"""
DeepShield V5 — Render-Ready Inference API
==========================================
Serves the DeepShield V5 multi-head deepfake audio detection model via Flask.

Endpoints
---------
GET  /health              — liveness / readiness probe
POST /api/predict         — single-file inference (multipart/form-data, key: "audio")
POST /api/predict/batch   — multi-file inference (multipart/form-data, key: "audio[]" or multiple "audio" fields)
GET  /api/model/info      — model metadata & architecture summary
"""

import io
import os
import sys
import time
import tempfile
import logging

# Render's free tier reports several logical cores to the container but only
# grants a tiny fractional CPU share. PyTorch's default intra-op thread pool
# sizes itself off the (misleading) core count, so threads end up fighting
# for a sliver of real CPU time instead of actually parallelizing — this is a
# well-known cause of CPU inference taking minutes instead of seconds on
# constrained containers. Pinning to a single thread removes that
# contention entirely; it has no effect on model outputs/accuracy. Must be
# set before numpy/torch spin up their thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deepshieldmodel")

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Reject oversized uploads before reading them into memory (only the first MAX_SCAN_SECONDS are analysed anyway).
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 25)) * 1024 * 1024

# Browser origins allowed to call the API. Defaults to the local demo UI; set CORS_ORIGINS="*" (or a
# comma-separated list) when the UI is served from elsewhere.
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS if CORS_ORIGINS == "*" else CORS_ORIGINS.split(",")}})

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR   = os.environ.get("DEEPSHIELD_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
BEST_CKPT   = os.path.join(MODEL_DIR, "deepshield_v5_best.pth")
NOVELTY_NPZ = os.path.join(MODEL_DIR, "novelty_scorer.npz")

# A retrained model ships gateway_calibration.json with settings fitted on validation data. Round 2 has only
# the uncertainty threshold (fallback: the original 0.15). Round 3 adds the ensemble member files, a
# temperature, silence trimming, and the combined-risk routing that replaces the rule gateway.
MIN_RELIABLE_SECONDS = 2.0
GATEWAY_CALIBRATION = os.path.join(MODEL_DIR, "gateway_calibration.json")
CALIBRATION = {}
if os.path.exists(GATEWAY_CALIBRATION):
    import json
    with open(GATEWAY_CALIBRATION) as _f:
        CALIBRATION = json.load(_f)
UNCERTAINTY_THRESH = float(CALIBRATION.get("uncertainty_threshold", 0.15))
TEMPERATURE        = float(CALIBRATION.get("temperature", 1.0))
_PREPROCESSING     = CALIBRATION.get("preprocessing", {})
TRIM_SILENCE       = bool(_PREPROCESSING.get("trim_silence", False))
TRIM_TOP_DB        = float(_PREPROCESSING.get("trim_top_db", 40))
COMBINED_RISK      = CALIBRATION.get("combined_risk")

# Optional inference-only add-ons (demo/addons.py, fitted by tools/fit_addons.py; no retraining):
# a reference bank fitted on labelled clips of the deployment's own audio, and a claimed-speaker check
# against enrolled genuine recordings. Each is active when its file is present in MODEL_DIR.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import addons  # noqa: E402
REFERENCE_BANK_PATH = os.path.join(MODEL_DIR, "reference_bank.json")
SPEAKER_FUSION_PATH = os.path.join(MODEL_DIR, "speaker_fusion.json")
ENROLL_DIR = os.environ.get("DEEPSHIELD_ENROLL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrollment"))
REFERENCE_BANK = (addons.ReferenceBank(REFERENCE_BANK_PATH)
                  if os.path.exists(REFERENCE_BANK_PATH) and os.environ.get("DEEPSHIELD_REFERENCE_BANK", "1") != "0" else None)
SPEAKER_CHECK = addons.SpeakerCheck(SPEAKER_FUSION_PATH, ENROLL_DIR) if os.path.exists(SPEAKER_FUSION_PATH) else None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR  = 16_000
DURATION   = 3        # seconds of audio per inference window
N_MELS     = 64
N_MFCC     = 64
N_FFT      = 1_024
HOP_LENGTH = 512

# A file longer than DURATION is scanned across multiple windows rather than
# only ever looking at its first few seconds -- synthetic artifacts can be
# localized to a short span (a spliced-in fake segment, a brief vocoder
# dropout) that a single 3s slice from the start of a long clip will simply
# never see. MAX_SCAN_SECONDS bounds how much of a very long file we decode
# and scan; MAX_WINDOWS bounds how many windows we run inference on (compute
# cost scales with this).
MAX_SCAN_SECONDS = 60
MAX_WINDOWS      = 120
# Windows overlap: one starts every WINDOW_HOP_SECONDS. The model's score swings sharply when its 3 s
# window shifts by half a second, so sparse back-to-back windows can all miss the synthetic stretch.
WINDOW_HOP_SECONDS = float(_PREPROCESSING.get("window_hop_s", 0.5))
# The clip score is the mean of the most suspicious third of its windows. Measured end to end on 800
# In-the-Wild clips (speaker-split): accuracy 79.8% vs 79.0% for the old "max of 2-12 windows" (dev 81.9
# vs 82.9, test 77.7 vs 75.2), genuine auto-blocked on test 3.4% vs 6.3%. Not an accuracy gain in itself;
# it removes the blind spots between sparse windows and matches how round-3 models are evaluated.
TOP_WINDOW_FRACTION = float(_PREPROCESSING.get("top_window_fraction", 1 / 3))
# MC-dropout passes for the uncertainty estimate; 20 matches the evaluation that calibrated the
# uncertainty threshold. The fixed seed makes the same clip always get the same route.
MC_SAMPLES = 20
MC_SEED    = 0

ATTACK_TYPES = [
    "real",
    "vocoder_artifact",
    "tts_prosody",
    "splice_edit",
    "unknown_synthetic",
]

ALLOWED_EXTENSIONS = {"wav", "flac", "mp3", "ogg", "m4a", "webm"}

# ─── Model Architecture (mirrors deepshieldv5.ipynb exactly) ──────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeepShieldBackbone(nn.Module):
    """Returns the pooled 128-d embedding shared by every head."""
    def __init__(self):
        super().__init__()
        self.init_conv   = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.layer1      = ResidualBlock(32, 32, stride=1)
        self.layer2      = ResidualBlock(32, 64, stride=2)
        self.attention   = SqueezeExcitationAttention(64)
        self.layer3      = ResidualBlock(64, 128, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, return_feature_map=False):
        x      = self.init_conv(x)
        x      = self.layer1(x)
        x      = self.layer2(x)
        x      = self.attention(x)
        fmap   = self.layer3(x)
        pooled = self.global_pool(fmap).flatten(1)
        if return_feature_map:
            return pooled, fmap
        return pooled


class DeepShieldMultiHead(nn.Module):
    """
    One backbone forward pass feeds four heads:
      - spoof_head  : binary real/fake logit
      - attack_head : multi-class attack-family logits (ATTACK_TYPES)
      - uncertainty : MC-dropout variance around the spoof probability
      - novelty     : handled separately by OpenSetNoveltyScorer
    """
    def __init__(self, embed_dim=128, num_attack_types=len(ATTACK_TYPES), dropout_p=0.35):
        super().__init__()
        self.backbone    = DeepShieldBackbone()
        self.spoof_head  = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.45),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(64, 1),
        )
        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_attack_types),
        )
        self.uncertainty_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mc_samples=1, return_embedding=False, mc_seed=None):
        embed        = self.backbone(x)
        spoof_logits = self.spoof_head(embed)
        attack_logits = self.attack_head(embed)

        if mc_samples > 1:
            # A fixed seed makes the dropout masks, and so the uncertainty and the gateway route,
            # identical every time the same clip is analysed (unseeded, a borderline clip could
            # be blocked on one upload and sent to review on the next).
            gen = torch.Generator(device=embed.device)
            if mc_seed is not None:
                gen.manual_seed(mc_seed)
            dropout_p = getattr(self.uncertainty_dropout, "p", 0.35)
            keep  = (torch.rand((mc_samples, *embed.shape), generator=gen, device=embed.device) >= dropout_p)
            e     = embed.unsqueeze(0) * keep / (1.0 - dropout_p)                 # [T, B, D]
            probs = torch.sigmoid(self.spoof_head(e.reshape(-1, embed.shape[-1])))
            probs = probs.reshape(mc_samples, embed.shape[0], 1)                 # [T, B, 1]
            mean_p      = probs.mean(dim=0)
            uncertainty = probs.std(dim=0)
        else:
            mean_p      = torch.sigmoid(spoof_logits)
            uncertainty = torch.zeros_like(mean_p)

        out = (spoof_logits, attack_logits, mean_p, uncertainty)
        return (embed, *out) if return_embedding else out


class OpenSetNoveltyScorer:
    """Mahalanobis-distance novelty scorer (calibrated on real-speech embeddings)."""
    def __init__(self):
        self.mean      = None
        self.inv_cov   = None
        self.threshold = None

    def load(self, path):
        data           = np.load(path)
        self.mean      = data["mean"]
        self.inv_cov   = data["inv_cov"]
        self.threshold = float(data["threshold"])

    def score(self, embedding: np.ndarray) -> float:
        diff = embedding - self.mean
        return float(np.sqrt(diff @ self.inv_cov @ diff.T))

    def is_novel(self, embedding: np.ndarray) -> bool:
        return self.score(embedding) > self.threshold


# ─── Global Model State ───────────────────────────────────────────────────────

model          = None
novelty_scorer = None
_model_loaded  = False
_load_error    = None


members        = []   # ensemble (round 3); members[0] is `model`


def _load_models():
    global model, members, novelty_scorer, _model_loaded, _load_error
    try:
        # Load the final trained DeepShield V5 checkpoint(s): every ensemble member listed in the
        # calibration file, or the single round-2 checkpoint.
        ckpt_paths = [os.path.join(MODEL_DIR, f) for f in CALIBRATION.get("ensemble_members", [])]
        if not ckpt_paths:
            ckpt_path = os.path.join(MODEL_DIR, "deepshield_v5_multihead.pth")
            ckpt_paths = [ckpt_path if os.path.exists(ckpt_path) else BEST_CKPT]

        members = []
        for ckpt_path in ckpt_paths:
            log.info("Loading DeepShield V5 checkpoint from %s …", ckpt_path)
            m = DeepShieldMultiHead().to(DEVICE)
            m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            m.eval()
            members.append(m)
        model = members[0]
        log.info("%d model(s) loaded on %s.", len(members), DEVICE)

        log.info("Loading novelty scorer from %s …", NOVELTY_NPZ)
        novelty_scorer = OpenSetNoveltyScorer()
        novelty_scorer.load(NOVELTY_NPZ)
        log.info("Novelty scorer loaded. Threshold = %.3f", novelty_scorer.threshold)

        _model_loaded = True
    except Exception as exc:
        _load_error = str(exc)
        log.exception("Failed to load model: %s", exc)


# Load at import time (Render spins up one worker; Gunicorn pre-loads)
_load_models()


# ─── Feature Extraction ───────────────────────────────────────────────────────

class AudioDecodeError(ValueError):
    """The upload is not decodable audio; reported to the caller as a 400, not a server error."""


def _load_audio(audio_bytes: bytes, file_ext: str = ".wav", duration: float | None = None) -> np.ndarray:
    """
    Load audio from raw bytes as mono @ TARGET_SR. Tries in-memory BytesIO
    first, with a safe flushed tempfile fallback. `duration` caps how many
    seconds are decoded (None = whole file).
    """
    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, duration=duration, mono=True)
    except Exception:
        suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
        try:
            y, _ = librosa.load(tmp_path, sr=TARGET_SR, duration=duration, mono=True)
        except Exception as exc:
            raise AudioDecodeError(f"Could not decode the file as audio ({type(exc).__name__}).") from exc
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    return y


def _features_from_waveform(y: np.ndarray) -> np.ndarray:
    """Pad/truncate a mono waveform to exactly DURATION seconds and extract the
    dual-channel (log-Mel, MFCC) feature tensor. Returns ndarray (2, N_MELS, 94)."""
    target_samples = TARGET_SR * DURATION
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    elif len(y) > target_samples:
        y = y[:target_samples]

    mel    = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc   = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    return np.stack([mel_db, mfcc], axis=0).astype(np.float32)


def _window_offsets(total_samples: int, win_samples: int) -> list:
    """
    Sample offsets (in samples) for each inference window. A clip no longer
    than one window is analyzed once. A longer clip is covered by overlapping
    windows every WINDOW_HOP_SECONDS (the last one ending at the end of the
    clip), thinned evenly if that would exceed MAX_WINDOWS.
    """
    max_start = total_samples - win_samples
    if max_start <= 0:
        return [0]
    hop = int(WINDOW_HOP_SECONDS * TARGET_SR)
    offsets = list(range(0, max_start + 1, hop))
    if offsets[-1] != max_start:
        offsets.append(max_start)
    if len(offsets) > MAX_WINDOWS:
        offsets = [offsets[int(round(i * (len(offsets) - 1) / (MAX_WINDOWS - 1)))] for i in range(MAX_WINDOWS)]
    return offsets


def _aggregate_windows(window_probs: np.ndarray) -> float:
    """Clip-level spoof probability: mean of the most suspicious TOP_WINDOW_FRACTION of windows."""
    k = max(1, int(len(window_probs) * TOP_WINDOW_FRACTION))
    return float(np.sort(window_probs)[-k:].mean())


# ─── Risk-Aware Gateway & Rationale (Notebook Section 11 & 12) ────────────────

def risk_aware_gateway(spoof_prob: float, novelty_score: float, novelty_threshold: float,
                       uncertainty: float, low_thresh: float = 0.30,
                       high_thresh: float = 0.70, uncertainty_thresh: float = 0.15) -> str:
    """
    Risk-Aware Gateway (Section 3.1 / 3.2, Figure 3 in the proposal).
    Novelty is checked first: an input unlike anything in training is routed
    to review regardless of what the spoof-probability head says.
    High predictive uncertainty similarly forces a review routing.
    """
    if novelty_score is not None and novelty_score > novelty_threshold:
        return "Stage 4: Unknown Pattern (novel / OOD) -> Human Review"
    if uncertainty is not None and uncertainty > uncertainty_thresh:
        return "Stage 3: High Uncertainty -> Quality-Control Review"
    if spoof_prob < low_thresh:
        return "Stage 1: Pass (Automated Clearance)"
    if spoof_prob > high_thresh:
        return "Stage 2: Block (Automated Termination)"
    return "Stage 3: Quality-Control Review Queue"


def combined_risk_score(spoof_prob: float, ensemble_spread: float, uncertainty: float, novelty_score: float) -> float:
    """Round-3 combined risk: a logistic model, fitted on validation clips, of how likely the 0.5 verdict is
    wrong, from the score margin and the other heads' signals (see eval/deepshield_r3_core.py)."""
    p = min(max(spoof_prob, 1e-6), 1 - 1e-6)
    signals = {"margin": -abs(float(np.log(p / (1 - p)))), "ens_std": ensemble_spread,
               "mc_unc": uncertainty, "novelty": novelty_score}
    cr = COMBINED_RISK
    x = [(signals[n] - mu) / sd for n, mu, sd in zip(cr["features"], cr["mean"], cr["scale"])]
    return float(np.dot(cr["coef"], x) + cr["intercept"])


def generate_rationale(spoof_prob: float, attack_type: str, attack_conf: float,
                       novelty_score: float, novelty_threshold: float, uncertainty: float) -> str:
    """Deterministic, template-based rationale — never hallucinated (Section 3.3)."""
    if novelty_score is not None and novelty_score > novelty_threshold:
        return (f"Unknown pattern (novelty={novelty_score:.2f} > threshold {novelty_threshold:.2f}): "
                f"does not match the real-speech manifold or any known attack family. "
                f"Routed to human review as an out-of-distribution sample.")
    if spoof_prob > 0.70:
        return (f"Flagged as synthetic (P={spoof_prob:.2f}, uncertainty={uncertainty:.2f}). "
                f"Attributed to '{attack_type}' family with {attack_conf:.0%} confidence.")
    if spoof_prob < 0.30:
        return f"Cleared as authentic human speech (P={spoof_prob:.2f}, uncertainty={uncertainty:.2f})."
    return f"Borderline (P={spoof_prob:.2f}, uncertainty={uncertainty:.2f}); routed to Quality-Control review queue."


# ─── Inference Helper (Notebook Section 13) ───────────────────────────────────

@torch.no_grad()
def score_audio(audio_bytes: bytes, file_ext: str = ".wav") -> dict:
    """Decode, window and score a clip with every ensemble member. Shared by /api/predict, /api/enroll and
    tools/fit_addons.py, so the add-ons are fitted on exactly the embeddings the app computes."""
    y_full     = _load_audio(audio_bytes, file_ext, duration=MAX_SCAN_SECONDS)
    # Round-3 models are trained on audio with leading/trailing silence removed; trim the same way.
    # Reported window times stay relative to the original upload (the UI seeks the player to them).
    lead_sec = 0.0
    if TRIM_SILENCE and len(y_full):
        y_trim, (t0_idx, _) = librosa.effects.trim(y_full, top_db=TRIM_TOP_DB)
        if len(y_trim) >= TARGET_SR // 2:
            y_full, lead_sec = y_trim, t0_idx / TARGET_SR
    win_samples = TARGET_SR * DURATION
    offsets     = _window_offsets(len(y_full), win_samples)

    # One deterministic pass (mc_samples=1) over every overlapping window, batched, for every
    # ensemble member. The clip score uses the members' mean window logit, temperature-scaled.
    feats = np.stack([_features_from_waveform(y_full[s:s + win_samples]) for s in offsets])
    xs    = torch.from_numpy(feats).to(DEVICE)
    window_logits, attack_probs_all = [], []
    for k, m in enumerate(members):
        m.eval()
        embed_k, logits_k, attack_k, _, _ = m(xs, mc_samples=1, return_embedding=True)
        if k == 0:
            embeds = embed_k
        window_logits.append(logits_k[:, 0].cpu().numpy())
        attack_probs_all.append(F.softmax(attack_k, dim=-1).cpu().numpy())
    window_logits = np.stack(window_logits)                              # [members, windows]
    window_probs  = 1.0 / (1.0 + np.exp(-window_logits.mean(0) / TEMPERATURE))
    return {"y": y_full, "lead_sec": lead_sec, "offsets": offsets, "xs": xs, "embeds": embeds,
            "window_logits": window_logits, "window_probs": window_probs, "attack_probs_all": attack_probs_all,
            "detector_probability": _aggregate_windows(window_probs),
            # clip embedding used by the add-ons: mean over windows, ensemble member 0
            "clip_embedding": embeds.cpu().numpy().mean(0)}


@torch.no_grad()
def _run_inference(audio_bytes: bytes, file_ext: str = ".wav", claimed_speaker: str | None = None) -> dict:
    t0 = time.perf_counter()

    sc = score_audio(audio_bytes, file_ext)
    y_full, lead_sec, offsets, xs, embeds = sc["y"], sc["lead_sec"], sc["offsets"], sc["xs"], sc["embeds"]
    window_logits, window_probs, attack_probs_all = sc["window_logits"], sc["window_probs"], sc["attack_probs_all"]
    b = int(np.argmax(window_probs))
    best = {
        "x":             xs[b:b + 1],
        "embed":         embeds[b:b + 1],
        "offset_sec":    round(lead_sec + offsets[b] / TARGET_SR, 2),
    }

    # 2. MC-dropout predictive uncertainty for the most suspicious window only (the expensive
    #    pass), seeded so repeated uploads of the same clip get the same answer.
    _, _, _, unc = model(best["x"], mc_samples=MC_SAMPLES, mc_seed=MC_SEED)

    spoof_prob   = sc["detector_probability"]
    uncertainty  = float(unc.item())

    # Add-ons. A claimed-speaker check (fitted on the detector's own score) takes precedence over the
    # reference bank; the detector's score is always reported alongside.
    scoring, speaker_check = "detector", None
    if REFERENCE_BANK is not None:
        spoof_prob, scoring = REFERENCE_BANK.prob(sc["clip_embedding"]), "reference_bank"
    if claimed_speaker:
        if SPEAKER_CHECK is None:
            speaker_check = {"applied": False, "reason": "speaker check not configured (no speaker_fusion.json)"}
        else:
            speaker_check = SPEAKER_CHECK.check(claimed_speaker, sc["clip_embedding"], sc["detector_probability"])
            if speaker_check["applied"]:
                spoof_prob, scoring = speaker_check["fused_probability"], "speaker_check"
    # Disagreement between ensemble members (0 with a single model).
    ensemble_spread = float(np.std([_aggregate_windows(1.0 / (1.0 + np.exp(-l))) for l in window_logits])) \
        if len(members) > 1 else 0.0

    attack_probs = np.mean(attack_probs_all, axis=0)[b]
    attack_idx   = int(np.argmax(attack_probs))
    attack_type  = ATTACK_TYPES[attack_idx]
    attack_conf  = float(attack_probs[attack_idx])

    embed_np          = best["embed"].cpu().numpy()[0]
    novelty_score     = float(novelty_scorer.score(embed_np)) if novelty_scorer is not None else 0.0
    novelty_threshold = float(novelty_scorer.threshold) if novelty_scorer is not None else 14.739
    is_novel          = bool(novelty_score > novelty_threshold)

    # Risk-Aware Gateway routing and explanation rationale. Round-3 models ship a combined risk score
    # (calibrated to a genuine-review budget on validation data) that replaces the rule cascade.
    risk_score = None
    if COMBINED_RISK:
        risk_score = combined_risk_score(spoof_prob, ensemble_spread, uncertainty, novelty_score)
        if risk_score > COMBINED_RISK["threshold"]:
            routing = "Stage 3: Elevated Combined Risk -> Human Review"
        elif spoof_prob > 0.5:
            routing = "Stage 2: Block (Automated Termination)"
        else:
            routing = "Stage 1: Pass (Automated Clearance)"
    else:
        routing = risk_aware_gateway(spoof_prob, novelty_score, novelty_threshold, uncertainty,
                                     uncertainty_thresh=UNCERTAINTY_THRESH)
    # Clips under MIN_RELIABLE_SECONDS carry too little speech for an automatic verdict: on the
    # In-the-Wild demo clips every genuine clip under 2 s was flagged fake. Route them to review.
    if len(y_full) < MIN_RELIABLE_SECONDS * TARGET_SR:
        routing = (f"Stage 3: Too Short To Judge Reliably (<{MIN_RELIABLE_SECONDS:g} s of audio) "
                   f"-> Quality-Control Review")
    # A silent or near-silent upload has nothing to judge; never let it be auto-passed or auto-blocked.
    if len(y_full) == 0 or float(np.max(np.abs(y_full))) < 1e-3:
        routing = "Stage 3: No Audible Signal -> Quality-Control Review"
    rationale = generate_rationale(spoof_prob, attack_type, attack_conf, novelty_score, novelty_threshold, uncertainty)
    if speaker_check and speaker_check.get("applied"):
        rationale += (f" Compared with {speaker_check['enrolled_clips']} enrolled genuine recordings of "
                      f"'{speaker_check['speaker']}': closest distance {speaker_check['l2_distance']:.1f} "
                      f"(detector alone P={speaker_check['detector_probability']:.2f}).")

    # Primary verdict and confidence (Cells 840 & 884 of notebook)
    is_deepfake = bool(spoof_prob >= 0.50)
    verdict     = "SYNTHETIC" if is_deepfake else "AUTHENTIC"
    confidence  = float(max(spoof_prob, 1.0 - spoof_prob))

    # Risk level classification
    if spoof_prob > 0.70:
        risk_level = "HIGH"
    elif spoof_prob >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "verdict":           verdict,
        "is_deepfake":       is_deepfake,
        "confidence":        round(confidence, 4),
        "spoof_probability": round(spoof_prob, 6),
        "real_probability":  round(1.0 - spoof_prob, 6),
        "uncertainty":       round(uncertainty, 6),
        "attack_type": {
            "label":         attack_type,
            "confidence":    round(attack_conf, 4),
            "probabilities": {
                t: round(float(p), 6) for t, p in zip(ATTACK_TYPES, attack_probs)
            },
        },
        "novelty": {
            "score":         round(novelty_score, 4),
            "is_novel":      is_novel,
            "threshold":     round(novelty_threshold, 4),
        },
        "gateway_routing":   routing,
        "rationale":         rationale,
        "risk_level":        risk_level,
        "scoring":           scoring,
        "detector_probability": round(sc["detector_probability"], 6),
        "speaker_check":     speaker_check,
        "ensemble_members":  len(members),
        "ensemble_spread":   round(ensemble_spread, 6),
        "routing_mode":      "combined_risk" if COMBINED_RISK else "rules",
        "risk_score":        None if risk_score is None else round(risk_score, 4),
        "windows_analyzed":  len(offsets),
        "segment_offset_sec": best["offset_sec"],
        "aggregation":       (f"mean of the top {TOP_WINDOW_FRACTION:.0%} of {len(offsets)} overlapping "
                              f"{DURATION} s windows (hop {WINDOW_HOP_SECONDS:g} s)"),
        # Per-window scores let a reviewer see where in the clip the synthetic evidence sits.
        "window_scores":     [{"start_sec": round(lead_sec + s / TARGET_SR, 2), "spoof_probability": round(float(p), 4)}
                              for s, p in zip(offsets, window_probs)],
        "inference_ms":      elapsed_ms,
        "device":            str(DEVICE),
    }


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    GET /
    Serves the interactive web diagnostic console.
    """
    return send_from_directory(os.path.dirname(__file__), "frontend_test.html")


@app.route("/health", methods=["GET"])
@app.route("/api/status", methods=["GET"])
@app.route("/status", methods=["GET"])
def health():
    """
    GET /health
    Liveness / readiness probe for Render.
    Returns 200 when the model is ready, 503 otherwise.
    """
    if _model_loaded:
        return jsonify({
            "status":  "ok",
            "model":   "DeepShield V5",
            "device":  str(DEVICE),
            "version": "5.0.0",
        }), 200
    else:
        return jsonify({
            "status": "unavailable",
            "error":  _load_error,
        }), 503


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: multipart/form-data with field "audio" (audio file)

    Response JSON:
    {
      "filename": "...",
      "is_deepfake": bool,
      "spoof_probability": float,
      "real_probability": float,
      "uncertainty": float,
      "attack_type": { "label": str, "probabilities": {str: float} },
      "novelty": { "score": float, "is_novel": bool, "threshold": float },
      "risk_level": "LOW" | "MEDIUM" | "HIGH",
      "windows_analyzed": int,     # how many DURATION-second windows across the clip were scanned
      "segment_offset_sec": float, # start time of the most-suspicious window used for the verdict
      "inference_ms": float,
      "device": str
    }
    """
    if not _model_loaded:
        return jsonify({"error": "Model not loaded", "detail": _load_error}), 503

    if "audio" not in request.files:
        raise BadRequest("Missing 'audio' field in multipart/form-data request.")

    file = request.files["audio"]
    if file.filename == "":
        raise BadRequest("No file selected.")
    if not _allowed_file(file.filename):
        raise BadRequest(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    try:
        audio_bytes = file.read()
        ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        result = _run_inference(audio_bytes, file_ext=ext, claimed_speaker=request.form.get("claimed_speaker") or None)
        result["filename"] = file.filename
        return jsonify(result), 200
    except (AudioDecodeError, ValueError) as exc:
        return jsonify({"error": "Bad Request", "detail": str(exc)}), 400
    except Exception as exc:
        log.exception("Inference error for file %s: %s", file.filename, exc)
        return jsonify({"error": "Inference failed", "detail": str(exc)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    POST /api/predict/batch
    Body: multipart/form-data with one or more "audio" fields (up to 10 files).

    Response JSON:
    {
      "results": [ { ...single predict result... }, ... ],
      "total_files": int,
      "errors": [ { "filename": str, "error": str }, ... ]
    }
    """
    if not _model_loaded:
        return jsonify({"error": "Model not loaded", "detail": _load_error}), 503

    files = request.files.getlist("audio")
    if not files:
        # also try "audio[]" key (common in some JS frameworks)
        files = request.files.getlist("audio[]")
    if not files:
        raise BadRequest("No 'audio' files found in the request.")
    if len(files) > 10:
        raise BadRequest("Batch endpoint accepts at most 10 files per request.")

    results = []
    errors  = []

    for file in files:
        if file.filename == "" or not _allowed_file(file.filename):
            errors.append({"filename": file.filename, "error": "Unsupported or missing file."})
            continue
        try:
            audio_bytes = file.read()
            ext = os.path.splitext(file.filename)[1].lower() or ".wav"
            res = _run_inference(audio_bytes, file_ext=ext)
            res["filename"] = file.filename
            results.append(res)
        except Exception as exc:
            log.exception("Batch inference error for %s: %s", file.filename, exc)
            errors.append({"filename": file.filename, "error": str(exc)})

    return jsonify({
        "results":     results,
        "total_files": len(files),
        "errors":      errors,
    }), 200


@app.route("/api/enroll", methods=["POST"])
def enroll():
    """
    POST /api/enroll
    Body: multipart/form-data with "speaker" (id) and one or more "audio" files of that speaker's VERIFIED
    GENUINE speech (at least 2 s each). A clip the detector scores as synthetic is skipped unless force=1.
    Later predictions with claimed_speaker=<id> are compared with these recordings.
    """
    if SPEAKER_CHECK is None:
        return jsonify({"error": "Speaker check not configured", "detail": "no speaker_fusion.json in the model folder"}), 503
    speaker = request.form.get("speaker", "")
    try:
        slug = SPEAKER_CHECK.slug(speaker)
    except ValueError as exc:
        raise BadRequest(str(exc))
    files = request.files.getlist("audio") or request.files.getlist("audio[]")
    if not files:
        raise BadRequest("No 'audio' files in the request.")
    force = request.form.get("force") == "1"
    added, skipped, embeddings, sources = [], [], [], []
    for f in files[:20]:
        if not f.filename or not _allowed_file(f.filename):
            skipped.append({"filename": f.filename, "reason": "unsupported or missing file"}); continue
        try:
            sc = score_audio(f.read(), os.path.splitext(f.filename)[1].lower() or ".wav")
        except AudioDecodeError as exc:
            skipped.append({"filename": f.filename, "reason": str(exc)}); continue
        if len(sc["y"]) < MIN_RELIABLE_SECONDS * TARGET_SR or float(np.max(np.abs(sc["y"]))) < 1e-3:
            skipped.append({"filename": f.filename, "reason": f"shorter than {MIN_RELIABLE_SECONDS:g} s or silent"}); continue
        if sc["detector_probability"] > 0.5 and not force:
            skipped.append({"filename": f.filename, "reason": f"detector scores it as synthetic "
                                                              f"(P={sc['detector_probability']:.2f}); send force=1 if verified genuine"})
            continue
        embeddings.append(sc["clip_embedding"]); sources.append(f.filename); added.append(f.filename)
    total = SPEAKER_CHECK.add(slug, embeddings, sources) if embeddings else len(SPEAKER_CHECK.enrolled(slug)[0])
    return jsonify({"speaker": slug, "added": added, "skipped": skipped, "enrolled_clips": total,
                    "ready": total >= SPEAKER_CHECK.min_enrolled}), 200


@app.route("/api/enroll", methods=["GET"])
def enrolled_speakers():
    """GET /api/enroll — enrolled speakers and how many genuine clips each has."""
    if SPEAKER_CHECK is None:
        return jsonify({"configured": False, "speakers": {}}), 200
    return jsonify({"configured": True, "min_enrolled": SPEAKER_CHECK.min_enrolled,
                    "speakers": SPEAKER_CHECK.speakers()}), 200


@app.route("/api/enroll/<speaker>", methods=["DELETE"])
def unenroll(speaker):
    """DELETE /api/enroll/<speaker> — remove a speaker's enrolled recordings."""
    if SPEAKER_CHECK is None:
        return jsonify({"error": "Speaker check not configured"}), 503
    try:
        removed = SPEAKER_CHECK.remove(speaker)
    except ValueError as exc:
        raise BadRequest(str(exc))
    return jsonify({"speaker": SPEAKER_CHECK.slug(speaker), "removed": removed}), 200 if removed else 404


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """
    GET /api/model/info
    Returns model metadata and architecture summary.
    """
    param_count = sum(p.numel() for p in model.parameters()) if _model_loaded else None
    return jsonify({
        "name":        "DeepShield V5",
        "version":     "5.0.0",
        "description": (
            "Dual-channel deepfake audio detection model. "
            "Compact residual CNN (3 blocks) + SE-Attention backbone with multi-head classification: "
            "spoof detection, attack-type attribution, MC-dropout uncertainty, "
            "and open-set Mahalanobis novelty scoring."
        ),
        "architecture": {
            "backbone":       "Residual CNN (3 residual blocks, 32-64-128 channels) + Squeeze-and-Excitation Attention",
            "embedding_dim":  128,
            "input_channels": 2,
            "input_shape":    [2, N_MELS, "T"],
            "heads": [
                "spoof_head (binary: real / fake)",
                "attack_head (5-class: real, vocoder_artifact, tts_prosody, splice_edit, unknown_synthetic)",
                f"uncertainty_estimator (MC-Dropout, {MC_SAMPLES} seeded passes)",
                "novelty_scorer (Mahalanobis distance on real-speech manifold)",
            ],
            "total_parameters": param_count,
        },
        "feature_extraction": {
            "sample_rate":  TARGET_SR,
            "duration_sec": DURATION,
            "n_fft":        N_FFT,
            "hop_length":   HOP_LENGTH,
            "channel_1":    f"log-power Mel-spectrogram ({N_MELS} bins)",
            "channel_2":    f"MFCC envelope ({N_MFCC} coefficients)",
        },
        "performance": {
            # Round-2 retrained model, measured on the held-out test set (6,648 clips; no recording,
            # sentence or speaker shared with training). The original 97.73 % / 0.9957 came from a
            # random split where ~28 % of validation clips had a copy in training.
            "val_accuracy":  "93.2 %",
            "val_roc_auc":   "0.978",
            "training_epochs": 40,
            "dataset_size":   40000,
        },
        "addons": {
            "reference_bank": None if REFERENCE_BANK is None else REFERENCE_BANK.meta,
            "speaker_check":  None if SPEAKER_CHECK is None else {**SPEAKER_CHECK.meta, "min_enrolled": SPEAKER_CHECK.min_enrolled},
        },
        "attack_types":  ATTACK_TYPES,
        "allowed_audio": sorted(ALLOWED_EXTENSIONS),
        "device":        str(DEVICE),
        "model_loaded":  _model_loaded,
    }), 200


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "detail": str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method Not Allowed", "detail": str(e)}), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File Too Large", "detail": str(e)}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Local-only by default; set HOST=0.0.0.0 to serve other machines (e.g. inside a container).
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
