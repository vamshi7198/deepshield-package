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

# Allow requests from any origin (configure to specific domains in production)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
BEST_CKPT   = os.path.join(MODEL_DIR, "deepshield_v5_best.pth")
NOVELTY_NPZ = os.path.join(MODEL_DIR, "novelty_scorer.npz")

# Uncertainty threshold for the risk gateway. A retrained model ships gateway_calibration.json with a
# threshold calibrated on validation data; otherwise fall back to the original 0.15.
MIN_RELIABLE_SECONDS = 2.0
GATEWAY_CALIBRATION = os.path.join(MODEL_DIR, "gateway_calibration.json")
UNCERTAINTY_THRESH = 0.15
if os.path.exists(GATEWAY_CALIBRATION):
    import json
    with open(GATEWAY_CALIBRATION) as _f:
        UNCERTAINTY_THRESH = float(json.load(_f)["uncertainty_threshold"])

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
MAX_WINDOWS      = 12

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

    def forward(self, x, mc_samples=1, return_embedding=False):
        embed        = self.backbone(x)
        spoof_logits = self.spoof_head(embed)
        attack_logits = self.attack_head(embed)

        if mc_samples > 1:
            probs = []
            dropout_p = getattr(self.uncertainty_dropout, "p", 0.35)
            for _ in range(mc_samples):
                e = F.dropout(embed, p=dropout_p, training=True)
                probs.append(torch.sigmoid(self.spoof_head(e)))
            probs       = torch.stack(probs)           # [T, B, 1]
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


def _load_models():
    global model, novelty_scorer, _model_loaded, _load_error
    try:
        # Load the final trained DeepShield V5 checkpoint
        ckpt_path = os.path.join(MODEL_DIR, "deepshield_v5_multihead.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = BEST_CKPT

        log.info("Loading DeepShield V5 checkpoint from %s …", ckpt_path)
        model = DeepShieldMultiHead().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        log.info("Model loaded successfully on %s.", DEVICE)

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
    than one window is analyzed once, exactly as before. A longer clip is
    scanned at evenly spaced offsets spanning its full (scanned) length, up
    to MAX_WINDOWS -- a synthetic artifact localized anywhere in the clip
    still has a chance to land inside a window instead of being silently
    cropped out by only ever looking at the start.
    """
    max_start = total_samples - win_samples
    if max_start <= 0:
        return [0]
    # roughly one window per 5s of extra content, capped at MAX_WINDOWS
    n = max(2, min(MAX_WINDOWS, int(max_start / TARGET_SR / 5) + 2))
    return [int(round(i * max_start / (n - 1))) for i in range(n)]


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
def _run_inference(audio_bytes: bytes, file_ext: str = ".wav") -> dict:
    t0 = time.perf_counter()

    y_full     = _load_audio(audio_bytes, file_ext, duration=MAX_SCAN_SECONDS)
    win_samples = TARGET_SR * DURATION
    offsets     = _window_offsets(len(y_full), win_samples)

    # Keep model in eval mode so BatchNorm and Dropout are stable
    model.eval()

    # 1. Cheap deterministic pass (mc_samples=1) per window to find the most
    #    suspicious segment -- exactly the notebook's single-window pass when
    #    there's only one window (short clip), but scanning every window on a
    #    longer clip instead of just cropping to its first DURATION seconds.
    best = None
    for start in offsets:
        segment = y_full[start:start + win_samples]
        feat = _features_from_waveform(segment)
        x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        embed, _, attack_logits, mean_p, _ = model(x, mc_samples=1, return_embedding=True)
        spoof_prob = float(mean_p.item())

        if best is None or spoof_prob > best["spoof_prob"]:
            best = {
                "spoof_prob":    spoof_prob,
                "x":             x,
                "embed":         embed,
                "attack_logits": attack_logits,
                "offset_sec":    round(start / TARGET_SR, 2),
            }

    # 2. Full 10-pass MC-dropout predictive uncertainty, only for the single
    #    most-suspicious window (the expensive pass, so we don't pay for it
    #    MAX_WINDOWS times over).
    _, _, _, unc = model(best["x"], mc_samples=10)

    spoof_prob   = best["spoof_prob"]
    uncertainty  = float(unc.item())

    attack_probs = F.softmax(best["attack_logits"], dim=-1).cpu().numpy()[0]
    attack_idx   = int(np.argmax(attack_probs))
    attack_type  = ATTACK_TYPES[attack_idx]
    attack_conf  = float(attack_probs[attack_idx])

    embed_np          = best["embed"].cpu().numpy()[0]
    novelty_score     = float(novelty_scorer.score(embed_np)) if novelty_scorer is not None else 0.0
    novelty_threshold = float(novelty_scorer.threshold) if novelty_scorer is not None else 14.739
    is_novel          = bool(novelty_score > novelty_threshold)

    # Risk-Aware Gateway routing and explanation rationale
    routing   = risk_aware_gateway(spoof_prob, novelty_score, novelty_threshold, uncertainty,
                                   uncertainty_thresh=UNCERTAINTY_THRESH)
    # Clips under MIN_RELIABLE_SECONDS carry too little speech for an automatic verdict: on the
    # In-the-Wild demo clips every genuine clip under 2 s was flagged fake. Route them to review.
    if len(y_full) < MIN_RELIABLE_SECONDS * TARGET_SR:
        routing = (f"Stage 3: Too Short To Judge Reliably (<{MIN_RELIABLE_SECONDS:g} s of audio) "
                   f"-> Quality-Control Review")
    rationale = generate_rationale(spoof_prob, attack_type, attack_conf, novelty_score, novelty_threshold, uncertainty)

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
        "windows_analyzed":  len(offsets),
        "segment_offset_sec": best["offset_sec"],
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
        result = _run_inference(audio_bytes, file_ext=ext)
        result["filename"] = file.filename
        return jsonify(result), 200
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
            "ResNet-34 + SE-Attention backbone with multi-head classification: "
            "spoof detection, attack-type attribution, MC-dropout uncertainty, "
            "and open-set Mahalanobis novelty scoring."
        ),
        "architecture": {
            "backbone":       "ResNet-34 + Squeeze-and-Excitation Attention",
            "embedding_dim":  128,
            "input_channels": 2,
            "input_shape":    [2, N_MELS, "T"],
            "heads": [
                "spoof_head (binary: real / fake)",
                "attack_head (5-class: real, vocoder_artifact, tts_prosody, splice_edit, unknown_synthetic)",
                "uncertainty_estimator (MC-Dropout, 20 passes)",
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

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
