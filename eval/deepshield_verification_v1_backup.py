# %% [markdown]
# # DeepShield V5 — Verification Run
#
# Re-runs the students' DeepShield V5 pipeline with a corrected **evaluation**.
# The architecture (dual-channel Mel+MFCC → ResNet + SE → spoof / attack / MC-dropout
# uncertainty heads, Mahalanobis novelty scorer, risk-aware gateway) is copied unchanged
# from `deepshieldv5.ipynb`, so the trained weights stay drop-in compatible with
# `deepshieldmodel/app.py`.
#
# What this run changes, and why:
#
# | # | Change | Why |
# |---|---|---|
# | 1 | Generator labels rebuilt from folder names | Original tagged all Fake-or-Real *real* clips as a fake generator, lumped 8 modern TTS systems together, and labelled every unidpro file fake |
# | 2 | Group-aware train / val / test split | ~23% of the original validation clips had the same recording (FoR versions) or sentence (LJSpeech) in training |
# | 3 | Separate test set, touched once | Original 97.73% was measured on the set used to pick the best epoch |
# | 4 | Genuine LJSpeech added as real audio (if attached) | 60% of fakes were one LJSpeech speaker with zero real recordings of her |
# | 5 | MC-dropout fixed | Dropout was disabled by `model.eval()`, so uncertainty was always 0 in the notebook |
# | 6 | LOGO actually run, per generator family | Code existed but was commented out |
# | 7 | Uncertainty, novelty and gateway evaluated | The integrated-system claim had no numbers behind it |
#
# **How to run on Kaggle:** attach the same four datasets the students used (plus, ideally,
# an LJ Speech dataset — search "LJ Speech" in *Add Input*), set Accelerator to **GPU T4 x2**
# or **P100**, Internet on, then **Save Version → Save & Run All (Commit)**. Runtime ≈ 2–3 hours.
# Everything lands in `/kaggle/working/deepshield_eval/`; the last cell prints a plain-English summary.

# %%
import os, glob, re, json, math, random, warnings, io, time
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Settings → Accelerator → GPU T4 x2 (or P100), then restart and run all.")
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print("GPU:", torch.cuda.get_device_name(0))

OUT = "/kaggle/working/deepshield_eval"
os.makedirs(OUT, exist_ok=True)

# ---- Run configuration ----
POOL_PER_CLASS   = 20000   # same 40k balanced pool size the students used
LJ_REAL_CAP      = 6000    # genuine LJSpeech clips to include as real (if dataset attached)
TEST_FRAC_FOLDS  = 3       # 3 of 20 group-folds = 15% test
VAL_FRAC_FOLDS   = 3       # 3 of 20 group-folds = 15% val
EPOCHS_MAIN, PATIENCE_MAIN = 40, 8
EPOCHS_LOGO, PATIENCE_LOGO = 25, 6
MC_SAMPLES       = 20
ROBUSTNESS_N     = 1000    # balanced test clips re-extracted under each degradation
RUN_NAIVE_BASELINE = True  # retrain on the original random split to show how much it inflated accuracy
N_JOBS           = os.cpu_count() or 2

# SMOKE_TEST = True runs the whole pipeline on a tiny pool with 1-2 epochs (~10-15 min) to catch errors.
# The numbers it prints are meaningless. Set it back to False for the real run.
SMOKE_TEST = True
if SMOKE_TEST:
    POOL_PER_CLASS, LJ_REAL_CAP = 1500, 300
    EPOCHS_MAIN, PATIENCE_MAIN, EPOCHS_LOGO, PATIENCE_LOGO = 2, 2, 1, 1
    MC_SAMPLES, ROBUSTNESS_N = 5, 100
    print("SMOKE TEST MODE — pipeline check only, results are not meaningful")

# %% [markdown]
# ## 1. Dataset discovery with corrected labels

# %%
DATASETS = {
    "adarsh_tts":   ("adarshsingh0903", "audio-deepfake-detection-dataset"),
    "unidpro":      ("unidpro", "real-vs-fake-human-voice-deepfake-audio"),
    "fakeaudio":    ("walimuhammadahmad", "fakeaudio"),
    "fake_or_real": ("mohammedabdeldayem", "the-fake-or-real-dataset"),
}

def resolve_root(owner, slug):
    for cand in (f"/kaggle/input/datasets/{owner}/{slug}", f"/kaggle/input/{slug}"):
        if os.path.isdir(cand):
            return cand
    return None

SOURCES = {name: resolve_root(*ids) for name, ids in DATASETS.items()}
for name, root in SOURCES.items():
    print(f"  [{'OK' if root else 'MISSING':7s}] {name:13s} -> {root}")
missing = [n for n, r in SOURCES.items() if r is None]
if missing:
    raise RuntimeError(f"Attach these Kaggle datasets first: {missing}")

AUDIO_EXTENSIONS = ("wav", "flac", "mp3")

def glob_audio(root):
    hits = []
    for ext in AUDIO_EXTENSIONS:
        hits.extend(glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
    return hits


def find_ljspeech_real():
    '''Look for an attached LJ Speech dataset (any slug) by its first clip's file name.'''
    known = {os.path.realpath(r) for r in SOURCES.values()}
    cands = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/datasets/*/*")
    for d in cands:
        if not os.path.isdir(d) or os.path.realpath(d) in known or d.rstrip("/").endswith("/datasets"):
            continue
        if any(os.path.realpath(d).startswith(k) for k in known):
            continue
        hits = glob.glob(os.path.join(d, "**", "LJ001-0001.wav"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def vocoder_family(folder):
    f = folder.lower().replace("_", "")
    if "conformer" in f or "fastspeech" in f:
        return "conformer_fastspeech2"      # full TTS system (checked first: its folder name also says 'pwg')
    if "melgan" in f:
        return "melgan_family"              # melgan, melgan_large, multi_band, full_band
    if "hifigan" in f:
        return "hifigan"
    if "waveglow" in f:
        return "waveglow"
    if "parallelwavegan" in f or "pwg" in f:
        return "parallel_wavegan"
    return "wavefake_other"


def describe(path, source):
    '''
    Returns (label, generator, detail, group) or None to skip.
      label     0 = real, 1 = fake
      generator generator FAMILY used for LOGO and per-generator reporting ('real' for genuine audio)
      detail    finest-grained tag available (sub-system / folder)
      group     recordings that must stay on the same side of every split
                (same FoR clip in its 4 versions, same LJSpeech sentence across vocoders, same speaker)
    '''
    p = path.replace("\\", "/")
    pl = p.lower()
    base = os.path.basename(p)

    if source == "fake_or_real":
        if "/fake/" in pl:
            label = 1
        elif "/real/" in pl:
            label = 0
        else:
            return None
        m = re.search(r"(file\d+)", base)
        fid = m.group(1) if m else base
        version = re.search(r"/(for-[a-z0-9]+)/", pl)
        # file id + label across for-original / for-norm / for-2sec / for-rerec = same underlying clip
        return label, ("real" if label == 0 else "for_tts"), (version.group(1) if version else "for"), f"for:{label}:{fid}"

    if source == "fakeaudio":
        m = re.search(r"generated_audio/([^/]+)", p)
        folder = m.group(1) if m else "unknown"
        stem = re.sub(r"_gen$", "", os.path.splitext(base)[0])
        corpus = "jsut" if folder.lower().startswith("jsut") else ("cv" if "common_voice" in folder.lower() else "lj")
        return 1, vocoder_family(folder), folder, f"{corpus}:{stem}"

    if source == "adarsh_tts":
        rel = p.split("audio-deepfake-detection-dataset/", 1)[-1]
        top = rel.split("/")[0]
        if "real_samples" in pl:
            spk = base.split("_")[0]           # LibriTTS naming: speaker_chapter_utt
            return 0, "real", "libritts", f"libritts:{spk}"
        return 1, "modern_tts", top, f"adarsh:{top}:{base}"

    if source == "unidpro":
        # decide on the FILE name — the dataset's folder name itself contains 'fake'
        label = 1 if re.search(r"synthetic|fake|clone|generated", base.lower()) else 0
        rel = p.split("real-vs-fake-human-voice-deepfake-audio/", 1)[-1]
        spk = os.path.dirname(rel)
        return label, ("unidpro_fake" if label else "real"), "unidpro", f"unidpro:{spk}"

    if source == "ljspeech_real":
        stem = os.path.splitext(base)[0]
        return 0, "real", "ljspeech", f"lj:{stem}"   # same group as that sentence's vocoded fakes
    return None


def discover():
    sources = dict(SOURCES)
    lj = find_ljspeech_real()
    if lj:
        sources["ljspeech_real"] = lj
        print(f"  [OK     ] ljspeech_real -> {lj}")
    else:
        print("  [ABSENT ] ljspeech_real — no LJ Speech dataset attached. The single-speaker shortcut "
              "cannot be ruled out; per-source results below will show whether it matters.")
    rows = []
    for source, root in sources.items():
        files = glob_audio(root)
        print(f"  {source:14s}: {len(files)} audio files")
        for f in files:
            d = describe(f, source)
            if d is None:
                continue
            label, gen, detail, group = d
            rows.append((f, label, source, gen, detail, group))
    df = pd.DataFrame(rows, columns=["file_path", "label", "source", "generator", "detail", "group"])
    return df.drop_duplicates("file_path").reset_index(drop=True)


all_df = discover()
print(f"\nDiscovered {len(all_df)} files")
print(all_df.groupby(["source", "label", "generator"]).size().to_string())

# %% [markdown]
# ## 2. Balanced 40k pool
# Rare, valuable sources (modern TTS, genuine LJSpeech, unidpro, LibriTTS reals) are kept in full;
# the rest is sampled at random to fill 20,000 real + 20,000 fake.

# %%
def build_pool(df):
    rng = np.random.RandomState(SEED)
    parts = []
    for label in (0, 1):
        sub = df[df.label == label]
        if label == 1:
            keep = sub[sub.generator.isin(["modern_tts", "unidpro_fake"])]
        else:
            keep = pd.concat([
                sub[sub.source.isin(["adarsh_tts", "unidpro"])],
                sub[sub.source == "ljspeech_real"].sample(
                    n=min(LJ_REAL_CAP, (sub.source == "ljspeech_real").sum()), random_state=SEED),
            ])
        if SMOKE_TEST:
            keep = keep.sample(n=min(len(keep), POOL_PER_CLASS // 3), random_state=SEED)
        rest = sub.drop(keep.index)
        need = max(0, min(POOL_PER_CLASS, len(sub)) - len(keep))
        parts += [keep, rest.sample(n=min(need, len(rest)), random_state=rng)]
    pool = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pool


pool = build_pool(all_df)
print(f"Pool: {len(pool)} files")
print(pool.groupby(["label", "source", "generator"]).size().to_string())

# %% [markdown]
# ## 3. Group-aware split (train 70 / val 15 / test 15)

# %%
strat = pool.label.astype(str) + "_" + pool.generator
sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
fold = np.zeros(len(pool), dtype=int)
for k, (_, te) in enumerate(sgkf.split(pool, strat, groups=pool.group)):
    fold[te] = k
pool["split"] = np.where(fold < TEST_FRAC_FOLDS, "test",
                np.where(fold < TEST_FRAC_FOLDS + VAL_FRAC_FOLDS, "val", "train"))

# hard check: no group may straddle two splits
straddle = pool.groupby("group").split.nunique()
assert (straddle == 1).all(), f"{(straddle > 1).sum()} groups straddle splits"
print("No recording / sentence / speaker group appears in more than one split. ✓")
print(pool.groupby(["split", "label"]).size().unstack().to_string())
print(pool.pivot_table(index="generator", columns="split", values="file_path", aggfunc="count").fillna(0).astype(int).to_string())
pool.to_csv(f"{OUT}/pool_with_splits.csv", index=False)

# %% [markdown]
# ## 4. Feature extraction (identical to the students' `extract_features`)
# Two copies per clip: **clean** (val / test) and **augmented** with their robustness
# augmentation (training), mirroring their train-aug / val-clean setup.

# %%
try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False
    print("pydub missing — MP3 augmentation skipped")


def mp3_roundtrip(y, sr, bitrate_kbps=64):
    if not PYDUB_OK:
        return y
    try:
        y16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
        seg = AudioSegment(y16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        buf = io.BytesIO(); seg.export(buf, format="mp3", bitrate=f"{bitrate_kbps}k"); buf.seek(0)
        out = np.array(AudioSegment.from_file(buf, format="mp3").get_array_of_samples()).astype(np.float32) / 32768.0
        if len(out) < len(y):
            out = np.pad(out, (0, len(y) - len(out)))
        return out[: len(y)]
    except Exception:
        return y


def add_noise_at_snr(y, snr_db):
    sig_power = np.mean(y ** 2) + 1e-12
    noise = np.random.normal(0, 1, size=y.shape)
    noise = noise * np.sqrt(sig_power / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return y + noise


def apply_reverb(y):
    length = np.random.choice([1000, 2500, 4000]); decay = np.random.choice([3.0, 6.0, 9.0])
    rir = (np.random.randn(length) * np.exp(-decay * np.linspace(0, 1, length))).astype(np.float32)
    wet = np.convolve(y, rir, mode="full")[: len(y)]
    return wet / (np.max(np.abs(wet)) + 1e-8) * (np.max(np.abs(y)) + 1e-8)


def telephone_bandlimit(y, sr):
    return librosa.resample(librosa.resample(y, orig_sr=sr, target_sr=8000), orig_sr=8000, target_sr=sr)


def apply_robustness_augmentation(y, sr):
    choice = np.random.choice(["mp3", "noise", "reverb", "telephone", "none"], p=[0.2, 0.25, 0.2, 0.15, 0.2])
    if choice == "mp3":
        y = mp3_roundtrip(y, sr, int(np.random.choice([32, 64, 128])))
    elif choice == "noise":
        y = add_noise_at_snr(y, float(np.random.choice([0, 10, 20])))
    elif choice == "reverb":
        y = apply_reverb(y)
    elif choice == "telephone":
        y = telephone_bandlimit(y, sr)
    return y


ROBUSTNESS_CONDITIONS = {
    "clean":          lambda y, sr: y,
    "mp3_32kbps":     lambda y, sr: mp3_roundtrip(y, sr, 32),
    "mp3_128kbps":    lambda y, sr: mp3_roundtrip(y, sr, 128),
    "noise_0dB":      lambda y, sr: add_noise_at_snr(y, 0),
    "noise_10dB":     lambda y, sr: add_noise_at_snr(y, 10),
    "noise_20dB":     lambda y, sr: add_noise_at_snr(y, 20),
    "reverb":         lambda y, sr: apply_reverb(y),
    "telephone_8kHz": lambda y, sr: telephone_bandlimit(y, sr),
}

SR, DUR, N_FRAMES = 16000, 3, 94


def features_from_wave(y, sr=SR):
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=512), ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=1024, hop_length=512)
    feat = np.stack([mel_db, mfcc], axis=0).astype(np.float32)
    if feat.shape[2] != N_FRAMES:   # guard: resampling can shift length by a sample
        feat = feat[:, :, :N_FRAMES] if feat.shape[2] > N_FRAMES else np.pad(feat, ((0, 0), (0, 0), (0, N_FRAMES - feat.shape[2])))
    return feat


def load_wave(path):
    y, _ = librosa.load(path, sr=SR, duration=DUR)
    if len(y) < SR * DUR:
        y = np.pad(y, (0, SR * DUR - len(y)))
    return y


def _extract(i, path, mode, seed):
    np.random.seed(seed)
    try:
        y = load_wave(path)
        if mode == "aug":
            y = apply_robustness_augmentation(y, SR)
        elif mode != "clean":
            y = ROBUSTNESS_CONDITIONS[mode](y, SR)
        return i, features_from_wave(y)
    except Exception:
        return i, None


def extract_all(paths, mode, chunk=2000):
    feats = np.zeros((len(paths), 2, 64, N_FRAMES), dtype=np.float32)
    ok = np.zeros(len(paths), dtype=bool)
    for s in tqdm(range(0, len(paths), chunk), desc=f"features[{mode}]"):
        res = Parallel(n_jobs=N_JOBS)(delayed(_extract)(i, paths[i], mode, SEED * 100003 + i)
                                      for i in range(s, min(s + chunk, len(paths))))
        for i, f in res:
            if f is not None:
                feats[i], ok[i] = f, True
    return feats, ok


t0 = time.time()
paths = pool.file_path.tolist()
X_clean, ok_clean = extract_all(paths, "clean")
X_aug, ok_aug = extract_all(paths, "aug")
pool["ok"] = ok_clean & ok_aug
print(f"Extraction done in {(time.time()-t0)/60:.1f} min | failed: {(~pool.ok).sum()} files")

# %% [markdown]
# ## 5. Model — copied unchanged from `deepshieldv5.ipynb` (one fix: MC-dropout)

# %%
ATTACK_TYPES = ["real", "vocoder_artifact", "tts_prosody", "splice_edit", "unknown_synthetic"]
ATTACK_TO_IDX = {t: i for i, t in enumerate(ATTACK_TYPES)}

def attack_type_of(generator, label):
    if label == 0:
        return ATTACK_TO_IDX["real"]
    if generator in ("melgan_family", "hifigan", "waveglow", "parallel_wavegan"):
        return ATTACK_TO_IDX["vocoder_artifact"]
    if generator in ("modern_tts", "conformer_fastspeech2", "for_tts"):
        return ATTACK_TO_IDX["tts_prosody"]
    return ATTACK_TO_IDX["unknown_synthetic"]

pool["attack"] = [attack_type_of(g, l) for g, l in zip(pool.generator, pool.label)]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)


class DeepShieldBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.attention = SqueezeExcitationAttention(64)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.attention(self.layer2(self.layer1(self.init_conv(x))))
        return self.global_pool(self.layer3(x)).flatten(1)


class DeepShieldMultiHead(nn.Module):
    def __init__(self, embed_dim=128, num_attack_types=len(ATTACK_TYPES), dropout_p=0.35):
        super().__init__()
        self.backbone = DeepShieldBackbone()
        self.spoof_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.45),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(64, 1))
        self.attack_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, num_attack_types))
        self.uncertainty_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mc_samples=1, return_embedding=False):
        embed = self.backbone(x)
        spoof_logits = self.spoof_head(embed)
        attack_logits = self.attack_head(embed)
        if mc_samples > 1:
            # FIX: F.dropout(training=True) so MC sampling stays active under model.eval()
            # (nn.Dropout is a no-op in eval mode, which made uncertainty always 0). Same fix as app.py.
            probs = torch.stack([torch.sigmoid(self.spoof_head(
                F.dropout(embed, p=self.uncertainty_dropout.p, training=True))) for _ in range(mc_samples)])
            mean_p, uncertainty = probs.mean(0), probs.std(0)
        else:
            mean_p = torch.sigmoid(spoof_logits)
            uncertainty = torch.zeros_like(mean_p)
        out = (spoof_logits, attack_logits, mean_p, uncertainty)
        return (embed, *out) if return_embedding else out


class OpenSetNoveltyScorer:
    '''Mahalanobis distance from the real-speech embedding manifold (unchanged).'''
    def fit(self, real_embeddings):
        self.mean = real_embeddings.mean(axis=0)
        cov = np.cov(real_embeddings, rowvar=False) + 1e-6 * np.eye(real_embeddings.shape[1])
        self.inv_cov = np.linalg.inv(cov)
        self.threshold = float(np.percentile(self.score_many(real_embeddings), 97.5))

    def score_many(self, E):
        d = E - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv_cov, d))

    def save(self, path):
        np.savez(path, mean=self.mean, inv_cov=self.inv_cov, threshold=self.threshold)


def risk_aware_gateway(p, unc, nov, nov_thr, low=0.30, high=0.70, unc_thr=0.15,
                       use_novelty=True, use_uncertainty=True):
    '''Vectorised version of the students' gateway: returns 'pass' / 'block' / 'review' per clip.'''
    route = np.where(p < low, "pass", np.where(p > high, "block", "review")).astype(object)
    if use_uncertainty:
        route[unc > unc_thr] = "review"
    if use_novelty:
        route[nov > nov_thr] = "review"
    return route

# %% [markdown]
# ## 6. Training (same recipe as the students: AdamW, warm-up + cosine, AMP, joint loss, best-val checkpoint)

# %%
XC = torch.from_numpy(X_clean)
XA = torch.from_numpy(X_aug)
Y = torch.tensor(pool.label.values, dtype=torch.float32)
A = torch.tensor(pool.attack.values, dtype=torch.long)


def batches(idx, bs=64, shuffle=False, balanced=False):
    idx = np.asarray(idx)
    if balanced:   # WeightedRandomSampler equivalent: equal expected real/fake per batch
        lab = pool.label.values[idx]
        w = np.where(lab == 1, 1.0 / max(1, (lab == 1).sum()), 1.0 / max(1, (lab == 0).sum()))
        order = np.random.choice(idx, size=len(idx), replace=True, p=w / w.sum())
    elif shuffle:
        order = np.random.permutation(idx)
    else:
        order = idx
    for s in range(0, len(order) - (bs - 1 if balanced else 0), bs):
        b = order[s:s + bs]
        if len(b):
            yield b


@torch.no_grad()
def infer(model, idx, X=XC, mc=MC_SAMPLES, bs=256):
    '''Deterministic spoof prob + MC-dropout uncertainty + embedding + attack probs for each clip.'''
    model.eval()
    P, U, E, AP = [], [], [], []
    for b in batches(idx, bs=bs):
        x = X[b].to(DEVICE, non_blocking=True)
        emb, _, atk_logits, p, _ = model(x, mc_samples=1, return_embedding=True)
        unc = model(x, mc_samples=mc)[3] if mc > 1 else torch.zeros_like(p)
        P.append(p.flatten().cpu()); U.append(unc.flatten().cpu()); E.append(emb.cpu()); AP.append(F.softmax(atk_logits, 1).cpu())
    return torch.cat(P).numpy(), torch.cat(U).numpy(), torch.cat(E).numpy(), torch.cat(AP).numpy()


def train_model(train_idx, val_idx, epochs, patience, tag, lr=1e-3, attack_w=0.4, ls=0.05, bs=64):
    seed_everything()
    model = DeepShieldMultiHead().to(DEVICE)
    spoof_crit, atk_crit = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(label_smoothing=ls)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm
                                              else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    scaler = torch.cuda.amp.GradScaler()
    best, best_ep, bad, ckpt = -1, -1, 0, f"{OUT}/_ckpt_{tag}.pth"
    yv = pool.label.values[val_idx]
    for ep in range(epochs):
        model.train()
        for b in batches(train_idx, bs=bs, balanced=True):
            x, y, a = XA[b].to(DEVICE), Y[b].to(DEVICE).unsqueeze(1), A[b].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                sl, al, _, _ = model(x)
                loss = spoof_crit(sl, y) + attack_w * atk_crit(al, a)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        p, _, _, _ = infer(model, val_idx, mc=1)
        acc = ((p > 0.5) == yv).mean()
        if acc > best:
            best, best_ep, bad = acc, ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"[{tag}] epoch {ep+1:02d} val_acc {acc:.4f}" + ("  *" if bad == 0 else ""))
        if bad >= patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"[{tag}] best val_acc {best:.4f} @ epoch {best_ep}")
    return model

# %% [markdown]
# ## 7. Metrics

# %%
def eer_of(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def auc_of(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def detection_metrics(y, p, thr=0.5):
    pred = (p > thr).astype(int)
    real, fake = y == 0, y == 1
    return {
        "n_real": int(real.sum()), "n_fake": int(fake.sum()),
        "accuracy": float((pred == y).mean()),
        "auc": auc_of(y, p), "eer": eer_of(y, p),
        "false_positive_rate": float(pred[real].mean()) if real.any() else float("nan"),   # genuine flagged fake
        "false_negative_rate": float(1 - pred[fake].mean()) if fake.any() else float("nan"),  # fake passed as genuine
    }


def gateway_table(y, p, unc, nov, nov_thr):
    '''Compares the plain spoof threshold with the gateway, adding uncertainty and novelty one at a time.'''
    configs = {
        "spoof head only (threshold 0.5)": np.where(p > 0.5, "block", "pass"),
        "+ confidence bands (0.3 / 0.7)":  risk_aware_gateway(p, unc, nov, nov_thr, use_novelty=False, use_uncertainty=False),
        "+ uncertainty":                   risk_aware_gateway(p, unc, nov, nov_thr, use_novelty=False),
        "+ novelty (full gateway)":        risk_aware_gateway(p, unc, nov, nov_thr),
    }
    rows = []
    for name, r in configs.items():
        real, fake = y == 0, y == 1
        rows.append({
            "configuration": name,
            "fakes_auto_passed": float((r[fake] == "pass").mean()) if fake.any() else float("nan"),
            "fakes_blocked": float((r[fake] == "block").mean()) if fake.any() else float("nan"),
            "fakes_sent_to_review": float((r[fake] == "review").mean()) if fake.any() else float("nan"),
            "genuine_auto_blocked": float((r[real] == "block").mean()) if real.any() else float("nan"),
            "genuine_sent_to_review": float((r[real] == "review").mean()) if real.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def fit_novelty(model, train_idx):
    real_idx = [i for i in train_idx if pool.label.values[i] == 0]
    _, _, E, _ = infer(model, real_idx, mc=1)
    ns = OpenSetNoveltyScorer(); ns.fit(E)
    return ns

# %% [markdown]
# ## 8. Main model — trained on all generators, evaluated once on the held-out test set
# This is also the model to deploy for the demo.

# %%
ok = pool.ok.values
tr_idx = np.where((pool.split == "train") & ok)[0]
va_idx = np.where((pool.split == "val") & ok)[0]
te_idx = np.where((pool.split == "test") & ok)[0]

main_model = train_model(tr_idx, va_idx, EPOCHS_MAIN, PATIENCE_MAIN, "main")
main_novelty = fit_novelty(main_model, tr_idx)

p, unc, E, AP = infer(main_model, te_idx)
nov = main_novelty.score_many(E)
y = pool.label.values[te_idx]
te = pool.iloc[te_idx].copy()
te["p"], te["unc"], te["nov"] = p, unc, nov
te["attack_pred"] = AP.argmax(1)
te.to_csv(f"{OUT}/test_predictions.csv", index=False)

main_metrics = detection_metrics(y, p)
print("MAIN TEST (group-clean, never used for model selection):")
print(json.dumps(main_metrics, indent=2))

# per source / per generator
rows = []
for (src, gen), g in te.groupby(["source", "generator"]):
    pred = (g.p > 0.5).astype(int)
    rows.append({"source": src, "generator": gen, "label": "real" if g.label.iloc[0] == 0 else "fake", "n": len(g),
                 "flagged_fake_rate": float(pred.mean()),
                 "error_rate": float((pred != g.label).mean())})
per_source = pd.DataFrame(rows).sort_values(["label", "error_rate"], ascending=[True, False])
per_source.to_csv(f"{OUT}/per_source_test.csv", index=False)
print("\nPer source / generator (error_rate = genuine flagged fake, or fake passed as genuine):")
print(per_source.to_string(index=False))

main_gateway = gateway_table(y, p, unc, nov, main_novelty.threshold)
main_gateway.to_csv(f"{OUT}/gateway_main_test.csv", index=False)
print("\nGateway on main test:\n", main_gateway.to_string(index=False))

# uncertainty usefulness: does it rise on mistakes?
wrong = ((p > 0.5).astype(int) != y)
unc_metrics = {
    "mean_uncertainty_correct": float(unc[~wrong].mean()),
    "mean_uncertainty_wrong": float(unc[wrong].mean()) if wrong.any() else float("nan"),
    "auc_uncertainty_flags_errors": auc_of(wrong.astype(int), unc),
    "share_of_clips_with_nonzero_uncertainty": float((unc > 1e-6).mean()),
}
print("\nUncertainty:", json.dumps(unc_metrics, indent=2))

# attack attribution among fakes
fk = y == 1
atk_true, atk_pred = pool.attack.values[te_idx][fk], te.attack_pred.values[fk]
attack_acc = float((atk_true == atk_pred).mean())
print(f"\nAttack-type attribution accuracy on test fakes: {attack_acc:.3f}")
print(pd.crosstab(pd.Series([ATTACK_TYPES[i] for i in atk_true], name="true"),
                  pd.Series([ATTACK_TYPES[i] for i in atk_pred], name="predicted")).to_string())

# save deployable artefacts (drop-in for deepshieldmodel/models/)
os.makedirs(f"{OUT}/models", exist_ok=True)
torch.save(main_model.state_dict(), f"{OUT}/models/deepshield_v5_multihead.pth")
torch.save(main_model.state_dict(), f"{OUT}/models/deepshield_v5_best.pth")
main_novelty.save(f"{OUT}/models/novelty_scorer.npz")
json.dump(ATTACK_TYPES, open(f"{OUT}/models/attack_types.json", "w"), indent=2)

# %% [markdown]
# ## 9. Naive-split baseline — how much did the original random split inflate accuracy?
# Same pool, same model, but split row-by-row at random (85/15) like the original notebook.

# %%
naive = None
if RUN_NAIVE_BASELINE:
    ok_idx = np.where(ok)[0]
    strat_n = (pool.label.astype(str) + "_" + pool.generator).values[ok_idx]
    vc = pd.Series(strat_n).value_counts()
    strat_n = np.where(np.isin(strat_n, vc[vc < 2].index), pool.label.astype(str).values[ok_idx], strat_n)
    n_tr, n_va = train_test_split(ok_idx, test_size=0.15, random_state=SEED, stratify=strat_n)
    naive_model = train_model(n_tr, n_va, EPOCHS_MAIN, PATIENCE_MAIN, "naive")
    pn, _, _, _ = infer(naive_model, n_va, mc=1)
    shared = pool.group.values[n_va]
    in_train = np.isin(shared, pool.group.values[n_tr])
    naive = {
        "naive_random_split_accuracy": float(((pn > 0.5) == pool.label.values[n_va]).mean()),
        "share_of_naive_val_with_same_recording_in_train": float(in_train.mean()),
        "group_clean_test_accuracy": main_metrics["accuracy"],
    }
    print(json.dumps(naive, indent=2))
    del naive_model; torch.cuda.empty_cache()

# %% [markdown]
# ## 10. Leave-One-Generator-Out (LOGO)
# For each generator family: remove it from training **and** validation entirely, train from scratch,
# then test on (a) its fakes from the val+test groups — never seen — vs. genuine test audio, and
# (b) the remaining (seen) generators on the same genuine test audio.

# %%
LOGO_FAMILIES = [g for g in ["modern_tts", "for_tts", "conformer_fastspeech2", "melgan_family",
                             "hifigan", "waveglow", "parallel_wavegan"]
                 if (pool.generator == g).sum() >= 100]
print("LOGO families:", LOGO_FAMILIES)

gen = pool.generator.values
test_real = te_idx[pool.label.values[te_idx] == 0]
logo_rows, logo_gateway = [], []
for fam in LOGO_FAMILIES:
    print(f"\n=== LOGO: holding out {fam} ===")
    l_tr = tr_idx[gen[tr_idx] != fam]
    l_va = va_idx[gen[va_idx] != fam]
    unseen_fake = np.where((gen == fam) & ok & pool.split.isin(["val", "test"]).values)[0]
    seen_fake = te_idx[(pool.label.values[te_idx] == 1) & (gen[te_idx] != fam)]

    m = train_model(l_tr, l_va, EPOCHS_LOGO, PATIENCE_LOGO, f"logo_{fam}")
    ns = fit_novelty(m, l_tr)

    res = {}
    for name, fakes in (("seen", seen_fake), ("unseen", unseen_fake)):
        idx = np.concatenate([test_real, fakes])
        pp, uu, ee, _ = infer(m, idx)
        nn_ = ns.score_many(ee)
        yy = pool.label.values[idx]
        dm = detection_metrics(yy, pp)
        res[name] = dm
        fk = yy == 1
        res[name + "_extra"] = {
            "novelty_auc_fake_vs_real": auc_of(yy, nn_),
            "fakes_flagged_novel": float((nn_[fk] > ns.threshold).mean()),
            "genuine_flagged_novel": float((nn_[~fk] > ns.threshold).mean()),
            "mean_uncertainty_fakes": float(uu[fk].mean()),
            "mean_uncertainty_genuine": float(uu[~fk].mean()),
        }
        gt = gateway_table(yy, pp, uu, nn_, ns.threshold)
        gt.insert(0, "held_out", fam); gt.insert(1, "fakes", name)
        logo_gateway.append(gt)

    row = {"held_out": fam, "n_unseen_fakes": len(unseen_fake)}
    for name in ("seen", "unseen"):
        for k in ("auc", "eer", "false_negative_rate", "false_positive_rate"):
            row[f"{name}_{k}"] = res[name][k]
        for k, v in res[name + "_extra"].items():
            row[f"{name}_{k}"] = v
    logo_rows.append(row)
    print(pd.Series(row).to_string())
    del m; torch.cuda.empty_cache()

logo_df = pd.DataFrame(logo_rows)
logo_gw = pd.concat(logo_gateway, ignore_index=True)
logo_df.to_csv(f"{OUT}/logo_results.csv", index=False)
logo_gw.to_csv(f"{OUT}/logo_gateway.csv", index=False)
print(logo_df[["held_out", "n_unseen_fakes", "seen_eer", "unseen_eer", "unseen_false_negative_rate",
               "unseen_fakes_flagged_novel", "unseen_novelty_auc_fake_vs_real"]].to_string(index=False))

# %% [markdown]
# ## 11. Robustness (main model, balanced test subset, with FPR / FNR, not just accuracy)
# Note: these degradations were also used as training augmentation, so this measures robustness to
# *known* degradations.

# %%
rng = np.random.RandomState(SEED)
half = ROBUSTNESS_N // 2
rb_idx = np.concatenate([rng.choice(te_idx[pool.label.values[te_idx] == c],
                                    size=min(half, (pool.label.values[te_idx] == c).sum()), replace=False) for c in (0, 1)])
rb_paths = pool.file_path.values[rb_idx].tolist()
rb_y = pool.label.values[rb_idx]
rob_rows = []
for cond in ROBUSTNESS_CONDITIONS:
    feats, okc = extract_all(rb_paths, cond)
    Xc = torch.from_numpy(feats[okc])
    pp, _, _, _ = infer(main_model, np.arange(okc.sum()), X=Xc, mc=1)
    rob_rows.append({"condition": cond, **detection_metrics(rb_y[okc], pp)})
rob_df = pd.DataFrame(rob_rows)
rob_df.to_csv(f"{OUT}/robustness.csv", index=False)
print(rob_df[["condition", "n_real", "n_fake", "accuracy", "eer", "false_positive_rate", "false_negative_rate"]].to_string(index=False))

# %% [markdown]
# ## 12. Plots

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
x = np.arange(len(logo_df))
ax[0].bar(x - 0.2, logo_df.seen_eer * 100, 0.4, label="seen generators")
ax[0].bar(x + 0.2, logo_df.unseen_eer * 100, 0.4, label="held-out (unseen) generator")
ax[0].set_xticks(x); ax[0].set_xticklabels(logo_df.held_out, rotation=30, ha="right")
ax[0].set_ylabel("Equal error rate (%) — lower is better"); ax[0].set_title("Leave-one-generator-out"); ax[0].legend()

u = logo_gw[logo_gw.fakes == "unseen"].groupby("configuration", sort=False).fakes_auto_passed.mean() * 100
ax[1].barh(u.index, u.values, color="#c05621")
ax[1].set_xlabel("Unseen-generator fakes auto-passed as genuine (%) — lower is better")
ax[1].set_title("What the gateway adds on unseen generators (mean over folds)")
plt.tight_layout(); plt.savefig(f"{OUT}/logo_and_gateway.png", dpi=150); plt.show()

plt.figure(figsize=(9, 4))
plt.bar(rob_df.condition, rob_df.eer * 100)
plt.xticks(rotation=30, ha="right"); plt.ylabel("EER (%)"); plt.title("Robustness under degradation")
plt.tight_layout(); plt.savefig(f"{OUT}/robustness.png", dpi=150); plt.show()

# %% [markdown]
# ## 13. Summary

# %%
def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100*v:.1f}%"

gw_unseen = logo_gw[logo_gw.fakes == "unseen"].groupby("configuration", sort=False)[
    ["fakes_auto_passed", "genuine_sent_to_review", "genuine_auto_blocked"]].mean()

summary = {
    "pool": {"n": int(len(pool)), "failed_extraction": int((~pool.ok).sum()),
             "ljspeech_real_included": bool((pool.source == "ljspeech_real").any()),
             "splits": pool.groupby("split").size().to_dict()},
    "main_test": main_metrics,
    "naive_vs_clean": naive,
    "uncertainty": unc_metrics,
    "attack_attribution_accuracy": attack_acc,
    "logo": logo_rows,
    "gateway_unseen_mean": gw_unseen.reset_index().to_dict(orient="records"),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, default=float)

lines = [
    "DEEPSHIELD VERIFICATION — PLAIN-ENGLISH SUMMARY",
    "",
    f"Clean test set ({main_metrics['n_real']} genuine, {main_metrics['n_fake']} fake, no recording shared with training):",
    f"  accuracy {pct(main_metrics['accuracy'])}, EER {pct(main_metrics['eer'])}",
    f"  genuine audio wrongly flagged fake: {pct(main_metrics['false_positive_rate'])}",
    f"  fake audio wrongly passed as genuine: {pct(main_metrics['false_negative_rate'])}",
]
if naive:
    lines += ["", f"Original-style random split gives {pct(naive['naive_random_split_accuracy'])} "
                  f"({pct(naive['share_of_naive_val_with_same_recording_in_train'])} of its validation clips "
                  f"had the same recording in training) vs {pct(naive['group_clean_test_accuracy'])} on the clean test."]
lines += ["", "Unseen generators (model never trained on them):"]
for r in logo_rows:
    lines.append(f"  {r['held_out']:22s} EER seen {pct(r['seen_eer'])} -> unseen {pct(r['unseen_eer'])}; "
                 f"unseen fakes passed as genuine {pct(r['unseen_false_negative_rate'])}; "
                 f"flagged novel {pct(r['unseen_fakes_flagged_novel'])}")
lines += ["", "Risk gateway on unseen generators (mean over held-out families):"]
for cfg, r in gw_unseen.iterrows():
    lines.append(f"  {cfg:34s} fakes auto-passed {pct(r.fakes_auto_passed)} | "
                 f"genuine sent to review {pct(r.genuine_sent_to_review)} | genuine auto-blocked {pct(r.genuine_auto_blocked)}")
lines += ["", f"Uncertainty: wrong predictions average {unc_metrics['mean_uncertainty_wrong']:.3f} vs "
              f"{unc_metrics['mean_uncertainty_correct']:.3f} for correct ones (AUC for spotting errors "
              f"{unc_metrics['auc_uncertainty_flags_errors']:.2f}; 0.5 = no better than chance).",
          f"Attack-type attribution accuracy on test fakes: {pct(attack_acc)}",
          "", "Robustness (EER): " + ", ".join(f"{r.condition} {pct(r.eer)}" for r in rob_df.itertuples())]
text = "\n".join(lines)
open(f"{OUT}/SUMMARY.txt", "w").write(text)
print(text)
print(f"\nAll outputs in {OUT}  (models/ is a drop-in replacement for deepshieldmodel/models/)")
