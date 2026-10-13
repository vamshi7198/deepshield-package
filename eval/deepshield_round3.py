# %% [markdown]
# # DeepShield V5 — Round 3: robust training, ensemble, and the equal-workload gateway test
#
# Same data, grouped split, architecture and heads as round 2 (weights stay drop-in compatible with
# `demo/app.py`). What changes, and why:
#
# | # | Change | Why (measured in round 2 or in the October 2026 laptop checks) |
# |---|---|---|
# | 1 | Train on several **random 3 s crops** per clip after trimming leading/trailing silence, instead of the first 3 s | The round-2 model only ever saw clip openings; its score swings from 0.00 to 0.99 when the window shifts by half a second |
# | 2 | Evaluate every clip with **overlapping windows** (0.5 s hop), clip score = mean of the most suspicious third | Matches the demo app; on 800 In-the-Wild clips this beat the old 2-12 window max (77.9% vs 75.4%) |
# | 3 | **Partial deepfakes**: 1-2 s of synthetic speech spliced into genuine recordings | The splice/edit attack class had no training data at all |
# | 4 | Optional **extra fake set** (`deepshield_extra_fakes`, e.g. Indian-accent TTS) | ~1,000 genuine Indian-accent clips in training, zero Indian-accent fakes |
# | 5 | **Ensemble** of N independently seeded models; uncertainty = spread across members | MC-dropout uncertainty barely beat the score margin and varied from run to run |
# | 6 | **Temperature scaling** fitted on validation | Raw probabilities sit at 0.00 / 1.00 almost always |
# | 7 | **Equal-workload gateway test**: combined risk score vs the score-only review rule, at fixed genuine-review budgets, on test, In-the-Wild and unseen generators | The patent's main claim needs this comparison; round 2 showed the review lane helps but not yet that the extra heads do |
#
# **Kaggle:** attach the round-2 datasets (and optionally a dataset containing a `deepshield_extra_fakes/` folder made
# with `tools/make_extra_fakes.py`), GPU T4 x2 or P100, Internet on. Run once with `SMOKE_TEST = True` (about
# 15 min), then build with `python eval/build_notebook.py --full --inline-core --src deepshield_round3.py
# --kernel kaggle_kernel_round3` for the real run. Results land in `/kaggle/working/deepshield_round3/`.
#
# **Laptop smoke test:** `DEEPSHIELD_INPUT=<mirror> DEEPSHIELD_OUT=<dir> DEEPSHIELD_ALLOW_CPU=1 python eval/deepshield_round3.py`
# with a small folder mirroring the Kaggle dataset layout (see `tools/make_local_mirror.py`).

# %%
import os, glob, re, json, math, random, warnings, io, time
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepshield_r3_core import *  # INLINE-CORE (build_notebook.py --inline-core pastes the module here)

warnings.filterwarnings("ignore")

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

INPUT_ROOT = os.environ.get("DEEPSHIELD_INPUT", "/kaggle/input")
OUT = os.environ.get("DEEPSHIELD_OUT", "/kaggle/working/deepshield_round3")
os.makedirs(OUT, exist_ok=True)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("GPU:", torch.cuda.get_device_name(0))
elif os.environ.get("DEEPSHIELD_ALLOW_CPU"):
    DEVICE = torch.device("cpu")
    print("CPU run (local smoke test)")
else:
    raise RuntimeError("No GPU. Settings → Accelerator → GPU T4 x2 (or P100), then restart and run all.")
USE_AMP = DEVICE.type == "cuda"

# ---- Run configuration ----
POOL_PER_CLASS   = 20000   # same 40k balanced pool as round 2
ITW_TEST_N       = 4000    # In-the-Wild clips (half real, half fake) for the external test
GATEWAY_TARGET_REVIEW = 0.15
GATEWAY_BUDGETS  = (0.10, 0.15, 0.25)   # genuine-review budgets for the equal-workload comparison
TEST_FRAC_FOLDS  = 3
VAL_FRAC_FOLDS   = 3
N_CROPS_TRAIN    = 3       # crop 0 is clean, the others are augmented
EVAL_MAX_WINDOWS = 24      # ~14.5 s of each val/test clip at a 0.5 s hop
MAX_LOAD_S       = 15
N_SPLICE         = 3000    # partial deepfakes added to the pool, split like the genuine clip they are built from
N_ENSEMBLE       = 3
EPOCHS_MAIN, PATIENCE_MAIN = 25, 6     # each epoch now sees N_CROPS_TRAIN crops per clip
EPOCHS_LOGO, PATIENCE_LOGO = 10, 3
MC_SAMPLES       = 20
MC_SEED          = 0
ROBUSTNESS_N     = 1000
N_JOBS           = os.cpu_count() or 2

TIME_BUDGET_H    = 10.5
T0 = time.time()
def elapsed_h():
    return (time.time() - T0) / 3600
def time_left_h():
    return TIME_BUDGET_H - elapsed_h()
def stamp(msg):
    line = f"[{elapsed_h():5.2f} h] {msg}"
    print(line)
    with open(f"{OUT}/progress.txt", "a") as f:
        f.write(line + "\n")

# SMOKE_TEST = True runs the whole pipeline on a tiny pool with 1-2 epochs to catch errors.
# The numbers it prints are meaningless. build_notebook.py --full switches it off.
SMOKE_TEST = True
if SMOKE_TEST:
    POOL_PER_CLASS, ITW_TEST_N, N_SPLICE = 1500, 200, 200
    EPOCHS_MAIN, PATIENCE_MAIN, EPOCHS_LOGO, PATIENCE_LOGO = 2, 2, 1, 1
    N_ENSEMBLE, N_CROPS_TRAIN, EVAL_MAX_WINDOWS = 2, 2, 6
    MC_SAMPLES, ROBUSTNESS_N = 5, 100
    print("SMOKE TEST MODE — pipeline check only, results are not meaningful")

json.dump({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, bool))},
          open(f"{OUT}/config.json", "w"), indent=2, default=str)

# %% [markdown]
# ## 1. Dataset discovery with corrected labels

# %%
DATASETS = {
    "adarsh_tts":   ("adarshsingh0903", "audio-deepfake-detection-dataset"),
    "unidpro":      ("unidpro", "real-vs-fake-human-voice-deepfake-audio"),
    "fakeaudio":    ("walimuhammadahmad", "fakeaudio"),
    "fake_or_real": ("mohammedabdeldayem", "the-fake-or-real-dataset"),
    "mlaad":        ("trapka", "mlaadthe-multi-languagaudioanti-spoofing-dataset"),
    "mlaad_2026":   ("smraj0198", "mlaad-english-500"),
    "vctk":         ("pratt3000", "vctk-corpus"),
    "accentdb":     ("imsparsh", "accentdb-core-extended"),
    "common_voice": ("mozillaorg", "common-voice"),
    "in_the_wild":  ("abdallamohamed312", "in-the-wild-audio-deepfake"),
}
EXTERNAL_TEST_SOURCES = {"in_the_wild"}   # never used for training, validation or model selection

# Only walk the relevant part of the big datasets (MLAAD has ~30 languages; Common Voice has several subsets)
SUBDIR_FILTER = {"mlaad": "fake/en", "common_voice": "cv-valid-train", "vctk": "wav48"}

def resolve_root(owner, slug):
    for cand in (f"{INPUT_ROOT}/datasets/{owner}/{slug}", f"{INPUT_ROOT}/{slug}"):
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

def _prune_towards(dirpath, dirnames, comps):
    '''
    Keeps only the child folders that can lead to the sub-folder path `comps` (e.g. ["fake", "en"]),
    so big datasets aren't walked in full. Wrapper folders above the first component are allowed.
    '''
    parts = dirpath.replace("\\", "/").rstrip("/").split("/")
    for k in range(len(comps), 0, -1):            # how much of comps the current path already ends with
        if parts[-k:] == comps[:k]:
            return [] if k == len(comps) else [d for d in dirnames if d == comps[k]]
    if comps[0] in dirnames:
        return [comps[0]]
    return dirnames                                # wrapper level (e.g. Mlaad_v5/mlaad_v5): keep looking


def glob_audio(root, subdir=None):
    '''One directory walk per source; hidden folders (e.g. .cache) skipped; optional sub-folder filter.'''
    hits = []
    comps = subdir.split("/") if subdir else None
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if comps and subdir not in dirpath.replace("\\", "/") + "/":
            # not inside the wanted sub-folder yet: only descend towards it
            dirnames[:] = _prune_towards(dirpath, dirnames, comps)
            continue
        hits.extend(os.path.join(dirpath, f) for f in files if f.lower().endswith(AUDIO_EXTENSIONS))
    return sorted(hits)


def find_ljspeech_real():
    '''Look for an attached LJ Speech dataset (any slug) by its first clip's file name.'''
    known = {os.path.realpath(r) for r in SOURCES.values()}
    cands = glob.glob(f"{INPUT_ROOT}/*") + glob.glob(f"{INPUT_ROOT}/datasets/*/*")
    for d in cands:
        if not os.path.isdir(d) or os.path.realpath(d) in known or d.rstrip("/").endswith("/datasets"):
            continue
        if any(os.path.realpath(d).startswith(k) for k in known):
            continue
        hits = glob.glob(os.path.join(d, "**", "LJ001-0001.wav"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def find_extra_fakes():
    '''Optional: any attached dataset containing a `deepshield_extra_fakes/` folder (tools/make_extra_fakes.py).'''
    hits = glob.glob(os.path.join(INPUT_ROOT, "**", "deepshield_extra_fakes"), recursive=True)
    return hits[0] if hits else None


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

    stem = os.path.splitext(base)[0]
    parts = p.split("/")

    if source in ("mlaad", "mlaad_2026"):
        # .../fake/en/<tts system>/<book>_<chapter>_<utt>.wav ; same sentence across systems = same group
        if "/fake/en/" not in p:
            return None
        system = p.split("/fake/en/", 1)[1].split("/")[0]
        family = "mlaad_tts" if source == "mlaad" else "tts_2026"
        return 1, family, system, f"mlaad:{stem}"

    if source == "vctk":
        spk = parts[-2]                                   # wav48/p225/p225_001.wav
        return 0, "real", "vctk", f"vctk:{spk}"

    if source == "accentdb":
        if "/data/" not in p:
            return None
        accent, spk = parts[-3], parts[-2]                # data/<accent>/speaker_01/<file>.wav
        return 0, "real", f"accentdb_{accent}", f"accentdb:{accent}:{spk}"

    if source == "common_voice":
        return 0, "real", "common_voice", f"cv:{stem}"

    if source == "extra_fakes":
        # deepshield_extra_fakes/<voice>/<text id>.wav ; the same sentence across voices is one group
        voice = parts[-2]
        return 1, "accent_tts", voice, f"extra:{stem}"

    if source == "in_the_wild":
        if "/fake/" in pl:
            return 1, "itw_fake", "in_the_wild", f"itw:{stem}"
        if "/real/" in pl:
            return 0, "real", "in_the_wild", f"itw:{stem}"
        return None
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
    extra = find_extra_fakes()
    if extra:
        sources["extra_fakes"] = extra
        print(f"  [OK     ] extra_fakes   -> {extra}")
    else:
        print("  [ABSENT ] extra_fakes   — no deepshield_extra_fakes/ folder attached (optional)")
    rows = []
    for source, root in sources.items():
        files = glob_audio(root, SUBDIR_FILTER.get(source))
        stamp(f"  scanned {source:14s}: {len(files)} audio files")
        for f in files:
            d = describe(f, source)
            if d is None:
                continue
            label, gen, detail, group = d
            rows.append((f, label, source, gen, detail, group))
    df = pd.DataFrame(rows, columns=["file_path", "label", "source", "generator", "detail", "group"])
    return df.drop_duplicates("file_path").reset_index(drop=True)


all_df = discover()
itw_df = all_df[all_df.source.isin(EXTERNAL_TEST_SOURCES)].reset_index(drop=True)
all_df = all_df[~all_df.source.isin(EXTERNAL_TEST_SOURCES)].reset_index(drop=True)
stamp("discovery done")
print(f"\nDiscovered {len(all_df)} files")
print(all_df.groupby(["source", "label", "generator"]).size().to_string())


# %% [markdown]
# ## 2. Balanced 40k pool, with a per-source budget
# Each source gets a cap so no single dataset (or speaker) dominates; `None` = take everything (rare sources).
# If a class ends up short, it is topped up at random from whatever is left.

# %%
SOURCE_CAPS = {
    0: {"fake_or_real": 6000, "ljspeech_real": 2500, "adarsh_tts": None, "unidpro": None,
        "vctk": 3500, "accentdb": 2500, "common_voice": 3500},
    1: {"fake_or_real": 4500, "fakeaudio": 5000, "adarsh_tts": None, "unidpro": None,
        "mlaad": 5500, "mlaad_2026": 3000, "extra_fakes": None},
}

def build_pool(df):
    scale = POOL_PER_CLASS / 20000
    parts = []
    for label in (0, 1):
        sub = df[df.label == label]
        taken = []
        for src, cap in SOURCE_CAPS[label].items():
            s_ = sub[sub.source == src]
            n = len(s_) if cap is None else min(len(s_), int(round(cap * scale)))
            if SMOKE_TEST and cap is None:
                n = min(n, int(round(600 * scale * 10)))
            taken.append(s_.sample(n=n, random_state=SEED))
        got = pd.concat(taken)
        target = min(POOL_PER_CLASS, len(sub))
        if len(got) > target:
            got = got.sample(n=target, random_state=SEED)
        elif len(got) < target:
            rest = sub.drop(got.index)
            got = pd.concat([got, rest.sample(n=min(target - len(got), len(rest)), random_state=SEED)])
        parts.append(got)
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


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
stamp("split done")

# %% [markdown]
# ## 3b. Partial deepfakes (splices)
# Each splice takes a genuine clip and a fake clip **from the same split**, and replaces 1-2 s of the genuine
# recording with loudness-matched synthetic speech. It is labelled fake (attack type `splice_edit`) and grouped
# with its genuine source clip, so no recording crosses splits. The audio is built at feature-extraction time
# from the two file paths and a per-row seed.

# %%
def add_splices(pool, n_total):
    rng = np.random.default_rng(SEED)
    rows = []
    for split, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        sp = pool[pool.split == split]
        gen_ok = sp[(sp.label == 0)]
        fake_ok = sp[(sp.label == 1)]
        n = min(int(round(n_total * frac)), len(gen_ok), len(fake_ok))
        if n == 0:
            continue
        g = gen_ok.sample(n=n, random_state=int(rng.integers(1 << 30)))
        f = fake_ok.sample(n=n, replace=len(fake_ok) < n, random_state=int(rng.integers(1 << 30)))
        for (_, gr), (_, fr) in zip(g.iterrows(), f.iterrows()):
            rows.append({"file_path": gr.file_path, "file_path2": fr.file_path, "label": 1, "source": "splice",
                         "generator": "splice", "detail": f"{gr.source}+{fr.generator}",
                         "group": f"splice:{gr.group}", "split": split})
    sp_df = pd.DataFrame(rows)
    out = pd.concat([pool.assign(file_path2=None), sp_df], ignore_index=True)
    # integrity: each splice's genuine group lives only in the splice's own split
    g_split = pool.drop_duplicates("group").set_index("group").split
    assert all(g_split[g[len("splice:"):]] == s for g, s in zip(sp_df.group, sp_df.split)), "splice crosses splits"
    return out


pool = add_splices(pool, N_SPLICE)
print(pool.groupby(["split", "source"]).size().unstack(fill_value=0).loc[:, ["splice"]].T.to_string())
pool.to_csv(f"{OUT}/pool_with_splits.csv", index=False)
stamp("splices added")


# %% [markdown]
# ## 4. Feature extraction: random training crops, overlapping evaluation windows
# Training: N_CROPS_TRAIN random 3 s crops per clip after silence trimming (crop 0 clean, the others with the
# round-2 robustness augmentation). Validation / test / In-the-Wild: every overlapping window, scored exactly as
# the demo app scores an upload. Features are stored as float16 to fit in memory.

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


def load_row(path, path2, seed):
    """Audio for one pool row: a trimmed recording, or for a splice row the genuine clip with a fake stretch
    inserted (same seed -> same splice). Returns (audio, (start, end) of the inserted segment or None)."""
    y = load_trimmed(path, MAX_LOAD_S)
    if isinstance(path2, str) and path2:
        fake = load_trimmed(path2, MAX_LOAD_S)
        return splice(y, fake, np.random.default_rng(seed))
    return y, None


def _train_crops(i, path, path2, seed):
    try:
        y, cover = load_row(path, path2, seed)
        rng = np.random.default_rng(seed + 1)
        np.random.seed(seed % (2 ** 32))          # apply_robustness_augmentation draws from np.random
        out = []
        for k in range(N_CROPS_TRAIN):
            c = random_crop(y, rng, must_cover=cover)
            if k > 0:
                c = apply_robustness_augmentation(c, SR)
            out.append(features(c).astype(np.float16))
        return i, np.stack(out)
    except Exception:
        return i, None


def _eval_windows(i, path, path2, seed, cond):
    try:
        y, cover = load_row(path, path2, seed)
        if cond != "clean":
            np.random.seed(seed % (2 ** 32))
            y = ROBUSTNESS_CONDITIONS[cond](y, SR).astype(np.float32)
        starts = window_starts(len(y), max_windows=EVAL_MAX_WINDOWS)
        feats = np.stack([features(y[s:s + WIN]) for s in starts]).astype(np.float16)
        return i, feats, np.array(starts) / SR, cover, len(y) / SR
    except Exception:
        return i, None, None, None, None


def row_seed(i):
    return SEED * 100003 + int(i)


def extract_train(df, idx, chunk=2000):
    """N_CROPS_TRAIN feature grids per clip. Returns X [n_ok*K, 2, 64, 94] float16, crop_row [n_ok*K], ok [len(idx)]."""
    pos = {int(j): k for k, j in enumerate(idx)}
    X, rows, ok = [], [], np.zeros(len(idx), bool)
    for s in tqdm(range(0, len(idx), chunk), desc="train crops"):
        part = idx[s:s + chunk]
        res = Parallel(n_jobs=N_JOBS)(delayed(_train_crops)(j, df.file_path.values[j], df.file_path2.values[j], row_seed(j))
                                      for j in part)
        for j, f in res:
            if f is not None:
                X.append(f); rows.extend([j] * len(f)); ok[pos[int(j)]] = True
    return np.concatenate(X), np.array(rows), ok


class WindowSet:
    """Overlapping-window features for a list of clips (val / test / In-the-Wild / robustness)."""
    def __init__(self, df, idx, cond="clean", chunk=2000):
        idx = np.asarray(idx)
        pos = {int(j): k for k, j in enumerate(idx)}
        X, clip, start, self.cover, self.dur = [], [], [], {}, {}
        ok = np.zeros(len(idx), bool)
        for s in tqdm(range(0, len(idx), chunk), desc=f"windows[{cond}]"):
            part = idx[s:s + chunk]
            res = Parallel(n_jobs=N_JOBS)(delayed(_eval_windows)(j, df.file_path.values[j], df.file_path2.values[j],
                                                                  row_seed(j), cond) for j in part)
            for j, f, st, cov, dur in res:
                if f is None:
                    continue
                k = pos[int(j)]
                ok[k] = True
                X.append(f); clip.extend([k] * len(f)); start.extend(st)
                self.cover[k], self.dur[k] = cov, dur
        self.idx, self.ok = idx, ok
        self.X = torch.from_numpy(np.concatenate(X)) if X else torch.zeros((0, 2, 64, N_FRAMES), dtype=torch.float16)
        self.clip = np.array(clip, dtype=int)      # position in idx of each window's clip
        self.start = np.array(start)
        self.n = len(idx)


t0 = time.time()
tr_all = np.where(pool.split == "train")[0]
va_all = np.where(pool.split == "val")[0]
te_all = np.where(pool.split == "test")[0]
XT_np, crop_row, ok_tr = extract_train(pool, tr_all)
XT = torch.from_numpy(XT_np); del XT_np
VW = WindowSet(pool, va_all)
TW = WindowSet(pool, te_all)
pool["ok"] = False
pool.loc[tr_all[ok_tr], "ok"] = True
pool.loc[va_all[VW.ok], "ok"] = True
pool.loc[te_all[TW.ok], "ok"] = True
stamp("feature extraction done")
print(f"Extraction done in {(time.time()-t0)/60:.1f} min | failed: {(~pool.ok).sum()} files | "
      f"train crops {len(XT)} | val windows {len(VW.X)} ({len(VW.X)/max(1, VW.ok.sum()):.1f}/clip) | "
      f"test windows {len(TW.X)} | feature memory {(XT.numel() + VW.X.numel() + TW.X.numel()) * 2 / 1e9:.1f} GB")

# %% [markdown]
# ## 5. Model — unchanged from round 2 (weights stay drop-in compatible with the demo app)

# %%
ATTACK_TYPES = ["real", "vocoder_artifact", "tts_prosody", "splice_edit", "unknown_synthetic"]
ATTACK_TO_IDX = {t: i for i, t in enumerate(ATTACK_TYPES)}

def attack_type_of(generator, label):
    if label == 0:
        return ATTACK_TO_IDX["real"]
    if generator in ("melgan_family", "hifigan", "waveglow", "parallel_wavegan"):
        return ATTACK_TO_IDX["vocoder_artifact"]
    if generator == "splice":
        return ATTACK_TO_IDX["splice_edit"]
    if generator in ("modern_tts", "conformer_fastspeech2", "for_tts", "mlaad_tts", "tts_2026", "accent_tts"):
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
# ## 6. Training on random crops; window-level inference; clip scores
# Same recipe as rounds 1-2 (AdamW, warm-up + cosine, AMP, joint spoof + attack loss, best-validation
# checkpoint), but each step samples crops, and the checkpoint is chosen by **clip-level** validation
# accuracy computed exactly as the demo app scores a clip.

# %%
Y = torch.tensor(pool.label.values, dtype=torch.float32)
A = torch.tensor(pool.attack.values, dtype=torch.long)


def crop_batches(crop_idx, bs=64):
    '''Balanced sampling (equal expected genuine / fake per batch) over training crops.'''
    lab = pool.label.values[crop_row[crop_idx]]
    w = np.where(lab == 1, 1.0 / max(1, (lab == 1).sum()), 1.0 / max(1, (lab == 0).sum()))
    order = np.random.choice(crop_idx, size=len(crop_idx), replace=True, p=w / w.sum())
    for s in range(0, len(order) - bs + 1, bs):
        yield order[s:s + bs]


@torch.no_grad()
def infer_windows(model, X, bs=512, want_embed=False):
    '''Deterministic pass over a stack of window grids: window logits, attack probs (and embeddings).'''
    model.eval()
    L, AP, E = [], [], []
    for s in range(0, len(X), bs):
        x = X[s:s + bs].to(DEVICE, non_blocking=True).float()
        emb, logits, atk, _, _ = model(x, mc_samples=1, return_embedding=True)
        L.append(logits[:, 0].float().cpu()); AP.append(F.softmax(atk.float(), 1).cpu())
        if want_embed:
            E.append(emb.float().cpu())
    L = torch.cat(L).numpy() if L else np.zeros(0)
    AP = torch.cat(AP).numpy() if AP else np.zeros((0, len(ATTACK_TYPES)))
    return (L, AP, torch.cat(E).numpy()) if want_embed else (L, AP)


@torch.no_grad()
def mc_uncertainty(model, X, rows, bs=256):
    '''Seeded MC-dropout spread of the spoof probability for the given windows (one per clip).'''
    model.eval()
    out = []
    for s in range(0, len(rows), bs):
        x = X[rows[s:s + bs]].to(DEVICE).float()
        torch.manual_seed(MC_SEED)
        out.append(model(x, mc_samples=MC_SAMPLES)[3][:, 0].float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def window_clip_scores(window_logits, ws, temperature=1.0):
    return clip_scores(1 / (1 + np.exp(-window_logits / temperature)), ws.clip, ws.n)


def train_model(train_rows_mask, val_ws, val_mask, epochs, patience, tag, seed=SEED, lr=1e-3, attack_w=0.4, ls=0.05, bs=64):
    '''train_rows_mask: bool over pool rows allowed in training; val_ws/val_mask: validation windows + clips to score.'''
    seed_everything(seed)
    crop_idx = np.where(train_rows_mask[crop_row])[0]
    model = DeepShieldMultiHead().to(DEVICE)
    spoof_crit, atk_crit = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(label_smoothing=ls)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm
                                              else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best, best_ep, bad, ckpt = -1, -1, 0, f"{OUT}/_ckpt_{tag}.pth"
    yv = pool.label.values[val_ws.idx]
    for ep in range(epochs):
        model.train()
        for b in crop_batches(crop_idx, bs=bs):
            r = crop_row[b]
            x, y, a = XT[b].to(DEVICE).float(), Y[r].to(DEVICE).unsqueeze(1), A[r].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                sl, al, _, _ = model(x)
                loss = spoof_crit(sl.float(), y) + attack_w * atk_crit(al.float(), a)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        lv, _ = infer_windows(model, val_ws.X)
        pv = window_clip_scores(lv, val_ws)
        m = val_mask & val_ws.ok
        acc = ((pv[m] > 0.5) == yv[m]).mean()
        if acc > best:
            best, best_ep, bad = acc, ep + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"[{tag}] epoch {ep+1:02d} clip val_acc {acc:.4f}" + ("  *" if bad == 0 else ""))
        if bad >= patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"[{tag}] best clip val_acc {best:.4f} @ epoch {best_ep}")
    return model

# %% [markdown]
# ## 7. Metrics and the novelty scorer

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
        "balanced_accuracy": float(((pred[real] == 0).mean() + (pred[fake] == 1).mean()) / 2) if real.any() and fake.any() else float("nan"),
        "auc": auc_of(y, p), "eer": eer_of(y, p),
        "false_positive_rate": float(pred[real].mean()) if real.any() else float("nan"),
        "false_negative_rate": float(1 - pred[fake].mean()) if fake.any() else float("nan"),
    }


def fit_novelty(model, rows_mask):
    '''Gaussian of the clean training crops (crop 0) of genuine clips.'''
    sel = np.where(rows_mask[crop_row] & (pool.label.values[crop_row] == 0))[0]
    first = sel[np.r_[True, crop_row[sel][1:] != crop_row[sel][:-1]]]          # crop 0 of each clip
    _, _, E = infer_windows(model, XT[first], want_embed=True)
    ns = OpenSetNoveltyScorer(); ns.fit(E)
    return ns


class Scored:
    '''Everything the gateway needs for one evaluation set, from an ensemble (or a single model).'''
    def __init__(self, models, novelty, ws, temperature=1.0):
        Ls, APs, E0 = [], [], None
        for k, m in enumerate(models):
            if k == 0:
                L, AP, E0 = infer_windows(m, ws.X, want_embed=True)
            else:
                L, AP = infer_windows(m, ws.X)
            Ls.append(L); APs.append(AP)
        Ls = np.stack(Ls)                                            # [members, windows]
        self.window_logit = Ls.mean(0)
        self.p = window_clip_scores(self.window_logit, ws, temperature)
        self.p_raw = window_clip_scores(self.window_logit, ws, 1.0)
        member_p = np.stack([window_clip_scores(L, ws, 1.0) for L in Ls])
        self.ens_std = member_p.std(0) if len(models) > 1 else np.zeros(ws.n)
        bw = best_window(1 / (1 + np.exp(-self.window_logit)), ws.clip, ws.n)
        has = bw >= 0
        self.best_window = bw
        self.mc_unc = np.full(ws.n, np.nan); self.mc_unc[has] = mc_uncertainty(models[0], ws.X, bw[has])
        self.novelty = np.full(ws.n, np.nan); self.novelty[has] = novelty.score_many(E0[bw[has]])
        self.attack_probs = np.full((ws.n, len(ATTACK_TYPES)), np.nan)
        self.attack_probs[has] = np.mean(APs, axis=0)[bw[has]]
        self.best_start = np.full(ws.n, np.nan); self.best_start[has] = ws.start[bw[has]]
        self.ok = ws.ok & has

    def feats(self, names, m=None):
        m = self.ok if m is None else m
        f = risk_features(self.p[m], ens_std=self.ens_std[m], mc_unc=self.mc_unc[m], novelty=self.novelty[m])
        return {k: f[k] for k in names}

# %% [markdown]
# ## 8. Main ensemble — trained on all generators, evaluated once on the held-out test set
# N models from different seeds on the same data. The first member doubles as the single-model
# drop-in for the current demo app.

# %%
ok = pool.ok.values
train_mask = (pool.split.values == "train") & ok
members = []
for k in range(N_ENSEMBLE):
    members.append(train_model(train_mask, VW, np.ones(VW.n, bool), EPOCHS_MAIN, PATIENCE_MAIN, f"main{k}", seed=SEED + 1000 * k))
    stamp(f"ensemble member {k} trained")
main_novelty = fit_novelty(members[0], train_mask)

# temperature on validation (clip-level log-loss of the ensemble mean logit)
_val_raw = Scored(members, main_novelty, VW)
TEMPERATURE = fit_temperature(_val_raw.window_logit, VW.clip, pool.label.values[VW.idx]) if VW.ok.any() else 1.0
print(f"Temperature: {TEMPERATURE:.2f}")
SV = Scored(members, main_novelty, VW, TEMPERATURE)
ST = Scored(members, main_novelty, TW, TEMPERATURE)
yv, yt = pool.label.values[VW.idx], pool.label.values[TW.idx]

main_metrics = detection_metrics(yt[ST.ok], ST.p[ST.ok])
single_metrics = detection_metrics(yt[ST.ok], Scored(members[:1], main_novelty, TW).p[ST.ok])
print("MAIN TEST, ensemble:", json.dumps(main_metrics, indent=2))
print("MAIN TEST, member 0 alone:", json.dumps(single_metrics, indent=2))

te = pool.iloc[TW.idx].copy()
te["p"], te["ens_std"], te["mc_unc"], te["nov"], te["best_start"] = ST.p, ST.ens_std, ST.mc_unc, ST.novelty, ST.best_start
te["attack_pred"] = np.nanargmax(np.nan_to_num(ST.attack_probs, nan=-1), 1)
te["ok_scored"] = ST.ok
te.to_csv(f"{OUT}/test_predictions.csv", index=False)

rows = []
for (src, gen), g in te[te.ok_scored].groupby(["source", "generator"]):
    pred = (g.p > 0.5).astype(int)
    rows.append({"source": src, "generator": gen, "label": "real" if g.label.iloc[0] == 0 else "fake", "n": len(g),
                 "flagged_fake_rate": float(pred.mean()), "error_rate": float((pred != g.label).mean())})
per_source = pd.DataFrame(rows).sort_values(["label", "error_rate"], ascending=[True, False])
per_source.to_csv(f"{OUT}/per_source_test.csv", index=False)
print(per_source.to_string(index=False))

# partial deepfakes: detected, and is the most suspicious window on the inserted segment?
sp = np.where((pool.source.values[TW.idx] == "splice") & ST.ok)[0]
splice_report = None
if len(sp):
    hit = []
    for k in sp:
        a, b = TW.cover[k]
        s0 = ST.best_start[k] * SR
        hit.append(min(s0 + WIN, b) - max(s0, a) > 0)
    splice_report = {"n": int(len(sp)), "detected": float((ST.p[sp] > 0.5).mean()),
                     "most_suspicious_window_overlaps_splice": float(np.mean(hit))}
    print("Splices:", splice_report)

fk = (yt == 1) & ST.ok
atk_true, atk_pred = pool.attack.values[TW.idx][fk], te.attack_pred.values[fk]
attack_acc = float((atk_true == atk_pred).mean()) if fk.any() else float("nan")
print(f"Attack-type attribution accuracy on test fakes: {attack_acc:.3f}")

# legacy rule gateway (novelty -> MC uncertainty -> 0.3/0.7 bands), uncertainty threshold calibrated on validation
def calibrate_unc_threshold(y, p, unc, nov, nov_thr, target=GATEWAY_TARGET_REVIEW):
    real = y == 0
    for t in np.unique(np.concatenate([np.quantile(unc[real], np.linspace(0, 1, 401)), [0.15]])):
        if (risk_aware_gateway(p, unc, nov, nov_thr, unc_thr=t)[real] == "review").mean() <= target:
            return float(t)
    return float(np.nanmax(unc))


main_unc_thr = calibrate_unc_threshold(yv[SV.ok], SV.p[SV.ok], SV.mc_unc[SV.ok], SV.novelty[SV.ok], main_novelty.threshold)
legacy_test = route_rates(yt[ST.ok], risk_aware_gateway(ST.p[ST.ok], ST.mc_unc[ST.ok], ST.novelty[ST.ok],
                                                         main_novelty.threshold, unc_thr=main_unc_thr))
print("Legacy rule gateway on test:", legacy_test)

# %% [markdown]
# ## 8b. External real-world test: In-the-Wild (never used for training, validation or calibration)

# %%
SI, itw, IW = None, None, None
if len(itw_df):
    try:
        itw = pd.concat([g.sample(n=min(ITW_TEST_N // 2, len(g)), random_state=SEED) for _, g in itw_df.groupby("label")]).reset_index(drop=True)
        itw["file_path2"] = None
        IW = WindowSet(itw, np.arange(len(itw)))
        SI = Scored(members, main_novelty, IW, TEMPERATURE)
        yi = itw.label.values
        itw_metrics = detection_metrics(yi[SI.ok], SI.p[SI.ok])
        itw["p"], itw["ens_std"], itw["mc_unc"], itw["nov"], itw["ok_scored"] = SI.p, SI.ens_std, SI.mc_unc, SI.novelty, SI.ok
        itw.to_csv(f"{OUT}/in_the_wild_predictions.csv", index=False)
        print("IN-THE-WILD:", json.dumps(itw_metrics, indent=2))
        stamp("in-the-wild done")
    except Exception as e:
        itw_metrics = None
        stamp(f"in-the-wild FAILED: {e!r}")
else:
    itw_metrics = None

# %% [markdown]
# ## 8c. Equal-workload gateway comparison (the patent's main claim)
# For each budget, every routing signal gets a review cut-off chosen on **validation** so that exactly that share
# of genuine validation clips goes to review; the cut-off is then frozen and applied to test and In-the-Wild.
# "score only" reviews the clips whose ensemble score is closest to 0.5. The combined risk score is a logistic
# model fitted on validation clips to predict a wrong verdict from the listed signals.

# %%
SIGNALS = {
    "score only": ["margin"],
    "score + ensemble spread": ["margin", "ens_std"],
    "score + ensemble spread + novelty": ["margin", "ens_std", "novelty"],
    "all signals": ["margin", "ens_std", "mc_unc", "novelty"],
}
if N_ENSEMBLE < 2:
    SIGNALS = {k: [n for n in v if n != "ens_std"] for k, v in SIGNALS.items()}
    SIGNALS = {k: v for k, v in SIGNALS.items() if v}
all_names = sorted({n for v in SIGNALS.values() for n in v})
eval_sets = {"test": (ST.feats(all_names), yt[ST.ok], ST.p[ST.ok])}
if SI is not None:
    eval_sets["in_the_wild"] = (SI.feats(all_names), yi[SI.ok], SI.p[SI.ok])
gw_rows = []
for budget in GATEWAY_BUDGETS:
    gw_rows += equal_workload_comparison(eval_sets, budget, SV.feats(all_names), yv[SV.ok], SV.p[SV.ok], SIGNALS)
gateway_eq = pd.DataFrame(gw_rows)
gateway_eq.to_csv(f"{OUT}/gateway_equal_workload.csv", index=False)
print(gateway_eq.to_string(index=False))

# shipped routing: the combined risk over all signals, cut-off at the target budget (validation)
cr_all = CombinedRisk(SIGNALS["all signals"]).fit(SV.feats(all_names), yv[SV.ok], SV.p[SV.ok])
cr_thr = review_threshold(cr_all.risk(SV.feats(all_names)), yv[SV.ok], GATEWAY_TARGET_REVIEW)

# %% [markdown]
# ## 8d. Save deployable artefacts
# `models/` replaces `demo/models/`. `deepshield_v5_multihead.pth` is member 0, so the current app still works;
# the updated app also loads the other members, the temperature and the combined-risk routing from
# `gateway_calibration.json`.

# %%
os.makedirs(f"{OUT}/models", exist_ok=True)
member_files = []
for k, m in enumerate(members):
    fn = f"deepshield_v5_member{k}.pth"
    torch.save(m.state_dict(), f"{OUT}/models/{fn}"); member_files.append(fn)
torch.save(members[0].state_dict(), f"{OUT}/models/deepshield_v5_multihead.pth")
torch.save(members[0].state_dict(), f"{OUT}/models/deepshield_v5_best.pth")
main_novelty.save(f"{OUT}/models/novelty_scorer.npz")
json.dump(ATTACK_TYPES, open(f"{OUT}/models/attack_types.json", "w"), indent=2)
json.dump({
    "uncertainty_threshold": main_unc_thr, "novelty_threshold": main_novelty.threshold,
    "target_genuine_review_rate": GATEWAY_TARGET_REVIEW,
    "temperature": TEMPERATURE, "ensemble_members": member_files,
    "preprocessing": {"trim_silence": True, "trim_top_db": TRIM_TOP_DB, "window_hop_s": EVAL_HOP_S,
                      "top_window_fraction": TOP_WINDOW_FRACTION},
    "combined_risk": {**cr_all.to_json(), "threshold": cr_thr, "mc_samples": MC_SAMPLES, "mc_seed": MC_SEED},
}, open(f"{OUT}/models/gateway_calibration.json", "w"), indent=2)
stamp("models saved")

# %% [markdown]
# ## 9. Leave-One-Generator-Out (LOGO), single model per fold
# The held-out family is removed from training and validation, the model is trained from scratch, and its
# fakes from val+test are scored as never-seen. The equal-workload comparison is repeated per fold with the
# signals a single model has (score margin, MC-dropout uncertainty, novelty), cut-offs fixed on the fold's
# validation data.

# %%
LOGO_FAMILIES = [g for g in ["tts_2026", "modern_tts", "mlaad_tts", "for_tts", "conformer_fastspeech2",
                             "melgan_family", "hifigan", "waveglow", "parallel_wavegan", "accent_tts"]
                 if (pool.generator == g).sum() >= 100]
print("LOGO families (priority order):", LOGO_FAMILIES)
gen = pool.generator.values
gen_v, gen_t = gen[VW.idx], gen[TW.idx]
LOGO_SIGNALS = {"score only": ["margin"], "score + novelty": ["margin", "novelty"],
                "score + MC uncertainty + novelty": ["margin", "mc_unc", "novelty"]}
logo_rows, logo_eq, logo_skipped, fold_hours = [], [], [], []
RESERVE_H = 0.8
for fam in LOGO_FAMILIES:
    est = 1.3 * max(fold_hours) if fold_hours else 0.6
    if time_left_h() - RESERVE_H < est:
        logo_skipped.append(fam); stamp(f"LOGO {fam}: skipped (time budget)"); continue
    t_fold = time.time()
    print(f"\n=== LOGO: holding out {fam} ===")
    m = train_model(train_mask & (gen != fam), VW, gen_v != fam, EPOCHS_LOGO, PATIENCE_LOGO, f"logo_{fam}")
    ns = fit_novelty(m, train_mask & (gen != fam))
    sv, st_ = Scored([m], ns, VW), Scored([m], ns, TW)
    vm = sv.ok & (gen_v != fam)
    unseen_v = sv.ok & (gen_v == fam)                                   # never-seen fakes in val ...
    unseen_t = st_.ok & (gen_t == fam)                                  # ... and in test
    real_t = st_.ok & (yt == 0)
    seen_t = st_.ok & (yt == 1) & (gen_t != fam)
    sets = {
        "seen": (st_.feats(["margin", "mc_unc", "novelty"], real_t | seen_t), yt[real_t | seen_t], st_.p[real_t | seen_t]),
        "unseen": ({k: np.concatenate([st_.feats(["margin", "mc_unc", "novelty"], real_t)[k],
                                       st_.feats(["margin", "mc_unc", "novelty"], unseen_t)[k],
                                       sv.feats(["margin", "mc_unc", "novelty"], unseen_v)[k]])
                    for k in ["margin", "mc_unc", "novelty"]},
                   np.concatenate([yt[real_t], yt[unseen_t], yv[unseen_v]]),
                   np.concatenate([st_.p[real_t], st_.p[unseen_t], sv.p[unseen_v]])),
    }
    row = {"held_out": fam, "n_unseen_fakes": int(unseen_t.sum() + unseen_v.sum())}
    for name, (f, yy, pp) in sets.items():
        mtr = detection_metrics(yy, pp)
        row.update({f"{name}_{k}": mtr[k] for k in ("auc", "eer", "false_negative_rate", "false_positive_rate")})
        row[f"{name}_novelty_auc_fake_vs_real"] = auc_of(yy, f["novelty"])
        row[f"{name}_fakes_flagged_novel"] = float((f["novelty"][yy == 1] > ns.threshold).mean()) if (yy == 1).any() else float("nan")
    for budget in GATEWAY_BUDGETS:
        for r in equal_workload_comparison(sets, budget, sv.feats(["margin", "mc_unc", "novelty"], vm), yv[vm], sv.p[vm], LOGO_SIGNALS):
            logo_eq.append({"held_out": fam, **r})
    logo_rows.append(row)
    print(pd.Series(row).to_string())
    del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pd.DataFrame(logo_rows).to_csv(f"{OUT}/logo_results.csv", index=False)
    pd.DataFrame(logo_eq).to_csv(f"{OUT}/logo_gateway_equal_workload.csv", index=False)
    fold_hours.append((time.time() - t_fold) / 3600)
    stamp(f"LOGO {fam}: done in {fold_hours[-1]:.2f} h")
logo_df, logo_eq_df = pd.DataFrame(logo_rows), pd.DataFrame(logo_eq)

# %% [markdown]
# ## 10. Robustness (ensemble, balanced test subset, degradation applied to the whole clip)

# %%
rob_df = pd.DataFrame()
if time_left_h() > 0.4:
    rng = np.random.RandomState(SEED)
    cand = [np.where(ST.ok & (yt == c) & (pool.source.values[TW.idx] != "splice"))[0] for c in (0, 1)]
    pick = np.sort(np.concatenate([rng.choice(c, size=min(ROBUSTNESS_N // 2, len(c)), replace=False) for c in cand]))
    rb_rows = TW.idx[pick]
    rob = []
    for cond in ROBUSTNESS_CONDITIONS:
        ws = WindowSet(pool, rb_rows, cond)
        s = Scored(members, main_novelty, ws, TEMPERATURE)
        rob.append({"condition": cond, **detection_metrics(pool.label.values[rb_rows][s.ok], s.p[s.ok])})
        pd.DataFrame(rob).to_csv(f"{OUT}/robustness.csv", index=False)
    rob_df = pd.DataFrame(rob)
    print(rob_df[["condition", "accuracy", "eer", "false_positive_rate", "false_negative_rate"]].to_string(index=False))
    stamp("robustness done")
else:
    stamp("robustness: skipped (time budget)")

# %% [markdown]
# ## 11. Summary

# %%
def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100*v:.1f}%"


summary = {
    "runtime_hours": round(elapsed_h(), 2), "smoke_test": SMOKE_TEST,
    "pool": {"n": int(len(pool)), "failed_extraction": int((~pool.ok).sum()), "splits": pool.groupby("split").size().to_dict(),
             "extra_fakes_included": bool((pool.source == "extra_fakes").any())},
    "temperature": TEMPERATURE, "main_test_ensemble": main_metrics, "main_test_member0": single_metrics,
    "splices": splice_report, "attack_attribution_accuracy": attack_acc,
    "legacy_gateway_test": legacy_test, "legacy_uncertainty_threshold": main_unc_thr,
    "in_the_wild": itw_metrics, "gateway_equal_workload": gw_rows,
    "logo": logo_rows, "logo_gateway_equal_workload": logo_eq, "logo_skipped_for_time": logo_skipped,
    "robustness": rob_df.to_dict(orient="records"),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, default=float)

L = ["DEEPSHIELD ROUND 3 — PLAIN-ENGLISH SUMMARY" + ("  (SMOKE TEST: numbers meaningless)" if SMOKE_TEST else ""),
     f"(runtime {elapsed_h():.1f} h; {N_ENSEMBLE}-model ensemble; temperature {TEMPERATURE:.2f})", "",
     f"Clean test set ({main_metrics['n_real']} genuine, {main_metrics['n_fake']} fake incl. splices; no recording shared with training):",
     f"  ensemble accuracy {pct(main_metrics['accuracy'])}, EER {pct(main_metrics['eer'])} "
     f"(single model: {pct(single_metrics['accuracy'])}, EER {pct(single_metrics['eer'])})",
     f"  genuine flagged fake {pct(main_metrics['false_positive_rate'])}, fakes passed as genuine {pct(main_metrics['false_negative_rate'])}"]
if splice_report:
    L.append(f"  partial deepfakes: {pct(splice_report['detected'])} detected; most suspicious window on the inserted part "
             f"{pct(splice_report['most_suspicious_window_overlaps_splice'])}")
if itw_metrics:
    L += ["", f"In-the-Wild ({itw_metrics['n_real']} genuine, {itw_metrics['n_fake']} fake, never trained on): "
              f"accuracy {pct(itw_metrics['accuracy'])}, EER {pct(itw_metrics['eer'])}, genuine flagged fake "
              f"{pct(itw_metrics['false_positive_rate'])}, fakes passed {pct(itw_metrics['false_negative_rate'])}"]
L += ["", "Equal-workload gateway (cut-offs fixed on validation; fakes auto-passed | genuine auto-blocked | genuine reviewed):"]
for (budget, set_name), g in gateway_eq.groupby(["budget", "set"], sort=False):
    L.append(f"  budget {budget:.0%}, {set_name}:")
    for r in g.itertuples():
        L.append(f"    {r.signals:36s} {pct(r.fakes_auto_passed)} | {pct(r.genuine_auto_blocked)} | {pct(r.genuine_reviewed)}")
if len(logo_eq_df):
    L += ["", "Unseen generators, equal workload (mean over held-out families, 'unseen' set):"]
    u = logo_eq_df[logo_eq_df.set == "unseen"].groupby(["budget", "signals"], sort=False)[
        ["fakes_auto_passed", "genuine_auto_blocked", "genuine_reviewed"]].mean()
    for (budget, sig), r in u.iterrows():
        L.append(f"  budget {budget:.0%}  {sig:36s} {pct(r.fakes_auto_passed)} | {pct(r.genuine_auto_blocked)} | {pct(r.genuine_reviewed)}")
    for r in logo_rows:
        L.append(f"  {r['held_out']:22s} EER seen {pct(r['seen_eer'])} -> unseen {pct(r['unseen_eer'])}; "
                 f"unseen fakes passed {pct(r['unseen_false_negative_rate'])}")
if logo_skipped:
    L.append(f"  (not run, out of time: {', '.join(logo_skipped)})")
L += ["", f"Attack-type attribution on test fakes: {pct(attack_acc)}",
      f"Legacy rule gateway on test (uncertainty threshold {main_unc_thr:.3f}): fakes auto-passed "
      f"{pct(legacy_test['fakes_auto_passed'])}, genuine auto-blocked {pct(legacy_test['genuine_auto_blocked'])}, "
      f"genuine reviewed {pct(legacy_test['genuine_reviewed'])}"]
if len(rob_df):
    L += ["", "Robustness (EER): " + ", ".join(f"{r.condition} {pct(r.eer)}" for r in rob_df.itertuples())]
text = "\n".join(L)
open(f"{OUT}/SUMMARY.txt", "w").write(text)
print(text)
print(f"\nAll outputs in {OUT}  (models/ replaces demo/models/)")
