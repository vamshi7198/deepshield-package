"""Round-3 building blocks shared by the Kaggle pipeline (deepshield_round3.py) and the local tests.

build_notebook.py --inline-core pastes this file into the notebook as its own cell, so everything here must
work both on Kaggle (GPU) and on a laptop (CPU). No top-level side effects.
"""
import numpy as np
import librosa

SR, DUR = 16000, 3
WIN = SR * DUR
N_FRAMES = 94
TRIM_TOP_DB = 40            # leading/trailing silence quieter than this (dB below peak) is cut
EVAL_HOP_S = 0.5            # overlapping evaluation windows, as in the demo app
TOP_WINDOW_FRACTION = 1 / 3  # clip score = mean of the most suspicious third of its windows


# ─── audio ────────────────────────────────────────────────────────────────────

def load_trimmed(path, max_seconds=12.0, trim=True):
    """Mono 16 kHz audio, at most max_seconds, with leading/trailing silence removed.

    Trimming takes away the silence-length shortcut: in several public fake datasets the generated clips
    start and end with cleaner, shorter silences than the genuine ones, and a model scored on the first
    3 s of each clip can learn that instead of the voice itself."""
    y, _ = librosa.load(path, sr=SR, duration=max_seconds, mono=True)
    if trim and len(y):
        yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        if len(yt) >= SR // 2:      # keep the original if trimming leaves almost nothing
            y = yt
    return y.astype(np.float32)


def pad_to(y, n=WIN):
    return np.pad(y, (0, n - len(y))) if len(y) < n else y


def random_crop(y, rng, must_cover=None, min_overlap=SR):
    """Random 3 s crop. With must_cover=(a, b) (sample range of a spliced-in segment) the crop is chosen so
    that it overlaps that range by at least min_overlap samples (or all of it, if shorter), so a spliced
    clip labelled fake is never trained on a crop that contains only its genuine part."""
    y = pad_to(y)
    hi = len(y) - WIN
    if hi <= 0:
        return y[:WIN]
    if must_cover is None:
        s = int(rng.integers(0, hi + 1))
    else:
        a, b = must_cover
        need = min(min_overlap, b - a)
        lo_s, hi_s = max(0, a + need - WIN), min(hi, b - need)
        s = int(rng.integers(lo_s, hi_s + 1)) if hi_s >= lo_s else int(np.clip(a, 0, hi))
    return y[s:s + WIN]


def window_starts(n_samples, hop_s=EVAL_HOP_S, max_windows=24):
    """Overlapping window offsets, identical to the demo app's _window_offsets (last window ends at the clip end)."""
    max_start = n_samples - WIN
    if max_start <= 0:
        return [0]
    hop = int(hop_s * SR)
    starts = list(range(0, max_start + 1, hop))
    if starts[-1] != max_start:
        starts.append(max_start)
    if len(starts) > max_windows:
        starts = [starts[int(round(i * (len(starts) - 1) / (max_windows - 1)))] for i in range(max_windows)]
    return starts


def splice(genuine, fake, rng, seg_s=(1.0, 2.0), fade_ms=10):
    """Insert a 1-2 s stretch of synthetic speech into a genuine recording (a 'partial deepfake').

    The fake stretch is loudness-matched to the genuine audio around it and joined with short crossfades.
    Returns the spliced audio and the (start, end) sample range of the inserted segment."""
    genuine, fake = pad_to(genuine, WIN), pad_to(fake, SR * 2)
    seg = int(rng.uniform(*seg_s) * SR)
    seg = min(seg, len(fake), len(genuine) - SR // 2)
    fs = int(rng.integers(0, len(fake) - seg + 1))
    piece = fake[fs:fs + seg].copy()
    gs = int(rng.integers(SR // 4, max(SR // 4 + 1, len(genuine) - seg - SR // 4)))
    ctx = genuine[max(0, gs - SR // 2): gs + seg + SR // 2]
    piece *= (np.sqrt(np.mean(ctx ** 2)) + 1e-6) / (np.sqrt(np.mean(piece ** 2)) + 1e-6)
    f = max(1, int(fade_ms * SR / 1000))
    out = genuine.copy()
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    piece[:f] = piece[:f] * ramp + out[gs:gs + f] * (1 - ramp)
    piece[-f:] = piece[-f:] * ramp[::-1] + out[gs + seg - f:gs + seg] * (1 - ramp[::-1])
    out[gs:gs + seg] = piece
    return out.astype(np.float32), (gs, gs + seg)


def features(y):
    """Dual-channel (log-Mel, MFCC) grid, 2 x 64 x 94, identical to the demo app's _features_from_waveform."""
    y = pad_to(y)[:WIN]
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, n_fft=1024, hop_length=512), ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=64, n_fft=1024, hop_length=512)
    feat = np.stack([mel_db, mfcc], axis=0).astype(np.float32)
    if feat.shape[2] != N_FRAMES:
        feat = feat[:, :, :N_FRAMES] if feat.shape[2] > N_FRAMES else np.pad(feat, ((0, 0), (0, 0), (0, N_FRAMES - feat.shape[2])))
    return feat


# ─── clip scores from window scores ───────────────────────────────────────────

def aggregate(window_probs, frac=TOP_WINDOW_FRACTION):
    p = np.asarray(window_probs, dtype=np.float64)
    k = max(1, int(len(p) * frac))
    return float(np.sort(p)[-k:].mean())


def clip_scores(window_probs, clip_of_window, n_clips, frac=TOP_WINDOW_FRACTION):
    """Aggregate a flat array of window probabilities into one score per clip."""
    order = np.argsort(clip_of_window, kind="stable")
    bounds = np.searchsorted(clip_of_window[order], np.arange(n_clips + 1))
    return np.array([aggregate(window_probs[order[bounds[i]:bounds[i + 1]]], frac) if bounds[i + 1] > bounds[i] else np.nan
                     for i in range(n_clips)])


def best_window(window_probs, clip_of_window, n_clips):
    """Index (into the flat window array) of each clip's most suspicious window."""
    best = np.full(n_clips, -1)
    top = np.full(n_clips, -np.inf)
    for i, (c, p) in enumerate(zip(clip_of_window, window_probs)):
        if p > top[c]:
            top[c], best[c] = p, i
    return best


# ─── calibration and routing ──────────────────────────────────────────────────

def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(window_logits, clip_of_window, y_clip, grid=np.exp(np.linspace(np.log(0.25), np.log(20), 120))):
    """Temperature T for window logits that minimises clip-level log-loss on validation data.
    The detector's raw probabilities sit almost always at 0.00 or 1.00; T > 1 spreads them out."""
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = clip_scores(1 / (1 + np.exp(-window_logits / t)), clip_of_window, len(y_clip))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(y_clip * np.log(p) + (1 - y_clip) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def review_threshold(risk, y, budget):
    """Risk cut-off that sends `budget` of the genuine clips to review (chosen on validation data)."""
    return float(np.quantile(risk[y == 0], 1 - budget))


def risk_features(p, ens_std=None, mc_unc=None, novelty=None):
    """Per-clip routing signals, each oriented so that larger means more likely to be wrong.
    margin = -|logit p|: clips scored near 0.5 come first (the score-only baseline)."""
    f = {"margin": -np.abs(logit(p))}
    if ens_std is not None:
        f["ens_std"] = np.asarray(ens_std, dtype=np.float64)
    if mc_unc is not None:
        f["mc_unc"] = np.asarray(mc_unc, dtype=np.float64)
    if novelty is not None:
        f["novelty"] = np.asarray(novelty, dtype=np.float64)
    return f


def route(p, risk, thr):
    """pass / block / review: review above the risk cut-off, otherwise the 0.5 verdict decides."""
    return np.where(risk > thr, "review", np.where(p > 0.5, "block", "pass"))


def route_rates(y, routes):
    real, fake = y == 0, y == 1
    rate = lambda m, r: float((routes[m] == r).mean()) if m.any() else float("nan")
    return {"fakes_auto_passed": rate(fake, "pass"), "fakes_reviewed": rate(fake, "review"),
            "genuine_auto_blocked": rate(real, "block"), "genuine_reviewed": rate(real, "review")}


class CombinedRisk:
    """Combined risk score (dependent claim 6): a logistic model, fitted on validation clips, that predicts
    whether the 0.5 verdict is wrong from the score margin and the other heads' signals. Reviewing the
    highest-risk clips under a fixed genuine-review budget is then compared, at equal workload, with the
    baseline of reviewing the clips whose score is closest to 0.5."""
    def __init__(self, names):
        self.names = list(names)

    def _x(self, feats):
        return np.column_stack([feats[n] for n in self.names])

    def fit(self, feats, y, p):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        wrong = ((p > 0.5).astype(int) != y).astype(int)
        self.scaler = StandardScaler().fit(self._x(feats))
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit(self.scaler.transform(self._x(feats)), wrong)
        return self

    def risk(self, feats):
        return self.lr.decision_function(self.scaler.transform(self._x(feats)))

    def to_json(self):
        return {"features": self.names, "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "coef": self.lr.coef_[0].tolist(), "intercept": float(self.lr.intercept_[0])}


def equal_workload_comparison(sets, budget, val_feats, val_y, val_p, signal_sets):
    """For each candidate routing signal, fix its review cut-off on validation so `budget` of genuine
    validation clips go to review, then apply it unchanged to every evaluation set.

    sets:        {name: (feats, y, p)} evaluation sets
    signal_sets: {label: list of feature names}; ["margin"] alone is the score-only baseline
    Returns a list of rows (dicts) with route rates per set and signal."""
    rows = []
    for label, names in signal_sets.items():
        if names == ["margin"]:
            risk_of = lambda f: f["margin"]
        else:
            cr = CombinedRisk(names).fit(val_feats, val_y, val_p)
            risk_of = cr.risk
        thr = review_threshold(risk_of(val_feats), val_y, budget)
        for set_name, (f, y, p) in sets.items():
            rows.append({"signals": label, "set": set_name, "budget": budget, **route_rates(y, route(p, risk_of(f), thr))})
    return rows
