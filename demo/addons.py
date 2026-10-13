"""Inference-only add-ons on top of the frozen DeepShield model. Neither retrains the network; both work on the
128-d clip embedding (mean of the window embeddings of ensemble member 0) that the model already computes.

ReferenceBank   a logistic layer fitted on labelled clips from the deployment's own audio (domain adaptation).
SpeakerCheck    compares a clip with verified genuine recordings of the speaker it claims to be, and fuses that
                distance with the detector's score.

Both are fitted by tools/fit_addons.py and stored as small JSON files next to the model.
"""
import json, os, re, threading
import numpy as np


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


class _Linear:
    """Standardise, then logistic: the stored form of a fitted sklearn StandardScaler + LogisticRegression."""
    def __init__(self, d):
        self.mean, self.scale = np.asarray(d["mean"]), np.asarray(d["scale"])
        self.coef, self.intercept = np.asarray(d["coef"]), float(d["intercept"])

    def prob(self, x):
        return _sigmoid(float(np.dot(self.coef, (np.asarray(x) - self.mean) / self.scale) + self.intercept))

    @staticmethod
    def export(scaler, lr):
        return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


class ReferenceBank:
    """P(fake) from the clip embedding, fitted on labelled reference clips. Replaces the detector's score."""
    def __init__(self, path):
        d = json.load(open(path))
        self.meta = d.get("meta", {})
        self.lin = _Linear(d)

    def prob(self, clip_embedding):
        return self.lin.prob(clip_embedding)


class SpeakerCheck:
    """Claimed-speaker check. Signals: the detector's logit, and the clip's distance to the closest enrolled genuine
    recording of the claimed speaker (scaled L2 and cosine). Fused by a logistic model fitted on labelled clips."""
    FEATURES = ["detector_logit", "l2_distance", "cosine_distance"]

    def __init__(self, fusion_path, enroll_dir):
        d = json.load(open(fusion_path))
        self.meta = d.get("meta", {})
        self.lin = _Linear(d)
        self.emb_scale = np.asarray(d["embedding_scale"])
        self.min_enrolled = int(d.get("min_enrolled", 3))
        self.dir = enroll_dir
        os.makedirs(enroll_dir, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def slug(speaker):
        s = re.sub(r"[^a-z0-9]+", "_", str(speaker).strip().lower()).strip("_")
        if not s or len(s) > 64:
            raise ValueError("speaker id must be 1-64 letters, digits or separators")
        return s

    def _path(self, speaker):
        return os.path.join(self.dir, self.slug(speaker) + ".npz")

    def enrolled(self, speaker):
        p = self._path(speaker)
        if not os.path.exists(p):
            return np.zeros((0, len(self.emb_scale))), []
        d = np.load(p, allow_pickle=False)
        return d["embeddings"], list(d["sources"])

    def add(self, speaker, embeddings, sources):
        with self._lock:
            E, S = self.enrolled(speaker)
            E = np.vstack([E, np.asarray(embeddings)]) if len(E) else np.asarray(embeddings)
            np.savez(self._path(speaker), embeddings=E, sources=np.array(S + list(sources)))
            return len(E)

    def remove(self, speaker):
        with self._lock:
            p = self._path(speaker)
            if os.path.exists(p):
                os.remove(p)
                return True
            return False

    def speakers(self):
        out = {}
        for f in sorted(os.listdir(self.dir)):
            if f.endswith(".npz"):
                out[f[:-4]] = int(np.load(os.path.join(self.dir, f))["embeddings"].shape[0])
        return out

    def distances(self, E, e):
        l2 = float(np.min(np.linalg.norm((E - e) / self.emb_scale, axis=1)))
        cos = float(1 - np.max(E @ e / (np.linalg.norm(E, axis=1) * np.linalg.norm(e) + 1e-9)))
        return l2, cos

    def check(self, speaker, clip_embedding, detector_prob):
        E, _ = self.enrolled(speaker)
        base = {"speaker": self.slug(speaker), "enrolled_clips": int(len(E))}
        if len(E) < self.min_enrolled:
            return {**base, "applied": False,
                    "reason": f"needs at least {self.min_enrolled} enrolled genuine clips (has {len(E)})"}
        l2, cos = self.distances(E, np.asarray(clip_embedding))
        fused = self.lin.prob([_logit(detector_prob), l2, cos])
        return {**base, "applied": True, "l2_distance": round(l2, 4), "cosine_distance": round(cos, 5),
                "detector_probability": round(float(detector_prob), 6), "fused_probability": round(fused, 6)}
