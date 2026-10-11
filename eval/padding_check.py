"""Checks whether short clips (zero-padded to the 3 s window) trigger false alarms.

Loads the demo model, cuts each genuine demo clip to a few seconds, and scores it with
(a) zero padding (what training and the app do) and (b) the clip repeated to fill the window.
Usage: ..\\.venv\\Scripts\\python padding_check.py
"""
import glob, os, sys
import numpy as np
import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
os.environ.setdefault("PORT", "0")
import app  # noqa: E402  (loads the model + novelty scorer from demo/models)

SR, WIN = app.TARGET_SR, app.TARGET_SR * app.DURATION
clips = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "demo", "clips", "in_the_wild", "*.wav")))


@torch.no_grad()
def score(y):
    x = torch.tensor(app._features_from_waveform(y), dtype=torch.float32).unsqueeze(0).to(app.DEVICE)
    return float(app.model(x, mc_samples=1)[2].item())


def zero_pad(y):
    return np.pad(y, (0, WIN - len(y))) if len(y) < WIN else y[:WIN]


def tile(y):
    return np.tile(y, int(np.ceil(WIN / len(y))))[:WIN]


print(f"{'clip':45s} {'len':>5s} | full-3s | cut 2.0s: zero / tile | cut 2.5s: zero / tile")
for c in clips:
    y, _ = librosa.load(c, sr=SR, mono=True)
    row = [f"{os.path.basename(c):45s} {len(y)/SR:5.1f}", f"{score(zero_pad(y[:WIN])):.2f}"]
    for sec in (2.0, 2.5):
        cut = y[: int(sec * SR)]
        row.append(f"{score(zero_pad(cut)):.2f} / {score(tile(cut)):.2f}")
    print(" | ".join(row))
