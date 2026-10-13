"""Fit the two inference-only add-ons in demo/addons.py from labelled clips, report how much they help on speakers
they were not fitted on, and optionally enroll speakers for the claimed-speaker check. No network retraining.

    python tools/fit_addons.py --labels itw/labels.csv --audio itw --models demo/models \
        --exclude-speakers "Barack Obama,Mark Zuckerberg,Arnold Schwarzenegger,Queen Elizabeth II" \
        --enroll-speakers "Barack Obama,Mark Zuckerberg,Arnold Schwarzenegger,Queen Elizabeth II" \
        --skip-files demo/clips/in_the_wild

labels.csv: columns file, speaker, label (bona-fide/spoof or real/fake), as in the In-the-Wild meta.csv.
Writes <models>/reference_bank.json and <models>/speaker_fusion.json; enrolls into demo/enrollment/.
Speakers in --exclude-speakers are left out of fitting; their clips (minus the enrolled ones and anything in
--skip-files) are scored at the end as a fully held-out check. Embeddings come from demo/app.py's own
score_audio(), so the add-ons see exactly what the app computes.
"""
import argparse, csv, datetime, importlib.util, json, os, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--models", default="demo/models")
ap.add_argument("--exclude-speakers", default="")
ap.add_argument("--enroll-speakers", default="")
ap.add_argument("--enroll-k", type=int, default=5)
ap.add_argument("--fusion-k", type=int, default=5, help="enrolled clips per speaker when fitting the speaker fusion")
ap.add_argument("--skip-files", default="", help="folder of clips (e.g. the demo clips) never used for anything")
ap.add_argument("--cache", default="", help=".npz cache of clip embeddings / scores")
a = ap.parse_args()

REPO = Path(__file__).resolve().parents[1]
os.environ["DEEPSHIELD_MODEL_DIR"] = str(Path(a.models).resolve())
os.environ["DEEPSHIELD_REFERENCE_BANK"] = "0"
spec = importlib.util.spec_from_file_location("dsapp", REPO / "demo" / "app.py")
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
addons = app.addons

split_list = lambda s: [x.strip() for x in s.split(",") if x.strip()]
excluded, to_enroll = set(split_list(a.exclude_speakers)), split_list(a.enroll_speakers)
skip_ids = {p.name.rsplit("_", 1)[-1] for p in Path(a.skip_files).glob("*.wav")} if a.skip_files else set()
rows = [r for r in csv.DictReader(open(a.labels, encoding="utf-8")) if r["file"] not in skip_ids]
for r in rows:
    r["y"] = int(r["label"] in ("spoof", "fake"))

# ── embeddings and detector scores, via the app ──────────────────────────────
cache = {}
if a.cache and os.path.exists(a.cache):
    c = np.load(a.cache, allow_pickle=False)
    cache = {f: (e, p) for f, e, p in zip(c["files"], c["emb"], c["p"])}
for i, r in enumerate(rows):
    if r["file"] not in cache:
        sc = app.score_audio((Path(a.audio) / r["file"]).read_bytes(), ".wav")
        cache[r["file"]] = (sc["clip_embedding"], sc["detector_probability"])
    r["emb"], r["p"] = cache[r["file"]]
    if (i + 1) % 200 == 0:
        print(f"  scored {i + 1}/{len(rows)}", flush=True)
if a.cache:
    np.savez(a.cache, files=np.array(list(cache)), emb=np.stack([v[0] for v in cache.values()]),
             p=np.array([v[1] for v in cache.values()]))

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    fpr, tpr, _ = roc_curve(y, s); i = np.nanargmin(np.abs(1 - tpr - fpr))
    pred = s > 0.5
    return {"accuracy": float((pred == y).mean()), "eer": float((fpr[i] + 1 - tpr[i]) / 2),
            "genuine_flagged_fake": float(pred[y == 0].mean()), "fakes_passed": float((~pred[y == 1]).mean()), "n": int(len(y))}


def fmt(m):
    return f"acc {m['accuracy']:.3f}  EER {m['eer']:.3f}  genuine->fake {m['genuine_flagged_fake']:.2f}  fake->genuine {m['fakes_passed']:.2f}  (n={m['n']})"


def fit_bank(rs):
    X = np.stack([r["emb"] for r in rs]); sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(X), [r["y"] for r in rs])


def enrollment_rows(rs, k, seed, emb_scale):
    """Simulate enrollment: per speaker, k genuine clips are enrolled; every other clip of that speaker is scored."""
    rng = np.random.default_rng(seed)
    out = []
    by = {}
    for r in rs:
        by.setdefault(r["speaker"], []).append(r)
    for s, items in by.items():
        gen = [r for r in items if r["y"] == 0]
        if len(gen) <= k:
            continue
        idx = rng.permutation(len(gen))
        enr = [gen[i] for i in idx[:k]]
        E = np.stack([r["emb"] for r in enr])
        for r in items:
            if any(r is e for e in enr):
                continue
            l2 = float(np.min(np.linalg.norm((E - r["emb"]) / emb_scale, axis=1)))
            cos = float(1 - np.max(E @ r["emb"] / (np.linalg.norm(E, axis=1) * np.linalg.norm(r["emb"]) + 1e-9)))
            out.append({"y": r["y"], "p": r["p"], "x": [float(logit(r["p"])), l2, cos]})
    return out


def fit_fusion(er):
    X = np.array([r["x"] for r in er]); sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=3000).fit(sc.transform(X), [r["y"] for r in er])


fit_rows = [r for r in rows if r["speaker"] not in excluded]
speakers = sorted({r["speaker"] for r in fit_rows})
rng = np.random.default_rng(1); rng.shuffle(speakers)
folds = [set(speakers[: len(speakers) // 2]), set(speakers[len(speakers) // 2:])]
print(f"\nFitting data: {len(fit_rows)} clips, {len(speakers)} speakers (excluded: {sorted(excluded) or 'none'})")

# ── honest estimate: 2-fold cross-validation by speaker ──────────────────────
cv = {"detector": [], "reference_bank": [], "speaker_detector": [], "speaker_check": []}
for k in (0, 1):
    tr = [r for r in fit_rows if r["speaker"] in folds[k]]
    te = [r for r in fit_rows if r["speaker"] in folds[1 - k]]
    sc, lr = fit_bank(tr)
    cv["detector"] += [(r["y"], r["p"]) for r in te]
    cv["reference_bank"] += list(zip([r["y"] for r in te], lr.predict_proba(sc.transform(np.stack([r["emb"] for r in te])))[:, 1]))
    es = np.stack([r["emb"] for r in tr if r["y"] == 0]).std(0) + 1e-6
    fsc, flr = fit_fusion(sum((enrollment_rows(tr, a.fusion_k, s, es) for s in range(3)), []))
    er = enrollment_rows(te, a.fusion_k, 99, es)
    cv["speaker_detector"] += [(r["y"], r["p"]) for r in er]
    cv["speaker_check"] += list(zip([r["y"] for r in er], flr.predict_proba(fsc.transform([r["x"] for r in er]))[:, 1]))
cv_m = {k: metrics(*zip(*v)) for k, v in cv.items()}
print("\nCross-validated by speaker (fitted on one half of the speakers, scored on the other):")
print(f"  detector alone                  {fmt(cv_m['detector'])}")
print(f"  + reference bank                {fmt(cv_m['reference_bank'])}")
print(f"  detector, same clips as below   {fmt(cv_m['speaker_detector'])}")
print(f"  + speaker check ({a.fusion_k} enrolled)     {fmt(cv_m['speaker_check'])}")

# ── final fit on all fitting speakers ────────────────────────────────────────
models = Path(a.models)
today = datetime.date.today().isoformat()
sc, lr = fit_bank(fit_rows)
json.dump({**addons._Linear.export(sc, lr), "meta": {
    "fitted": today, "labels": os.path.basename(a.labels), "n_clips": len(fit_rows), "n_speakers": len(speakers),
    "excluded_speakers": sorted(excluded), "cross_validated": {"detector": cv_m["detector"], "reference_bank": cv_m["reference_bank"]}}},
    open(models / "reference_bank.json", "w"), indent=1)
emb_scale = np.stack([r["emb"] for r in fit_rows if r["y"] == 0]).std(0) + 1e-6
fsc, flr = fit_fusion(sum((enrollment_rows(fit_rows, a.fusion_k, s, emb_scale) for s in range(3)), []))
json.dump({**addons._Linear.export(fsc, flr), "features": addons.SpeakerCheck.FEATURES,
           "embedding_scale": emb_scale.tolist(), "min_enrolled": 3, "meta": {
               "fitted": today, "labels": os.path.basename(a.labels), "n_speakers": len(speakers), "fusion_k": a.fusion_k,
               "excluded_speakers": sorted(excluded),
               "cross_validated": {"detector": cv_m["speaker_detector"], "speaker_check": cv_m["speaker_check"]}}},
          open(models / "speaker_fusion.json", "w"), indent=1)
print(f"\nwrote {models / 'reference_bank.json'} and {models / 'speaker_fusion.json'}")

# ── enroll speakers, then score their remaining clips (never used in any fit) ─
if to_enroll:
    check = addons.SpeakerCheck(str(models / "speaker_fusion.json"), app.ENROLL_DIR)
    bank = addons.ReferenceBank(str(models / "reference_bank.json"))
    held = {"detector": [], "reference_bank": [], "speaker_check": []}
    rng = np.random.default_rng(7)
    for s in to_enroll:
        gen = [r for r in rows if r["speaker"] == s and r["y"] == 0]
        rng.shuffle(gen)
        enr = gen[: a.enroll_k]
        check.remove(s)
        n = check.add(s, [r["emb"] for r in enr], [r["file"] for r in enr])
        rest = [r for r in rows if r["speaker"] == s and not any(r is e for e in enr)]
        for r in rest:
            res = check.check(s, r["emb"], r["p"])
            held["detector"].append((r["y"], r["p"]))
            held["reference_bank"].append((r["y"], bank.prob(r["emb"])))
            held["speaker_check"].append((r["y"], res["fused_probability"] if res["applied"] else r["p"]))
        print(f"  enrolled '{check.slug(s)}': {n} genuine clips; {len(rest)} other clips of this speaker held out")
    if held["detector"] and s in excluded:
        print("\nHeld-out speakers (never in any fit; scored on their non-enrolled clips):")
        for k, v in held.items():
            print(f"  {k:16s} {fmt(metrics(*zip(*v)))}")
