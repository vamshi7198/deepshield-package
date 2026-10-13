# DeepShield: verification, demo and patent package

Everything produced after the students' repository was reviewed, verified and retrained (October 2026).
The students' original code and GitHub repository were **not modified**; all work here is additional.

## What's in this package

| Item | Where | What it is |
|---|---|---|
| Management briefing | `deliverables/DeepShield_Management_Briefing_v2.pptx` | 22 slides for senior management, speaker notes on every slide; v2 (Oct 2026) adds the add-ons, robustness changes and the new demo sequence |
| Patent pre-disclosure | `deliverables/DeepShield_Patent_PreDOI_v2.0.pptx` | The inventors' IP Pre-DOI template, corrected and extended (30 slides); v2.0 adds claims 14–15, the add-on module and evidence, SASV prior art, corrected claim statuses |
| Live demo | `demo/` → after setup, double-click `demo/start_demo.bat` | Runs fully on the laptop at http://localhost:8080 (students' UI, unchanged) |
| Demo clips | `demo/clips/in_the_wild/` | 24 clips: 4 public figures × 3 genuine + 3 deepfake, never used in training |
| Verification notebook | `eval/deepshield_verification.ipynb` | Round 1: corrected evaluation of the students' design |
| Round-2 notebook | `eval/deepshield_round2.ipynb` | Final model: more data, In-the-Wild external test, calibrated gateway |
| Results | `eval/runs/round2/deepshield_round2/` | `SUMMARY.txt`, `summary.json`, per-clip predictions, plots |

## Headline results (round-2 model, frozen)

- **93.2%** accuracy on a strict test of 6,648 clips (no recording, sentence or speaker shared with training).
  The earlier 97.7% came from a random split where 28% of test clips had a copy in training.
- **91%** correct on the type of AI tool. **4 in 5** of the newest commercial AI voices caught without ever training on them.
- **73%** on real-world internet clips (In-the-Wild, never trained on): the hardest test.
- A human-review lane at about 15% of genuine audio **roughly halves** both kinds of mistake.
  Most of that gain comes from the review lane itself; the uncertainty and novelty heads add a small
  but consistent extra benefit on never-seen AI tools. This is stated plainly in the patent deck.

## First-time setup (any Windows machine)

```
python -m venv .venv
.venv\Scripts\pip install -r demo\requirements.txt
```
`start_demo.bat` expects that `.venv` folder next to `demo\`. The built web interface is already in `demo/web-dist/`
(built from the students' `web/` folder, unchanged, pointed at `http://localhost:8765`).

## Running the demo

1. Double-click `demo/start_demo.bat`. Two black windows open, then the browser.
2. Click **Detector**, then upload any clip from `demo/clips/in_the_wild/`, or record from the microphone.
3. The first analysis takes up to 40 seconds while the model warms up; after that, under a second.
4. To stop, close the two black windows.

Suggested demo sequence (routes are deterministic since the robustness update, so this always plays out the same;
reference bank on, see below):
`Mark_Zuckerberg_GENUINE_31559.wav` → Pass, `Barack_Obama_DEEPFAKE_23208.wav` → Block,
`Mark_Zuckerberg_DEEPFAKE_11316.wav` → Quality-control review.
The speaker notes of slide 15 still list the earlier sequence, which no longer holds.

## What changed relative to the students' code (all in our copy only)

- **Evaluation:** corrected generator labels, grouped train/val/test split, MC-dropout fix, leave-one-family-out
  run for 9 families, external In-the-Wild test, robustness with error rates.
- **Data (round 2):** added MLAAD, 2025–26 commercial voices, VCTK, AccentDB, Common Voice, LJ Speech.
- **Demo backend (`demo/app.py`, a copy of the students' newer `app.py`):** loads the round-2 model, reads the
  calibrated uncertainty threshold from `models/gateway_calibration.json`, routes clips under 2 s to review,
  and reports the honest 93.2% accuracy. The architecture is unchanged.
- **UI:** unchanged. Note that its landing page still says "ResNet-34"; the actual network is a compact
  ResNet-style CNN (3 residual blocks, 341,254 parameters).

Previous models are kept for comparison: `demo/models_round1/` and `demo/models_students_original/`.

## Robustness update (October 2026, branch `robustness-v3`)

Measured on 800 In-the-Wild clips downloaded separately (54 speakers, split by speaker into dev and test halves;
none of the 24 demo clips), plus 13 laptop-made samples and 14 accent TTS clips. Scripts and raw results are not
in the repo; numbers below are what they produced.

**Demo backend (`demo/app.py`), same round-2 model:**
- **Overlapping windows:** a 3 s window every 0.5 s (was 2–12 back-to-back windows); clip score = mean of the most
  suspicious third. The model's score can swing from 0.00 to 0.99 when its window moves by half a second, so sparse
  windows could miss a fake entirely (a 6 s Indian-English TTS clip scored 0.08; now 0.997). Overall accuracy is
  about the same (79.8% vs 79.0%; test speakers 77.7% vs 75.2%, dev speakers 81.9% vs 82.9%); genuine clips
  auto-blocked fell from 6.3% to 3.4% on the test speakers. Each response now lists every window's score.
- **Deterministic routes:** MC-dropout uses a fixed seed and 20 passes (as in the evaluation), so the same upload
  always gets the same route. Before, borderline clips switched between Block and Review on repeat uploads.
- **Input handling:** undecodable files return 400 (was 500); silent uploads go to review; uploads over 25 MB are
  refused (`MAX_UPLOAD_MB`).
- **Exposure:** listens on 127.0.0.1 unless `HOST` is set; browser access only from the local UI unless
  `CORS_ORIGINS` is set.
- **Round-3 ready:** if `models/gateway_calibration.json` lists ensemble members, a temperature, silence trimming
  and a combined risk score, the app uses them; with the round-2 files it behaves as described above.
- **Tried and dropped:** re-scoring each clip under faint noise, gain change and telephone filtering, and sending
  unstable verdicts to review. It did not beat the plain score margin at equal review workload, so it is not used.

**Known limits of the current model, confirmed:** a partial deepfake (1.8 s of TTS spliced into genuine speech,
`SYNTHETIC_splice...` in the laptop samples) passes as genuine at 0.000; most of its In-the-Wild mistakes are
confident ones, which routing cannot catch. Both need retraining.

## Add-ons without retraining (`demo/addons.py`, fitted by `tools/fit_addons.py`)

Both sit on top of the frozen model and use the 128-d clip embedding it already computes (mean over windows).
Fitted on 741 In-the-Wild clips from 50 speakers; the four demo speakers (Obama, Zuckerberg, Schwarzenegger,
Queen Elizabeth II) and the 24 demo clips were kept out of all fitting.

| | Detector alone | With add-on |
|---|---|---|
| **Reference bank**, cross-validated by speaker (741 clips) | 80.3% acc, EER 19.7% | **91.8%**, EER **8.6%** |
| **Speaker check**, 5 enrolled genuine clips, cross-validated by speaker (479 clips) | 80.6%, EER 17.2% | **89.1%**, EER **14.8%** |
| Demo speakers, never in any fit (39 non-demo clips): reference bank / speaker check | 64.1%, EER 30.5% | **84.6% / 84.6%**, EER 10.2% / 11.9% |
| The 24 demo clips: reference bank / speaker check | 13/24 | **21/24 / 15/24** |
| 14 laptop samples incl. the partial deepfake, reference bank | 11/14 | **14/14** |

- **Reference bank** (`models/reference_bank.json`, on whenever the file is present; `DEEPSHIELD_REFERENCE_BANK=0`
  turns it off): a logistic layer on the embedding, fitted on labelled clips of the kind of audio the system will see.
  It replaces the detector's score; the detector's own score is still returned as `detector_probability`.
  Caveat: fitted and tested on one source (In-the-Wild); refit it on the deployment's own labelled audio.
- **Speaker check** (`models/speaker_fusion.json` + `demo/enrollment/`): enroll a speaker's verified genuine
  recordings with `POST /api/enroll` (fields `speaker`, `audio` files; at least 3), then send `claimed_speaker` with
  `/api/predict`. The clip's distance to the closest enrolled recording is fused with the detector's score. The four
  demo speakers are enrolled with 5 non-demo clips each. `GET /api/enroll` lists speakers, `DELETE /api/enroll/<id>` removes one.
  Caveats: In-the-Wild clips are cut from longer recordings, so enrolled and test clips may share a session, which
  flatters the result; confirm with enrollment audio from separate recordings. It lowers fakes passed (21% → 6%) but
  raises genuine flagged (15% → 24%) at the 0.5 threshold. Close prior art: spoofing-aware speaker verification
  (SASV 2022) fuses a speaker-verification score with a countermeasure score; the difference here is that the
  detector's own embedding is used, with no separate speaker model.
- Refit: `python tools/fit_addons.py --labels <labels.csv> --audio <dir> --models demo/models --exclude-speakers ...
  --enroll-speakers ... --skip-files demo/clips/in_the_wild` (full usage in its docstring).

## Round 3 (retraining, ready to run on Kaggle)

`eval/deepshield_round3.py` (+ `eval/deepshield_r3_core.py`) keeps the round-2 data, grouped split and architecture
(weights stay drop-in compatible) and changes the training and evaluation:

| Change | Why |
|---|---|
| Several random 3 s crops per clip, silence trimmed (was: first 3 s only) | The model only ever saw clip openings |
| Val/test scored with the app's overlapping windows | Evaluation matches deployment |
| 3,000 partial deepfakes (1–2 s of fake speech spliced into genuine recordings) | The splice/edit class had no data |
| Optional extra fakes, e.g. Indian-accent TTS (`tools/make_extra_fakes.py`) | ~1,000 genuine Indian-accent clips, zero fakes |
| 3-model ensemble, temperature scaling | Stabler uncertainty; probabilities no longer stuck at 0/1 |
| Equal-workload gateway comparison on test, In-the-Wild and every unseen-generator fold | The evidence the patent's main claim needs |

Run it: `python eval/build_notebook.py --smoke --inline-core --src deepshield_round3.py --kernel kaggle_kernel_round3`,
`kaggle kernels push -p eval/kaggle_kernel_round3`; check the smoke run, then rebuild with `--full` and push again.
Copy the run's `models/` folder over `demo/models/`. To check the pipeline on a laptop first:
`tools/make_local_mirror.py` builds a small fake copy of the Kaggle inputs (instructions in its docstring).

## Rebuilding

- Notebooks: `python eval/build_notebook.py --full --src deepshield_round2.py --kernel kaggle_kernel_round2`,
  then `kaggle kernels push -p eval/kaggle_kernel_round2` (GPU, about 1.8 h).
- Decks: `tools/build_management_deck.js` (Node, pptxgenjs; run from `tools/` after `npm install`, with `NODE_PATH=tools/node_modules` and `PPTX_SKILL_DIR` pointing at the pptx skill that provides `apply_theme.js`) and `tools/build_patent_deck.py` (python-pptx). Both read `deliverables/_work/robustness_addons_results.json` for the October 2026 numbers. Demo screenshots: `tools/capture_demo.py` (needs the demo running).

## Repository layout

```
deliverables/   the two decks, demo screenshots and figures, analysis JSON used by the patent deck
demo/           laptop demo: backend (app.py), round-2 model + earlier models, built UI, 24 demo clips
eval/           Kaggle notebooks (round 1 and round 2), build tool, results of both runs
tools/          deck builders, screenshot capture, PowerPoint rendering for visual checks
DeepSheild_Templete_IP_PreDOI ... .pptx   the inventors' original patent template (input to the patent deck)
```

This repository is private. It contains inventor contact details (in the patent template) and should not be made public;
public disclosure before filing can also affect patentability.
