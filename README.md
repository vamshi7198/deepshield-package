# DeepShield: verification, demo and patent package

Everything produced after the students' repository was reviewed, verified and retrained (October 2026).
The students' original code and GitHub repository were **not modified**; all work here is additional.

## What's in this package

| Item | Where | What it is |
|---|---|---|
| Management briefing | `deliverables/DeepShield_Management_Briefing.pptx` | 19 slides for senior management, with speaker notes on every slide |
| Patent pre-disclosure | `deliverables/DeepShield_Patent_PreDOI_v1.0.pptx` | The inventors' IP Pre-DOI template, corrected and extended (27 slides) |
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
.venv\Scripts\pip install -r demoequirements.txt
```
`start_demo.bat` expects that `.venv` folder next to `demo\`. The built web interface is already in `demo/web-dist/`
(built from the students' `web/` folder, unchanged, pointed at `http://localhost:8765`).

## Running the demo

1. Double-click `demo/start_demo.bat`. Two black windows open, then the browser.
2. Click **Detector**, then upload any clip from `demo/clips/in_the_wild/`, or record from the microphone.
3. The first analysis takes up to 40 seconds while the model warms up; after that, under a second.
4. To stop, close the two black windows.

Suggested demo sequence (also in the speaker notes of slide 15):
`Mark_Zuckerberg_GENUINE_8168.wav` → Pass, `Barack_Obama_DEEPFAKE_23208.wav` → Block,
`Mark_Zuckerberg_DEEPFAKE_11316.wav` → Human review.

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

## Rebuilding

- Notebooks: `python eval/build_notebook.py --full --src deepshield_round2.py --kernel kaggle_kernel_round2`,
  then `kaggle kernels push -p eval/kaggle_kernel_round2` (GPU, about 1.8 h).
- Decks: `tools/build_management_deck.js` (Node, pptxgenjs) and `tools/build_patent_deck.py` (python-pptx).
- Demo screenshots: `tools/capture_demo.py` (needs the demo running).

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
