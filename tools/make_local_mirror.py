"""Build a small local folder that mimics the Kaggle input layout of every dataset the round-3 pipeline reads,
filled with copies of labelled clips you already have, so `eval/deepshield_round3.py` can be smoke-tested on a
laptop (CPU) before spending GPU time:

    python tools/make_local_mirror.py --labels itw/labels.csv --audio itw --out mirror
    set DEEPSHIELD_INPUT=mirror & set DEEPSHIELD_OUT=smoke_out & set DEEPSHIELD_ALLOW_CPU=1
    python eval/deepshield_round3.py          (from inside eval/, so deepshield_r3_core imports)

labels.csv needs columns file,label with label bona-fide/spoof (the In-the-Wild meta.csv format) or real/fake.
The numbers such a run prints are meaningless; it only proves the pipeline runs end to end.
"""
import argparse, csv, random, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-source", type=int, default=30)
ap.add_argument("--logo-family-size", type=int, default=120, help="fakeaudio melgan clips, enough for one LOGO fold")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.labels, encoding="utf-8")))
real = [r["file"] for r in rows if r["label"] in ("bona-fide", "real")]
fake = [r["file"] for r in rows if r["label"] in ("spoof", "fake")]
rng = random.Random(0)
rng.shuffle(real); rng.shuffle(fake)
out, src = Path(a.out), Path(a.audio)
n = a.per_source


def take(pool, k):
    got = [pool[i % len(pool)] for i in range(take.pos.get(id(pool), 0), take.pos.get(id(pool), 0) + k)]
    take.pos[id(pool)] = take.pos.get(id(pool), 0) + k
    return got
take.pos = {}


def put(files, make_path):
    for i, f in enumerate(files):
        p = out / make_path(i)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, p)


put(take(real, n), lambda i: f"audio-deepfake-detection-dataset/real_samples/{100 + i % 6}_{i}_{i:04d}.wav")
put(take(fake, n), lambda i: f"audio-deepfake-detection-dataset/{['OpenAI', 'xTTS'][i % 2]}/clip_{i}.wav")
put(take(real, n // 2), lambda i: f"real-vs-fake-human-voice-deepfake-audio/spk{i % 4}/real_{i}.wav")
put(take(fake, n // 2), lambda i: f"real-vs-fake-human-voice-deepfake-audio/spk{i % 4}/fake_{i}.wav")
put(take(fake, a.logo_family_size), lambda i: f"fakeaudio/generated_audio/ljspeech_melgan/LJ001-{i:04d}_gen.wav")
put(take(fake, n), lambda i: f"fakeaudio/generated_audio/ljspeech_hifiGAN/LJ002-{i:04d}_gen.wav")
put(take(real, n), lambda i: f"the-fake-or-real-dataset/for-norm/training/real/file{i}.wav")
put(take(fake, n), lambda i: f"the-fake-or-real-dataset/for-norm/training/fake/file{i}.wav")
put(take(fake, n), lambda i: f"mlaadthe-multi-languagaudioanti-spoofing-dataset/fake/en/tts_sys_{i % 3}/book_{i}_f{i:05d}.wav")
put(take(fake, n), lambda i: f"mlaad-english-500/fake/en/vendor_{i % 3}/book_{i}_f{i:05d}.wav")
put(take(real, n), lambda i: f"vctk-corpus/wav48/p{225 + i % 8}/p{225 + i % 8}_{i:03d}.wav")
put(take(real, n), lambda i: f"accentdb-core-extended/data/{['indian', 'bangla', 'american'][i % 3]}/speaker_0{i % 4}/s_{i}.wav")
put(take(real, n), lambda i: f"common-voice/cv-valid-train/sample-{i:06d}.wav")
put(take(real, n), lambda i: f"the-lj-speech-dataset/wavs/LJ001-{i + 1:04d}.wav")
put(take(fake, n), lambda i: f"extra/deepshield_extra_fakes/en-IN-voice{i % 3}/text{i:03d}.wav")
put(take(real, n), lambda i: f"in-the-wild-audio-deepfake/real/{i}.wav")
put(take(fake, n), lambda i: f"in-the-wild-audio-deepfake/fake/{i}.wav")
print("mirror written to", out, "-", sum(1 for _ in out.rglob("*.wav")), "files")
