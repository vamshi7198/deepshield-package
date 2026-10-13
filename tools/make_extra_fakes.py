"""Generate the optional `deepshield_extra_fakes/` training set for round 3: synthetic speech in voices the
earlier rounds never had as fakes (by default Indian-accented English, where training had ~1,000 genuine clips
and no fakes at all).

    pip install edge-tts
    python tools/make_extra_fakes.py --out extra_fakes_dataset --n 1500

Then upload `extra_fakes_dataset/` as a private Kaggle dataset and attach it to the round-3 notebook; the
pipeline finds the `deepshield_extra_fakes/` folder by name. Layout: deepshield_extra_fakes/<voice>/<text id>.wav,
16 kHz mono, leading/trailing silence trimmed. The same text id across voices is one group, so a sentence never
appears in both training and test.

Uses Microsoft's online neural voices through edge-tts (needs internet; check that this use fits your
organisation's terms). Any TTS system can be swapped in: only the folder layout matters.
"""
import argparse, asyncio, itertools, random
from pathlib import Path

import edge_tts
import librosa
import numpy as np
import soundfile as sf

VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaExpressiveNeural",
          "hi-IN-SwaraNeural", "hi-IN-MadhurNeural", "ta-IN-PallaviNeural", "te-IN-MohanNeural"]

# Short, varied, call-centre / voice-note style sentences built from slots, so each text id is distinct.
OPENERS = ["Hi, this is", "Hello, my name is", "Good morning, I am", "Hey, it's", "Namaste, this is"]
NAMES = ["Ravi", "Priya", "Arjun", "Meera", "Karthik", "Ananya", "Vikram", "Divya", "Rahul", "Sneha"]
REQUESTS = [
    "please transfer {amt} rupees to the account I shared yesterday",
    "I need my one time password resent to this number",
    "can you confirm my appointment on {day} at {time}",
    "the delivery for order {num} has not reached me yet",
    "I am calling about the payment that failed this morning",
    "please call me back after {time}, I am in a meeting",
    "my card was blocked when I tried to pay at the {place}",
    "I would like to update the address on my account",
    "the meeting with the {place} team is moved to {day}",
    "send the documents to my email before {time} today",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TIMES = ["ten thirty", "four o'clock", "half past two", "noon", "six in the evening"]
PLACES = ["airport", "hospital", "bank", "petrol pump", "supermarket", "Bangalore", "Chennai", "Pune"]


def texts(n, rng):
    out = set()
    for o, nm, rq in itertools.product(OPENERS, NAMES, REQUESTS):
        out.add(f"{o} {nm}, " + rq.format(amt=rng.choice([500, 2000, 15000, 48000]), day=rng.choice(DAYS),
                                           time=rng.choice(TIMES), num=rng.randint(10000, 99999), place=rng.choice(PLACES)) + ".")
    out = sorted(out)
    rng.shuffle(out)
    return out[:n]


async def main(a):
    rng = random.Random(a.seed)
    root = Path(a.out) / "deepshield_extra_fakes"
    ts = texts(a.n, rng)
    for i, t in enumerate(ts):
        v = VOICES[i % len(VOICES)] if not a.all_voices else None
        for voice in ([v] if v else VOICES):
            wav = root / voice / f"text{i:05d}.wav"
            if wav.exists():
                continue
            wav.parent.mkdir(parents=True, exist_ok=True)
            mp3 = wav.with_suffix(".mp3")
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(t, voice).save(str(mp3))
                    break
                except Exception as e:
                    if attempt == 2:
                        print("failed", voice, i, e)
                    await asyncio.sleep(2 + 3 * attempt)
            if not mp3.exists():
                continue
            y, _ = librosa.load(mp3, sr=16000)
            y, _ = librosa.effects.trim(y, top_db=40)
            sf.write(wav, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), 16000, subtype="PCM_16")
            mp3.unlink()
        if (i + 1) % 100 == 0:
            print(i + 1, "texts done")
    print("wrote", sum(1 for _ in root.rglob("*.wav")), "clips to", root)


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1500, help="number of distinct sentences")
ap.add_argument("--all-voices", action="store_true", help="every sentence in every voice (default: one voice each)")
ap.add_argument("--seed", type=int, default=0)
asyncio.run(main(ap.parse_args()))
