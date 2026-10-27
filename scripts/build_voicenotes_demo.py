"""Assemble a voice-note demo folder from real crowd recordings (NaijaVoices, cached by `validate`/`heldout`).

Usage: uv run python scripts/build_voicenotes_demo.py [--n 120] [--source naija-hau]
Then:  uv run afriembed index-audio demo/voicenotes --out indexes/voicenotes
       uv run afriembed search indexes/voicenotes --ui --adapter artifacts/audio-lora

Each note is one clip; `transcripts.csv` next to the clips keeps the reference text so you know what to ask for.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="naija-hau")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "demo" / "voicenotes"))
    a = ap.parse_args()

    from afriembed.data.heldout import SOURCES, load_source

    utts = load_source(SOURCES[a.source], max_items=a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, u in enumerate(utts):
        dst = out / f"{a.source}-{i:03d}.wav"
        shutil.copy(u.wav, dst)
        rows.append({"audio": dst.name, "text": u.text, "lang": u.lang, "seconds": round(u.duration_s, 2)})
    with open(out / "transcripts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio", "text", "lang", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} notes -> {out}  ({sum(r['seconds'] for r in rows) / 60:.1f} min)")


if __name__ == "__main__":
    main()
