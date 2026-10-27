"""Assemble a small long-form "radio archive" for product 8 from public read speech.

Concatenates BibleTTS verses (Hausa by default, Ewe also available) into programme-length files of a few
minutes each with short pauses, plus the chapter files left by `afriembed align eval` if present, so the
archive has real continuous speech to search through in 10 s windows.

Usage: uv run python scripts/build_archive_demo.py [--minutes 12] [--lang hau]
Then:  uv run afriembed index-audio demo/archive --out indexes/archive --adapter artifacts/audio-lora
       uv run afriembed search indexes/archive --ui --adapter artifacts/audio-lora
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hau")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--programme-minutes", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT / "demo" / "archive"))
    a = ap.parse_args()

    from afriembed.config import cache_dir
    from afriembed.data.audio_utils import TARGET_SR, load_any, to_mono_16k, write_wav
    from afriembed.data.heldout import SOURCES, load_source

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = {"hau": "bible-hau", "ewe": "bible-ewe"}[a.lang]
    utts = load_source(SOURCES[src], max_items=int(a.minutes * 60 / 4) + 50)
    gap = np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)
    rows, pieces, t, n, k = [], [], 0.0, 0, 0
    for u in utts:
        y, sr = load_any(u.wav)
        y = to_mono_16k(y, sr)
        rows.append({"file": f"programme{k:02d}.wav", "start_s": round(t, 2), "end_s": round(t + len(y) / TARGET_SR, 2), "text": u.text})
        pieces += [y, gap]
        t += (len(y) + len(gap)) / TARGET_SR
        n += 1
        if t >= a.programme_minutes * 60:
            write_wav(out / f"programme{k:02d}.wav", np.concatenate(pieces))
            pieces, t, k = [], 0.0, k + 1
            if k * a.programme_minutes >= a.minutes:
                break
    if pieces:
        write_wav(out / f"programme{k:02d}.wav", np.concatenate(pieces))
    chapters = sorted((cache_dir() / "align-eval" / a.lang).glob("chapter*.wav"))
    for c in chapters[:4]:
        shutil.copy(c, out / f"{a.lang}-{c.name}")
    with open(out / "segments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "start_s", "end_s", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{n} verses into {k + 1} programmes (+{len(chapters[:4])} chapters) -> {out}")


if __name__ == "__main__":
    main()
