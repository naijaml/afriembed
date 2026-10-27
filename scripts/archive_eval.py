"""Product 8 measurement: can a typed or spoken query find the right 10 s window in a long-form audio archive?

Uses demo/archive (built by build_archive_demo.py, with segments.csv giving each verse's true position) and
indexes/archive. Typed query = the verse text; spoken query = the original verse recording. A hit counts when
the top-ranked window overlaps the verse's true span.

Usage: uv run python scripts/archive_eval.py [--adapter artifacts/audio-lora] [--n 97]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def overlaps(hit: dict, row: dict) -> bool:
    return (
        Path(hit["file"]).name == row["file"]
        and hit["start_s"] < float(row["end_s"])
        and hit["end_s"] > float(row["start_s"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/archive")
    ap.add_argument("--adapter", default="artifacts/audio-lora")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    from afriembed.data.heldout import SOURCES, load_source
    from afriembed.experiments.common import ModelOpts
    from afriembed.experiments.search import Searcher
    from afriembed.utils import save_result

    rows = list(csv.DictReader(open(ROOT / "demo" / "archive" / "segments.csv", encoding="utf-8")))[: a.n]
    clips = {u.text: str(u.wav) for u in load_source(SOURCES["bible-hau"], max_items=400)}
    s = Searcher(ModelOpts(model="embeddinggemma-2", adapter=a.adapter), Path(a.index), need_audio=True)
    typed_r1, typed_r3, spoken_r1, spoken_r3, n_spoken = [], [], [], [], 0
    for r in rows:
        hits = s(query=r["text"], k=3)
        typed_r1.append(overlaps(hits[0], r))
        typed_r3.append(any(overlaps(h, r) for h in hits))
        if r["text"] in clips:
            hits = s(audio_query=Path(clips[r["text"]]), k=3)
            spoken_r1.append(overlaps(hits[0], r))
            spoken_r3.append(any(overlaps(h, r) for h in hits))
            n_spoken += 1
    res = {
        "n_queries": len(rows),
        "n_windows": len(s.idx.meta),
        "archive_minutes": round(sum(1 for _ in s.idx.meta) * 10 / 60, 1),
        "typed_window_hit@1": float(np.mean(typed_r1)),
        "typed_window_hit@3": float(np.mean(typed_r3)),
        "spoken_n": n_spoken,
        "spoken_window_hit@1": float(np.mean(spoken_r1)) if spoken_r1 else None,
        "spoken_window_hit@3": float(np.mean(spoken_r3)) if spoken_r3 else None,
    }
    print(json.dumps(res, indent=1))
    save_result("archive", s.emb.name, {**res, "model_info": s.emb.describe()}, tag="adapter")


if __name__ == "__main__":
    main()
