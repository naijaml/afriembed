"""Experiment 15: accent robustness on AfriSpeech-200 (African-accented English, clinical + general)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..data.afrispeech import load_afrispeech
from ..metrics import ranks_of_targets, sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def run(
    opts: ModelOpts,
    accents: list[str] | None = None,
    split: str = "test",
    max_per_accent: int | None = 30,
    text_task: str = "document",
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "audio"))
    utts = load_afrispeech(accents, split=split, max_per_accent=max_per_accent)
    with timer(f"encode {len(utts)} accented utterances"):
        A = emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True)
    T = emb.encode_text([u.text for u in utts], text_task)
    # transcripts repeat across speakers; the target for utterance i is the first row with identical text
    first = {}
    targets = []
    for i, u in enumerate(utts):
        first.setdefault(u.text.strip().lower(), i)
        targets.append(first[u.text.strip().lower()])
    ranks = ranks_of_targets(sim_matrix(A, T), np.array(targets))
    by = {"accent": defaultdict(list), "country": defaultdict(list), "domain": defaultdict(list)}
    for u, r in zip(utts, ranks, strict=True):
        by["accent"][u.accent].append(r)
        by["country"][u.country].append(r)
        by["domain"][u.domain].append(r)

    def agg(rs):
        rs = np.array(rs)
        return {
            "recall@1": float((rs <= 1).mean()),
            "recall@10": float((rs <= 10).mean()),
            "mrr": float((1 / rs).mean()),
            "n": int(len(rs)),
        }

    payload = {
        "dataset": "AfriSpeech-200",
        "split": split,
        "pool": len(utts),
        "model_info": emb.describe(),
        "overall": agg(ranks),
        "by_accent": {k: agg(v) for k, v in sorted(by["accent"].items())},
        "by_country": {k: agg(v) for k, v in sorted(by["country"].items())},
        "by_domain": {k: agg(v) for k, v in sorted(by["domain"].items())},
    }
    log.info(
        "overall R@1=%.3f over %d utterances / %d accents",
        payload["overall"]["recall@1"],
        len(utts),
        len(by["accent"]),
    )
    save_result("accent", emb.name, payload, tag=opts.full_tag(tag))
    return payload
