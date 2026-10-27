"""Product 5: cross-corpus overlap and leakage check from audio alone.

Given corpus A (recordings) and corpus B (sentences), find recordings in A whose nearest sentence in B is
the same sentence: shared prompts between datasets, test sentences that leaked into a training set, or
duplicate recordings. Because every source here ships transcripts, precision and recall are measured
exactly against normalised-text equality; the tool is meant for the case where one side has no
transcripts (e.g. an untranscribed collection against a benchmark's test sentences).

Sources: fleurs:<lang>:<split> (one recording per sentence), or the held-out ids (naija-hau, bible-hau, ...).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from ..metrics import sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


@dataclass
class Item:
    text: str
    wav: str


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", "", s).strip()


def load(spec: str, max_items: int = 1000, seed: int = 0) -> list[Item]:
    """spec: fleurs:<lang>:<split>[@seed] or a held-out source id. The seed picks which recording of each
    FLEURS sentence is kept, so fleurs:hau:test@1 vs fleurs:hau:test is a positive control (same sentences,
    mostly different readers)."""
    if "@" in spec:
        spec, seed_s = spec.split("@", 1)
        seed = int(seed_s)
    if spec.startswith("fleurs:"):
        _, lang, split = spec.split(":")
        split = {"dev": "validation"}.get(split, split)
        from .. import languages as L
        from ..data.fleurs import load_fleurs

        utts = load_fleurs(L.get(lang), split=split, max_items=max_items, seed=seed)
        return [Item(u.text, str(u.wav)) for u in utts]
    from ..data.heldout import SOURCES, load_source

    utts = load_source(SOURCES[spec], max_items=max_items, seed=seed)
    return [Item(u.text, str(u.wav)) for u in utts]


def run(
    opts: ModelOpts,
    a: str,
    b: str,
    max_items: int = 1000,
    threshold: float | None = None,
    calibrate_quantile: float = 0.05,
    tag: str | None = None,
) -> dict:
    """Find sentences of B spoken in A. If `threshold` is None it is calibrated on A's own audio/text pairs:
    the similarity below which only `calibrate_quantile` of true pairs fall."""
    emb = opts.load(("text", "audio"))
    A = load(a, max_items)
    B = load(b, max_items)
    with timer(f"embed {len(A)} recordings of {a} and {len(B)} sentences of {b}"):
        EA = emb.encode_audio([x.wav for x in A], "none", show_progress=True)
        TA = emb.encode_text([x.text for x in A], "document")
        TB = emb.encode_text([x.text for x in B], "document")
    own = np.sum(EA * TA, axis=1)  # true-pair similarities, for calibration
    thr = float(np.quantile(own, calibrate_quantile)) if threshold is None else threshold
    S = sim_matrix(EA, TB)
    best = S.argmax(1)
    best_sim = S.max(1)
    found = best_sim >= thr

    nb = {norm(x.text): i for i, x in enumerate(B)}
    truth = np.array([norm(x.text) in nb for x in A])  # A recording's sentence exists in B
    correct_target = np.array([truth[i] and nb[norm(A[i].text)] == best[i] for i in range(len(A))])
    tp = int((found & correct_target).sum())
    fp = int((found & ~correct_target).sum())
    fn = int((truth & ~found).sum())
    examples = [
        {
            "a_text": A[i].text[:100],
            "b_text": B[best[i]].text[:100],
            "sim": round(float(best_sim[i]), 3),
            "true": bool(correct_target[i]),
        }
        for i in np.argsort(-best_sim)[:8]
    ]
    payload = {
        "a": a,
        "b": b,
        "n_a": len(A),
        "n_b": len(B),
        "threshold": thr,
        "calibrated": threshold is None,
        "true_overlap": int(truth.sum()),
        "flagged": int(found.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn) if truth.any() else None,
        "own_pair_sim_mean": float(own.mean()),
        "best_cross_sim_mean": float(best_sim.mean()),
        "top_matches": examples,
        "model_info": emb.describe(),
    }
    log.info(
        "%s vs %s: true overlap %d, flagged %d (tp %d fp %d fn %d) at thr %.3f",
        a,
        b,
        payload["true_overlap"],
        payload["flagged"],
        tp,
        fp,
        fn,
        thr,
    )
    save_result("dedup", emb.name, payload, tag=opts.full_tag(tag))
    return payload
