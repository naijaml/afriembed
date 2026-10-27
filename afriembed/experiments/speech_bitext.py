"""Experiment 3: cross-lingual speech bitext mining on FLEURS (n-way parallel by sentence id)."""

from __future__ import annotations

import itertools
import logging

import numpy as np

from ..data.fleurs import align_by_id, load_fleurs
from ..languages import get, with_fleurs
from ..metrics import bitext_accuracy, sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

DIRECTIONS = ("audio->text", "audio->audio", "text->text", "text->audio")


def run(
    opts: ModelOpts,
    languages: list[str],
    pivot: str = "eng",
    split: str = "test",
    max_items: int | None = 400,
    all_pairs: bool = False,
    text_task: str = "bitext",
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "audio"))
    langs = with_fleurs(languages)
    pv = get(pivot)
    cache: dict[str, dict] = {}

    def enc(lang):
        if lang.code not in cache:
            utts = load_fleurs(lang, split=split, max_items=None)  # full split so ids overlap maximally
            with timer(f"{lang.code}: encode {len(utts)} audio+text"):
                A = emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True)
                T = emb.encode_text([u.text for u in utts], text_task)
            cache[lang.code] = {"utts": utts, "A": A, "T": T}
        return cache[lang.code]

    pairs = (
        list(itertools.permutations([l for l in langs], 2))
        if all_pairs
        else [(l, pv) for l in langs if l.code != pv.code]
    )
    results = {}
    for src, tgt in pairs:
        s, t = enc(src), enc(tgt)
        su, tu = align_by_id(s["utts"], t["utts"])
        if max_items and len(su) > max_items:
            su, tu = su[:max_items], tu[:max_items]
        if len(su) < 10:
            log.warning("%s->%s: only %d aligned ids, skipping", src.code, tgt.code, len(su))
            continue
        si = {u.sent_id: i for i, u in enumerate(s["utts"])}
        ti = {u.sent_id: i for i, u in enumerate(t["utts"])}
        sidx = np.array([si[u.sent_id] for u in su])
        tidx = np.array([ti[u.sent_id] for u in tu])
        mats = {
            "audio->text": (s["A"][sidx], t["T"][tidx]),
            "audio->audio": (s["A"][sidx], t["A"][tidx]),
            "text->text": (s["T"][sidx], t["T"][tidx]),
            "text->audio": (s["T"][sidx], t["A"][tidx]),
        }
        res = {k: bitext_accuracy(sim_matrix(a, b)) for k, (a, b) in mats.items()}
        results[f"{src.code}->{tgt.code}"] = {"n": len(su), **res}
        log.info(
            "%s->%s n=%d  a->t %.3f  a->a %.3f  t->t %.3f  t->a %.3f",
            src.code,
            tgt.code,
            len(su),
            *(res[k]["accuracy"] for k in DIRECTIONS),
        )
    summary = {k: float(np.mean([r[k]["accuracy"] for r in results.values()])) for k in DIRECTIONS if results}
    payload = {
        "dataset": "mteb/fleurs",
        "split": split,
        "pivot": pivot,
        "all_pairs": all_pairs,
        "text_task": text_task,
        "model_info": emb.describe(),
        "pairs": results,
        "mean_accuracy": summary,
    }
    save_result("speech-bitext", emb.name, payload, tag=opts.full_tag(tag))
    return payload
