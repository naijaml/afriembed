"""Experiment 6: image <-> African-language caption alignment (Afri-Aya, AfriCaption)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..data.captions import load_afri_aya, load_africaption
from ..languages import get
from ..metrics import retrieval_report, sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def _eval_group(emb, items, text_task: str) -> dict:
    with timer(f"encode {len(items)} images"):
        I = emb.encode_images([it.image for it in items], "none", show_progress=True)
    L = emb.encode_text([it.caption_local for it in items], text_task)
    E = emb.encode_text([it.caption_en for it in items], text_task)
    tgt = np.arange(len(items))
    return {
        "n": len(items),
        "image->local": retrieval_report(sim_matrix(I, L), tgt),
        "image->english": retrieval_report(sim_matrix(I, E), tgt),
        "local->image": retrieval_report(sim_matrix(L, I), tgt),
        "english->image": retrieval_report(sim_matrix(E, I), tgt),
        "local<->english_text": retrieval_report(sim_matrix(L, E), tgt),
        "language_gap_cos": float(np.mean(np.sum(I * E, axis=1) - np.sum(I * L, axis=1))),
    }


def run(
    opts: ModelOpts,
    dataset: str = "afri-aya",
    languages: list[str] | None = None,
    max_items: int | None = 300,
    min_per_language: int = 20,
    text_task: str = "document",
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "image"))
    groups: dict[str, list] = defaultdict(list)
    if dataset == "afri-aya":
        for it in load_afri_aya(max_items=None, languages=languages):
            groups[it.language].append(it)
    elif dataset == "africaption":
        cols = [get(c).africaption for c in (languages or ["hau", "yor", "ibo", "amh", "kin", "lug"])]
        for col, items in load_africaption([c for c in cols if c], max_items=max_items).items():
            groups[col] = items
    else:
        raise ValueError("dataset must be afri-aya or africaption")
    per_lang = {}
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_per_language:
            continue
        items = items[:max_items] if max_items else items
        per_lang[name] = _eval_group(emb, items, text_task)
        log.info(
            "%s n=%d image->local R@1=%.3f image->english R@1=%.3f",
            name,
            len(items),
            per_lang[name]["image->local"]["recall@1"],
            per_lang[name]["image->english"]["recall@1"],
        )
    summary = {
        k: float(np.mean([v[k]["recall@1"] for v in per_lang.values()]))
        for k in ("image->local", "image->english", "local->image")
        if per_lang
    }
    payload = {
        "dataset": dataset,
        "text_task": text_task,
        "model_info": emb.describe(),
        "per_language": per_lang,
        "mean_recall@1": summary,
    }
    save_result("vision", emb.name, payload, tag=opts.full_tag(tag))
    return payload
