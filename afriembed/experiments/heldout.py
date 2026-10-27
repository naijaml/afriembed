"""Held-out speech check: speech<->text retrieval on non-FLEURS sources (NaijaVoices crowd speech, BibleTTS verses)."""

from __future__ import annotations

import logging

import numpy as np

from ..data.heldout import SOURCES, load_source
from ..utils import save_result, timer
from .common import ModelOpts
from .speech_text import evaluate_pair

log = logging.getLogger(__name__)


def run(
    opts: ModelOpts,
    sources: list[str] | None = None,
    max_items: int = 1500,
    text_task: str = "document",
    dims=(768, 512, 256, 128),
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "audio"))
    names = sources or list(SOURCES)
    per_source = {}
    for name in names:
        src = SOURCES[name]
        try:
            utts = load_source(src, max_items=max_items)
        except Exception as e:
            log.warning("%s unavailable: %s", name, str(e)[:200])
            per_source[name] = {"error": str(e)[:300]}
            continue
        if len(utts) < 20:
            per_source[name] = {"error": f"only {len(utts)} utterances"}
            continue
        with timer(f"{name}: encode {len(utts)} audio"):
            A = emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True)
        T = emb.encode_text([u.text for u in utts], text_task)
        per_source[name] = {
            "lang": src.lang,
            "kind": src.kind,
            "n": len(utts),
            "mean_duration_s": float(np.mean([u.duration_s for u in utts])),
            "dims": evaluate_pair(A, T, tuple(d for d in dims if d <= emb.dim)),
        }
        log.info(
            "%s audio->text R@1 = %.3f (pool %d)",
            name,
            per_source[name]["dims"]["768"]["audio->text"]["recall@1"],
            len(utts),
        )
        # save after every source so a dropped connection keeps what was measured
        save_result(
            "heldout",
            emb.name,
            {
                "sources": names,
                "text_task": text_task,
                "model_info": emb.describe(),
                "per_source": per_source,
                "partial": True,
            },
            tag=opts.full_tag(tag),
        )
    ok = {k: v for k, v in per_source.items() if "dims" in v}
    summary = {
        str(d): float(np.mean([v["dims"][str(d)]["audio->text"]["recall@1"] for v in ok.values()]))
        for d in dims
        if ok
    }
    payload = {
        "sources": names,
        "text_task": text_task,
        "model_info": emb.describe(),
        "per_source": per_source,
        "mean_audio->text_recall@1": summary,
    }
    save_result("heldout", emb.name, payload, tag=opts.full_tag(tag))
    return payload
