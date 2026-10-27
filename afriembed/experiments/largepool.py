"""Large-pool test: speech queries against nested candidate pools from ~300 up to hundreds of thousands of sentences."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..data.ciral import load_ciral
from ..data.fleurs import load_fleurs
from ..data.heldout import SOURCES, load_source
from ..languages import get
from ..metrics import ranks_of_targets
from ..models.embedder import truncate
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?؟])\s+")


def _sentences(text: str, min_chars: int = 20) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) >= min_chars]


def run(
    opts: ModelOpts,
    languages: list[str],
    max_queries: int = 300,
    ciral_passages: int | None = 50_000,
    dims=(768, 256),
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "audio"))
    # distractor passages can run to thousands of tokens; 512 is plenty for them to act as distractors and keeps
    # the padded batch small (whole passages at 8K context x batch 128 asked for >50 GB on an H100)
    full_len = int(getattr(emb.model, "max_seq_length", 8192) or 8192)
    pool_len = min(512, full_len)
    per_lang = {}
    for code in languages:
        lang = get(code)
        # ---- queries: FLEURS test audio (+ NaijaVoices crowd audio where available)
        test = load_fleurs(lang, split="test", max_items=max_queries)
        queries = [(str(u.wav), u.text, "fleurs") for u in test]
        naija = [s for s in SOURCES.values() if s.lang == code and s.kind == "crowd"]
        if naija:
            try:
                nv = load_source(naija[0], max_items=max_queries)
                queries += [(str(u.wav), u.text, "naija") for u in nv]
            except Exception as e:
                log.warning("NaijaVoices %s unavailable: %s", code, str(e)[:120])
        # ---- nested pools (each level adds candidates; the query's own transcript is always in level 0)
        levels: list[tuple[str, list[str]]] = []
        own = [q[1] for q in queries]
        levels.append(("own transcripts", own))
        extra_fleurs = []
        for split in ("train", "validation"):
            try:
                extra_fleurs += [u.text for u in load_fleurs(lang, split=split, max_items=None)]
            except Exception as e:
                log.warning("FLEURS %s/%s unavailable: %s", code, split, str(e)[:100])
        levels.append(("+ all FLEURS sentences", extra_fleurs))
        if code in ("hau", "som", "swh", "yor") and ciral_passages:
            data = load_ciral(code, split="test-a", max_passages=ciral_passages)
            passages = [t for _, t in data.passages.values()]
            levels.append(("+ CIRAL passages (whole)", passages))
            sents = [s for p in passages for s in _sentences(p)]
            levels.append(("+ CIRAL passages split into sentences", sents))
        # dedupe across levels, keep order
        seen: set[str] = set()
        pool: list[str] = []
        level_end: list[tuple[str, int]] = []
        for name, texts in levels:
            for t in texts:
                k = t.strip()
                if k and k not in seen:
                    seen.add(k)
                    pool.append(k)
            level_end.append((name, len(pool)))
        targets = np.array([pool.index(q[1].strip()) for q in queries])
        with timer(f"{code}: encode pool of {len(pool)} texts"):
            emb.model.max_seq_length = (
                pool_len  # text distractors only; audio needs the full context (25 tokens/s)
            )
            P = emb.encode_text(pool, "document", show_progress=True, batch_size=64)
            emb.model.max_seq_length = full_len
        with timer(f"{code}: encode {len(queries)} audio queries"):
            A = emb.encode_audio([q[0] for q in queries], "none", show_progress=True)
        res = {
            "n_queries": len(queries),
            "query_sources": {s: sum(1 for q in queries if q[2] == s) for s in ("fleurs", "naija")},
            "levels": [],
        }
        for d in dims:
            a, p = truncate(A, d), truncate(P, d)
            for name, end in level_end:
                sim = a @ p[:end].T
                ranks = ranks_of_targets(sim, targets)
                entry = {
                    "dim": d,
                    "pool": name,
                    "pool_size": end,
                    "recall@1": float((ranks <= 1).mean()),
                    "recall@10": float((ranks <= 10).mean()),
                    "mrr": float((1 / ranks).mean()),
                }
                for s in ("fleurs", "naija"):
                    m = np.array([q[2] == s for q in queries])
                    if m.any():
                        entry[f"recall@1_{s}"] = float((ranks[m] <= 1).mean())
                res["levels"].append(entry)
                log.info(
                    "%s d=%d %-40s pool=%7d R@1=%.3f R@10=%.3f",
                    code,
                    d,
                    name,
                    end,
                    entry["recall@1"],
                    entry["recall@10"],
                )
        per_lang[code] = res
        save_result(
            "largepool",
            emb.name,
            {"model_info": emb.describe(), "per_language": per_lang, "partial": True},
            tag=opts.full_tag(tag),
        )
        del P, A
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    payload = {"model_info": emb.describe(), "per_language": per_lang}
    save_result("largepool", emb.name, payload, tag=opts.full_tag(tag))
    return payload
