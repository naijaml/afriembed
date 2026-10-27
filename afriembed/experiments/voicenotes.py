"""Product 2 measurement: voice-note search = query-by-example audio->audio retrieval, speaker-independent.

FLEURS records each sentence several times (different readers). For every sentence with at least two
recordings we take one recording as the query and a different recording as the target; the pool holds one
target recording per sentence. R@1 then means "another person saying the same thing is the nearest note".
Text->audio on the same pool is the typed-search case.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..data.fleurs import load_fleurs
from ..languages import with_fleurs
from ..metrics import retrieval_report, sim_matrix
from ..models.embedder import truncate
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test",
    max_sentences: int = 400,
    dims: tuple[int, ...] = (768, 256),
    text_task: str = "document",
    tag: str | None = None,
    seed: int = 0,
) -> dict:
    emb = opts.load(("text", "audio"))
    dims = tuple(d for d in dims if d <= emb.dim)
    rng = np.random.default_rng(seed)
    per_lang = {}
    for lang in with_fleurs(languages):
        utts = load_fleurs(lang, split=split, dedupe=False)
        by_id = defaultdict(list)
        for u in utts:
            by_id[u.sent_id].append(u)
        multi = [v for v in by_id.values() if len(v) >= 2]
        if len(multi) < 20:
            log.warning("%s: only %d sentences with 2+ recordings; skipping", lang.code, len(multi))
            continue
        rng.shuffle(multi)
        multi = multi[:max_sentences]
        queries, targets = [], []
        for recs in multi:
            i, j = rng.choice(len(recs), size=2, replace=False)
            queries.append(recs[i])
            targets.append(recs[j])
        with timer(f"{lang.code}: encode {len(queries)} query + {len(targets)} target recordings"):
            Q = emb.encode_audio([str(u.wav) for u in queries], "none", show_progress=True)
            P = emb.encode_audio([str(u.wav) for u in targets], "none", show_progress=True)
        T = emb.encode_text([u.text for u in targets], text_task)
        idx = np.arange(len(targets))
        res = {}
        for d in dims:
            q, p, t = truncate(Q, d), truncate(P, d), truncate(T, d)
            res[str(d)] = {
                "audio->audio": retrieval_report(sim_matrix(q, p), idx),
                "text->audio": retrieval_report(sim_matrix(t, p), idx),
                "audio->text": retrieval_report(sim_matrix(q, t), idx),
            }
        same_gender = float(np.mean([a.gender == b.gender for a, b in zip(queries, targets, strict=True)]))
        per_lang[lang.code] = {
            "n_sentences": len(targets),
            "n_recordings_total": len(utts),
            "query_target_same_gender_rate": same_gender,
            "dims": res,
        }
        log.info(
            "%s audio->audio R@1 %.3f | text->audio %.3f | audio->text %.3f (pool %d)",
            lang.code,
            res[str(dims[0])]["audio->audio"]["recall@1"],
            res[str(dims[0])]["text->audio"]["recall@1"],
            res[str(dims[0])]["audio->text"]["recall@1"],
            len(targets),
        )
    summary = {
        str(d): {
            k: float(np.mean([v["dims"][str(d)][k]["recall@1"] for v in per_lang.values()]))
            for k in ("audio->audio", "text->audio", "audio->text")
        }
        for d in dims
        if per_lang
    }
    payload = {
        "dataset": "mteb/fleurs",
        "split": split,
        "text_task": text_task,
        "model_info": emb.describe(),
        "per_lang": per_lang,
        "mean_recall@1": summary,
    }
    save_result("voicenotes", emb.name, payload, tag=opts.full_tag(tag))
    return payload
