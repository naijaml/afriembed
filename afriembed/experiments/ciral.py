"""Experiment 13: cross-lingual IR on CIRAL (English queries -> African-language passages)."""

from __future__ import annotations

import logging

import numpy as np

from ..data.ciral import load_ciral
from ..metrics import ndcg_at_k, recall_qrels_at_k
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def _passage_inputs(emb, passages: dict[str, tuple[str, str]]) -> tuple[list[str], str]:
    """EmbeddingGemma gets real titles in its document prefix; other families get 'title. text' with their own prefix."""
    ids = list(passages)
    if emb.family.startswith("embeddinggemma") and emb.use_prompts:
        texts = [f"title: {passages[i][0] or 'none'} | text: {passages[i][1]}" for i in ids]
        return texts, "none"
    texts = [f"{passages[i][0]}. {passages[i][1]}" if passages[i][0] else passages[i][1] for i in ids]
    return texts, "document"


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test-a",
    max_passages: int | None = 50_000,
    translated: bool = False,
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text",))
    per_lang = {}
    for code in languages:
        data = load_ciral(code, split=split, max_passages=max_passages, translated=translated)
        qids = list(data.queries)
        doc_ids = list(data.passages)
        texts, task = _passage_inputs(emb, data.passages)
        with timer(f"CIRAL {code} encode {len(texts)} passages"):
            D = emb.encode_text(texts, task, show_progress=True)
        Q = emb.encode_text([data.queries[q] for q in qids], "query")
        sim = Q @ D.T
        top = np.argsort(-sim, axis=1)[:, :100]
        ranked = [[doc_ids[j] for j in row] for row in top]
        qrels = {q: data.qrels[q] for q in qids}
        per_lang[code] = {
            "ndcg@20": ndcg_at_k(ranked, qrels, 20),
            "ndcg@10": ndcg_at_k(ranked, qrels, 10),
            "recall@100": recall_qrels_at_k(ranked, qrels, 100),
            "n_queries": len(qids),
            "n_passages": len(doc_ids),
            "subsampled": bool(max_passages and len(doc_ids) >= max_passages),
        }
        log.info("CIRAL %s: %s", code, per_lang[code])
    payload = {
        "split": split,
        "translated_passages": translated,
        "model_info": emb.describe(),
        "per_language": per_lang,
        "mean_ndcg@20": float(np.mean([v["ndcg@20"] for v in per_lang.values()])),
    }
    save_result("ciral", emb.name, payload, tag=opts.full_tag(tag))
    return payload
