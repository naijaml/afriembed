"""CIRAL: cross-lingual IR, English queries -> Hausa/Somali/Swahili/Yoruba news passages."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

from ..config import hf_token

log = logging.getLogger(__name__)

_LANG = {"hau": ("hausa", "ha"), "som": ("somali", "so"), "swh": ("swahili", "sw"), "yor": ("yoruba", "yo")}


@dataclass
class CiralSplit:
    lang: str
    split: str
    queries: dict[str, str]  # qid -> English query
    qrels: dict[str, dict[str, int]]  # qid -> {docid: rel}
    passages: dict[str, tuple[str, str]]  # docid -> (title, text)


def load_ciral(
    lang: str, split: str = "test-a", max_passages: int | None = None, seed: int = 0, translated: bool = False
) -> CiralSplit:
    """split in {dev, test-a, test-b}. max_passages subsamples the corpus but always keeps judged docs (flagged in results)."""
    if lang not in _LANG:
        raise KeyError(f"CIRAL covers {sorted(_LANG)}")
    full, short = _LANG[lang]
    tok = hf_token()
    topics = hf_hub_download(
        "CIRAL/ciral",
        f"ciral-{full}/topics/topics.ciral-v1.0-{short}-{split}.tsv",
        repo_type="dataset",
        token=tok,
    )
    qrels_p = hf_hub_download(
        "CIRAL/ciral",
        f"ciral-{full}/qrels/qrels.ciral-v1.0-{short}-{split}.tsv",
        repo_type="dataset",
        token=tok,
    )
    sub = "translated-passages-v1.0" if translated else "passages-v1.0"
    corpus = hf_hub_download(
        "CIRAL/ciral-corpus", f"{sub}/{full}_passages.jsonl", repo_type="dataset", token=tok
    )

    queries = {}
    with open(topics, encoding="utf-8") as f:
        for line in f:
            qid, q = line.rstrip("\n").split("\t", 1)
            queries[qid] = q
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_p, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    queries = {q: t for q, t in queries.items() if q in qrels}
    judged = {d for rels in qrels.values() for d in rels}

    passages: dict[str, tuple[str, str]] = {}
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            passages[r["docid"]] = ((r.get("title") or "").strip(), r.get("text") or "")
    if max_passages and len(passages) > max_passages:
        rng = random.Random(seed)
        others = [d for d in passages if d not in judged]
        keep = set(rng.sample(others, max(0, max_passages - len(judged)))) | judged
        passages = {d: passages[d] for d in keep}
        log.warning(
            "CIRAL %s: subsampled corpus to %d passages (all %d judged docs kept)",
            lang,
            len(passages),
            len(judged),
        )
    log.info("CIRAL %s/%s: %d queries, %d passages", lang, split, len(queries), len(passages))
    return CiralSplit(lang, split, queries, qrels, passages)
