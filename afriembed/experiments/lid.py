"""Experiment 14: spoken language identification and language clustering from embeddings alone (FLEURS)."""

from __future__ import annotations

import logging

import numpy as np

from ..data.fleurs import load_fleurs
from ..languages import with_fleurs
from ..metrics import clustering_report, knn_accuracy, sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def _confusions(
    emb: np.ndarray, labels: np.ndarray, names: list[str], k: int = 5, top: int = 8
) -> list[dict]:
    sim = sim_matrix(emb, emb)
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1)[:, :k]
    pred = np.array([np.bincount(labels[row]).argmax() for row in idx])
    conf: dict[tuple[int, int], int] = {}
    for y, p in zip(labels, pred, strict=True):
        if y != p:
            conf[(y, p)] = conf.get((y, p), 0) + 1
    return [
        {"true": names[a], "pred": names[b], "count": c}
        for (a, b), c in sorted(conf.items(), key=lambda x: -x[1])[:top]
    ]


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test",
    per_lang: int = 100,
    k: int = 5,
    tag: str | None = None,
) -> dict:
    emb = opts.load(("text", "audio"))
    langs = with_fleurs(languages)
    names = [l.code for l in langs]
    A_all, T_all, y = [], [], []
    for i, lang in enumerate(langs):
        utts = load_fleurs(lang, split=split, max_items=per_lang)
        with timer(f"{lang.code}: encode {len(utts)}"):
            A_all.append(emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True))
            T_all.append(emb.encode_text([u.text for u in utts], "clustering"))
        y += [i] * len(utts)
    A, T, y = np.vstack(A_all), np.vstack(T_all), np.array(y)
    res = {
        "audio": {
            "knn_accuracy": knn_accuracy(A, y, k),
            **clustering_report(A, y),
            "top_confusions": _confusions(A, y, names, k),
        },
        "text": {
            "knn_accuracy": knn_accuracy(T, y, k),
            **clustering_report(T, y),
            "top_confusions": _confusions(T, y, names, k),
        },
        "chance": 1.0 / len(langs),
    }
    log.info(
        "LID audio kNN=%.3f NMI=%.3f | text kNN=%.3f NMI=%.3f",
        res["audio"]["knn_accuracy"],
        res["audio"]["nmi"],
        res["text"]["knn_accuracy"],
        res["text"]["nmi"],
    )
    payload = {
        "dataset": "mteb/fleurs",
        "languages": names,
        "per_lang": per_lang,
        "k": k,
        "model_info": emb.describe(),
        **res,
    }
    save_result("lid", emb.name, payload, tag=opts.full_tag(tag))
    return payload
