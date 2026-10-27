"""Product 4: ASR-free spoken intent routing (Spoken Injongo).

Injongo (Yu et al., 2025) has 40 banking/travel/home intents with utterances in 16 African languages. We embed
the training utterances once (text tower) and classify a new utterance by nearest neighbours in embedding
space, with no ASR: a typed utterance goes through the text tower, a spoken one (MMS-TTS here, a microphone in
the product) through the audio tower. Two classifiers are reported: k-NN vote over training utterances and
nearest intent centroid. Accuracy is on the test split.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

from ..config import hf_token
from ..metrics import sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

HF_ID = "masakhane/InjongoIntent"


def load_split(lang: str, split: str) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, lang, split=split, token=hf_token())
    texts = [r["text"].strip() for r in ds]
    intents = [r["intent"] for r in ds]
    return texts, intents


def knn_predict(S: np.ndarray, train_labels: list[str], k: int = 5) -> list[str]:
    out = []
    for row in S:
        top = np.argsort(-row)[:k]
        votes = Counter()
        for j in top:
            votes[train_labels[j]] += float(row[j])  # similarity-weighted vote
        out.append(votes.most_common(1)[0][0])
    return out


def centroids(E: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    groups = defaultdict(list)
    for e, lab in zip(E, labels, strict=True):
        groups[lab].append(e)
    names = sorted(groups)
    C = np.stack([np.mean(groups[n], axis=0) for n in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C, names


def run(
    opts: ModelOpts,
    languages: list[str],
    k: int = 5,
    max_test: int | None = None,
    train_split: str = "train",
    test_split: str = "test",
    tag: str | None = None,
) -> dict:
    from ..data.tts import TTS_MODELS, synthesize_all

    emb = opts.load(("text", "audio"))
    per_lang = {}
    for lang in languages:
        tr_text, tr_lab = load_split(lang, train_split)
        te_text, te_lab = load_split(lang, test_split)
        if max_test:
            te_text, te_lab = te_text[:max_test], te_lab[:max_test]
        with timer(f"{lang}: embed {len(tr_text)} train + {len(te_text)} test utterances"):
            TR = emb.encode_text(tr_text, "document", show_progress=True)
            TE = emb.encode_text(te_text, "query")
        C, names = centroids(TR, tr_lab)
        res = {"n_train": len(tr_text), "n_test": len(te_text), "n_intents": len(names), "k": k}

        def acc(pred: list[str], gold: list[str] = te_lab) -> float:
            return float(np.mean([p == g for p, g in zip(pred, gold, strict=True)]))

        res["typed_knn_acc"] = acc(knn_predict(sim_matrix(TE, TR), tr_lab, k))
        res["typed_centroid_acc"] = acc([names[i] for i in sim_matrix(TE, C).argmax(1)])
        if lang in TTS_MODELS:
            with timer(f"{lang}: synthesise {len(te_text)} spoken test utterances"):
                wavs = synthesize_all(te_text, lang, namespace="tts-injongo")
            with timer(f"{lang}: embed {len(wavs)} spoken utterances"):
                A = emb.encode_audio([str(w) for w in wavs], "none", show_progress=True)
            res["spoken_knn_acc"] = acc(knn_predict(sim_matrix(A, TR), tr_lab, k))
            res["spoken_centroid_acc"] = acc([names[i] for i in sim_matrix(A, C).argmax(1)])
            # confusions of the spoken k-NN classifier
            pred = knn_predict(sim_matrix(A, TR), tr_lab, k)
            conf = Counter((g, p) for p, g in zip(pred, te_lab, strict=True) if p != g)
            res["spoken_top_confusions"] = [f"{g} -> {p} ({n})" for (g, p), n in conf.most_common(6)]
        per_lang[lang] = res
        log.info(
            "%s: typed kNN %.3f centroid %.3f | spoken kNN %s centroid %s (%d intents)",
            lang,
            res["typed_knn_acc"],
            res["typed_centroid_acc"],
            f"{res['spoken_knn_acc']:.3f}" if "spoken_knn_acc" in res else "-",
            f"{res['spoken_centroid_acc']:.3f}" if "spoken_centroid_acc" in res else "-",
            len(names),
        )
    keys = ("typed_knn_acc", "typed_centroid_acc", "spoken_knn_acc", "spoken_centroid_acc")
    payload = {
        "dataset": HF_ID,
        "model_info": emb.describe(),
        "per_lang": per_lang,
        "mean": {
            key: float(np.mean([v[key] for v in per_lang.values() if key in v]))
            for key in keys
            if any(key in v for v in per_lang.values())
        },
    }
    save_result("spoken-intent", emb.name, payload, tag=opts.full_tag(tag))
    return payload
