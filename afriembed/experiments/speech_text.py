"""Experiment 2 (+5 for audio): speech<->text retrieval on FLEURS per African language, all Matryoshka dims from one encoding."""

from __future__ import annotations

import logging

import numpy as np

from ..data.fleurs import load_fleurs
from ..languages import with_fleurs
from ..metrics import retrieval_report, sim_matrix
from ..models.embedder import truncate
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def evaluate_pair(A: np.ndarray, T: np.ndarray, dims: tuple[int, ...]) -> dict[str, dict]:
    """A: audio embeddings, T: text embeddings, aligned row-wise. Returns per-dim metrics in both directions."""
    out = {}
    n = A.shape[0]
    targets = np.arange(n)
    for d in dims:
        a, t = truncate(A, d), truncate(T, d)
        out[str(d)] = {
            "audio->text": retrieval_report(sim_matrix(a, t), targets),
            "text->audio": retrieval_report(sim_matrix(t, a), targets),
            "diag_cos_mean": float(np.mean(np.sum(a * t, axis=1))),
        }
    return out


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test",
    max_items: int | None = 500,
    text_task: str = "document",
    dims: tuple[int, ...] = (768, 512, 256, 128),
    tag: str | None = None,
    shuffle_control: bool = False,
) -> dict:
    emb = opts.load(("text", "audio"))
    dims = tuple(d for d in dims if d <= emb.dim)
    per_lang = {}
    for lang in with_fleurs(languages):
        utts = load_fleurs(lang, split=split, max_items=max_items)
        if len(utts) < 10:
            log.warning("Skipping %s: only %d utterances", lang.code, len(utts))
            continue
        with timer(f"{lang.code}: encode {len(utts)} audio"):
            A = emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True)
        T = emb.encode_text([u.text for u in utts], text_task)
        if shuffle_control:  # negative control: break the audio/transcript pairing; R@1 must fall to ~1/pool
            T = T[np.random.default_rng(0).permutation(len(utts))]
        per_lang[lang.code] = {
            "tier": lang.tier,
            "script": lang.script,
            "n": len(utts),
            "mean_duration_s": float(np.mean([u.duration_s or 0 for u in utts])),
            "dims": evaluate_pair(A, T, dims),
        }
        r1 = per_lang[lang.code]["dims"][str(dims[0])]["audio->text"]["recall@1"]
        log.info("%s audio->text R@1 = %.3f (pool %d)", lang.code, r1, len(utts))
    summary = {}
    for d in dims:
        vals = [v["dims"][str(d)]["audio->text"]["recall@1"] for v in per_lang.values()]
        summary[str(d)] = {
            "mean_audio->text_recall@1": float(np.mean(vals)) if vals else None,
            "retention_vs_full": None,
        }
    full = summary[str(dims[0])]["mean_audio->text_recall@1"]
    for d in dims:
        m = summary[str(d)]["mean_audio->text_recall@1"]
        summary[str(d)]["retention_vs_full"] = (m / full) if (m is not None and full) else None
    payload = {
        "dataset": "mteb/fleurs",
        "split": split,
        "shuffle_control": shuffle_control,
        "text_task": text_task,
        "model_info": emb.describe(),
        "dims": list(dims),
        "per_language": per_lang,
        "summary": summary,
    }
    save_result("speech-text", emb.name, payload, tag=opts.full_tag(tag))
    return payload
