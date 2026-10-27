"""Experiment 4: tokenizer fertility ('African language tax') for Gemma 4 vs Gemma 3 vs XLM-R vs NLLB."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..config import hf_token, tokenizer_registry
from ..data.text_corpora import sib200_texts
from ..languages import get
from ..utils import save_result

log = logging.getLogger(__name__)

_BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def tokenizer_stats(tok, texts: list[str]) -> dict[str, float]:
    n_tokens = n_words = n_chars = n_unk = n_bytes = 0
    unk_id = getattr(tok, "unk_token_id", None)
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tokens += len(ids)
        n_words += max(1, len(t.split()))
        n_chars += len(t)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
        pieces = tok.convert_ids_to_tokens(ids)
        n_bytes += sum(1 for p in pieces if isinstance(p, str) and _BYTE_TOKEN.match(p))
    return {
        "tokens_per_word": n_tokens / n_words,
        "tokens_per_char": n_tokens / max(1, n_chars),
        "chars_per_token": n_chars / max(1, n_tokens),
        "unk_rate": n_unk / max(1, n_tokens),
        "byte_fallback_rate": n_bytes / max(1, n_tokens),
        "n_sentences": len(texts),
        "n_tokens": n_tokens,
    }


def run(
    languages: list[str],
    tokenizers: dict[str, str] | None = None,
    split: str = "test",
    max_sentences: int | None = None,
) -> dict:
    from transformers import AutoTokenizer

    tokenizers = tokenizers or tokenizer_registry()
    corpora = {}
    for code in languages:
        lang = get(code)
        if not lang.sib:
            log.warning("%s has no SIB-200 config; skipped", code)
            continue
        try:
            texts = sib200_texts(lang, split)
        except Exception as e:
            log.warning("%s: SIB-200 %s unavailable (%s); skipped", code, lang.sib, str(e)[:80])
            continue
        corpora[code] = texts[:max_sentences] if max_sentences else texts
    languages = list(corpora)
    table: dict[str, dict[str, dict]] = {}
    for tname, tid in tokenizers.items():
        try:
            tok = AutoTokenizer.from_pretrained(tid, token=hf_token())
        except Exception as e:
            log.warning("Skipping tokenizer %s (%s): %s", tname, tid, e)
            continue
        per_lang = {code: tokenizer_stats(tok, texts) for code, texts in corpora.items()}
        eng = per_lang.get("eng", {}).get("tokens_per_word")
        for code, s in per_lang.items():
            s["premium_vs_eng"] = (s["tokens_per_word"] / eng) if eng else None
            s["tier"] = get(code).tier
            s["script"] = get(code).script
        table[tname] = {
            "hf_id": tid,
            "vocab_size": int(getattr(tok, "vocab_size", 0) or len(tok)),
            "languages": per_lang,
        }
        log.info(
            "%s: mean tokens/word over African langs = %.2f",
            tname,
            np.mean([s["tokens_per_word"] for c, s in per_lang.items() if c not in ("eng", "fra", "arb")]),
        )
    payload = {"languages": languages, "split": split, "tokenizers": table}
    save_result("fertility", "tokenizers", payload)
    return payload
