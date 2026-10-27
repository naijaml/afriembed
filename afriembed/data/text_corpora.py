"""Small text corpora used for tokenizer fertility, vocab trimming, and text fine-tuning pairs."""

from __future__ import annotations

import logging

from ..config import hf_token
from ..languages import Language, get

log = logging.getLogger(__name__)


def sib200_texts(lang: Language, split: str = "test") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("mteb/sib200", lang.sib, split=split, token=hf_token())
    return [r["text"] for r in ds]


def sib200_labeled(lang: Language, split: str) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset("mteb/sib200", lang.sib, split=split, token=hf_token())
    return [r["text"] for r in ds], [int(r["label"]) for r in ds]


def injongo_texts(lang: Language, split: str = "train") -> list[str]:
    from datasets import load_dataset

    if not lang.injongo:
        return []
    ds = load_dataset("masakhane/InjongoIntent", lang.injongo, split=split, token=hf_token())
    return [r["text"] for r in ds]


def africaption_pairs(lang: Language, max_items: int | None = None) -> list[tuple[str, str]]:
    """(english, local) caption pairs. Not part of any AfriMTEB test task."""
    from datasets import load_dataset

    if not lang.africaption:
        return []
    ds = load_dataset("AfriMM/AFRICaption", split="train", token=hf_token())
    pairs = [(r["eng"], r[lang.africaption]) for r in ds if r.get(lang.africaption) and r.get("eng")]
    return pairs[:max_items] if max_items else pairs


def afri_aya_pairs(max_items: int | None = None) -> list[tuple[str, str, str]]:
    """(language, caption_en, caption_local)."""
    from datasets import load_dataset

    ds = load_dataset("CohereLabsCommunity/afri-aya", split="train", token=hf_token())
    out = [
        (str(r["language"]), r["caption_en"], r["caption_local"])
        for r in ds
        if r.get("caption_en") and r.get("caption_local")
    ]
    return out[:max_items] if max_items else out


# Sources that are *test* data in AfriMTEB and must never feed fine-tuning.
CONTAMINATION_BLOCKLIST = (
    "FloresBitextMining",
    "NTREXBitextMining",
    "SIB200 test",
    "Belebele",
    "AfriXNLI test",
    "MasakhaNEWS test",
    "InjongoIntent test",
)


def mixed_monolingual(lang_codes: list[str], max_per_source: int = 2000) -> dict[str, list[str]]:
    """Texts per language from SIB-200 (train+validation only) and Injongo train, for vocab statistics."""
    out: dict[str, list[str]] = {}
    for c in lang_codes:
        lang = get(c)
        texts: list[str] = []
        for split in ("train", "validation"):
            try:
                texts += sib200_texts(lang, split)[:max_per_source]
            except Exception as e:  # pragma: no cover
                log.warning("SIB200 %s/%s unavailable: %s", c, split, e)
        try:
            texts += injongo_texts(lang, "train")[:max_per_source]
        except Exception as e:  # pragma: no cover
            log.warning("Injongo %s unavailable: %s", c, e)
        out[c] = texts
    return out
