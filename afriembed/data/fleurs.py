"""FLEURS (mteb/fleurs parquet mirror): aligned speech + transcript, n-way parallel across languages via `id`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import cache_dir, hf_token
from ..languages import Language
from .audio_utils import cached_wav_path, decode_hf_audio, ensure_wav

log = logging.getLogger(__name__)

HF_ID = "mteb/fleurs"


@dataclass
class Utterance:
    sent_id: int  # FLORES sentence id, shared across languages
    lang: str  # ISO code from our registry
    text: str  # raw_transcription (cased, punctuated)
    norm_text: str  # transcription (lowercased, normalised)
    wav: Path
    gender: str | None = None
    duration_s: float | None = None


def load_fleurs(
    lang: Language, split: str = "test", max_items: int | None = None, seed: int = 0, dedupe: bool = True
) -> list[Utterance]:
    """Download one FLEURS language split, write 16 kHz wavs to the cache, return utterances.

    FLEURS has several recordings per sentence id; `dedupe` keeps one recording per sentence so that
    retrieval pools have unique targets.
    """
    from datasets import Audio, load_dataset

    assert lang.fleurs, f"{lang.code} has no FLEURS config"
    ds = load_dataset(HF_ID, lang.fleurs, split=split, token=hf_token())
    ds = ds.cast_column("audio", Audio(decode=False))  # decode bytes ourselves (no torchcodec dependency)
    if max_items:
        ds = ds.shuffle(seed=seed)
    seen: set[int] = set()
    out: list[Utterance] = []
    root = cache_dir()
    for row in ds:
        sid = int(row["id"])
        if dedupe and sid in seen:
            continue
        key = f"{lang.fleurs}/{split}/{sid}/{row.get('num_samples', 0)}"
        wav_path = cached_wav_path(root, "fleurs", key)
        if wav_path.exists():
            n_samples = int(row.get("num_samples") or 0)
            duration = n_samples / 16_000 if n_samples else None
        else:
            y, sr = decode_hf_audio(row["audio"])
            ensure_wav(root, "fleurs", key, y, sr)
            duration = len(y) / sr
        out.append(
            Utterance(
                sent_id=sid,
                lang=lang.code,
                text=row.get("raw_transcription") or row["transcription"],
                norm_text=row["transcription"],
                wav=wav_path,
                gender=str(row.get("gender")),
                duration_s=duration,
            )
        )
        seen.add(sid)
        if max_items and len(out) >= max_items:
            break
    log.info("FLEURS %s/%s: %d utterances (%s)", lang.code, split, len(out), "deduped" if dedupe else "all")
    return out


def align_by_id(a: list[Utterance], b: list[Utterance]) -> tuple[list[Utterance], list[Utterance]]:
    """Return the sub-lists of a and b with common sentence ids, in the same order."""
    bi = {u.sent_id: u for u in b}
    pairs = [(u, bi[u.sent_id]) for u in a if u.sent_id in bi]
    return [p[0] for p in pairs], [p[1] for p in pairs]
