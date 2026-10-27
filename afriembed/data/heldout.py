"""Held-out speech sources that are not FLEURS: crowd-recorded NaijaVoices and verse-level BibleTTS sets.

Each source yields (text, wav_path) pairs through one generic loader that detects the audio and text columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ..config import cache_dir, hf_token
from .audio_utils import decode_hf_audio, ensure_wav

log = logging.getLogger(__name__)

TEXT_COLS = ("transcription", "transcript", "text", "sentence", "raw_transcription")


@dataclass(frozen=True)
class Source:
    name: str  # short id used in results
    lang: str  # ISO code
    repo: str
    split: str = "train"
    parquet_prefix: str | None = None  # for gated/sharded repos: load the first shard under this prefix
    kind: str = "read"  # bible | crowd, for grouping


SOURCES: dict[str, Source] = {
    "bible-hau": Source(
        "bible-hau", "hau", "vpetukhov/bible_tts_hausa", parquet_prefix="data/train-00000-", kind="bible"
    ),  # test split has only 124 verses; the train split is 44 shards / 21 GB, so load one shard (~1,000 verses)
    "bible-ewe": Source(
        "bible-ewe",
        "ewe",
        "analist/bibletts-ewe-16k-public",
        split="test",
        parquet_prefix="data/test-",
        kind="bible",
    ),  # parquet mirror test split (409 verses, one shard); loading via load_dataset would pull the 16 GB train split too
    "naija-hau": Source(
        "naija-hau", "hau", "naijavoices/naijavoices-dataset", parquet_prefix="hausa-batch-0/", kind="crowd"
    ),
    "naija-ibo": Source(
        "naija-ibo", "ibo", "naijavoices/naijavoices-dataset", parquet_prefix="igbo-batch-0/", kind="crowd"
    ),
    "naija-yor": Source(
        "naija-yor", "yor", "naijavoices/naijavoices-dataset", parquet_prefix="yoruba-batch-0/", kind="crowd"
    ),
}
# WAXAL (Google + African universities, Feb 2026): conversational ASR test shards from the community mirror.
for _code in ("amh", "ewe", "ful", "lin", "lug", "orm", "sna", "ach", "aka", "tir", "nyn", "mlg"):
    SOURCES[f"waxal-{_code}"] = Source(
        f"waxal-{_code}",
        _code,
        "ngoloan/WaxalNLP",
        parquet_prefix=f"data/ASR/{_code}/{_code}-test-",
        kind="conversational",
    )


@dataclass
class HeldoutUtterance:
    source: str
    lang: str
    text: str
    wav: Path
    duration_s: float


def _detect_cols(columns: list[str], features) -> tuple[str, str]:
    audio_col = next(
        (c for c in columns if getattr(features[c], "_type", "") == "Audio" or c == "audio"), None
    )
    text_col = next((c for c in TEXT_COLS if c in columns), None)
    if not audio_col or not text_col:
        raise RuntimeError(f"Could not detect audio/text columns in {columns}")
    return audio_col, text_col


def load_source(src: Source, max_items: int = 1500, seed: int = 0) -> list[HeldoutUtterance]:
    import pandas as pd
    from datasets import Audio, Dataset, load_dataset

    tok = hf_token()
    if src.parquet_prefix:
        files = sorted(
            f
            for f in HfApi().list_repo_files(src.repo, repo_type="dataset", token=tok)
            if f.startswith(src.parquet_prefix) and f.endswith(".parquet")
        )
        if not files:
            raise RuntimeError(f"No parquet shards under {src.parquet_prefix} in {src.repo}")
        p = hf_hub_download(src.repo, files[0], repo_type="dataset", token=tok)
        df = pd.read_parquet(p)
        if len(df) > max_items:
            df = df.sample(n=max_items, random_state=seed)
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        log.info("%s: shard %s (%d rows, using %d)", src.name, files[0], len(df), len(ds))
    else:
        ds = load_dataset(src.repo, split=src.split, token=tok)
        if len(ds) > max_items:
            ds = ds.shuffle(seed=seed).select(range(max_items))
    audio_col, text_col = _detect_cols(ds.column_names, ds.features)
    try:
        ds = ds.cast_column(audio_col, Audio(decode=False))
    except Exception:  # already raw bytes struct
        pass
    root = cache_dir()
    out: list[HeldoutUtterance] = []
    seen: set[str] = set()
    for i, r in enumerate(ds):
        text = (r[text_col] or "").strip()
        if not text or text in seen:
            continue
        try:
            y, sr = decode_hf_audio(r[audio_col])
        except Exception as e:
            log.debug("skip row %d: %s", i, e)
            continue
        if len(y) < sr * 0.5:
            continue
        wav = ensure_wav(root, f"heldout/{src.name}", f"{src.split}/{i}/{text[:40]}", y, sr)
        out.append(HeldoutUtterance(src.name, src.lang, text, wav, len(y) / sr))
        seen.add(text)
    log.info(
        "%s: %d unique utterances (%.1f s mean)",
        src.name,
        len(out),
        sum(u.duration_s for u in out) / max(1, len(out)),
    )
    return out
