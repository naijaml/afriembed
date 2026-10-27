"""AfriSpeech-200: African-accented English speech (clinical + general), 120 accents, 16 kHz parquet mirror."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ..config import cache_dir, hf_token
from .audio_utils import decode_hf_audio, ensure_wav

log = logging.getLogger(__name__)

HF_ID = "Professor/afrispeech-200-16khz"


@dataclass
class AccentedUtterance:
    uid: str
    accent: str
    country: str
    domain: str
    text: str
    wav: Path
    duration_s: float


def list_accents(split: str = "test") -> list[str]:
    files = HfApi().list_repo_files(HF_ID, repo_type="dataset", token=hf_token())
    return sorted(
        f.split(f"{split}-", 1)[1].removesuffix(".parquet")
        for f in files
        if f.startswith(f"data/{split}-") and f.endswith(".parquet")
    )


def load_afrispeech(
    accents: list[str] | None = None, split: str = "test", max_per_accent: int | None = None, seed: int = 0
) -> list[AccentedUtterance]:
    import pandas as pd
    from datasets import Dataset

    accents = accents or list_accents(split)
    root = cache_dir()
    out = []
    for acc in accents:
        p = hf_hub_download(HF_ID, f"data/{split}-{acc}.parquet", repo_type="dataset", token=hf_token())
        df = pd.read_parquet(p)
        if max_per_accent and len(df) > max_per_accent:
            df = df.sample(n=max_per_accent, random_state=seed)
        ds = Dataset.from_pandas(df.reset_index(drop=True))
        from datasets import Audio

        ds = ds.cast_column("audio", Audio(decode=False))
        for r in ds:
            y, sr = decode_hf_audio(r["audio"])
            wav = ensure_wav(root, "afrispeech", f"{split}/{acc}/{r['audio_id']}", y, sr)
            out.append(
                AccentedUtterance(
                    r["audio_id"],
                    acc,
                    str(r.get("country")),
                    str(r.get("domain")),
                    r["transcript"],
                    wav,
                    float(r.get("duration") or 0),
                )
            )
    log.info("AfriSpeech-200 %s: %d utterances across %d accents", split, len(out), len(accents))
    return out
