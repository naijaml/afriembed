"""Synthesised spoken queries with MMS-TTS (VITS), cached as 16 kHz wavs.

Used to turn text benchmarks (AfriQA questions, Injongo intents) into spoken ones when no recordings exist.
Synthetic speech is cleaner than real speech, so numbers from it are an upper bound on the spoken case.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import cache_dir
from .audio_utils import cached_wav_path, ensure_wav

log = logging.getLogger(__name__)

# ISO 639-3 -> MMS-TTS checkpoint. Meta ships no Igbo voice; the WAXAL benchmarking community one fills the gap.
TTS_MODELS = {
    "hau": "facebook/mms-tts-hau",
    "yor": "facebook/mms-tts-yor",
    "swa": "facebook/mms-tts-swh",
    "swh": "facebook/mms-tts-swh",
    "ibo": "waxal-benchmarking/mms-tts-ibo-rnjema101",
    "amh": "facebook/mms-tts-amh",
    "som": "facebook/mms-tts-som",
    "eng": "facebook/mms-tts-eng",
    "lug": "facebook/mms-tts-lug",
    "xho": "facebook/mms-tts-xho",
    "zul": "facebook/mms-tts-zul",
}


class TTS:
    def __init__(self, lang: str, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, VitsModel

        if lang not in TTS_MODELS:
            raise KeyError(f"No MMS-TTS voice registered for {lang}; known: {sorted(TTS_MODELS)}")
        self.lang = lang
        self.repo = TTS_MODELS[lang]
        self.tok = AutoTokenizer.from_pretrained(self.repo)
        self.model = VitsModel.from_pretrained(self.repo)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.sr = self.model.config.sampling_rate

    def __call__(self, text: str) -> tuple:
        import torch

        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = self.model(**inputs).waveform[0].float().cpu().numpy()
        return wav, self.sr

    def to_wav(self, text: str, namespace: str = "tts") -> Path:
        """Synthesise once and cache under data_cache/audio/<namespace>/<sha1(lang/text)>.wav."""
        key = f"{self.lang}/{text}"
        p = cached_wav_path(cache_dir(), namespace, key)
        if p.exists():
            return p
        wav, sr = self(text)
        return ensure_wav(cache_dir(), namespace, key, wav, sr)


def synthesize_all(
    texts: list[str], lang: str, namespace: str = "tts", device: str | None = None
) -> list[Path]:
    tts = TTS(lang, device=device)
    out = []
    for i, t in enumerate(texts):
        out.append(tts.to_wav(t, namespace))
        if i % 100 == 99:
            log.info("%s tts: %d/%d", lang, i + 1, len(texts))
    return out
