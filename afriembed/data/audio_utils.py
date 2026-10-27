"""Audio helpers: decode anything to 16 kHz mono wav, window long files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


def to_mono_16k(array: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = np.asarray(array, dtype=np.float32)
    if y.ndim == 2:  # (channels, n) or (n, channels)
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y


def write_wav(path: Path, y: np.ndarray, sr: int = TARGET_SR) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


def load_any(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode wav/flac/mp3/ogg/m4a via soundfile, falling back to librosa (ffmpeg/audioread)."""
    import soundfile as sf

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y, sr
    except Exception:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def decode_hf_audio(value) -> tuple[np.ndarray, int]:
    """Decode a `datasets` Audio value without torchcodec.

    Accepts the decode=False form {"bytes": ..., "path": ...} (preferred), an already-decoded
    {"array": ..., "sampling_rate": ...}, or a plain path.
    """
    import io

    import soundfile as sf

    if isinstance(value, dict):
        if value.get("array") is not None:
            return np.asarray(value["array"], dtype=np.float32), int(value["sampling_rate"])
        if value.get("bytes"):
            y, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            return y, int(sr)
        if value.get("path"):
            return load_any(value["path"])
        raise ValueError(f"Unrecognised audio value keys: {sorted(value)}")
    return load_any(value)


def cached_wav_path(cache_root: Path, namespace: str, key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_root / "audio" / namespace / f"{h}.wav"


def ensure_wav(cache_root: Path, namespace: str, key: str, array: np.ndarray, sr: int) -> Path:
    """Write (array, sr) as 16 kHz mono wav under the cache if not already present; return path."""
    p = cached_wav_path(cache_root, namespace, key)
    if not p.exists():
        write_wav(p, to_mono_16k(array, sr))
    return p


@dataclass(frozen=True)
class Window:
    index: int
    start_s: float
    end_s: float


def windows(
    n_samples: int,
    sr: int = TARGET_SR,
    window_s: float = 10.0,
    hop_s: float | None = None,
    min_s: float = 1.0,
) -> list[Window]:
    """Fixed windows over a signal. Default: 10 s windows (250 audio tokens each), no overlap."""
    hop_s = hop_s or window_s
    total = n_samples / sr
    out, i, t = [], 0, 0.0
    while t < total:
        end = min(t + window_s, total)
        if end - t >= min_s or not out:
            out.append(Window(i, round(t, 3), round(end, 3)))
            i += 1
        t += hop_s
    return out


def slice_window(y: np.ndarray, w: Window, sr: int = TARGET_SR) -> np.ndarray:
    return y[int(w.start_s * sr) : int(w.end_s * sr)]
