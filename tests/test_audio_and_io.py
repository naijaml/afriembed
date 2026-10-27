import json

import numpy as np

from afriembed import utils
from afriembed.data import audio_utils as A
from afriembed.experiments.rag import exact_match, token_f1
from afriembed.experiments.search import Index, _chunk_text


def test_windows_cover_signal():
    ws = A.windows(16_000 * 25, 16_000, window_s=10.0)
    assert [(w.start_s, w.end_s) for w in ws] == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    ws = A.windows(16_000 * 20 + 8000, 16_000, window_s=10.0)  # trailing 0.5 s below min_s is dropped
    assert len(ws) == 2
    ws = A.windows(16_000 * 15, 16_000, window_s=10.0, hop_s=5.0)
    assert [(w.start_s, w.end_s) for w in ws] == [(0.0, 10.0), (5.0, 15.0), (10.0, 15.0)]


def test_mono_and_resample():
    y = np.zeros((2, 32_000), dtype=np.float32)
    out = A.to_mono_16k(y, 32_000)
    assert out.ndim == 1 and abs(len(out) - 16_000) <= 1


def test_wav_roundtrip(tmp_path):
    y = (0.1 * np.sin(np.linspace(0, 100, 16_000))).astype(np.float32)
    p = A.ensure_wav(tmp_path, "t", "k1", y, 16_000)
    assert p.exists()
    y2, sr = A.load_any(p)
    assert sr == 16_000 and np.allclose(y, y2, atol=1e-3)
    assert A.ensure_wav(tmp_path, "t", "k1", y, 16_000) == p  # cached


def test_chunk_text():
    text = "para one\n\npara two\n\n" + "x" * 2500
    chunks = _chunk_text(text, max_chars=1000)
    assert chunks[0] == "para one\n\npara two"
    assert all(len(c) <= 1000 for c in chunks) and sum(len(c) for c in chunks) >= 2500


def test_index_roundtrip(tmp_path):
    idx = Index(
        "docs",
        "m",
        4,
        np.eye(3, 4, dtype=np.float32),
        [{"file": "a", "text": "é"}, {"file": "b"}, {"file": "c"}],
    )
    idx.save(tmp_path / "ix")
    back = Index.load(tmp_path / "ix")
    assert back.kind == "docs" and back.emb.shape == (3, 4) and back.meta[0]["text"] == "é"


def test_qa_metrics():
    assert token_f1("babban birnin Abuja ne", ["Abuja ne babban birnin"]) == 1.0
    assert abs(token_f1("Abuja ne babban birnin", ["babban birnin Abuja"]) - 6 / 7) < 1e-9
    assert exact_match("Abuja.", ["abuja"]) == 1.0
    assert token_f1("nothing", ["abuja"]) == 0.0


def test_save_and_load_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AFRIEMBED_RESULTS", str(tmp_path))
    p = utils.save_result("unit", "model/x", {"score": np.float32(0.5), "arr": np.arange(2)}, tag="t")
    data = json.loads(p.read_text())
    assert data["score"] == 0.5 and data["arr"] == [0, 1] and data["experiment"] == "unit"
    assert (p.parent / "latest-t.json").exists()  # one latest pointer per tag/variant
    assert utils.load_results("unit")[0]["model"] == "model/x"
