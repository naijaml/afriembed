import numpy as np
import pytest

from afriembed.models.embedder import (
    TASKS,
    Embedder,
    guard_text,
    l2_normalize,
    looks_like_media,
    mteb_prompt_catalog,
    truncate,
)


class _FakeST:
    """Mimics the bits of SentenceTransformer the wrapper touches."""

    def __init__(self, prompts=None, dim=8):
        self.prompts = prompts if prompts is not None else {}
        self.dim = dim
        self.calls = []

    def encode(self, items, **kw):
        self.calls.append((items, kw))
        rng = np.random.default_rng(len(items))
        x = rng.normal(size=(len(items), self.dim)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


GEMMA_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "STS": "task: sentence similarity | query: ",
    "BitextMining": "task: search result | query: ",
}


def test_gemma_prompt_names_map_to_catalog():
    e = Embedder(
        "g2",
        "google/embeddinggemma-2",
        "embeddinggemma2",
        _FakeST(GEMMA_PROMPTS),
        modalities=("text", "audio"),
        dim=8,
    )
    assert e.prompt_kwargs("query") == {"prompt_name": "query"}
    assert e.prompt_kwargs("classification") == {"prompt_name": "Classification"}
    assert e.prompt_kwargs("bitext") == {"prompt_name": "BitextMining"}
    assert e.prompt_kwargs("none") == {"prompt": ""}
    assert e.prompt_string("document") == "title: none | text: "


def test_gemma_falls_back_to_literal_prefix_when_catalog_missing():
    e = Embedder("g2", "x", "embeddinggemma2", _FakeST({}), dim=8)
    assert e.prompt_kwargs("query") == {"prompt": "task: search result | query: "}


def test_no_prompts_ablation_sends_empty_prompt():
    e = Embedder("g2", "x", "embeddinggemma2", _FakeST(GEMMA_PROMPTS), dim=8, use_prompts=False)
    assert e.prompt_kwargs("query") == {"prompt": ""}
    e5 = Embedder("e5", "x", "e5", _FakeST(), dim=8, use_prompts=False)
    assert e5.prompt_kwargs("query") == {}


def test_e5_families_use_literal_prefixes():
    e = Embedder("me5", "x", "e5", _FakeST(), dim=8)
    assert e.prompt_kwargs("query") == {"prompt": "query: "}
    assert e.prompt_kwargs("document") == {"prompt": "passage: "}
    i = Embedder("me5i", "x", "e5-instruct", _FakeST(), dim=8)
    assert i.prompt_kwargs("document") == {"prompt": ""}
    assert i.prompt_kwargs("query")["prompt"].startswith("Instruct:")
    for t in TASKS:
        i.prompt_kwargs(t)  # all tasks resolvable


def test_unknown_task_rejected():
    e = Embedder("g2", "x", "embeddinggemma2", _FakeST(GEMMA_PROMPTS), dim=8)
    with pytest.raises(ValueError):
        e.prompt_kwargs("passage")


def test_encode_truncates_and_renormalises():
    st = _FakeST(GEMMA_PROMPTS, dim=8)
    e = Embedder("g2", "x", "embeddinggemma2", st, dim=8, truncate_dim=4)
    out = e.encode_text(["a", "b", "c"], "query")
    assert out.shape == (3, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert st.calls[-1][1]["prompt_name"] == "query"


def test_media_requires_supported_modality():
    e = Embedder("g2", "x", "embeddinggemma2", _FakeST(GEMMA_PROMPTS), modalities=("text",), dim=8)
    with pytest.raises(RuntimeError):
        e.encode_audio(["x.wav"])
    ok = Embedder("g2", "x", "embeddinggemma2", _FakeST(GEMMA_PROMPTS), modalities=("text", "audio"), dim=8)
    out = ok.encode_audio(
        ["x.wav"], task="classification"
    )  # non query/document tasks drop to no prefix for media
    assert out.shape == (1, 8)
    assert ok.model.calls[-1][1]["prompt"] == ""


def test_truncate_helpers():
    x = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(l2_normalize(x), [[0.6, 0.8, 0, 0]])
    assert np.allclose(truncate(x, 1), [[1.0]])
    assert truncate(np.zeros((0, 4), dtype=np.float32), 2).shape == (0, 2)


def test_mteb_prompt_catalog():
    assert mteb_prompt_catalog("embeddinggemma2", True) is None  # ships its own
    assert mteb_prompt_catalog("e5", False) is None
    assert mteb_prompt_catalog("plain", True) is None
    e5 = mteb_prompt_catalog("e5", True)
    assert (
        e5["Retrieval-query"] == "query: "
        and e5["Retrieval-document"] == "passage: "
        and e5["Classification"] == "query: "
    )
    inst = mteb_prompt_catalog("e5-instruct", True)
    assert "Retrieval-document" not in inst and inst["query"].startswith("Instruct:")


def test_media_looking_strings_are_forced_to_text():
    assert looks_like_media("https://www.bbc.com/hausa/video-123")
    assert looks_like_media("www.vdc.gov.ng")
    assert looks_like_media("clip.mp4")
    assert looks_like_media("labarai/2024/hira.MP4?x=1")
    assert not looks_like_media("Ana shuka masara a farkon damina.")
    assert not looks_like_media("Kwatancin 802.11n na aiki akan mita 2.4Ghz da 5.0Ghz.")
    out = guard_text(["Sannu da zuwa.", "www.vdc.gov.ng", {"audio": "a.wav"}])
    assert (
        out[0] == "Sannu da zuwa." and out[1] == {"text": "www.vdc.gov.ng"} and out[2] == {"audio": "a.wav"}
    )
