import pytest

from afriembed import languages as L
from afriembed.config import model_registry, tokenizer_registry


def test_registry_loads_and_lite_set_matches_mteb():
    langs = L.all_languages()
    lite = L.language_set("lite")
    assert lite == ["amh", "gaz", "ibo", "yor", "hau", "swh", "kin", "xho", "zul"]
    assert all(langs[c].lite for c in lite)


def test_every_language_has_sib_and_consistent_fleurs():
    sets = L._registry()["sets"]
    in_sets = {c for members in sets.values() for c in members}
    for code, l in L.all_languages().items():
        usable = bool(l.sib or l.fleurs or l.injongo or l.africaption)
        if not usable:  # registry-only entries (e.g. Nigerian Pidgin) must not be in any experiment set
            assert code not in in_sets, code
        if l.fleurs:
            assert "_" in l.fleurs, (code, l.fleurs)


def test_resolve_specs():
    assert L.resolve(None, "ciral") == ["hau", "som", "swh", "yor"]
    assert L.resolve("hau, yor", "lite") == ["hau", "yor"]
    with pytest.raises(KeyError):
        L.resolve("xxx", "lite")


def test_with_fleurs_skips_missing(recwarn):
    out = L.with_fleurs(["hau", "kin"])
    assert [l.code for l in out] == ["hau"]
    assert any("kin" in str(w.message) for w in recwarn)


def test_model_registry_shape():
    reg = model_registry()
    assert "embeddinggemma-2" in reg and reg["embeddinggemma-2"]["family"] == "embeddinggemma2"
    assert set(reg["embeddinggemma-2"]["modalities"]) == {"text", "image", "audio", "video"}
    assert len(tokenizer_registry()) >= 3


def test_every_registered_model_is_pinned_to_a_revision():
    for name, spec in model_registry().items():
        rev = spec.get("revision")
        assert isinstance(rev, str) and len(rev) == 40, f"{name} must pin a full 40-char commit hash"
