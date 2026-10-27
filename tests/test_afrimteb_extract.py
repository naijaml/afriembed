from afriembed.experiments.afrimteb import _per_language


def test_per_language_folds_aliases_and_skips_english():
    flat = {
        "FloresBitextMining": {
            "main_score": 0.4,
            "subsets": {
                "test/eng_Latn-swh_Latn": {"main_score": 0.4},
                "test/orm_Latn-eng_Latn": {"main_score": 0.2},
            },
        },
        "AfriSentiClassification": {
            "main_score": 0.5,
            "subsets": {"test/swa": {"main_score": 0.6}, "test/hau": {"main_score": 0.5}},
        },
    }
    pl = _per_language(flat)
    assert "eng" not in pl and "orm" not in pl and "swa" not in pl
    assert abs(pl["swh"] - 0.5) < 1e-9 and pl["gaz"] == 0.2 and pl["hau"] == 0.5
