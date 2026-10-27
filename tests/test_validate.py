import csv
import json

import numpy as np

from afriembed.experiments.validate import Thresholds, decide, read_manifest


def test_decide_pass_borderline_fail():
    # 4 clips, 4 prompts; clip i claims prompt i
    sims = np.array(
        [
            [0.90, 0.50, 0.40, 0.30],  # clear pass
            [0.70, 0.71, 0.40, 0.30],  # own is rank 2 -> borderline
            [0.20, 0.10, 0.60, 0.65],  # own (0.60) rank 2 but fine; margin negative -> borderline
            [0.80, 0.70, 0.60, 0.20],  # own 0.20 < min_similarity -> fail
        ]
    )
    own = np.array([0, 0, 2, 3])
    status, own_sim, rank, margin = decide(
        sims, own, Thresholds(pass_margin=0.02, borderline_rank=3, min_similarity=0.3)
    )
    assert status.tolist() == [0, 1, 1, 2]
    assert rank.tolist() == [1, 2, 2, 4]
    assert np.isclose(margin[0], 0.40)


def test_decide_thin_margin_is_borderline():
    sims = np.array([[0.60, 0.59]])
    status, *_ = decide(sims, np.array([0]), Thresholds(pass_margin=0.02))
    assert status.tolist() == [1]


def test_decide_rank_beyond_k_is_fail():
    sims = np.array([[0.5, 0.9, 0.8, 0.7, 0.6, 0.55]])
    status, _, rank, _ = decide(sims, np.array([0]), Thresholds(borderline_rank=3))
    assert rank.tolist() == [6]
    assert status.tolist() == [2]


def test_read_manifest_csv_and_jsonl(tmp_path):
    (tmp_path / "clips").mkdir()
    p = tmp_path / "m.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "text"])
        w.writerow(["clips/a.wav", " ina kwana "])
    rows = read_manifest(p)
    assert rows == [(str(tmp_path / "clips" / "a.wav"), "ina kwana")]
    j = tmp_path / "m.jsonl"
    j.write_text(json.dumps({"audio": "/abs/b.wav", "text": "sannu"}) + "\n")
    assert read_manifest(j) == [("/abs/b.wav", "sannu")]
