import numpy as np

from afriembed.experiments.align import Segment, _subdivide, monotonic_align


def test_monotonic_align_recovers_block_structure():
    # 6 segments, 3 sentences; segments (0,1)->s0, (2,3,4)->s1, (5)->s2
    S = np.full((6, 3), 0.1)
    for i in (0, 1):
        S[i, 0] = 0.9
    for i in (2, 3, 4):
        S[i, 1] = 0.9
    S[5, 2] = 0.9
    assert monotonic_align(S, max_group=4) == [(0, 1), (2, 4), (5, 5)]


def test_monotonic_align_uses_every_segment_in_order():
    rng = np.random.default_rng(0)
    S = rng.random((9, 4))
    spans = monotonic_align(S, max_group=4)
    assert spans[0][0] == 0 and spans[-1][1] == 8
    for (a, b), (c, _d) in zip(spans, spans[1:], strict=False):
        assert a <= b and c == b + 1


def test_subdivide_until_enough_segments():
    segs = [Segment(0.0, 10.0), Segment(10.0, 12.0)]
    out = _subdivide(segs, 4)
    assert len(out) == 4
    assert out[0].start_s == 0.0 and out[-1].end_s == 12.0
