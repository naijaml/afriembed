"""Product 6: zero-ASR alignment of long read audio (audio Bibles, audiobooks) to its text.

Given one long recording and its text as a list of sentences (verses), split the audio at pauses, embed
every pause-delimited segment with the audio tower and every sentence with the text tower, then find the
monotonic assignment of consecutive segments to consecutive sentences that maximises total similarity
(dynamic programming, each sentence gets 1..G consecutive segments). No recogniser for the language is
needed, so it runs for any language the audio tower handles, including ones it was never trained on.

Evaluation builds synthetic chapters from BibleTTS verse clips (concatenated with short silences, so the
true boundaries are known) and reports boundary and verse-assignment accuracy.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import cache_dir
from ..data.audio_utils import TARGET_SR, load_any, to_mono_16k, write_wav
from ..utils import save_result
from .common import ModelOpts

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start_s: float
    end_s: float


def split_on_pauses(
    y: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 30.0,
    min_gap_s: float = 0.3,
    min_len_s: float = 0.8,
) -> list[Segment]:
    """Energy-based split at silences of at least min_gap_s; segments shorter than min_len_s merge into a neighbour.

    Short intra-sentence pauses are bridged so the segment count stays near the sentence count (sentence
    boundaries in read speech carry the longer pauses)."""
    import librosa

    ivals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    segs: list[Segment] = []
    for a, b in ivals:
        s = Segment(a / sr, b / sr)
        if segs and (s.start_s - segs[-1].end_s) < min_gap_s:
            segs[-1] = Segment(segs[-1].start_s, s.end_s)  # bridge a short pause
        else:
            segs.append(s)
    merged: list[Segment] = []
    for s in segs:
        if merged and (s.end_s - s.start_s) < min_len_s:
            merged[-1] = Segment(merged[-1].start_s, s.end_s)
        else:
            merged.append(s)
    if len(merged) > 1 and (merged[0].end_s - merged[0].start_s) < min_len_s:
        merged[1] = Segment(merged[0].start_s, merged[1].end_s)
        merged.pop(0)
    return merged


def monotonic_align(S: np.ndarray, max_group: int = 4) -> list[tuple[int, int]]:
    """Per-segment similarities S[i, j]; group score = mean over the group's segments. Kept for tests/fallback."""
    m, n = S.shape
    csum = np.vstack([np.zeros((1, n)), np.cumsum(S, axis=0)])
    span = np.full((m, max_group + 1, n), -np.inf)
    for i in range(m):  # span ending at segment i with g segments
        for g in range(1, min(max_group, i + 1) + 1):
            span[i, g] = (csum[i + 1] - csum[i + 1 - g]) / g
    return span_align(span)


def span_align(
    span: np.ndarray,
    seg_dur: np.ndarray | None = None,
    expected_dur: np.ndarray | None = None,
    duration_weight: float = 0.15,
) -> list[tuple[int, int]]:
    """span[i, g, j]: score of sentence j covering the g segments ending at segment i (g >= 1).

    Exact monotonic DP: every segment used once, in order; each sentence gets 1..G consecutive segments.
    With seg_dur (per segment) and expected_dur (per sentence, from its share of the text), a span is
    penalised by duration_weight * |log(span_dur / expected_dur)|: read speech is close to proportional to
    text length, and this keeps the alignment from drifting. Proportional split if infeasible."""
    m, G, n = span.shape
    NEG = -1e9
    dp = np.full((m + 1, n + 1), NEG)
    back = np.zeros((m + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0
    cdur = np.concatenate([[0.0], np.cumsum(seg_dur)]) if seg_dur is not None else None
    for j in range(1, n + 1):
        for i in range(j, m + 1):
            best, arg = NEG, 0
            for g in range(1, min(G - 1, i) + 1):
                prev = dp[i - g, j - 1]
                if prev <= NEG / 2 or not np.isfinite(span[i - 1, g, j - 1]):
                    continue
                sc = prev + span[i - 1, g, j - 1]
                if cdur is not None and expected_dur is not None:
                    d = max(cdur[i] - cdur[i - g], 1e-3)
                    sc -= duration_weight * abs(np.log(d / max(expected_dur[j - 1], 1e-3)))
                if sc > best:
                    best, arg = sc, g
            dp[i, j], back[i, j] = best, arg
    if dp[m, n] <= NEG / 2:
        bounds = np.linspace(0, m, n + 1).round().astype(int)
        return [(int(bounds[j]), int(max(bounds[j], bounds[j + 1] - 1))) for j in range(n)]
    spans = []
    i = m
    for j in range(n, 0, -1):
        g = back[i, j]
        spans.append((i - g, i - 1))
        i -= g
    return spans[::-1]


def align(
    opts_or_emb,
    audio: Path,
    sentences: list[str],
    max_group: int | None = None,
    top_db: float = 30.0,
    emb=None,
):
    emb = emb or opts_or_emb.load(("text", "audio"))
    y, sr = load_any(audio)
    y = to_mono_16k(y, sr)
    segs = split_on_pauses(y, TARGET_SR, top_db=top_db)
    if len(segs) < len(sentences):  # not enough pauses: cut long segments into pieces
        segs = _subdivide(segs, len(sentences))
    # a sentence may span several segments; allow enough so that every segment can be used
    max_group = max_group or max(3, int(np.ceil(2.0 * len(segs) / len(sentences))))
    tmp = cache_dir() / "align" / audio.stem
    tmp.mkdir(parents=True, exist_ok=True)
    m = len(segs)
    # every candidate span (g consecutive segments ending at i) is embedded as the actual audio it covers
    paths, keys = [], []
    for i in range(m):
        for g in range(1, min(max_group, i + 1) + 1):
            a = i - g + 1
            p = tmp / f"{a:05d}_{i:05d}.wav"
            if not p.exists():
                write_wav(p, y[int(segs[a].start_s * TARGET_SR) : int(segs[i].end_s * TARGET_SR)])
            paths.append(str(p))
            keys.append((i, g))
    A = emb.encode_audio(paths, "none", show_progress=len(paths) > 100)
    T = emb.encode_text(sentences, "document")
    sims = A @ T.T  # [n_spans, n_sentences]
    span = np.full((m, max_group + 1, len(sentences)), -np.inf, dtype=np.float32)
    for (i, g), row in zip(keys, sims, strict=True):
        span[i, g] = row
    seg_dur = np.array([sg.end_s - sg.start_s for sg in segs])
    chars = np.array([max(1, len(t)) for t in sentences], dtype=float)
    expected_dur = seg_dur.sum() * chars / chars.sum()
    spans = span_align(span, seg_dur=seg_dur, expected_dur=expected_dur)
    out = []
    for j, (a, b) in enumerate(spans):
        conf = float(span[b, b - a + 1, j]) if np.isfinite(span[b, b - a + 1, j]) else float("nan")
        out.append(
            {
                "start_s": segs[a].start_s,
                "end_s": segs[b].end_s,
                "text": sentences[j],
                "confidence": round(conf, 4),
            }
        )
    return out, segs


def _subdivide(segs: list[Segment], target: int) -> list[Segment]:
    """Cut the longest segments in half until there are at least `target` segments."""
    segs = list(segs)
    while len(segs) < target:
        k = int(np.argmax([s.end_s - s.start_s for s in segs]))
        s = segs.pop(k)
        mid = (s.start_s + s.end_s) / 2
        segs[k:k] = [Segment(s.start_s, mid), Segment(mid, s.end_s)]
    return segs


def align_file(
    opts: ModelOpts, audio: Path, text_file: Path, out: Path | None = None, max_group: int = 4
) -> Path:
    sentences = [l.strip() for l in text_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, _ = align(opts, audio, sentences, max_group=max_group)
    out = out or audio.with_suffix(".aligned.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start_s", "end_s", "confidence", "text"])
        w.writeheader()
        w.writerows(rows)
    log.info("aligned %d sentences over %d s -> %s", len(rows), round(rows[-1]["end_s"]), out)
    return out


# ------------------------------------------------------------------ evaluation on BibleTTS


def _bible_rows(lang: str, max_rows: int = 400):
    from ..data.heldout import SOURCES, load_source

    src = {"hau": "bible-hau", "ewe": "bible-ewe"}[lang]
    return load_source(SOURCES[src], max_items=max_rows)


def evaluate(
    opts: ModelOpts,
    lang: str = "hau",
    n_chapters: int = 10,
    verses_per_chapter: int = 20,
    gap_s: float = 0.4,
    max_group: int | None = None,
    tag: str | None = None,
    seed: int = 0,
) -> dict:
    emb = opts.load(("text", "audio"))
    utts = _bible_rows(lang, max_rows=n_chapters * verses_per_chapter + 50)
    rng = np.random.default_rng(seed)
    rng.shuffle(utts)
    tmp = cache_dir() / "align-eval" / lang
    tmp.mkdir(parents=True, exist_ok=True)
    per = []
    for c in range(n_chapters):
        chunk = utts[c * verses_per_chapter : (c + 1) * verses_per_chapter]
        if len(chunk) < verses_per_chapter:
            break
        pieces, truth, t = [], [], 0.0
        gap = np.zeros(int(gap_s * TARGET_SR), dtype=np.float32)
        for u in chunk:
            y, sr = load_any(u.wav)
            y = to_mono_16k(y, sr)
            pieces += [y, gap]
            # truth = the speech inside the clip (clips may carry their own leading/trailing silence)
            nz = np.where(np.abs(y) > 0.01)[0]
            a0 = (nz[0] / TARGET_SR) if len(nz) else 0.0
            a1 = (nz[-1] / TARGET_SR) if len(nz) else len(y) / TARGET_SR
            truth.append((t + a0, t + a1))
            t += (len(y) + len(gap)) / TARGET_SR
        chapter = tmp / f"chapter{c:02d}.wav"
        write_wav(chapter, np.concatenate(pieces))
        rows, segs = align(None, chapter, [u.text for u in chunk], max_group=max_group, emb=emb)
        ious, starts_ok = [], []
        for (ts, te), r in zip(truth, rows, strict=True):
            inter = max(0.0, min(te, r["end_s"]) - max(ts, r["start_s"]))
            union = max(te, r["end_s"]) - min(ts, r["start_s"])
            ious.append(inter / union if union > 0 else 0.0)
            starts_ok.append(abs(r["start_s"] - ts) <= 0.5)
        (tmp / f"chapter{c:02d}.aligned.json").write_text(
            json.dumps({"truth": truth, "pred": rows}, ensure_ascii=False)
        )
        per.append(
            {
                "chapter": c,
                "duration_s": round(t, 1),
                "n_segments": len(segs),
                "verse_iou>=0.5": float(np.mean([i >= 0.5 for i in ious])),
                "mean_iou": float(np.mean(ious)),
                "start_within_0.5s": float(np.mean(starts_ok)),
                "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            }
        )
        log.info(
            "%s chapter %d: %d verses, %d segments, IoU>=0.5 %.2f, start±0.5s %.2f",
            lang,
            c,
            len(chunk),
            len(segs),
            per[-1]["verse_iou>=0.5"],
            per[-1]["start_within_0.5s"],
        )
    payload = {
        "lang": lang,
        "n_chapters": len(per),
        "verses_per_chapter": verses_per_chapter,
        "gap_s": gap_s,
        "max_group": max_group,
        "model_info": emb.describe(),
        "per_chapter": per,
        "mean": {
            k: float(np.mean([p[k] for p in per]))
            for k in ("verse_iou>=0.5", "mean_iou", "start_within_0.5s")
        },
    }
    save_result("align", emb.name, payload, tag=opts.full_tag(f"{lang}-{tag}" if tag else lang))
    (tmp / "summary.json").write_text(json.dumps(payload["mean"], indent=2))
    return payload
