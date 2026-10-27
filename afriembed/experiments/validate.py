"""Product 1: speech dataset validator.

Given clips with their claimed transcripts (a read-speech or prompted crowd dataset), embed each clip and
every prompt, and check that each clip's own prompt is its nearest prompt. Clips whose nearest prompt is a
different sentence are mismatches (wrong label, misread, wrong file); clips that win by a thin margin are
borderline. No ASR for the language is needed.

Input is a manifest (csv/tsv/jsonl with `audio` and `text` columns) or one of the held-out sources for a
self-test; `--inject p` shuffles a fraction p of the labels so detection precision/recall can be measured.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

STATUSES = ("pass", "borderline", "fail")


@dataclass
class Thresholds:
    pass_margin: float = 0.02  # own prompt must beat the best other prompt by this much to pass
    borderline_rank: int = 5  # own prompt within the top-k nearest prompts -> borderline rather than fail
    min_similarity: float = 0.30  # below this the clip matches nothing in the manifest -> fail


@dataclass
class Row:
    audio: str
    text: str
    status: str
    own_sim: float
    rank: int
    margin: float
    predicted: str
    injected: bool = False


def read_manifest(path: Path) -> list[tuple[str, str]]:
    """Return (audio_path, text) pairs from csv / tsv / jsonl. Relative audio paths resolve from the manifest's dir."""
    rows: list[tuple[str, str]] = []
    base = path.parent
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append((r["audio"], r["text"]))
    else:
        delim = "\t" if path.suffix == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter=delim):
                rows.append((r["audio"], r["text"]))
    out = []
    for a, t in rows:
        p = Path(a)
        if not p.is_absolute():
            p = base / p
        out.append((str(p), t.strip()))
    return out


def decide(
    sims: np.ndarray, own: np.ndarray, th: Thresholds
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised decision. sims: [n_clips, n_prompts]; own: [n_clips] index of each clip's claimed prompt.

    Returns (status codes 0/1/2 for pass/borderline/fail, own_sim, rank, margin).
    """
    n = sims.shape[0]
    own_sim = sims[np.arange(n), own]
    rank = (sims > own_sim[:, None]).sum(axis=1) + 1
    others = sims.copy()
    others[np.arange(n), own] = -np.inf
    best_other = others.max(axis=1) if sims.shape[1] > 1 else np.full(n, -np.inf)
    margin = own_sim - best_other
    status = np.full(n, 2)  # fail
    status[(rank <= th.borderline_rank)] = 1  # borderline
    status[(rank == 1) & (margin >= th.pass_margin)] = 0  # pass
    status[own_sim < th.min_similarity] = 2
    return status, own_sim, rank, margin


def run(
    opts: ModelOpts,
    manifest: Path | None = None,
    source: str | None = None,
    max_items: int = 1000,
    inject: float = 0.0,
    out_dir: Path | None = None,
    thresholds: Thresholds | None = None,
    seed: int = 0,
    tag: str | None = None,
) -> dict:
    th = thresholds or Thresholds()
    rng = np.random.default_rng(seed)
    if manifest is not None:
        pairs = read_manifest(manifest)
        name = manifest.stem
    elif source is not None:
        from ..data.heldout import SOURCES, load_source

        utts = load_source(SOURCES[source], max_items=max_items, seed=seed)
        pairs = [(str(u.wav), u.text) for u in utts]
        name = source
    else:
        raise SystemExit("Give --manifest or --source")
    if len(pairs) < 2:
        raise SystemExit("Need at least two clips")

    # unique prompts; a clip's claimed prompt is an index into them
    prompts = sorted({t for _, t in pairs})
    pidx = {t: i for i, t in enumerate(prompts)}
    own = np.array([pidx[t] for _, t in pairs])
    injected = np.zeros(len(pairs), dtype=bool)
    if inject > 0:
        k = max(1, int(round(inject * len(pairs))))
        pick = rng.choice(len(pairs), size=k, replace=False)
        # give each picked clip a different prompt than its own
        for i in pick:
            j = own[i]
            while j == own[i]:
                j = int(rng.integers(len(prompts)))
            own[i] = j
            injected[i] = True
        log.info("injected %d wrong labels (%.0f%%)", k, 100 * k / len(pairs))

    emb = opts.load(("text", "audio"))
    with timer(f"{name}: embed {len(pairs)} clips and {len(prompts)} prompts"):
        A = emb.encode_audio([a for a, _ in pairs], "none", show_progress=True)
        T = emb.encode_text(prompts, "document", show_progress=True)
    sims = A @ T.T
    status, own_sim, rank, margin = decide(sims, own, th)
    top1 = sims.argmax(axis=1)

    rows = [
        Row(
            audio=pairs[i][0],
            text=prompts[own[i]],
            status=STATUSES[status[i]],
            own_sim=round(float(own_sim[i]), 4),
            rank=int(rank[i]),
            margin=round(float(margin[i]), 4),
            predicted=prompts[top1[i]] if top1[i] != own[i] else "",
            injected=bool(injected[i]),
        )
        for i in range(len(pairs))
    ]
    counts = {s: int((status == i).sum()) for i, s in enumerate(STATUSES)}
    summary: dict = {
        "name": name,
        "n_clips": len(pairs),
        "n_prompts": len(prompts),
        "thresholds": asdict(th),
        "counts": counts,
        "flag_rate": round(float((status > 0).mean()), 4),
        "model_info": emb.describe(),
    }
    if inject > 0:
        flagged = status > 0
        hard_fail = status == 2
        tp = int((flagged & injected).sum())
        summary["injection"] = {
            "n_injected": int(injected.sum()),
            "recall_flagged": round(float(tp / max(1, injected.sum())), 4),
            "precision_flagged": round(float(tp / max(1, flagged.sum())), 4),
            "recall_fail_only": round(float((hard_fail & injected).sum() / max(1, injected.sum())), 4),
            "precision_fail_only": round(float((hard_fail & injected).sum() / max(1, hard_fail.sum())), 4),
            "false_alarm_rate_clean": round(
                float((flagged & ~injected).sum() / max(1, (~injected).sum())), 4
            ),
        }
        log.info("injection: %s", summary["injection"])

    out_dir = out_dir or Path("reports") / f"validate-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (STATUSES.index(r.status) * -1, r.margin)):
            w.writerow(asdict(r))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    summary["report"] = str(out_dir / "report.csv")
    log.info("%s: %s -> %s", name, counts, out_dir)
    save_result("validate", emb.name, summary, tag=opts.full_tag(tag))
    return summary
