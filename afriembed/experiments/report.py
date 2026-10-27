"""Aggregate results/**/latest.json into Markdown tables (results/REPORT.md)."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import results_dir


def _latest() -> list[dict]:
    out = []
    for p in sorted(results_dir().glob("*/*/latest*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(out)


def build() -> str:
    res = _latest()
    md = ["# afriembed results", ""]
    by_exp: dict[str, list[dict]] = {}
    for r in res:
        by_exp.setdefault(r["experiment"], []).append(r)

    for exp in sorted(by_exp):
        rows = by_exp[exp]
        md.append(f"## {exp}")
        md.append("")
        if exp.startswith("afrimteb"):
            langs = sorted({l for r in rows for l in r.get("per_language", {})})
            md.append(
                _table(
                    ["model", "run", "prompts", "dim", "macro avg", *langs],
                    [
                        [
                            r["model"],
                            r["run_id"],
                            r["model_info"].get("use_prompts"),
                            r["model_info"].get("truncate_dim") or r["model_info"].get("dim"),
                            r.get("macro_average"),
                            *[r.get("per_language", {}).get(l) for l in langs],
                        ]
                        for r in rows
                    ],
                )
            )
            md.append("")
            tasks = sorted({t for r in rows for t in r.get("tasks", {})})
            md.append(
                _table(
                    ["task", *[f"{r['model']}:{r['run_id'][-6:]}" for r in rows]],
                    [[t, *[r.get("tasks", {}).get(t, {}).get("main_score") for r in rows]] for t in tasks],
                )
            )
        elif exp == "speech-text":
            for r in rows:
                dims = r.get("dims", [])
                md.append(f"**{r['model']}** ({r['run_id']}, text_task={r.get('text_task')})")
                md.append("")
                md.append(
                    _table(
                        ["lang", "tier", "n", *[f"a→t R@1 d{d}" for d in dims], "t→a R@1 (full)", "MRR a→t"],
                        [
                            [
                                l,
                                v["tier"],
                                v["n"],
                                *[v["dims"][str(d)]["audio->text"]["recall@1"] for d in dims],
                                v["dims"][str(dims[0])]["text->audio"]["recall@1"],
                                v["dims"][str(dims[0])]["audio->text"]["mrr"],
                            ]
                            for l, v in r["per_language"].items()
                        ],
                    )
                )
                md.append("")
                md.append(
                    _table(
                        ["dim", "mean a→t R@1", "retention"],
                        [
                            [d, s["mean_audio->text_recall@1"], s["retention_vs_full"]]
                            for d, s in r["summary"].items()
                        ],
                    )
                )
                md.append("")
        elif exp == "speech-bitext":
            for r in rows:
                md.append(f"**{r['model']}** ({r['run_id']}, pivot={r.get('pivot')})")
                md.append("")
                md.append(
                    _table(
                        ["pair", "n", "audio→text", "audio→audio", "text→text", "text→audio", "a→t margin"],
                        [
                            [
                                p,
                                v["n"],
                                v["audio->text"]["accuracy"],
                                v["audio->audio"]["accuracy"],
                                v["text->text"]["accuracy"],
                                v["text->audio"]["accuracy"],
                                v["audio->text"].get("margin_accuracy"),
                            ]
                            for p, v in r["pairs"].items()
                        ],
                    )
                )
                md.append("")
        elif exp == "fertility":
            for r in rows:
                toks = r["tokenizers"]
                langs = r["languages"]
                md.append(
                    _table(
                        [
                            "lang",
                            "tier",
                            "script",
                            *[f"{t} tok/word" for t in toks],
                            *[f"{t} ×eng" for t in toks],
                        ],
                        [
                            [
                                l,
                                toks[next(iter(toks))]["languages"][l]["tier"],
                                toks[next(iter(toks))]["languages"][l]["script"],
                                *[toks[t]["languages"][l]["tokens_per_word"] for t in toks],
                                *[toks[t]["languages"][l]["premium_vs_eng"] for t in toks],
                            ]
                            for l in langs
                        ],
                    )
                )
                md.append("")
        elif exp == "ciral":
            langs = sorted({l for r in rows for l in r["per_language"]})
            md.append(
                _table(
                    [
                        "model",
                        "run",
                        "split",
                        *[f"{l} nDCG@20" for l in langs],
                        *[f"{l} R@100" for l in langs],
                    ],
                    [
                        [
                            r["model"],
                            r["run_id"],
                            r["split"],
                            *[r["per_language"].get(l, {}).get("ndcg@20") for l in langs],
                            *[r["per_language"].get(l, {}).get("recall@100") for l in langs],
                        ]
                        for r in rows
                    ],
                )
            )
            md.append("")
        elif exp == "vision":
            for r in rows:
                md.append(f"**{r['model']}** ({r['run_id']}, {r['dataset']})")
                md.append("")
                md.append(
                    _table(
                        [
                            "language",
                            "n",
                            "image→local R@1",
                            "image→english R@1",
                            "local→image R@1",
                            "local↔english text R@1",
                            "gap (en−local cos)",
                        ],
                        [
                            [
                                l,
                                v["n"],
                                v["image->local"]["recall@1"],
                                v["image->english"]["recall@1"],
                                v["local->image"]["recall@1"],
                                v["local<->english_text"]["recall@1"],
                                v["language_gap_cos"],
                            ]
                            for l, v in r["per_language"].items()
                        ],
                    )
                )
                md.append("")
        elif exp == "lid":
            md.append(
                _table(
                    [
                        "model",
                        "run",
                        "langs",
                        "audio kNN",
                        "audio NMI",
                        "audio purity",
                        "text kNN",
                        "text NMI",
                        "chance",
                    ],
                    [
                        [
                            r["model"],
                            r["run_id"],
                            len(r["languages"]),
                            r["audio"]["knn_accuracy"],
                            r["audio"]["nmi"],
                            r["audio"]["purity"],
                            r["text"]["knn_accuracy"],
                            r["text"]["nmi"],
                            r["chance"],
                        ]
                        for r in rows
                    ],
                )
            )
            md.append("")
        elif exp == "accent":
            for r in rows:
                md.append(
                    f"**{r['model']}** ({r['run_id']}) overall R@1={_fmt(r['overall']['recall@1'])} pool={r['pool']}"
                )
                md.append("")
                md.append(
                    _table(
                        ["country", "n", "R@1", "R@10", "MRR"],
                        [
                            [c, v["n"], v["recall@1"], v["recall@10"], v["mrr"]]
                            for c, v in r["by_country"].items()
                        ],
                    )
                )
                md.append("")
                md.append(
                    _table(
                        ["domain", "n", "R@1", "MRR"],
                        [[c, v["n"], v["recall@1"], v["mrr"]] for c, v in r["by_domain"].items()],
                    )
                )
                md.append("")
        elif exp == "vocab-trim":
            md.append(
                _table(
                    ["model", "run", "languages", "kept", "vocab", "MB saved (bf16)", "held-out coverage"],
                    [
                        [
                            r["model"],
                            r["run_id"],
                            ",".join(r["languages"]),
                            r["kept_tokens"],
                            r["vocab_size"],
                            r["embedding_mb_saved_bf16"],
                            r["heldout_token_coverage"],
                        ]
                        for r in rows
                    ],
                )
            )
            md.append("")
        elif exp == "rag":
            md.append(
                _table(
                    ["model", "run", "generator", "k", "n", "answer in ctx@k", "token F1", "EM"],
                    [
                        [
                            r["model"],
                            r["run_id"],
                            r.get("generator"),
                            r["k"],
                            r["n"],
                            r.get("answer_in_context@k"),
                            r.get("token_f1"),
                            r.get("exact_match"),
                        ]
                        for r in rows
                    ],
                )
            )
            md.append("")
        else:
            md.append(
                _table(
                    ["model", "run", "keys"],
                    [[r["model"], r["run_id"], ", ".join(k for k in r if k not in ("env",))] for r in rows],
                )
            )
            md.append("")
    text = "\n".join(md)
    out = results_dir() / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    return text


def main() -> Path:
    build()
    return results_dir() / "REPORT.md"


if __name__ == "__main__":  # pragma: no cover
    print(build())
