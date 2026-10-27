"""Build the paper figures from results/*.json (static PNG, light surface, reference palette slots 1-3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def latest(exp: str, model: str, pick=None) -> dict:
    files = sorted((R / exp / model).glob("2026*.json"))
    runs = [json.loads(p.read_text()) for p in files]
    if pick:
        runs = [r for r in runs if pick(r)]
    if not runs:
        raise SystemExit(f"no results for {exp}/{model}")
    return runs[-1]


def lite_runs(model: str):
    """All complete, prompts-on AfriMTEB-Lite runs for a model, from timestamped files (latest pointers can collide)."""
    out = []
    for p in sorted((R / "afrimteb-lite" / model).glob("2026*.json")):
        d = json.loads(p.read_text())
        mi = d["model_info"]
        if d["n_tasks"] < 13 or not mi["use_prompts"]:
            continue
        # diagnostic ablations (causal mask, precision, short context) are not leaderboard runs
        if any(t in d["run_id"] for t in ("-causal", "-fp16", "-fp32", "-seq")):
            continue
        out.append((p, d))
    return out


def fig_leaderboard():
    rows = {}
    for model_dir in sorted((R / "afrimteb-lite").iterdir()):
        if not model_dir.is_dir():
            continue
        for _p, d in lite_runs(model_dir.name):
            mi = d["model_info"]
            if mi.get("adapter") or mi.get("truncate_dim") or mi.get("trim"):
                continue
            rows[d["model"]] = d["macro_average"] * 100  # later runs (full context) overwrite earlier
    rows = list(rows.items())
    rows = sorted(set(rows), key=lambda r: r[1])
    names = {
        "embeddinggemma-2": "EmbeddingGemma 2 (271M)",
        "embeddinggemma-300m": "EmbeddingGemma v1 (308M)",
        "me5-small": "mE5-small (118M)",
        "me5-base": "mE5-base (278M)",
        "bge-m3": "bge-m3 (568M)",
        "me5-large-instruct": "mE5-large-instruct (560M)",
        "afrie5-large-instruct": "AfriE5-large-instruct (560M)",
    }
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = range(len(rows))
    ax.barh(list(y), [r[1] for r in rows], color=BLUE, height=0.62)
    ax.set_yticks(list(y), [names.get(r[0], r[0]) for r in rows])
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", color=INK2, fontsize=9)
    ax.axvline(29.1, color=ORANGE, lw=1.5, ls="--")
    ax.text(29.6, -0.75, "paper's EmbeddingGemma row (29.1)", color=ORANGE, fontsize=8.5, va="bottom")
    ax.set_xlim(0, 75)
    ax.set_xlabel("AfriMTEB-Lite macro average (13 tasks, 9 languages)")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "EmbeddingGemma trails size-matched mE5 by ~12 points",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_afrimteb_lite.png", dpi=200)


def fig_speech():
    base = latest(
        "speech-text",
        "embeddinggemma-2",
        lambda r: (
            r["text_task"] == "document"
            and not r["model_info"].get("adapter")
            and len(r["per_language"]) >= 14
        ),
    )
    lora = latest(
        "speech-text", "embeddinggemma-2", lambda r: "audio-lora" in str(r["model_info"].get("adapter"))
    )
    langs = sorted(
        lora["per_language"], key=lambda l: base["per_language"][l]["dims"]["768"]["audio->text"]["recall@1"]
    )
    b = [base["per_language"][l]["dims"]["768"]["audio->text"]["recall@1"] for l in langs]
    a = [lora["per_language"][l]["dims"]["768"]["audio->text"]["recall@1"] for l in langs]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = list(range(len(langs)))
    for i in y:
        ax.plot([b[i], a[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(b, y, color=BLUE, s=42, zorder=2, label="zero-shot")
    ax.scatter(a, y, color=ORANGE, s=42, zorder=3, label="+ audio LoRA (5M params, 1 h)")
    ax.set_yticks(y, [f"{l}*" if l == "afr" else l for l in langs])
    ax.set_xlim(0.7, 1.02)
    ax.set_xlabel("speech → transcript Recall@1 (FLEURS test, pool ≈ 300 sentences)\n"
                  "* afr: test sentences appear in the training split; excluded from transfer claims")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", frameon=False, fontsize=9)
    ax.set_title(
        "Within-language speech retrieval: 0.94 zero-shot → 0.997 with a small adapter",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_speech_text.png", dpi=200)


def fig_matryoshka():
    dims = [768, 512, 256, 128]
    text = {}
    for _p, d in lite_runs("embeddinggemma-2"):
        mi = d["model_info"]
        if mi.get("adapter") or mi.get("trim"):
            continue
        text[mi.get("truncate_dim") or 768] = d["macro_average"]
    base = latest(
        "speech-text",
        "embeddinggemma-2",
        lambda r: (
            r["text_task"] == "document"
            and not r["model_info"].get("adapter")
            and len(r["per_language"]) >= 14
        ),
    )
    lora = latest(
        "speech-text", "embeddinggemma-2", lambda r: "audio-lora" in str(r["model_info"].get("adapter"))
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    series = [
        ("AfriMTEB-Lite text (macro)", BLUE, [text[d] / text[768] for d in dims]),
        (
            "FLEURS speech→text, zero-shot",
            ORANGE,
            [base["summary"][str(d)]["retention_vs_full"] for d in dims],
        ),
        (
            "FLEURS speech→text, + audio LoRA",
            AQUA,
            [lora["summary"][str(d)]["retention_vs_full"] for d in dims],
        ),
    ]
    for name, c, ys in series:
        ax.plot(dims, ys, color=c, lw=2, marker="o", ms=5, label=name)
        ax.text(128 * 0.93, ys[-1], f"{ys[-1]:.2f}", color=c, ha="left", va="center", fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_xlim(900, 95)
    ax.set_ylim(0.7, 1.03)
    ax.set_xlabel("embedding dimension (Matryoshka truncation)")
    ax.set_ylabel("retention vs 768d")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title(
        "128-dim speech embeddings: near-lossless with the adapter",
        fontsize=10.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_matryoshka.png", dpi=200)


def fig_crosslingual():
    runs = {}
    for p in (R / "speech-bitext" / "embeddinggemma-2").glob("2026*.json"):
        d = json.loads(p.read_text())
        if d["all_pairs"]:
            continue
        a = str(d["model_info"].get("adapter") or "")
        key = (
            "stacked"
            if "," in a
            else "audio LoRA"
            if "audio" in a
            else "text LoRA"
            if "text" in a
            else "zero-shot"
        )
        runs[key] = d
    order = [k for k in ("zero-shot", "text LoRA", "audio LoRA", "stacked") if k in runs]
    dirs = [
        ("audio->text", "African speech\n→ English text"),
        ("audio->audio", "African speech\n→ English speech"),
        ("text->text", "African text\n→ English text"),
    ]
    colors = {"zero-shot": BLUE, "text LoRA": ORANGE, "audio LoRA": AQUA, "stacked": "#eda100"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(order)
    for j, k in enumerate(order):
        xs = [i + (j - (len(order) - 1) / 2) * w for i in range(len(dirs))]
        ys = [runs[k]["mean_accuracy"][d] for d, _ in dirs]
        ax.bar(xs, ys, width=w - 0.03, color=colors[k], label=k)
        for x, v in zip(xs, ys, strict=True):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(dirs)), [d[1] for d in dirs])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("bitext accuracy (13 languages, FLEURS)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(order), loc="upper left")
    ax.set_title(
        "Cross-lingual alignment is the gap; adapters help and stack (+9 points)", fontsize=10.5, loc="left"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_crosslingual.png", dpi=200)


def fig_heldout():
    runs = {}
    for p in sorted((R / "heldout" / "embeddinggemma-2").glob("2026*.json")):
        d = json.loads(p.read_text())
        key = "adapter" if d["model_info"].get("adapter") else "base"
        runs.setdefault(key, {}).update({k: v for k, v in d["per_source"].items() if "dims" in v})
    names = {
        "bible-hau": "Bible verses · Hausa",
        "bible-ewe": "Bible verses · Ewe (unseen language)",
        "naija-hau": "NaijaVoices · Hausa",
        "naija-ibo": "NaijaVoices · Igbo",
        "naija-yor": "NaijaVoices · Yoruba",
        "waxal-lug": "WAXAL · Luganda",
        "waxal-orm": "WAXAL · Oromo",
        "waxal-sna": "WAXAL · Shona",
        "waxal-lin": "WAXAL · Lingala",
        "waxal-ewe": "WAXAL · Ewe",
    }
    srcs = [k for k in names if k in runs.get("base", {}) and k in runs.get("adapter", {})]
    if not srcs:
        raise SystemExit("no held-out results")
    b = [runs["base"][k]["dims"]["768"]["audio->text"]["recall@1"] for k in srcs]
    a = [runs["adapter"][k]["dims"]["768"]["audio->text"]["recall@1"] for k in srcs]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    y = list(range(len(srcs)))
    for i in y:
        ax.plot([b[i], a[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(b, y, color=BLUE, s=42, zorder=2, label="zero-shot")
    ax.scatter(a, y, color=ORANGE, s=42, zorder=3, label="+ audio LoRA (trained on FLEURS only)")
    for i in y:
        ax.text(b[i] - 0.012, i, f"{b[i]:.3f}", ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(a[i] + 0.012, i, f"{a[i]:.3f}", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"{names[k]} (n={runs['base'][k]['n']})" for k in srcs])
    ax.set_xlim(0.4, 1.1)
    ax.set_xlabel("speech → transcript Recall@1, sources outside FLEURS")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", frameon=False, fontsize=8.5)
    ax.set_title(
        "Held-out sources: the result survives and the adapter transfers",
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig5_heldout.png", dpi=200)


def fig_largepool():
    runs = {}
    for p in sorted((R / "largepool" / "embeddinggemma-2").glob("2026*.json")):
        d = json.loads(p.read_text())
        if d.get("partial"):
            continue
        runs["adapter" if d["model_info"].get("adapter") else "base"] = d
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
    names = {"hau": "Hausa", "yor": "Yoruba", "swh": "Swahili"}
    for ax, lang in zip(axes, ("hau", "yor", "swh"), strict=True):
        for key, c, label in (("base", BLUE, "zero-shot"), ("adapter", ORANGE, "+ audio LoRA")):
            lv = [e for e in runs[key]["per_language"][lang]["levels"] if e["dim"] == 768]
            xs = [e["pool_size"] for e in lv]
            ys = [e["recall@1"] for e in lv]
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=5, label=label)
            ax.text(xs[-1], ys[-1] - 0.045, f"{ys[-1]:.2f}", color=c, ha="right", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.03)
        ax.set_title(names[lang], fontsize=10, loc="left")
        ax.set_xlabel("candidate sentences")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("speech → transcript Recall@1")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(
        "Retrieval holds at a quarter-million candidates with the adapter", x=0.01, ha="left", fontsize=10.5
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig6_largepool.png", dpi=200)


if __name__ == "__main__":
    for f in (fig_leaderboard, fig_speech, fig_matryoshka, fig_crosslingual, fig_heldout, fig_largepool):
        try:
            f()
            print("ok", f.__name__)
        except SystemExit as e:
            print("skip", f.__name__, e, file=sys.stderr)
