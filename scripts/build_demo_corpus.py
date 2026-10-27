"""Populate demo/corpus/news with real BBC articles from MasakhaNEWS (Adelani et al., 2023).

Usage: uv run python scripts/build_demo_corpus.py [--per-lang 60] [--langs hau,ibo,yor,pcm,swa,eng]

Each article becomes one markdown file: headline as title, category and source URL in a metadata line,
then the article body (trimmed to --max-chars so each file is a single index chunk).
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "corpus" / "news"
NAMES = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "yor": "Yoruba",
    "pcm": "Nigerian Pidgin",
    "swa": "Swahili",
    "eng": "English",
    "amh": "Amharic",
    "fra": "French",
    "lin": "Lingala",
    "lug": "Luganda",
    "orm": "Oromo",
    "run": "Rundi",
    "sna": "Shona",
    "som": "Somali",
    "tir": "Tigrinya",
    "xho": "Xhosa",
}


def slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    return re.sub(r"[\s_-]+", "-", s)[:n].strip("-") or "article"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hau,ibo,yor,pcm,swa,eng")
    ap.add_argument("--per-lang", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    total = 0
    for lang in a.langs.split(","):
        ds = load_dataset("masakhane/masakhanews", lang, split="train")
        rows = [r for r in ds if len(r["text"]) > 300]
        rng.shuffle(rows)
        # spread across categories so the corpus is not all politics
        by_cat: dict[str, list] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        picked: list = []
        while len(picked) < a.per_lang and any(by_cat.values()):
            for c in sorted(by_cat):
                if by_cat[c] and len(picked) < a.per_lang:
                    picked.append(by_cat[c].pop())
        d = OUT / lang
        d.mkdir(parents=True, exist_ok=True)
        for r in picked:
            body = r["text"].strip()
            if len(body) > a.max_chars:
                body = body[: a.max_chars].rsplit(" ", 1)[0] + " …"
            md = (
                f"# {r['headline'].strip()}\n\n"
                f"*{NAMES.get(lang, lang)} · {r['category']} · {r['url']}*\n\n"
                f"{body}\n"
            )
            (d / f"{slug(r['headline'])}.md").write_text(md, encoding="utf-8")
        total += len(picked)
        print(f"{lang}: {len(picked)} articles -> {d}")
    print(f"total {total} articles under {OUT}")


if __name__ == "__main__":
    main()
