"""Self-check for the African speech-retrieval adapter: does spoken audio retrieve its own transcript?

    uv run python scripts/try_adapter.py --lang hau --n 40                  # FLEURS clips for one language
    uv run python scripts/try_adapter.py --folder my_clips                   # your own: name.wav + name.txt pairs

Prints Recall@1 with and without the adapter, plus one worked example with its top-3 candidates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ADAPTER = "naijaml/embeddinggemma-2-african-speech-lora"
BASE = "google/embeddinggemma-2"
REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def load_pairs(args) -> list[tuple[str, str]]:
    if args.folder:
        pairs = []
        for wav in sorted(Path(args.folder).glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                pairs.append((str(wav), txt.read_text(encoding="utf-8").strip()))
        return pairs[: args.n]
    from afriembed.data.fleurs import load_fleurs
    from afriembed.languages import get

    utts = load_fleurs(get(args.lang), split="test", max_items=args.n)
    return [(str(u.wav), u.text) for u in utts]


def score(model, pairs, label):
    from sentence_transformers import SentenceTransformer  # noqa: F401

    A = model.encode([{"audio": w} for w, _ in pairs], batch_size=4, normalize_embeddings=True)
    T = model.encode([t for _, t in pairs], prompt_name="Document", batch_size=16, normalize_embeddings=True)
    sim = A @ T.T
    ranks = (sim > sim[np.arange(len(pairs)), np.arange(len(pairs))][:, None]).sum(1) + 1
    r1 = float((ranks == 1).mean())
    print(f"{label:>18}: Recall@1 = {r1:.3f}  (pool {len(pairs)}, MRR {float((1 / ranks).mean()):.3f})")
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hau")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    pairs = load_pairs(args)
    if len(pairs) < 5:
        raise SystemExit("need at least 5 (audio, transcript) pairs")
    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    kw = dict(
        revision=REV,
        device=args.device,
        config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": dtype},
    )
    print(f"{len(pairs)} clips, device={args.device}, dtype={dtype}")
    base = SentenceTransformer(BASE, **kw)
    sim_base = score(base, pairs, "base model")
    base.load_adapter(ADAPTER)
    sim_ad = score(base, pairs, "+ adapter")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pairs))
    shuffled = float((sim_ad[:, perm].argmax(1) == np.arange(len(pairs))).mean())
    print(f"{'shuffled control':>18}: Recall@1 = {shuffled:.3f}  (expected ≈ {1 / len(pairs):.3f})")
    i = int(np.argmin(np.diag(sim_base)))  # the clip the base model found hardest
    print(f"\nhardest clip for the base model: {Path(pairs[i][0]).name}\n  truth: {pairs[i][1][:90]}")
    for label, sim in (("base", sim_base), ("adapter", sim_ad)):
        top = np.argsort(-sim[i])[:3]
        print(f"  {label} top-3:", " | ".join(f"#{j}{'*' if j == i else ''} {sim[i, j]:.2f}" for j in top))


if __name__ == "__main__":
    main()
