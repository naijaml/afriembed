"""Experiment 9: vocabulary trimming by id-remap.

Gemma's 262k-token vocabulary dominates the 270M text tower (token embeddings + per-layer embeddings).
We keep the tokenizer untouched (so tokenisation is bit-identical) and prune every vocab-sized
embedding table to the ids observed in a target-language corpus, remapping ids with a forward
pre-hook. Unseen ids map to <unk>/<pad>. This is lossless on covered text and reports coverage on
held-out text so the trade-off is explicit.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.text_corpora import mixed_monolingual
from ..utils import save_result
from .common import ModelOpts

log = logging.getLogger(__name__)


def _vocab_tables(auto_model, vocab_size: int) -> list[tuple[str, nn.Embedding]]:
    return [
        (n, m)
        for n, m in auto_model.named_modules()
        if isinstance(m, nn.Embedding) and m.num_embeddings == vocab_size
    ]


def _count_ids(tok, texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(tok(t, add_special_tokens=True)["input_ids"])
    return c


def build(
    opts: ModelOpts,
    languages: list[str],
    out_dir: Path,
    extra_text: Path | None = None,
    keep_top_general: int = 8000,
    min_count: int = 1,
    holdout_frac: float = 0.1,
) -> dict:
    emb = opts.load(("text",))
    tok = emb.model.tokenizer
    auto = emb.model[0].auto_model
    vocab_size = len(tok)
    corpora = mixed_monolingual(languages)
    if extra_text:
        corpora["extra"] = [l for l in Path(extra_text).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_texts, held_texts = [], []
    for code, texts in corpora.items():
        n_hold = int(len(texts) * holdout_frac)
        held_texts += texts[:n_hold]
        train_texts += texts[n_hold:]
        log.info("%s: %d texts", code, len(texts))
    counts = _count_ids(tok, train_texts)
    keep = {i for i, c in counts.items() if c >= min_count}
    keep |= set(tok.all_special_ids)
    for attr in ("unk_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            keep.add(int(v))
    # a slice of the most frequent ids overall (ascii/punct/English function words sit low in the id space for Gemma)
    keep |= set(range(min(keep_top_general, vocab_size)))
    keep_ids = sorted(keep)
    unk_old = tok.unk_token_id if tok.unk_token_id is not None else (tok.pad_token_id or 0)
    held_counts = _count_ids(tok, held_texts)
    covered = sum(c for i, c in held_counts.items() if i in keep) / max(1, sum(held_counts.values()))

    tables = _vocab_tables(auto, vocab_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = torch.tensor(keep_ids, dtype=torch.long)
    state = {}
    before = after = 0
    for name, mod in tables:
        w = mod.weight.detach().cpu()
        state[name] = w[idx].contiguous()
        before += w.numel()
        after += state[name].numel()
    from safetensors.torch import save_file

    save_file({k: v.to(torch.float32) for k, v in state.items()}, str(out_dir / "embeddings.safetensors"))
    meta = {
        "model": emb.hf_id,
        "vocab_size": vocab_size,
        "kept": len(keep_ids),
        "keep_ids": keep_ids,
        "unk_old_id": int(unk_old),
        "tables": [n for n, _ in tables],
        "languages": languages,
        "heldout_coverage": covered,
        "params_before": before,
        "params_after": after,
    }
    (out_dir / "trim.json").write_text(json.dumps(meta))
    bytes_per = 2  # bf16 deployment
    report = {
        "kept_tokens": len(keep_ids),
        "vocab_size": vocab_size,
        "tables": [n for n, _ in tables],
        "embedding_params_before": before,
        "embedding_params_after": after,
        "embedding_mb_saved_bf16": (before - after) * bytes_per / 1e6,
        "heldout_token_coverage": covered,
        "languages": languages,
        "out_dir": str(out_dir),
    }
    log.info(
        "Trimmed %d -> %d tokens over %d tables; saves %.0f MB (bf16); held-out coverage %.4f",
        vocab_size,
        len(keep_ids),
        len(tables),
        report["embedding_mb_saved_bf16"],
        covered,
    )
    save_result("vocab-trim", emb.name, report, tag="build")
    return report


def apply_trim(emb, trim_dir: str | Path) -> None:
    """Prune embedding tables in-place and install id remapping hooks on a loaded Embedder."""
    from safetensors.torch import load_file

    trim_dir = Path(trim_dir)
    meta = json.loads((trim_dir / "trim.json").read_text())
    state = load_file(str(trim_dir / "embeddings.safetensors"))
    auto = emb.model[0].auto_model
    keep_ids = meta["keep_ids"]
    remap = torch.full((meta["vocab_size"],), -1, dtype=torch.long)
    remap[torch.tensor(keep_ids)] = torch.arange(len(keep_ids))
    unk_new = int(remap[meta["unk_old_id"]].item())
    remap[remap < 0] = unk_new
    modules = dict(auto.named_modules())
    for name in meta["tables"]:
        mod: nn.Embedding = modules[name]
        w = state[name].to(device=mod.weight.device, dtype=mod.weight.dtype)
        mod.weight = nn.Parameter(w, requires_grad=False)
        mod.num_embeddings = w.shape[0]
        if mod.padding_idx is not None:
            mod.padding_idx = int(remap[mod.padding_idx].item())
        r = remap.to(mod.weight.device)

        def _hook(module, args, _r=r):
            ids = args[0]
            return (_r[ids],) + tuple(args[1:])

        mod.register_forward_pre_hook(_hook)
    emb.extra["trim"] = {"dir": str(trim_dir), "kept": len(keep_ids), "vocab_size": meta["vocab_size"]}
    log.info("Applied vocab trim from %s: %d/%d ids kept", trim_dir, len(keep_ids), meta["vocab_size"])


def coverage(trim_dir: Path, texts: list[str], tokenizer) -> float:
    meta = json.loads((Path(trim_dir) / "trim.json").read_text())
    keep = set(meta["keep_ids"])
    c = _count_ids(tokenizer, texts)
    return sum(v for i, v in c.items() if i in keep) / max(1, sum(c.values()))


def param_count(emb) -> int:
    return int(sum(p.numel() for p in emb.model[0].auto_model.parameters()))


def _unused():  # keep numpy import for type parity in reports
    return np
