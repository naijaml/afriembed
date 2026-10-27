"""Experiment 7: contrastive LoRA fine-tune of the text tower on African cross-lingual pairs (AfriE5 recipe, 270M)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from ..data.text_corpora import CONTAMINATION_BLOCKLIST, afri_aya_pairs, africaption_pairs
from ..languages import get
from ..models.registry import add_lora, save_adapter
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def build_pairs(
    languages: list[str],
    max_per_lang: int | None = 4000,
    include_afri_aya: bool = True,
    extra_tsv: Path | None = None,
    seed: int = 0,
    include_captions: bool = True,
) -> list[dict]:
    """anchor/positive rows: English<->African caption pairs (both directions). Never uses AfriMTEB test sets.

    With include_captions=False only `extra_tsv` pairs are used (e.g. real parallel sentences), one direction per
    line as given; both directions are added here."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for code in languages if include_captions else []:
        lang = get(code)
        pairs = africaption_pairs(lang)
        rng.shuffle(pairs)
        pairs = pairs[:max_per_lang] if max_per_lang else pairs
        for en, loc in pairs:
            rows.append({"anchor": en, "positive": loc, "lang": code})
            rows.append({"anchor": loc, "positive": en, "lang": code})
        log.info("%s: %d AfriCaption pairs", code, len(pairs))
    if include_afri_aya and include_captions:
        aya = afri_aya_pairs()
        for lang_name, en, loc in aya:
            rows.append({"anchor": en, "positive": loc, "lang": f"aya:{lang_name}"})
            rows.append({"anchor": loc, "positive": en, "lang": f"aya:{lang_name}"})
        log.info("Afri-Aya: %d pairs", len(aya))
    if extra_tsv:
        n = 0
        for line in Path(extra_tsv).read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                a, p = line.split("\t", 1)
                rows.append({"anchor": a.strip(), "positive": p.strip(), "lang": "extra"})
                rows.append({"anchor": p.strip(), "positive": a.strip(), "lang": "extra"})
                n += 1
        log.info("extra: %d pairs from %s", n, extra_tsv)
    rng.shuffle(rows)
    log.info("Total training pairs: %d  (blocklist: %s)", len(rows), ", ".join(CONTAMINATION_BLOCKLIST))
    return rows


def train(
    opts: ModelOpts,
    languages: list[str],
    out_dir: Path,
    epochs: float = 1.0,
    lr: float = 2e-5,
    batch_size: int = 32,
    mini_batch_size: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    full_finetune: bool = False,
    max_pairs: int | None = None,
    max_per_lang: int | None = 4000,
    extra_tsv: Path | None = None,
    anchor_task: str = "query",
    positive_task: str = "document",
    seed: int = 0,
    include_captions: bool = True,
) -> Path:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    emb = opts.load(("text",))
    rows = build_pairs(
        languages,
        max_per_lang=max_per_lang,
        extra_tsv=extra_tsv,
        seed=seed,
        include_captions=include_captions,
    )
    if max_pairs:
        rows = rows[:max_pairs]
    ds = Dataset.from_list([{"anchor": r["anchor"], "positive": r["positive"]} for r in rows])
    model = emb.model
    if not full_finetune:
        add_lora(model, lora_r, lora_alpha, lora_dropout)
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    prompts = {"anchor": emb.prompt_string(anchor_task), "positive": emb.prompt_string(positive_task)}
    import torch

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=seed,
        prompts=prompts,
        gradient_checkpointing=False,  # EmbeddingGemma2Model rejects transformers 5.19 checkpointing kwargs; H100 has the memory
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    with timer("train text"):
        trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    if full_finetune:
        model.save_pretrained(str(out_dir / "model"))
        saved = out_dir / "model"
    else:
        saved = save_adapter(model, out_dir / "adapter")
    save_result(
        "finetune-text",
        emb.name,
        {
            "saved": str(saved),
            "n_pairs": len(rows),
            "languages": languages,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "lora": None if full_finetune else {"r": lora_r, "alpha": lora_alpha},
            "prompts": prompts,
        },
        tag="train",
    )
    log.info("Saved -> %s  (evaluate with: afriembed afrimteb --adapter %s)", saved, saved)
    return saved
