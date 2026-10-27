"""Experiment 8: align the audio tower to African speech with (audio, transcript) contrastive pairs from FLEURS train."""

from __future__ import annotations

import logging
from pathlib import Path

from ..data.fleurs import load_fleurs
from ..languages import with_fleurs
from ..models.registry import add_lora, save_adapter, text_tower_regex
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)


def _set_audio_trainable(auto_model, trainable: bool) -> int:
    n = 0
    for name, p in auto_model.named_parameters():
        if "audio" in name:
            p.requires_grad_(trainable)
            n += p.numel() if trainable else 0
    return n


def train(
    opts: ModelOpts,
    languages: list[str],
    out_dir: Path,
    split: str = "train",
    max_per_lang: int | None = 1500,
    epochs: float = 1.0,
    lr: float = 2e-5,
    batch_size: int = 8,
    mini_batch_size: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    train_audio_tower: bool = False,
    text_task: str = "document",
    seed: int = 0,
) -> Path:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    emb = opts.load(("text", "audio"))
    rows = []
    for lang in with_fleurs(languages):
        utts = load_fleurs(lang, split=split, max_items=max_per_lang, dedupe=False)
        rows += [{"anchor": {"audio": str(u.wav)}, "positive": u.text} for u in utts]
    log.info("%d (audio, transcript) pairs", len(rows))
    ds = Dataset.from_list(rows).shuffle(seed=seed)

    model = emb.model
    tf = model[0]
    add_lora(model, lora_r, lora_alpha, 0.0, target_modules=text_tower_regex(tf.auto_model))
    n_audio = _set_audio_trainable(tf.auto_model, train_audio_tower)
    if train_audio_tower:
        log.info("Audio tower unfrozen: %d params", n_audio)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
        prompts={"positive": emb.prompt_string(text_task)},
        gradient_checkpointing=False,  # EmbeddingGemma2Model rejects transformers 5.19 checkpointing kwargs; H100 has the memory
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    with timer("train audio"):
        trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(model, out_dir / "adapter")
    if train_audio_tower:
        audio_state = {k: v.detach().cpu() for k, v in tf.auto_model.named_parameters() if "audio" in k}
        torch.save(audio_state, out_dir / "audio_tower.pt")
    save_result(
        "finetune-audio",
        emb.name,
        {
            "saved": str(out_dir / "adapter"),
            "n_pairs": len(rows),
            "languages": languages,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "train_audio_tower": train_audio_tower,
            "lora": {"r": lora_r, "alpha": lora_alpha},
        },
        tag="train",
    )
    log.info(
        "Saved -> %s  (evaluate with: afriembed speech-text --adapter %s)",
        out_dir / "adapter",
        out_dir / "adapter",
    )
    return out_dir / "adapter"
