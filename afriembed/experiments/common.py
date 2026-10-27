"""Shared CLI option bundle -> Embedder."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Embedder, load_embedder


@dataclass
class ModelOpts:
    model: str = "embeddinggemma-2"
    dim: int | None = None
    use_prompts: bool = True
    batch_size: int = 32
    device: str | None = None
    adapter: str | None = None
    local_path: str | None = None
    trim_dir: str | None = None
    max_seq_length: int | None = None
    dtype: str | None = None  # bf16 | fp16 | fp32; None = bf16 on capable GPUs
    causal_attention: bool = False  # ablation: run the encoder with a causal mask (pre-fix transformers)
    modalities: tuple[str, ...] = field(default_factory=lambda: ("text",))

    def load(self, modalities: tuple[str, ...] | None = None) -> Embedder:
        emb = load_embedder(
            self.model,
            modalities=modalities or self.modalities,
            device=self.device,
            use_prompts=self.use_prompts,
            truncate_dim=self.dim,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            local_path=self.local_path,
            adapter_path=self.adapter,
            dtype_override=self.dtype,
            causal_attention=self.causal_attention,
        )
        if self.trim_dir:
            from .vocab_trim import apply_trim

            apply_trim(emb, self.trim_dir)
        return emb

    def full_tag(self, user_tag: str | None) -> str | None:
        """Variant tag (dim/noprompt/adapter/trimmed) combined with a user --tag so pointers never collide."""
        base = self.tag()
        if base and user_tag:
            return f"{base}-{user_tag}"
        return user_tag or base or None

    def tag(self) -> str:
        parts = []
        if self.dim:
            parts.append(f"d{self.dim}")
        if not self.use_prompts:
            parts.append("noprompt")
        if self.adapter:
            parts.append("adapter")
        if self.trim_dir:
            parts.append("trimmed")
        if self.dtype:
            parts.append(self.dtype)
        if self.causal_attention:
            parts.append("causal")
        if self.max_seq_length and self.max_seq_length != 1024:
            parts.append(f"seq{self.max_seq_length}")
        return "-".join(parts)
