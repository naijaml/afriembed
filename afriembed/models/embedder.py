"""Uniform wrapper over sentence-transformers models with per-family task prompts.

Generic task names used across experiments:
    query, document, classification, clustering, sts, bitext, none
EmbeddingGemma families map these to the prompt catalog shipped in the model's
config_sentence_transformers.json (prompt_name). E5 families prepend literal prefixes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

TASKS = ("query", "document", "classification", "clustering", "sts", "bitext", "none")

# prompt_name values from the EmbeddingGemma 2 catalog (also valid for EmbeddingGemma 1).
_GEMMA_PROMPT_NAMES = {
    "query": "query",
    "document": "document",
    "classification": "Classification",
    "clustering": "Clustering",
    "sts": "STS",
    "bitext": "BitextMining",
    "none": None,
}

# Literal prefixes for prompt-less families.
_E5_INSTRUCT_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)
_E5_INSTRUCT_CLS = "Instruct: Classify the given text\nQuery: "
_E5_INSTRUCT_CLU = "Instruct: Identify the topic or theme of the given text\nQuery: "
_E5_INSTRUCT_STS = "Instruct: Retrieve semantically similar text\nQuery: "
_E5_INSTRUCT_BITEXT = "Instruct: Retrieve parallel sentences\nQuery: "

_FAMILY_PREFIXES: dict[str, dict[str, str]] = {
    "e5": {
        "query": "query: ",
        "document": "passage: ",
        "classification": "query: ",
        "clustering": "query: ",
        "sts": "query: ",
        "bitext": "query: ",
        "none": "",
    },
    "e5-instruct": {
        "query": _E5_INSTRUCT_QUERY,
        "document": "",
        "classification": _E5_INSTRUCT_CLS,
        "clustering": _E5_INSTRUCT_CLU,
        "sts": _E5_INSTRUCT_STS,
        "bitext": _E5_INSTRUCT_BITEXT,
        "none": "",
    },
    "plain": {t: "" for t in TASKS},
}


_MEDIA_LIKE = re.compile(
    r"^(?:https?://|www\.)|\.(?:mp4|mov|avi|mkv|webm|m4v|wav|mp3|flac|ogg|m4a|opus|jpe?g|png|gif|webp|bmp|tiff?)(?:[?#]|$)",
    re.I,
)


def looks_like_media(s: str) -> bool:
    """True for strings sentence-transformers' multimodal auto-detection could mistake for a file/URL input."""
    t = s.strip()
    return (
        bool(t)
        and (" " not in t or t.startswith(("http://", "https://", "www.")))
        and bool(_MEDIA_LIKE.search(t))
        or (" " not in t and "/" in t and "." in t)
    )


def guard_text(items: list) -> list:
    """Wrap media-looking plain strings as {'text': ...} so they are always embedded as text."""
    return [{"text": it} if isinstance(it, str) and looks_like_media(it) else it for it in items]


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def pick_dtype(device: str, override: str | None = None) -> torch.dtype:
    """bf16 on CUDA that supports it, else fp32. fp16 only when asked for explicitly (an ablation): EmbeddingGemma
    produces NaN / degenerate embeddings in fp16, which is a candidate explanation for the AfriMTEB paper's v1 row."""
    if override:
        return DTYPES[override]
    if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def truncate(x: np.ndarray, dim: int | None) -> np.ndarray:
    """Matryoshka slice + re-normalise (sentence-transformers' truncate_dim slices normalised vectors, so renorm)."""
    if dim is None or dim >= x.shape[-1]:
        return l2_normalize(x)
    return l2_normalize(x[:, :dim])


def mteb_prompt_catalog(family: str, use_prompts: bool) -> dict[str, str] | None:
    """A prompts dict in the key scheme mteb resolves (prompt type, task type, task-type+prompt-type).

    EmbeddingGemma checkpoints ship their own catalog, so None is returned for them (mteb uses the built-in one).
    """
    if family.startswith("embeddinggemma") or not use_prompts:
        return None
    pre = _FAMILY_PREFIXES.get(family)
    if pre is None or not any(pre.values()):
        return None
    cat = {
        "query": pre["query"],
        "document": pre["document"],
        "Retrieval-query": pre["query"],
        "Retrieval-document": pre["document"],
        "Reranking-query": pre["query"],
        "Reranking-document": pre["document"],
        "InstructionRetrieval-query": pre["query"],
        "InstructionRetrieval-document": pre["document"],
        "Classification": pre["classification"],
        "MultilabelClassification": pre["classification"],
        "Clustering": pre["clustering"],
        "STS": pre["sts"],
        "PairClassification": pre["sts"],
        "Summarization": pre["sts"],
        "BitextMining": pre["bitext"],
    }
    return {k: v for k, v in cat.items() if v}


@dataclass
class Embedder:
    name: str
    hf_id: str
    family: str
    model: Any  # sentence_transformers.SentenceTransformer
    modalities: tuple[str, ...] = ("text",)
    dim: int = 768
    use_prompts: bool = True
    truncate_dim: int | None = None
    batch_size: int = 32
    extra: dict = field(default_factory=dict)

    # ---- prompts -------------------------------------------------------
    def prompt_kwargs(self, task: str) -> dict:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if not self.use_prompts or task == "none":
            return {"prompt": ""} if self.family.startswith("embeddinggemma") else {}
        if self.family.startswith("embeddinggemma"):
            name = _GEMMA_PROMPT_NAMES[task]
            available = getattr(self.model, "prompts", {}) or {}
            if name is None:
                return {"prompt": ""}
            if name in available:
                return {"prompt_name": name}
            # fall back to literal prefix (same strings as the catalog) if a checkpoint lacks the entry
            literal = {
                "query": "task: search result | query: ",
                "document": "title: none | text: ",
                "classification": "task: classification | query: ",
                "clustering": "task: clustering | query: ",
                "sts": "task: sentence similarity | query: ",
                "bitext": "task: search result | query: ",
            }[task]
            return {"prompt": literal}
        prefixes = _FAMILY_PREFIXES.get(self.family)
        if prefixes is None:
            raise ValueError(f"Unknown model family {self.family!r}")
        return {"prompt": prefixes[task]}

    def prompt_string(self, task: str) -> str:
        """Human-readable prefix actually applied (for logging / results metadata)."""
        kw = self.prompt_kwargs(task)
        if "prompt" in kw:
            return kw["prompt"]
        return (getattr(self.model, "prompts", {}) or {}).get(kw.get("prompt_name", ""), "")

    # ---- encoding ------------------------------------------------------
    def supports(self, modality: str) -> bool:
        return modality in self.modalities

    def encode(
        self,
        items: list,
        task: str = "none",
        *,
        batch_size: int | None = None,
        dim: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode strings or multimodal dicts ({'text':..,'audio':path,'image':..,'video':..}).

        Prompts are applied to text; for multimodal dicts the EmbeddingGemma catalog is still valid
        (query/document) but we default to no prefix for pure media unless task is query/document.
        """
        if len(items) == 0:
            return np.zeros((0, self.truncate_dim or self.dim), dtype=np.float32)
        has_media = any(
            isinstance(it, dict) and any(k in it for k in ("audio", "image", "video")) for it in items
        )
        if has_media:
            missing = {
                k
                for it in items
                if isinstance(it, dict)
                for k in ("audio", "image", "video")
                if k in it and not self.supports(k)
            }
            if missing:
                raise RuntimeError(
                    f"Model {self.name} does not support modalities {sorted(missing)} (loaded with towers disabled?)"
                )
            if task not in ("query", "document"):
                task = "none"
        kw = self.prompt_kwargs(task)
        if self.family.startswith("embeddinggemma2"):
            items = guard_text(items)
        emb = self.model.encode(
            items,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            **kw,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return truncate(emb, dim if dim is not None else self.truncate_dim)

    def encode_text(self, texts: list[str], task: str = "none", **kw) -> np.ndarray:
        return self.encode(list(texts), task, **kw)

    def encode_audio(self, paths: list[str], task: str = "none", **kw) -> np.ndarray:
        return self.encode([{"audio": p} for p in paths], task, **kw)

    def encode_images(self, images: list, task: str = "none", **kw) -> np.ndarray:
        return self.encode([{"image": im} for im in images], task, **kw)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "family": self.family,
            "modalities": list(self.modalities),
            "dim": self.dim,
            "truncate_dim": self.truncate_dim,
            "use_prompts": self.use_prompts,
            "prompts": {t: self.prompt_string(t) for t in TASKS if t != "none"},
            **self.extra,
        }
