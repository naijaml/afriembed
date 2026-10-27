"""Load any registered model (or an arbitrary HF id / local path) as an Embedder."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import hf_token, model_registry
from .embedder import Embedder, pick_device, pick_dtype

log = logging.getLogger(__name__)


def load_embedder(
    name: str,
    *,
    modalities: tuple[str, ...] | list[str] = ("text",),
    device: str | None = None,
    use_prompts: bool = True,
    truncate_dim: int | None = None,
    batch_size: int = 32,
    max_seq_length: int | None = None,
    local_path: str | None = None,
    adapter_path: str | None = None,
    dtype_override: str | None = None,
    causal_attention: bool = False,
) -> Embedder:
    """Load a model by registry name, HF id, or local directory.

    modalities: which towers to instantiate for EmbeddingGemma 2. ('text',) skips vision+audio (271M);
    ('text','audio') skips vision; ('text','image','audio','video') loads everything (744M).
    """
    from sentence_transformers import SentenceTransformer

    reg = model_registry()
    if name in reg:
        spec = dict(reg[name])
        hf_id = spec["hf_id"]
        family = spec["family"]
        revision = spec.get("revision")
    else:
        revision = None
        spec = {"modalities": ["text"], "dim": None}
        hf_id = name
        family = (
            "embeddinggemma2"
            if "embeddinggemma-2" in name
            else "embeddinggemma1"
            if "embeddinggemma" in name
            else "plain"
        )
        log.warning("Model %s not in registry; assuming family=%s", name, family)

    source = local_path or hf_id
    device = pick_device(device)
    dtype = pick_dtype(device, dtype_override)
    model_kwargs = {"torch_dtype": dtype}
    config_kwargs = {}
    wanted = tuple(modalities)
    if causal_attention:
        # Ablation: transformers releases before EmbeddingGemma support ignored `use_bidirectional_attention`
        # and ran the encoder with a causal mask. This reproduces that behaviour.
        config_kwargs["use_bidirectional_attention"] = False
    if family == "embeddinggemma2":
        if "image" not in wanted and "video" not in wanted:
            config_kwargs["vision_config"] = None
        if "audio" not in wanted:
            config_kwargs["audio_config"] = None
    supported = tuple(m for m in spec.get("modalities", ["text"]) if m in wanted or m == "text")

    st_kwargs = dict(device=device, model_kwargs=model_kwargs, token=hf_token())
    if revision and not local_path:
        st_kwargs["revision"] = revision
    if config_kwargs:
        st_kwargs["config_kwargs"] = config_kwargs
    log.info("Loading %s (%s) on %s dtype=%s towers=%s", name, source, device, dtype, supported)
    model = SentenceTransformer(source, **st_kwargs)
    if max_seq_length:
        # never exceed the backbone's positional limit (XLM-R models: 512) -> CUDA device-side asserts otherwise
        # sentence-transformers initialises max_seq_length to the backbone's real limit (512 for XLM-R,
        # whose config says 514 but only 512 positions are usable), so use that as the hard cap.
        hard = int(getattr(model, "max_seq_length", 0) or 0)
        if hard and max_seq_length > hard:
            log.warning("max_seq_length %d exceeds model limit %d; capping", max_seq_length, hard)
            max_seq_length = hard
        model.max_seq_length = max_seq_length

    if adapter_path:
        _attach_adapter(model, adapter_path)

    if not use_prompts:
        # Clear the prompt catalog so mteb's wrapper and encode_query/encode_document cannot re-apply prefixes.
        model.prompts = {}
        model.default_prompt_name = None

    dim = spec.get("dim") or model.get_sentence_embedding_dimension()
    return Embedder(
        name=name if not local_path else f"{name}@{Path(local_path).name}",
        hf_id=hf_id,
        family=family,
        model=model,
        modalities=supported,
        dim=int(dim),
        use_prompts=use_prompts,
        truncate_dim=truncate_dim,
        batch_size=batch_size,
        extra={
            "device": device,
            "dtype": str(dtype),
            "causal_attention": causal_attention,
            "source": source,
            "adapter": adapter_path,
            "max_seq_length": int(getattr(model, "max_seq_length", 0) or 0),
            "revision": revision,
        },
    )


def _attach_adapter(model, adapter_path: str) -> None:
    """Attach one or more PEFT LoRA adapters (comma-separated paths) via sentence-transformers' native API.

    Several adapters are loaded under distinct names and all activated, so their LoRA deltas sum (stacking)."""
    paths = [p.strip() for p in str(adapter_path).split(",") if p.strip()]
    names = []
    for i, p in enumerate(paths):
        name = f"adapter{i}" if len(paths) > 1 else "default"
        model.load_adapter(_text_tower_scoped(model, p), adapter_name=name)
        names.append(name)
        log.info("Attached adapter %s from %s", name, p)
    if len(names) > 1:
        model.set_adapter(names)
        log.info("Activated stacked adapters: %s", names)
    for p in paths:
        tower = Path(p).parent / "audio_tower.pt"
        if tower.exists():
            import torch

            state = torch.load(tower, map_location="cpu")
            _, unexpected = model[0].auto_model.load_state_dict(state, strict=False)
            log.info(
                "Loaded fine-tuned audio tower from %s (%d tensors)", tower, len(state) - len(unexpected)
            )


LORA_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def text_tower_regex(auto_model, leaves=LORA_LEAVES) -> str:
    """PEFT target_modules regex matching exactly the text-tower projection modules (never audio/vision
    encoders), so an adapter trained text-only still loads when other towers are instantiated."""
    import re

    names = [
        n
        for n, _ in auto_model.named_modules()
        if n.endswith(leaves) and "audio" not in n and "vision" not in n
    ]
    if not names:
        raise RuntimeError("Could not locate text-tower projections for LoRA")
    return "^(?:" + "|".join(re.escape(n) for n in sorted(names)) + ")$"


def _text_tower_scoped(model, adapter_path: str) -> str:
    """If an adapter's config lists bare module names (trained text-only), return a temp copy whose
    target_modules is a text-tower regex for the currently loaded towers; otherwise return the path unchanged."""
    import json
    import shutil
    import tempfile

    src = Path(adapter_path)
    cfg_p = src / "adapter_config.json"
    if not cfg_p.exists():
        return str(src)
    cfg = json.loads(cfg_p.read_text())
    tm = cfg.get("target_modules")
    if not isinstance(tm, list):
        return str(src)
    leaves = tuple(tm)
    regex = text_tower_regex(model[0].auto_model, leaves)
    tmp = Path(tempfile.mkdtemp(prefix="afriembed-adapter-"))
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, tmp / f.name)
    cfg["target_modules"] = regex
    (tmp / "adapter_config.json").write_text(json.dumps(cfg))
    log.info("Adapter targets %s rewritten to text-tower regex", tm)
    return str(tmp)


def add_lora(model, r: int, alpha: int, dropout: float = 0.0, target_modules=None) -> dict:
    """Inject a LoRA adapter into the loaded SentenceTransformer (transformers PEFT integration), freeze everything
    else, and return parameter counts."""
    from peft import LoraConfig

    cfg = LoraConfig(
        task_type="FEATURE_EXTRACTION",  # the Hub config parser requires a string here
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.add_adapter(cfg)
    auto = model[0].auto_model
    for name, prm in auto.named_parameters():
        prm.requires_grad_("lora_" in name)
    trainable = sum(x.numel() for x in auto.parameters() if x.requires_grad)
    total = sum(x.numel() for x in auto.parameters())
    log.info(
        "LoRA r=%d alpha=%d: trainable %d / %d (%.2f%%)", r, alpha, trainable, total, 100 * trainable / total
    )
    return {"trainable": trainable, "total": total}


def save_adapter(model, out_dir) -> Path:
    """Save only the adapter weights (adapter_config.json + adapter_model.safetensors)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(out_dir))
    return out_dir
