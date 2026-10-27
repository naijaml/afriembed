"""Project paths, environment, and YAML registries."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

load_dotenv(ROOT / ".env")


def cache_dir() -> Path:
    p = Path(os.environ.get("AFRIEMBED_CACHE", ROOT / "data_cache")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = Path(os.environ.get("AFRIEMBED_RESULTS", ROOT / "results")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def mteb_cache_dir() -> Path:
    p = Path(os.environ.get("MTEB_CACHE", ROOT / "mteb_cache")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@cache
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_registry() -> dict[str, dict]:
    return load_yaml("models.yaml")["models"]


def tokenizer_registry() -> dict[str, str]:
    return load_yaml("models.yaml")["tokenizers"]
