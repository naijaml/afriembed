"""Results I/O, seeding, timing, and environment capture."""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import ROOT, results_dir

log = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    for noisy in ("httpx", "urllib3", "filelock", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    env = {
        "afriembed": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_sha(),
    }
    for mod in ("torch", "transformers", "sentence_transformers", "mteb", "datasets", "peft"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch

        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


def run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.@" else "_" for c in name)


def save_result(experiment: str, model_name: str, payload: dict[str, Any], tag: str | None = None) -> Path:
    """Write results/<experiment>/<model>/<run_id>[-tag].json and update latest.json."""
    d = results_dir() / experiment / _safe(model_name)
    d.mkdir(parents=True, exist_ok=True)
    rid = run_id() + (f"-{_safe(tag)}" if tag else "")
    payload = {
        "experiment": experiment,
        "model": model_name,
        "run_id": rid,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "env": environment(),
        **payload,
    }
    p = d / f"{rid}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    latest = d / (
        f"latest-{_safe(tag)}.json" if tag else "latest.json"
    )  # one pointer per variant (dim / prompts / adapter)
    latest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    log.info("Saved %s", p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return p


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_results(experiment: str | None = None) -> list[dict]:
    base = results_dir()
    out = []
    for p in sorted(base.glob(f"{experiment or '*'}/*/*.json")):
        if p.name.startswith("latest"):
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # pragma: no cover
            log.warning("Skipping %s: %s", p, e)
    return out


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    log.info("%s took %.1fs", label, time.perf_counter() - t0)


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
