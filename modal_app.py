"""Run any afriembed command on Modal GPUs.

    modal run modal_app.py --cmd "smoke --audio"
    modal run modal_app.py --cmd "afrimteb --suite lite --no-prompts"
    modal run modal_app.py --cmd "speech-text --langs fleurs_all"
    modal run modal_app.py --cmds "afrimteb --suite lite ; afrimteb --suite lite --model embeddinggemma-300m ; report"
    AFRIEMBED_GPU=A100-80GB modal run modal_app.py --cmd "finetune-audio --langs hau,yor,swh,amh --train-audio-tower --out /vol/checkpoints/audio-full"

Setup once:
    modal secret create huggingface-secret HF_TOKEN=hf_...        # after accepting the google/embeddinggemma-2 license
Results, model cache, datasets and checkpoints persist in the `afriembed-vol` volume:
    modal volume get afriembed-vol results ./results        # pull results back, then `uv run afriembed report`
    modal volume ls afriembed-vol checkpoints
"""

from __future__ import annotations

import os
import shlex
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).parent
GPU = os.environ.get(
    "AFRIEMBED_GPU", "A10G"
)  # A10G (24 GB) fits every eval; L40S/A100 for audio-tower training
TIMEOUT_S = int(os.environ.get("AFRIEMBED_TIMEOUT", 6 * 3600))

HF_SECRET = os.environ.get("AFRIEMBED_HF_SECRET", "huggingface-secret")  # must contain HF_TOKEN
VOL = modal.Volume.from_name("afriembed-vol", create_if_missing=True)
VOL_PATH = "/vol"

# This module is re-imported inside the container (no pyproject there); only parse deps on the local side.
_deps = (
    tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    if modal.is_local()
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(*_deps, "gradio>=5.0")
    .env(
        {
            "HF_HOME": f"{VOL_PATH}/hf",
            "AFRIEMBED_CACHE": f"{VOL_PATH}/data_cache",
            "AFRIEMBED_RESULTS": f"{VOL_PATH}/results",
            "MTEB_CACHE": f"{VOL_PATH}/mteb_cache",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
    .add_local_python_source("afriembed")
)

app = modal.App("afriembed", image=image)


def _run_one(argv: list[str]) -> dict:
    """Run one CLI command; errors are returned as text so the local side needs no remote packages to show them."""
    import traceback

    from afriembed.cli import app as cli

    print(f"\n=== afriembed {' '.join(argv)}", flush=True)
    try:
        cli(argv, standalone_mode=False)
        return {"code": 0, "error": None}
    except SystemExit as e:  # typer exits
        return {"code": int(e.code or 0), "error": None}
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return {"code": 1, "error": tb[-4000:]}


@app.function(
    gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_PATH: VOL}, secrets=[modal.Secret.from_name(HF_SECRET)]
)
def run(commands: list[str]) -> list[dict]:
    """Run afriembed commands sequentially in one container (model and data stay cached between them)."""
    os.chdir(VOL_PATH)  # relative --out paths (checkpoints/..., indexes/...) land on the volume
    results = []
    for c in commands:
        results.append({"cmd": c, **_run_one(shlex.split(c))})
        VOL.commit()
    return results


@app.function(timeout=600, volumes={VOL_PATH: VOL})
def report() -> str:
    """CPU-only: rebuild results/REPORT.md on the volume and return it."""
    os.chdir(VOL_PATH)
    from afriembed.experiments.report import build

    text = build()
    VOL.commit()
    return text


@app.local_entrypoint()
def main(cmd: str = "", cmds: str = ""):
    """--cmd "afrimteb --suite lite"   or   --cmds "a ; b ; c" (semicolon separated)."""
    commands = [c.strip() for c in (cmds.split(";") if cmds else [cmd]) if c.strip()]
    if not commands:
        raise SystemExit('Pass --cmd "..." or --cmds "a ; b"')
    if commands == ["report"]:
        print(report.remote())
        return
    results = run.remote(commands)
    for r in results:
        print(f"{'ok ' if r['code'] == 0 else 'ERR'} {r['cmd']}")
        if r["error"]:
            print(r["error"])
    print(report.remote())
    if any(r["code"] for r in results):
        raise SystemExit(1)
