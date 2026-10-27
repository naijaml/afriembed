"""Cross-language sentence-overlap check for the FLEURS speech results.

FLEURS sentence ids (FLORES ids) are shared across languages. This intersects the union of train ids of the
adapter's six training languages with the test ids of every evaluated language. Result (2026-10-17): 0 overlap
for 20 languages; Afrikaans 233/233 (excluded from transfer claims). Reads locally cached arrow files where
present and downloads only the test parquet files for the rest.

Usage: uv run python scripts/check_fleurs_overlap.py
"""

import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

root = os.path.expanduser("~/.cache/huggingface/datasets/mteb___fleurs")


def local_ids(cfg, split):
    out = set()
    for f in glob.glob(f"{root}/{cfg}/**/fleurs-{split}*.arrow", recursive=True):
        with pa.memory_map(f) as src:
            try:
                t = pa.ipc.open_stream(src).read_all()
            except Exception:
                src.seek(0)
                t = pa.ipc.open_file(src).read_all()
        out |= set(int(x) for x in t.column("id").to_pylist())
    return out


train = set()
for cfg in ("ha_ng", "yo_ng", "sw_ke", "am_et", "ig_ng", "so_so"):
    train |= local_ids(cfg, "train")
files = HfApi().list_repo_files("mteb/fleurs", repo_type="dataset")
CACHED = [
    c for c in sorted(os.listdir(root)) if glob.glob(f"{root}/{c}/**/fleurs-test*.arrow", recursive=True)
]
for cfg in CACHED:
    s = local_ids(cfg, "test")
    print(f"test {cfg}: {len(s)} ids, in any train: {len(s & train)}", flush=True)
for cfg in ("ln_cd", "sn_zw", "nso_za", "ny_mw", "umb_ao", "ff_sn", "kam_ke", "kea_cv"):
    tf = [
        f
        for f in files
        if f"/{cfg}/" in f"/{f}" and "/test" in f"/{f}" or (f.startswith(f"{cfg}/") and "test" in f)
    ]
    tf = sorted(set(f for f in tf if f.endswith(".parquet") and cfg in f and "test" in f))
    ids = set()
    for f in tf:
        for attempt in range(5):
            try:
                p = hf_hub_download("mteb/fleurs", f, repo_type="dataset")
                break
            except Exception as e:
                print(cfg, "retry", attempt, str(e)[:80], flush=True)
                time.sleep(10)
        ids |= set(int(x) for x in pq.read_table(p, columns=["id"]).column("id").to_pylist())
    print(f"test {cfg}: {len(ids)} ids, in any train: {len(ids & train)}", flush=True)
print("DONE", flush=True)
