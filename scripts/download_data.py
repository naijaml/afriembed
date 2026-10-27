"""Pre-download the datasets used by the experiments (so runs are offline-friendly)."""

from __future__ import annotations

import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

from afriembed.config import hf_token
from afriembed.languages import all_languages, language_set

tok = hf_token()
langs = sys.argv[1:] or language_set("fleurs_core")
reg = all_languages()
for c in langs:
    l = reg[c]
    if l.fleurs:
        load_dataset("mteb/fleurs", l.fleurs, split="test", token=tok)
    load_dataset("mteb/sib200", l.sib, split="test", token=tok)
load_dataset("AfriMM/AFRICaption", split="train", token=tok)
load_dataset("CohereLabsCommunity/afri-aya", split="train", token=tok)
snapshot_download("CIRAL/ciral", repo_type="dataset", token=tok)
print("done")
