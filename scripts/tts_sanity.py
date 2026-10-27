"""How intelligible is each MMS-TTS voice to the audio tower? R@1 of synthesised utterance -> its own text.

Used to read the spoken-intent / spoken-QA numbers correctly: a low score here means the synthetic voice, not the
retrieval model, is the bottleneck for that language.

Usage: uv run python scripts/tts_sanity.py --langs hau,yor,swa,ibo --n 200
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hau,yor,swa,ibo")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adapter", default="artifacts/audio-lora")
    a = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    from afriembed.data.tts import synthesize_all
    from afriembed.experiments.common import ModelOpts
    from afriembed.experiments.spoken_intent import load_split
    from afriembed.metrics import retrieval_report, sim_matrix
    from afriembed.utils import save_result

    emb = ModelOpts(model="embeddinggemma-2", adapter=a.adapter).load(("text", "audio"))
    out = {}
    for lang in a.langs.split(","):
        texts, _ = load_split(lang, "test")
        texts = texts[: a.n]
        wavs = synthesize_all(texts, lang, namespace="tts-injongo")
        A = emb.encode_audio([str(w) for w in wavs], "none")
        T = emb.encode_text(texts, "document")
        r = retrieval_report(sim_matrix(A, T), np.arange(len(texts)))
        out[lang] = {
            "n": len(texts),
            "tts_audio->own_text": r,
            "mean_own_sim": float(np.mean(np.sum(A * T, axis=1))),
        }
        print(
            lang,
            "R@1",
            round(r["recall@1"], 3),
            "R@5",
            round(r["recall@5"], 3),
            "own-sim",
            round(out[lang]["mean_own_sim"], 3),
        )
    save_result("tts-sanity", emb.name, {"per_lang": out, "model_info": emb.describe()}, tag="adapter")
    print(json.dumps({k: round(v["tts_audio->own_text"]["recall@1"], 3) for k, v in out.items()}))


if __name__ == "__main__":
    main()
