"""Product 3: spoken question answering over documents (Spoken AfriQA).

AfriQA (Ogundepo et al., 2023) gives questions in African languages with their English Wikipedia gold
passage. We pool every gold passage of the chosen languages as the retrieval corpus, then ask each
question three ways: typed in the African language, typed in English (translated_question, the upper
bound), and spoken in the African language (MMS-TTS of the question, embedded by the audio tower with no
transcription). Metric: rank of the gold passage. Optionally a small Gemma answers from the top-k
passages of the spoken query and is scored with token F1 against the English gold answer.

Data files come straight from the AfriQA GitHub repo (the HF loader script is no longer supported).
"""

from __future__ import annotations

import json
import logging
import urllib.request

import numpy as np

from ..config import cache_dir
from ..metrics import retrieval_report, sim_matrix
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

RAW = "https://github.com/masakhane-io/afriqa/raw/main/data/gold_passages/{lang}/gold_span_passages.afriqa.{lang}.{pivot}.{split}.json"
PIVOT = {
    "hau": "en",
    "ibo": "en",
    "yor": "en",
    "swa": "en",
    "kin": "en",
    "twi": "en",
    "zul": "en",
    "bem": "en",
    "fon": "fr",
    "wol": "fr",
}


def load_gold(lang: str, split: str = "test") -> list[dict]:
    p = cache_dir() / "afriqa" / f"{lang}.gold.{split}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW.format(lang=lang, pivot=PIVOT[lang], split=split), p)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for r in rows:
        ctx = (r.get("context") or "").strip()
        if not ctx or not r.get("question_lang"):
            continue
        ans = r.get("answer_pivot") or {}
        out.append(
            {
                "id": f"{lang}-{r['id']}",
                "lang": lang,
                "question": r["question_lang"].strip(),
                "question_en": (r.get("question_translated") or "").strip(),
                "answer_en": (ans.get("text") or [None])[0],
                "answer_lang": r.get("answer_lang"),
                "title": r.get("title"),
                "context": ctx,
            }
        )
    log.info("AfriQA %s/%s: %d questions with gold passages", lang, split, len(out))
    return out


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test",
    max_questions: int | None = None,
    generator_id: str | None = None,
    n_gen: int = 50,
    k: int = 3,
    tag: str | None = None,
) -> dict:
    from ..data.tts import TTS_MODELS, synthesize_all

    emb = opts.load(("text", "audio"))
    data = {lang: load_gold(lang, split)[:max_questions] for lang in languages}
    # shared corpus: unique passages across all languages (natural distractors)
    passages: list[str] = []
    pid: dict[str, int] = {}
    for rows in data.values():
        for r in rows:
            if r["context"] not in pid:
                pid[r["context"]] = len(passages)
                passages.append(r["context"])
    with timer(f"embed {len(passages)} passages"):
        P = emb.encode_text(passages, "document", show_progress=True)

    gen = None
    if generator_id:
        from .rag import PROMPT, Generator, token_f1

        gen = Generator(generator_id)

    per_lang = {}
    for lang, rows in data.items():
        if lang not in TTS_MODELS:
            log.warning("%s: no TTS voice; typed only", lang)
        gold = np.array([pid[r["context"]] for r in rows])
        res = {"n": len(rows)}
        Q_typed = emb.encode_text([r["question"] for r in rows], "query")
        res["typed_lang"] = retrieval_report(sim_matrix(Q_typed, P), gold)
        if all(r["question_en"] for r in rows):
            Q_en = emb.encode_text([r["question_en"] for r in rows], "query")
            res["typed_english"] = retrieval_report(sim_matrix(Q_en, P), gold)
        if lang in TTS_MODELS:
            with timer(f"{lang}: synthesise {len(rows)} spoken questions"):
                wavs = synthesize_all([r["question"] for r in rows], lang, namespace="tts-afriqa")
            with timer(f"{lang}: embed {len(wavs)} spoken questions"):
                A = emb.encode_audio([str(w) for w in wavs], "none", show_progress=True)
            S = sim_matrix(A, P)
            res["spoken_lang"] = retrieval_report(S, gold)
            # audio question vs its own typed question: how much the audio tower loses vs text
            res["spoken_vs_typed_agreement@1"] = float(
                np.mean(S.argmax(1) == sim_matrix(Q_typed, P).argmax(1))
            )
            if gen is not None:
                f1s, ems_in_ctx, samples = [], [], []
                for i in range(min(n_gen, len(rows))):
                    top = np.argsort(-S[i])[:k]
                    ctx = "\n\n".join(f"[{j + 1}] {passages[t]}" for j, t in enumerate(top))
                    ans = gen(PROMPT.format(context=ctx, question=rows[i]["question"]))
                    golds = [g for g in (rows[i]["answer_en"], rows[i]["answer_lang"]) if g]
                    f1s.append(token_f1(ans, golds))
                    ems_in_ctx.append(float(gold[i] in top))
                    if len(samples) < 8:
                        samples.append(
                            {
                                "q": rows[i]["question"],
                                "gold": golds,
                                "answer": ans,
                                "gold_in_top_k": gold[i] in top,
                            }
                        )
                res["generation"] = {
                    "generator": generator_id,
                    "n": len(f1s),
                    "k": k,
                    "gold_passage_in_top_k": float(np.mean(ems_in_ctx)),
                    "token_f1": float(np.mean(f1s)),
                    "samples": samples,
                }
        per_lang[lang] = res
        save_result(  # partial save so a crash in a later language keeps this one
            "spoken-qa",
            emb.name,
            {"partial": True, "n_passages": len(passages), "per_lang": per_lang},
            tag=opts.full_tag(tag),
        )
        log.info(
            "%s: gold passage R@1 typed %.3f | english %.3f | spoken %s (pool %d)",
            lang,
            res["typed_lang"]["recall@1"],
            res.get("typed_english", {}).get("recall@1", float("nan")),
            f"{res['spoken_lang']['recall@1']:.3f}" if "spoken_lang" in res else "-",
            len(passages),
        )
    payload = {
        "dataset": "AfriQA gold passages (masakhane-io/afriqa)",
        "split": split,
        "n_passages": len(passages),
        "model_info": emb.describe(),
        "per_lang": per_lang,
        "mean_recall@1": {
            key: float(np.mean([v[key]["recall@1"] for v in per_lang.values() if key in v]))
            for key in ("typed_lang", "typed_english", "spoken_lang")
            if any(key in v for v in per_lang.values())
        },
    }
    save_result("spoken-qa", emb.name, payload, tag=opts.full_tag(tag))
    return payload
