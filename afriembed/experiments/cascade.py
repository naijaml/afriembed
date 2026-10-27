"""Cascade baseline: ASR (Meta MMS-1B-all) -> text embedding -> retrieval, scored on the same FLEURS pools as the
direct audio->text path, with word error rate of the transcripts."""

from __future__ import annotations

import logging
import re

import numpy as np

from ..data.fleurs import load_fleurs
from ..languages import with_fleurs
from ..metrics import retrieval_report, sim_matrix
from ..models.embedder import truncate
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

MMS_ID = "facebook/mms-1b-all"
# ISO 639-3 codes MMS uses where ours differ
MMS_LANG = {"gaz": "orm"}  # MMS codes Oromo as orm; every other code in our registry matches MMS


class MMS:
    def __init__(self, device: str):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.device = device
        self.processor = AutoProcessor.from_pretrained(MMS_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MMS_ID, torch_dtype=torch.float32).to(device).eval()
        self.lang = None

    def set_lang(self, code: str) -> bool:
        lang = MMS_LANG.get(code, code)
        try:
            self.processor.tokenizer.set_target_lang(lang)
            self.model.load_adapter(lang)
            self.lang = lang
            return True
        except Exception as e:
            log.warning("MMS has no adapter for %s (%s): %s", code, lang, str(e)[:120])
            return False

    def transcribe(self, wavs: list[str], batch_size: int = 8) -> list[str]:
        import soundfile as sf
        import torch

        out = []
        for i in range(0, len(wavs), batch_size):
            arrays = [sf.read(w, dtype="float32")[0] for w in wavs[i : i + batch_size]]
            inputs = self.processor(arrays, sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.get("attention_mask", None).to(self.device)
                    if inputs.get("attention_mask") is not None
                    else None,
                ).logits
            ids = torch.argmax(logits, dim=-1)
            out += self.processor.batch_decode(ids)
        return out


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def run(
    opts: ModelOpts,
    languages: list[str],
    split: str = "test",
    max_items: int | None = 300,
    text_task: str = "document",
    query_task: str = "query",
    dims=(768, 128),
    tag: str | None = None,
) -> dict:
    import jiwer

    emb = opts.load(("text", "audio"))
    asr = MMS(emb.extra.get("device", "cuda"))
    per_lang = {}
    for lang in with_fleurs(languages):
        if not asr.set_lang(lang.code):
            per_lang[lang.code] = {"error": "no MMS adapter"}
            continue
        utts = load_fleurs(lang, split=split, max_items=max_items)
        refs = [u.text for u in utts]
        with timer(f"{lang.code}: MMS transcribe {len(utts)}"):
            hyps = asr.transcribe([str(u.wav) for u in utts])
        wer = float(jiwer.wer([_norm(r) for r in refs], [_norm(h) or "-" for h in hyps]))
        T = emb.encode_text(refs, text_task)
        H = emb.encode_text(hyps, query_task)
        with timer(f"{lang.code}: encode {len(utts)} audio"):
            A = emb.encode_audio([str(u.wav) for u in utts], "none", show_progress=True)
        targets = np.arange(len(utts))
        res = {"n": len(utts), "wer": wer, "dims": {}}
        for d in dims:
            res["dims"][str(d)] = {
                "cascade_asr_text->text": retrieval_report(
                    sim_matrix(truncate(H, d), truncate(T, d)), targets
                ),
                "direct_audio->text": retrieval_report(sim_matrix(truncate(A, d), truncate(T, d)), targets),
            }
        res["sample"] = [{"ref": refs[i], "hyp": hyps[i]} for i in range(min(3, len(utts)))]
        per_lang[lang.code] = res
        log.info(
            "%s WER=%.3f  cascade R@1=%.3f  direct R@1=%.3f (pool %d)",
            lang.code,
            wer,
            res["dims"]["768"]["cascade_asr_text->text"]["recall@1"],
            res["dims"]["768"]["direct_audio->text"]["recall@1"],
            len(utts),
        )
    ok = {k: v for k, v in per_lang.items() if "dims" in v}
    summary = (
        {
            "mean_wer": float(np.mean([v["wer"] for v in ok.values()])),
            "mean_cascade_recall@1": float(
                np.mean([v["dims"]["768"]["cascade_asr_text->text"]["recall@1"] for v in ok.values()])
            ),
            "mean_direct_recall@1": float(
                np.mean([v["dims"]["768"]["direct_audio->text"]["recall@1"] for v in ok.values()])
            ),
        }
        if ok
        else {}
    )
    payload = {
        "asr": MMS_ID,
        "split": split,
        "text_task": text_task,
        "query_task": query_task,
        "model_info": emb.describe(),
        "per_language": per_lang,
        "summary": summary,
    }
    save_result("cascade", emb.name, payload, tag=opts.full_tag(tag))
    return payload
