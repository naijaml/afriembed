"""Experiment 10: on-device RAG in African languages — afriembed retrieval + a small Gemma generator.

Answer quality is scored with token-F1 / exact match on a JSONL of {"question", "answers": [...]} rows
(e.g. exported from AfriQA), using an afriembed docs index built with `afriembed index-docs`.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import hf_token
from ..utils import save_result, timer
from .common import ModelOpts
from .search import Index

log = logging.getLogger(__name__)

DEFAULT_GENERATOR = "google/gemma-3-1b-it"  # swap for a Gemma 4 instruct id when available locally

PROMPT = """You answer questions using only the provided context. Reply in the same language as the question, in one short sentence.

Context:
{context}

Question: {question}
Answer:"""


class Generator:
    def __init__(
        self, model_id: str = DEFAULT_GENERATOR, device: str | None = None, max_new_tokens: int = 64
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id, token=hf_token())
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", token=hf_token()
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        import torch

        msgs = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        n_in = enc["input_ids"].shape[1]
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


class RAG:
    def __init__(self, opts: ModelOpts, index_dir: Path, generator: Generator | None, k: int = 4):
        self.idx = Index.load(index_dir)
        self.emb = opts.load(("text",))
        self.gen = generator
        self.k = k

    def retrieve(self, question: str) -> list[dict]:
        q = self.emb.encode_text([question], "query")
        sims = (q @ self.idx.emb.T)[0]
        top = np.argsort(-sims)[: self.k]
        return [{"score": float(sims[i]), **self.idx.meta[i]} for i in top]

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        ctx = "\n\n".join(f"[{i + 1}] {h.get('text', '')}" for i, h in enumerate(hits))
        ans = self.gen(PROMPT.format(context=ctx, question=question)) if self.gen else None
        return {"question": question, "answer": ans, "contexts": hits}


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split()
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))


def evaluate(
    opts: ModelOpts,
    index_dir: Path,
    qa_jsonl: Path,
    generator_id: str | None = DEFAULT_GENERATOR,
    k: int = 4,
    max_questions: int | None = None,
    tag: str | None = None,
) -> dict:
    rows = [json.loads(l) for l in Path(qa_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:max_questions] if max_questions else rows
    gen = Generator(generator_id) if generator_id else None
    rag = RAG(opts, index_dir, gen, k=k)
    f1s, ems, hit_at_k, outputs = [], [], [], []
    with timer(f"RAG over {len(rows)} questions"):
        for r in rows:
            golds = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
            res = rag.answer(r["question"])
            ctx_text = " ".join(_norm(h.get("text", "")) for h in res["contexts"])
            hit_at_k.append(float(any(_norm(g) and _norm(g) in ctx_text for g in golds)))
            if res["answer"] is not None:
                f1s.append(token_f1(res["answer"], golds))
                ems.append(exact_match(res["answer"], golds))
            outputs.append({**res, "golds": golds})
    payload = {
        "index": str(index_dir),
        "qa_file": str(qa_jsonl),
        "generator": generator_id,
        "k": k,
        "n": len(rows),
        "answer_in_context@k": float(np.mean(hit_at_k)) if hit_at_k else None,
        "token_f1": float(np.mean(f1s)) if f1s else None,
        "exact_match": float(np.mean(ems)) if ems else None,
        "model_info": rag.emb.describe(),
        "samples": outputs[:20],
    }
    save_result("rag", rag.emb.name, payload, tag=opts.full_tag(tag))
    return payload


def export_afriqa(lang: str, out: Path, split: str = "test") -> Path:
    """Best-effort export of AfriQA (masakhane/afriqa) to the JSONL format above; column names are probed at runtime."""
    from datasets import load_dataset

    ds = load_dataset("masakhane/afriqa", lang, split=split, token=hf_token())
    cols = ds.column_names
    qcol = next(
        (c for c in ("question_translated", "translated_question", "question_lang", "question") if c in cols),
        None,
    )
    acol = next((c for c in ("answers", "answer_pivot", "answer_lang", "answer") if c in cols), None)
    if not qcol or not acol:
        raise RuntimeError(f"Unrecognised AfriQA columns: {cols}")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in ds:
            ans = r[acol]
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)
                except Exception:
                    ans = [ans]
            if isinstance(ans, dict):
                ans = ans.get("text") or ans.get("answers") or list(ans.values())[0]
            if not isinstance(ans, list):
                ans = [str(ans)]
            ans = [str(a) for a in ans if a]
            if not ans:
                continue
            f.write(
                json.dumps({"question": r[qcol], "answers": ans, "lang": lang}, ensure_ascii=False) + "\n"
            )
            n += 1
    log.info("Exported %d AfriQA questions (%s/%s) -> %s", n, lang, split, out)
    return out
