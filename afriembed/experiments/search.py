"""Experiments 11 + 12: local semantic search indexes.

index-audio: folder of recordings -> 10 s windows -> embeddings (radio/oral archive search)
index-docs:  folder of text files   -> chunks     -> embeddings (voice search over documents)
search:      query by text or by a spoken .wav against either index
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import cache_dir
from ..data.audio_utils import TARGET_SR, load_any, slice_window, to_mono_16k, windows, write_wav
from ..utils import timer
from .common import ModelOpts

log = logging.getLogger(__name__)
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}
TEXT_EXT = {".txt", ".md", ".jsonl"}


@dataclass
class Index:
    kind: str  # audio | docs
    model: str
    dim: int
    emb: np.ndarray
    meta: list[dict]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.emb.astype(np.float32))
        with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        (d / "index.json").write_text(
            json.dumps(
                {"kind": self.kind, "model": self.model, "dim": self.dim, "n": len(self.meta)}, indent=2
            )
        )

    @classmethod
    def load(cls, d: Path) -> Index:
        info = json.loads((d / "index.json").read_text())
        emb = np.load(d / "embeddings.npy")
        meta = [
            json.loads(l) for l in (d / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        return cls(info["kind"], info["model"], info["dim"], emb, meta)


def _chunk_text(text: str, max_chars: int = 1200, min_tail_chars: int = 400) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars:]
            cur = p
    if cur:
        chunks.append(cur)
    # a short trailing fragment scores unfairly well on its own; fold it into the previous chunk
    if len(chunks) > 1 and len(chunks[-1]) < min_tail_chars:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def index_audio(
    opts: ModelOpts, folder: Path, out: Path, window_s: float = 10.0, hop_s: float | None = None
) -> Index:
    emb = opts.load(("text", "audio"))
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"No audio files under {folder}")
    tmp = cache_dir() / "windows" / out.name
    items, meta = [], []
    total_s = 0.0
    for f in files:
        y, sr = load_any(f)
        y = to_mono_16k(y, sr)
        total_s += len(y) / TARGET_SR
        for w in windows(len(y), TARGET_SR, window_s, hop_s):
            p = tmp / f"{f.stem}_{w.index:05d}.wav"
            if not p.exists():
                write_wav(p, slice_window(y, w))
            items.append(str(p))
            meta.append({"file": str(f), "start_s": w.start_s, "end_s": w.end_s})
    with timer(f"embed {len(items)} windows ({total_s / 60:.1f} min of audio)"):
        E = emb.encode_audio(items, "none", show_progress=True)
    idx = Index("audio", emb.name, E.shape[1], E, meta)
    idx.save(out)
    log.info("Indexed %d files / %d windows -> %s", len(files), len(items), out)
    return idx


def index_docs(opts: ModelOpts, folder: Path, out: Path, max_chars: int = 1200) -> Index:
    emb = opts.load(("text",))
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in TEXT_EXT)
    if not files:
        raise SystemExit(f"No text files under {folder}")
    texts, meta = [], []
    for f in files:
        if f.suffix == ".jsonl":
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("text") or r.get("content") or ""
                for j, c in enumerate(_chunk_text(t, max_chars)):
                    texts.append(c)
                    meta.append({"file": str(f), "row": i, "chunk": j, "title": r.get("title"), "text": c})
        else:
            for j, c in enumerate(_chunk_text(f.read_text(encoding="utf-8", errors="replace"), max_chars)):
                texts.append(c)
                meta.append({"file": str(f), "chunk": j, "title": f.stem, "text": c})
    with timer(f"embed {len(texts)} chunks"):
        E = emb.encode_text(texts, "document", show_progress=True)
    idx = Index("docs", emb.name, E.shape[1], E, meta)
    idx.save(out)
    log.info("Indexed %d files / %d chunks -> %s", len(files), len(texts), out)
    return idx


class Searcher:
    """Index + embedder loaded once; query many times (used by the CLI and the UI)."""

    def __init__(self, opts: ModelOpts, index_dir: Path, need_audio: bool = False):
        self.idx = Index.load(index_dir)
        need_audio = need_audio or self.idx.kind == "audio"
        self.emb = opts.load(("text", "audio") if need_audio else ("text",))
        self.has_audio = need_audio

    def __call__(self, query: str | None = None, audio_query: Path | None = None, k: int = 10) -> list[dict]:
        if audio_query is not None:
            if not self.has_audio:
                raise SystemExit("Searcher was loaded without the audio tower; reload with need_audio=True")
            y, sr = load_any(audio_query)
            p = cache_dir() / "queries" / f"{Path(audio_query).stem}_16k.wav"
            write_wav(p, to_mono_16k(y, sr))
            q = self.emb.encode_audio([str(p)], "none")
        elif query:
            q = self.emb.encode_text([query], "query")
        else:
            raise SystemExit("Provide --query text or --audio file")
        sims = (q @ self.idx.emb.T)[0]
        top = np.argsort(-sims)[: int(k)]
        return [{"score": float(sims[i]), **self.idx.meta[i]} for i in top]


def search(
    opts: ModelOpts, index_dir: Path, query: str | None = None, audio_query: Path | None = None, k: int = 10
) -> list[dict]:
    return Searcher(opts, index_dir, need_audio=audio_query is not None)(query, audio_query, k)


def launch_ui(opts: ModelOpts, index_dir: Path, port: int = 7860):  # pragma: no cover - interactive
    try:
        import gradio as gr
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "The browser UI needs gradio. Install it with:  uv sync --extra ui   (then rerun with --ui)"
        ) from None

    log.info("Loading index and model once for the UI ...")
    searcher = Searcher(opts, index_dir, need_audio=True)

    def _run(text, audio, k):
        text = (text or "").strip()
        if not text and not audio:
            return []
        # a spoken query wins if both are given
        hits = searcher(query=None if audio else text, audio_query=Path(audio) if audio else None, k=int(k))
        return [
            [
                round(h["score"], 3),
                h.get("title") or h.get("file"),
                h.get("start_s"),
                h.get("end_s"),
                (h.get("text") or "")[:300],
            ]
            for h in hits
        ]

    with gr.Blocks(title="afriembed search") as demo:
        gr.Markdown(
            f"### afriembed search over `{index_dir}`\n"
            "Type a question in any language, or press the microphone and ask it aloud. "
            f"Model: `{opts.model}`" + (f", adapter: `{opts.adapter}`" if opts.adapter else "")
        )
        text = gr.Textbox(label="Text query", placeholder="yaushe ake shuka masara?")
        audio = gr.Audio(
            label="Spoken query (record or upload)", type="filepath", sources=["microphone", "upload"]
        )
        k = gr.Slider(1, 50, value=5, step=1, label="top-k")
        out = gr.Dataframe(headers=["score", "title", "start_s", "end_s", "text"], label="results", wrap=True)
        btn = gr.Button("Search", variant="primary")
        btn.click(_run, [text, audio, k], out)
        text.submit(_run, [text, audio, k], out)
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
