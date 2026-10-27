"""FastAPI backend for the search UI: loads the index and model once, serves the built web app.

    uv run afriembed search indexes/demo --ui --adapter artifacts/audio-lora

Endpoints
  GET  /api/info                      index stats, model, adapter, language counts
  GET  /api/search?q=...&k=10         typed query
  POST /api/search/audio  (wav file)  spoken query; the browser records PCM and encodes WAV itself
"""

import logging
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

from .experiments.common import ModelOpts
from .experiments.search import Searcher

log = logging.getLogger(__name__)
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
META_RE = re.compile(r"^\*([^·*]+)·([^·*]+)·\s*(\S+)\*\s*$", re.M)
LANG_CODES = {
    "hausa": "hau", "igbo": "ibo", "yoruba": "yor", "nigerian pidgin": "pcm", "swahili": "swa",
    "english": "eng", "amharic": "amh", "somali": "som", "xhosa": "xho", "french": "fra",
}  # fmt: skip


LANG_NAMES = {v: k.title() if k != "nigerian pidgin" else "Nigerian Pidgin" for k, v in LANG_CODES.items()}


def _doc_lang(h: dict) -> tuple[str | None, str | None, str | None]:
    """(language name, category, url) from the article's metadata line, else from its path."""
    text = h.get("text") or ""
    m = META_RE.search(text)
    if m:
        lang, cat, url = (x.strip() for x in m.groups())
        return lang, cat, url
    parts = Path(h.get("file", "")).parts
    stem = Path(h.get("file", "")).stem
    if "news" in parts:
        code = parts[parts.index("news") + 1]
        return LANG_NAMES.get(code, code), None, None
    if stem.startswith("hausa_"):
        return "Hausa", None, None
    if stem.startswith("english_"):
        return "English", None, None
    return None, None, None


def _enrich(h: dict, title_by_file: dict[str, str] | None = None) -> dict:
    text = h.get("text") or ""
    lang, cat, url = _doc_lang(h)
    file = h.get("file", "")
    title = (title_by_file or {}).get(file) or h.get("title") or Path(file).stem
    text = META_RE.sub("", text)
    text = re.sub(r"^#\s*.*\n+", "", text.strip())  # drop the markdown title from the snippet
    return {
        "score": round(float(h["score"]), 4),
        "title": title,
        "lang": lang,
        "lang_code": LANG_CODES.get((lang or "").lower(), (lang or "").lower()[:3] or None),
        "category": cat,
        "url": url,
        "file": file,
        "chunk": h.get("chunk"),
        "start_s": h.get("start_s"),
        "end_s": h.get("end_s"),
        "snippet": text[:600],
    }


def build_app(opts: ModelOpts, index_dir: Path):
    from fastapi import FastAPI, File, Query, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    searcher = Searcher(opts, index_dir, need_audio=True)
    idx = searcher.idx
    # per-document title (from the first chunk's "# heading"), category and url, so later chunks inherit them
    title_by_file: dict[str, str] = {}
    meta_by_file: dict[str, tuple] = {}
    for m in idx.meta:
        f = m.get("file", "")
        t = (m.get("text") or "").lstrip("\ufeff")
        if f not in title_by_file and t.startswith("# "):
            title_by_file[f] = t.splitlines()[0][2:].strip()
        if f not in meta_by_file or meta_by_file[f][0] is None:
            meta_by_file[f] = _doc_lang(m)
    langs = Counter(v[0] for v in meta_by_file.values() if v[0])
    n_docs = len(meta_by_file)

    def enrich(h: dict) -> dict:
        e = _enrich(h, title_by_file)
        lang, cat, url = meta_by_file.get(h.get("file", ""), (None, None, None))
        e["lang"] = e["lang"] or lang
        e["lang_code"] = LANG_CODES.get((e["lang"] or "").lower(), e["lang_code"])
        e["category"] = e["category"] or cat
        e["url"] = e["url"] or url
        return e

    app = FastAPI(title="afriembed search", docs_url="/api/docs", redoc_url=None)

    @app.get("/api/info")
    def info():
        return {
            "index": str(index_dir),
            "kind": idx.kind,
            "model": opts.model,
            "adapter": opts.adapter,
            "dim": idx.dim,
            "n_chunks": len(idx.meta),
            "n_docs": n_docs,
            "languages": dict(langs.most_common()),
            "device": opts.device or "auto",
            "reading_sentences": len(sentences),
        }

    def dedupe(hits: list[dict], k: int) -> list[dict]:
        """One row per document: keep its best-scoring chunk."""
        seen: set[str] = set()
        out = []
        for h in hits:
            f = h.get("file", "")
            if f in seen:
                continue
            seen.add(f)
            out.append(h)
            if len(out) >= k:
                break
        return out

    @app.get("/api/search")
    def search_text(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=50)):
        t0 = time.perf_counter()
        hits = dedupe(searcher(query=q, k=k * 4), k)
        return {
            "query": q,
            "mode": "text",
            "ms": round((time.perf_counter() - t0) * 1000),
            "hits": [enrich(h) for h in hits],
        }

    @app.post("/api/search/audio")
    async def search_audio(file: UploadFile = File(...), k: int = Query(10, ge=1, le=50)):
        t0 = time.perf_counter()
        suffix = Path(file.filename or "q.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(await file.read())
            p = Path(f.name)
        try:
            hits = dedupe(searcher(audio_query=p, k=k * 4), k)
        finally:
            p.unlink(missing_ok=True)
        return {
            "query": file.filename,
            "mode": "audio",
            "ms": round((time.perf_counter() - t0) * 1000),
            "hits": [enrich(h) for h in hits],
        }

    # ---- product 7: reading game -------------------------------------------------------------
    sentences: list[dict] = []
    sent_emb = None
    reading_csv = Path("demo/voicenotes/transcripts.csv")
    if reading_csv.exists():
        import csv

        with open(reading_csv, encoding="utf-8") as f:
            sentences = [r for r in csv.DictReader(f) if r.get("text")]
        if sentences:
            sent_emb = searcher.emb.encode_text([r["text"] for r in sentences], "document")
            log.info("reading game: %d sentences from %s", len(sentences), reading_csv)

    @app.get("/api/read/sentence")
    def read_sentence(lang: str | None = None):
        import random

        pool = [i for i, r in enumerate(sentences) if not lang or r.get("lang") == lang]
        if not pool:
            return {"error": "no sentences loaded (build demo/voicenotes first)"}
        i = random.choice(pool)
        return {"id": i, "text": sentences[i]["text"], "lang": sentences[i].get("lang")}

    @app.post("/api/read/score")
    async def read_score(file: UploadFile = File(...), id: int = Query(...)):
        """Score a read-aloud attempt: how clearly the recording points at the target sentence among all others."""
        import numpy as np

        if sent_emb is None or not (0 <= id < len(sentences)):
            return {"error": "bad sentence id"}
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(await file.read())
            p = Path(f.name)
        try:
            from .data.audio_utils import load_any, to_mono_16k, write_wav

            y, sr = load_any(p)
            write_wav(p, to_mono_16k(y, sr))
            a = searcher.emb.encode_audio([str(p)], "none")[0]
        finally:
            p.unlink(missing_ok=True)
        sims = sent_emb @ a
        own = float(sims[id])
        others = np.delete(sims, id)
        best_other = float(others.max())
        rank = int((sims > own).sum()) + 1
        margin = own - best_other
        # 0-100: similarity scaled over the typical 0.35..0.75 range, with a margin bonus and a rank penalty
        score = 100 * min(1.0, max(0.0, (own - 0.35) / 0.40))
        score = score * (0.6 + 0.4 * min(1.0, max(0.0, margin / 0.08))) if rank == 1 else score * 0.5
        nearest = int(np.argmax(sims))
        verdict = "clear" if rank == 1 and margin >= 0.04 else ("close" if rank <= 3 else "unclear")
        return {
            "score": round(score),
            "verdict": verdict,
            "similarity": round(own, 3),
            "margin": round(margin, 3),
            "rank": rank,
            "pool": len(sentences),
            "sounded_like": sentences[nearest]["text"] if nearest != id else None,
        }

    @app.get("/api/audio")
    def audio_window(file: str, start: float | None = None, end: float | None = None):
        """Stream one indexed recording (or a [start, end] window of it) as 16 kHz WAV, for the player."""
        import io

        import soundfile as sf
        from fastapi import HTTPException
        from fastapi.responses import Response

        from .data.audio_utils import TARGET_SR, load_any, to_mono_16k

        p = Path(file)
        allowed = {str(Path(m.get("file", "")).resolve()) for m in idx.meta}
        if str(p.resolve()) not in allowed:
            raise HTTPException(404, "not an indexed file")
        y, sr = load_any(p)
        y = to_mono_16k(y, sr)
        a = int((start or 0) * TARGET_SR)
        b = int(end * TARGET_SR) if end else len(y)
        buf = io.BytesIO()
        sf.write(buf, y[a:b], TARGET_SR, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            target = WEB_DIST / path
            if path and target.is_file():
                return FileResponse(target)
            return FileResponse(WEB_DIST / "index.html")
    else:
        log.warning("web/dist not found; build it with `cd web && pnpm install && pnpm build`. API only.")

    return app


def serve(opts: ModelOpts, index_dir: Path, host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    app = build_app(opts, index_dir)
    log.info("UI at http://localhost:%d  (API docs at /api/docs)", port)
    try:  # phones on the same Wi-Fi can open the LAN address and "add to home screen"
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        log.info("on your phone (same network): http://%s:%d", s.getsockname()[0], port)
        s.close()
    except Exception:
        pass
    uvicorn.run(app, host=host, port=port, log_level="warning")
