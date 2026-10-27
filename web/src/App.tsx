import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, ExternalLink, Loader2, Mic, Square, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { WavRecorder } from "@/lib/recorder";
import { cn, highlight } from "@/lib/utils";

type Hit = {
  score: number;
  title: string;
  lang: string | null;
  lang_code: string | null;
  category: string | null;
  url: string | null;
  file: string | null;
  start_s: number | null;
  end_s: number | null;
  snippet: string;
};
type Result = { query: string; mode: "text" | "audio"; ms: number; hits: Hit[] };
type Info = {
  kind: "docs" | "audio";
  model: string;
  adapter: string | null;
  n_docs: number;
  n_chunks: number;
  dim: number;
  languages: Record<string, number>;
  device: string;
  reading_sentences: number;
};

const EXAMPLES: { q: string; lang: string }[] = [
  { q: "yaushe ake shuka masara?", lang: "hau" },
  { q: "me ya sa mutum ke jin an danne shi a cikin barci?", lang: "hau" },
  { q: "parents of the conjoined twins born in Ilorin", lang: "eng" },
  { q: "woman who put Gorilla Glue in her hair went to hospital", lang: "eng" },
  { q: "kwa nini Wakenya hawapendi deni jipya kutoka IMF?", lang: "swa" },
  { q: "Buhari ati Osinbajo gba abere ajesara covid", lang: "yor" },
  { q: "children dey die for one Abuja village because hospital no dey", lang: "pcm" },
  { q: "Hushpuppi", lang: "yor" },
];
const WEAK = 0.6;

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [tab, setTab] = useState<"search" | "read">("search");
  const [q, setQ] = useState("");
  const [k, setK] = useState(10);
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [rec, setRec] = useState<{ on: boolean; secs: number; level: number }>({ on: false, secs: 0, level: 0 });
  const recorder = useRef<WavRecorder | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/info")
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setErr("Backend not reachable. Start it with: uv run afriembed search indexes/demo --ui"));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") {
        setRes(null);
        setQ("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const runText = useCallback(
    async (text: string) => {
      const t = text.trim();
      if (!t) return;
      setBusy(true);
      setErr(null);
      try {
        const r = await fetch(`/api/search?q=${encodeURIComponent(t)}&k=${k}`);
        if (!r.ok) throw new Error(await r.text());
        setRes(await r.json());
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [k],
  );

  const runAudio = useCallback(
    async (blob: Blob) => {
      setBusy(true);
      setErr(null);
      try {
        const fd = new FormData();
        fd.append("file", blob, "query.wav");
        const r = await fetch(`/api/search/audio?k=${k}`, { method: "POST", body: fd });
        if (!r.ok) throw new Error(await r.text());
        setRes(await r.json());
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [k],
  );

  const toggleRec = useCallback(async () => {
    if (rec.on) {
      const blob = await recorder.current!.stop();
      setRec({ on: false, secs: 0, level: 0 });
      recorder.current = null;
      if (blob.size > 44 + 16000) await runAudio(blob); // ignore sub-half-second taps
      return;
    }
    try {
      const r = new WavRecorder();
      await r.start();
      recorder.current = r;
      setQ("");
      setRec({ on: true, secs: 0, level: 0 });
    } catch {
      setErr("Microphone permission was refused, or no microphone was found.");
    }
  }, [rec.on, runAudio]);

  useEffect(() => {
    if (!rec.on) return;
    const id = setInterval(() => {
      const r = recorder.current;
      if (r) setRec((s) => ({ ...s, secs: r.seconds, level: r.level }));
    }, 80);
    return () => clearInterval(id);
  }, [rec.on]);

  const langs = info ? Object.entries(info.languages) : [];

  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-line">
        <div className="mx-auto max-w-3xl px-5 py-4 flex items-center justify-between gap-4">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-sm tracking-tight">afriembed</span>
            <span className="text-ink-3 text-sm hidden sm:inline">voice &amp; text search · African languages</span>
          </div>
          {info && info.reading_sentences > 0 && (
            <nav className="flex gap-1 font-mono text-xs">
              {(["search", "read"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  className={cn(
                    "rounded px-2 py-1 transition-colors",
                    tab === t ? "bg-ink text-paper" : "text-ink-2 hover:bg-paper-2",
                  )}
                >
                  {t === "search" ? "search" : "read aloud"}
                </button>
              ))}
            </nav>
          )}
          {info && (
            <div className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
              <span className="hidden md:inline">{info.model}</span>
              {info.adapter && <Badge variant="accent">adapter</Badge>}
              <span>{info.dim}d</span>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl px-5 flex-1">
        {tab === "read" && <ReadAloud />}
        {tab === "search" && (
        <>
        <section className="pt-10 pb-6">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight [text-wrap:balance]">
            {info?.kind === "audio" ? (
              <>
                Search your voice notes.
                <span className="text-ink-3 font-normal"> Say what you remember, or type it.</span>
              </>
            ) : (
              <>
                Ask in Hausa, Yoruba, Igbo, Swahili, Pidgin or English.
                <span className="text-ink-3 font-normal"> Type it or say it.</span>
              </>
            )}
          </h1>
          {info && (
            <p className="mt-3 font-mono text-xs text-ink-3">
              {info.n_docs} {info.kind === "audio" ? "recordings" : "documents"} · {info.n_chunks}{" "}
              {info.kind === "audio" ? "windows" : "chunks"}
              {langs.length > 0 && " · "}
              {langs.map(([l, n], i) => (
                <span key={l}>
                  {i > 0 && " · "}
                  {l.toLowerCase()} {n}
                </span>
              ))}
            </p>
          )}
        </section>

        <form
          className="flex gap-2 items-stretch"
          onSubmit={(e) => {
            e.preventDefault();
            runText(q);
          }}
        >
          <div className="relative flex-1 min-w-0">
            <Input
              ref={inputRef}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={rec.on ? "listening…" : "yaushe ake shuka masara?   ( / to focus )"}
              disabled={rec.on}
              autoFocus
              spellCheck={false}
              className={cn("pr-10", rec.on && "opacity-50")}
            />
            {q && !rec.on && (
              <button
                type="button"
                aria-label="clear"
                onClick={() => {
                  setQ("");
                  inputRef.current?.focus();
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-3 hover:text-ink"
              >
                <X className="size-4" />
              </button>
            )}
          </div>
          <Button
            type="button"
            size="iconLg"
            variant={rec.on ? "accent" : "outline"}
            onClick={toggleRec}
            aria-pressed={rec.on}
            aria-label={rec.on ? "stop and search" : "speak your query"}
            className={cn(rec.on && "recording")}
            disabled={busy}
          >
            {rec.on ? <Square /> : <Mic />}
          </Button>
          <Button type="submit" size="lg" disabled={busy || rec.on || !q.trim()} className="px-5">
            {busy ? <Loader2 className="animate-spin" /> : <ArrowRight />}
            <span className="hidden sm:inline">Search</span>
          </Button>
        </form>

        {rec.on && (
          <div className="mt-3 flex items-center gap-3 font-mono text-xs text-ink-2">
            <span className="inline-block size-2 rounded-full bg-accent animate-pulse" />
            recording {rec.secs.toFixed(1)}s
            <div className="h-1.5 flex-1 rounded bg-paper-2 overflow-hidden">
              <div className="h-full bg-accent transition-[width] duration-75" style={{ width: `${rec.level * 100}%` }} />
            </div>
            <span className="text-ink-3">press the square to search</span>
          </div>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {info?.kind !== "audio" && <span className="font-mono text-[11px] text-ink-3 mr-1">try</span>}
          {(info?.kind === "audio" ? [] : EXAMPLES).map((e) => (
            <button
              key={e.q}
              type="button"
              onClick={() => {
                setQ(e.q);
                runText(e.q);
              }}
              className="group inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-xs hover:border-ink-2 hover:bg-paper-2 transition-colors"
            >
              <span className="font-mono text-[10px] text-ink-3 group-hover:text-accent">{e.lang}</span>
              <span className="truncate max-w-[220px]">{e.q}</span>
            </button>
          ))}
          <label className="ml-auto flex items-center gap-2 font-mono text-[11px] text-ink-3">
            top
            <select
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="bg-transparent border border-line rounded px-1 py-0.5 text-ink"
            >
              {[5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>

        {err && (
          <p className="mt-6 rounded-md border border-accent/40 bg-accent/10 px-4 py-3 text-sm text-ink">{err}</p>
        )}

        {res && (
          <section className="mt-8 pb-16">
            <div className="flex items-baseline justify-between gap-4 border-b border-line pb-2 font-mono text-[11px] text-ink-3">
              <span>
                {res.mode === "audio" ? "spoken query" : <>“{res.query}”</>}
              </span>
              <span>
                {res.hits.length} hits · {res.ms} ms
              </span>
            </div>
            <ol className="divide-y divide-line">
              {res.hits.map((h, i) => (
                <HitRow key={`${h.file}-${i}`} h={h} i={i} query={res.mode === "text" ? res.query : ""} />
              ))}
            </ol>
          </section>
        )}

        {!res && !err && (
          <p className="mt-10 text-sm text-ink-3 max-w-prose">
            Searches a local index with EmbeddingGemma 2 and a 10 MB speech adapter, entirely on this machine. Spoken
            queries are recorded in the browser as 16 kHz WAV and embedded directly, with no transcription step.
          </p>
        )}
        </>
        )}
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-3xl px-5 py-3 font-mono text-[11px] text-ink-3 flex justify-between">
          <span>{info?.kind === "audio" ? "index: recordings, 10 s windows" : "corpus: MasakhaNEWS (BBC) + demo guides"}</span>
          <span>{info?.device}</span>
        </div>
      </footer>
    </div>
  );
}

type Sentence = { id: number; text: string; lang: string | null };
type ReadResult = {
  score: number;
  verdict: "clear" | "close" | "unclear";
  similarity: number;
  margin: number;
  rank: number;
  pool: number;
  sounded_like: string | null;
};

function ReadAloud() {
  const [sent, setSent] = useState<Sentence | null>(null);
  const [rec, setRec] = useState<{ on: boolean; secs: number; level: number }>({ on: false, secs: 0, level: 0 });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ReadResult | null>(null);
  const [history, setHistory] = useState<{ text: string; score: number; verdict: string }[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const recorder = useRef<WavRecorder | null>(null);

  const next = useCallback(async () => {
    setResult(null);
    setErr(null);
    const r = await fetch("/api/read/sentence");
    const j = await r.json();
    if (j.error) setErr(j.error);
    else setSent(j);
  }, []);
  useEffect(() => {
    next();
  }, [next]);

  useEffect(() => {
    if (!rec.on) return;
    const id = setInterval(() => {
      const r = recorder.current;
      if (r) setRec((s) => ({ ...s, secs: r.seconds, level: r.level }));
    }, 80);
    return () => clearInterval(id);
  }, [rec.on]);

  const toggle = useCallback(async () => {
    if (!sent) return;
    if (rec.on) {
      const blob = await recorder.current!.stop();
      recorder.current = null;
      setRec({ on: false, secs: 0, level: 0 });
      if (blob.size < 44 + 16000) return;
      setBusy(true);
      try {
        const fd = new FormData();
        fd.append("file", blob, "read.wav");
        const r = await fetch(`/api/read/score?id=${sent.id}`, { method: "POST", body: fd });
        const j: ReadResult = await r.json();
        setResult(j);
        setHistory((h) => [{ text: sent.text, score: j.score, verdict: j.verdict }, ...h].slice(0, 8));
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
      return;
    }
    try {
      const r = new WavRecorder();
      await r.start();
      recorder.current = r;
      setResult(null);
      setRec({ on: true, secs: 0, level: 0 });
    } catch {
      setErr("Microphone permission was refused, or no microphone was found.");
    }
  }, [rec.on, sent]);

  const avg = history.length ? Math.round(history.reduce((a, b) => a + b.score, 0) / history.length) : null;
  return (
    <section className="pt-10 pb-16">
      <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight [text-wrap:balance]">
        Read it aloud.
        <span className="text-ink-3 font-normal"> The model tells you how clearly it heard this sentence, not another.</span>
      </h1>
      {err && <p className="mt-6 rounded-md border border-accent/40 bg-accent/10 px-4 py-3 text-sm">{err}</p>}
      {sent && (
        <div className="mt-8 rounded-lg border border-line bg-paper-2/60 p-6">
          <div className="font-mono text-[11px] text-ink-3 mb-3">
            {sent.lang ?? ""} · sentence #{sent.id}
          </div>
          <p className="text-xl sm:text-2xl leading-relaxed [text-wrap:balance]">{sent.text}</p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button
              type="button"
              size="lg"
              variant={rec.on ? "accent" : "default"}
              onClick={toggle}
              disabled={busy}
              className={cn(rec.on && "recording")}
            >
              {busy ? <Loader2 className="animate-spin" /> : rec.on ? <Square /> : <Mic />}
              {rec.on ? `stop (${rec.secs.toFixed(1)}s)` : busy ? "scoring…" : "record"}
            </Button>
            <Button type="button" variant="outline" size="lg" onClick={next} disabled={rec.on || busy}>
              skip
            </Button>
            {rec.on && (
              <div className="h-1.5 w-40 rounded bg-paper overflow-hidden">
                <div className="h-full bg-accent transition-[width] duration-75" style={{ width: `${rec.level * 100}%` }} />
              </div>
            )}
          </div>
        </div>
      )}
      {result && (
        <div className="mt-6 flex items-start gap-6">
          <div className="shrink-0">
            <div
              className={cn(
                "text-5xl font-semibold tabular-nums",
                result.verdict === "clear" ? "text-ok" : result.verdict === "close" ? "text-accent" : "text-weak",
              )}
            >
              {result.score}
            </div>
            <div className="font-mono text-[11px] text-ink-3 uppercase tracking-wide">{result.verdict}</div>
          </div>
          <div className="text-sm text-ink-2 space-y-1">
            <p>
              Your reading was the nearest match for this sentence out of {result.pool}
              {result.rank === 1 ? "." : `, but it ranked ${result.rank}.`}
            </p>
            <p className="font-mono text-[11px] text-ink-3">
              similarity {result.similarity.toFixed(3)} · margin over next sentence {result.margin.toFixed(3)}
            </p>
            {result.sounded_like && (
              <p>
                It sounded more like: <span className="text-ink">“{result.sounded_like}”</span>
              </p>
            )}
            <div className="pt-2">
              <Button type="button" size="sm" onClick={next}>
                next sentence <ArrowRight />
              </Button>
            </div>
          </div>
        </div>
      )}
      {history.length > 0 && (
        <div className="mt-10 border-t border-line pt-4">
          <div className="flex justify-between font-mono text-[11px] text-ink-3">
            <span>last {history.length}</span>
            <span>average {avg}</span>
          </div>
          <ul className="mt-2 divide-y divide-line">
            {history.map((h, i) => (
              <li key={i} className="py-2 flex items-baseline gap-3 text-sm">
                <span className="font-mono tabular-nums w-8 text-right">{h.score}</span>
                <Badge variant={h.verdict === "clear" ? "ok" : h.verdict === "close" ? "accent" : "weak"}>{h.verdict}</Badge>
                <span className="text-ink-2 truncate">{h.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function HitRow({ h, i, query }: { h: Hit; i: number; query: string }) {
  const weak = h.score < WEAK;
  const pct = Math.round(Math.max(0, Math.min(1, (h.score - 0.3) / 0.6)) * 100);
  const parts = query ? highlight(h.snippet, query) : [h.snippet];
  const isAudio = h.start_s != null && h.file;
  const src = isAudio
    ? `/api/audio?file=${encodeURIComponent(h.file!)}&start=${h.start_s}&end=${h.end_s ?? ""}`
    : null;
  const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  return (
    <li className={cn("py-4 flex gap-4", weak && "opacity-70")}>
      <div className="w-7 shrink-0 pt-0.5 font-mono text-xs text-ink-3 tabular-nums">{String(i + 1).padStart(2, "0")}</div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-medium leading-snug [text-wrap:balance]">{h.title}</h3>
          {h.lang_code && <Badge variant={weak ? "weak" : "accent"}>{h.lang_code}</Badge>}
          {h.category && <Badge>{h.category}</Badge>}
          {weak && <Badge variant="weak">weak match</Badge>}
        </div>
        {isAudio ? (
          <div className="mt-2 flex items-center gap-3">
            <audio controls preload="none" src={src!} className="h-8 max-w-full" />
            <span className="font-mono text-[11px] text-ink-3 tabular-nums">
              {fmt(h.start_s!)}–{h.end_s != null ? fmt(h.end_s) : "end"}
            </span>
          </div>
        ) : (
          <p className="mt-1.5 text-sm text-ink-2 leading-relaxed line-clamp-3">
            {parts.map((p, j) => (typeof p === "string" ? <span key={j}>{p}</span> : <mark key={j}>{p.m}</mark>))}
          </p>
        )}
        <div className="mt-2 flex items-center gap-3">
          <div className="h-1 w-28 rounded bg-paper-2 overflow-hidden">
            <div className={cn("h-full", weak ? "bg-weak" : "bg-ok")} style={{ width: `${pct}%` }} />
          </div>
          <span className="font-mono text-[11px] text-ink-3 tabular-nums">{h.score.toFixed(3)}</span>
          {h.url && (
            <a
              href={h.url.startsWith("http") ? h.url : `https://www.bbc.com${h.url}`}
              target="_blank"
              rel="noreferrer"
              className="ml-auto inline-flex items-center gap-1 font-mono text-[11px] text-ink-3 hover:text-accent"
            >
              source <ExternalLink className="size-3" />
            </a>
          )}
        </div>
      </div>
    </li>
  );
}
