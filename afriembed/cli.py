"""afriembed command line. One sub-command per experiment; see docs/EXPERIMENTS.md."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import languages as L
from .experiments.common import ModelOpts
from .utils import seed_everything, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="EmbeddingGemma 2 on African languages: benchmarks, fine-tuning, search tools.",
)

# ---------------------------------------------------------------- shared options
Model = Annotated[
    str, typer.Option("--model", "-m", help="Registry name (configs/models.yaml), HF id, or local path")
]
Dim = Annotated[int | None, typer.Option("--dim", help="Matryoshka truncation (768/512/256/128)")]
NoPrompts = Annotated[bool, typer.Option("--no-prompts", help="Ablation: disable task prefixes")]
Batch = Annotated[int, typer.Option("--batch-size", "-b")]
Device = Annotated[str | None, typer.Option("--device", help="cuda / cpu (auto)")]
Adapter = Annotated[
    str | None,
    typer.Option("--adapter", help="PEFT adapter dir from finetune-*; comma-separate several to stack them"),
]
Local = Annotated[
    str | None, typer.Option("--local-path", help="Load weights from a local directory instead of the hub")
]
Trim = Annotated[
    str | None, typer.Option("--trim-dir", help="Apply a vocab-trim built by `vocab-trim build`")
]
Langs = Annotated[
    str | None,
    typer.Option(
        "--langs", "-l", help="Comma list of ISO codes or a named set (lite, fleurs_core, fleurs_all, ciral)"
    ),
]
Tag = Annotated[str | None, typer.Option("--tag", help="Suffix for the results file name")]
MaxSeq = Annotated[
    int | None, typer.Option("--max-seq-length", help="Cap input tokens (memory); default per command")
]
Verbose = Annotated[bool, typer.Option("--verbose", "-v")]


def _opts(
    model,
    dim,
    no_prompts,
    batch_size,
    device,
    adapter,
    local_path,
    trim_dir,
    verbose,
    seed=0,
    max_seq_length=None,
    dtype=None,
    causal_attention=False,
) -> ModelOpts:
    setup_logging(verbose)
    seed_everything(seed)
    return ModelOpts(
        model=model,
        dim=dim,
        use_prompts=not no_prompts,
        batch_size=batch_size,
        device=device,
        adapter=adapter,
        local_path=local_path,
        trim_dir=trim_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        causal_attention=causal_attention,
    )


# ---------------------------------------------------------------- 0. smoke
@app.command()
def smoke(
    model: Model = "embeddinggemma-2", audio: bool = False, device: Device = None, verbose: Verbose = False
):
    """Load the model, embed a few sentences (and optionally a synthetic tone), print shapes and similarities."""
    import numpy as np

    opts = _opts(model, None, False, 8, device, None, None, None, verbose)
    emb = opts.load(("text", "audio") if audio else ("text",))
    q = emb.encode_text(["Wane ne shugaban Najeriya?"], "query")  # Hausa: who is the president of Nigeria?
    d = emb.encode_text(
        [
            "Bola Ahmed Tinubu shi ne shugaban kasar Najeriya.",
            "Ruwa yana da muhimmanci ga rayuwa.",
            "The capital of France is Paris.",
        ],
        "document",
    )
    print("query", q.shape, "docs", d.shape, "cos:", np.round(q @ d.T, 3))
    print("prompts:", emb.describe()["prompts"])
    if audio:
        import soundfile as sf

        from .config import cache_dir

        t = np.linspace(0, 1, 16_000, endpoint=False)
        p = cache_dir() / "smoke_tone.wav"
        sf.write(str(p), (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16_000)
        a = emb.encode_audio([str(p)], "none")
        print("audio", a.shape, "cos to docs:", np.round(a @ d.T, 3))


# ---------------------------------------------------------------- 1 + 5. AfriMTEB
@app.command()
def afrimteb(
    model: Model = "embeddinggemma-2",
    suite: Annotated[str, typer.Option(help="lite | full")] = "lite",
    tasks: Annotated[str | None, typer.Option(help="Comma list of mteb task names to restrict to")] = None,
    langs: Langs = None,
    matryoshka: Annotated[bool, typer.Option("--matryoshka", help="Run at 768/512/256/128")] = False,
    dim: Dim = None,
    no_prompts: NoPrompts = False,
    batch_size: Batch = 32,
    device: Device = None,
    adapter: Adapter = None,
    local_path: Local = None,
    trim_dir: Trim = None,
    no_overwrite: bool = False,
    max_seq_length: MaxSeq = 1024,
    dtype: Annotated[
        str | None, typer.Option(help="bf16 | fp16 | fp32 (ablation; default bf16 on GPU)")
    ] = None,
    causal_attention: Annotated[
        bool,
        typer.Option(
            "--causal-attention", help="Ablation: causal mask, as pre-fix transformers ran EmbeddingGemma"
        ),
    ] = False,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 1: AfriMTEB(-Lite) via mteb. Add --no-prompts for the ablation, --matryoshka for experiment 5.

    --max-seq-length defaults to 1024: MasakhaNEWS articles at the 8K context OOM a 24 GB GPU at batch 128."""
    from .experiments import afrimteb as E

    opts = _opts(
        model,
        dim,
        no_prompts,
        batch_size,
        device,
        adapter,
        local_path,
        trim_dir,
        verbose,
        max_seq_length=max_seq_length,
        dtype=dtype,
        causal_attention=causal_attention,
    )
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None
    lang_list = L.resolve(langs, "lite") if langs else None
    if matryoshka:
        E.run_matryoshka(opts, suite=suite, tasks=task_list, languages=lang_list, overwrite=not no_overwrite)
    else:
        E.run(opts, suite=suite, tasks=task_list, languages=lang_list, overwrite=not no_overwrite, tag=tag)


# ---------------------------------------------------------------- 2 + 5. speech <-> text
@app.command("speech-text")
def speech_text(
    model: Model = "embeddinggemma-2",
    langs: Langs = "fleurs_core",
    split: str = "test",
    max_items: Annotated[int | None, typer.Option(help="Utterances per language (pool size)")] = 500,
    text_task: Annotated[
        str, typer.Option(help="Prompt for the transcript side: document | none | query")
    ] = "document",
    dims: Annotated[str, typer.Option(help="Comma list of Matryoshka dims to score")] = "768,512,256,128",
    shuffle_control: Annotated[
        bool,
        typer.Option("--shuffle-control", help="Negative control: permute transcripts; expect R@1 ≈ 1/pool"),
    ] = False,
    no_prompts: NoPrompts = False,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    local_path: Local = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 2: FLEURS speech->text and text->speech retrieval per language; experiment 5 for audio via --dims."""
    from .experiments import speech_text as E

    opts = _opts(model, None, no_prompts, batch_size, device, adapter, local_path, None, verbose)
    E.run(
        opts,
        L.resolve(langs, "fleurs_core"),
        split=split,
        max_items=max_items,
        text_task=text_task,
        dims=tuple(int(d) for d in dims.split(",")),
        shuffle_control=shuffle_control,
        tag=tag,
    )


# ---------------------------------------------------------------- 2b. held-out speech
@app.command("heldout")
def heldout(
    model: Model = "embeddinggemma-2",
    sources: Annotated[
        str | None,
        typer.Option(
            help="Comma list of source ids (bible-hau, bible-ewe, naija-hau, naija-ibo, naija-yor, waxal-lug, waxal-orm, waxal-sna, waxal-lin, waxal-ewe)"
        ),
    ] = None,
    max_items: int = 1500,
    text_task: str = "document",
    no_prompts: NoPrompts = False,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 2b: speech<->text retrieval on non-FLEURS sources (BibleTTS, NaijaVoices, WAXAL) to rule out leakage."""
    from .experiments import heldout as E

    opts = _opts(model, None, no_prompts, batch_size, device, adapter, None, None, verbose)
    E.run(
        opts,
        sources=[s.strip() for s in sources.split(",")] if sources else None,
        max_items=max_items,
        text_task=text_task,
        tag=tag,
    )


# ---------------------------------------------------------------- P2. voice-note search measurement
@app.command("voice-notes")
def voice_notes(
    langs: Langs = "fleurs_core",
    max_sentences: int = 400,
    model: Model = "embeddinggemma-2",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Product 2: speaker-independent audio->audio retrieval on FLEURS (another reader saying the same sentence)."""
    from .experiments import voicenotes as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    r = E.run(opts, L.resolve(langs, "fleurs_core"), max_sentences=max_sentences, tag=tag)
    print(r["mean_recall@1"])


# ---------------------------------------------------------------- P4. spoken intent routing
@app.command("spoken-intent")
def spoken_intent(
    langs: Annotated[str, typer.Option(help="Injongo languages, comma list")] = "hau,yor,swa,ibo",
    k: int = 5,
    max_test: int | None = None,
    model: Model = "embeddinggemma-2",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Product 4: Spoken Injongo. Intent accuracy by nearest-neighbour embedding, typed vs spoken (MMS-TTS), no ASR."""
    from .experiments import spoken_intent as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    r = E.run(opts, [x.strip() for x in langs.split(",")], k=k, max_test=max_test, tag=tag)
    print(r["mean"])


# ---------------------------------------------------------------- P5. overlap / leakage check
@app.command()
def dedup(
    a: Annotated[
        str, typer.Option(help="Recordings: fleurs:<lang>:<split> or naija-hau / bible-hau ...")
    ] = "fleurs:hau:test",
    b: Annotated[
        str, typer.Option(help="Sentences to search for in A (same spec format)")
    ] = "fleurs:hau:dev",
    max_items: int = 1000,
    threshold: float | None = None,
    model: Model = "embeddinggemma-2",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Product 5: find sentences of B that are spoken in A (shared prompts, leaked test sentences) from audio alone."""
    from .experiments import dedup as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    r = E.run(opts, a, b, max_items=max_items, threshold=threshold, tag=tag)
    print(
        {k: r[k] for k in ("true_overlap", "flagged", "tp", "fp", "fn", "precision", "recall", "threshold")}
    )


# ---------------------------------------------------------------- P6. long-audio alignment
align_app = typer.Typer(help="Product 6: align long read audio to its sentences without ASR.")
app.add_typer(align_app, name="align")


@align_app.command("run")
def align_run(
    audio: Path,
    text: Annotated[Path, typer.Argument(help="One sentence / verse per line, in reading order")],
    out: Path | None = None,
    max_group: int | None = None,
    model: Model = "embeddinggemma-2",
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    verbose: Verbose = False,
):
    """Align one recording to its text; writes <audio>.aligned.csv with start_s, end_s, confidence, text."""
    from .experiments import align as E

    opts = _opts(model, None, False, 8, device, adapter, None, None, verbose)
    print(E.align_file(opts, audio, text, out=out, max_group=max_group))


@align_app.command("eval")
def align_eval(
    lang: str = "hau",
    n_chapters: int = 10,
    verses_per_chapter: int = 20,
    gap_s: float = 0.4,
    max_group: int | None = None,
    model: Model = "embeddinggemma-2",
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Synthetic chapters from BibleTTS verses (known boundaries): verse IoU and boundary accuracy."""
    from .experiments import align as E

    opts = _opts(model, None, False, 8, device, adapter, None, None, verbose)
    r = E.evaluate(
        opts,
        lang=lang,
        n_chapters=n_chapters,
        verses_per_chapter=verses_per_chapter,
        gap_s=gap_s,
        max_group=max_group,
        tag=tag,
    )
    print(r["mean"])


# ---------------------------------------------------------------- P1. dataset validator
@app.command()
def validate(
    manifest: Annotated[
        Path | None, typer.Option(help="csv/tsv/jsonl with `audio` and `text` columns (paths relative to it)")
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            help="Self-test on a held-out source: bible-hau, bible-ewe, naija-hau, naija-ibo, naija-yor, waxal-*"
        ),
    ] = None,
    max_items: int = 1000,
    inject: Annotated[float, typer.Option(help="Shuffle this fraction of labels to measure detection")] = 0.0,
    out: Annotated[
        Path | None, typer.Option(help="Report directory (default reports/validate-<name>)")
    ] = None,
    pass_margin: float = 0.02,
    borderline_rank: int = 5,
    min_similarity: float = 0.30,
    model: Model = "embeddinggemma-2",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Product 1: flag clips whose audio does not match their transcript (pass / borderline / fail), no ASR needed."""
    from .experiments import validate as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    s = E.run(
        opts,
        manifest=manifest,
        source=source,
        max_items=max_items,
        inject=inject,
        out_dir=out,
        thresholds=E.Thresholds(pass_margin, borderline_rank, min_similarity),
        tag=tag,
    )
    print({k: s[k] for k in ("n_clips", "n_prompts", "counts", "flag_rate", "report")})
    if "injection" in s:
        print("injection:", s["injection"])


# ---------------------------------------------------------------- 2c. cascade baseline + large pool
@app.command()
def cascade(
    model: Model = "embeddinggemma-2",
    langs: Langs = "hau,yor,swh,amh,ibo,som,xho,zul,wol,lug",
    max_items: int | None = 300,
    batch_size: Batch = 16,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Cascade baseline: MMS-1B-all ASR -> text embedding -> retrieval, versus direct audio->text, plus WER."""
    from .experiments import cascade as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    E.run(opts, L.resolve(langs, "lite"), max_items=max_items, tag=tag)


@app.command()
def largepool(
    model: Model = "embeddinggemma-2",
    langs: Langs = "hau,yor,swh",
    max_queries: int = 300,
    ciral_passages: int | None = 50_000,
    batch_size: Batch = 16,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Large-pool test: speech queries against nested pools from ~300 to hundreds of thousands of sentences."""
    from .experiments import largepool as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    E.run(opts, L.resolve(langs, "ciral"), max_queries=max_queries, ciral_passages=ciral_passages, tag=tag)


# ---------------------------------------------------------------- 3. speech bitext
@app.command("speech-bitext")
def speech_bitext(
    model: Model = "embeddinggemma-2",
    langs: Langs = "fleurs_core",
    pivot: str = "eng",
    all_pairs: Annotated[
        bool, typer.Option("--all-pairs", help="African<->African pairs instead of each->pivot")
    ] = False,
    split: str = "test",
    max_items: int | None = 400,
    text_task: str = "bitext",
    no_prompts: NoPrompts = False,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    local_path: Local = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 3: cross-lingual speech bitext mining (audio->text, audio->audio, text->text, text->audio)."""
    from .experiments import speech_bitext as E

    opts = _opts(model, None, no_prompts, batch_size, device, adapter, local_path, None, verbose)
    E.run(
        opts,
        L.resolve(langs, "fleurs_core"),
        pivot=pivot,
        split=split,
        max_items=max_items,
        all_pairs=all_pairs,
        text_task=text_task,
        tag=tag,
    )


# ---------------------------------------------------------------- 4. fertility
@app.command()
def fertility(
    langs: Langs = "fertility",
    tokenizers: Annotated[
        str | None, typer.Option(help="Comma list name=hf_id; default from configs/models.yaml")
    ] = None,
    max_sentences: int | None = None,
    verbose: Verbose = False,
):
    """Experiment 4: tokenizer fertility and the English premium across African languages."""
    from .experiments import fertility as E

    setup_logging(verbose)
    toks = dict(kv.split("=", 1) for kv in tokenizers.split(",")) if tokenizers else None
    E.run(L.resolve(langs, "fertility"), tokenizers=toks, max_sentences=max_sentences)


# ---------------------------------------------------------------- 6. vision
@app.command()
def vision(
    model: Model = "embeddinggemma-2",
    dataset: Annotated[str, typer.Option(help="afri-aya | africaption")] = "afri-aya",
    langs: Annotated[
        str | None,
        typer.Option(help="afri-aya: comma list of community/language names; africaption: ISO codes"),
    ] = None,
    max_items: int | None = 300,
    min_per_language: int = 20,
    text_task: str = "document",
    no_prompts: NoPrompts = False,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 6: image <-> African-language caption retrieval and the English/local language gap."""
    from .experiments import vision as E

    opts = _opts(model, None, no_prompts, batch_size, device, adapter, None, None, verbose)
    E.run(
        opts,
        dataset=dataset,
        languages=[s.strip() for s in langs.split(",")] if langs else None,
        max_items=max_items,
        min_per_language=min_per_language,
        text_task=text_task,
        tag=tag,
    )


# ---------------------------------------------------------------- 7. finetune text
@app.command("finetune-text")
def finetune_text(
    out: Annotated[Path, typer.Option(help="Output dir for the adapter")] = Path("checkpoints/text-lora"),
    model: Model = "embeddinggemma-2",
    langs: Langs = "lite",
    epochs: float = 1.0,
    lr: float = 2e-5,
    batch_size: Batch = 32,
    mini_batch_size: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    full_finetune: bool = False,
    max_pairs: int | None = None,
    max_per_lang: int | None = 4000,
    extra_tsv: Annotated[Path | None, typer.Option(help="Extra anchor<TAB>positive pairs")] = None,
    no_captions: Annotated[
        bool, typer.Option("--no-captions", help="Train on --extra-tsv only (no caption pairs)")
    ] = False,
    device: Device = None,
    verbose: Verbose = False,
):
    """Experiment 7: LoRA contrastive fine-tune on English<->African caption pairs (never AfriMTEB test data)."""
    from .experiments import finetune_text as E

    opts = _opts(model, None, False, batch_size, device, None, None, None, verbose)
    E.train(
        opts,
        L.resolve(langs, "lite"),
        out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        full_finetune=full_finetune,
        max_pairs=max_pairs,
        max_per_lang=max_per_lang,
        extra_tsv=extra_tsv,
        include_captions=not no_captions,
    )


# ---------------------------------------------------------------- 8. finetune audio
@app.command("finetune-audio")
def finetune_audio(
    out: Annotated[Path, typer.Option()] = Path("checkpoints/audio-lora"),
    model: Model = "embeddinggemma-2",
    langs: Langs = "fleurs_core",
    split: str = "train",
    max_per_lang: int | None = 1500,
    epochs: float = 1.0,
    lr: float = 2e-5,
    batch_size: Batch = 8,
    mini_batch_size: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    train_audio_tower: Annotated[
        bool, typer.Option("--train-audio-tower", help="Unfreeze the 300M audio encoder too")
    ] = False,
    seed: Annotated[int, typer.Option(help="LoRA init, data order and trainer seed")] = 0,
    device: Device = None,
    verbose: Verbose = False,
):
    """Experiment 8: align speech to transcripts for African languages with (audio, text) contrastive pairs."""
    from .experiments import finetune_audio as E

    opts = _opts(model, None, False, batch_size, device, None, None, None, verbose, seed=seed)
    E.train(
        opts,
        L.resolve(langs, "fleurs_core"),
        out,
        split=split,
        max_per_lang=max_per_lang,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        train_audio_tower=train_audio_tower,
        seed=seed,
    )


# ---------------------------------------------------------------- 9. vocab trim
trim_app = typer.Typer(help="Experiment 9: vocabulary trimming for micro variants.")
app.add_typer(trim_app, name="vocab-trim")


@trim_app.command("build")
def vocab_trim_build(
    out: Annotated[Path, typer.Option()] = Path("checkpoints/trim-lite"),
    model: Model = "embeddinggemma-2",
    langs: Langs = "lite",
    extra_text: Path | None = None,
    keep_top_general: int = 8000,
    device: Device = None,
    verbose: Verbose = False,
):
    """Count token ids over target-language corpora and save pruned embedding tables + id remap."""
    from .experiments import vocab_trim as E

    opts = _opts(model, None, False, 32, device, None, None, None, verbose)
    E.build(
        opts,
        L.resolve(langs, "lite") + ["eng"],
        out,
        extra_text=extra_text,
        keep_top_general=keep_top_general,
    )


# ---------------------------------------------------------------- 10. RAG
rag_app = typer.Typer(help="Experiment 10: on-device RAG in African languages.")
app.add_typer(rag_app, name="rag")


@rag_app.command("ask")
def rag_ask(
    question: str,
    index: Annotated[Path, typer.Option()] = Path("indexes/docs"),
    model: Model = "embeddinggemma-2",
    generator: str | None = None,
    k: int = 4,
    device: Device = None,
    verbose: Verbose = False,
):
    """Answer one question over a docs index (built with `index-docs`). --generator google/gemma-3-1b-it to generate."""
    from .experiments import rag as E

    opts = _opts(model, None, False, 16, device, None, None, None, verbose)
    gen = E.Generator(generator) if generator else None
    res = E.RAG(opts, index, gen, k=k).answer(question)
    for i, h in enumerate(res["contexts"], 1):
        print(f"[{i}] {h['score']:.3f} {h.get('title') or h.get('file')}: {(h.get('text') or '')[:160]!r}")
    if res["answer"] is not None:
        print("\nAnswer:", res["answer"])


@rag_app.command("eval")
def rag_eval(
    qa_jsonl: Path,
    index: Annotated[Path, typer.Option()] = Path("indexes/docs"),
    model: Model = "embeddinggemma-2",
    generator: str | None = "google/gemma-3-1b-it",
    k: int = 4,
    max_questions: int | None = None,
    device: Device = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Score retrieval (answer-in-context@k) and generation (token F1 / EM) on a {"question","answers"} JSONL."""
    from .experiments import rag as E

    opts = _opts(model, None, False, 16, device, None, None, None, verbose)
    E.evaluate(
        opts, index, qa_jsonl, generator_id=generator or None, k=k, max_questions=max_questions, tag=tag
    )


@rag_app.command("spoken-eval")
def rag_spoken_eval(
    langs: Annotated[
        str, typer.Option(help="AfriQA languages (hau, ibo, yor, swa, ...)")
    ] = "hau,yor,swa,ibo",
    max_questions: int | None = None,
    generator: Annotated[
        str | None, typer.Option(help="e.g. google/gemma-3-1b-it; omit for retrieval only")
    ] = None,
    n_gen: int = 50,
    k: int = 3,
    model: Model = "embeddinggemma-2",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = "artifacts/audio-lora",
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Product 3: Spoken AfriQA. Gold-passage retrieval from typed vs spoken (MMS-TTS) questions, optional generation."""
    from .experiments import spoken_qa as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    r = E.run(
        opts,
        [x.strip() for x in langs.split(",")],
        max_questions=max_questions,
        generator_id=generator,
        n_gen=n_gen,
        k=k,
        tag=tag,
    )
    print(r["mean_recall@1"])


@rag_app.command("export-afriqa")
def rag_export_afriqa(lang: str, out: Path, split: str = "test", verbose: Verbose = False):
    """Export AfriQA questions for a language to the eval JSONL format."""
    from .experiments import rag as E

    setup_logging(verbose)
    E.export_afriqa(lang, out, split=split)


# ---------------------------------------------------------------- 11 + 12. search
@app.command("index-audio")
def index_audio(
    folder: Path,
    out: Annotated[Path, typer.Option()] = Path("indexes/audio"),
    model: Model = "embeddinggemma-2",
    window_s: float = 10.0,
    hop_s: float | None = None,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    verbose: Verbose = False,
):
    """Experiment 11: index a folder of recordings (radio, oral histories) as 10 s windows."""
    from .experiments import search as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    E.index_audio(opts, folder, out, window_s=window_s, hop_s=hop_s)


@app.command("index-docs")
def index_docs(
    folder: Path,
    out: Annotated[Path, typer.Option()] = Path("indexes/docs"),
    model: Model = "embeddinggemma-2",
    max_chars: int = 1200,
    batch_size: Batch = 32,
    device: Device = None,
    adapter: Adapter = None,
    trim_dir: Trim = None,
    verbose: Verbose = False,
):
    """Experiment 12: index text documents (.txt/.md/.jsonl) for voice or text search."""
    from .experiments import search as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, trim_dir, verbose)
    E.index_docs(opts, folder, out, max_chars=max_chars)


@app.command()
def search(
    index: Path,
    query: str | None = None,
    audio: Annotated[Path | None, typer.Option(help="Spoken query (.wav/.mp3)")] = None,
    k: int = 10,
    model: Model = "embeddinggemma-2",
    device: Device = None,
    adapter: Adapter = None,
    ui: Annotated[bool, typer.Option("--ui", help="Serve the web UI (React app + API) instead")] = False,
    gradio: Annotated[
        bool, typer.Option("--gradio", help="With --ui: use the plain Gradio page instead")
    ] = False,
    port: int = 7860,
    verbose: Verbose = False,
):
    """Search an audio or docs index by typed text or a spoken query."""
    from .experiments import search as E

    opts = _opts(model, None, False, 8, device, adapter, None, None, verbose)
    if ui and gradio:
        E.launch_ui(opts, index, port=port)
        return
    if ui:
        from .server import serve

        serve(opts, index, port=port)
        return
    for h in E.search(opts, index, query=query, audio_query=audio, k=k):
        loc = (
            f"{h['file']} [{h['start_s']:.1f}s–{h['end_s']:.1f}s]"
            if "start_s" in h
            else f"{h.get('title') or h['file']}"
        )
        print(f"{h['score']:.3f}  {loc}  {(h.get('text') or '')[:120]!r}")


# ---------------------------------------------------------------- 13. CIRAL
@app.command()
def ciral(
    model: Model = "embeddinggemma-2",
    langs: Langs = "ciral",
    split: str = "test-a",
    max_passages: int | None = 50_000,
    translated: bool = False,
    dim: Dim = None,
    no_prompts: NoPrompts = False,
    batch_size: Batch = 32,
    device: Device = None,
    adapter: Adapter = None,
    trim_dir: Trim = None,
    max_seq_length: MaxSeq = 2048,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 13: English queries -> Hausa/Somali/Swahili/Yoruba passages (nDCG@20, R@100)."""
    from .experiments import ciral as E

    opts = _opts(
        model,
        dim,
        no_prompts,
        batch_size,
        device,
        adapter,
        None,
        trim_dir,
        verbose,
        max_seq_length=max_seq_length,
    )
    E.run(
        opts,
        L.resolve(langs, "ciral"),
        split=split,
        max_passages=max_passages,
        translated=translated,
        tag=tag,
    )


# ---------------------------------------------------------------- 14. LID
@app.command()
def lid(
    model: Model = "embeddinggemma-2",
    langs: Langs = "fleurs_core",
    per_lang: int = 100,
    k: int = 5,
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 14: spoken language ID (kNN) and language clustering from raw embeddings."""
    from .experiments import lid as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    E.run(opts, L.resolve(langs, "fleurs_core"), per_lang=per_lang, k=k, tag=tag)


# ---------------------------------------------------------------- 15. accents
@app.command()
def accent(
    model: Model = "embeddinggemma-2",
    accents: Annotated[str | None, typer.Option(help="Comma list; default all 100+")] = None,
    max_per_accent: int | None = 30,
    text_task: str = "document",
    batch_size: Batch = 8,
    device: Device = None,
    adapter: Adapter = None,
    tag: Tag = None,
    verbose: Verbose = False,
):
    """Experiment 15: African-accented English (AfriSpeech-200) speech->transcript retrieval by accent/country/domain."""
    from .experiments import accent as E

    opts = _opts(model, None, False, batch_size, device, adapter, None, None, verbose)
    E.run(
        opts,
        accents=[a.strip() for a in accents.split(",")] if accents else None,
        max_per_accent=max_per_accent,
        text_task=text_task,
        tag=tag,
    )


# ---------------------------------------------------------------- report
@app.command()
def report():
    """Aggregate results/**/latest.json into results/REPORT.md."""
    from .experiments import report as E

    print(E.build())


@app.command("list-langs")
def list_langs():
    """Print the language registry."""
    for code, l in L.all_languages().items():
        print(
            f"{code:4} {l.name:16} {l.script} tier={l.tier:6} fleurs={l.fleurs or '-':7} sib={l.sib or '-':9} lite={'y' if l.lite else '-'}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
