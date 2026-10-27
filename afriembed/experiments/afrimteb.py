"""Experiment 1 (+5 for text): AfriMTEB / AfriMTEB-Lite via the mteb package, with prompt and Matryoshka ablations."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import mteb_cache_dir
from ..utils import save_result, timer
from .common import ModelOpts

log = logging.getLogger(__name__)

SUITES = {"lite": "MTEB(Africa, v1, lite)", "full": "MTEB(Africa, v1)"}

# Task lists mirror mteb/benchmarks/benchmarks.py (MTEB_AFRICA / MTEB_AFRICA_LITE) so the suites work even
# on mteb releases that ship the tasks but not yet the Benchmark objects.
LITE_TASKS = [
    "AfriHateClassification",
    "AfriSentiClassification",
    "MasakhaNEWSClassification",
    "KinNewsClassification",
    "AfriXNLI",
    "EmotionAnalysisPlus",
    "FloresBitextMining",
    "InjongoIntent",
    "NTREXBitextMining",
    "SIB200Classification.v2",
    "SIB200Classification",
    "SIB200ClusteringS2S",
    "BelebeleRetrieval",
]
FULL_TASKS = [
    "SIB200Classification",
    "FloresBitextMining",
    "SIB200ClusteringS2S",
    "NTREXBitextMining",
    "AfriXNLI",
    "EmotionAnalysisPlus",
    "AfriSentiClassification",
    "MasakhaNEWSClassification",
    "AfriHateClassification",
    "KinNewsClassification",
    "InjongoIntent",
    "SIB200Classification.v2",
    "BelebeleRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "XQuADRetrieval",
    # "XM3600T2IRetrieval",  # text->image task; text-only baselines cannot run it, so it is excluded from the text suite
    "AfriSentiLangClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultilingualSentimentClassification",
    "NaijaSenti",
    "SouthAfricanLangClassification",
    "SiswatiNewsClassification",
    "SwahiliNewsClassification",
    "TswanaNewsClassification",
    "IsiZuluNewsClassification",
    "BibleNLPBitextMining",
    "NollySentiBitextMining",
    "Tatoeba",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "XNLI",
    "MIRACLReranking",
    "SemRel24STS",
]
LITE_LANGS = ["amh", "gaz", "ibo", "yor", "hau", "swh", "kin", "xho", "zul"]
FULL_LANGS = [
    "aeb",
    "afr",
    "aka",
    "amh",
    "arb",
    "arq",
    "ary",
    "arz",
    "bam",
    "bem",
    "cjk",
    "dik",
    "dyu",
    "eng",
    "ewe",
    "fon",
    "fuv",
    "gaz",
    "hau",
    "ibo",
    "kab",
    "kam",
    "kbp",
    "kea",
    "kik",
    "kin",
    "kmb",
    "knc",
    "kon",
    "lin",
    "lua",
    "lug",
    "luo",
    "mos",
    "nbl",
    "nqo",
    "nso",
    "nus",
    "nya",
    "pcm",
    "plt",
    "por",
    "run",
    "sag",
    "sna",
    "som",
    "sot",
    "ssw",
    "swa",
    "swh",
    "taq",
    "tir",
    "tsn",
    "tso",
    "tum",
    "twi",
    "tzm",
    "umb",
    "ven",
    "wol",
    "xho",
    "yor",
    "zul",
]


# Languages present in mteb/sib200.v2 (128 configs; no swh/xho/yor/zul). Queried live when possible.
SIB200_V2_FALLBACK_LANGS = ["amh", "gaz", "hau", "ibo", "kin"]


def _sib200_v2_languages(requested: list[str]) -> list[str]:
    try:
        from huggingface_hub import HfApi

        from ..config import hf_token

        card = HfApi().dataset_info("mteb/sib200.v2", token=hf_token()).card_data or {}
        configs = {c.get("config_name", "")[:3] for c in (card.get("configs") or [])}
        if configs:
            return [l for l in requested if l in configs]
    except Exception as e:  # pragma: no cover
        log.warning("Could not query mteb/sib200.v2 configs (%s); using fallback list", e)
    return [l for l in requested if l in SIB200_V2_FALLBACK_LANGS]


def _install_split_safe_loader(task, splits: tuple[str, ...]) -> None:
    """Replace task.load_data with one that loads only `splits` per language subset.

    mteb's default loader calls load_dataset(name=subset, **dataset) which materialises every split and
    raises 'Instruction "validation" corresponds to no data!' when a split is declared but empty.
    """
    import types

    from datasets import DatasetDict, load_dataset

    def load_data(self, num_proc=None, *, timer=None, **kwargs):
        if self.data_loaded:
            return
        self.dataset = {}
        for hf_subset in self.hf_subsets:
            parts = {}
            for split in splits:
                parts[split] = load_dataset(
                    name=hf_subset, split=split, **self.metadata.dataset, num_proc=num_proc
                )
            self.dataset[hf_subset] = DatasetDict(parts)
        self.dataset_transform(num_proc=num_proc)
        self.data_loaded = True

    task.load_data = types.MethodType(load_data, task)


def patch_tasks(task_objs: list, languages: list[str] | None = None) -> list:
    """Work around two upstream data problems in the AfriMTEB task definitions (mteb 2.24):

    * EmotionAnalysisPlus trains its classifier on the `validation` split, which is empty for every
      language in mteb/EmotionAnalysis -> train on `train` instead (recorded in results as a patch).
    * SIB200Classification.v2 points at mteb/sib200.v2, which lacks swh/xho/yor/zul configs -> restrict
      the task to the languages that exist rather than failing the whole task.
    """
    import mteb

    out = []
    for t in task_objs:
        name = t.metadata.name
        if name == "EmotionAnalysisPlus":
            _install_split_safe_loader(t, splits=("train", "test"))
            log.warning(
                "Patched EmotionAnalysisPlus: loading only train/test splits (upstream validation split is empty)"
            )
        if name == "SIB200Classification.v2":
            requested = languages or [s[:3] for s in (t.hf_subsets or [])]
            avail = _sib200_v2_languages(requested)
            if set(avail) != set(requested):
                log.warning(
                    "Patched SIB200Classification.v2: languages %s -> %s (missing in mteb/sib200.v2)",
                    requested,
                    avail,
                )
                if not avail:
                    continue
                t = list(mteb.get_tasks(tasks=[name], languages=avail, exclusive_language_filter=True))[0]
        out.append(t)
    return out


PATCH_NOTES = {
    "EmotionAnalysisPlus": "loaded train+test splits only (upstream `validation` split is empty)",
    "SIB200Classification.v2": "restricted to languages present in mteb/sib200.v2 (no swh/xho/yor/zul)",
}


def get_suite_tasks(suite: str, languages: list[str] | None = None) -> list:
    """Benchmark tasks from mteb if registered, else rebuilt from the mirrored task + language lists."""
    import mteb

    name = SUITES.get(suite, suite)
    try:
        tasks = list(mteb.get_benchmark(name).tasks)
    except KeyError:
        if suite == "lite":
            task_names, langs = LITE_TASKS, LITE_LANGS
        elif suite == "full":
            task_names, langs = FULL_TASKS, FULL_LANGS
        else:
            raise
        log.warning(
            "%s not registered in mteb %s; rebuilding from task list (%d tasks)",
            name,
            mteb.__version__,
            len(task_names),
        )
        tasks = list(mteb.get_tasks(tasks=task_names, languages=langs, exclusive_language_filter=True))
    suite_langs = LITE_LANGS if suite == "lite" else FULL_LANGS if suite == "full" else None
    if languages:
        names = [t.metadata.name for t in tasks]
        tasks = list(mteb.get_tasks(tasks=names, languages=languages, exclusive_language_filter=True))
    return patch_tasks(tasks, languages or suite_langs)


def _result_cache(path: Path):
    try:
        from mteb.cache import ResultCache
    except ImportError:  # pragma: no cover
        from mteb import ResultCache  # type: ignore
    try:
        return ResultCache(cache_path=path)
    except TypeError:  # pragma: no cover
        return ResultCache(path)


def _extract(model_result) -> dict:
    """Flatten mteb ModelResult into {task: {main_score, per_subset: {...}}}."""
    out = {}
    for tr in getattr(model_result, "task_results", []) or []:
        name = getattr(tr, "task_name", None) or getattr(getattr(tr, "task", None), "metadata", {}).name
        entry = {"main_score": None, "subsets": {}}
        try:
            entry["main_score"] = float(tr.get_score())
        except Exception:
            pass
        scores = getattr(tr, "scores", {}) or {}
        for split, rows in scores.items():
            for row in rows:
                subset = row.get("hf_subset", "default")
                entry["subsets"][f"{split}/{subset}"] = {
                    k: v for k, v in row.items() if isinstance(v, int | float)
                }
        out[name] = entry
    return out


# mteb subsets use a few alternative ISO codes for the same language; fold them onto the registry codes.
_ISO_ALIASES = {"orm": "gaz", "swa": "swh", "kir": "kin"}


def _per_language(flat: dict) -> dict[str, float]:
    """Average main_score per ISO code found in subset names like 'hau', 'eng-hau', 'hau_Latn', 'hau-eng'.

    English pivots ('eng') are excluded so bitext pairs count toward the African side only."""
    acc: dict[str, list[float]] = {}
    for _task, e in flat.items():
        for key, row in e["subsets"].items():
            subset = key.split("/", 1)[1]
            ms = row.get("main_score")
            if ms is None:
                continue
            for tok in subset.replace("_", "-").split("-"):
                if len(tok) == 3 and tok.isalpha() and tok.islower() and tok != "eng":
                    acc.setdefault(_ISO_ALIASES.get(tok, tok), []).append(ms)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}


def _mteb_model(opts: ModelOpts, emb):
    """Prefer mteb's own registry wrapper for stock baselines (it carries the published instructions);
    otherwise hand mteb our SentenceTransformer with a prompt catalog it can resolve."""
    import mteb

    from ..config import model_registry
    from ..models.embedder import mteb_prompt_catalog

    stock = not (opts.adapter or opts.local_path or opts.trim_dir or opts.dim) and opts.use_prompts
    if stock and not emb.family.startswith("embeddinggemma") and emb.name in model_registry():
        try:
            m = mteb.get_model(emb.hf_id, revision=model_registry()[emb.name].get("revision"))
            log.info("Using mteb registry wrapper for %s", emb.hf_id)
            return m, "mteb-registry"
        except Exception as e:  # pragma: no cover
            log.warning("mteb.get_model(%s) failed (%s); falling back to SentenceTransformer", emb.hf_id, e)
    catalog = mteb_prompt_catalog(emb.family, emb.use_prompts)
    if catalog is not None:
        emb.model.prompts = catalog
        log.info("Installed %d-entry prompt catalog for family %s", len(catalog), emb.family)
    return emb.model, "sentence-transformer"


def run(
    opts: ModelOpts,
    suite: str = "lite",
    tasks: list[str] | None = None,
    languages: list[str] | None = None,
    overwrite: bool = True,
    tag: str | None = None,
) -> dict:
    import mteb

    emb = opts.load(("text",))
    if opts.dim:
        emb.model.truncate_dim = opts.dim  # mteb re-normalises for cosine tasks; see docs/EXPERIMENTS.md
    model_for_mteb, wrapper_kind = _mteb_model(opts, emb)
    bench_name = SUITES.get(suite, suite)
    task_objs = get_suite_tasks(suite, languages)
    if tasks:
        task_objs = [t for t in task_objs if t.metadata.name in set(tasks)]
    log.info("Running %s: %d tasks, prompts=%s dim=%s", bench_name, len(task_objs), emb.use_prompts, opts.dim)

    variant = f"{emb.name}-{opts.tag() or 'base'}-{suite}"  # suite in the path: Full and Lite must never share a cache
    cache = _result_cache(mteb_cache_dir() / variant)
    with timer(bench_name):
        res = mteb.evaluate(
            model_for_mteb,
            tasks=task_objs,
            encode_kwargs={"batch_size": opts.batch_size},
            cache=cache,
            overwrite_strategy="always" if overwrite else "only-missing",
            raise_error=False,
        )
    flat = _extract(res)
    mains = [e["main_score"] for e in flat.values() if e["main_score"] is not None]
    payload = {
        "suite": bench_name,
        "model_info": emb.describe(),
        "mteb_wrapper": wrapper_kind,
        "task_patches": {k: v for k, v in PATCH_NOTES.items() if k in flat},
        "n_tasks": len(flat),
        "macro_average": (sum(mains) / len(mains)) if mains else None,
        "tasks": flat,
        "per_language": _per_language(flat),
        "mteb_cache": str(cache.__dict__.get("cache_path", "")),
    }
    save_result(f"afrimteb-{suite}", emb.name, payload, tag=opts.full_tag(tag))
    return payload


def run_matryoshka(opts: ModelOpts, suite: str = "lite", dims=(768, 512, 256, 128), **kw) -> dict[int, dict]:
    out = {}
    for d in dims:
        o = ModelOpts(**{**opts.__dict__, "dim": d})
        out[d] = run(o, suite=suite, tag=f"d{d}", **kw)
    return out
