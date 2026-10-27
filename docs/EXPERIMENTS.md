# Experiments

Every command writes `results/<experiment>/<model>/<run-id>.json` (plus `latest.json`) with the full
metric tree, the exact prompts used, library versions, GPU, and git SHA. `afriembed report` turns the
latest run of each into `results/REPORT.md`.

Common flags: `--model` (registry name in `configs/models.yaml`, HF id, or local path), `--dim`
(Matryoshka truncation), `--no-prompts` (prefix ablation), `--adapter` (LoRA from a fine-tune),
`--trim-dir` (vocab-trimmed tables), `--batch-size`, `--device`, `--tag`.

| # | Experiment | Command | Data | Headline metric | Module |
|---|------|---------|------|-----------------|--------|
| 1 | AfriMTEB-Lite / Full for v2, and the v1 re-run | `afrimteb --suite lite` / `--suite full`, `--no-prompts` | mteb `MTEB(Africa, v1[, lite])` | macro average main score; per task; per language | `experiments/afrimteb.py` |
| 2 | African speech→text retrieval benchmark | `speech-text --langs fleurs_all` | FLEURS test (≈20 African langs) | R@1 / MRR audio→text and text→audio per language | `experiments/speech_text.py` |
| 3 | Cross-lingual speech bitext mining | `speech-bitext --pivot eng`, `--all-pairs` | FLEURS (n-way parallel ids) | accuracy audio→text, audio→audio, text→text, text→audio | `experiments/speech_bitext.py` |
| 4 | Tokenizer fertility / African language tax | `fertility` | SIB-200 (FLORES sentences) | tokens per word, premium vs English, byte-fallback rate | `experiments/fertility.py` |
| 5 | Matryoshka degradation by language | `afrimteb --matryoshka`; `speech-text --dims 768,512,256,128` | as 1 and 2 | retention vs 768d, per language | same as 1 / 2 |
| 6 | Image ↔ African caption alignment | `vision --dataset afri-aya` / `africaption` | Afri-Aya; AfriCaption + Flickr8k | image→local R@1 vs image→English R@1, language gap | `experiments/vision.py` |
| 7 | AfriEmbeddingGemma-2 text (LoRA) | `finetune-text` then `afrimteb --adapter` | AfriCaption + Afri-Aya caption pairs (never AfriMTEB test) | Δ macro average | `experiments/finetune_text.py` |
| 8 | African speech-aligned audio tower | `finetune-audio` then `speech-text --adapter` | FLEURS train (audio, transcript) | Δ audio→text R@1 | `experiments/finetune_audio.py` |
| 9 | Vocab-trimmed micro variants | `vocab-trim build` then any eval with `--trim-dir` | SIB-200 train/val + Injongo train | MB saved, held-out token coverage, Δ score | `experiments/vocab_trim.py` |
| 10 | On-device African RAG | `index-docs`, `rag ask`, `rag eval`, `rag export-afriqa` | your corpus; AfriQA questions | answer-in-context@k, token F1, EM | `experiments/rag.py` |
| 11 | Radio / oral archive search | `index-audio <folder>`, `search --query`, `--ui` | any recordings | interactive; indexing speed logged | `experiments/search.py` |
| 12 | Offline voice search over documents | `index-docs <folder>`, `search --audio q.wav` | any text | interactive | `experiments/search.py` |
| 13 | Cross-lingual spoken/text query → English or African corpora | `ciral` (text side); `speech-bitext` (speech side) | CIRAL (hau, som, swh, yor) | nDCG@20, R@100 | `experiments/ciral.py` |
| 14 | Spoken language ID and corpus sorting | `lid --langs fleurs_all` | FLEURS | kNN accuracy, NMI, purity, confusions | `experiments/lid.py` |
| 15 | Accented English (AfriSpeech-200) | `accent` | AfriSpeech-200 (120 accents, clinical+general) | R@1 by accent / country / domain | `experiments/accent.py` |

## Notes that affect interpretation

- **Prompts.** EmbeddingGemma's catalog is applied by `prompt_name` (`query`, `document`, `Classification`,
  `Clustering`, `STS`, `BitextMining`). `--no-prompts` clears the catalog on the loaded model, so mteb's
  wrapper cannot re-apply it either. The prompts actually used are recorded in `model_info.prompts`.
- **Media inputs take no prefix** (per the model card). For audio→text retrieval the transcript side
  defaults to the `document` prompt; `--text-task none` is the ablation.
- **Matryoshka.** Truncation is a prefix slice, so `speech-text` scores all dims from one encoding.
  For mteb, `--matryoshka` re-runs per dim and sets `model.truncate_dim`; cosine-based tasks are
  scale-invariant, so the missing re-normalisation does not change rankings.
- **FLEURS pools.** One recording per sentence id (`dedupe=True`), so each query has exactly one
  correct target. Pool size = `--max-items` (default 500) or the whole split.
- **CIRAL corpus subsampling.** `--max-passages` keeps every judged passage and samples the rest;
  the results file records `subsampled: true`. Use `--max-passages 0` for the full corpus.
- **Contamination.** Text fine-tuning pairs come from AfriCaption and Afri-Aya only; audio pairs from FLEURS train. FLORES, NTREX, SIB-200
  test, Belebele, AfriXNLI, MasakhaNEWS test and Injongo test are AfriMTEB test data and are blocklisted.
- **Sequence length.** `afrimteb` caps inputs at 1024 tokens by default (`--max-seq-length`); MasakhaNEWS articles at the
  model's 8K context OOM a 24 GB GPU at batch 128. The cap is recorded in `model_info.max_seq_length`.
- **Precision.** bf16 on CUDA, fp32 elsewhere, never fp16 (NaNs with EmbeddingGemma 2).
- **Vocab trimming** keeps the tokenizer intact and prunes every vocab-sized table (token embeddings and
  per-layer embeddings) with an id-remap hook; unseen ids map to `<unk>`. Held-out coverage is reported.
- **4 GB GPUs.** Text-only inference and all evaluations fit. Fine-tuning uses LoRA + cached MNRL with
  `--mini-batch-size 2–8` and gradient checkpointing; `--train-audio-tower` needs more memory.

## Baselines

`me5-base` (278M) and `me5-small` (118M) are the size-matched text baselines; `me5-large-instruct`,
`afrie5-large-instruct` and `bge-m3` are the published AfriMTEB references (≈560M). `embeddinggemma-300m`
is v1 for the generation comparison.
