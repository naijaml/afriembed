# Voice / text search demo

`demo/corpus/` holds four hand-written guides (two Hausa, two English). The `news/` set of 360 BBC articles from
MasakhaNEWS (60 each in Hausa, Igbo, Yoruba, Nigerian Pidgin, Swahili and English) is not in the repo because the
articles are BBC copyright; build it first with `uv run python scripts/build_demo_corpus.py --per-lang 60`.
Indexing all 364 files (647 chunks) takes about a minute on an RTX 3050.

Things to ask (typed or spoken): "yaushe ake shuka masara?", "labarin kwallon kafa a Najeriya",
"ìròyìn nípa Buhari àti ìjọba", "ọrịa coronavirus na Naijirịa", "wetin dey happen for Lagos",
"habari za uchaguzi Tanzania", "Nigeria's debt to the World Bank".

```bash
# index (EmbeddingGemma 2; or --model me5-small for an ungated CPU-only check)
uv run afriembed index-docs demo/corpus --out indexes/demo
# typed query in Hausa, English documents also searchable
uv run afriembed search indexes/demo --query "yaushe ake shuka masara?"      # when is maize planted?
uv run afriembed search indexes/demo --query "how do I prevent malaria"
# spoken query (16 kHz wav of someone asking in Hausa), with the speech adapter
uv run afriembed search indexes/demo --audio question.wav --adapter artifacts/audio-lora
# web UI: React + Tailwind front end over a FastAPI backend, microphone recorded in the browser as 16 kHz WAV
uv run afriembed search indexes/demo --ui --adapter artifacts/audio-lora     # http://localhost:7860
# (first time: uv sync --extra ui --extra dev; rebuild the front end after editing web/src with: cd web && pnpm install && pnpm build)
# --gradio gives the plain Gradio page instead
```

## Spoken queries (verified on an RTX 3050 4 GB, 2026-10-15)

`demo/queries/` holds nine 16 kHz clips. The three used here are Hausa: two questions synthesised with MMS-TTS
(`hausa_when_to_plant_maize.wav`, `hausa_how_to_prevent_malaria.wav`) and one real FLEURS
recording about mosquitoes carrying a virus (`fleurs_hau_105_mosquito_virus.wav`). The other six belong to the
stress test below. Apart from the FLEURS recording (CC-BY 4.0), every clip was synthesised with MMS-TTS voices
(Meta's for Hausa, Yoruba and Swahili, a community fine-tune for Igbo). Meta's MMS-TTS weights are CC-BY-NC 4.0,
so treat the synthetic clips as non-commercial.

```bash
uv run afriembed search indexes/demo --audio demo/queries/hausa_when_to_plant_maize.wav --adapter artifacts/audio-lora
# 0.654  hausa_farming   (maize)        0.541  hausa_health
uv run afriembed search indexes/demo --audio demo/queries/hausa_how_to_prevent_malaria.wav --adapter artifacts/audio-lora
# 0.640  hausa_health    (malaria)      0.558  hausa_farming
uv run afriembed search indexes/demo --audio demo/queries/fleurs_hau_105_mosquito_virus.wav --adapter artifacts/audio-lora
# 0.505  hausa_health    (malaria)      0.459  hausa_farming
```

Indexing the four documents took 2.1 s on the laptop GPU; each spoken query returns in a few seconds.

## Stress test

`scripts/demo_stress_test.py` runs queries with known target articles and prints the target's rank, for the
embeddings and, for typed queries, for a BM25 keyword baseline over the same chunks. Re-run on 2026-10-19 with the
364-document index and the audio adapter:

| Query set | Scored | Embeddings, target at rank 1 | BM25, target at rank 1 |
|---|---|---|---|
| Typed, sharing some words with the target (paraphrase, cross-lingual, stripped diacritics, code-switching, near-duplicates, bare entities) | 22 | 18 | 18 |
| Typed, no word in common with the target | 12 | 6 | 0 |
| Spoken, Hausa, Yoruba, Igbo and Swahili | 6 | 3 (5 in the top 3) | - |

An off-topic query and the FLEURS clip have no target and are not scored. Misses are the known weaknesses:
Hausa queries for English articles, queries in one African language for an article in another, a deep Hausa
paraphrase with no entity named, and Igbo with diacritics stripped.

## Tools

### P1. Dataset validator

```bash
# your own data: a csv/tsv/jsonl with `audio` and `text` columns (audio paths relative to the file)
uv run afriembed validate --manifest my_clips.csv --out reports/my_clips
# self-test on a public source, 10% of labels deliberately shuffled
uv run afriembed validate --source naija-hau --max-items 1000 --inject 0.1
```
Writes `report.csv` (pass / borderline / fail per clip, with the sentence the clip actually sounds like) and
`summary.json`. On NaijaVoices Hausa, BibleTTS Hausa and WAXAL Luganda it caught 251 of 251 injected wrong labels with 0.93-1.00
precision; see docs/RESULTS.md "Tools".

### P2. Voice-note search

```bash
uv run python scripts/build_voicenotes_demo.py --n 120          # 120 real NaijaVoices Hausa clips -> demo/voicenotes
uv run afriembed index-audio demo/voicenotes --out indexes/voicenotes --adapter artifacts/audio-lora
uv run afriembed search indexes/voicenotes --ui --adapter artifacts/audio-lora   # speak or type; results play inline
uv run afriembed voice-notes --langs hau,yor,ibo,swh,amh,som     # the measurement (another reader, same sentence)
```

### P4-P5. Spoken intent, overlap check

```bash
uv run afriembed spoken-intent --langs hau,yor,swa          # Injongo intents, typed vs spoken (MMS-TTS), no ASR
uv run afriembed dedup --a naija-hau --b fleurs:hau:test    # which FLEURS test sentences are spoken in NaijaVoices?
uv run afriembed dedup --a fleurs:hau:test@1 --b fleurs:hau:test   # positive control: other readers, same sentences
```

### P7-P8. Read-aloud game, archive search

```bash
uv run afriembed search indexes/voicenotes --ui --adapter artifacts/audio-lora   # header gains a "read aloud" tab
uv run python scripts/build_archive_demo.py --lang hau --minutes 12               # 23 min of continuous Hausa speech
uv run afriembed index-audio demo/archive --out indexes/archive --adapter artifacts/audio-lora
uv run afriembed search indexes/archive --ui --adapter artifacts/audio-lora      # type or speak; play the window
uv run python scripts/archive_eval.py                                            # window hit@1/@3 for 97 known verses
```
On a phone on the same Wi-Fi: open the address the server prints, then "add to home screen".

### P6. Align long audio to its text (no ASR)

```bash
uv run afriembed align run chapter.wav verses.txt      # one sentence per line -> chapter.aligned.csv (start_s, end_s, confidence, text)
uv run afriembed align eval --lang hau --n-chapters 10 # synthetic BibleTTS chapters with known boundaries
```
