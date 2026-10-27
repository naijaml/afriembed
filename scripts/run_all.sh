#!/usr/bin/env bash
# Full pipeline on one GPU. Each step writes results/<experiment>/<model>/*.json; `afriembed report` builds tables.
set -euo pipefail
M=${MODEL:-embeddinggemma-2}
run() { echo; echo "=== $*"; uv run afriembed "$@"; }

run smoke --model "$M" --audio
# 1 + 5  text benchmarks (+ prompt ablation, + Matryoshka)
run afrimteb --model "$M" --suite lite
run afrimteb --model "$M" --suite lite --no-prompts
run afrimteb --model embeddinggemma-300m --suite lite
run afrimteb --model embeddinggemma-300m --suite lite --no-prompts
run afrimteb --model me5-base --suite lite
run afrimteb --model bge-m3 --suite lite
run afrimteb --model "$M" --suite lite --matryoshka
# 4  tokenizer fertility
run fertility
# 2 + 5  speech <-> text (all dims from one encoding)
run speech-text --model "$M" --langs fleurs_all --max-items 500
run speech-text --model "$M" --langs fleurs_core --text-task none --tag texttask-none
# 3  speech bitext
run speech-bitext --model "$M" --langs fleurs_core --pivot eng
run speech-bitext --model "$M" --langs hau,yor,swh,amh --all-pairs --tag allpairs
# 13 cross-lingual IR
run ciral --model "$M"
run ciral --model me5-base
# 14 spoken LID / clustering, 15 accents
run lid --model "$M" --langs fleurs_all --per-lang 80
run accent --model "$M" --max-per-accent 20
# 6 vision
run vision --model "$M" --dataset afri-aya
# 9 vocab trim then re-evaluate
run vocab-trim build --model "$M" --langs lite --out checkpoints/trim-lite
run afrimteb --model "$M" --suite lite --trim-dir checkpoints/trim-lite --tag trimmed
# 7 + 8 fine-tunes then re-evaluate
run finetune-text --model "$M" --langs lite --out checkpoints/text-lora --epochs 1
run afrimteb --model "$M" --suite lite --adapter checkpoints/text-lora/adapter --tag text-lora
run finetune-audio --model "$M" --langs hau,yor,swh,amh --out checkpoints/audio-lora --epochs 1
run speech-text --model "$M" --langs fleurs_core --adapter checkpoints/audio-lora/adapter --tag audio-lora
run report
