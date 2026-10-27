#!/usr/bin/env bash
# Everything on H100s, in parallel jobs that do not share dataset downloads.
#   A: text benchmarks (mteb datasets, SIB-200)        B: speech / vision / CIRAL (FLEURS, AfriSpeech, Afri-Aya)
#   C: fine-tunes + their re-evaluations (after A & B)  D: AfriMTEB Full for v2 (long; parallel with C)
# Logs: logs/modal-{A,B,C,D}.log.  Results: `modal volume get afriembed-vol results ./results`
set -uo pipefail
cd "$(dirname "$0")/.."
export AFRIEMBED_GPU=${AFRIEMBED_GPU:-H100}
export AFRIEMBED_TIMEOUT=${AFRIEMBED_TIMEOUT:-43200}
mkdir -p logs
run() { local name=$1; shift; uv run modal run modal_app.py --cmds "$*" > "logs/modal-$name.log" 2>&1; echo "job $name exit $?"; }

A="afrimteb --suite lite --tasks MasakhaNEWSClassification,KinNewsClassification --max-seq-length 8192 --batch-size 32 --tag fullctx ;
   afrimteb --suite lite --no-overwrite --max-seq-length 8192 --batch-size 128 --tag fullctx ;
   afrimteb --suite lite --no-prompts --tasks MasakhaNEWSClassification,KinNewsClassification --max-seq-length 8192 --batch-size 32 --tag fullctx ;
   afrimteb --suite lite --no-prompts --no-overwrite --max-seq-length 8192 --batch-size 128 --tag fullctx ;
   afrimteb --suite lite --model embeddinggemma-300m --tasks MasakhaNEWSClassification,KinNewsClassification --max-seq-length 2048 --batch-size 32 --tag fullctx ;
   afrimteb --suite lite --model embeddinggemma-300m --no-overwrite --max-seq-length 2048 --batch-size 128 --tag fullctx ;
   afrimteb --suite lite --model me5-base --batch-size 128 ;
   afrimteb --suite lite --model me5-small --batch-size 128 ;
   afrimteb --suite lite --model bge-m3 --batch-size 64 ;
   afrimteb --suite lite --model me5-large-instruct --batch-size 64 ;
   afrimteb --suite lite --model afrie5-large-instruct --batch-size 64 ;
   afrimteb --suite lite --dim 512 --batch-size 128 ;
   afrimteb --suite lite --dim 256 --batch-size 128 ;
   afrimteb --suite lite --dim 128 --batch-size 128 ;
   fertility ;
   vocab-trim build --langs lite --out checkpoints/trim-lite ;
   afrimteb --suite lite --trim-dir checkpoints/trim-lite --batch-size 128 --tag trimmed"

B="speech-text --langs fleurs_all --max-items 500 --batch-size 64 ;
   speech-text --model embeddinggemma-2 --langs fleurs_core --text-task none --batch-size 64 --tag texttask-none ;
   speech-bitext --langs fleurs_core --pivot eng --batch-size 64 ;
   speech-bitext --langs hau,yor,swh,amh,ibo,som --all-pairs --batch-size 64 --tag allpairs ;
   lid --langs fleurs_all --per-lang 80 --batch-size 64 ;
   accent --max-per-accent 20 --batch-size 64 ;
   vision --dataset afri-aya --batch-size 32 ;
   ciral --batch-size 256 ;
   ciral --model me5-base --batch-size 256 ;
   ciral --model embeddinggemma-300m --batch-size 256"

C="finetune-text --langs lite --out checkpoints/text-lora --epochs 1 --batch-size 128 --mini-batch-size 32 ;
   afrimteb --suite lite --adapter checkpoints/text-lora/adapter --batch-size 128 --tag text-lora ;
   ciral --adapter checkpoints/text-lora/adapter --batch-size 256 --tag text-lora ;
   finetune-audio --langs hau,yor,swh,amh,ibo,som --out checkpoints/audio-lora --epochs 1 --batch-size 32 --mini-batch-size 8 ;
   speech-text --langs fleurs_core --adapter checkpoints/audio-lora/adapter --batch-size 64 --tag audio-lora ;
   speech-bitext --langs fleurs_core --pivot eng --adapter checkpoints/audio-lora/adapter --batch-size 64 --tag audio-lora ;
   finetune-audio --langs hau,yor,swh,amh,ibo,som --out checkpoints/audio-full --epochs 1 --batch-size 32 --mini-batch-size 8 --train-audio-tower ;
   speech-text --langs fleurs_core --adapter checkpoints/audio-full/adapter --batch-size 64 --tag audio-full"

D="afrimteb --suite full --max-seq-length 2048 --batch-size 64 ;
   afrimteb --suite full --model embeddinggemma-300m --max-seq-length 2048 --batch-size 64"

run A "$A" &
PA=$!
run B "$B" &
PB=$!
wait $PA $PB
run C "$C" &
PC=$!
run D "$D" &
PD=$!
wait $PC $PD
echo "all jobs finished"
