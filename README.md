# afriembed

Code, results and adapter for *EmbeddingGemma 2 on African Languages: Corrected Text Benchmarks and Speech
Retrieval with a 5M-Parameter Adapter* (Onyebuchi, 2026). The paper is in `paper/main_acl.pdf`.

The harness evaluates Google's EmbeddingGemma 2 (text, image, audio and video in one 768-d space) and its
predecessor on African languages, under the same settings as five text baselines. Main findings:

- AfriMTEB-Lite: EmbeddingGemma 300m scores 45.0 on the 12-task macro, not the published 29.1. Running it with a
  causal attention mask, as transformers did before 4.57, gives 30.5. EmbeddingGemma 2 scores within a point of v1.
- Speech: spoken utterances retrieve their transcript at 0.934 Recall@1 across 21 African languages (FLEURS,
  zero-shot). A 5M-parameter LoRA trained on six languages raises the 14 languages it was evaluated on
  from 0.938 to 0.997, including seven it never saw in training.
- Cross-lingual alignment is weak: African speech retrieves English text at 0.43.

The adapter is on Hugging Face:
[naijaml/embeddinggemma-2-african-speech-lora](https://huggingface.co/naijaml/embeddinggemma-2-african-speech-lora).

## Setup

```bash
uv sync --extra dev
cp .env.example .env            # set HF_TOKEN; accept the google/embeddinggemma-2 license on Hugging Face first
uv run afriembed smoke --audio  # loads the model and prints a few similarities
```

Python 3.13, transformers 5.19, sentence-transformers 6.1, mteb 2.24. Evaluation fits a 4 GB GPU in bf16. Do not
use fp16: EmbeddingGemma produces NaN embeddings in it.

## Reproducing the paper

```bash
# Text (Section 3)
uv run afriembed afrimteb --suite lite                                   # EmbeddingGemma 2
uv run afriembed afrimteb --suite lite --model embeddinggemma-300m       # v1
uv run afriembed afrimteb --suite lite --model embeddinggemma-300m --causal-attention   # reproduces 29.1
uv run afriembed afrimteb --suite lite --model me5-base                  # also me5-small, me5-large-instruct,
                                                                         # bge-m3, afrie5-large-instruct
# Speech (Section 4)
uv run afriembed speech-text --langs fleurs_all
uv run afriembed finetune-audio --langs hau,yor,swh,amh,ibo,som --out checkpoints/audio-lora --epochs 1
uv run afriembed speech-text --langs fleurs_core --adapter checkpoints/audio-lora/adapter
uv run afriembed speech-text --langs hau,ibo,amh --adapter checkpoints/audio-lora/adapter --shuffle-control
uv run afriembed heldout --adapter checkpoints/audio-lora/adapter        # BibleTTS, NaijaVoices
uv run afriembed heldout --sources waxal-lug,waxal-orm,waxal-sna,waxal-lin,waxal-ewe --max-items 600 --adapter checkpoints/audio-lora/adapter
uv run afriembed cascade --langs hau,yor,swh,amh,ibo,som,xho,zul,wol,lug
uv run afriembed largepool --langs hau,yor,swh --adapter checkpoints/audio-lora/adapter

# Cross-lingual and other sections
uv run afriembed speech-bitext --langs fleurs_core --pivot eng
uv run afriembed vision --dataset afri-aya
uv run afriembed ciral
uv run afriembed fertility
uv run afriembed lid --langs fleurs_all

uv run afriembed report                                                  # results/REPORT.md
uv run python scripts/make_figures.py                                    # docs/figures/
```

`uv run afriembed --help` lists every command, including the tools from Section 6 (`validate`, `voice-notes`,
`dedup`, `align`, `index-audio`, `index-docs`, `search`). `docs/EXPERIMENTS.md` describes each one.

The paper's runs used Modal: `modal run modal_app.py --cmd "<any command above>"`. Set `AFRIEMBED_GPU` to pick
the GPU (default A10G). Results land in the `afriembed-vol` volume; fetch them with
`modal volume get afriembed-vol results ./results`. The whole set took about 9 H100-hours and 4 A10G-hours.

## Results format

Each run writes `results/<experiment>/<model>/<run-id>.json` with the metrics, the prompts used, the model
revision and the library versions. `latest.json` points at the most recent run, so the paper's numbers are tied
to named run files in `docs/RESULTS.md`. The canonical v1 AfriMTEB-Lite run is
`results/afrimteb-lite/embeddinggemma-300m/20261014-195419-fullctx.json`.

## Layout

```
afriembed/cli.py          one command per experiment
afriembed/models/         prompts per model family, loading, LoRA adapters, Matryoshka truncation
afriembed/data/           FLEURS, BibleTTS, NaijaVoices, WAXAL, CIRAL, Afri-Aya, AfriSpeech-200 loaders
afriembed/experiments/    one module per experiment
configs/                  language and model registries
paper/                    LaTeX source (IEEE and ACL versions)
results/                  every run behind the paper
artifacts/audio-lora/     the released adapter
tests/                    unit tests that run without the model
```

## Citation

```bibtex
@misc{onyebuchi2026embeddinggemma2african,
  title  = {EmbeddingGemma 2 on African Languages: Corrected Text Benchmarks and Speech Retrieval
            with a 5M-Parameter Adapter},
  author = {Onyebuchi, Ugochukwu},
  year   = {2026}
}
```

Code and adapter: Apache 2.0. Datasets keep their own licenses; none are redistributed here.
