---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: peft
tags: [embeddinggemma, african-languages, speech-retrieval, audio, lora, multimodal-embedding]
language: [ha, yo, sw, am, ig, so, xh, zu, af, wo, lg, ln, sn, ee]
datasets: [mteb/fleurs]
---

# EmbeddingGemma 2 · African speech-retrieval LoRA

Ugochukwu Onyebuchi, Carnegie Mellon University / NaijaML.

A LoRA adapter (5M parameters; rank 16, alpha 32, on the text tower's attention and MLP projections) for
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89`. With it, a spoken
African-language utterance embeds next to its own transcript. It was trained for one epoch (about one H100-hour) on
9,000 FLEURS-train (audio, transcript) pairs in Hausa, Yoruba, Swahili, Amharic, Igbo and Somali, with
`CachedMultipleNegativesRankingLoss`. The audio and vision encoders are frozen.

Typical uses are spoken search over text documents in the same language without an ASR model, typed or spoken
search over recordings indexed in 10-second windows, and matching untranscribed clips to candidate sentences when
building speech corpora. It does not transcribe or translate, and it barely helps cross-lingual retrieval: African
speech to English text goes from 0.434 to 0.484 over 13 languages.

## Results

### FLEURS test (speech → transcript Recall@1, pool ≈ 300 sentences per language)

| Language | base | + adapter |
|---|---|---|
| Amharic | 0.882 | 0.986 |
| Igbo | 0.786 | 0.983 |
| Somali | 0.871 | 0.991 |
| Hausa | 0.949 | 1.000 |
| Yoruba | 0.953 | 1.000 |
| Swahili | 0.971 | 0.997 |
| Xhosa (not in training) | 0.937 | 1.000 |
| Zulu (not in training) | 0.965 | 1.000 |
| Wolof, Luganda, Lingala, Shona (not in training) | 0.945–0.962 | 0.996–1.000 |
| Afrikaans (not in training, but its test sentences occur in the training split of other languages; not a transfer result) | 0.991 | 1.000 |
| **mean, 14 languages** | **0.938** | **0.997** |

### Held-out sources (outside FLEURS and outside this adapter's training data)

| Source | Pool | base | + adapter |
|---|---|---|---|
| BibleTTS Hausa, read verses | 998 | 0.861 | 0.994 |
| BibleTTS Ewe, a language in neither FLEURS nor the training set | 409 | 0.883 | 0.995 |
| NaijaVoices Hausa, crowd recordings (~3 s) | 1000 | 0.823 | 0.959 |
| NaijaVoices Igbo | 998 | 0.599 | 0.837 |
| NaijaVoices Yoruba | 980 | 0.765 | 0.913 |

### Negative control

With transcripts shuffled so that no clip is paired with its own text, Recall@1 is 0.003 for Hausa, Igbo and Amharic,
with and without the adapter, i.e. exactly chance for a pool of 300. The metric is not leaking targets.

### Seed variance

Retraining with two more seeds (same data and settings) gives a 14-language FLEURS mean of 0.9964-0.9965. On WAXAL
spontaneous speech in five unseen languages (600 clips each), the mean is 0.926-0.928 and Ewe, the hardest, 0.782-0.787.
This card's weights are seed 0.

### Matryoshka

128-dimensional embeddings retain 0.99 of full quality with the adapter (0.82 without), so a 6x smaller index costs about one point.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None})
model.load_adapter("naijaml/embeddinggemma-2-african-speech-lora")

audio = model.encode([{"audio": "question.wav"}])                       # 16 kHz mono, no prefix
docs = model.encode(["Bola Tinubu shi ne shugaban Najeriya."], prompt_name="Document")
print(model.similarity(audio, docs))
```

Use bf16 or fp32 (never fp16). Transcripts and documents take the `Document` prompt; audio takes no prefix.

## Verify it yourself

Put `name.wav` (16 kHz mono) and `name.txt` pairs in a folder, then:

```bash
git clone https://github.com/naijaml/afriembed && cd afriembed && uv sync   # public with the preprint
uv run python scripts/try_adapter.py --folder my_clips      # or: --lang hau --n 40  (uses FLEURS)
```

It prints Recall@1 for the base model, for the adapter, and for a shuffled control, plus the hardest clip's top-3
candidates. If the adapter is working you will see the middle number rise and the control stay near 1/pool.

## Caveats

FLEURS test shares domain, and possibly speakers, with FLEURS train; the held-out table above is the clean evidence.
Pools are hundreds to a thousand candidates; real indexes can be larger, so expect lower absolute numbers there.

## Reproduce

```bash
afriembed finetune-audio --langs hau,yor,swh,amh,ibo,som --out checkpoints/audio-lora --epochs 1
afriembed speech-text --langs fleurs_core --adapter checkpoints/audio-lora/adapter
afriembed heldout --adapter checkpoints/audio-lora/adapter
afriembed speech-text --langs hau,ibo,amh --adapter checkpoints/audio-lora/adapter --shuffle-control
```

## Citation

```bibtex
@misc{onyebuchi2026afrispeechlora,
  title        = {EmbeddingGemma 2 African speech-retrieval LoRA},
  author       = {Onyebuchi, Ugochukwu},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/naijaml/embeddinggemma-2-african-speech-lora}}
}
```
