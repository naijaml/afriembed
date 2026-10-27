# Results: EmbeddingGemma 2 on African languages

All runs: Modal H100, bf16, 2026-10-14. Raw JSON per run is in `results/` (synced from the `afriembed-vol`
volume); `results/REPORT.md` is the auto-generated full table set. Caveats that apply everywhere are listed at the end.

## 1. AfriMTEB-Lite (13 tasks, 9 languages)

| Model | Params | Macro | Flores bitext | NTREX bitext | Belebele | AfriXNLI | MasakhaNEWS | SIB-200 | Intent | AfriSenti | AfriHate | Emotion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AfriE5-large-instruct | 560M | 66.2 | 97.9 | 94.8 | 76.4 | 68.8 | 83.5 | 71.9 | 75.0 | 55.1 | 50.9 | 31.3 |
| mE5-large-instruct | 560M | 64.7 | 97.8 | 94.4 | 76.0 | 64.2 | 82.9 | 71.0 | 75.5 | 49.7 | 51.1 | 29.6 |
| bge-m3 | 568M | 57.5 | 85.0 | 82.8 | 69.9 | 68.3 | 76.8 | 55.6 | 74.8 | 51.2 | 49.5 | 28.0 |
| mE5-base | 278M | 57.3 | 91.3 | 88.6 | 62.9 | 63.5 | 74.9 | 54.9 | 75.0 | 51.3 | 49.0 | 29.3 |
| mE5-small | 118M | 53.5 | 82.8 | 81.7 | 53.0 | 62.1 | 73.4 | 52.5 | 65.0 | 46.9 | 47.2 | 28.0 |
| EmbeddingGemma v1 (300m) | 308M | 45.3 | 42.1 | 49.0 | 54.9 | 59.2 | 75.0 | 44.2 | 59.5 | 44.2 | 48.1 | 26.5 |
| EmbeddingGemma 2 | 271M | 44.8 | 39.5 | 47.2 | 51.8 | 59.7 | 75.0 | 43.6 | 57.0 | 49.7 | 47.9 | 28.0 |

- **The published EmbeddingGemma row (29.1, EACL 2026 AfriMTEB paper) does not reproduce.** Our v1 run scores 45.0 on the
  paper's own 12 tasks (Belebele 54.9 vs 10.6, Flores 42.1 vs 12.2, NTREX 49.0 vs 17.9).
- **Prompts are worth 1-2 points** (v2: 44.9 -> 43.9 without; v1: 45.4 -> 43.4; 13-task macros), so they do not explain the gap.
- **Full 8K context changes nothing** on the two long-document tasks (MasakhaNEWS 75.0 vs 75.5 at 1K).
- **v2 == v1 on African text.** Same tokenizer (identical fertility on 30 languages), same scores.
- **Baselines re-run through mteb's registry** land about 3 points above the paper on its 12-task macro (mostly SIB-200 v2, which
  is missing four languages upstream, plus Flores, NTREX and AfriSenti); the ranking is unchanged.
- **Matryoshka (v2):** 512d 44.5 (0.99), 256d 43.5 (0.97), 128d 40.5 (0.90) of the 44.9 full-dimension run; bitext degrades fastest.
- **Vocab trim** to 23k tokens (Lite languages + English, 98% held-out coverage, -245 MB bf16): macro 44.5 -> 39.6.
- **Text LoRA (45k caption pairs, 1 epoch):** macro 44.8 -> 44.0; bitext +1.5, intent -8.9. Negative at this scale.


### Reconciliation with the AfriMTEB paper (ours minus paper, per task)

| Task | mE5-large-instruct | AfriE5 | bge-m3 | EmbeddingGemma v1 |
|---|---|---|---|---|
| AfriHate | -0.4 | -0.8 | -0.6 | +2.7 |
| AfriSenti | +2.7 | +4.4 | +3.3 | +5.2 |
| AfriXNLI | -0.3 | -0.2 | -0.4 | +5.6 |
| Belebele retrieval | +0.3 | -1.3 | -0.0 | **+44.3** |
| EmotionAnalysisPlus | -1.9 | -1.5 | -1.4 | -0.1 |
| Flores bitext | +6.4 | +6.7 | +6.9 | **+29.9** |
| NTREX bitext | +2.9 | +2.8 | +2.4 | **+31.1** |
| Injongo intent | +0.0 | -0.4 | -0.6 | +10.1 |
| News (MasakhaNEWS + KinNews merged over languages, as the paper does) | -0.4 | -0.3 | -0.3 | +10.8 |
| SIB-200 v2 (4 languages missing upstream; not comparable) | +27.1 | +26.1 | +24.1 | +24.8 |
| SIB-200 classification | -0.2 | -0.1 | +0.0 | +14.6 |
| SIB-200 clustering | -0.2 | -0.0 | +0.1 | +12.0 |

Seven tasks reproduce the paper's baselines within 1.3 (Belebele, AfriXNLI, Injongo, SIB-200 classification and
clustering, AfriHate, News). Flores, NTREX and AfriSenti run 2-7 points higher for every baseline (and Emotion 1.4-1.9 lower), consistent with
task-definition changes between the paper's mteb version and 2.24. SIB-200 v2 is excluded from paper comparisons. The
EmbeddingGemma correction rests on the seven reproducing tasks, where v1 differs from the paper by +3 to +44.
(Corrected 2026-10-18: this table previously compared MasakhaNEWS alone, +4, against the paper's merged News column,
and claimed agreement within 0.6, but AfriE5 differs by 1.3 on Belebele and 0.8 on AfriHate.)

**What the paper actually specifies (read 2026-10-15, arXiv:2510.23896v2).** No evaluation framework version, prompts,
loader, precision or sequence length for any model; its limitations say "some open models may be sensitive to prompt
formats or poolers we did not exhaustively tune." AfriMTEB-Lite is 12 tasks: MasakhaNEWS and KinNews are one "News"
task, and "SIB-200 14 Classes" covers 56 languages including all nine Lite languages. Scores are macro over languages
within a task, then over tasks. The paper does not state which Flores/NTREX language pairs it mines; mteb's Lite
definition (which we ran) mines only the 72 African-African Flores pairs and 30 NTREX pairs among the nine languages,
with no English, so our bitext columns are not guaranteed to be the paper's. mteb also evaluates SIB-200
classification on train, validation and test splits (mE5: 71.5 / 69.8 / 71.8); the paper's split is unstated.

Re-aggregated the paper's way (12 tasks, news merged):

| Model | Our 13-task macro | Paper-style 12 | 11 (SIB-200 v2 dropped) | Paper |
|---|---|---|---|---|
| EmbeddingGemma v1 | 45.3 | 45.0 | 46.7 | 29.1 |
| EmbeddingGemma 2 | 44.8 | 44.4 | 46.2 | - |
| mE5-large-instruct | 64.7 | 65.0 | 66.5 | 62.0 |
| AfriE5-large-instruct | 66.2 | 66.6 | 67.9 | 63.7 |
| bge-m3 | 57.5 | 57.8 | 59.9 | 55.0 |

Merging the news tasks moves every macro by under 0.5; dropping SIB-200 v2 raises each by a further 1.3-2.1. Every baseline runs about 3 points above the paper; v1 runs
16 above. The correction claim rests on the seven tasks whose baselines reproduce the paper within 1.3.

**Bitext with English pairs (rerun 2026-10-15, Flores devtest / NTREX test, F1 macro over pairs):**

| Model | Flores eng↔X (18 pairs) | Flores X↔X (72) | NTREX eng↔X (14) | NTREX X↔X (30) | Paper Flores | Paper NTREX |
|---|---|---|---|---|---|---|
| EmbeddingGemma v1 | 53.2 | 42.2 | 52.5 | 49.0 | 12.2 | 17.9 |
| EmbeddingGemma 2 | 51.4 | 39.5 | 49.6 | 47.2 | - | - |
| mE5-large-instruct | 99.2 | 97.9 | 98.1 | 94.3 | 91.4 | 91.5 |
| AfriE5-large-instruct | 99.3 | 97.9 | 98.1 | 94.8 | 91.2 | 92.0 |
| bge-m3 | 91.6 | 85.0 | 88.9 | 82.8 | 78.1 | 80.4 |

English pairs score higher than African-African pairs for every model. No pair set reproduces the paper's baseline
bitext numbers (ours are 2-14 points above theirs under either set), so the paper's bitext protocol differs from mteb
2.24's task in some way we cannot recover from the text. EmbeddingGemma v1 scores 42-53 on Flores under both sets,
against the published 12.2; Swahili and Amharic pairs with English reach 84-94 even for v1.



### The published 29.1 reproduced: EmbeddingGemma run with a causal attention mask (2026-10-17)

EmbeddingGemma is a bidirectional encoder; its config sets `use_bidirectional_attention: true`, a flag added in a
transformers pre-release (`4.57.0.dev0` in the model's config). A loader that ignores the flag runs the encoder with
a causal (left-to-right) mask. Rerunning v1 on AfriMTEB-Lite with `--causal-attention` (same harness, bf16, A10G):

| Task | Correct (bidirectional) | Causal mask | AfriMTEB paper |
|---|---|---|---|
| SIB-200 classification | 44.2 | 29.5 | 29.6 |
| AfriHate | 48.1 | 46.4 | 45.4 |
| Emotion | 26.5 | 24.8 | 26.6 |
| AfriXNLI | 59.2 | 55.6 | 53.6 |
| SIB-200 clustering | 16.6 | 3.4 | 4.6 |
| Flores bitext | 42.1 | 14.0 | 12.2 |
| NTREX bitext | 49.0 | 20.1 | 17.9 |
| Belebele retrieval | 54.9 | 16.4 | 10.6 |
| Injongo intent | 59.5 | 41.8 | 49.4 |
| News (MasakhaNEWS + KinNews, language-averaged) | 69.8 | 57.2 | 59.0 |
| SIB-200 v2 (not comparable, 4 languages missing upstream) | 26.2 | 14.2 | 1.4 |
| AfriSenti | 44.2 | 43.0 | 39.0 |
| **Macro, paper's 12-task aggregation** | **45.0** | **30.5** | **29.1** |
| Macro, 11 tasks (SIB-200 v2 dropped) | 46.7 | 32.0 | 31.6 |

Correction 2026-10-18: an earlier version of this table said 29.4, which merged the two news tasks by a plain
average while the 45.0 beside it averaged news over languages (the paper's method). Consistently aggregated,
causal is 30.5 vs 29.1, and 32.0 vs 31.6 on the 11 comparable tasks. Runs: 20261014-195419-fullctx (correct, the
leaderboard run; 20261014-185236 at 1K context differs only on News, 70.0) and 20261017-072853-causal.

The causal run reproduces the published macro to within 1.4 points (0.4 on comparable tasks) and the published per-task pattern: sentiment, hate, emotion and NLI lose 1-4 points, topic and intent classification 13-18, and bitext, retrieval and clustering collapse. AfriSenti sits 4
above the paper here exactly as every baseline does (task-definition drift, see the reconciliation above).

Ruled out on the same day: fp32 (45.0 on 12 tasks, equal to bf16), a 128-token limit (44.1), dropping the Dense and Normalize
modules (Flores 36.8, AfriHate 48.3), and fp16 (every embedding NaN; classification tasks crash, retrieval and
bitext at chance, so fp16 cannot produce a score of 29). The correction is now a diagnosis: the AfriMTEB paper's
EmbeddingGemma row was measured with causal attention, most likely a transformers version that predates
EmbeddingGemma support.

Prior reports of the same fallback (found 2026-10-18): sentence-transformers issue #3725 (Apr 2026, transformers
< 4.57 silently causal; in-house Recall@20 0.465 vs 0.669), mteb issue #3144 (Sep 2025, WikipediaRerankingMultilingual
map 0.81 vs 0.93, fixed by changing transformers), HF model discussion #36, and an MLX Swift port bug. None mentions
AfriMTEB. The AfriMTEB EACL 2026 camera-ready (Jan 2026) still prints 29.1, and the SOTA2 leaderboard copies it.
No other source reports EmbeddingGemma on AfriMTEB-Lite.

EmbeddingGemma 2 does not carry the flag: its config has no `use_bidirectional_attention` key at any level, and the
same `--causal-attention` run scores 44.8, identical to the normal run. So the switch is a no-op for v2 and this
particular failure cannot happen to v2 through that flag; we make no claim beyond that.

## 1b. AfriMTEB Full (33 text tasks, 59 languages)

EmbeddingGemma 2: macro **52.0**; EmbeddingGemma v1: **52.6** (paper: AfriE5 62.4, mE5-large 61.3, bge-m3 55.8).
Weakest languages (v2): Tamazight 6.4, N'Ko 14.5, Tamasheq 22.4, Nuer 23.4, Kabyle 24.0. Strongest: Nigerian Pidgin 59.3,
Portuguese 54.2, Venda 51.7, Afrikaans 51.6, Swahili 50.6. BibleNLP bitext 19.1; language-ID classification 75-83.
v2 vs v1 differ by more than 3 points on only a few tasks (v2 better on AfriSenti language-ID +10, SouthAfricanLang +3;
v1 better on Siswati news +6, Swahili news +5, NollySenti bitext +4, SIB-200 clustering +4).

## 2. Speech -> text retrieval, FLEURS, 21 African languages (zero-shot)

Mean R@1 **0.934** at 768d (pools ~300 unique sentences per language), text->audio 0.940.
Top: Afrikaans 0.991, Kabuverdianu 0.985, Luo 0.974, Swahili 0.971, Zulu 0.965, Fulfulde 0.964, Wolof 0.962.
Bottom: Igbo 0.786, Umbundu 0.806, Somali 0.871, Kamba 0.879, Amharic 0.882.
Matryoshka: 512d 0.991, 256d 0.958, 128d 0.816 of full. Removing the `document` prefix on transcripts costs up to 23 points (0-23)
(Somali 0.871 -> 0.643, Amharic 0.882 -> 0.777). Resource tier barely predicts audio performance.

## 2b. Held-out speech check (non-FLEURS sources, zero-shot vs audio LoRA)

| Source | Lang | Pool | Zero-shot R@1 | + audio LoRA |
|---|---|---|---|---|
| BibleTTS Hausa (read verses) | hau | 998 | 0.861 | 0.994 |
| BibleTTS Ewe (language in neither FLEURS nor the adapter's training) | ewe | 409 | 0.883 | 0.995 |
| NaijaVoices Hausa (crowd recordings, ~3 s) | hau | 1000 | 0.823 | 0.959 |
| NaijaVoices Igbo | ibo | 998 | 0.599 | 0.837 |
| NaijaVoices Yoruba | yor | 980 | 0.765 | 0.913 |

The FLEURS result holds on other sources with 3x larger pools, and the adapter transfers to an unseen source and an
unseen language, so FLEURS leakage cannot explain the speech results.

## 2c. Cascade baseline (MMS-1B-all ASR -> text embedding -> retrieval)

| Language | MMS WER | Cascade R@1 | Direct zero-shot R@1 | Direct + audio LoRA R@1 |
|---|---|---|---|---|
| Hausa | 0.256 | 0.993 | 0.953 | 1.000 |
| Yoruba | 0.535 | 0.997 | 0.957 | 1.000 |
| Swahili | 0.147 | 0.997 | 0.970 | 0.997 |
| Amharic | 0.273 | 1.000 | 0.882 | 0.986 |
| Igbo | 0.442 | 0.977 | 0.777 | 0.983 |
| Somali | 0.440 | 0.980 | 0.877 | 0.991 |
| Xhosa | 0.370 | 0.987 | 0.943 | 1.000 |
| Zulu | 0.337 | 0.993 | 0.963 | 1.000 |
| Wolof | 0.388 | 0.996 | 0.962 | 1.000 |
| Luganda | 0.439 | 0.983 | 0.957 | 0.997 |
| **mean** | **0.363** | **0.990** | **0.924** | **0.995** |

Pools of ~300 (FLEURS test). Retrieval from noisy MMS transcripts is robust even at 54% WER (Yoruba): the cascade beats
the zero-shot direct path by 7 points on average (20 on Igbo). The adapted model matches or beats the cascade in every
language in one pass, with no ASR model (1B parameters) and no transcription step.

## 2d. Large-pool test (speech queries vs nested candidate pools, R@1 at 768d)

Queries: 300 FLEURS test clips per language plus 300 NaijaVoices crowd clips for Hausa and Yoruba. Pools grow from the
queries' own transcripts to all FLEURS sentences, 50k CIRAL news passages, and those passages split into sentences.

| Language | Pool 300-600 | ~2k | ~52k | ~235-273k | ~235-273k, R@10 |
|---|---|---|---|---|---|
| Hausa, zero-shot | 0.863 | 0.860 | 0.840 | 0.700 | 0.822 |
| Hausa, + audio LoRA | 0.988 | 0.988 | 0.988 | **0.967** | 0.988 |
| Yoruba, zero-shot | 0.878 | 0.832 | 0.770 | 0.530 | 0.727 |
| Yoruba, + audio LoRA | 0.973 | 0.973 | 0.973 | **0.943** | 0.961 |
| Swahili, zero-shot | 0.970 | 0.947 | 0.867 | 0.697 | 0.863 |
| Swahili, + audio LoRA | 0.997 | 0.997 | 0.990 | **0.987** | 0.993 |

At 256 dimensions the adapter still scores 0.953 / 0.918 / 0.967 at the largest pools (zero-shot 0.622 / 0.475 / 0.620).
Crowd-recorded NaijaVoices queries at the largest pool: Hausa 0.76 -> 0.94, Yoruba 0.48 -> 0.89.

## 3. Cross-lingual speech bitext, FLEURS (n-way parallel ids), 13 languages -> English

| Direction | Base | + text LoRA | + audio LoRA | both stacked |
|---|---|---|---|---|
| audio -> English text | 0.434 | 0.487 | 0.484 | 0.526 |
| audio -> English audio | 0.409 | 0.481 | 0.446 | 0.492 |
| text -> English text | 0.597 | 0.619 | 0.597 | 0.644 |
| text -> English audio | 0.554 | 0.573 | 0.546 | 0.602 |

Stacking (both LoRA deltas active) is additive: every one of the 13 languages improves on audio -> English text
(Lingala 0.38 -> 0.52, Xhosa 0.44 -> 0.57, Hausa 0.34 -> 0.45).

Stacked adapters within-language (speech -> own transcript, FLEURS): the run was cut off after five languages
(Amharic 0.966, Oromo 1.000, Igbo 0.974, Yoruba 1.000, Hausa 0.997), i.e. 1-2 points below the audio adapter alone
(0.986, 1.000, 0.983, 1.000, 1.000). Stacking trades a little within-language accuracy for the cross-lingual gain.


Swahili (0.72) and Amharic (0.63) are the only well-aligned languages; most sit at 0.33-0.45. African<->African pairs
average 0.34. The bottleneck is the text space, not the audio tower.

## 4. Tokenizer fertility (SIB-200 sentences, tokens per word relative to English)

Gemma (v1 and v2 identical): 1.95x; XLM-R: 1.66x; NLLB-200: 1.33x. Worst: Zulu/Xhosa/Tigrinya 2.6-2.7x, Amharic 2.4x.
Byte fallback < 1% everywhere except Ewe (2.5%).

## 6. Image <-> caption alignment, Afri-Aya

image->English caption R@1 **0.79** vs image->local caption **0.17**. Local<->English text alignment ~0.1 for
Kinyarwanda, Luganda, Twi, Hausa, Yoruba, Zulu; 0.96 Egyptian Arabic, 0.84 Krio, 0.38 Igbo.

## 8. Audio fine-tunes (9,000 FLEURS-train speech/transcript pairs, 6 languages, 1 epoch)

| Setting | mean a->t R@1 (14 langs) | 128d retention | Igbo | Amharic | Somali |
|---|---|---|---|---|---|
| base | 0.938 | 0.816 | 0.786 | 0.882 | 0.871 |
| LoRA r16 on text tower (5M params, ~20 MB) | 0.997 | 0.988 | 0.983 | 0.986 | 0.991 |
| + audio tower unfrozen (306M) | 0.998 | 0.991 | 0.991 | 0.993 | 0.994 |

Transfers to languages not in the fine-tune (Xhosa, Zulu, Wolof, Luganda, Lingala, Shona, Oromo -> 0.99-1.00).

**Seed variance (2026-10-27).** The LoRA row retrained with seeds 1 and 2 (`--seed`, which sets the LoRA
initialisation, data order and trainer seed; same data and hyperparameters, H100). Evaluation is unchanged.

| Seed | FLEURS mean R@1 (14) | 128d retention | WAXAL mean R@1 (5, 600 clips) | WAXAL Lingala | WAXAL Ewe |
|---|---|---|---|---|---|
| 0 (released) | 0.9965 | 0.988 | 0.926 | 0.875 | 0.783 |
| 1 | 0.9965 | 0.989 | 0.926 | 0.873 | 0.782 |
| 2 | 0.9964 | 0.989 | 0.928 | 0.877 | 0.787 |

FLEURS is at ceiling, so the seeds miss nearly the same clips (seed 2 swaps one Afrikaans miss for one Luganda hit).
On WAXAL the spread is 0.2 points on the mean and 0.5 on Ewe. Files: `results/finetune-audio/embeddinggemma-2/20261026-203024-train.json`
and `20261026-204822-train.json`; FLEURS `results/speech-text/embeddinggemma-2/20261026-230715-adapter.json` (seed 1),
`20261027-090912-adapter.json` (seed 2), `20261027-103607-adapter.json` (seed 0 re-scored locally, identical to the
H100 run); WAXAL `results/heldout/embeddinggemma-2/20261027-084119-adapter.json` (seed 1), `20261027-092957-adapter.json` (seed 2).

**Cross-language sentence overlap check (2026-10-17).** FLEURS sentence ids are shared across languages, so an
adapter trained on Hausa could have seen the meaning of a Zulu test sentence. We intersected the union of train
ids of the six training languages (1,509 sentences) with the test ids of all 21 evaluated languages. Twenty
languages: 0 overlap. **Afrikaans: all 233 test sentences overlap** (FLEURS assigns Afrikaans a different split),
so Afrikaans is excluded from every transfer claim. Without it the adapted 13-language mean is 0.9962 (was
0.9965 with 14) and the zero-shot 20-language mean 0.931 (was 0.934 with 21). NaijaVoices, BibleTTS and WAXAL
share no FLORES sentences and are unaffected. Script: `scripts/check_fleurs_overlap.py`.
Cross-lingual audio->English improves 0.434 -> 0.484 with the audio LoRA.

## 13. CIRAL (English queries -> African passages, 50k-passage pools, nDCG@20)

| Model | Hausa | Somali | Swahili | Yoruba | Mean |
|---|---|---|---|---|---|
| EmbeddingGemma v1 | 0.387 | 0.357 | 0.486 | 0.373 | 0.401 |
| EmbeddingGemma 2 | 0.352 | 0.363 | 0.452 | 0.355 | 0.380 |
| mE5-base | 0.384 | 0.357 | 0.400 | 0.377 | 0.379 |
| EmbeddingGemma 2 + text LoRA | 0.338 | 0.356 | 0.457 | 0.336 | 0.372 |

## 14. Spoken language ID, FLEURS, 21 languages, 80 clips each

Audio embeddings: kNN 0.670 (chance 0.048), NMI 0.706, purity 0.699. Text: kNN 0.708. Top confusions: Zulu<->Xhosa (44),
Swahili->Shona, Chichewa->Shona, Yoruba->Igbo.

## 15. Accents, AfriSpeech-200 (African-accented English, 108 accents, 1,967 utterances)

R@1 0.888, R@10 0.949, MRR 0.911; clinical 0.892 vs general 0.882. Kenya 0.98, South Africa 0.93, Nigeria 0.88, Ghana 0.85.
Weakest accents: Gerawa 0.15, Kanuri 0.25, Angas 0.50. The transcript prefix makes no difference for English.

## Tools (built 2026-10-15/16, all on the local RTX 3050 4 GB, $0 cloud)

### P1. Speech dataset validator (`afriembed validate`)

Each clip is embedded by the audio tower, each claimed transcript by the text tower (with the audio adapter);
a clip passes if its own transcript is its nearest prompt by a margin >= 0.02, is borderline if its transcript is
within the top 5, and fails otherwise or if the similarity is below 0.30. Self-test: 10% of labels shuffled.

| Source | Clips | Injected mismatches caught (fail) | Fail precision | Clean clips flagged (incl. borderline) | Clean clips failed |
|---|---|---|---|---|---|
| NaijaVoices Hausa (crowd, ~3 s) | 1,000 | 100 / 100 | 0.93 | 7.6% | 0.9% (8) |
| BibleTTS Hausa (studio verses) | 910 | 91 / 91 | 1.00 | 3.1% | 0 |
| WAXAL Luganda (spontaneous, ~24 s, unseen language) | 600 | 60 / 60 | 1.00 | 1.3% | 0 |

Every injected wrong label was caught as a hard fail in all three sets, including spontaneous Luganda, a language the adapter never saw. Of the eight clean NaijaVoices clips that
failed, one is a 4 s clip paired with a 20-word sentence (own similarity 0.29, rank 994 of 1,000), i.e. a genuine
label error in the dataset; the rest are short, generic sentences confused with other short sentences in a
1,000-prompt pool. Report: `reports/validate-<source>/report.csv` (status, own_sim, rank, margin, predicted text).
Throughput on the laptop GPU: about 1,000 clips in 5 minutes.

### P2. Voice-note search (`afriembed voice-notes`, UI over `index-audio`)

Query-by-example: one FLEURS recording of a sentence as the query, a *different* recording of the same sentence
(another reader) as the target, one target per sentence in the pool. Audio adapter on.

| Language | Sentences (pool) | audio->audio R@1 | R@5 | text->audio R@1 | audio->audio R@1 at 256d |
|---|---|---|---|---|---|
| Hausa | 215 | 1.000 | 1.000 | 1.000 | 1.000 |
| Yoruba | 300 | 1.000 | 1.000 | 1.000 | 1.000 |
| Igbo | 300 | 0.977 | 1.000 | 0.987 | 0.980 |
| Swahili | 175 | 1.000 | 1.000 | 0.994 | 1.000 |
| Amharic | 175 | 1.000 | 1.000 | 0.994 | 1.000 |
| Somali | 300 | 0.997 | 0.997 | 0.997 | 0.997 |
| **mean** | | **0.996** | | **0.995** | **0.996** |

Saying a sentence finds another person's recording of it essentially every time, and typing it does the same.
Caveat: the FLEURS mirror carries no speaker ids, so "another reader" is inferred from FLEURS having several
recordings per sentence; the query/target gender differs in 49% of Igbo and 41% of Swahili pairs and in under 1% of
pairs in the other four languages. The demo index (`demo/voicenotes`, 120 NaijaVoices crowd clips, built by
`scripts/build_voicenotes_demo.py`) is searchable by voice or text in the web UI with an inline player.

### P4. ASR-free spoken intent routing (`afriembed spoken-intent`, Spoken Injongo)

Injongo intents (40 classes, 2,240 train / 640 test utterances per language). Training utterances are embedded
once by the text tower; a test utterance is classified by similarity-weighted 5-NN vote (or nearest intent
centroid). Typed = text tower; spoken = MMS-TTS of the test utterance through the audio tower, no ASR.

| Language | Typed 5-NN | Typed centroid | Spoken 5-NN | Spoken centroid | Top spoken confusion |
|---|---|---|---|---|---|
| Hausa | 0.919 | 0.889 | 0.805 | 0.752 | balance -> transfer |
| Yoruba | 0.836 | 0.836 | 0.669 | 0.664 | time -> timezone |
| Swahili | 0.861 | 0.877 | 0.684 | 0.722 | freeze_account -> balance |
| Igbo | 0.814 | 0.827 | 0.134 | 0.150 | (voice unintelligible, see below) |
| **mean** | **0.857** | **0.857** | 0.573 | 0.572 | |

Typed intent classification by embedding alone is 81-92% over 40 intents with no classifier training. Spoken
Hausa reaches 0.80 against 0.92 typed; Yoruba and Swahili lose 17-18 points to the synthetic voice; Igbo collapses because
Meta ships no Igbo MMS-TTS voice and the community one used here is not intelligible to the model. Real
microphone speech is the product setting; these numbers bound it from the synthetic side only.

**Voice intelligibility check** (`scripts/tts_sanity.py`: 200 synthesised Injongo utterances, R@1 of the audio
to its own text): Hausa 0.93, Yoruba 0.95, Swahili 0.975, Igbo 0.28. The three Meta voices are heard almost
perfectly; the Igbo community voice is not, so every Igbo "spoken" number in P3 and P4 measures that voice, not
the model. Real Igbo speech scored 0.98 on FLEURS and 0.84 on NaijaVoices with the adapter.

### P3. Spoken question answering over documents (`afriembed rag spoken-eval`, Spoken AfriQA) -- negative result

AfriQA questions (Hausa 300, Yoruba 253, Swahili 295, Igbo 409) against the pool of all 1,206 English gold
Wikipedia passages. Three forms of the same question; metric is the gold passage at rank 1 / in the top 5.

| Language | Typed, English translation | Typed, African language | Spoken (MMS-TTS), African language | Spoken R@5 | Spoken top-1 agrees with typed |
|---|---|---|---|---|---|
| Hausa | 0.823 | 0.407 | 0.133 | 0.260 | 0.26 |
| Yoruba | 0.791 | 0.538 | 0.130 | 0.277 | 0.20 |
| Swahili | 0.861 | 0.729 | 0.447 | 0.603 | 0.55 |
| Igbo | 0.819 | 0.555 | 0.046 | 0.115 | 0.08 |
| **mean** | **0.823** | **0.557** | **0.189** | | |

This is the African-language-to-English direction that the paper identifies as the model's weakness, and the
spoken form inherits it and adds the synthetic-voice loss (Igbo's community voice is not intelligible to the
model; see the TTS intelligibility check). Generating an answer with google/gemma-3-1b-it from the top 3 passages
of the spoken query gives token F1 of 0.01-0.20, i.e. not usable. Conclusion: spoken QA over same-language
documents (as in the search demo) works; spoken QA over English documents needs the cross-lingual fine-tune
first. The benchmark and harness are in place to measure that fine-tune when it exists.

### P9. WAXAL spontaneous speech (`afriembed heldout --sources waxal-*`)

WAXAL (Google + African universities, released Feb 2026; community parquet mirror ngoloan/WaxalNLP) ASR test
shards: spontaneous image-prompted speech (not conversational), 17-24 s per clip, 600 clips per language, pools of 600. None of these five
languages was in the adapter's training set (Hausa, Yoruba, Swahili, Amharic, Igbo, Somali).

| Language | Mean clip | Zero-shot R@1 | Zero-shot R@5 | + audio LoRA R@1 | + LoRA R@5 | LoRA at 256d |
|---|---|---|---|---|---|---|
| Luganda | 24.2 s | 0.705 | 0.862 | **0.995** | 1.000 | 0.983 |
| Oromo | 17.0 s | 0.757 | 0.917 | **0.993** | 0.998 | 0.983 |
| Shona | 20.7 s | 0.645 | 0.878 | **0.985** | 0.990 | 0.980 |
| Lingala | 17.8 s | 0.547 | 0.747 | **0.875** | 0.907 | 0.860 |
| Ewe | 19.1 s | 0.488 | 0.677 | **0.783** | 0.928 | 0.652 |
| **mean** | | **0.628** | | **0.926** | | |

Seeds 1 and 2 of the adapter (Section 8) give means of 0.926 and 0.928, Ewe 0.782 and 0.787.

Spontaneous speech is a much harder domain than FLEURS read speech for the zero-shot model (0.63 vs 0.93), and
the 5M-parameter adapter trained on six other languages' read speech recovers most of it (+30 points on average,
+24 to +34 per language) without ever seeing spontaneous audio or these languages. Lingala and Ewe keep a gap;
Ewe's R@5 of 0.93 suggests the remaining errors are near-misses within long clips.

### P5. Cross-corpus overlap and leakage check from audio alone (`afriembed dedup`)

For every recording in corpus A, the nearest sentence in corpus B; flagged if the similarity clears a threshold
calibrated on A's own audio/transcript pairs (the 5th percentile of true-pair similarity, so by construction
about 5% of true matches are missed). Precision and recall are exact because both corpora carry transcripts.

| A (recordings) | B (sentences) | True shared | Flagged | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| FLEURS Hausa test, other reader of each sentence | FLEURS Hausa test | 331 | 314 | 1.00 | 0.95 | 0.644 |
| FLEURS Hausa test | FLEURS Hausa validation | 0 | 0 | - | - | 0.634 |
| NaijaVoices Hausa (1,000) | FLEURS Hausa test | 0 | 0 | - | - | 0.651 |
| BibleTTS Hausa (910) | FLEURS Hausa test | 0 | 0 | - | - | 0.695 |

No false positive for any of the 2,241 recordings checked against a corpus they share no sentence with, and 95% of true shared sentences found from a different
reader's recording. The strongest unrelated matches sit at 0.60-0.64, below every calibrated threshold, so the
calibration leaves a usable gap. Confirms the three Hausa sources used in this work share no sentences.

### P6. Zero-ASR alignment of long read audio to its text (`afriembed align run|eval`)

Split the recording at pauses >= 0.3 s, embed the audio of every candidate span of 1..G consecutive segments
with the audio tower and every sentence with the text tower, then an exact monotonic dynamic programme assigns
consecutive spans to consecutive sentences, maximising similarity minus a text-length duration prior. No
recogniser for the language. Evaluation: synthetic chapters of 20 BibleTTS verses concatenated with 0.4 s
pauses (true spans = the speech inside each clip), 10 chapters per language.

| Language | Verse length | Verses with IoU >= 0.5 | Mean IoU | Start within 0.5 s | Mean confidence |
|---|---|---|---|---|---|
| Hausa (studio, 7 s verses) | 7.1 s | 0.905 | 0.856 | 0.78 | 0.74 |
| Ewe (unseen language, 10 s verses with internal pauses) | 10.3 s | 0.61 | 0.57 | 0.27 | 0.53 |

Hausa aligns well enough to build verse-level corpora from chapter audio without ASR (9 of 10 chapters at
0.90-1.00). Ewe, which the adapter never saw and whose verses are long with 0.4-0.7 s silences inside and
around them, aligns 61% of verses; the mean span confidence (0.53 vs 0.74) tracks this, and the worst chapter
in each language (confidence 0.35-0.67) is flagged by it, so confidence is a usable "needs review" signal.
Before the span-embedding DP and the 0.3 s pause threshold, the first version scored 0.31 on Hausa: the pause
splitter at 0.5 s was recovering only 5% of verse boundaries. Each chapter of ~2.5 min takes 20-60 s on the
laptop GPU (about 600 span embeddings).

### P7. Read-aloud game (web UI tab "read aloud")

The page shows a sentence from `demo/voicenotes/transcripts.csv`; the learner records it; the recording is embedded
and compared with every sentence in the pool (120). Score 0-100 from the similarity to the target, a margin
bonus, and a rank penalty; verdict clear / close / unclear; if another sentence is nearer, it is shown as "it
sounded more like". End-to-end check with real NaijaVoices clips: the clip of the shown sentence scores 94
("clear", rank 1, margin 0.13); a clip of a different sentence scores 16 ("unclear", rank 75). The score is a
proxy for intelligibility to the model, not a pronunciation assessment.

### P8. Long-form archive search (`index-audio` + web UI with inline player)

`demo/archive`: 23.5 minutes of continuous Hausa read speech (five "programmes" assembled from BibleTTS verses
with 0.5 s pauses, plus four synthetic chapters), indexed as 141 ten-second windows. 97 verses with known
positions; a hit is a returned window that overlaps the verse's true span.

| Query | Window hit@1 | Window hit@3 |
|---|---|---|
| Typed verse text | 0.753 | 0.948 |
| Spoken (the verse's own recording) | 0.814 | 0.979 |

Ten-second windows hold about 1.5 verses each, so the ceiling for exact top-1 is below 1.0; top-3 is the
practical figure for a "jump to where it was said" product. The UI streams any window through `/api/audio` for
playback.

### P10. Phone and offline use

The web app ships a manifest and icon, so Chrome on a phone on the same Wi-Fi can "add to home screen" and open
it full-screen; the server prints its LAN address on start. Everything runs on the laptop; the phone only needs
the browser. No native build: the audio tower on-device would need the LiteRT export, which is out of scope here.

## Audit (2026-10-15)

- **Negative control.** `speech-text --shuffle-control` permutes transcripts so no clip is paired with its own text.
  Hausa / Igbo / Amharic (pools 300, 300, 296): R@1 = 0.003 / 0.003 / 0.003 both zero-shot and with the audio adapter,
  i.e. exactly chance (1/pool). The un-shuffled recheck in the same job gave 1.000 / 0.980 / 0.986. The metric is not
  leaking targets.
- **Extraction check.** mteb's own cached result files were compared with the AfriMTEB-Lite tables for two variants
  that only ever ran the Lite suite (EmbeddingGemma 2 without prompts; mE5-base): 13/13 tasks identical to 1e-6.
  (The Full and Lite suites briefly shared one cache directory; this never affected a reported number and is fixed.)
- **Bitext prefix ablation.** The model card defines no prefix for bitext mining. On six languages -> English (FLEURS),
  transcript prefix = search result (used throughout): audio->text 0.458, text->text 0.669, text->audio 0.621;
  sentence similarity: 0.480 / 0.659 / 0.505; none: 0.434 / 0.637 / 0.578. The search prefix is best overall and the
  choice moves cross-lingual numbers by at most 2 points, so it does not explain the alignment gap.
- **Revisions.** All models pinned; EmbeddingGemma 2 unchanged on the hub since release, so every run used `914f7f89`.
- **Bug provenance.** Every fix in the git history corresponds to a traceback preserved in `logs/`.

## Artifacts

- Audio adapter: https://huggingface.co/naijaml/embeddinggemma-2-african-speech-lora (model card included; copy in `artifacts/audio-lora`).
- Text adapter (caption pairs, weak): `artifacts/text-lora`.

## Caveats

- Models are pinned to Hugging Face revisions in `configs/models.yaml` (EmbeddingGemma 2: `914f7f89`, unchanged since its
  2026-10-06 release, so every run here used exactly that snapshot). Baselines run through mteb's registry use mteb's own pins.

- Two AfriMTEB tasks needed upstream workarounds (EmotionAnalysisPlus: empty validation split; SIB200Classification.v2:
  dataset lacks swh/xho/yor/zul). Both are recorded in every results file under `task_patches`.
- FLEURS is public and may be in the audio encoder's training data; section 2b checks this on NaijaVoices and BibleTTS
  (different sources, one language absent from FLEURS) and the results hold.
- Retrieval pools in experiments 2, 3, 6 are ~300; CIRAL corpora are subsampled to 50k with all judged passages kept.
- Experiments 10-12 (RAG, archive search, voice-over-documents) are tools; they need a corpus to produce numbers.
