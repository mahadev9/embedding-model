# Multilingual Embedding Models (50M / 100M / 200M): Implementation Plan

Educational project: build three from-scratch multilingual text-embedding models on one RTX 5090 (32 GB),
aiming to be the best of their size class. This document records the research behind each design choice,
the architecture, the training pipeline, and the techniques to use or ablate.

**Status (2026-11-07):** scaffold, model configs and model code are done and verified (see 5.6). Data,
tokenizer and training loops are not started.

**Confidence markers.** Claims tagged *(paper)* were read in the paper's own text. Claims tagged *(summary)*
came from partial or machine-extracted summaries of a paper. *(blog)* is a secondary source.
*(mine)* is my own suggestion or arithmetic, not from a source. Verify *(summary)* and *(mine)* items before
depending on them.

---

## 1. Decisions already made

| Topic | Decision |
|---|---|
| Sizes | 50M, 100M, 200M **non-embedding** parameters. 200M is the main target, 100M is optional and last |
| Start with | **50M**: a cheap end-to-end pipeline test and the platform for ablations |
| Languages | 12 languages (confirmed by user): en, zh, es, fr, de, pt, ru, ja, ar, hi, ko, it |
| Tokenizer | Custom, shared by all sizes, 64k–100k vocabulary, vocabulary size a multiple of 64 |
| Output embedding | 768 dimensions, Matryoshka (MRL) so it can be truncated to 512/256/128 |
| Context | Train short, extend to 8192. 32k only via inference-time RoPE scaling if wanted |
| Architecture family | Dense bidirectional encoder, ModernBERT-style, trained from scratch |
| Reranker | Phase 2, after the embedder works |
| Teacher distillation | An ablation at 50M first (with vs without) |

---

## 2. What the research says

### 2.1 Reference recipes

| Model | Size | What it contributes | Source |
|---|---|---|---|
| ModernBERT | 149M / 395M | The architecture template and an encoder pretraining recipe | paper |
| mmBERT | 140M (42M non-emb) / 307M (110M non-emb) | The multilingual version of that template: language and mask-rate schedules | paper (partly summary) |
| NeoBERT | 250M | Ablation evidence on what moves quality in an encoder recipe | paper (partly summary) |
| mGTE | 305M | A complete multilingual encoder-to-embedder-to-reranker pipeline at 8k context | paper (partly summary) |
| EmbeddingGemma | 300M | Distillation, spread-out regularizer, MRL, souping | paper/blog |
| Qwen3-Embedding | 0.6B–8B | Synthetic data, merging, and the reranker recipe | paper/blog |
| jina-embeddings-v5 | small | Distillation beat plain contrastive training, and the combination beat both | paper (blog summary) |
| "Retrofitting Small Multilingual Models" (Salesforce) | 305M | What data matters for multilingual retrieval | paper |

### 2.2 Findings that shape this plan

1. **Architecture alone does not decide embedding quality.** In NeoBERT's table (MTEB-en after the same
   contrastive fine-tuning), ModernBERT-base scored 45.0 and ModernBERT-large 46.9, while BERT-base scored
   48.1 and NeoBERT 51.3. Its biggest single ablation gain was better pretraining data (+3.6% on GLUE).
   Data and recipe matter at least as much as the block design. *(paper)*
2. **Deep and narrow wins at small scale.** ModernBERT aims to be "as Deep & Narrow as possible without a
   significant inference slowdown." NeoBERT moved from 16×1056 to 28×768. mmBERT-small is 22 layers at 384
   hidden. MobileLLM reports the same for sub-billion decoders. *(paper)*
3. **Sequence packing needs masking across documents.** NeoBERT lost about 2.8% when packing without
   cross-attention masking. *(paper)*
4. **Mask rate:** ModernBERT uses 30%, mmBERT starts at 30%, then 15%, then 5% across phases, and NeoBERT uses
   20%. mmBERT says a lower rate in the final decay phase helped ("the smaller the masking rate the better"),
   but there is no pretraining-scale ablation. *(paper)*
5. **Language mix:** mmBERT adds languages late (annealed language learning), with sampling temperature
   0.7 → 0.5 → 0.3. mGTE uses α = 0.5. Neither ran an ablation on the exact value. *(paper)*
6. **Multilingual retrieval fine-tuning data:** in the Salesforce study, task diversity mattered more than
   language diversity, and LLM-written questions over Wikipedia documents helped. Parallel (translation) data
   gave little or no gain and hurt one backbone. Mined hard negatives consistently beat random in-batch
   negatives. Gains saturated after about 4k synthetic examples per language. *(paper)*
7. **mGTE's multilingual embedder pipeline:** MLM pretraining, then contrastive pretraining (about 2.9B pairs,
   batch 16,384, τ = 0.01), then fine-tuning with hard negatives. CLS pooling, MRL during fine-tuning. The
   reranker is a cross-encoder trained with InfoNCE from the 8k encoder. *(paper/summary)*
8. **Distillation:** jina-v5 found distillation from a larger embedding model beat contrastive-only training
   and the two combined did best. This was with pretrained LLM backbones, not from-scratch encoders, so the
   gap for us is unknown. *(blog/paper)*
9. **Scale caveat:** these recipes use 1T–3T tokens of pretraining. We can afford about 30–40B. Expect the
   learning-rate, mask-rate and batch-size values to need retuning, and treat every number below as a
   starting point to sweep, not a rule.

### 2.3 Gaps that remain after research

- No published ablation of temperature, batch size or false-negative filtering for sub-100M models.
- Muon vs AdamW: the evidence is for decoder LMs only (30M–200M decoders in one study, small gains of about
  3% in loss in another). Nothing for encoder pretraining.
- No controlled study of 64k vs 128k vocabulary for small encoders. Fertility falls as vocabulary grows
  (one study: 1.77 at 32k, 1.52 at 128k) but cost per token rises.
- Query/document prefixes and mean vs CLS pooling were not settled by the sources. mGTE and BGE-M3 use CLS.

---

## 3. Architecture

### 3.1 Block design (ModernBERT-style)

All items below are from the ModernBERT paper unless marked.

| Component | Choice | Notes |
|---|---|---|
| Attention layout | Global attention every 3rd layer, sliding-window (128 tokens) in the others | Matched global-only quality in ablations at 100B tokens, with large speedups |
| Positional encoding | RoPE. Theta 10,000 for local layers. Global layers 10,000 during pretraining, raised to **160,000** at context extension | Both ModernBERT and mmBERT do this |
| Normalization | Pre-norm. Extra norm after the embedding layer, no norm on the first layer's attention input | ModernBERT used LayerNorm. LayerNorm vs RMSNorm made little difference in its ablations. We use RMSNorm (PyTorch has it natively now) *(mine)* |
| Activation | GeGLU | GeGLU and SwiGLU were about equal in ablations *(summary)* |
| Biases | None in linear layers or norms. Keep a bias on the output (MLM) decoder, because embeddings are tied | |
| Embeddings | Tied input/output | |
| Dimensions | Multiples of 64, head dim 64, vocabulary a multiple of 64 | Tensor-core efficiency |
| Padding | Padding-free: pack documents and mask across them | See 5.2 for the kernel choice |
| Pooling | Test **CLS vs mean** at 50M | mGTE and BGE-M3 use CLS |
| Output head | Linear projection to 768 where hidden size differs, then L2 normalize | MRL applied to this 768-dim output |

### 3.2 Starting configurations

Parameter counts are non-embedding, computed from my formula
`layers × (4·hidden² + 3·hidden·intermediate)` *(mine)*. Vocabulary 64k–100k is extra.

| Model | Hidden | Layers | Heads | GeGLU intermediate | Non-emb params | Embedding table (64k vocab) |
|---|---|---|---|---|---|---|
| 50M | 384 | 22 | 6 | 1536 | ~52M | ~25M |
| 100M | 576 | 22 | 9 | 1728 | ~95M | ~37M |
| 200M | 768 | 26 | 12 | 2304 | ~200M | ~49M |

For comparison, mmBERT-small is 384 hidden, 22 layers, 1152 intermediate (42M non-embedding), and
ModernBERT-base is 768 hidden, 22 layers, 1152 intermediate (110M non-embedding). Ours use a larger
intermediate size to reach the target counts.

### 3.3 Architecture ablations to run at 50M

Run each for the same token budget (about 5–8B tokens, hours each) and compare by MLM loss plus a quick
embedding probe.

1. Depth vs width at fixed 52M: 22×384 vs 16×448 vs 28×336.
2. Local/global ratio: every 3rd vs every 4th layer global. Window 128 vs 256.
3. Mask rate: 20% vs 30%, and the 30 → 15 → 5 decay schedule vs constant.
4. Vocabulary size: 48k vs 64k vs 96k (fertility on the 12 languages vs parameter cost).
5. Pooling: CLS vs mean (after a short contrastive run).
6. Norm type: RMSNorm vs LayerNorm (likely a tie, so keep whichever is faster under `torch.compile`).

---

## 4. Training pipeline

```
Stage 0  Data + tokenizer
Stage 1  MLM pretraining (warmup-stable) ──► Stage 2 Context extension to 8192 (RoPE 160k)
         ──► Stage 3 Decay phase (low mask rate, language mix annealed) ──► merge checkpoints
Stage 4  Contrastive pretraining (large batch, weak pairs, in-batch negatives)
Stage 5  Supervised fine-tuning (hard negatives, MRL, instruction prefixes)
Stage 6  Distillation from a teacher (ablation)           ──► merge / soup
Stage 7  Long-context embedding fine-tune (short stage)
Phase 2  Cross-encoder reranker
```

### Stage 0: Data and tokenizer

- **Text for MLM:** FineWeb-2 for the non-English languages (ODC-By 1.0), and an English quality source
  (FineWeb-Edu or DCLM) plus Wikipedia. mmBERT's English share ranged from 10% to 34% by stage.
- **Language sampling:** `q_i ∝ p_i^α`. Start at α = 0.5 (mGTE) and ablate lowering it to 0.3 in the decay
  phase (mmBERT-like).
- **Tokenizer:** train on a balanced sample of the 12 languages, BPE or Unigram, 64k–96k vocabulary,
  character coverage high enough for CJK. Measure fertility per language before committing. Special
  tokens: CLS, SEP, MASK, PAD, plus a few unused slots.
- **Pair data:** see section 6.

### Stage 1: MLM pretraining

- Objective: MLM only, no next-sentence prediction. Mask rate 30% (ablate against 20%).
- Optimizer: StableAdamW (AdamW with Adafactor-style update clipping), betas (0.90, 0.98), epsilon 1e-6
  *(summary)*. Weight decay small (ModernBERT 1e-5, mmBERT 8e-5). Keep as the default over Muon, which has
  no encoder evidence (test it as an ablation only if time allows).
- Schedule: warmup-stable-decay. ModernBERT peak LR 8e-4 for base, held constant. mmBERT peak 8e-4 with
  3–4B-token warmup. For our scale, **sweep the peak LR** (2e-4 to 1e-3) at 50M with short runs.
- Batch-size warmup (ModernBERT: 768 → 4,608 sequences over 50B tokens) accelerated training in its
  ablations. Scale this down proportionally.
- Initial context 1024 tokens, with documents packed and masked at document boundaries.
- Initialization: standard Megatron-style scaled init for the first run.

### Stage 2: Context extension

- Raise the **global-layer RoPE theta from 10k to 160k** and continue MLM at 8192 tokens.
- ModernBERT's best result used both parts: the theta change and upsampling higher-quality, longer
  documents. Do both. Keep this stage short (a few billion tokens at our scale), since 8192-token steps cost
  about 2× per token in my benchmark.

### Stage 3: Decay and merge

- Decay LR to near zero with a 1−√ shape. Drop the mask rate (mmBERT: 15% in mid-training, 5% in decay).
- Anneal the language mix at this stage (mmBERT's late-language idea). With only 12 fixed languages,
  the simpler version is to lower α toward 0.3 for a more uniform language mix.
- Train two or three variants of the decay mixture (for example, English-heavy, balanced, and
  multilingual-heavy) and merge them (mmBERT: TIES for base, exponentially weighted checkpoint average for
  small). The model-merging step is cheap, so try it.

### Stage 4: Contrastive pretraining

- Loss: InfoNCE with in-batch negatives over (query, positive) pairs, symmetric if the data allows.
- Temperature: sweep 0.01–0.05. Evidence: mGTE 0.01 for pretraining, Salesforce 0.02, other guides 0.05.
- Batch size: as large as possible. mGTE used 16,384. Use **GradCache** (exact large-batch gradients, roughly
  20% overhead *(blog/paper)*). Sweep 1k, 4k, 16k at 50M to see where the gains stop.
- Each batch drawn from a single data source (mGTE) to make in-batch negatives harder and cleaner.
- Max lengths: queries 128–256, documents 512 for this stage.
- Data: cleaned weak pairs (title–body, adjacent passages, heading–section, QA, duplicates). Quality over
  volume. Start with 50–100M pairs and measure before collecting more.

### Stage 5: Supervised fine-tuning

- Positives plus **mined hard negatives** (mGTE: 1 positive + 8 hard negatives; Salesforce: 7). Mine with the
  stage-4 model, using a rank window (for example, ranks 10–100) and filtering likely false negatives
  (drop negatives scoring above 95% of the positive's score, or use a teacher reranker to filter).
- Add the MRL loss over dimensions 768/512/256/128 (EmbeddingGemma's setup).
- Temperature 0.02–0.05.
- Task and language diversity: mix retrieval, STS, classification-style pairs, clustering-style pairs, and
  multilingual synthetic retrieval. Salesforce found task diversity mattered most.
- Instruction prefixes: short task prefixes on the query side. Ablate on/off.
- Synthetic data: use a local LLM to write queries for Wikipedia and FineWeb-2 passages in the 12 languages.
  About 4k per language gave most of the gain in the Salesforce study, so start there.
- Avoid MTEB **training** splits unless you plan to report scores as "trained on MTEB". The Salesforce and
  NeoBERT recipes both use them. Keep your reported evaluation clean.

### Stage 6: Distillation (ablation)

- Teacher: an open embedding model run locally (Qwen3-Embedding 0.6B/4B). Precompute and cache teacher
  scores or embeddings over the training set so the 5090 isn't running two models at once.
- Dimension-agnostic option *(mine)*: distill the **similarity distribution**, meaning a KL divergence
  between teacher and student scores over each query's positive, hard negatives and in-batch candidates.
  No projection layer is needed.
- EmbeddingGemma's version matches embeddings directly. Add its **spread-out regularizer** (a global-orthogonal
  second-moment term on in-batch embeddings) at equal weight.
- Run at 50M: contrastive only vs distill only vs both. This is the educational result of the project.

### Stage 7: Long-context fine-tune

- Short stage on 2k–8k token pairs (long documents with generated questions) so retrieval holds up at 8192.
- Length-grouped batches (mGTE used batch sizes from 768 for short sequences down to 80 for 3000–8000).
- Evaluate on a long-document retrieval benchmark (MLDR) before and after.

### Merging

Keep the checkpoints from stages 5–6 and merge. Qwen3 reports both its weak-supervision stage and its merging
step mattered, and EmbeddingGemma averages "souped" runs trained on different data mixtures. Try linear
averaging, SLERP and TIES.

### Phase 2: Reranker

- Cross-encoder `[CLS] query [SEP] document`, linear head on CLS (mGTE).
- Initialize from the same MLM backbone. Train with InfoNCE over 1 positive and about 10 negatives (mGTE: 6
  hard and 4 random), with hard negatives mined by your trained embedder.
- Optional: distill from an open reranker (Qwen3-Reranker). Qwen3's own reranker is a pointwise yes/no
  model on an LLM, a different design.
- mGTE found contrastive pretraining did not help its reranker. Skip it.

---

## 5. Implementation notes

### 5.1 Stack

- Python 3.13, PyTorch 2.14.1+cu130 (installed and benchmarked on this GPU), bf16 autocast, `torch.compile`.
- A custom training loop for MLM and contrastive training. Sentence-Transformers v5 as an optional
  baseline for the fine-tuning stages.

### 5.2 Attention

Measured on the RTX 5090 (see [src/benchmarks/throughput.py](../src/benchmarks/throughput.py)):

| Kernel | 512 tokens | 8192 tokens |
|---|---|---|
| PyTorch SDPA flash backend | 118 TFLOP/s | 195 TFLOP/s |
| cuDNN | 106 | 188 |
| Memory-efficient | 72 | 84 |

- The built-in SDPA flash backend works on sm_120 with no flash-attn build, and is the fastest dense kernel here.
- FlexAttention with a sliding window compiled and ran (about 1.5 ms at 8192 tokens, 32k tokens per batch).
  Use it for the local layers.
- **Verified:** FlexAttention with a document mask handles packed sequences on sm_120. Building both block
  masks (global and local) costs about 2 ms per batch. The model uses FlexAttention for all layers, so the
  SDPA flash numbers above are a reference, not what the model runs. See 5.6.

### 5.3 Measured throughput (training, no MLM head, bf16, `torch.compile`)

| Model | Non-emb params | 512 tokens | Tokens/day |
|---|---|---|---|
| 50M | 52M | 331k tok/s | ~28.6B |
| 200M | 199M | 98k tok/s | ~8.5B |

At 8192 tokens, throughput is about half. A 200M run at 64k tokens per micro-batch runs out of memory
without checkpointing, and 16k tokens uses about 15 GB.

### 5.4 Rough compute plan *(mine, from the real-model benchmark in 5.6, ignoring data-loading overhead)*

| Stage | 50M | 200M |
|---|---|---|
| MLM pretraining, 1024 context | ~30B tokens, about 1.4 days | ~25B tokens, about 3.5–4.5 days |
| Context extension + decay | ~5B tokens, a few hours | ~5B tokens, about 1 day |
| Contrastive + fine-tuning | 1–2 days | 3–5 days |
| Distillation + merging | hours | 1–2 days |

Expect real numbers to run 20–40% slower. Budget in total roughly a week for 50M including ablations, and
two to three weeks for the 200M.

### 5.5 Repository layout (proposed)

```
src/
  benchmarks/      throughput.py, model_throughput.py (exist)
  data/            download, filter, language sampling, pair mining, synthetic generation
  tokenizer/       train + fertility report
  model/           config, rope, attention, encoder, heads (exist: see 5.6)
  training/        mlm.py, contrastive.py (GradCache), distill.py, optimizers, schedules, merge.py
  eval/            MMTEB subset, MIRACL, MLDR, throughput, contamination checks
  rerank/          cross-encoder + training
configs/           one YAML per stage and size (configs/model/ exists)
docs/              this file
```

### 5.6 Implementation status and measured results

**Built in `src/model/`:** `config.py` (validated `ModelConfig`, loaded from `configs/model/*.yaml`, analytic
parameter counts), `rope.py`, `attention.py` (FlexAttention with a document mask and a sliding window),
`encoder.py` (pre-norm blocks, GeGLU, no biases, extra norm after the embeddings, first block without an
attention norm), and `heads.py` (`MLMModel` with a tied decoder, `EmbeddingModel` with CLS or mean pooling,
projection, L2 normalization and Matryoshka truncation).

**Interface:** every forward pass takes `input_ids` and `doc_ids` (document index per token, `-1` for
padding). Packed pretraining rows and padded one-text-per-row embedding batches use the same path. Sequence
length should be a multiple of 128 (FlexAttention block size).

**Correctness checks** (ad hoc script, not committed as tests):

| Check | Result |
|---|---|
| Attention vs dense reference mask (global and local layers, fp32) | max difference ~1e-7 |
| Two documents packed in one row vs run separately (full model, fp32) | max difference ~1e-6 |
| RoPE positions restart at document boundaries | confirmed |
| MLM loss at initialization | 11.12 (ln 64000 = 11.07) |
| Embedding output and Matryoshka truncation (768 and 128 dims) | unit norm at both |
| Parameter count vs `ModelConfig` | exact match for 50M, 100M, 200M |

**MLM training throughput, real model** (bf16, `torch.compile`, packed documents, 30% masked,
`python -m src.benchmarks.model_throughput`):

| Model | Seq len | Packed docs | Tokens/s | Tokens/day | Peak memory |
|---|---|---|---|---|---|
| 50M | 1024 | ~400 tokens | 255k | ~22B | 18.3 GB at 32k tokens/batch |
| 50M | 8192 | ~400 tokens | 235k | ~20B | 18.3 GB at 32k tokens/batch |
| 200M | 1024 | ~400 tokens | 84k | ~7.3B | 18.7 GB at 16k tokens/batch |
| 200M | 8192 | ~400 tokens | 81k | ~7.0B | 18.7 GB at 16k tokens/batch |
| 200M | 8192 | one 8192-token doc | 64k | ~5.6B | 18.7 GB at 16k tokens/batch |

Takeaways:
- Packing short documents into 8192-token rows costs almost nothing extra per token, because the mask skips
  cross-document blocks. Pretraining at 8192 on packed short text is nearly as cheap as at 1024.
- Long single documents are the expensive case (the 200M drops to 64k tokens/s). The extension stage should
  budget for that, since long documents are what it trains on.
- The real MLM numbers are lower than the toy benchmark in 5.3 (no head, no masking): the 200M is about
  5.6–7B tokens/day rather than 8.5B.

---

## 6. Datasets

| Purpose | Dataset | License / note |
|---|---|---|
| MLM, multilingual | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-By 1.0. Check per-language sizes for the 12 languages |
| MLM, English quality | FineWeb-Edu or DCLM, Wikipedia | Check licenses before use |
| Weak pairs, English | [nomic-embed-unsupervised-data](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data), [sentence-transformers/embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) | Mostly English |
| Weak pairs, multilingual | Mined by us from FineWeb-2 and Wikipedia (title–body, adjacent passages) | multilingual-e5 used Wikipedia, mC4, CC News and NLLB for this |
| Parallel pairs | OpenSubtitles pairs, [allenai/nllb](https://huggingface.co/datasets/allenai/nllb) | Low value per the Salesforce study. Optional, with filtering. NLLB license unchecked |
| Supervised retrieval | MS MARCO, NQ, HotpotQA, FEVER, AllNLI | Standard. Verify licenses |
| Supervised, multilingual | MIRACL, Mr. TyDi, mMARCO | Small. MIRACL is also an evaluation set, so keep it out of training if you evaluate on it. License reports conflict |
| Synthetic | LLM-generated queries over Wikipedia/FineWeb-2 passages in the 12 languages | About 4k per language gets most of the gain |

Contamination: hold out evaluation sets before building any training mix, and deduplicate against them.

---

## 7. Evaluation plan

1. **During MLM:** validation MLM loss per language, plus a tiny probe (frozen-embedding STS or retrieval on
   a small multilingual set) every N tokens.
2. **Embeddings:** an MMTEB subset (retrieval, STS, classification, clustering, bitext mining) in the 12
   languages, MIRACL for retrieval, MLDR for long documents. Compare against same-size baselines
   (multilingual-e5 small/base, mmBERT-small fine-tuned) rather than the large models.
3. **Efficiency:** tokens per second and memory at 512 and 8192 tokens.
4. **MRL:** quality at 768/512/256/128 dimensions.
5. **Reference points** (third-party reports): mmBERT-small, fine-tuned on MS MARCO only, scored 50.7 on
   multilingual MTEB v2 and mmBERT-base 54.1. The 305M Salesforce retrofit scored 60.56 on MMTEB retrieval.
   Those are the kinds of numbers to compare against.

---

## 8. Milestones

1. **M0 Environment and data audit:** confirm the 12 languages, per-language FineWeb-2 sizes, and licenses.
2. **M1 Tokenizer:** train candidates (48k/64k/96k), pick on fertility per language.
3. **M2 Model and MLM loop at 50M:** model code, correctness checks and the packing throughput check are
   done (5.6). The MLM training loop, data loader and a short sanity run remain.
4. **M3 50M ablations** (section 3.3), then a full 50M pretraining run.
5. **M4 Contrastive and fine-tuning pipeline at 50M**, with GradCache, hard-negative mining, MRL, evaluation.
6. **M5 Distillation and merging ablations at 50M.**
7. **M6 Scale to 200M** with the best recipe. 100M if time allows.
8. **M7 Reranker.**

---

## 9. Risks and open questions

- **Scale gap:** the reference recipes use 30–100× more pretraining tokens than we can afford. Retune before
  trusting them.
- **From-scratch multilingual quality:** with no pretrained backbone, expect to trail Jina-v5, EmbeddingGemma
  and Harrier, which start from large pretrained models.
- **Pair-data quality** is likely to be the dominant factor, given finding 1 and the NeoBERT ablations.
- **Packing on sm_120:** resolved. FlexAttention document masking works and is fast (5.6).
- **Licenses** for several datasets are unconfirmed (MIRACL, Mr. TyDi, mMARCO, NLLB, FineWeb-Edu/DCLM).
- **Unverified details:** several numbers from mmBERT, mGTE and NeoBERT came from machine summaries of
  truncated pages. Where a number drives a decision, read the paper section directly.
- **Open decisions:** tokenizer size and whether to try Muon.

---

## 10. Sources

- [ModernBERT](https://arxiv.org/html/2412.13663)
- [mmBERT](https://arxiv.org/html/2509.06888v1)
- [NeoBERT](https://arxiv.org/html/2502.19587)
- [mGTE](https://arxiv.org/html/2407.19669v2)
- [Retrofitting Small Multilingual Models for Retrieval](https://arxiv.org/html/2510.14274v2)
- [EmbeddingGemma](https://arxiv.org/abs/2509.20354) and [its architecture/recipe blog](https://developers.googleblog.com/en/gemma-explained-embeddinggemma-architecture-and-recipe/)
- [Qwen3 Embedding](https://arxiv.org/html/2506.05176v1)
- [jina-embeddings-v5-text](https://arxiv.org/html/2602.15547v1)
- [Multilingual E5](https://arxiv.org/html/2402.05672v1)
- [MobileLLM](https://arxiv.org/pdf/2402.14905)
- [GradCache](https://arxiv.org/pdf/2101.06983)
- [FineWeb-2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/cab26bb3cab1ca7c26bb13a48f80fc3d2def4620/README.md)
