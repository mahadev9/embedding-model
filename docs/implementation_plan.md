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
| Tokenizer | Custom, shared by all sizes. Candidate sizes 48k / 64k / 96k / 128k, vocabulary size a multiple of 64 (see section 10) |
| Output embedding | 768 dimensions for 100M and 200M, 384 for 50M; Matryoshka (MRL) so it can be truncated (768/512/256/128, or 384/256/128/64 for the 50M) |
| Context | Train short, extend to 8192. 32k only via inference-time RoPE scaling if wanted |
| Architecture family | Dense bidirectional encoder, ModernBERT-style, trained from scratch |
| Reranker | Phase 2, after the embedder works |
| Teacher distillation | **Main recipe** (see section 11): distill from cached teacher embeddings, then contrastive fine-tune. Contrastive-only is kept as the baseline run |
| Parallel (translation) pairs | Included, filtered, weighted toward pairs among the 12 languages (see section 11) |
| 50M output dimension | **384** (equals its hidden size, Matryoshka 384/256/128/64). Decided 2026-11-07, see 11.3 |

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
| Attention layout | Global attention every 3rd layer, sliding-window (128 tokens) in the others. **The last layer is always global** so the pooled CLS token sees the whole text (the 200M, 26 layers, would otherwise end on a local layer) | Matched global-only quality in ablations at 100B tokens, with large speedups |
| Positional encoding | RoPE. Theta 10,000 for local layers. Global layers 10,000 during pretraining, raised to **160,000** at context extension | Both ModernBERT and mmBERT do this |
| Normalization | Pre-norm. Extra norm after the embedding layer, no norm on the first layer's attention input | ModernBERT used LayerNorm. LayerNorm vs RMSNorm made little difference in its ablations. We use RMSNorm (PyTorch has it natively now) *(mine)* |
| Activation | GeGLU | GeGLU and SwiGLU were about equal in ablations *(summary)* |
| Biases | None in linear layers or norms. Keep a bias on the output (MLM) decoder, because embeddings are tied | |
| Embeddings | Tied input/output | |
| Dimensions | Multiples of 64, head dim 64, vocabulary a multiple of 64 | Tensor-core efficiency |
| Padding | Padding-free: pack documents and mask across them | See 5.2 for the kernel choice |
| Pooling | Test **CLS vs mean** at 50M | mGTE and BGE-M3 use CLS |
| Output head | Linear projection to `output_dim` where it differs from the hidden size (100M only: 576 to 768), then L2 normalize. The 50M and 200M need no projection | MRL applied to the output vector |

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
7. Pretrain at 8192 with global RoPE theta 160k from the start vs 1024 then extend. Packed short
   documents cost only 4–8% more per token at 8192 (5.6), but real documents are longer than the benchmark's
   ~400 tokens, so measure on real data.
8. Token budget slope: the same 50M model at 30B vs about 100B tokens, to see how much a longer run buys
   before committing a week of the 5090 to either size.

---

## 4. Training pipeline

```
Stage 0  Data + tokenizer
Stage 1  MLM pretraining ──► Stage 2 Context extension (RoPE 160k) ──► Stage 3 Decay ──► merge
                              (one shared checkpoint: everything below starts from it)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
   Arm A: Stage 4                      Arm B: Stage 6 ──► Stage 5           Arm C: Stage 4 ──► Stage 5
   distill only                        contrastive only                     distill, then contrastive
         └────────────────────────────────────┼────────────────────────────────────┘
                                              ▼
                       best arm ──► merge / soup ──► Stage 7 long-context ──► Phase 2 reranker
```

Stage 6 is a separate branch from the Stage 3 checkpoint, not a step after Stage 5.

### 4.1 The central experiment: three arms with matched compute

Question: does distillation help a from-scratch student, and does it replace or complement contrastive
training? Run at 50M first. All three arms start from the same Stage 3 checkpoint.

| Arm | Recipe | Compute after Stage 3 |
|---|---|---|
| A, distill only | Stage 4 | X + Y |
| B, contrastive only | Stage 6, then Stage 5 | X for Stage 6, then Y for Stage 5 |
| C, both | Stage 4, then Stage 5 | X for Stage 4, then Y for Stage 5 |

Rules so the comparison is fair:
- **Same total budget:** every arm gets X + Y GPU-hours (A gets all of it in Stage 4, since it has no Stage 5).
  Also record examples seen, since the stages consume different data.
- **Same starting checkpoint**, tokenizer, evaluation subset, and at least two seeds if time allows.
- **Identical hard negatives for B and C.** Mine them once, before the experiment, with a common miner (for
  example the teacher plus a reranker filter), not with each arm's own earlier model. Otherwise arms differ in
  their negatives as well as their recipe.
- **Teacher caching is a shared one-time cost.** Report it separately instead of charging it to X or Y.
- Report results by task type and by Matryoshka size, not only the overall mean.
- Stage 7 and the reranker run only for the winning arm.

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

### Stage 4: Distillation pretraining (main recipe)

Why: the strongest small models are distilled, not trained contrastively from scratch. jina-embeddings-v5-text
(EuroBERT-210M student, Qwen3-Embedding-4B teacher) found distillation beat plain contrastive training and the
two combined did best. LEAF (23M English student, teacher mxbai-embed-large-v1) trains only to match the
teacher's embeddings with an L2 loss, needs no hard negatives, and works with small batches. Both start from
pretrained backbones, so whether this holds for a from-scratch student is untested here. Hence the 50M
comparison (distill vs contrastive-only vs both) is the project's central experiment.

- **Teacher:** Qwen3-Embedding-0.6B first (Apache 2.0, multilingual, reported to support Matryoshka-truncated
  output; language coverage of our 12 and truncation quality are unverified). Try 4B if the 0.6B teacher is
  the bottleneck. **Measure its encoding speed on the 5090 before sizing the corpus.**
- **Corpus to embed:** FineWeb2-HQ-Plus passages of mixed lengths, mined pairs, synthetic queries, and
  filtered parallel sentences. Start with 20–50M texts. Cached embeddings cost about 1.5 KB per text
  (768 dims, fp16), so 20–50M texts is 30–75 GB, which fits on the HDD.
- **Loss:** L2 or cosine between student and cached teacher embeddings (LEAF). Add a KL loss on the
  in-batch similarity matrices *(mine)*, and optionally EmbeddingGemma's spread-out regularizer. If the
  student's output size differs from the teacher's, do not add a training-only head (see Targets below).
- **Targets (no projection head):** cache the teacher's embeddings once at 768 dims, fp16 (the 0.6B teacher's
  native 1024, truncated and re-normalized). The 100M and 200M match the 768-dim vector directly. The 50M
  matches the first 384 dims, re-normalized. The student's nested Matryoshka sizes match the same-size
  prefixes of the teacher vector, re-normalized. So the vector you ship is exactly the vector you trained,
  and the 384-dim targets cost nothing extra to store.
- **Gating check before caching (M4):** the model card lists "MRL Support: Yes" with output sizes from 32 to
  1024, but gives no quality numbers at smaller sizes, and the paper does not describe Matryoshka training.
  So measure the teacher itself at 1024, 768, 384, 256, 128 and 64 dims on our 12-language subset. If 384 is
  clearly weak, fall back to distilling from the full 1024-dim vector through a training-only head.
- **Queries vs documents:** embed queries with the teacher's query instruction and documents without, and
  give the student matching prefixes.
- **Batch size** can be small, so GradCache is optional here, unlike stage 6.
- The teacher's quality caps the student's. Keep checking the student against the teacher on held-out text.

### Stage 5: Contrastive fine-tuning (after distillation)

- Positives plus **mined hard negatives** (mGTE: 1 positive + 8 hard negatives; Salesforce: 7). Mine once with a
  common miner fixed before the arms experiment (section 4.1), using a rank window (for example, ranks 10–100) and filtering likely false negatives
  (drop negatives scoring above 95% of the positive's score, or use a teacher reranker to filter).
- Add the MRL loss over the model's Matryoshka sizes (768/512/256/128, or 384/256/128/64 for the 50M;
  EmbeddingGemma's setup).
- Temperature 0.02–0.05.
- Task and language diversity: mix retrieval, STS, classification-style pairs, clustering-style pairs, and
  multilingual synthetic retrieval. Salesforce found task diversity mattered most.
- Instruction prefixes: short task prefixes on the query side. Ablate on/off.
- Synthetic data: use a local LLM to write queries for Wikipedia and FineWeb-2 passages in the 12 languages.
  About 4k per language gave most of the gain in the Salesforce study, so start there.
- Avoid MTEB **training** splits unless you plan to report scores as "trained on MTEB". The Salesforce and
  NeoBERT recipes both use them. Keep your reported evaluation clean.

### Stage 6: Contrastive-only baseline (and optional mix-in)

- Loss: InfoNCE with in-batch negatives over (query, positive) pairs, symmetric if the data allows.
- Temperature: sweep 0.01–0.05. Evidence: mGTE 0.01 for pretraining, Salesforce 0.02, other guides 0.05.
- Batch size: as large as possible. mGTE used 16,384. Use **GradCache** (exact large-batch gradients, roughly
  20% overhead *(blog/paper)*). Sweep 1k, 4k, 16k at 50M to see where the gains stop.
- Each batch drawn from a single data source (mGTE) to make in-batch negatives harder and cleaner.
- Max lengths: queries 128–256, documents 512 for this stage.
- Data: cleaned weak pairs (title–body, adjacent passages, heading–section, QA, duplicates). Quality over
  volume. Start with 50–100M pairs and measure before collecting more.

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
| Option: longer MLM run | ~100B tokens, about 4.5 days | ~100B tokens, about 14 days |
| Context extension + decay | ~5B tokens, a few hours | ~5B tokens, about 1 day |
| **Teacher caching** (encode 20–50M texts with Qwen3-Embedding-0.6B, shared by arms A and C) | **TBD, measured in M4** | same cache, no extra cost |
| Experiment arms A, B, C (section 4.1), budget X + Y each | TBD after teacher speed is known | TBD, run only for the winning recipe |
| Merging, long-context stage | hours | about 1 day |

The earlier placeholders for contrastive and distillation (1–2 days, hours) are removed: they predate the
distillation-first recipe and had no basis once the teacher cost became the largest new item. Teacher
caching is a one-time cost, so its size (corpus of 20–50M texts at whatever the teacher's speed allows)
decides how large X and Y can be. Expect real numbers to run 20–40% slower than benchmarks. The longer MLM
run is only worth its days if the 30B vs 100B slope ablation (3.3, item 8) shows a gain.

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
padding). Packed pretraining rows and padded one-text-per-row embedding batches use the same path. Any
sequence length works (checked at 100 and 300 tokens against the padded-to-128 result); multiples of 128 just
avoid partial FlexAttention blocks.
- **Build block masks outside `torch.compile`** (`build_block_masks`) and pass them in. Building them inside the
  compiled region caused graph breaks.
- **`MLMModel`** takes a fixed number of masked positions per batch (`mlm_positions`, `mlm_labels`, with `-100`
  for unused slots; `select_masked` builds them from a label tensor). Boolean-mask indexing produced a
  different shape each step, which broke the compiled graph and recompiled.
- **`EmbeddingModel.encode_packed`** pools many texts per row without padding between them (CLS = first
  token of each text, or a per-text mean), so short queries no longer pay for padding to 128 tokens.
- RoPE cos/sin tables are computed once per forward per theta and shared by all layers.

**Correctness checks** (ad hoc script, not committed as tests):

| Check | Result |
|---|---|
| Attention vs dense reference mask (global and local layers, fp32) | max difference ~1e-7 |
| Two documents packed in one row vs run separately (full model, fp32) | max difference ~1e-6 |
| RoPE positions restart at document boundaries | confirmed |
| MLM loss at initialization | 11.15 (ln 64000 = 11.07) |
| Embedding output and Matryoshka truncation (768 and 128 dims) | unit norm at both |
| `encode_packed` vs padded `forward`, CLS and mean pooling | max difference ~3e-8 |
| Last layer is global on 50M, 100M and 200M | confirmed |
| Graph breaks in the compiled MLM step (varying masked counts) | none |
| Parameter count vs `ModelConfig` | exact match for 50M, 100M, 200M |

**MLM training throughput, real model** (bf16, `torch.compile`, packed documents, 30% masked,
`python -m src.benchmarks.model_throughput`):

| Model | Seq len | Packed docs | Tokens/s | Tokens/day | Peak memory |
|---|---|---|---|---|---|
| 50M | 1024 | ~400 tokens | 257k | ~22B | 14.8 GB at 32k tokens/batch |
| 50M | 8192 | ~400 tokens | 248k | ~21B | 14.8 GB at 32k tokens/batch |
| 200M | 1024 | ~400 tokens | 87k | ~7.5B | 16.9 GB at 16k tokens/batch |
| 200M | 8192 | ~400 tokens | 83k | ~7.2B | 16.9 GB at 16k tokens/batch |
| 200M | 8192 | one 8192-token doc | 64k | ~5.5B | 16.9 GB at 16k tokens/batch |

Each step builds a fresh batch (new documents and masks) and includes mask building, as in real training.
The first benchmark (previous revision) reused the same label count every step, which hid a recompile and
two graph-break problems; fixing them also cut memory by about 3 GB and raised throughput slightly.

Takeaways:
- Packing short documents into 8192-token rows costs only 4–8% extra per token, because the mask skips
  cross-document blocks. Pretraining at 8192 on packed short text is nearly as cheap as at 1024.
- Long single documents are the expensive case (the 200M drops to 64k tokens/s). The extension stage should
  budget for that, since long documents are what it trains on.
- The real MLM numbers are lower than the toy benchmark in 5.3 (no head, no masking): the 200M is about
  5.5–7.5B tokens/day rather than 8.5B.

---

## 6. Datasets

| Purpose | Dataset | License / note |
|---|---|---|
| MLM, multilingual | [FineWeb2-HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus) (preferred, see section 10) or [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Both odc-by. HQ-Plus is the top 10% of FineWeb-2 by a quality classifier |
| MLM, English quality | FineWeb-Edu or DCLM, Wikipedia | Check licenses before use |
| Weak pairs, English | [nomic-embed-unsupervised-data](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data), [sentence-transformers/embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) | Mostly English |
| Weak pairs, multilingual | Mined by us from FineWeb-2 and Wikipedia (title–body, adjacent passages) | multilingual-e5 used Wikipedia, mC4, CC News and NLLB for this |
| Parallel pairs | OpenSubtitles pairs, [allenai/nllb](https://huggingface.co/datasets/allenai/nllb) | Low value per the Salesforce study. Optional, with filtering. NLLB license unchecked |
| Supervised retrieval | MS MARCO, NQ, HotpotQA, FEVER, AllNLI | Standard. Verify licenses |
| Supervised, multilingual | MIRACL, Mr. TyDi, mMARCO | Small. MIRACL is also an evaluation set, so keep it out of training if you evaluate on it. License reports conflict |
| Synthetic | LLM-generated queries over Wikipedia/FineWeb-2 passages in the 12 languages | About 4k per language gets most of the gain |

Contamination: hold out evaluation sets before building any training mix, and deduplicate against them.

### 6.1 Audit results (2026-11-07, `python -m src.data.audit`)

Licenses, from the dataset cards: FineWeb-2 **odc-by**, FineWeb-Edu **odc-by**, Wikipedia
**cc-by-sa-3.0 / gfdl** (share-alike, so keep it out of anything you plan to relicense; it is also small
next to the web data), DCLM **cc-by-4.0** (not used so far). The MIRACL, Mr. TyDi, mMARCO and NLLB licenses
are still unchecked.

FineWeb-2 has no English, so English comes from FineWeb-Edu (a 10B-token sample is enough).

| Language | FineWeb-2 config | Train shards | Parquet size | Rows (documents) | Wikipedia (parquet) |
|---|---|---|---|---|---|
| en | (FineWeb-Edu `sample/10BT`) | n/a | n/a | n/a | 10.8 GiB |
| zh | cmn_Hani | 370 | 1510.6 GiB | 636M | 1.6 GiB |
| es | spa_Latn | 146 | 593.8 GiB | n/a | 3.3 GiB |
| fr | fra_Latn | 135 | 502.8 GiB | 360M | 4.3 GiB |
| de | deu_Latn | 181 | 719.1 GiB | 496M | 5.4 GiB |
| pt | por_Latn | 66 | 256.9 GiB | 200M | 1.5 GiB |
| ru | rus_Cyrl | 440 | 1851.1 GiB | 699M | 4.5 GiB |
| ja | jpn_Jpan | 175 | 667.4 GiB | 400M | 3.7 GiB |
| ar | arb_Arab | 25 | 98.7 GiB | 62M | 1.2 GiB |
| hi | hin_Deva | 8 | 31.9 GiB | 22M | 0.2 GiB |
| ko | kor_Hang | 25 | 98.5 GiB | 61M | 0.7 GiB |
| it | ita_Latn | 85 | 332.5 GiB | 239M | 2.7 GiB |

Conclusions:
- **Every language has far more text than we can train on.** Even Hindi, the smallest, is about 22M documents.
  The binding constraint is our 25–40B token budget and disk, not data availability. With α = 0.5 sampling,
  the small languages (hi, ar, ko) are upsampled relative to their share, but they are not exhausted.
- FineWeb-2 shards are about 4 GiB each, with uneven sizes (some trailing shards are small), so downloads
  should be by shard count, not by language.
- Rough raw-text need for 30B tokens is on the order of 120 GB (about 4 bytes per token on average, higher
  for CJK and Devanagari), so about 60–80 GB of zstd parquet after download. This fits the 200–300 GB budget,
  provided raw shards are deleted after tokenization.

### 6.2 Local data layout and download script

Small and derived data lives under `data/` (gitignored). Bulk raw downloads can be redirected to another disk
by setting `EMBEDDING_RAW_DIR`. On this machine the 1.9 TB HDD is available:

```bash
export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw
```

Hugging Face caches follow `RAW_DIR` (your stored login token is still used). To remove everything:
`rm -rf data/ $EMBEDDING_RAW_DIR`. Training reads tokenized data at only about 160 KB/s for the 200M model
(80k tokens/s x 2 bytes), so even the HDD is fast enough for the tokenized corpus.

```
data/                              (repo, local SSD)
  audit/sources.json               output of src.data.audit
  sample/<lang>.parquet            ~300 MB of text per language, for tokenizer training
$EMBEDDING_RAW_DIR/                (default data/raw; HDD recommended)
  <source>/<lang>/...              whole shards for the main corpus (src.data.download shards)
  .hf_cache/                       Hugging Face caches
```

`src/data/download.py`:
- `sample` streams only the first pages of randomly ordered shards (no multi-GiB download), until the text
  budget per language is met. Rerunning skips files that exist.
- `shards` downloads whole parquet shards for a source and languages (`--source fineweb2 | fineweb-edu |
  wikipedia`, `--langs`, `--num-shards`, `--random`).

Tokenizer sample downloaded: 12 languages x ~300 MB of text (about 1.4 GB on disk). Languages are balanced by
text bytes, so no language dominates the vocabulary.

---

## 7. Evaluation plan

1. **During MLM:** validation MLM loss per language, plus a tiny probe (frozen-embedding STS or retrieval on
   a small multilingual set) every N tokens.
2. **Embeddings:** an MMTEB subset (retrieval, STS, classification, clustering, bitext mining) in the 12
   languages, MIRACL for retrieval, MLDR for long documents. Compare against same-size baselines, all run
   on the same 12-language subset: multilingual-e5 small/base, mmBERT-small fine-tuned, EmbeddingGemma-300m,
   **granite-embedding-97m-multilingual-r2** (Apache 2.0), **jina-embeddings-v5-text-nano** (CC BY-NC, so for
   comparison only), and for the English subset **LEAF**. Report scores by task type (retrieval, STS,
   classification, clustering, bitext mining), not only the overall mean.
3. **Efficiency:** tokens per second and memory at 512 and 8192 tokens.
4. **MRL:** quality at each Matryoshka size.
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
5. **M4 Teacher benchmark and distillation pipeline at 50M**: measure teacher encoding speed and its quality
   at truncated dimensions (gating check), cache teacher embeddings, distillation loop, evaluation harness with all baselines.
6. **M5 Contrastive pipeline and ablations at 50M**: hard-negative mining, MRL, GradCache baseline run,
   distill vs contrastive vs both, merging.
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
- **Licenses** for several datasets are unconfirmed (MIRACL, Mr. TyDi, mMARCO, NLLB, FineWeb-Edu/DCLM). The Qwen3-Embedding teacher license (Apache 2.0) was
  reported in search results, not read from the model card.
- **Unverified details:** several numbers from mmBERT, mGTE and NeoBERT came from machine summaries of
  truncated pages. Where a number drives a decision, read the paper section directly.
- **Teacher cost is unmeasured:** how fast Qwen3-Embedding-0.6B encodes on the 5090 sets the corpus size.
- **Realistic goal:** the leading small models start from trillion-token pretrained backbones. Aim for best
  from-scratch model in each class, or best on specific task types or languages, not best overall.
- **Teacher truncation quality is unverified:** the Qwen3-Embedding card lists Matryoshka support (32–1024
  dimensions) but gives no quality numbers, and the paper does not describe Matryoshka training. The student
  targets at 384 and smaller depend on this, so it is a gating check in M4 (see Stage 4).
- **Open decisions:** tokenizer size, Muon, and whether to test muP/CompleteP.

---

## 10. Research round 2 (2026-11-07): tokenizer, data quality, evaluation

### 10.1 Tokenizer design

What the evidence says, and what it doesn't:
- **BPE vs Unigram:** no clear winner. One 2025 study of cross-lingual tokenizer inequities found no
  significant difference in token premiums between them. A Telugu study found Unigram better downstream. A
  multilingual design study found Unigram only slightly worse than BPE at 32k but degrading as the vocabulary
  grows. *(paper/blog snippets)*
- **Vocabulary size:** a multilingual tokenizer design study found 128k the most balanced. Tao et al. (NeurIPS
  2024) show the optimal vocabulary grows with model size and that many models use too small a vocabulary,
  but their models are decoders (33M–3B) and not multilingual. *(paper)*
- **Parity-aware BPE** improves compression fairness across languages, but its variants trail classical BPE on
  CJK. *(paper)*
- **Byte fallback and digits:** GPT-SW3 used byte fallback, character coverage 0.9999, and single-digit
  tokens. Llama 2's tokenizer triggers many byte fallbacks on CJK. *(paper)*
- **Unicode normalization:** no evidence found. My default is NFKC (sentencepiece's default for XLM-R-style
  tokenizers): it folds full-width CJK and Latin characters, but check what it does to Arabic and Devanagari
  before committing. *(mine)*

Plan for the tokenizer step:
1. Candidates: BPE at 48k, 64k, 96k, 128k (Unigram as a secondary comparison), byte fallback on, digits split,
   character coverage ~0.9999, balanced by text bytes across the 12 languages.
2. Metrics: **bytes per token** and **characters per token** per language (words per token is ill-defined for
   zh and ja), plus tokens per word for the space-delimited languages.
3. Baselines measured on our sample: XLM-R (250k), Gemma (256k), Qwen3 (~151k), Llama 3 (128k), ModernBERT
   (50k). This shows what a good number looks like.
4. Embedding cost of the choice: at hidden 384, a 128k vocabulary is ~49M embedding parameters, about equal
   to the whole 50M body. Embedding lookups cost almost no compute and the MLM head only projects masked
   tokens, so the cost is memory and parameter count, not speed. Report both total and non-embedding sizes.

### 10.2 Data quality: use FineWeb2-HQ-Plus for the non-English languages

- [FineWeb2-HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus) is the top 10% of FineWeb-2
  (v2.1.1) by a multilingual quality classifier, covering 100 languages, license odc-by. It covers all 11
  non-English target languages. (The original FineWeb2-HQ has 20 languages and lacks Hindi and Korean.)
- The authors report the original HQ matches FineWeb-2 with about 6x fewer tokens for decoder LLM
  pretraining. For MLM encoders this is unproven. NeoBERT's results point the same way (data quality was its
  largest ablation gain), so it is worth testing. *(blog/dataset card)*
- Layout is `<lang>/NNN_NNNNN.parquet` (no `train/` folder). Sizes (parquet):

| Language | Files | Size |
|---|---|---|
| zh (cmn_Hani) | 680 | 174.6 GiB |
| ru (rus_Cyrl) | 840 | 304.3 GiB |
| de (deu_Latn) | 337 | 132.8 GiB |
| es (spa_Latn) | 278 | 119.9 GiB |
| fr (fra_Latn) | 243 | 98.4 GiB |
| ja (jpn_Jpan) | 315 | 96.0 GiB |
| it (ita_Latn) | 157 | 60.6 GiB |
| pt (por_Latn) | 122 | 50.5 GiB |
| ar (arb_Arab) | 45 | 18.4 GiB |
| ko (kor_Hang) | 45 | 17.2 GiB |
| hi (hin_Deva) | 15 | 6.9 GiB |

- Is it enough? Rough estimate *(mine)*: with α = 0.5 sampling and a 30B token budget, Hindi's share is about
  0.7B tokens, and 6.9 GiB of parquet is a few billion tokens, so no language needs repeating.
- `src/data/download.py` now accepts `--source fineweb2-hq-plus`. The tokenizer sample currently on disk came
  from plain FineWeb-2, so it should be regenerated from HQ-Plus before tokenizer training, so the
  tokenizer sees the same distribution as the pretraining data.
- Add a 50M ablation: FineWeb-2 vs HQ-Plus at the same token budget.

### 10.3 Evaluation coverage and the real bar to beat

- **MIRACL** has 18 languages (16 known plus 2 revealed later). Italian and Portuguese are not among the known
  ones. German's status is unconfirmed here. **Mr. TyDi** (ar, bn, en, fi, id, ja, ko, ru, sw, te, th) has
  none of es, fr, de, pt, it, zh, hi. So retrieval training and evaluation data for Italian and Portuguese must
  come from elsewhere (MTEB-BR exists for Brazilian Portuguese; Italian is unchecked).
- **MMTEB** has 500+ tasks across 250+ languages. I found no per-language task counts for our 12. The `mteb`
  package can filter tasks by language, which we need to do, and then run every baseline on the same subset.
- **Official numbers**, MTEB(Multilingual, v2) mean over tasks, from the EmbeddingGemma paper (third-party
  snapshot): EmbeddingGemma-300m 61.2, gte-multilingual-base 58.2 (305M), multilingual-e5-base 57.0 (278M),
  multilingual-e5-small 55.8 (118M). These are averages over all languages, so **they are not comparable to
  scores on our 12-language subset**. Run each baseline on our subset ourselves.
- **Size classes, by non-embedding parameters** *(mine, from vocabulary x hidden size)*: multilingual-e5-small
  is 118M total but about 96M of that is its 250k-token embedding table (about 21M non-embedding).
  multilingual-e5-base and gte-multilingual-base have roughly 86M and 113M non-embedding parameters, and
  EmbeddingGemma-300m about 107M. So our 100M model has a similar transformer body to the "300M-class"
  models, and our 200M has about twice the body, with a total of about 250M. State both numbers when
  comparing ("200M non-embedding, ~250M total").
- **Runtime:** MMTEB reports up to two days on one A100 for moderate LLMs on the original English benchmark,
  and uses downsampling to cut cost. A small model's runtime is unknown. Time one on a few tasks first.

### 10.4 Still open (deferred until needed)

Hyperparameter transfer from 50M to 200M, token budget scaling for small multilingual encoders, verified
StableAdamW values, contrastive pair-data licenses (MIRACL, mMARCO, NLLB and others), teacher and
synthetic-data LLM choice and license.

---

## 11. External plan review (2026-11-07): assessment and changes

A reviewer read this plan and the code. I checked each claim: code claims by testing them, model and recipe
claims against sources. Status: **confirmed** (verified and acted on), **accepted** (reasonable, adopted as a
test or plan change), **open** (needs your decision).

### 11.1 Code findings

| # | Finding | Status | What I did |
|---|---|---|---|
| 1 | 200M (26 layers) ends on a local layer, so the final CLS token sees only +-64 tokens | **confirmed** (layer 25 was local; 50M and 100M end global) | Last layer is now always global, for any depth. 200M has 10 global layers |
| 2 | `EmbeddingModel` pads short texts to 128-token blocks | **confirmed** | Added `encode_packed` (CLS and mean pooling), equal to the padded path within 3e-8 |
| 3 | Padded inputs not rounded to 128 | **partly**: lengths 100 and 300 work and match padded-to-128 within 1e-6, so it is not a correctness bug | No code change. Packing removes the efficiency concern. Docstring corrected |
| 4 | `hidden[masked]` has a data-dependent shape and recompiles | **confirmed**, and worse: block-mask construction inside the compiled region also graph-broke | Fixed-count masked positions (`select_masked`), masks built outside compile. My earlier benchmark reused one label count and hid this |
| 5 | 50M projects 384 to 768, so rank is at most 384 | **accepted** | 50M now outputs 384, see 11.3 |
| 6 | RoPE cos/sin recomputed per layer | **confirmed** (minor) | Computed once per forward per theta |
| 7 | Optimizer parameter groups | **accepted** | No weight decay on norms, biases or embeddings when `training/` is written |

### 11.2 Strategy

| # | Suggestion | Status | Notes |
|---|---|---|---|
| 1 | Newer baselines | **confirmed** | granite-embedding-97m-multilingual-r2: 97M, 384-dim, ModernBERT architecture, built by layer pruning, vocabulary selection (about 180k tokens), contrastive fine-tuning and distillation; 60.3 on multilingual MTEB retrieval; Apache 2.0. jina-embeddings-v5-text-nano: 239M, EuroBERT-210M backbone, distilled from Qwen3-Embedding-4B, 65.5 MMTEB task mean, CC BY-NC. LEAF: 23M, English, L2 distillation from mxbai-embed-large-v1, no hard negatives, #1 among models of about 30M or less on MTEB v2 English as of late 2025. Added to section 7 |
| 2 | Distillation as the main recipe | **accepted** | Supported by jina-v5 and LEAF, but both use pretrained backbones. Section 4 reordered: stage 4 is distillation, stage 6 the contrastive-only baseline. The 50M comparison is the central experiment |
| 3 | Many more pretraining tokens for the 50M | **accepted as a test** | Plausible, since competitors saw trillions. Unproven for this setup. Ablation 8 in 3.3 measures the 30B vs 100B slope first |
| 4 | Add parallel pairs for bitext mining | **accepted** | mGTE also used translation pairs. The Salesforce result concerned retrieval only. Parallel data is now included, filtered, and weighted among our 12 languages |
| 5 | Pretrain at 8192 from the start | **accepted as an ablation** | Ablation 7 in 3.3 |
| 6 | muP or depth-aware muP for transfer | **open** | CompleteP (depth-aware muP) and Completed(d)P exist and report transfer across width and depth, with 12–34% compute efficiency gains, but on decoder LLMs. Completed(d)P also reports that optimal hyperparameters do not transfer across training duration out of the box. Untested for MLM encoders, so a short 50M-to-100M check would come first |
| 7 | Muon as a late ablation | **accepted** | Unchanged |

### 11.3 Decision: 50M output dimension (resolved: option B)

The 50M has hidden size 384, so a 768-dim output would be a rank-384 projection: twice the storage for the
same information. IBM's 97M granite model also uses 384-dim output. Decided: the **50M outputs 384**
(Matryoshka 384/256/128/64) with no projection layer, and the 100M and 200M stay at 768. The 100M (hidden
576) is rank-limited to 576 by the same argument, to a smaller degree, and keeps 768 for now. During
distillation the teacher targets are Matryoshka-truncated prefixes of its embedding, so no training-only head
is needed (see Stage 4 for the gating check).

---

## 12. Sources

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
- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) and [IBM blog](https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2)
- [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
- [LEAF](https://arxiv.org/html/2509.12539v1) and [MongoDB blog](https://mongodb.com/company/blog/engineering/leaf-distillation-state-of-the-art-text-embedding-models)
- [Ettin: paired encoders and decoders](https://arxiv.org/html/2507.11412v1)
- [CompleteP](https://arxiv.org/html/2505.01618v4) and [Completed(d)P](https://arxiv.org/pdf/2512.22382)
- [FineWeb2-HQ-Plus](https://huggingface.co/datasets/epfml/FineWeb2-HQ-Plus)
