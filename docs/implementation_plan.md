# Multilingual Embedding Models (50M and 200M): Implementation Plan

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
| Sizes | 50M and 200M **non-embedding** parameters. 200M is the main target (a 100M size was dropped 2026-11-13) |
| Start with | **50M**: a cheap end-to-end pipeline test and the platform for ablations |
| Languages | 12 languages (confirmed by user): en, zh, es, fr, de, pt, ru, ja, ar, hi, ko, it |
| Tokenizer | Custom byte-level BPE, shared by all sizes. **96,000-token vocabulary** (decided 2026-11-07, see 10.5) |
| Output embedding | 768 dimensions for the 200M, 384 for the 50M; Matryoshka (MRL) so it can be truncated (768/512/256/128, or 384/256/128/64 for the 50M) |
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
| Output head | Linear projection to `output_dim` where it differs from the hidden size, then L2 normalize. The 50M and 200M need no projection | MRL applied to the output vector |

### 3.2 Starting configurations

Parameter counts are non-embedding, computed from my formula
`layers × (4·hidden² + 3·hidden·intermediate)` *(mine)*. The 96k vocabulary is extra.

| Model | Hidden | Layers | Heads | GeGLU intermediate | Non-emb params | Embedding table (96k vocab) | Total params |
|---|---|---|---|---|---|---|
| 50M | 384 | 22 | 6 | 1536 | ~52M | ~37M | ~89M |
| 200M | 768 | 26 | 12 | 2304 | ~200M | ~74M | ~273M |

For comparison, mmBERT-small is 384 hidden, 22 layers, 1152 intermediate (42M non-embedding), and
ModernBERT-base is 768 hidden, 22 layers, 1152 intermediate (110M non-embedding). Ours use a larger
intermediate size to reach the target counts.

### 3.3 Architecture ablations to run at 50M

Run each for the same token budget (about 5–8B tokens, hours each) and compare by MLM loss plus a quick
embedding probe.

1. Depth vs width at fixed 52M: 22×384 vs 16×448 vs 28×336.
2. Local/global ratio: every 3rd vs every 4th layer global. Window 128 vs 256.
3. Mask rate: 20% vs 30%, and the 30 → 15 → 5 decay schedule vs constant.
4. Vocabulary size: 96k is decided on compression and throughput grounds (10.5). Optional check: 64k vs 96k
   on the 50M's downstream embedding quality, since compression is only a proxy.
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

### 4.2 Optional but recommended: contrastive pretraining as a main stage (proposed 2026-11-10)

Status: **not adopted yet.** Decide after the continued MLM training (the 20B trunk, section 5.10). Until
then sections 4 and 4.1 above stand as written.

Proposal (external review): make large-scale contrastive pretraining a shared main stage instead of arm B
only, so the arms experiment asks what distillation adds on top of it, not whether it replaces it.

```
MLM (Stages 1-3) ──► weak-pair contrastive pretraining (Stage 6: 100M+ pairs, batch 8k+, GradCache)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
   Arm A: hard negatives only          Arm B: distillation only          Arm C: hard negatives + distillation
         └────────────────────────────────────┼────────────────────────────────────┘
                                              ▼
                       best arm ──► merge / soup ──► Stage 7 long-context ──► Phase 2 reranker
```

Optional fourth arm at 50M only: distil straight from the MLM checkpoint (today's arm A). It keeps the
original question, whether distillation can replace pair pretraining for a from-scratch student.

Why it is recommended:
- The "distillation first" evidence (jina-embeddings-v5, LEAF) is from pretrained backbones; for a
  from-scratch student it is untested (Stage 4).
- The reviewer reports that the strong small models checked (granite, harrier, DenseOn, Bekko) run hundreds
  of millions of pairs first and distil afterwards. This matches e5, mGTE and granite; the harrier, DenseOn
  and Bekko recipes are not verified by us.
- Our probes (5.12, 5.13): with 1M pairs, retrieval is about 20 points below e5-small, and neither pooling
  nor parallel pairs moved it.

What it needs before it can start:
- **Data (the main work):** we hold about 1M WebFAQ pairs and 440k parallel pairs. 100M+ needs much more
  of WebFAQ, mined title-body and adjacent-passage pairs from FineWeb2 and Wikipedia, and the English
  weak-pair sets in section 6.
- **Compute:** the probe ran 1M pairs in 23 minutes, so about 38 hours per 100M pairs for the 50M at the
  same text lengths (64/256 tokens); the 200M is not measured.
- **Batch 8k+:** the trainer runs GradCache at batch 512; confirm memory and throughput with a short run.

### Stage 0: Data and tokenizer

- **Text for MLM:** FineWeb-2 for the non-English languages (ODC-By 1.0), and an English quality source
  (FineWeb-Edu or DCLM) plus Wikipedia. mmBERT's English share ranged from 10% to 34% by stage.
- **Language sampling:** `q_i ∝ p_i^α`. Start at α = 0.5 (mGTE) and ablate lowering it to 0.3 in the decay
  phase (mmBERT-like).
- **Tokenizer:** done (10.5): byte-level BPE, 96k vocabulary, NFKC, trained on a balanced sample of the
  12 languages. Special tokens: PAD, UNK, CLS, SEP, MASK plus 11 unused slots (ids 0..15).
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
  native 1024, truncated and re-normalized). The 200M matches the 768-dim vector directly. The 50M
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
| Last layer is global on 50M and 200M | confirmed |
| Graph breaks in the compiled MLM step (varying masked counts) | none |
| Parameter count vs `ModelConfig` | exact match for 50M and 200M |

**MLM training throughput, real model** (bf16, `torch.compile`, packed documents, 30% masked,
`python -m src.benchmarks.model_throughput`):

| Model | Seq len | Packed docs | Tokens/s | Tokens/day | Peak memory |
|---|---|---|---|---|---|
| 50M | 1024 | ~400 tokens | 257k | ~22B | 14.8 GB at 32k tokens/batch |
| 50M | 8192 | ~400 tokens | 248k | ~21B | 14.8 GB at 32k tokens/batch |
| 200M | 1024 | ~400 tokens | 87k | ~7.5B | 16.9 GB at 16k tokens/batch |
| 200M | 8192 | ~400 tokens | 83k | ~7.2B | 16.9 GB at 16k tokens/batch |
| 200M | 8192 | one 8192-token doc | 64k | ~5.5B | 16.9 GB at 16k tokens/batch |
| **With the 96k vocabulary and realistic documents (~1,000 tokens, as in the corpus):** | | | | | |
| 50M | 1024 | ~1000 tokens | 244k | ~21B | 15.5 GB at 32k tokens/batch |
| 50M | 8192 | ~1000 tokens | 214k | ~18.5B | 15.5 GB at 32k tokens/batch |
| 200M | 1024 | ~1000 tokens | 83k | ~7.1B | 17.5 GB at 16k tokens/batch |
| 200M | 8192 | ~1000 tokens | 76k | ~6.6B | 17.5 GB at 16k tokens/batch |

The rows above the bold line used a 64k vocabulary and 400-token documents. Each step builds a fresh batch (new documents and masks) and includes mask building, as in real training.
The first benchmark (previous revision) reused the same label count every step, which hid a recompile and
two graph-break problems; fixing them also cut memory by about 3 GB and raised throughput slightly.

Takeaways:
- Packing short documents into 8192-token rows costs only 4–8% extra per token (4–12% with the corpus's
  ~1,000-token documents), because the mask skips
  cross-document blocks. Pretraining at 8192 on packed short text is nearly as cheap as at 1024.
- Long single documents are the expensive case (the 200M drops to 64k tokens/s). The extension stage should
  budget for that, since long documents are what it trains on.
- The real MLM numbers are lower than the toy benchmark in 5.3 (no head, no masking): the 200M is about
  5.5–7.5B tokens/day rather than 8.5B.

### 5.7 MLM pretraining defaults (Stage 1, decided 2026-11-07)

Defaults for the first runs. Items marked *(sweep)* are tuned before the long runs; *(ablate)* items are
tested at 50M. Numbers are from ModernBERT, mmBERT and NeoBERT unless marked *(mine)*; the optimizer
constants come from machine summaries of those papers and are unverified.

| Setting | Default | Notes |
|---|---|---|
| Objective | MLM only, no next-sentence loss | |
| Mask rate | **30%**, exactly K masked positions per batch | ModernBERT and mmBERT use 30%, NeoBERT 20% *(ablate)*. mmBERT lowers it to 15% then 5% in later phases *(ablate in decay)* |
| Replacement | 100% `[MASK]` | NeoBERT dropped BERT's 80/10/10 split *(mine, not ablated)*. `[CLS]` and `[SEP]` are never masked |
| Optimizer | StableAdamW: AdamW plus per-parameter update clipping, betas (0.90, 0.98), eps 1e-6 | replaces global gradient clipping, which is off |
| Weight decay | 1e-5 on 2-D weights only; none on norms, embeddings, biases | ModernBERT 1e-5, mmBERT 8e-5 *(sweep if time)* |
| Peak learning rate | **sweep first**: 2e-4, 4e-4, 8e-4, 1.6e-3 at 50M, short runs | ModernBERT and mmBERT used 8e-4 for the base size. Short sweeps favor larger rates and optimal rates do not transfer across training length (Completed(d)P), so the winner is a starting point |
| Schedule | Warmup-stable-decay: linear warmup over 1% of tokens (at least 200 steps), constant, then 1 - sqrt decay to 0 over the last 10% | ModernBERT used 1 - sqrt decay *(decay share is mine, since our runs are far shorter)* |
| Batch | 524,288 tokens per step (512 rows of 1024), via gradient accumulation | ModernBERT ramped to 4.7M tokens over 1.7T tokens; at 30B tokens, 0.5M gives about 57k steps. No batch-size warmup initially *(ablate)* |
| Context | 1024 tokens; 8192 in the extension stage | see ablation 7 in 3.3 |
| Precision | bf16 autocast, fp32 weights, `torch.compile` | |
| Dropout | none | few epochs; ModernBERT used attention-output dropout 0.1 *(ablate if overfitting appears)* |
| Language mix | quotas of the corpus plan (alpha = 0.5) as sampling probabilities | section 10.6 |
| Rows | contiguous 1024-token windows of each language's token stream; documents packed, attention masked across `[CLS]` boundaries; a document cut by a window edge becomes two | no padding, nothing dropped |
| Disk reads | runs of 16 consecutive rows per read | the HDD handles about 100–150 random reads/s; training needs about 250 rows/s at 50M |
| Determinism | batch contents depend only on (seed, step); per-language row cursors are saved in checkpoints | exact resume |
| Data pin | `snapshots/corpus_10B.json` for sweeps and ablations; `corpus_40B.json` for the long runs | loaders never read past the snapshot counts, for train or validation |
| Validation | per-language MLM loss on 0.5M held-out tokens per language, fixed masks, every 100M training tokens; macro average over languages is the headline number | |
| Checkpoints | `latest` every 100M tokens, a kept copy every 1B | |
| Logging | JSONL file per run (loss, learning rate, tokens/s, update-clip statistic, memory) | no external services |

### 5.8 Training pipeline status (2026-11-07)

Code in `src/training/`: `data.py` (corpus reader, deterministic row sampling, prefetcher, GPU masking),
`optim.py` (StableAdamW, parameter groups, warmup-stable-decay schedule), `mlm.py` (the trainer).
Configs: `configs/train/mlm_50m.yaml` and `mlm_200m.yaml`. Run from the project root:

```bash
export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw
export EMBEDDING_TOKENIZED_DIR=/mnt/hdd-2tb/embedding-model/tokenized
export EMBEDDING_RUNS_DIR=/mnt/hdd-2tb/embedding-model/runs      # checkpoints and logs (default: ./runs)
python -m src.training.mlm --config configs/train/mlm_50m.yaml --run-name lr8e-4 \
    --set optim.lr=8e-4 schedule.total_tokens=1e9
```

Checks run on a 100M-token smoke run of the 50M model (deleted afterwards):

| Check | Result |
|---|---|
| Batch contents after resuming vs a continuous run (`iter_batches`) | identical |
| Loss after killing at a checkpoint and resuming (step 120) | 8.3623 in both runs |
| Language shares observed over 200 steps vs the mix | within 0.2 percentage points |
| Masking: masked positions, special tokens | exactly 9,830 of 32,768 (30%); `[CLS]` and `[SEP]` never masked |
| Data loader speed from the HDD (one thread, extension running at the same time) | 1.4M tokens/s, against about 0.25M needed |
| End-to-end training speed, 50M, 1024 context | 229–239k tokens/s (the isolated benchmark: 244k), so loader and optimizer overhead is small |
| Memory | 16.2 GB at 32k tokens per micro-batch |
| Loss | 11.48 at start (ln of the 96k vocabulary is 11.47), 7.21 macro validation loss after 100M tokens (still warming up) |

At about 230k tokens/s a 1B-token run takes about 72 minutes and the planned 30B-token run about 36 hours.
The first learning-rate sweep (1B tokens per value, about 5 hours for four values) is the next GPU job.
Notes: the optimizer step is a per-tensor Python loop, which costs little at this step size (about 2 s);
checkpoints hold weights and optimizer (1.07 GB for the 50M) and a weights-only copy every 1B tokens.

### 5.9 Learning-rate sweep results (50M, 1B tokens each, 2026-11-07)

Four runs of the 50M model, same seed, same data order and masks, same schedule shape (warmup 200 steps,
constant, 1 - sqrt decay over the last 190 steps), `python -m src.training.sweep`. Each took 75–77 minutes
at about 230k tokens/s.

| Peak LR | Final macro val loss | Gain vs previous rate | Max update RMS | Loss spikes | NaN |
|---|---|---|---|---|---|
| 2e-4 | 4.335 | | 2.26 | 0 | 0 |
| 4e-4 | 3.546 | -0.789 | 2.21 | 0 | 0 |
| 8e-4 | 3.181 | -0.365 | 2.38 | 0 | 0 |
| **1.6e-3** | **3.055** | **-0.127** | 2.76 | 0 | 0 |

Per-language final loss (1B tokens):

| LR | en | zh | es | fr | de | pt | ru | ja | ar | hi | ko | it |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2e-4 | 4.01 | 5.37 | 3.85 | 3.73 | 4.19 | 4.08 | 3.56 | 4.78 | 4.96 | 4.27 | 5.01 | 4.20 |
| 4e-4 | 3.38 | 4.50 | 3.18 | 3.03 | 3.38 | 3.31 | 2.92 | 3.92 | 4.04 | 3.41 | 4.05 | 3.43 |
| 8e-4 | 3.06 | 4.06 | 2.87 | 2.71 | 3.00 | 2.96 | 2.59 | 3.55 | 3.63 | 3.05 | 3.62 | 3.06 |
| 1.6e-3 | 2.95 | 3.93 | 2.75 | 2.59 | 2.87 | 2.83 | 2.47 | 3.44 | 3.49 | 2.93 | 3.49 | 2.92 |

**Findings.**
1. **The best rate is the top of the grid, and it is stable.** 1.6e-3 won in every language, with no spikes,
   no NaN, and the lowest loss jitter (0.017). The sweep did not find where training becomes unstable.
2. **Returns diminish quickly:** each doubling gained 0.79, 0.37, then 0.13. At this horizon the optimum is
   near 1.6e-3, possibly a little above it.
3. **The lead of 1.6e-3 over 8e-4 narrowed, then reopened.** It was -0.78 at 199M tokens, shrank to -0.09 at
   897M, and widened to -0.13 during the decay phase (an earlier prediction of mine, that it would keep
   shrinking, was wrong).
4. **Annealing helps higher rates more.** The loss improvement per 100M tokens during the decay phase,
   relative to the slope just before it, was 0.62x (2e-4), 1.05x (4e-4), 1.98x (8e-4) and 3.13x (1.6e-3).
   High rates do bounce around the minimum, and lowering the rate recovers it, but after annealing the
   higher rate still wins at this horizon.
5. **This is an early-training measurement.** The loss is about 3, still dropping fast, and the horizon is
   1B of the planned 30B tokens. Optimal rates generally fall as the horizon grows (and, per the
   Completed(d)P paper, do not transfer across training length), so 1.6e-3 is an upper bound for the long run.

**Decision pending:** whether to (a) run 3.2e-3 at 1B tokens to find the top of the curve, and (b) run
8e-4 and 1.6e-3 (and possibly 4e-4) at about 3B tokens to see whether the order changes before fixing the
rate for the 30B run.

---

### 5.10 Constant-rate trunks with decay branches (50M): plan and status

Follow-up to 5.9, based on the WSD paper (Wen et al., 2024): run constant-rate trunks and branch
short decays from their checkpoints, instead of fixing the horizon in advance. Research notes: the
constant phase makes progress along the loss valley, the decay phase removes the oscillation, and a
late decay beats an early one; re-warming after a decay (Ibrahim et al., 2024) is meant for new
data and risks forgetting, so we continue from pre-decay checkpoints instead.

**Tools.** `schedule.decay_fraction=0` keeps the rate constant. Full checkpoints (with optimizer
state) are written every 1B tokens as `ckpt_<label>.pt` (label like `3.00B`). `--stop-at-tokens`
pauses a run; rerunning the same command continues it. `python -m src.training.branch --from-run
RUN --label 3.00B --name NAME --decay-tokens 3e8 [--hold-lr 4e-4 --hold-tokens 3e8]` copies a
checkpoint and trains on the same data stream with an optional lower-rate hold, then a 1-sqrt decay
to zero; the trunk is untouched. Verified: a branch whose hold rate equals the trunk's reproduces
the trunk's losses bit for bit, and the decay reaches ~0.

**Setup.** 50M model, corpus snapshot `corpus_40B` (20B tokens is half an epoch, no repetition),
same seed and data stream for every run. Trunk A is named `main-lr1.6e-3`, trunk B `main-lr8e-4`.

| # | Step | Time | Status |
|---|---|---|---|
| 1 | Trunk A: 1.6e-3 constant, 0 to 3B tokens (`--stop-at-tokens 3e9`) | 3.6 h | done 2026-11-08: macro val 2.760 at 3B, no spikes, max update RMS 2.82 |
| 2a | Branch from A at 1B: decay 0.1B tokens (`decay-1.00B`) | 7 min | dropped: 1B/2B checkpoints were not kept (bug, fixed) |
| 2b | Branch from A at 3B: decay 0.3B (`decay-3.00B`), the reference result | 21 min | done: macro val 2.760 -> 2.612 (all 12 languages improved 0.12-0.17) |
| 2c | Branch from A at 3B: hold 4e-4 for 0.3B, then decay 0.3B (`hold4e-4-3.00B`) | 42 min | done: 2.588 (-0.024 vs 2b in every language) but it trained 0.6B vs 0.3B tokens, so confounded; the control (plain 0.6B decay) is dropped in favour of a token-matched pair at 10B |
| 3a | Trunk B: 8e-4 constant, 0 to 3B tokens | 3.6 h | on hold: decide after 2b/2c (weak predictor of the 20B optimum; 4 h) |
| 3b | Branch from B at 3B: decay 0.3B | 21 min | todo |
| - | **Decision:** pick the peak rate from 2b, 2c, 3b (lowest decayed macro val loss at 3B) | | todo |
| 4 | Continue trunk A (1.6e-3) from 3B to 10B (`--stop-at-tokens 1e10`) | 8.3 h | done: macro val 2.465 at 10B (undecayed), no plateau, max update RMS 2.04 |
| 4b | Branches at 10B, token-matched: plain decay 1B (`decay-10.00B`), and optionally hold 4e-4 for 0.5B + decay 0.5B | 71 min each | plain decay done: 2.465 -> 2.348 at 11.0B (decayed 3B result was 2.612); hold variant done: 2.351, no benefit over plain decay at matched tokens |
| 5 | Continue the winning trunk from 10B to 20B | 11.8 h | done 2026-11-11: macro val 2.347 at 20B (undecayed), still falling about 0.008 per 1B, max update RMS 2.05 over 10-20B. Paused once at 16.14B and resumed exactly |
| 5b | Branch at 20B: decay 2B tokens (final 50M model) | 2.4 h | done: macro val 2.244 at 22.0B (`decay-20.00B`, the final 50M model), see 5.14 |

Total about 32.5 h, one job on the GPU at a time. Both trunks are constant-rate runs, so the loser
simply stops at 3B and the winner continues: nothing is restarted. What this answers: whether the
best peak rate holds at a longer horizon (2b vs 3b), whether high-then-lower beats constant-then-
decay (2c vs 2b), and whether the decayed loss keeps improving with horizon (1B, 3B, 10B, 20B).
Not covered: peak rates above 1.6e-3 (3.2e-3 would need its own stability probe).

Note: the first launch kept no `ckpt_*` files at 1B and 2B, because keep points were only checked
when a regular save was due (the two intervals do not line up); fixed in the trainer, so later runs
keep a full checkpoint every 1B tokens. A's 3B branch point is the pause checkpoint (step 5723).
Steps 2b and 2c run back to back (3a and 3b are on hold); results go below as each step finishes.

### 5.11 Evaluating the pretrained encoder (50M, 10B tokens, 2026-11-09)

Layers of evaluation, cheapest first: (1) per-language MLM validation loss (in every run log);
(2) cross-model masked-token loss in bits per byte; (3) fine-tuning probes (XNLI, PAWS-X, WikiANN),
English-trained and tested zero-shot on other languages; (4) a short contrastive fine-tune and the
MMTEB subset (needs the Stage 5 code, not built yet); (5) long-context and Matryoshka checks later.
Code: `src/eval/mlm_crossmodel.py`, `src/eval/finetune.py` (extra `pip install -e ".[eval]"` for
transformers==5.22.2). Baseline models are cached on the HDD with the other Hugging Face files.

**Layer 2** (`python -m src.eval.mlm_crossmodel`): the same held-out text (300 documents per
language, 1200 UTF-8 bytes each, decoded from our validation files), the same masked 12-character
blocks for every model, loss in bits per byte of masked tokens. Lower is better, macro over 12
languages:

| Model | mask 15% | mask 30% |
|---|---|---|
| ours, 10B + 1B-token decay (`decay-10.00B`) | 1.308 | 1.469 |
| ours, 10B before decay | 1.341 | 1.501 |
| mmBERT-small | 1.402 | 1.610 |
| mmBERT-base | 1.275 | 1.485 |
| XLM-R base | 1.429 | 1.627 |

Ours beats mmBERT-small and XLM-R base, ties mmBERT-base at 30% masking and trails it slightly at
15%. By language (15%) mmBERT-base is better in English (1.02 vs 1.25) and most European languages
and Russian; ours is better in Hindi, Korean and Arabic; Chinese and Japanese are level. The decay
is worth 0.033 bits at both rates. Caveats: the text is close to our own training distribution,
the public models may have seen the same FineWeb-2 documents, and this is not an embedding test.

**Layer 3** (`python -m src.eval.finetune`): linear head on [CLS] (token head for NER), AdamW, batch
32, one seed, learning rate chosen from two values by English validation. Test scores, English-trained
(en = English test; zero-shot = mean over the other languages; NER excludes zh and ja, whose WikiANN
text is split into single characters, in the last column):

| Task | Model | en | zero-shot | all langs |
|---|---|---|---|---|
| XNLI (accuracy) | ours, 10B tokens | 75.9 | 64.4 | 65.8 |
| | mmBERT-small | 82.7 | 72.9 | 74.1 |
| | mmBERT-base | 85.5 | 76.0 | 77.1 |
| | XLM-R base | 81.7 | 72.4 | 73.5 |
| PAWS-X (accuracy) | ours | 93.8 | 78.1 | 80.3 |
| | mmBERT-small | 95.0 | 84.7 | 86.1 |
| | mmBERT-base | 96.1 | 85.6 | 87.1 |
| | XLM-R base | 93.5 | 82.7 | 84.3 |
| NER (F1, excl. zh/ja zero-shot) | ours | 78.7 | 56.5 | 51.7 |
| | mmBERT-small | 82.2 | 64.1 | 58.4 |
| | mmBERT-base | 82.7 | 67.9 | 62.3 |
| | XLM-R base | 82.8 | 66.1 | 60.0 |

Ours trails mmBERT-small (the same 22 x 384 shape, about 300x more pretraining tokens) by 7 to 9
points on every task, and XLM-R base by 6 to 8. English is closer (3 to 7 points) than the other
languages. Our fine-tuning is learning-rate sensitive: PAWS at 4e-4 stayed at chance while 1.5e-4
gave 93.8. The baselines' best rates sit at the bottom of the grid, so their scores may be slightly
low; one seed, so differences under about 0.5 are noise. Masked-byte loss (layer 2) under-predicts
these gaps, so use the tasks, not the loss, to judge pretraining changes.

### 5.12 Layer 4: embedding probe on our MMTEB subset (50M, 2026-11-10)

**Setup.** Contrastive probe `probe-a`: the 10B-token decayed encoder, fine-tuned on about 1M WebFAQ
question-answer pairs (CC BY-4.0; 8 languages at 100k, zh 64k, ar 55k, hi 40k, ko 36k pairs;
`src/data/build_pairs.py`), in-batch negatives with 512-pair single-language batches, GradCache,
temperature 0.05, Matryoshka loss over 384/256/128/64, lr 1e-4, 1,954 steps, **23 minutes** (about
85-90k tokens/s; `configs/train/contrastive_50m.yaml`, `src/training/contrastive.py`). Validation MRR
(positive among 1,000 same-language candidates): 0.603 at step 250, 0.674 at the end; Japanese was
lowest (0.46), the rest 0.65-0.75. WebFAQ is not among the evaluation tasks.

**Evaluation.** `src/eval/mmteb.py` runs 17 MTEB/MMTEB tasks (bitext mining 2, classification 4,
clustering 1, STS 3, pair classification 2, retrieval 5) on the language subsets that lie entirely
within our 12 languages. `src/eval/compare.py` builds the comparison: baseline scores come from the
public embeddings-benchmark/results repository where it has the subset and from our own runs where it
does not (about 1 hour of gap runs); ours is always local. Our runner reproduces the public e5-small
numbers (identical to two decimals on 4 tasks, within 1.2 points on 3 tasks with randomized
classifiers). Percent, mean of the task-type means (macro) over all 17 tasks:

| Model | Bitext | Classif. | Cluster | Pair | Retrieval | STS | Macro |
|---|---|---|---|---|---|---|---|
| EmbeddingGemma-300m | 94.9 | 78.7 | 44.0 | 71.5 | 76.5 | 81.2 | **74.5** |
| multilingual-e5-base | 92.0 | 68.2 | 34.0 | 62.3 | 72.5 | 76.6 | 67.6 |
| multilingual-e5-small | 90.5 | 66.1 | 36.2 | 63.2 | 67.2 | 75.3 | 66.4 |
| granite-embedding-97m | 84.3 | 62.2 | 41.6 | 59.2 | 64.4 | 72.4 | 64.0 |
| **ours, probe-a (50M)** | 53.8 | 59.1 | 41.0 | 55.9 | 47.4 | 63.3 | **53.4** |

Per language (macro over monolingual subsets), ours vs e5-small: en 60.3/71.0, zh 60.8/65.9,
es 58.9/68.2, fr 58.1/63.5, de 55.1/65.8, pt 56.9/60.4, ru 53.9/65.5, ja 54.3/62.7, ar 57.4/67.6,
hi 51.5/65.1, ko 49.7/58.3, it 62.2/68.8.

**Reading.** The probe is far smaller than the baselines' training (e5 and EmbeddingGemma use
billions of pairs), so the 11-21 point macro gap is expected, not a ranking of recipes. The
informative part is where the gap sits: bitext mining (53.8 vs 84-95) is by far the largest, because
the training pairs are monolingual and nothing aligns translations; retrieval on large corpora
(MIRACL, MLQA about 30) is second. Clustering (41.0, above e5) and SIB200 classification (47.0, level
with e5 and granite) are close. Next experiments, cheapest first: add parallel-sentence pairs to the
mix; distill from the teacher (central experiment, section 4.1); more pairs and hard negatives;
CLS vs mean pooling. Caveats: single seed, single run; Hindi and Korean pairs were seen about twice.

### 5.13 Pooling and parallel-sentence probes (50M, 2026-11-10)

**Setup.** Two variants of `probe-a`, same encoder, steps (1,954) and hyperparameters, one change each:

- `probe-b`: mean pooling instead of CLS (`configs/model/50m_mean.yaml`).
- `probe-c`: CLS, with translation pairs mixed in. Each batch is a parallel batch with probability
  0.25 (`data.name=[webfaq,parallel]`, `data.parallel_fraction=0.25`), otherwise a WebFAQ batch.
  Parallel data: WikiMatrix English-X sentence pairs for our 11 other languages, up to 40,000 per
  pair after filtering (`src/data/build_parallel.py`, CC BY-SA).

Validation MRR on WebFAQ (macro over 12 languages): probe-a 0.674, probe-b 0.665, probe-c 0.669.
probe-c reaches 0.988 on held-out parallel pairs, which is saturated and not informative.

Same 17 tasks and subsets as 5.12 (percent):

| Model | Bitext | Classif. | Cluster | Pair | Retrieval | STS | Macro |
|---|---|---|---|---|---|---|---|
| probe-a (CLS) | 53.8 | 59.1 | 41.0 | 55.9 | 47.4 | 63.3 | 53.4 |
| probe-b (mean) | 52.0 | 59.0 | 40.6 | 56.1 | 47.1 | 63.6 | 53.1 |
| probe-c (CLS + parallel) | **73.2** | 59.1 | 37.8 | 55.8 | 47.1 | 61.6 | **55.8** |

**Reading.**

- Pooling: CLS is ahead by 0.3 macro, almost all of it bitext mining (Tatoeba 59.3 vs 56.0); the
  other types are within 0.4. Keep CLS.
- Parallel pairs: bitext mining rises 19.4 points (Tatoeba 59.3 to 70.2, NTREX 48.3 to 76.2), which
  is the whole macro gain. This task mirrors the added training signal. Elsewhere it costs a little:
  clustering -3.2, STS -1.7, and every language's monolingual average drops 1-2 points; retrieval and
  classification are flat. The 0.25 fraction was not tuned; a smaller one may keep most of the gain.
- Neither change moves retrieval (47 vs 64-77 for the baselines) or STS, the largest remaining gaps.
  Those need the distillation and hard-negative experiments, not more mixing.

Caveats: single seed, single run each. The first probe-c evaluation stalled when the GPU driver hung
(no output for over 2 hours); after a reboot with updated drivers the rerun resumed from mteb's cached
tasks and finished normally.

### 5.14 Final 50M model after 20B tokens: results (2026-11-11)

**Model.** `decay-20.00B`: constant 1.6e-3 trunk to 20B tokens (`main-lr1.6e-3`), then a 2B-token 1-sqrt
decay to zero, 22B tokens in total. The 20B trunk ran in two legs with one pause at 16.14B; the resume is
exact (same data stream), loss and update RMS continued unchanged.

**Validation loss (macro over 12 languages, lower is better).**

| Model | Tokens | Decay | Loss |
|---|---|---|---|
| decay-3.00B | 3.3B | 0.3B | 2.612 |
| decay-10.00B | 11.0B | 1B | 2.348 |
| **decay-20.00B** | **22.0B** | **2B** | **2.244** |

The undecayed trunk went 2.760 (3B), 2.465 (10B), 2.388 (15B), 2.347 (20B), about 0.008 per 1B at the end.
The decay gain shrinks with horizon: -0.148 at 3B, -0.117 at 10B, -0.103 at 20B. All 12 languages improved
(en 2.18, zh 3.00, es 2.01, fr 1.86, de 2.02, pt 2.04, ru 1.69, ja 2.64, ar 2.59, hi 2.16, ko 2.64, it 2.10).

**Layer 2** (bits per masked byte, macro): 20B model 1.276 (mask 15%) and 1.442 (30%), against 1.308 and
1.469 for the 10B model.

**Layer 3** (English fine-tune, zero-shot average test over languages, best rate by English dev):

| Task | 10B model | 20B model |
|---|---|---|
| XNLI | 65.8 | 66.3 |
| PAWS-X | 80.3 | 80.0 |
| WikiANN | 51.7 | 51.0 |

All three differences are within single-seed noise (about 0.5).

**Layer 4** (`probe-d`: the probe-a recipe, WebFAQ only, CLS, started from the 20B model; WebFAQ validation
MRR 0.674, identical to probe-a). 17-task benchmark, task-type means (percent):

| | Bitext | Classif. | Cluster | Pair | Retrieval | STS | Macro |
|---|---|---|---|---|---|---|---|
| probe-a (10B model) | 53.8 | 59.1 | 41.0 | 55.9 | 47.4 | 63.3 | 53.4 |
| probe-d (20B model) | 56.7 | 60.0 | 40.8 | 56.7 | 48.0 | 63.3 | 54.3 |

**Reading.** Doubling the MLM tokens (about 12 GPU hours) gave about +0.9 macro on the embedding benchmark,
mostly bitext mining (+2.9; Tatoeba 59.3 to 61.2, NTREX 48.3 to 52.2). 13 of 17 tasks rose, 11 of 12
languages rose, but single tasks are within noise and the layer 3 probes are flat. The parallel-pair mix gave
+2.4 macro for a 25-minute fine-tune (5.13), so the lever is the pair stage, not more MLM tokens. This
supports the proposal in 4.2. Caveats: single seed per run; MLM loss improved 0.10 while downstream moved
little, so loss alone is a weak guide at this size.

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
2. **M1 Tokenizer and corpus:** candidates trained and evaluated (10.5), 96k chosen and set in the YAMLs,
   10B-token corpus built and verified (10.6).
3. **M2 Model and MLM loop at 50M:** done: model code, correctness checks, packing throughput (5.6), data
   loader, trainer and a verified smoke run (5.8). Next: the learning-rate sweep.
4. **M3 50M ablations** (section 3.3), then a full 50M pretraining run.
5. **M4 Teacher benchmark and distillation pipeline at 50M**: measure teacher encoding speed and its quality
   at truncated dimensions (gating check), cache teacher embeddings, distillation loop, evaluation harness with all baselines.
6. **M5 Contrastive pipeline and ablations at 50M**: hard-negative mining, MRL, GradCache baseline run,
   distill vs contrastive vs both, merging.
7. **M6 Scale to 200M** with the best recipe.
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
- **Open decisions:** tokenizer vocabulary size (96k recommended, 10.5), Muon, and whether to test
  muP/CompleteP.

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
  EmbeddingGemma-300m about 107M. So our 200M model has about twice the transformer body of the "300M-class"
  models, with a total of about 250M. State both numbers when
  comparing ("200M non-embedding, ~250M total").
- **Runtime:** MMTEB reports up to two days on one A100 for moderate LLMs on the original English benchmark,
  and uses downsampling to cut cost. A small model's runtime is unknown. Time one on a few tasks first.

### 10.5 Tokenizer results (2026-11-07)

Code: `src/tokenizer/` (`train.py`, `evaluate.py`, `common.py`). Commands:

```bash
python -m src.data.download sample --mb 300      # 12 languages from FineWeb2-HQ-Plus / FineWeb-Edu
python -m src.tokenizer.train --mb-per-lang 50   # ~2.5 min, ~16 GB RAM peak
python -m src.tokenizer.evaluate                 # candidates vs public baselines
```

**Method.** Byte-level BPE (no unknown tokens), NFKC normalization, a Qwen-style pre-tokenizer with single
digits, 16 special tokens at ids 0..15. Equal text bytes per language (so English is only 1/12 of the
training text). One training run at 128k; the 48k, 64k and 96k candidates are exact prefixes of its merge
list (BPE merges are learned in order). Evaluation uses a fixed 5% holdout of the sample, 3–8 MB of text per
language, counted on raw text so normalization is not rewarded or penalized.

**Characters per token** (higher is better; tokenizer trained on 12 languages only, so baselines that cover
100+ languages are at a disadvantage here):

| Tokenizer | en | zh | es | fr | de | pt | ru | ja | ar | hi | ko | it | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours 48k | 3.91 | 1.31 | 4.02 | 3.74 | 3.71 | 3.84 | 3.06 | 1.60 | 2.92 | 3.02 | 1.83 | 3.85 | 3.07 |
| ours 64k | 4.06 | 1.36 | 4.18 | 3.89 | 3.89 | 4.00 | 3.22 | 1.68 | 3.08 | 3.19 | 1.92 | 4.01 | 3.21 |
| **ours 96k** | 4.25 | 1.45 | 4.39 | 4.10 | 4.13 | 4.22 | 3.49 | 1.80 | 3.32 | 3.40 | 2.04 | 4.24 | **3.40** |
| ours 128k | 4.37 | 1.51 | 4.52 | 4.24 | 4.30 | 4.35 | 3.67 | 1.88 | 3.48 | 3.54 | 2.13 | 4.39 | 3.53 |
| XLM-R (250k) | 4.22 | 1.45 | 4.35 | 3.90 | 4.22 | 4.19 | 3.87 | 1.70 | 3.23 | 3.51 | 1.91 | 4.19 | 3.40 |
| granite-97m-r2 (180k) | 4.80 | 1.25 | 4.52 | 4.25 | 4.26 | 4.35 | 3.64 | 1.32 | 3.12 | 3.19 | 1.70 | 3.90 | 3.36 |
| EmbeddingGemma (262k) | 4.67 | 1.41 | 4.45 | 4.06 | 4.10 | 4.13 | 3.64 | 1.70 | 2.88 | 3.73 | 1.76 | 4.04 | 3.38 |
| mmBERT / Gemma-2 (256k) | 4.68 | 1.40 | 4.56 | 4.15 | 4.33 | 4.21 | 3.63 | 1.73 | 2.83 | 2.64 | 1.50 | 4.13 | 3.32 |

Other baselines (macro): EuroBERT 128k 2.91, Qwen3 151k 2.80, ModernBERT 50k 2.28. Bytes per token and
tokens per word tables are in `data/tokenizer/report.md` (generated, not committed).

**Findings.**
1. **A bug caught by the per-language view.** The first pre-tokenizer copied Qwen's pattern, which treats only
   letters as word characters. Devanagari vowel signs are combining marks, so Hindi words were split at every
   mark: 2.0 characters/token even at 128k (Qwen3 shows the same flaw at 1.11). Adding `\p{M}` to the word
   class raised Hindi to 3.40 at 96k and the macro average from 3.30 to 3.40. Any other script whose vowel
   signs are marks (Thai, Bengali and so on) would need the same fix.
2. **Tokenizer training data size does not matter here:** 25 MB and 50 MB per language gave the same scores
   (3.12 / 3.29 / 3.41 vs 3.12 / 3.30 / 3.42 at 64k / 96k / 128k before the fix), so 100 MB is unnecessary.
   (The trainer's memory is the limit: 150 MB per language needed over 31 GB.)
3. **NFKC changes 2.8% of Chinese and 1.0% of Japanese characters** (mostly full-width punctuation folded to
   ASCII) and 0.01–0.04% elsewhere. NFC changes nothing on this text. So the choice matters only for zh and
   ja. Kept NFKC; it is lossy for symbols such as circled digits.
4. **Published tokenizers can mislead a benchmark.** Granite's tokenizer ships with padding enabled and others
   with truncation, which first made granite and EuroBERT look 20x worse. `evaluate.py` now disables both.
5. **English is under-compressed (4.25 vs 4.7–4.8 for the big-vocabulary baselines)** because it is 1/12 of
   the training text. If English-heavy retrieval matters most, up-weight it in tokenizer training.
6. **Cost of a larger vocabulary, measured** (`model_throughput --vocab-sizes`, MLM tokens/s at 1024 context):

| Vocab | 50M tok/s | 200M tok/s | 50M embedding / total params | 200M embedding / total params |
|---|---|---|---|---|
| 48k | 281k | 86k | 18.4M / 70M | 36.9M / 236M |
| 64k | 269k | 85k | 24.6M / 76M | 49.2M / 249M |
| 96k | 252k | 84k | 36.9M / 89M | 73.7M / 273M |
| 128k | 234k | 81k | 49.2M / 101M | 98.3M / 298M |

   Multiplying by characters per token gives text throughput. For the 50M it is flat from 48k to 96k
   (about 855–864k characters/s) and 4% lower at 128k. For the 200M it rises about 4% from 64k to 96k and
   1% more at 128k. In the later stages (distillation, contrastive) the MLM head is not used, so fewer
   tokens per text is a pure saving: 96k needs about 5.6% fewer tokens than 64k.

**Decision: 96k, shared by all three sizes (confirmed 2026-11-07).** It matches XLM-R's 250k vocabulary and beats granite's
180k on these 12 languages, costs the 50M nothing in text throughput, and saves tokens in every later stage.
128k adds only 3.8% more compression for +12M parameters and a 4% throughput loss at 50M. 64k was the
fallback if total model size mattered most (the 50M would be 76M total instead of 89M). Applied to the
model YAMLs (`vocab_size: 96000`) and the `ModelConfig` default.

### 10.6 Tokenized corpus (first 10B tokens)

Code: `src/data/build_corpus.py`. Commands (the two environment variables put the bulk data on the HDD):

```bash
export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw
export EMBEDDING_TOKENIZED_DIR=/mnt/hdd-2tb/embedding-model/tokenized
python -m src.data.build_corpus --total-tokens 10e9 --plan-only   # estimate availability, show the mix
python -m src.data.build_corpus --total-tokens 10e9               # download, tokenize, delete raw shards
```

**Format.** Per language, `train.bin` and `val.bin`: little-endian **uint32** token ids (the 96k vocabulary
does not fit in 16 bits, so 10B tokens is about 40 GB, not 20). Each document is `[CLS] tokens [SEP]`
(ids 2 and 3), so document boundaries are recoverable from the CLS ids. Every 500th document goes to
`val.bin`. The tokenizer and a manifest with its SHA-256 are copied next to the data. The run is resumable
(per-language `progress.json`) and deletes each raw shard after tokenizing it.

**Language mix.** Quota per language is proportional to available tokens to the power alpha = 0.5, capped at
half of what a language has. Availability is estimated by tokenizing one downloaded shard per language and
scaling by the total shard bytes.

**English share correction.** With English limited to FineWeb-Edu's 10B-token sample, alpha = 0.5 gave
English only 5.3% of the corpus. mmBERT used 10–34%, and English tasks are a large part of MTEB. FineWeb-Edu
also has a 100B-token sample, so English availability was under-stated, not the mix rule wrong. Using
`sample/100BT` gives English 15.3%. The mix for 10B tokens:

| Language | Available (est.) | Quota | Share |
|---|---|---|---|
| en | 112.9B | 1.53B | 15.3% |
| zh | 52.0B | 1.04B | 10.4% |
| es | 39.4B | 0.90B | 9.0% |
| fr | 35.0B | 0.85B | 8.5% |
| de | 51.5B | 1.03B | 10.3% |
| pt | 15.8B | 0.57B | 5.7% |
| ru | 143.3B | 1.72B | 17.2% |
| ja | 30.5B | 0.79B | 7.9% |
| ar | 6.6B | 0.37B | 3.7% |
| hi | 2.0B | 0.21B | 2.1% |
| ko | 4.9B | 0.32B | 3.2% |
| it | 22.1B | 0.68B | 6.8% |

**Result (built 2026-11-07, about 1 h 20 min at 2.5–3.5M tokens/s; tokenization on 16 cores is the
bottleneck).** 10.02B training tokens and 20.4M validation tokens, 38 GB on the HDD:

| Language | Train tokens | Documents | Tokens per document |
|---|---|---|---|
| en | 1.53B | 1.37M | ~1,120 |
| zh | 1.04B | 1.03M | ~1,010 |
| es | 0.90B | 0.92M | ~985 |
| fr | 0.85B | 0.84M | ~1,015 |
| de | 1.03B | 1.10M | ~935 |
| pt | 0.57B | 0.62M | ~930 |
| ru | 1.72B | 0.87M | ~1,990 |
| ja | 0.80B | 0.96M | ~830 |
| ar | 0.37B | 0.35M | ~1,055 |
| hi | 0.21B | 0.22M | ~960 |
| ko | 0.32B | 0.31M | ~1,025 |
| it | 0.68B | 0.70M | ~970 |

Checks run on the output: every file size equals the recorded token count times 4; all ids are below 96,000
(max 95,999); documents start with `[CLS]` and end with `[SEP]`; a decoded document per language reads as
clean text (including hi, ar, zh, ja, ko). Documents average about 1,000 tokens (Russian about 2,000), more than
the 400 assumed in the earlier throughput benchmark, so the benchmark was re-run with 1,000 (5.6). The corpus can be grown later (see below).

**Corpus size decision (2026-11-07): extend to 40B tokens.** The 10B corpus is smaller than the planned runs
(about 30B for the main 50M and 200M runs, up to 100B for the optional long 50M run), which would mean 3
and 10 epochs. The best evidence found (Muennighoff et al., decoder language models, not verified for MLM
encoders) is that up to about 4 epochs of repetition costs little, with returns decaying after that. At 40B
a 30B run is under one epoch and a 100B run is 2.5. 40B is also about the ceiling for this mix: at alpha 0.5
Hindi would use 40% of the data it has and Korean 26%; at 80B both hit the 50% cap and the mix would need
rebalancing. Per-language quotas at 40B: en 6.1B, ru 6.9B, zh 4.1B, de 4.1B, es 3.6B, fr 3.4B, ja 3.2B, it 2.7B,
pt 2.3B, ar 1.5B, ko 1.3B, hi 0.8B.

**Growing and pinning.** The corpus files only ever grow (append-only, fixed seeded shard order). To extend,
move `plan.json` aside and rerun with a larger `--total-tokens`. To make experiments reproducible across
growth, `--snapshot NAME` records each language's token counts to `snapshots/NAME.json`; readers use only up
to those counts. `snapshots/corpus_10B.json` (10,016,130,123 train tokens) pins the first 10B, which the 50M
ablations use while the extension runs in the background at low CPU priority (`nice`).

**Extension result (finished 2026-11-07 evening, about 4.5 h at low priority).** 40.01B training tokens and
80.6M validation tokens, 150 GB on the HDD (1.7 TB free). Every language reached its quota:

| Language | Train tokens | Documents | | Language | Train tokens | Documents |
|---|---|---|---|---|---|---|
| en | 6.11B | 5.47M | | ja | 3.18B | 3.84M |
| zh | 4.14B | 4.14M | | ar | 1.48B | 1.39M |
| es | 3.61B | 3.74M | | hi | 0.82B | 0.85M |
| fr | 3.40B | 3.33M | | ko | 1.28B | 1.25M |
| de | 4.13B | 4.43M | | it | 2.70B | 2.80M |
| pt | 2.29B | 2.46M | | ru | 6.88B | 3.47M |

Checks: file sizes equal the recorded counts; all ids below 96,000; the original 10B tokens are unchanged
(the old data still ends in `[SEP]` and the appended data starts with `[CLS]`); appended text decodes
cleanly in all 12 languages. `snapshots/corpus_40B.json` pins the full 40.01B; `corpus_10B.json` still pins
the first 10B (sweeps and ablations). Validation text is pinned the same way, so a snapshot's validation
loss stays comparable after growth.

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
| 1 | 200M (26 layers) ends on a local layer, so the final CLS token sees only +-64 tokens | **confirmed** (layer 25 was local; the 50M ends global) | Last layer is now always global, for any depth. 200M has 10 global layers |
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
| 6 | muP or depth-aware muP for transfer | **open** | CompleteP (depth-aware muP) and Completed(d)P exist and report transfer across width and depth, with 12–34% compute efficiency gains, but on decoder LLMs. Completed(d)P also reports that optimal hyperparameters do not transfer across training duration out of the box. Untested for MLM encoders, so a short 50M-to-200M check would come first |
| 7 | Muon as a late ablation | **accepted** | Unchanged |

### 11.3 Decision: 50M output dimension (resolved: option B)

The 50M has hidden size 384, so a 768-dim output would be a rank-384 projection: twice the storage for the
same information. IBM's 97M granite model also uses 384-dim output. Decided: the **50M outputs 384**
(Matryoshka 384/256/128/64) with no projection layer, and the 200M stays at 768. During
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
