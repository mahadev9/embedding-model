# Loom (loom-embed)

From-scratch multilingual text embedding models (50M and 200M non-embedding parameters), built on a
single RTX 5090. Languages: English, Chinese, Spanish, French, German, Portuguese, Russian, Japanese,
Arabic, Hindi, Korean, Italian. See [docs/implementation_plan.md](docs/implementation_plan.md) for the
research, plan and all measured results.

Loom is not affiliated with the Loom video product.

## Status

The 50M pipeline runs end to end: tokenizer, MLM pretraining (constant-rate trunk plus decay branches),
contrastive fine-tuning, and a four-layer evaluation (MLM loss, fine-tuning probes, a 17-task MMTEB
subset). Results so far are in sections 5.9 to 5.14 of the plan. The embedding model is still a probe:
about 11 to 13 points below multilingual-e5-small on our benchmark, and well behind English-only small
models on English. Large-scale pair pretraining and distillation (plan section 4.2) come next. No model
weights are published yet.

## Setup

Requires Python 3.13 and an NVIDIA driver that supports CUDA 13.0.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu130
```

The extra index is needed so pip picks the CUDA 13.0 build of PyTorch. Dependencies are declared in
[pyproject.toml](pyproject.toml). `-e` installs the project in editable mode, so `import src...` works from
any directory.

## Usage

```bash
python -m src.model.config                      # print parameter counts for all model sizes
python -m src.benchmarks.throughput attention   # attention kernel benchmark
python -m src.benchmarks.model_throughput --sizes 50m   # real-model MLM throughput

python -m src.data.audit                        # source sizes and licenses (API only)
python -m src.data.download sample --mb 300     # ~300 MB text per language, into data/
python -m src.tokenizer.train --mb-per-lang 50  # BPE candidates (48k/64k/96k/128k), ~2.5 min
python -m src.tokenizer.evaluate                # compare candidates with public tokenizers
python -m src.data.build_corpus --total-tokens 10e9   # download + tokenize the pretraining corpus
rm -rf data/                                    # remove all downloaded data

# Optional: send bulk raw shards to another disk (caches follow it)
export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw
export EMBEDDING_TOKENIZED_DIR=/mnt/hdd-2tb/embedding-model/tokenized
export EMBEDDING_RUNS_DIR=/mnt/hdd-2tb/embedding-model/runs

python -m src.training.mlm --config configs/train/mlm_50m.yaml --run-name my-run   # MLM pretraining
python -m src.training.branch --from-run RUN --label 20.00B --name NAME --decay-tokens 2e9   # decay branch

python -m src.data.build_pairs --per-language 100000    # WebFAQ question-answer pairs
python -m src.data.build_parallel --per-pair 40000      # WikiMatrix translation pairs
python -m src.training.contrastive --config configs/train/contrastive_50m.yaml --run-name probe

python -m src.eval.mlm_crossmodel --models ours:RUN/latest.pt hf:jhu-clsp/mmBERT-small   # needs ".[eval]"
python -m src.eval.finetune --models ours:RUN/latest.pt --tasks xnli paws ner
python -m src.eval.mmteb --models ours:RUN/latest.pt    # 17-task MMTEB subset
python -m src.eval.compare --ours ours__RUN__latest.pt  # compare with public baselines

ruff format src && ruff check src               # format and lint
```
