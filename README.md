# embedding-model

From-scratch multilingual text embedding models (50M / 100M / 200M non-embedding parameters), built on a
single RTX 5090. See [docs/implementation_plan.md](docs/implementation_plan.md) for the research and plan.

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
rm -rf data/                                    # remove all downloaded data

# Optional: send bulk raw shards to another disk (caches follow it)
export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw

ruff format src && ruff check src               # format and lint
```
