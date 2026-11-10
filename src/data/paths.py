"""Where data lives. Everything is under <repo>/data/ by default so it is easy to delete in one go.

Bulk raw downloads (multi-GiB shards) can be moved to another disk by setting EMBEDDING_RAW_DIR,
e.g. on a large HDD:

    export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw

The tokenized corpus can be moved the same way with EMBEDDING_TOKENIZED_DIR. The Hugging Face, MTEB,
sentence-transformers and torch caches follow RAW_DIR. Small data (audit, tokenizer sample,
tokenizer) stays under DATA_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = Path(os.environ.get("EMBEDDING_RAW_DIR", DATA_DIR / "raw"))
TOKENIZED_DIR = Path(os.environ.get("EMBEDDING_TOKENIZED_DIR", DATA_DIR / "tokenized"))
RUNS_DIR = Path(os.environ.get("EMBEDDING_RUNS_DIR", REPO_ROOT / "runs"))  # checkpoints and logs
SAMPLE_DIR = DATA_DIR / "sample"  # ~300 MB of text per language, for tokenizer training
AUDIT_DIR = DATA_DIR / "audit"
HF_CACHE_DIR = RAW_DIR / ".hf_cache"


def use_local_hf_cache() -> None:
    """Keep Hugging Face caches next to the raw data instead of ~/.cache.

    Must run before huggingface_hub is imported. HF_HOME is left alone so the stored login token
    (~/.cache/huggingface/token) is still found.
    """
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
    # Other libraries that would otherwise write under ~/.cache (the SSD).
    os.environ.setdefault("MTEB_CACHE", str(RAW_DIR / ".mteb_cache"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(RAW_DIR / ".sentence_transformers"))
    os.environ.setdefault("TORCH_HOME", str(RAW_DIR / ".torch"))
