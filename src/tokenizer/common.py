"""Shared tokenizer constants, text loading, and the BPE prefix-truncation helper."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from src.data.languages import LANGUAGES
from src.data.paths import DATA_DIR, SAMPLE_DIR

TOKENIZER_DIR = DATA_DIR / "tokenizer"

# 16 special tokens: ids 0..15, then the 256 byte tokens, then merges.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
    f"[unused{i}]" for i in range(11)
]

# Qwen-style pre-tokenizer pattern, with combining marks (\p{M}) treated as part of words.
# Without \p{M}, scripts whose vowel signs are marks (Devanagari, and others) are split at every
# vowel sign: Hindi dropped to ~2.0 characters/token while other tokenizers get 3.2-3.7.
# Single digits (\p{N}) mean numbers are always split digit by digit.
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

HOLDOUT_EVERY = 20  # every 20th document is held out for evaluation (5%)


def sample_texts(code: str, split: str, max_bytes: int | None = None) -> Iterator[str]:
    """Yield documents of one language's sample. split is 'train' or 'eval' (a fixed 5% holdout)."""
    path = SAMPLE_DIR / f"{code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python -m src.data.download sample")
    total = 0
    texts = pq.read_table(path, columns=["text"]).column("text")
    for i in range(len(texts)):
        is_eval = i % HOLDOUT_EVERY == 0
        if is_eval != (split == "eval"):
            continue
        text = texts[i].as_py()
        yield text
        total += len(text.encode("utf-8"))
        if max_bytes is not None and total >= max_bytes:
            return


def all_codes() -> list[str]:
    return [lang.code for lang in LANGUAGES]


def truncate_bpe(tokenizer: Tokenizer, vocab_size: int) -> Tokenizer:
    """Return the BPE tokenizer with only the first `vocab_size` token ids.

    BPE merges are learned greedily in order, so a vocabulary of size V trained on the same data
    is exactly the first V - 272 merges of a larger one (ids follow merge order). That lets one
    training run give every smaller candidate.
    """
    data = json.loads(tokenizer.to_str())
    vocab: dict[str, int] = data["model"]["vocab"]
    kept = {token: idx for token, idx in vocab.items() if idx < vocab_size}
    merges = data["model"]["merges"]
    kept_merges = []
    for merge in merges:
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        if left + right in kept:
            kept_merges.append(merge)
    data["model"]["vocab"] = kept
    data["model"]["merges"] = kept_merges
    return Tokenizer.from_str(json.dumps(data))


def candidate_path(vocab_size: int) -> Path:
    return TOKENIZER_DIR / f"bpe_{vocab_size // 1000}k" / "tokenizer.json"
