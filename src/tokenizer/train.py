"""Train a byte-level BPE on the balanced 12-language sample and derive smaller candidates.

    python -m src.tokenizer.train --mb-per-lang 150

Trains once at the largest vocabulary (default 128000) and writes the smaller ones (48k, 64k, 96k)
as exact merge-prefixes of it. Output goes to data/tokenizer/bpe_<N>k/tokenizer.json.

Design: NFKC normalization, a Qwen-style pre-tokenizer (single digits, punctuation and whitespace
runs), byte-level alphabet so no input is ever unknown, 16 special tokens at ids 0..15.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from .common import (
    PRETOKENIZE_PATTERN,
    SPECIAL_TOKENS,
    all_codes,
    candidate_path,
    sample_texts,
    truncate_bpe,
)


def build_untrained_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def training_texts(codes: list[str], mb_per_lang: int) -> Iterator[str]:
    """Equal text bytes per language, interleaved so no language arrives as one long block."""
    streams = {code: sample_texts(code, "train", mb_per_lang * 10**6) for code in codes}
    while streams:
        for code in list(streams):
            try:
                yield next(streams[code])
            except StopIteration:
                del streams[code]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mb-per-lang", type=int, default=150)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[48000, 64000, 96000, 128000])
    parser.add_argument("--langs", nargs="+", default=all_codes())
    args = parser.parse_args()

    largest = max(args.vocab_sizes)
    assert all(v % 64 == 0 for v in args.vocab_sizes), "vocab sizes must be multiples of 64"

    tokenizer = build_untrained_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=largest,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"training BPE vocab={largest} on {len(args.langs)} languages x {args.mb_per_lang} MB")
    start = time.time()
    tokenizer.train_from_iterator(training_texts(args.langs, args.mb_per_lang), trainer)
    print(
        f"trained in {(time.time() - start) / 60:.1f} min; vocab size {tokenizer.get_vocab_size()}"
    )

    for size in sorted(args.vocab_sizes):
        candidate = tokenizer if size == largest else truncate_bpe(tokenizer, size)
        out = candidate_path(size)
        out.parent.mkdir(parents=True, exist_ok=True)
        candidate.save(str(out))
        print(f"wrote {out} (vocab {candidate.get_vocab_size()})")


if __name__ == "__main__":
    main()
