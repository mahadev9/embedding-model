"""MLM training throughput of the real model with packed, document-masked batches.

Examples:
    python -m src.benchmarks.model_throughput --sizes 50m --seq-lens 1024 8192
    python -m src.benchmarks.model_throughput --sizes 200m --tokens-per-batch 16384
"""

from __future__ import annotations

import argparse
import time

import torch

from src.model import MLMModel, build_block_masks, load_model_config

DEVICE = "cuda"


def make_packed_batch(
    batch: int, seq_len: int, vocab: int, mean_doc_len: int, mask_rate: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random tokens, geometric-ish document lengths packed into each row, 30% masked labels."""
    input_ids = torch.randint(5, vocab, (batch, seq_len), device=DEVICE)
    boundaries = torch.rand(batch, seq_len, device=DEVICE) < (1.0 / mean_doc_len)
    doc_ids = boundaries.cumsum(dim=1).to(torch.int32)
    labels = torch.full_like(input_ids, -100)
    masked = torch.rand(batch, seq_len, device=DEVICE) < mask_rate
    labels[masked] = input_ids[masked]
    return input_ids, doc_ids, labels


def bench(size: str, seq_len: int, tokens_per_batch: int, mean_doc_len: int, iters: int) -> None:
    cfg = load_model_config(size)
    batch = tokens_per_batch // seq_len
    model = MLMModel(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    compiled = torch.compile(model)
    input_ids, doc_ids, labels = make_packed_batch(batch, seq_len, cfg.vocab_size, mean_doc_len)

    def step() -> None:
        with torch.autocast("cuda", torch.bfloat16):
            compiled(input_ids, doc_ids, labels).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    try:
        for _ in range(3):  # warmup / compile
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        sec = (time.perf_counter() - start) / iters

        # Cost of building the two block masks, which happens once per batch.
        build_block_masks(doc_ids, cfg.local_window)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            build_block_masks(doc_ids, cfg.local_window)
        torch.cuda.synchronize()
        mask_ms = (time.perf_counter() - start) / iters * 1e3
    except torch.OutOfMemoryError:
        print(f"{size} T={seq_len}: OOM at {tokens_per_batch} tokens/batch")
        return

    tok_per_s = tokens_per_batch / sec
    print(
        f"{size} non-emb={cfg.non_embedding_params / 1e6:.0f}M T={seq_len:5d} "
        f"docs~{mean_doc_len} {tok_per_s / 1e3:7.1f}k tok/s  mask-build {mask_ms:5.1f} ms  "
        f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB  "
        f"-> {tok_per_s * 86400 / 1e9:.1f}B tok/day"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["50m"])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--mean-doc-len", type=int, default=400)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    for size in args.sizes:
        for seq_len in args.seq_lens:
            bench(size, seq_len, args.tokens_per_batch, args.mean_doc_len, args.iters)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
