"""
Throughput benchmarks for an RTX 5090 (sm_120): attention kernels and encoder training.

Examples:
    python -m src.benchmarks.throughput attention
    python -m src.benchmarks.throughput train --sizes 50M 100M --tokens-per-batch 65536
    python -m src.benchmarks.throughput train --sizes 200M --tokens-per-batch 16384
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = "cuda"
DTYPE = torch.bfloat16
VOCAB_SIZE = 65536


def time_fn(fn: Callable[[], None], iters: int = 10, warmup: int = 3) -> float:
    """Average seconds per call, after warmup, with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


# --------------------------------------------------------------------------- attention


def bench_attention(tokens_per_batch: int = 32768, heads: int = 12, head_dim: int = 64) -> None:
    print(
        f"== attention fwd+bwd, {tokens_per_batch} tokens/batch, {heads} heads x {head_dim}, bf16"
    )
    backends = {
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    compiled_flex = torch.compile(flex_attention)
    for seq_len in (512, 8192):
        _bench_attention_len(seq_len, tokens_per_batch, heads, head_dim, backends, compiled_flex)


def _bench_attention_len(
    seq_len: int,
    tokens_per_batch: int,
    heads: int,
    head_dim: int,
    backends: dict[str, SDPBackend],
    compiled_flex: Callable,
    window: int = 128,
) -> None:
    batch = tokens_per_batch // seq_len
    q, k, v = (
        torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE).requires_grad_()
        for _ in range(3)
    )
    # forward = 4 * B*H*T^2*D FLOPs; backward is roughly 2.5x forward.
    flops = 4 * batch * heads * seq_len**2 * head_dim * 3.5

    def run_sdpa(backend: SDPBackend) -> None:
        with sdpa_kernel(backend):
            F.scaled_dot_product_attention(q, k, v).sum().backward()

    for name, backend in backends.items():
        try:
            sec = time_fn(lambda backend=backend: run_sdpa(backend))
            print(f"T={seq_len} {name:9s} {flops / sec / 1e12:6.1f} TFLOP/s  {sec * 1e3:7.1f} ms")
        except RuntimeError as err:
            print(f"T={seq_len} {name:9s} FAIL {str(err)[:80]}")

    block_mask = create_block_mask(
        lambda b, h, qi, ki: (qi - ki).abs() <= window // 2,
        batch,
        None,
        seq_len,
        seq_len,
        device=DEVICE,
    )

    def run_flex() -> None:
        compiled_flex(q, k, v, block_mask=block_mask).sum().backward()

    try:
        print(f"T={seq_len} flex-local{window} {time_fn(run_flex) * 1e3:7.1f} ms")
    except RuntimeError as err:
        print(f"T={seq_len} flex-local{window} FAIL {str(err)[:100]}")


# --------------------------------------------------------------------------- encoder


@dataclass(frozen=True)
class EncoderConfig:
    hidden: int
    layers: int
    heads: int
    ffn: int


# Starting-point configs; see the architecture notes in the project plan.
CONFIGS: dict[str, EncoderConfig] = {
    "50M": EncoderConfig(hidden=384, layers=22, heads=6, ffn=1536),
    "100M": EncoderConfig(hidden=576, layers=22, heads=9, ffn=1728),
    "200M": EncoderConfig(hidden=768, layers=26, heads=12, ffn=2304),
}


class Block(nn.Module):
    """Pre-norm transformer block with GeGLU feed-forward and no biases."""

    def __init__(self, hidden: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = nn.RMSNorm(hidden)
        self.ffn_in = nn.Linear(hidden, 2 * ffn, bias=False)
        self.ffn_out = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(self.attn_norm(x)).view(batch, seq_len, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, seq_len, hidden)
        )
        x = x + self.attn_out(attn)
        gate, up = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.ffn_out(F.gelu(gate) * up)


class Encoder(nn.Module):
    def __init__(self, vocab: int, cfg: EncoderConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, cfg.hidden)
        self.blocks = nn.ModuleList(
            Block(cfg.hidden, cfg.heads, cfg.ffn) for _ in range(cfg.layers)
        )
        self.final_norm = nn.RMSNorm(cfg.hidden)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def _time_train_step(cfg: EncoderConfig, seq_len: int, tokens_per_batch: int):
    """Return (seconds per step, non-embedding params, peak GB); locals are freed on return."""
    model = Encoder(VOCAB_SIZE, cfg).to(DEVICE)
    non_embedding = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("embed"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    compiled = torch.compile(model)
    ids = torch.randint(0, VOCAB_SIZE, (tokens_per_batch // seq_len, seq_len), device=DEVICE)

    def step() -> None:
        with torch.autocast("cuda", DTYPE):
            compiled(ids).float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    sec = time_fn(step, iters=5)
    return sec, non_embedding, torch.cuda.max_memory_allocated() / 2**30


def bench_train_step(name: str, cfg: EncoderConfig, seq_len: int, tokens_per_batch: int) -> None:
    try:
        sec, non_embedding, peak_gb = _time_train_step(cfg, seq_len, tokens_per_batch)
    except torch.OutOfMemoryError:
        print(f"{name} T={seq_len:5d} OOM at {tokens_per_batch} tokens/batch")
        return
    finally:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tok_per_s = tokens_per_batch / sec
    tflops = 6 * non_embedding * tok_per_s / 1e12
    print(
        f"{name} non-emb={non_embedding / 1e6:.0f}M T={seq_len:5d} "
        f"{tok_per_s / 1e3:7.1f}k tok/s  ~{tflops:5.1f} TFLOP/s (matmul only)  "
        f"mem={peak_gb:.1f}GB  -> {tok_per_s * 86400 / 1e9:.1f}B tok/day"
    )


def bench_train(sizes: list[str], seq_lens: list[int], tokens_per_batch: int) -> None:
    print(f"== encoder train step (bf16, torch.compile, vocab {VOCAB_SIZE}, no MLM head)")
    for name in sizes:
        for seq_len in seq_lens:
            bench_train_step(name, CONFIGS[name], seq_len, tokens_per_batch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("attention", help="benchmark attention kernels")
    train = sub.add_parser("train", help="benchmark encoder training steps")
    train.add_argument("--sizes", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    train.add_argument("--seq-lens", nargs="+", type=int, default=[512, 2048, 8192])
    train.add_argument("--tokens-per-batch", type=int, default=16384)
    args = parser.parse_args()

    if args.command == "attention":
        bench_attention()
    else:
        bench_train(args.sizes, args.seq_lens, args.tokens_per_batch)


if __name__ == "__main__":
    main()
