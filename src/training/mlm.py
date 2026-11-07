"""MLM pretraining loop.

    python -m src.training.mlm --config configs/train/mlm_50m.yaml --run-name lr8e-4
    python -m src.training.mlm --config configs/train/mlm_50m.yaml --run-name lr4e-4 \\
        --set optim.lr=4e-4 schedule.total_tokens=3e8

Reads the pinned token files, trains with StableAdamW and a warmup-stable-decay schedule, evaluates
per-language validation loss, writes JSONL logs and checkpoints to <RUNS_DIR>/<run-name>, and
resumes automatically if that directory already has a latest.pt. Run it from the project root.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from src.data.paths import RUNS_DIR
from src.model import MLMModel, build_block_masks, load_model_config

from .data import CorpusReader, Prefetcher, make_mlm_batch
from .optim import StableAdamW, build_param_groups, wsd_lr

DEVICE = "cuda"


def _coerce_numbers(node):
    """PyYAML reads `3.0e10` and `1e8` as strings (it wants `3.0e+10`); turn them into floats."""
    if isinstance(node, dict):
        return {k: _coerce_numbers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_numbers(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return node
    return node


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise SystemExit(f"unknown config key {key}")
        node[leaf] = yaml.safe_load(value)
    return _coerce_numbers(cfg)


def run_micro_batches(compiled, rows, cfg, model_cfg, generator, backward: bool) -> torch.Tensor:
    """Run one optimizer step's rows in micro-batches; returns the mean loss (a GPU scalar)."""
    micro_rows = cfg["batch"]["micro_tokens"] // cfg["data"]["seq_len"]
    n_micro = rows.shape[0] // micro_rows
    total = torch.zeros((), device=DEVICE)
    for m in range(n_micro):
        micro = rows[m * micro_rows : (m + 1) * micro_rows]
        inputs, doc_ids, positions, targets = make_mlm_batch(
            micro, cfg["data"]["mask_rate"], generator
        )
        block_masks = build_block_masks(
            doc_ids, model_cfg.local_window
        )  # outside the compiled model
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(inputs, doc_ids, positions, targets, block_masks=block_masks)
        if backward:
            (loss / n_micro).backward()
        total += loss.detach() / n_micro
    return total


@torch.no_grad()
def evaluate(compiled, reader: CorpusReader, cfg: dict, model_cfg) -> dict[str, float]:
    """Per-language MLM loss on fixed validation rows with fixed masks."""
    seq_len = cfg["data"]["seq_len"]
    micro_rows = cfg["batch"]["micro_tokens"] // seq_len
    rows = max(micro_rows, cfg["eval"]["tokens_per_language"] // seq_len // micro_rows * micro_rows)
    losses = {}
    for k, code in enumerate(reader.codes):
        val = reader.val_rows(code, rows)
        val = val[: len(val) // micro_rows * micro_rows].to(DEVICE)
        generator = torch.Generator(DEVICE).manual_seed(10_000 + k)  # same masks every evaluation
        losses[code] = run_micro_batches(compiled, val, cfg, model_cfg, generator, False).item()
    losses["macro"] = sum(losses.values()) / len(losses)
    return losses


def save_checkpoint(
    run_dir: Path, model, optimizer, step: int, cfg: dict, keep: str | None
) -> None:
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    tmp = run_dir / "latest.pt.tmp"
    torch.save(payload, tmp)
    tmp.rename(run_dir / "latest.pt")
    if keep:
        torch.save({"model": model.state_dict(), "step": step}, run_dir / f"model_{keep}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides"
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    torch.manual_seed(cfg["seed"])
    model_cfg = load_model_config(cfg["model"])
    model = MLMModel(model_cfg).to(DEVICE)
    compiled = torch.compile(model) if cfg["compile"] else model
    o = cfg["optim"]
    optimizer = StableAdamW(
        build_param_groups(model, o["weight_decay"]),
        lr=o["lr"],
        betas=tuple(o["betas"]),
        eps=o["eps"],
    )

    seq_len = cfg["data"]["seq_len"]
    tokens_per_step = cfg["batch"]["tokens_per_step"]
    micro_tokens = cfg["batch"]["micro_tokens"]
    assert tokens_per_step % micro_tokens == 0 and micro_tokens % seq_len == 0
    rows_per_step = tokens_per_step // seq_len
    sch = cfg["schedule"]
    total_steps = math.ceil(sch["total_tokens"] / tokens_per_step)
    warmup_steps = max(sch["min_warmup_steps"], int(sch["warmup_fraction"] * total_steps))
    decay_steps = int(sch["decay_fraction"] * total_steps)

    reader = CorpusReader(
        cfg["data"]["snapshot"], seq_len, cfg["data"]["run_rows"], seed=cfg["seed"]
    )
    start_step = 0
    latest = run_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{model_cfg.summary()}\nsteps {total_steps} (warmup {warmup_steps}, decay {decay_steps}), "
        f"{tokens_per_step} tokens/step, params {n_params / 1e6:.1f}M, languages {reader.codes}"
    )

    log_path = run_dir / "log.jsonl"
    eval_every = max(1, int(cfg["eval"]["every_tokens"] // tokens_per_step))
    ckpt_every = max(1, int(cfg["checkpoint"]["every_tokens"] // tokens_per_step))
    keep_every = max(1, int(cfg["checkpoint"]["keep_every_tokens"] // tokens_per_step))
    log_every = cfg["log_every_steps"]

    def log(record: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    loader = Prefetcher(reader.iter_batches(start_step, total_steps, rows_per_step))
    window_loss = torch.zeros((), device=DEVICE)
    window_start = time.time()
    excluded = 0.0  # seconds in eval/checkpoint within the current logging window
    window_steps = 0
    for step, rows in zip(range(start_step, total_steps), loader, strict=False):
        lr = wsd_lr(step, total_steps, o["lr"], warmup_steps, decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        generator = torch.Generator(DEVICE).manual_seed(cfg["seed"] * 1_000_003 + step)
        loss = run_micro_batches(
            compiled, rows.to(DEVICE, non_blocking=True), cfg, model_cfg, generator, True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        window_loss += loss
        window_steps += 1
        done = step + 1
        tokens = done * tokens_per_step

        if done % log_every == 0 or done == total_steps:
            torch.cuda.synchronize()
            elapsed = time.time() - window_start - excluded
            rms = optimizer.last_rms
            record = {
                "type": "train",
                "step": done,
                "tokens": tokens,
                "loss": window_loss.item() / window_steps,
                "lr": lr,
                "tok_per_s": window_steps * tokens_per_step / elapsed,
                "update_rms_mean": rms.mean().item(),
                "update_rms_max": rms.max().item(),
                "mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            log(record)
            print(
                f"step {done}/{total_steps} tokens {tokens / 1e6:.0f}M loss {record['loss']:.4f} "
                f"lr {lr:.2e} {record['tok_per_s'] / 1e3:.0f}k tok/s "
                f"rms max {record['update_rms_max']:.2f} mem {record['mem_gb']:.1f}GB",
                flush=True,
            )
            window_loss.zero_()
            window_steps = 0
            excluded = 0.0
            window_start = time.time()

        if done % eval_every == 0 or done == total_steps:
            side_start = time.time()
            losses = evaluate(compiled, reader, cfg, model_cfg)
            log({"type": "eval", "step": done, "tokens": tokens, "loss": losses})
            print(
                f"  eval @ {tokens / 1e6:.0f}M tokens: macro {losses['macro']:.4f}  "
                + " ".join(f"{c} {v:.2f}" for c, v in losses.items() if c != "macro"),
                flush=True,
            )
            excluded += time.time() - side_start
        if done % ckpt_every == 0 or done == total_steps:
            side_start = time.time()
            keep = f"{tokens / 1e9:.1f}B" if done % keep_every == 0 or done == total_steps else None
            save_checkpoint(run_dir, model, optimizer, done, cfg, keep)
            excluded += time.time() - side_start


if __name__ == "__main__":
    main()
