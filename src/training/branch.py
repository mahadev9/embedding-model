"""Branch a decay (optionally after a lower-rate hold) from a constant-rate run's checkpoint.

    python -m src.training.branch --from-run main-lr1.6e-3 --label 1.00B --name decay-1.00B \\
        --decay-tokens 1e8
    python -m src.training.branch --from-run main-lr1.6e-3 --label 3.00B --name hold4e-4-3.00B \\
        --hold-lr 4e-4 --hold-tokens 3e8 --decay-tokens 3e8

The branch copies <from-run>/ckpt_<label>.pt (weights and optimizer state) into a new run, then
trains on the same data stream the main run would have seen next, with the rate going from the
main run's peak to [hold-lr for hold-tokens] and then decaying (1 - sqrt) to zero over
decay-tokens. The main run is untouched. Rerun the same command to resume an interrupted branch.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys

import torch
import yaml

from src.data.paths import RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--label", required=True, help="checkpoint label, e.g. 3.00B")
    parser.add_argument("--name", required=True, help="name of the new run")
    parser.add_argument("--decay-tokens", type=float, required=True)
    parser.add_argument("--hold-lr", type=float)
    parser.add_argument("--hold-tokens", type=float, default=0.0)
    args = parser.parse_args()
    if bool(args.hold_lr) != bool(args.hold_tokens):
        raise SystemExit("--hold-lr and --hold-tokens go together")

    source = RUNS_DIR / args.from_run
    ckpt_path = source / f"ckpt_{args.label}.pt"
    cfg = yaml.safe_load((source / "config.yaml").read_text())
    tokens_per_step = cfg["batch"]["tokens_per_step"]
    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "latest.pt").exists():
        shutil.copy(ckpt_path, run_dir / "latest.pt")
    step = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)["step"]

    side_tokens = args.hold_tokens + args.decay_tokens
    total_tokens = step * tokens_per_step + side_tokens
    total_steps = math.ceil(total_tokens / tokens_per_step)
    sch = cfg["schedule"]
    warmup = max(sch["min_warmup_steps"], int(sch["warmup_fraction"] * total_steps))
    if step < warmup:
        raise SystemExit(f"checkpoint step {step} is inside the warmup of the branch ({warmup})")
    eval_every = max(tokens_per_step, side_tokens / 5)

    overrides = [
        f"schedule.total_tokens={total_tokens:.0f}",
        f"schedule.decay_tokens={args.decay_tokens:.0f}",
        f"schedule.hold_tokens={args.hold_tokens:.0f}",
        f"eval.every_tokens={eval_every:.0f}",
        f"checkpoint.every_tokens={eval_every:.0f}",
        "checkpoint.keep_every_tokens=1e13",
    ]
    if args.hold_lr:
        overrides.append(f"schedule.hold_lr={args.hold_lr:g}")
    print(f"branch {args.name}: from step {step}, {side_tokens / 1e6:.0f}M tokens", flush=True)
    cmd = [
        sys.executable, "-u", "-m", "src.training.mlm",
        "--config", str(source / "config.yaml"),
        "--run-name", args.name,
        "--set", *overrides,
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
