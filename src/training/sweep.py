"""Learning-rate sweep: the same short training run at several peak learning rates.

    python -m src.training.sweep --lrs 2e-4 4e-4 8e-4 1.6e-3 --tokens 1e9     # run (resumable)
    python -m src.training.sweep --lrs 2e-4 4e-4 8e-4 1.6e-3 --tokens 1e9 --summarize

Runs go one after another (they would only slow each other down on one GPU). Everything except the
learning rate is identical across runs: model initialization, data order, masks and schedule shape.
Each run is named sweep-lr<value>; finished runs are skipped and interrupted ones resume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from src.data.paths import RUNS_DIR


def run_name(lr: float) -> str:
    return f"sweep-lr{lr:g}"


def read_log(lr: float) -> list[dict]:
    path = RUNS_DIR / run_name(lr) / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def is_finished(lr: float, tokens: float) -> bool:
    evals = [r for r in read_log(lr) if r["type"] == "eval"]
    return bool(evals) and evals[-1]["tokens"] >= tokens


def summarize(lrs: list[float]) -> None:
    header = f"{'peak lr':>9s} {'tokens':>8s} {'val macro':>10s} {'train loss':>11s}"
    print(header + f" {'max rms':>8s}  worst-lang")
    for lr in lrs:
        log = read_log(lr)
        evals = [r for r in log if r["type"] == "eval"]
        trains = [r for r in log if r["type"] == "train"]
        if not evals:
            print(f"{lr:9g} (no evaluation yet)")
            continue
        last = evals[-1]
        per_lang = {c: v for c, v in last["loss"].items() if c != "macro"}
        worst = max(per_lang, key=per_lang.get)
        print(
            f"{lr:9g} {last['tokens'] / 1e6:7.0f}M {last['loss']['macro']:10.4f} "
            f"{trains[-1]['loss']:11.4f} {max(t['update_rms_max'] for t in trains):8.2f}  "
            f"{worst} {per_lang[worst]:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--lrs", nargs="+", type=float, required=True)
    parser.add_argument("--tokens", type=float, default=1e9)
    parser.add_argument("--config", default="configs/train/mlm_50m.yaml")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if not args.summarize:
        for lr in args.lrs:
            if is_finished(lr, args.tokens):
                print(f"{run_name(lr)}: finished, skipping", flush=True)
                continue
            print(f"=== {run_name(lr)} ({args.tokens / 1e9:g}B tokens) ===", flush=True)
            cmd = [
                sys.executable, "-u", "-m", "src.training.mlm",
                "--config", args.config,
                "--run-name", run_name(lr),
                "--set", f"optim.lr={lr:g}", f"schedule.total_tokens={args.tokens:g}",
                "checkpoint.keep_every_tokens=1e12",
            ]  # fmt: skip
            subprocess.run(cmd, check=True)
    summarize(args.lrs)


if __name__ == "__main__":
    main()
