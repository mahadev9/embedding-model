"""Contrastive fine-tuning of the embedding model with in-batch negatives and GradCache.

    python -m src.training.contrastive --config configs/train/contrastive_50m.yaml \\
        --run-name probe-a
    python -m src.training.contrastive --config configs/train/contrastive_50m.yaml \\
        --run-name smoke --set schedule.total_pairs=20000 data.name=smoke

Starts from a pretrained encoder (`init_from`), trains on (query, positive) text pairs with a
temperature-scaled InfoNCE loss in both directions, summed over the Matryoshka dimensions. Every
batch comes from a single language, so the in-batch negatives are same-language. GradCache lets
the batch (default 512 pairs) be much bigger than what fits in memory at once: all embeddings are
computed without gradients, the loss gradient is taken with respect to the embeddings, and each
chunk is then re-encoded with gradients and back-propagated with its cached gradient.

Resumes automatically from <RUNS_DIR>/<run>/latest.pt. Batches depend only on (seed, step).
Writes <RUNS_DIR>/<run>/{log.jsonl,latest.pt}; latest.pt holds the EmbeddingModel state dict
under "model", which src/eval/mmteb.py loads directly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from src.data.build_pairs import PAIRS_DIR
from src.data.paths import RUNS_DIR, TOKENIZED_DIR
from src.model import EmbeddingModel, build_block_masks, load_model_config

from .data import Prefetcher
from .mlm import load_config

DEVICE = "cuda"
CLS_ID, SEP_ID = 2, 3


# ------------------------------------------------------------------------------------- data


class PairData:
    """(query, positive) pairs per language, with a held-out validation slice."""

    def __init__(self, name: str, val_per_lang: int, seed: int) -> None:
        self.seed = seed
        self.train: dict[str, tuple[list[str], list[str]]] = {}
        self.val: dict[str, tuple[list[str], list[str]]] = {}
        for path in sorted((PAIRS_DIR / name).glob("*.parquet")):
            t = pq.read_table(path, columns=["query", "positive"]).to_pydict()
            q, p = t["query"], t["positive"]
            self.val[path.stem] = (q[:val_per_lang], p[:val_per_lang])
            self.train[path.stem] = (q[val_per_lang:], p[val_per_lang:])
        self.codes = list(self.train)
        if not self.codes:
            raise SystemExit(f"no pair files in {PAIRS_DIR / name}")

    def language(self, step: int) -> str:
        return self.codes[np.random.default_rng([self.seed, step, 5]).integers(len(self.codes))]

    def _order(self, code: str, epoch: int) -> np.ndarray:
        n = len(self.train[code][0])
        return np.random.default_rng([self.seed, self.codes.index(code), epoch, 9]).permutation(n)

    def batch(self, step: int, size: int, cursors: dict[str, int]) -> tuple[list[str], list[str]]:
        code = self.language(step)
        n = len(self.train[code][0])
        idx = []
        while len(idx) < size:
            epoch, within = divmod(cursors[code], n)
            take = min(size - len(idx), n - within)
            idx.extend(self._order(code, epoch)[within : within + take].tolist())
            cursors[code] += take
        q, p = self.train[code]
        return [q[i] for i in idx], [p[i] for i in idx]

    def cursors_at(self, step: int, size: int) -> dict[str, int]:
        cursors = dict.fromkeys(self.codes, 0)
        for s in range(step):
            cursors[self.language(s)] += size
        return cursors


def tokenize(tok: Tokenizer, texts: list[str], max_len: int) -> list[list[int]]:
    enc = tok.encode_batch(texts, add_special_tokens=False)
    return [[CLS_ID, *e.ids[: max_len - 2], SEP_ID] for e in enc]


def pack(seqs: list[list[int]], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-pack token lists into rows of `width`: (ids, segment_ids), segments in row order."""
    rows, current, used = [], [], 0
    for i, s in enumerate(seqs):
        if used + len(s) > width and current:
            rows.append(current)
            current, used = [], 0
        current.append(i)
        used += len(s)
    rows.append(current)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    seg = torch.full((len(rows), width), -1, dtype=torch.int32)
    order = {}
    for r, members in enumerate(rows):
        pos = 0
        for i in members:
            ids[r, pos : pos + len(seqs[i])] = torch.tensor(seqs[i])
            seg[r, pos : pos + len(seqs[i])] = len(order)
            order[i] = len(order)
            pos += len(seqs[i])
    assert [order[i] for i in range(len(seqs))] == list(range(len(seqs)))  # row-major order
    return ids, seg


def encode(model: EmbeddingModel, seqs: list[list[int]], width: int) -> torch.Tensor:
    ids, seg = pack(seqs, width)
    ids, seg = ids.to(DEVICE), seg.to(DEVICE)
    masks = build_block_masks(seg, model.cfg.local_window)
    with torch.autocast("cuda", torch.bfloat16):
        return model.encode_packed(ids, seg, len(seqs), dim=None, block_masks=masks)


# ------------------------------------------------------------------------------------- loss


def mrl_loss(q: torch.Tensor, p: torch.Tensor, dims: list[int], temperature: float):
    """InfoNCE in both directions, averaged over Matryoshka dims. Returns (loss, top-1 accuracy)."""
    total, acc = 0.0, 0.0
    target = torch.arange(q.size(0), device=q.device)
    for d in dims:
        qd, pd = F.normalize(q[:, :d], dim=-1), F.normalize(p[:, :d], dim=-1)
        logits = qd @ pd.T / temperature
        total = total + 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        if d == dims[0]:
            acc = (logits.argmax(dim=1) == target).float().mean()
    return total / len(dims), acc


def grad_cache_step(model, q_seqs, p_seqs, cfg) -> tuple[float, float]:
    """One optimizer step's forward/backward over the whole batch. Returns (loss, accuracy)."""
    chunk_q, chunk_p = cfg["batch"]["chunk_queries"], cfg["batch"]["chunk_passages"]
    wq, wp = cfg["data"]["query_width"], cfg["data"]["passage_width"]
    dims, temp = model.cfg.matryoshka_dims, cfg["loss"]["temperature"]

    with torch.no_grad():
        q = torch.cat(
            [encode(model, q_seqs[i : i + chunk_q], wq) for i in range(0, len(q_seqs), chunk_q)]
        )
        p = torch.cat(
            [encode(model, p_seqs[i : i + chunk_p], wp) for i in range(0, len(p_seqs), chunk_p)]
        )
    q.requires_grad_(True)
    p.requires_grad_(True)
    loss, acc = mrl_loss(q, p, dims, temp)
    loss.backward()
    for reps, seqs, chunk, width in ((q, q_seqs, chunk_q, wq), (p, p_seqs, chunk_p, wp)):
        for i in range(0, len(seqs), chunk):
            emb = encode(model, seqs[i : i + chunk], width)
            emb.backward(reps.grad[i : i + chunk])
    return loss.item(), float(acc)


@torch.no_grad()
def validate(model, data: PairData, tok, cfg) -> dict[str, float]:
    """Mean reciprocal rank of the positive among all validation positives of a language."""
    model.eval()
    out = {}
    for code, (qs, ps) in data.val.items():
        q = torch.cat(
            [
                encode(model, s, cfg["data"]["query_width"])
                for s in chunks(tokenize(tok, qs, cfg["data"]["max_query"]), 256)
            ]
        )
        p = torch.cat(
            [
                encode(model, s, cfg["data"]["passage_width"])
                for s in chunks(tokenize(tok, ps, cfg["data"]["max_passage"]), 64)
            ]
        )
        scores = q @ p.T
        rank = (scores > scores.diag()[:, None]).sum(dim=1) + 1
        out[code] = (1.0 / rank.float()).mean().item()
    out["macro"] = sum(out.values()) / len(out)
    model.train()
    return out


def chunks(items, n):
    return [items[i : i + n] for i in range(0, len(items), n)]


# ------------------------------------------------------------------------------------ main


def load_encoder(model: EmbeddingModel, init_from: str) -> None:
    run, _, ckpt = init_from.partition("/")
    state = torch.load(RUNS_DIR / run / ckpt, map_location="cpu", weights_only=True)["model"]
    enc = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total - step) / max(total - warmup, 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.manual_seed(cfg["seed"])

    model_cfg = load_model_config(cfg["model"])
    model = EmbeddingModel(model_cfg).to(DEVICE)
    if cfg.get("init_from"):
        load_encoder(model, cfg["init_from"])
    model.train()
    o = cfg["optim"]
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": o["weight_decay"]},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=o["lr"],
        betas=tuple(o["betas"]),
    )

    tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    data = PairData(cfg["data"]["name"], cfg["data"]["val_per_language"], cfg["seed"])
    batch_size = cfg["batch"]["pairs"]
    total_steps = math.ceil(cfg["schedule"]["total_pairs"] / batch_size)
    warmup = max(1, int(cfg["schedule"]["warmup_fraction"] * total_steps))

    start_step = 0
    latest = run_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")
    print(
        f"{model_cfg.summary()}\nsteps {total_steps} x {batch_size} pairs, languages {data.codes}, "
        f"matryoshka {model_cfg.matryoshka_dims}",
        flush=True,
    )

    cursors = data.cursors_at(start_step, batch_size)

    def batches():
        for step in range(start_step, total_steps):
            qs, ps = data.batch(step, batch_size, cursors)
            yield (
                step,
                tokenize(tok, qs, cfg["data"]["max_query"]),
                tokenize(tok, ps, cfg["data"]["max_passage"]),
            )

    log_path = run_dir / "log.jsonl"

    def log(rec: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    eval_every, save_every = cfg["eval"]["every_steps"], cfg["checkpoint"]["every_steps"]
    window_start, window_loss, window_acc, window_tokens, n = time.time(), 0.0, 0.0, 0, 0
    for step, q_seqs, p_seqs in Prefetcher(batches()):
        lr = lr_at(step, total_steps, o["lr"], warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss, acc = grad_cache_step(model, q_seqs, p_seqs, cfg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        window_loss, window_acc, n = window_loss + loss, window_acc + acc, n + 1
        window_tokens += sum(map(len, q_seqs)) + sum(map(len, p_seqs))
        done = step + 1
        if done % cfg["log_every_steps"] == 0 or done == total_steps:
            rec = {
                "type": "train",
                "step": done,
                "pairs": done * batch_size,
                "loss": window_loss / n,
                "acc": window_acc / n,
                "lr": lr,
                "tok_per_s": window_tokens / (time.time() - window_start),
            }
            log(rec)
            print(
                f"step {done}/{total_steps} loss {rec['loss']:.4f} acc {rec['acc']:.3f} "
                f"lr {lr:.2e} {rec['tok_per_s'] / 1e3:.0f}k tok/s",
                flush=True,
            )
            window_start, window_loss, window_acc, window_tokens, n = time.time(), 0.0, 0.0, 0, 0
        if done % eval_every == 0 or done == total_steps:
            res = validate(model, data, tok, cfg)
            log({"type": "eval", "step": done, "mrr": res})
            print(
                f"  validation MRR macro {res['macro']:.4f}  "
                + " ".join(f"{c} {v:.2f}" for c, v in res.items() if c != "macro"),
                flush=True,
            )
        if done % save_every == 0 or done == total_steps:
            tmp = Path(str(latest) + ".tmp")
            torch.save(
                {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": done}, tmp
            )
            tmp.rename(latest)


if __name__ == "__main__":
    main()
