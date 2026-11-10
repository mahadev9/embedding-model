"""Fine-tuning probes: fine-tune an encoder on English, test zero-shot on other languages.

    python -m src.eval.finetune --models ours:decay-10.00B/latest.pt hf:jhu-clsp/mmBERT-small \\
        --tasks xnli paws ner

Tasks (all train on English only, then score every language, the XTREME protocol):
  xnli  natural-language inference, 3 classes, accuracy (ar de en es fr hi ru zh)
  paws  paraphrase identification, 2 classes, accuracy (de en es fr ja ko zh)
  ner   named entities (WikiANN), entity-level F1 (all 12 of our languages)

Every model gets the same recipe: a linear head on the [CLS] vector (token head for ner), AdamW
with 6% warmup and linear decay, batch 32, bf16. For each model the learning rate is picked from a
small grid by English validation score, and that run's test scores are reported. One seed: gaps
below about 0.5 points (accuracy) are within noise. Results go to <RUNS_DIR>/eval/finetune_*.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from bisect import bisect_right

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch import nn

from src.data.paths import RUNS_DIR, TOKENIZED_DIR, use_local_hf_cache
from src.model import PAD_DOC_ID, MLMModel, build_block_masks, load_model_config

use_local_hf_cache()
from huggingface_hub import hf_hub_download  # noqa: E402

DEVICE = "cuda"
CLS_ID, SEP_ID = 2, 3
EVAL_DIR = RUNS_DIR / "eval"

TASKS = {
    "xnli": dict(
        repo="facebook/xnli",
        langs=["en", "ar", "de", "es", "fr", "hi", "ru", "zh"],
        epochs=2,
        max_train=100_000,
        max_len=128,
    ),
    "paws": dict(
        repo="google-research-datasets/paws-x",
        langs=["en", "de", "es", "fr", "ja", "ko", "zh"],
        epochs=3,
        max_train=None,
        max_len=160,
    ),
    "ner": dict(
        repo="unimelb-nlp/wikiann",
        langs=["en", "zh", "es", "fr", "de", "pt", "ru", "ja", "ar", "hi", "ko", "it"],
        epochs=4,
        max_train=None,
        max_len=128,
    ),
}
DEFAULT_LRS = {
    "ours": [1.5e-4, 4e-4],
    "mmBERT-small": [5e-5, 1.5e-4],
    "mmBERT-base": [3e-5, 1e-4],
    "xlm-roberta-base": [2e-5, 5e-5],
}
NUM_TAGS = 7  # O, B/I-PER, B/I-ORG, B/I-LOC


# ------------------------------------------------------------------------------------- data


def read_split(task: str, lang: str, split: str) -> list[dict]:
    repo = TASKS[task]["repo"]
    path = hf_hub_download(repo, f"{lang}/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


def load_task(task: str, lang: str, split: str, limit: int | None = None, seed: int = 0):
    rows = read_split(task, lang, split)
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    if task == "xnli":
        return [((r["premise"], r["hypothesis"]), r["label"]) for r in rows]
    if task == "paws":
        return [((r["sentence1"], r["sentence2"]), r["label"]) for r in rows]
    return [(r["tokens"], r["ner_tags"]) for r in rows]


# -------------------------------------------------------------------------------- tokenizers


def words_to_text(words: list[str]) -> tuple[str, list[int]]:
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    return " ".join(words), starts


def token_words(text: str, starts: list[int], offsets) -> list[int]:
    """Word index of each token (-1 for special/empty tokens)."""
    out = []
    for a, b in offsets:
        if b <= a or not text[a:b].strip():
            out.append(-1)
        else:
            out.append(bisect_right(starts, b - 1) - 1)
    return out


class OursTok:
    pad_id = 0

    def __init__(self) -> None:
        self.tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()

    def pair(self, a: str, b: str, max_len: int) -> list[int]:
        ia = self.tok.encode(a, add_special_tokens=False).ids
        ib = self.tok.encode(b, add_special_tokens=False).ids
        while len(ia) + len(ib) > max_len - 3:
            (ia if len(ia) > len(ib) else ib).pop()
        return [CLS_ID, *ia, SEP_ID, *ib, SEP_ID]

    def words(self, words: list[str], max_len: int) -> tuple[list[int], list[int]]:
        text, starts = words_to_text(words)
        e = self.tok.encode(text, add_special_tokens=False)
        ids, offsets = e.ids[: max_len - 2], e.offsets[: max_len - 2]
        return [CLS_ID, *ids, SEP_ID], [-1, *token_words(text, starts, offsets), -1]


class HFTok:
    def __init__(self, repo: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo)
        self.pad_id = self.tok.pad_token_id or 0

    def pair(self, a: str, b: str, max_len: int) -> list[int]:
        return self.tok(a, b, truncation=True, max_length=max_len)["input_ids"]

    def words(self, words: list[str], max_len: int) -> tuple[list[int], list[int]]:
        text, starts = words_to_text(words)
        e = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
        return e["input_ids"], token_words(text, starts, [tuple(o) for o in e["offset_mapping"]])


# ---------------------------------------------------------------------------------- backbones


class OursBackbone(nn.Module):
    pad_multiple = 128  # FlexAttention works in 128-token blocks

    def __init__(self, spec: str) -> None:
        super().__init__()
        run, _, ckpt = spec.partition("/")
        run_dir = RUNS_DIR / run
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        mlm = MLMModel(load_model_config(cfg["model"]))
        mlm.load_state_dict(
            torch.load(run_dir / ckpt, map_location="cpu", weights_only=True)["model"]
        )
        self.encoder = mlm.encoder
        self.window = mlm.cfg.local_window
        self.hidden_size = mlm.cfg.hidden_size

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        doc = torch.where(attn.bool(), 0, PAD_DOC_ID).to(torch.int32)
        return self.encoder(ids, doc, None, build_block_masks(doc, self.window))


class HFBackbone(nn.Module):
    pad_multiple = 1

    def __init__(self, repo: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids, attention_mask=attn).last_hidden_state


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int, token_level: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.token_level = token_level
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.hidden_size, num_labels)

    def forward(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(ids, attn)
        return self.head(self.dropout(hidden if self.token_level else hidden[:, 0]))


# ------------------------------------------------------------------------------- batching


def collate(seqs: list[list[int]], pad_id: int, multiple: int):
    width = max(len(s) for s in seqs)
    width = -(-width // multiple) * multiple
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), width, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s)
        attn[r, : len(s)] = 1
    return ids, attn


def encode_examples(task: str, tok, examples, max_len: int):
    """(token ids, label) for pair tasks; (token ids, per-token word index, tags) for ner."""
    if task == "ner":
        out = []
        for words, tags in examples:
            ids, widx = tok.words(words, max_len)
            out.append((ids, widx, tags))
        return out
    return [(tok.pair(a, b, max_len), label) for (a, b), label in examples]


def make_batch(task: str, items, pad_id: int, multiple: int):
    ids, attn = collate([it[0] for it in items], pad_id, multiple)
    if task != "ner":
        return ids, attn, torch.tensor([it[1] for it in items])
    labels = torch.full(ids.shape, -100, dtype=torch.long)
    for r, (_seq, widx, tags) in enumerate(items):
        prev = -1
        for t, w in enumerate(widx):
            if w >= 0 and w != prev:
                labels[r, t] = tags[w]
            if w >= 0:
                prev = w
    return ids, attn, labels


# ------------------------------------------------------------------------------------ metrics


def entities(tags: list[int]) -> set[tuple[int, int, int]]:
    """(type, start, end) spans from WikiANN tags (1/2 PER, 3/4 ORG, 5/6 LOC; odd = B, even = I)."""
    spans, start, kind = set(), None, None
    for i, t in enumerate([*tags, 0]):
        begin = t in (1, 3, 5)
        inside = t in (2, 4, 6)
        t_kind = (t + 1) // 2 if t else None
        if start is not None and not (inside and t_kind == kind):
            spans.add((kind, start, i))
            start = None
        if begin or (inside and start is None):
            start, kind = i, t_kind
    return spans


@torch.no_grad()
def evaluate(model, task: str, items, pad_id: int, multiple: int, batch_size: int = 64) -> float:
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    preds = {}
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids, attn, labels = make_batch(task, [items[i] for i in idx], pad_id, multiple)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(ids.to(DEVICE), attn.to(DEVICE))
        guess = logits.argmax(-1).cpu()
        for r, i in enumerate(idx):
            preds[i] = guess[r]
    if task != "ner":
        correct = sum(int(preds[i]) == items[i][1] for i in range(len(items)))
        return 100.0 * correct / len(items)
    tp = n_pred = n_gold = 0
    for i, (_ids, widx, tags) in enumerate(items):
        pred_tags, prev = [0] * len(tags), -1
        for t, w in enumerate(widx):
            if w >= 0 and w != prev:
                pred_tags[w] = int(preds[i][t])
            if w >= 0:
                prev = w
        gold, pred = entities(tags), entities(pred_tags)
        tp += len(gold & pred)
        n_pred += len(pred)
        n_gold += len(gold)
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return 100.0 * 2 * p * r / max(p + r, 1e-9)


# ------------------------------------------------------------------------------------ training


def run_task(spec: str, task: str, lrs: list[float], args) -> dict:
    cfg = TASKS[task]
    kind, _, rest = spec.partition(":")
    tok = OursTok() if kind == "ours" else HFTok(rest)
    multiple = OursBackbone.pad_multiple if kind == "ours" else 1
    max_len = cfg["max_len"]
    limit = args.max_train or cfg["max_train"]

    train = encode_examples(task, tok, load_task(task, "en", "train", limit), max_len)
    dev = encode_examples(task, tok, load_task(task, "en", "validation"), max_len)
    tests = {
        lang: encode_examples(task, tok, load_task(task, lang, "test", args.max_test), max_len)
        for lang in cfg["langs"]
    }
    num_labels = NUM_TAGS if task == "ner" else 3 if task == "xnli" else 2

    best = None
    for lr in lrs:
        torch.manual_seed(0)
        backbone = OursBackbone(rest) if kind == "ours" else HFBackbone(rest)
        model = Classifier(backbone, num_labels, task == "ner").to(DEVICE)
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
        rest_params = [
            p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)
        ]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.01}, {"params": rest_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.98),
        )
        steps = cfg["epochs"] * (len(train) // args.batch_size)
        warm = max(1, int(0.06 * steps))

        def schedule(s: int, steps: int = steps, warm: int = warm) -> float:
            return min((s + 1) / warm, max(0.0, (steps - s) / max(steps - warm, 1)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        rng = random.Random(0)
        start, step = time.time(), 0
        for _epoch in range(cfg["epochs"]):
            model.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            for s in range(0, len(order) - args.batch_size + 1, args.batch_size):
                ids, attn, labels = make_batch(
                    task, [train[i] for i in order[s : s + args.batch_size]], tok.pad_id, multiple
                )
                with torch.autocast("cuda", torch.bfloat16):
                    logits = model(ids.to(DEVICE), attn.to(DEVICE))
                loss = F.cross_entropy(
                    logits.float().reshape(-1, num_labels),
                    labels.to(DEVICE).reshape(-1),
                    ignore_index=-100,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 500 == 0:
                    print(
                        f"  {task} lr {lr:g} step {step}/{steps} loss {loss.item():.3f} "
                        f"({time.time() - start:.0f}s)",
                        flush=True,
                    )
        dev_score = evaluate(model, task, dev, tok.pad_id, multiple)
        scores = {
            lang: evaluate(model, task, items, tok.pad_id, multiple)
            for lang, items in tests.items()
        }
        print(
            f"{spec} {task} lr {lr:g}: English dev {dev_score:.2f}, test "
            + " ".join(f"{k} {v:.1f}" for k, v in scores.items())
            + f"  ({time.time() - start:.0f}s)",
            flush=True,
        )
        if best is None or dev_score > best["dev"]:
            best = {"lr": lr, "dev": dev_score, "test": scores}
        del model, backbone, opt
        torch.cuda.empty_cache()
    return best


def default_lrs(spec: str) -> list[float]:
    kind, _, rest = spec.partition(":")
    if kind == "ours":
        return DEFAULT_LRS["ours"]
    return DEFAULT_LRS[rest.split("/")[-1]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True, help="ours:RUN/CKPT or hf:REPO")
    parser.add_argument("--tasks", nargs="+", default=["xnli", "paws", "ner"], choices=list(TASKS))
    parser.add_argument("--lrs", nargs="+", type=float, help="override the learning-rate grid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train", type=int, help="cap English training examples")
    parser.add_argument("--max-test", type=int, help="cap test examples per language")
    parser.add_argument("--out", default="finetune.json")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / args.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec in args.models:
        for task in args.tasks:
            key = f"{spec}|{task}"
            if key in results:
                print(f"skip {key} (done)")
                continue
            results[key] = run_task(spec, task, args.lrs or default_lrs(spec), args)
            out_path.write_text(json.dumps(results, indent=1))

    print("\nscores (test, English-trained):")
    for task in args.tasks:
        langs = TASKS[task]["langs"]
        print(f"\n{task}: {' '.join(f'{lang:>5s}' for lang in langs)}   avg  zero-shot avg   lr")
        for spec in args.models:
            r = results[f"{spec}|{task}"]
            vals = [r["test"][lang] for lang in langs]
            zs = sum(vals[1:]) / (len(vals) - 1)
            print(
                f"{spec[:34]:34s} {' '.join(f'{v:5.1f}' for v in vals)}  "
                f"{sum(vals) / len(vals):5.1f}  {zs:5.1f}  {r['lr']:g}"
            )


if __name__ == "__main__":
    main()
