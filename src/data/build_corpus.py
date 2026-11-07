"""Build the tokenized pretraining corpus: stream shards, tokenize, write flat token files.

    python -m src.data.build_corpus --total-tokens 10e9 --plan-only   # show the language plan
    python -m src.data.build_corpus --total-tokens 10e9               # download + tokenize

Per language it writes <TOKENIZED_DIR>/<lang>/train.bin and val.bin: little-endian uint32 token ids
(the 96k vocabulary does not fit in 16 bits). Every document is stored as [CLS] tokens [SEP], so
document boundaries are recoverable from the CLS ids. One document in 500 goes to val.bin.

Language mix: q_i is proportional to available_tokens_i ** alpha (alpha 0.5 by default), capped at
half of what a language has. Raw shards are deleted after tokenizing (use --keep-raw to keep them).
The run is resumable: rerun the same command and it continues from the last finished shard.
To extend a finished corpus, move plan.json aside and rerun with a larger --total-tokens: the shard
order is fixed by a seed, so new data is appended. Pin experiments to a size with --snapshot NAME.

    export EMBEDDING_RAW_DIR=/mnt/hdd-2tb/embedding-model/raw
    export EMBEDDING_TOKENIZED_DIR=/mnt/hdd-2tb/embedding-model/tokenized
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .paths import RAW_DIR, TOKENIZED_DIR, use_local_hf_cache

use_local_hf_cache()

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from src.tokenizer.common import candidate_path  # noqa: E402

from .download import default_source, source_folder  # noqa: E402
from .languages import LANGUAGES  # noqa: E402

CLS_ID, SEP_ID = 2, 3  # see SPECIAL_TOKENS in src/tokenizer/common.py
VAL_EVERY = 500  # every 500th document of a language goes to the validation file
MAX_DOC_CHARS = 200_000  # guard against pathological documents
BATCH_DOCS = 2048
assert array.array("I").itemsize == 4


def shards_with_sizes(api: HfApi, code: str) -> tuple[str, list[tuple[str, int]]]:
    """(repo, [(path, bytes)]) of a language's parquet shards, in a seeded random order."""
    repo, folder = source_folder(default_source(code), code)
    entries = api.list_repo_tree(repo, repo_type="dataset", path_in_repo=folder, recursive=True)
    shards = sorted((e.path, e.size) for e in entries if e.path.endswith(".parquet"))
    random.Random(f"shards-{code}").shuffle(shards)
    return repo, shards


def download(repo: str, path: str, code: str) -> Path:
    local_dir = RAW_DIR / "corpus" / code
    return Path(hf_hub_download(repo, path, repo_type="dataset", local_dir=local_dir))


def tokenize_docs(tokenizer: Tokenizer, docs: list[str]) -> list[list[int]]:
    docs = [d[:MAX_DOC_CHARS] for d in docs]
    return [
        [CLS_ID, *e.ids, SEP_ID] for e in tokenizer.encode_batch(docs, add_special_tokens=False)
    ]


# ------------------------------------------------------------------------------------ planning


def estimate_available(tokenizer: Tokenizer, api: HfApi) -> dict[str, dict]:
    """Estimate each language's total tokens from one downloaded shard and the total shard bytes."""
    info = {}
    for lang in LANGUAGES:
        repo, shards = shards_with_sizes(api, lang.code)
        first_path, first_bytes = shards[0]
        local = download(repo, first_path, lang.code)
        pf = pq.ParquetFile(local)
        docs = next(pf.iter_batches(batch_size=2000, columns=["text"])).column("text").to_pylist()
        tokens = sum(len(x) for x in tokenize_docs(tokenizer, docs))
        tokens_per_byte = tokens / len(docs) * pf.metadata.num_rows / first_bytes
        total_bytes = sum(size for _, size in shards)
        info[lang.code] = {
            "repo": repo,
            "shards": len(shards),
            "total_bytes": total_bytes,
            "tokens_per_parquet_byte": tokens_per_byte,
            "available_tokens": tokens_per_byte * total_bytes,
        }
        print(
            f"{lang.code}: {len(shards)} shards, {total_bytes / 2**30:.1f} GiB, "
            f"~{info[lang.code]['available_tokens'] / 1e9:.1f}B tokens available",
            flush=True,
        )
    return info


def allocate(available: dict[str, float], total: float, alpha: float, cap: float = 0.5):
    """Quotas q_i ~ available_i ** alpha summing to `total`, each capped at cap * available_i."""
    quota: dict[str, float] = {}
    free = set(available)
    remaining = total
    while free:
        weights = {c: available[c] ** alpha for c in free}
        norm = sum(weights.values())
        capped = {c for c in free if remaining * weights[c] / norm > cap * available[c]}
        if not capped:
            quota.update({c: remaining * weights[c] / norm for c in free})
            break
        for c in capped:
            quota[c] = cap * available[c]
            remaining -= quota[c]
            free.discard(c)
    return quota


# ------------------------------------------------------------------------------------ building


def build_language(tokenizer: Tokenizer, api: HfApi, code: str, quota: int, keep_raw: bool) -> dict:
    out_dir = TOKENIZED_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    state = {"shards_done": [], "train_tokens": 0, "val_tokens": 0, "docs": 0}
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
    # A crash can leave bytes past the last recorded shard; cut the files back to the recorded size.
    for name, key in (("train.bin", "train_tokens"), ("val.bin", "val_tokens")):
        path = out_dir / name
        path.touch()
        with open(path, "r+b") as f:
            f.truncate(state[key] * 4)

    repo, shards = shards_with_sizes(api, code)
    todo = [p for p, _ in shards if p not in set(state["shards_done"])]
    train_goal = quota
    start = time.time()
    with (
        open(out_dir / "train.bin", "ab") as train_f,
        open(out_dir / "val.bin", "ab") as val_f,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        pending = pool.submit(download, repo, todo[0], code) if todo else None
        for i, shard in enumerate(todo):
            if state["train_tokens"] >= train_goal:
                break
            local = pending.result()
            pending = pool.submit(download, repo, todo[i + 1], code) if i + 1 < len(todo) else None
            for batch in pq.ParquetFile(local).iter_batches(
                batch_size=BATCH_DOCS, columns=["text"]
            ):
                docs = tokenize_docs(tokenizer, batch.column("text").to_pylist())
                train_ids, val_ids = array.array("I"), array.array("I")
                for ids in docs:
                    target = val_ids if state["docs"] % VAL_EVERY == 0 else train_ids
                    target.extend(ids)
                    state["docs"] += 1
                train_ids.tofile(train_f)
                val_ids.tofile(val_f)
                state["train_tokens"] += len(train_ids)
                state["val_tokens"] += len(val_ids)
                if state["train_tokens"] >= train_goal:
                    break
            train_f.flush()
            val_f.flush()
            state["shards_done"].append(shard)
            progress_path.write_text(json.dumps(state))
            if not keep_raw:
                local.unlink(missing_ok=True)
            rate = state["train_tokens"] / max(time.time() - start, 1)
            print(
                f"{code}: {state['train_tokens'] / 1e6:8.0f}M / {train_goal / 1e6:.0f}M tokens "
                f"({len(state['shards_done'])} shards, ~{rate / 1e6:.2f}M tok/s this run)",
                flush=True,
            )
    shutil.rmtree(RAW_DIR / "corpus" / code, ignore_errors=True) if not keep_raw else None
    return state


def write_snapshot(name: str) -> Path:
    """Record each language's current train/val token counts as snapshots/<name>.json.

    The corpus files only ever grow (append-only), so a snapshot pins "the first N tokens of each
    file". Experiments that read only up to the snapshot's counts keep seeing exactly the same data
    even after the corpus is extended.
    """
    snapshot = {}
    for lang in LANGUAGES:
        state = json.loads((TOKENIZED_DIR / lang.code / "progress.json").read_text())
        snapshot[lang.code] = {
            "train_tokens": state["train_tokens"],
            "val_tokens": state["val_tokens"],
        }
    snapshot["total_train_tokens"] = sum(v["train_tokens"] for v in snapshot.values())
    out = TOKENIZED_DIR / "snapshots" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} already exists; snapshots are never overwritten")
    out.write_text(json.dumps(snapshot, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--total-tokens", type=float, default=10e9)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--vocab-size", type=int, default=96000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--snapshot", metavar="NAME", help="record current token counts and exit")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--langs", nargs="+", default=[lang.code for lang in LANGUAGES])
    args = parser.parse_args()

    if args.snapshot:
        print(f"wrote {write_snapshot(args.snapshot)}")
        return

    tok_path = candidate_path(args.vocab_size)
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    api = HfApi()

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = TOKENIZED_DIR / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(f"using existing plan {plan_path}")
    else:
        info = estimate_available(tokenizer, api)
        quota = allocate(
            {c: v["available_tokens"] for c, v in info.items()}, args.total_tokens, args.alpha
        )
        plan = {"total_tokens": args.total_tokens, "alpha": args.alpha, "languages": {}}
        for code, v in info.items():
            plan["languages"][code] = {**v, "quota_tokens": int(quota[code])}
        plan_path.write_text(json.dumps(plan, indent=2))

    print(f"\n{'lang':4s} {'available':>11s} {'quota':>9s} {'share':>6s}")
    total_quota = sum(v["quota_tokens"] for v in plan["languages"].values())
    for code, v in plan["languages"].items():
        print(
            f"{code:4s} {v['available_tokens'] / 1e9:10.1f}B {v['quota_tokens'] / 1e9:8.2f}B "
            f"{v['quota_tokens'] / total_quota * 100:5.1f}%"
        )
    if args.plan_only:
        return

    manifest = {
        "tokenizer": str(tok_path),
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
        "vocab_size": args.vocab_size,
        "cls_id": CLS_ID,
        "sep_id": SEP_ID,
        "dtype": "uint32",
    }
    shutil.copy(tok_path, TOKENIZED_DIR / "tokenizer.json")
    (TOKENIZED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {}
    for code in args.langs:
        quota = plan["languages"][code]["quota_tokens"]
        results[code] = build_language(tokenizer, api, code, quota, args.keep_raw)
    print("\ndone:", {c: f"{r['train_tokens'] / 1e9:.2f}B" for c, r in results.items()})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
