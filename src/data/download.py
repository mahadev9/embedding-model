"""Download pretraining text into <repo>/data/. Safe to re-run: existing files are skipped.

# Small balanced text sample per language (for the tokenizer): streams ~300 MB of text each.
# Saved under data/sample/ (always local). Bulk shards go to $EMBEDDING_RAW_DIR if set.
python -m src.data.download sample --mb 300

# Whole parquet shards for the main corpus (each FineWeb-2 shard is ~4 GiB).
python -m src.data.download shards --source fineweb2-hq-plus --langs es fr --num-shards 2
python -m src.data.download shards --source fineweb-edu --num-shards 3
python -m src.data.download shards --source wikipedia --langs hi ko

# Remove everything this script downloaded or derived:  rm -rf data/  (and $EMBEDDING_RAW_DIR)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .paths import RAW_DIR, SAMPLE_DIR, use_local_hf_cache

use_local_hf_cache()

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download  # noqa: E402

from .languages import (  # noqa: E402
    BY_CODE,
    FINEWEB2_HQ_PLUS_REPO,
    FINEWEB2_REPO,
    FINEWEB_EDU_REPO,
    LANGUAGES,
    WIKIPEDIA_REPO,
    WIKIPEDIA_SNAPSHOT,
)

FINEWEB_EDU_SUBSET = "sample/10BT"  # 10B-token sample; plenty for English at our budget


def parquet_files(api: HfApi, repo: str, path: str) -> list[str]:
    """Sorted parquet paths under a dataset repo folder."""
    entries = api.list_repo_tree(repo, repo_type="dataset", path_in_repo=path, recursive=True)
    return sorted(e.path for e in entries if e.path.endswith(".parquet"))


def source_folder(source: str, code: str) -> tuple[str, str]:
    """(repo, folder inside repo) holding a language's text for a given source."""
    lang = BY_CODE[code]
    if source == "fineweb2":
        if lang.fineweb2 is None:
            raise SystemExit(f"{code} is not in FineWeb-2; use --source fineweb-edu or wikipedia")
        return FINEWEB2_REPO, f"data/{lang.fineweb2}/train"
    if source == "fineweb2-hq-plus":
        if lang.fineweb2 is None:
            raise SystemExit(f"{code} is not in FineWeb-2-HQ-Plus; use --source fineweb-edu")
        return FINEWEB2_HQ_PLUS_REPO, lang.fineweb2  # layout: <config>/NNN_NNNNN.parquet
    if source == "fineweb-edu":
        if code != "en":
            raise SystemExit("FineWeb-Edu is English only")
        return FINEWEB_EDU_REPO, FINEWEB_EDU_SUBSET
    if source == "wikipedia":
        return WIKIPEDIA_REPO, f"{WIKIPEDIA_SNAPSHOT}.{lang.wikipedia}"
    raise SystemExit(f"unknown source {source}")


def default_source(code: str) -> str:
    return "fineweb-edu" if code == "en" else "fineweb2-hq-plus"


def resolve_source(code: str, requested: str | None) -> str:
    """The requested source, except English never uses the (non-English) FineWeb-2 variants."""
    if requested is None or (code == "en" and requested in ("fineweb2", "fineweb2-hq-plus")):
        return default_source(code)
    return requested


SOURCES = ["fineweb2", "fineweb2-hq-plus", "fineweb-edu", "wikipedia"]


# ----------------------------------------------------------------------------- sample


def stream_text_sample(
    shard_paths: list[str], repo: str, budget_bytes: int, out_path: Path
) -> tuple[int, int]:
    """Stream rows from remote parquet shards, in order, until `budget_bytes` of text is read.

    Only the parquet pages that are read are transferred, not whole multi-GiB shards. Later
    shards are used only if earlier ones run out (some are small). Returns (text bytes, shards).
    """
    fs = HfFileSystem()
    texts: list[str] = []
    total = 0
    used = 0
    for path_in_repo in shard_paths:
        used += 1
        with fs.open(f"datasets/{repo}/{path_in_repo}", "rb") as f:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    texts.append(text)
                    total += len(text.encode("utf-8"))
                if total >= budget_bytes:
                    break
        if total >= budget_bytes:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table({"text": texts}), tmp, compression="zstd")
    tmp.rename(out_path)
    return total, used


def cmd_sample(args: argparse.Namespace) -> None:
    api = HfApi()
    rng = random.Random(args.seed)
    out_dir = SAMPLE_DIR
    for code in args.langs:
        out_path = out_dir / f"{code}.parquet"
        if out_path.exists():
            print(f"{code}: exists, skipping ({out_path})")
            continue
        repo, folder = source_folder(resolve_source(code, args.source), code)
        files = parquet_files(api, repo, folder)
        # Random shard order avoids always taking the oldest crawl snapshot.
        rng.shuffle(files)
        print(f"{code}: streaming {args.mb} MB from {repo}/{folder} ...", flush=True)
        total, used = stream_text_sample(files, repo, args.mb * 10**6, out_path)
        print(
            f"{code}: wrote {total / 1e6:.0f} MB text from {used} shard(s) -> "
            f"{out_path.stat().st_size / 1e6:.0f} MB"
        )


# ----------------------------------------------------------------------------- shards


def cmd_shards(args: argparse.Namespace) -> None:
    api = HfApi()
    rng = random.Random(args.seed)
    for code in args.langs:
        source = resolve_source(code, args.source)
        repo, folder = source_folder(source, code)
        files = parquet_files(api, repo, folder)
        chosen = files[: args.num_shards] if not args.random else rng.sample(files, args.num_shards)
        local_dir = RAW_DIR / source / code
        print(f"{code}: {len(chosen)} of {len(files)} shards from {repo}/{folder} -> {local_dir}")
        for path in chosen:
            local = hf_hub_download(repo, path, repo_type="dataset", local_dir=local_dir)
            print(f"  {path} -> {local}", flush=True)


# ------------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    all_codes = [lang.code for lang in LANGUAGES]

    sample = sub.add_parser("sample", help="stream a small text sample per language")
    sample.add_argument("--langs", nargs="+", default=all_codes, choices=all_codes)
    sample.add_argument(
        "--source", choices=SOURCES, help="source for non-English languages (English: FineWeb-Edu)"
    )
    sample.add_argument("--mb", type=int, default=300, help="text megabytes per language")
    sample.add_argument("--seed", type=int, default=0)
    sample.set_defaults(func=cmd_sample)

    shards = sub.add_parser("shards", help="download whole parquet shards")
    shards.add_argument("--source", choices=SOURCES)
    shards.add_argument("--langs", nargs="+", default=None, choices=all_codes)
    shards.add_argument("--num-shards", type=int, default=1)
    shards.add_argument("--random", action="store_true", help="pick shards at random")
    shards.add_argument("--seed", type=int, default=0)
    shards.set_defaults(func=cmd_shards)

    args = parser.parse_args()
    if args.command == "shards" and args.langs is None:
        args.langs = ["en"] if args.source == "fineweb-edu" else all_codes
    args.func(args)


if __name__ == "__main__":
    main()
