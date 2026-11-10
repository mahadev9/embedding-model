"""Build text-pair training data for contrastive fine-tuning (question -> answer pairs).

    python -m src.data.build_pairs --per-language 100000          # WebFAQ, our 12 languages
    python -m src.data.build_pairs --per-language 2000 --langs de --out-name smoke

Source: PaDaS-Lab/webfaq (CC BY-4.0), FAQ question/answer pairs from web pages. The files are huge
(English alone is 27 GB) and sorted by website, so this streams random byte ranges of each language
file instead of downloading it, keeps at most `--max-per-origin` pairs per website, drops very
short texts and duplicate questions, and writes <PAIRS_DIR>/<out-name>/<lang>.parquet with columns
query, positive, lang, origin. WebFAQ is not among the evaluation tasks in src/eval/mmteb.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .paths import RAW_DIR, use_local_hf_cache

use_local_hf_cache()
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem, get_session, hf_hub_url  # noqa: E402

PAIRS_DIR = RAW_DIR / "pairs"
WEBFAQ = {
    "en": "eng", "zh": "zho", "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "ru": "rus", "ja": "jpn", "ar": "ara", "hi": "hin", "ko": "kor", "it": "ita",
}  # fmt: skip
CHUNK_BYTES = 4_000_000


def fetch_chunk(url: str, offset: int, session) -> list[dict]:
    """Rows from one random byte range of a jsonl file (first and last lines are partial)."""
    for attempt in range(5):
        try:
            r = session.get(
                url, headers={"Range": f"bytes={offset}-{offset + CHUNK_BYTES - 1}"}, timeout=60
            )
            r.raise_for_status()
            break
        except Exception:
            if attempt == 4:
                return []
            time.sleep(2**attempt)
    rows = []
    for line in r.content.decode("utf-8", errors="ignore").split("\n")[1:-1]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sample_language(code: str, target: int, max_per_origin: int, seed: int, workers: int = 8):
    url = hf_hub_url("PaDaS-Lab/webfaq", f"data/{WEBFAQ[code]}.jsonl", repo_type="dataset")
    size = HfFileSystem().info(f"datasets/PaDaS-Lab/webfaq/data/{WEBFAQ[code]}.jsonl")["size"]
    rng = random.Random(f"{seed}-{code}")
    session = get_session()
    rows, seen, per_origin = [], set(), Counter()
    chunks_read = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(rows) < target and chunks_read < 600:
            offsets = [rng.randrange(0, max(size - CHUNK_BYTES, 1)) for _ in range(workers)]
            for batch in pool.map(lambda o: fetch_chunk(url, o, session), offsets):
                chunks_read += 1
                for r in batch:
                    q, a, origin = r["question"].strip(), r["answer"].strip(), r["origin"]
                    if (
                        len(q) < 8
                        or len(a) < 20
                        or q in seen
                        or per_origin[origin] >= max_per_origin
                    ):
                        continue
                    seen.add(q)
                    per_origin[origin] += 1
                    rows.append((q, a[:2000], code, origin))
            print(f"  {code}: {len(rows)}/{target} pairs after {chunks_read} chunks", flush=True)
    return rows[:target]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--per-language", type=int, default=100_000)
    parser.add_argument("--langs", nargs="+", default=list(WEBFAQ))
    parser.add_argument("--max-per-origin", type=int, default=20)
    parser.add_argument("--out-name", default="webfaq")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = PAIRS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in args.langs:
        out = out_dir / f"{code}.parquet"
        if out.exists():
            print(f"{code}: exists, skipping")
            continue
        rows = sample_language(code, args.per_language, args.max_per_origin, args.seed)
        table = pa.table(
            dict(zip(("query", "positive", "lang", "origin"), zip(*rows, strict=True), strict=True))
        )
        pq.write_table(table, out)
        origins = len({r[3] for r in rows})
        print(f"{code}: {len(rows)} pairs from {origins} websites -> {out}", flush=True)


if __name__ == "__main__":
    main()
