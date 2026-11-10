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
from collections import Counter

from .paths import RAW_DIR, use_local_hf_cache

use_local_hf_cache()
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

PAIRS_DIR = RAW_DIR / "pairs"
WEBFAQ = {
    "en": "eng", "zh": "zho", "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "ru": "rus", "ja": "jpn", "ar": "ara", "hi": "hin", "ko": "kor", "it": "ita",
}  # fmt: skip
CHUNK_BYTES = 4_000_000


def sample_language(fs: HfFileSystem, code: str, target: int, max_per_origin: int, seed: int):
    path = f"datasets/PaDaS-Lab/webfaq/data/{WEBFAQ[code]}.jsonl"
    size = fs.info(path)["size"]
    rng = random.Random(f"{seed}-{code}")
    rows, seen, per_origin = [], set(), Counter()
    with fs.open(path, "rb") as f:
        tries = 0
        while len(rows) < target and tries < 400:
            tries += 1
            f.seek(rng.randrange(0, max(size - CHUNK_BYTES, 1)))
            block = f.read(CHUNK_BYTES).decode("utf-8", errors="ignore")
            for line in block.split("\n")[1:-1]:  # first and last lines are partial
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q, a, origin = r["question"].strip(), r["answer"].strip(), r["origin"]
                if len(q) < 8 or len(a) < 20 or q in seen or per_origin[origin] >= max_per_origin:
                    continue
                seen.add(q)
                per_origin[origin] += 1
                rows.append((q, a[:2000], code, origin))
                if len(rows) >= target:
                    break
    return rows


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

    fs = HfFileSystem()
    out_dir = PAIRS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in args.langs:
        out = out_dir / f"{code}.parquet"
        if out.exists():
            print(f"{code}: exists, skipping")
            continue
        rows = sample_language(fs, code, args.per_language, args.max_per_origin, args.seed)
        table = pa.table(
            dict(zip(("query", "positive", "lang", "origin"), zip(*rows, strict=True), strict=True))
        )
        pq.write_table(table, out)
        origins = len({r[3] for r in rows})
        print(f"{code}: {len(rows)} pairs from {origins} websites -> {out}", flush=True)


if __name__ == "__main__":
    main()
