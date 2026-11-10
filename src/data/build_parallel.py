"""Build translation-pair training data (English <-> one of our other languages) from WikiMatrix.

    python -m src.data.build_parallel --per-pair 40000

Source: sentence-transformers/parallel-sentences-wikimatrix (pairs mined from Wikipedia, CC BY-SA).
For each of the 11 English-X pairs it keeps up to --per-pair random pairs after dropping identical
sides, very short or very long sentences and duplicates, and writes
<PAIRS_DIR>/parallel/par_en-<code>.parquet with columns query (English), positive (other language),
lang, origin. These files are mixed into contrastive training (see data.parallel_fraction in
configs/train/contrastive_50m.yaml). Tatoeba, which is an evaluation task, is not used.
"""

from __future__ import annotations

import argparse
import random

from .build_pairs import PAIRS_DIR
from .paths import use_local_hf_cache

use_local_hf_cache()
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "sentence-transformers/parallel-sentences-wikimatrix"
CODES = ["ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "zh"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--per-pair", type=int, default=40_000)
    parser.add_argument("--out-name", default="parallel")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = PAIRS_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in CODES:
        out = out_dir / f"par_en-{code}.parquet"
        if out.exists():
            print(f"en-{code}: exists, skipping")
            continue
        path = hf_hub_download(REPO, f"en-{code}/train-00000-of-00001.parquet", repo_type="dataset")
        table = pq.read_table(path).to_pydict()
        rng = random.Random(f"{args.seed}-{code}")
        order = list(range(len(table["english"])))
        rng.shuffle(order)
        rows, seen = [], set()
        for i in order:
            en, other = table["english"][i].strip(), table["non_english"][i].strip()
            if en == other or en in seen:
                continue
            if not (20 <= len(en) <= 400) or not (4 <= len(other) <= 400):
                continue
            seen.add(en)
            rows.append((en, other))
            if len(rows) >= args.per_pair:
                break
        cols = {
            "query": [r[0] for r in rows],
            "positive": [r[1] for r in rows],
            "lang": [f"en-{code}"] * len(rows),
            "origin": ["wikimatrix"] * len(rows),
        }
        pq.write_table(pa.table(cols), out)
        print(f"en-{code}: {len(rows)} pairs (of {len(order)}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
