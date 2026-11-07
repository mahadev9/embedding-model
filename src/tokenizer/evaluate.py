"""Compare candidate tokenizers and public baselines on held-out text from the 12 languages.

    python -m src.tokenizer.evaluate

Metrics per language, on a fixed 5% holdout of the tokenizer sample (never seen in training):
    chars/token and bytes/token (higher = better compression), tokens/word for space-delimited
    languages (lower = better). Character and byte counts use the raw text, so tokenizers that
    normalize text internally are measured fairly. Writes data/tokenizer/report.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from src.data.paths import use_local_hf_cache

use_local_hf_cache()

from tokenizers import Tokenizer  # noqa: E402

from .common import TOKENIZER_DIR, all_codes, candidate_path, sample_texts  # noqa: E402

NO_SPACES = {"zh", "ja"}  # tokens/word is not meaningful without whitespace word boundaries

BASELINES = {
    "xlm-roberta-base (250k)": "FacebookAI/xlm-roberta-base",
    "mmBERT / Gemma-2 (256k)": "jhu-clsp/mmBERT-small",
    "EmbeddingGemma (262k)": "google/embeddinggemma-300m",
    "granite-97m-r2 (180k)": "ibm-granite/granite-embedding-97m-multilingual-r2",
    "Qwen3 (151k)": "Qwen/Qwen3-0.6B",
    "EuroBERT (128k)": "EuroBERT/EuroBERT-210m",
    "ModernBERT (50k)": "answerdotai/ModernBERT-base",
}


@dataclass
class LangStats:
    chars: int = 0
    bytes: int = 0
    words: int = 0
    tokens: int = 0

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words


def load_eval_texts(codes: list[str], mb_per_lang: float) -> dict[str, list[str]]:
    return {c: list(sample_texts(c, "eval", int(mb_per_lang * 10**6))) for c in codes}


def measure(tokenizer: Tokenizer, texts: dict[str, list[str]]) -> dict[str, LangStats]:
    # Some published tokenizers ship with truncation (e.g. 512 tokens) or padding enabled, which
    # would silently distort token counts.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    stats = {}
    for code, docs in texts.items():
        encodings = tokenizer.encode_batch(docs, add_special_tokens=False)
        s = LangStats()
        s.tokens = sum(len(e.ids) for e in encodings)
        s.chars = sum(len(d) for d in docs)
        s.bytes = sum(len(d.encode("utf-8")) for d in docs)
        s.words = sum(len(d.split()) for d in docs)
        stats[code] = s
    return stats


def summarize(stats: dict[str, LangStats]) -> dict:
    cpt = {c: s.chars_per_token for c, s in stats.items()}
    bpt = {c: s.bytes_per_token for c, s in stats.items()}
    tpw = {c: s.tokens_per_word for c, s in stats.items() if c not in NO_SPACES}
    worst = min(cpt, key=cpt.get)
    return {
        "chars_per_token": cpt,
        "bytes_per_token": bpt,
        "tokens_per_word": tpw,
        "macro_chars_per_token": sum(cpt.values()) / len(cpt),
        "macro_bytes_per_token": sum(bpt.values()) / len(bpt),
        "macro_tokens_per_word": sum(tpw.values()) / len(tpw),
        "worst_lang": worst,
        "worst_chars_per_token": cpt[worst],
    }


def load_tokenizers(vocab_sizes: list[int]) -> dict[str, Tokenizer]:
    tokenizers: dict[str, Tokenizer] = {}
    for size in vocab_sizes:
        path = candidate_path(size)
        if path.exists():
            tokenizers[f"OURS bpe-{size // 1000}k"] = Tokenizer.from_file(str(path))
        else:
            print(f"missing candidate {path}; run python -m src.tokenizer.train")
    for name, repo in BASELINES.items():
        try:
            tokenizers[name] = Tokenizer.from_pretrained(repo)
        except Exception as err:
            print(f"skipping baseline {name} ({repo}): {type(err).__name__}: {str(err)[:80]}")
    return tokenizers


def to_markdown(results: dict[str, dict], codes: list[str]) -> str:
    def table(title: str, key: str, fmt: str, cols: list[str], macro: str) -> str:
        lines = [f"### {title}", "", "| tokenizer | " + " | ".join(cols) + " | macro |"]
        lines.append("|---|" + "---|" * (len(cols) + 1))
        for name, r in results.items():
            cells = [format(r[key][c], fmt) if c in r[key] else "n/a" for c in cols]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {format(r[macro], fmt)} |")
        return "\n".join(lines) + "\n"

    space_cols = [c for c in codes if c not in NO_SPACES]
    return "\n".join(
        [
            "# Tokenizer comparison",
            "",
            table(
                "Characters per token (higher is better)",
                "chars_per_token",
                ".2f",
                codes,
                "macro_chars_per_token",
            ),
            table(
                "Bytes per token (higher is better)",
                "bytes_per_token",
                ".2f",
                codes,
                "macro_bytes_per_token",
            ),
            table(
                "Tokens per word (lower is better; zh and ja excluded)",
                "tokens_per_word",
                ".2f",
                space_cols,
                "macro_tokens_per_word",
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[48000, 64000, 96000, 128000])
    parser.add_argument("--mb-per-lang", type=float, default=8)
    parser.add_argument("--langs", nargs="+", default=all_codes())
    args = parser.parse_args()

    texts = load_eval_texts(args.langs, args.mb_per_lang)
    print({c: f"{sum(len(d) for d in docs) / 1e6:.1f}M chars" for c, docs in texts.items()})
    results = {}
    for name, tokenizer in load_tokenizers(args.vocab_sizes).items():
        results[name] = summarize(measure(tokenizer, texts))
        r = results[name]
        print(
            f"{name:28s} macro chars/token {r['macro_chars_per_token']:.2f}  "
            f"bytes/token {r['macro_bytes_per_token']:.2f}  "
            f"tokens/word {r['macro_tokens_per_word']:.2f}  "
            f"worst {r['worst_lang']} {r['worst_chars_per_token']:.2f}"
        )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    (TOKENIZER_DIR / "report.json").write_text(json.dumps(results, indent=2))
    (TOKENIZER_DIR / "report.md").write_text(to_markdown(results, args.langs))
    print(f"wrote {TOKENIZER_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
