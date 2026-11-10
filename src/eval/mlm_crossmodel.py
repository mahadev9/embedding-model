"""Compare masked-language-modelling quality across models with different tokenizers.

    python -m src.eval.mlm_crossmodel \\
        --models ours:decay-10.00B/latest.pt ours:decay-3.00B/latest.pt \\
                 hf:jhu-clsp/mmBERT-small hf:jhu-clsp/mmBERT-base hf:FacebookAI/xlm-roberta-base

Token-level losses are not comparable across tokenizers, so every model is scored on the same text
with the same masked character spans, and the loss is reported as bits per masked byte:

  * Text: held-out documents from our validation files, decoded back to text (NFKC-normalized),
    cut to `--chunk-bytes` UTF-8 bytes. The same texts go to every model.
  * Masking: the text is cut into blocks of `--span-chars` characters and each block is masked with
    probability `--mask-rate` (seeded per text, identical for every model). A model masks every
    token that overlaps a masked block, 100% as its [MASK] token; the other tokens stay visible.
  * Score: sum of the negative log-likelihood (bits) of the masked tokens divided by the UTF-8
    bytes of those tokens. `overhang` is the masked tokens' bytes divided by the bytes of the masked
    blocks (1.0 = the tokens cover the blocks exactly; larger means the model had to predict extra
    text because its tokens straddle block edges).

Caveats: the public models were trained on FineWeb-2 and may have seen these documents; ours has
not. They were trained with other mask rates and masking schemes. Read the result as a rough
comparison of language-modelling strength, not as an embedding-quality score.
"""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from src.data.languages import LANGUAGES
from src.data.paths import RUNS_DIR, TOKENIZED_DIR, use_local_hf_cache
from src.model import PAD_DOC_ID, MLMModel, build_block_masks, load_model_config

use_local_hf_cache()

DEVICE = "cuda"
CLS_ID, SEP_ID, MASK_ID = 2, 3, 4
PAD_TO = 128  # our encoder works in 128-token blocks
EVAL_DIR = RUNS_DIR / "eval"


# ------------------------------------------------------------------------------------- texts


def load_texts(docs_per_lang: int, chunk_bytes: int) -> dict[str, list[str]]:
    """Held-out texts per language, decoded from our validation token files."""
    cache = EVAL_DIR / f"texts_{docs_per_lang}_{chunk_bytes}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
    out: dict[str, list[str]] = {}
    for lang in LANGUAGES:
        ids = np.fromfile(TOKENIZED_DIR / lang.code / "val.bin", dtype="<u4")
        starts = np.flatnonzero(ids == CLS_ID)
        texts: list[str] = []
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(ids)
            doc = tok.decode(ids[s + 1 : end - 1].tolist(), skip_special_tokens=True)
            raw = doc.encode()
            if len(raw) >= chunk_bytes:
                texts.append(raw[:chunk_bytes].decode(errors="ignore"))
            if len(texts) == docs_per_lang:
                break
        out[lang.code] = texts
        print(f"{lang.code}: {len(texts)} texts", flush=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def masked_blocks(text: str, rate: float, span: int, seed: int) -> list[tuple[int, int]]:
    """Character ranges [start, end) to mask: each `span`-char block with probability `rate`."""
    rng = random.Random(seed)
    return [(s, min(s + span, len(text))) for s in range(0, len(text), span) if rng.random() < rate]


def token_mask(offsets: list[tuple[int, int]], blocks: list[tuple[int, int]]) -> list[bool]:
    """True for tokens (with real offsets) that overlap any masked block."""
    flags = []
    for a, b in offsets:
        flags.append(b > a and any(a < be and b > bs for bs, be in blocks))
    return flags


# ------------------------------------------------------------------------------------ models


class OursScorer:
    def __init__(self, spec: str) -> None:
        run, _, ckpt = spec.partition("/")
        run_dir = RUNS_DIR / run
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.model = MLMModel(load_model_config(cfg["model"])).to(DEVICE).eval()
        state = torch.load(run_dir / ckpt, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.local_window = self.model.cfg.local_window
        self.tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.name = f"ours {spec}"

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        e = self.tok.encode(text, add_special_tokens=False)
        return [CLS_ID, *e.ids, SEP_ID], [(0, 0), *e.offsets, (0, 0)]

    @torch.no_grad()
    def nll(self, batch: list[tuple[list[int], list[bool]]]) -> list[torch.Tensor]:
        """Per-token NLL (nats) of each example's masked tokens."""
        width = -(-max(len(ids) for ids, _ in batch) // PAD_TO) * PAD_TO
        inputs = torch.zeros(len(batch), width, dtype=torch.long)
        doc = torch.full((len(batch), width), PAD_DOC_ID, dtype=torch.int32)
        flags = torch.zeros(len(batch), width, dtype=torch.bool)
        for r, (ids, mask) in enumerate(batch):
            inputs[r, : len(ids)] = torch.tensor(ids)
            doc[r, : len(ids)] = 0
            flags[r, : len(mask)] = torch.tensor(mask)
        targets = inputs.clone()
        inputs[flags] = MASK_ID
        inputs, doc, flags, targets = (x.to(DEVICE) for x in (inputs, doc, flags, targets))
        masks = build_block_masks(doc, self.local_window)
        with torch.autocast("cuda", torch.bfloat16):
            hidden = self.model.encoder(inputs, doc, None, masks)
            logits = self.model._logits(hidden[flags])
        loss = F.cross_entropy(logits.float(), targets[flags], reduction="none")
        counts = flags.sum(dim=1).tolist()
        return list(loss.split(counts))


class HFScorer:
    def __init__(self, repo: str) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForMaskedLM.from_pretrained(repo).to(DEVICE).eval()
        positions = getattr(self.model.config, "max_position_embeddings", 512)
        self.max_len = min(positions, 1024) - 4  # leave room for special tokens and offsets
        self.name = f"hf {repo}"

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        e = self.tok(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len + 2,
        )
        return e["input_ids"], [tuple(o) for o in e["offset_mapping"]]

    @torch.no_grad()
    def nll(self, batch: list[tuple[list[int], list[bool]]]) -> list[torch.Tensor]:
        width = max(len(ids) for ids, _ in batch)
        pad = self.tok.pad_token_id or 0
        inputs = torch.full((len(batch), width), pad, dtype=torch.long)
        attn = torch.zeros(len(batch), width, dtype=torch.long)
        flags = torch.zeros(len(batch), width, dtype=torch.bool)
        for r, (ids, mask) in enumerate(batch):
            inputs[r, : len(ids)] = torch.tensor(ids)
            attn[r, : len(ids)] = 1
            flags[r, : len(mask)] = torch.tensor(mask)
        targets = inputs.clone()
        inputs[flags] = self.tok.mask_token_id
        inputs, attn, flags, targets = (x.to(DEVICE) for x in (inputs, attn, flags, targets))
        with torch.autocast("cuda", torch.bfloat16):
            logits = self.model(input_ids=inputs, attention_mask=attn).logits
        loss = F.cross_entropy(logits[flags].float(), targets[flags], reduction="none")
        counts = flags.sum(dim=1).tolist()
        return list(loss.split(counts))


# ------------------------------------------------------------------------------------- score


def score(scorer, texts: dict[str, list[str]], rate: float, span: int, batch_size: int):
    """Per-language (bits per masked byte, overhang, nats per masked token) for one model."""
    results = {}
    for k, (code, items) in enumerate(texts.items()):
        examples, byte_counts, target_bytes = [], [], []
        for i, text in enumerate(items):
            blocks = masked_blocks(text, rate, span, seed=1000 * k + i)
            ids, offsets = scorer.encode(text)
            flags = token_mask(offsets, blocks)
            if not any(flags):
                continue
            examples.append((ids, flags))
            byte_counts.append(
                sum(len(text[a:b].encode()) for (a, b), f in zip(offsets, flags, strict=True) if f)
            )
            target_bytes.append(sum(len(text[a:b].encode()) for a, b in blocks))
        order = sorted(range(len(examples)), key=lambda j: len(examples[j][0]))
        nll_sum = tokens = 0.0
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            for loss in scorer.nll([examples[j] for j in idx]):
                nll_sum += loss.sum().item()
                tokens += loss.numel()
        masked_bytes = sum(byte_counts)
        results[code] = {
            "bpb": nll_sum / math.log(2) / masked_bytes,
            "overhang": masked_bytes / sum(target_bytes),
            "nats_per_token": nll_sum / tokens,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True, help="ours:RUN/CKPT or hf:REPO")
    parser.add_argument("--docs-per-lang", type=int, default=300)
    parser.add_argument("--chunk-bytes", type=int, default=1200)
    parser.add_argument("--mask-rates", nargs="+", type=float, default=[0.15, 0.30])
    parser.add_argument("--span-chars", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="mlm_crossmodel.json")
    args = parser.parse_args()

    texts = load_texts(args.docs_per_lang, args.chunk_bytes)
    all_results: dict = {}
    for spec in args.models:
        kind, _, rest = spec.partition(":")
        scorer = OursScorer(rest) if kind == "ours" else HFScorer(rest)
        for rate in args.mask_rates:
            res = score(scorer, texts, rate, args.span_chars, args.batch_size)
            all_results[f"{scorer.name} @ {rate}"] = res
            macro = sum(v["bpb"] for v in res.values()) / len(res)
            over = sum(v["overhang"] for v in res.values()) / len(res)
            print(f"{scorer.name:55s} mask {rate:.2f}  bits/byte {macro:.4f}  overhang {over:.2f}")
        del scorer
        torch.cuda.empty_cache()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / args.out).write_text(json.dumps(all_results, indent=1))
    codes = [lang.code for lang in LANGUAGES]
    for rate in args.mask_rates:
        print(f"\nbits per masked byte, mask rate {rate} (lower is better)")
        print(f"{'model':48s} " + " ".join(f"{c:>5s}" for c in codes) + "  macro")
        for name, res in all_results.items():
            if not name.endswith(f"@ {rate}"):
                continue
            row = " ".join(f"{res[c]['bpb']:5.2f}" for c in codes)
            macro = sum(res[c]["bpb"] for c in codes) / len(codes)
            print(f"{name.rsplit(' @', 1)[0]:48s} {row}  {macro:.3f}")


if __name__ == "__main__":
    main()
