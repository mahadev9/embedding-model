"""Embedding benchmark on our 12 languages: a fixed subset of MTEB / MMTEB tasks.

    python -m src.eval.mmteb --models ours:RUN/CKPT intfloat/multilingual-e5-small
    python -m src.eval.mmteb --models intfloat/multilingual-e5-small --tasks STS22.v2 --dry-run
    python -m src.eval.mmteb --prefetch                  # download every task's data, run nothing

Models: `ours:RUN/CKPT` (a checkpoint under RUNS_DIR: a trained embedding model, or a raw MLM
checkpoint as a zero-training baseline) or any Hugging Face name that `mteb.get_model` knows
(multilingual-e5, granite, EmbeddingGemma, ...), which gets the model's own prompts.

Tasks (SUBSET below) cover bitext mining, classification, clustering, STS, pair classification and
retrieval. Only subsets whose languages are all among ours are scored (`exclusive_language_filter`),
so cross-lingual pairs such as zh-en are kept and fr-pl is dropped. Results are cached under
MTEB_CACHE (on the HDD) and a summary is written to <RUNS_DIR>/eval/mmteb_<name>.json.

The score table reports, per model: the mean over tasks of each task type, the mean of those type
means ("macro"), and per language the mean over every monolingual subset. These are our own
subset numbers; they are not comparable to the official MTEB leaderboard averages.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from src.data.paths import RUNS_DIR, TOKENIZED_DIR, use_local_hf_cache

use_local_hf_cache()
import mteb  # noqa: E402
from mteb.cache import ResultCache  # noqa: E402
from mteb.models import ModelMeta  # noqa: E402

from src.model import PAD_DOC_ID, EmbeddingModel, build_block_masks, load_model_config  # noqa: E402

DEVICE = "cuda"
CLS_ID, SEP_ID = 2, 3
LANGS = ["eng", "cmn", "spa", "fra", "deu", "por", "rus", "jpn", "arb", "hin", "kor", "ita"]
ISO1 = {
    "eng": "en", "cmn": "zh", "spa": "es", "fra": "fr", "deu": "de", "por": "pt",
    "rus": "ru", "jpn": "ja", "arb": "ar", "hin": "hi", "kor": "ko", "ita": "it",
}  # fmt: skip
ISO1_ALT = {"zh": "zh", "cmn": "zh", "ar": "ar", "arb": "ar"}

SCRIPT = {
    "eng": "Latn", "cmn": "Hans", "spa": "Latn", "fra": "Latn", "deu": "Latn", "por": "Latn",
    "rus": "Cyrl", "jpn": "Jpan", "arb": "Arab", "hin": "Deva", "kor": "Hang", "ita": "Latn",
}  # fmt: skip

SUBSET = {
    "BitextMining": ["Tatoeba", "NTREXBitextMining"],
    "Classification": [
        "MassiveIntentClassification",
        "SIB200Classification.v3",
        "MultilingualSentimentClassification",
        "MTOPDomainClassification",
    ],
    "Clustering": ["SIB200ClusteringS2S"],
    "STS": ["STS17", "STS22.v2", "STSBenchmarkMultilingualSTS"],
    "PairClassification": ["XNLI", "PawsXPairClassification"],
    "Retrieval": [
        "XPQARetrieval",
        "BelebeleRetrieval",
        "MIRACLRetrievalHardNegatives",
        "WikipediaRetrievalMultilingual",
        "MLQARetrieval",
    ],
}
ALL_TASKS = [t for names in SUBSET.values() for t in names]


# --------------------------------------------------------------------------------- our model


class OursEncoder:
    """Our encoder as an mteb model. Pooling and Matryoshka dimension come from the config."""

    def __init__(self, spec: str, max_len: int = 512, dim: int | None = None, prefix: str = ""):
        run, _, ckpt = spec.partition("/")
        run_dir = RUNS_DIR / run
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.model_cfg = load_model_config(cfg["model"])
        self.model = EmbeddingModel(self.model_cfg).to(DEVICE).eval()
        state = torch.load(run_dir / ckpt, map_location=DEVICE, weights_only=True)["model"]
        if any(k.startswith("head_dense") for k in state):  # a raw MLM checkpoint
            enc = {
                k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")
            }
            self.model.encoder.load_state_dict(enc)
            self.kind = "raw MLM encoder (no embedding training)"
        else:
            self.model.load_state_dict(state)
            self.kind = "embedding model"
        self.tok = Tokenizer.from_file(str(TOKENIZED_DIR / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.max_len, self.dim, self.prefix = max_len, dim, prefix
        n_params = sum(p.numel() for p in self.model.parameters())
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"ours/{spec}",
            revision="local",
            release_date=None,
            languages=[f"{c}-{SCRIPT[c]}" for c in LANGS],
            n_parameters=n_params,
            memory_usage_mb=None,
            max_tokens=max_len,
            embed_dim=dim or self.model_cfg.output_dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    @torch.no_grad()
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs) -> np.ndarray:  # fmt: skip
        texts = [self.prefix + t for batch in inputs for t in batch["text"]]
        enc = self.tok.encode_batch(texts, add_special_tokens=False)
        ids = [[CLS_ID, *e.ids[: self.max_len - 2], SEP_ID] for e in enc]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = [None] * len(ids)
        budget = 32768  # tokens per batch (after padding)
        i = 0
        while i < len(order):
            width = -(-len(ids[order[i]]) // 128) * 128
            n = max(1, min(256, budget // width))
            idx = order[i : i + n]
            i += n
            batch = torch.zeros(len(idx), width, dtype=torch.long)
            attn = torch.zeros(len(idx), width, dtype=torch.long)
            for r, j in enumerate(idx):
                batch[r, : len(ids[j])] = torch.tensor(ids[j])
                attn[r, : len(ids[j])] = 1
            batch, attn = batch.to(DEVICE), attn.to(DEVICE)
            doc = torch.where(attn.bool(), 0, PAD_DOC_ID).to(torch.int32)
            masks = build_block_masks(doc, self.model_cfg.local_window)
            with torch.autocast("cuda", torch.bfloat16):
                emb = self.model(batch, attn, dim=self.dim, block_masks=masks)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            for r, j in enumerate(idx):
                out[j] = emb[r]
        return np.stack(out)

    def similarity(self, embeddings1, embeddings2):
        """Cosine similarity matrix (our embeddings are L2-normalized)."""
        a = torch.as_tensor(embeddings1, dtype=torch.float32).reshape(-1, embeddings1.shape[-1])
        b = torch.as_tensor(embeddings2, dtype=torch.float32).reshape(-1, embeddings2.shape[-1])
        return a @ b.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        """Cosine similarity of corresponding rows."""
        a = torch.as_tensor(embeddings1, dtype=torch.float32).reshape(-1, embeddings1.shape[-1])
        b = torch.as_tensor(embeddings2, dtype=torch.float32).reshape(-1, embeddings2.shape[-1])
        return (a * b).sum(dim=-1)


def load_model(spec: str, args):
    if spec.startswith("ours:"):
        return OursEncoder(spec[5:], max_len=args.max_len, dim=args.dim, prefix=args.prefix)
    return mteb.get_model(spec)


# ------------------------------------------------------------------------------------ summary


def summarize(model_result, tasks) -> dict:
    """Per task: mean main score over scored subsets; per language: monolingual subsets."""
    per_task, per_lang = {}, defaultdict(list)
    for tr in model_result.task_results:
        scores = []
        for split_scores in tr.scores.values():
            for entry in split_scores:
                scores.append(entry["main_score"] * 100)
                sub = entry["hf_subset"]
                if re.fullmatch(r"[a-z]{2,3}", sub) or re.fullmatch(r"[a-z]{3}-[A-Z][a-z]{3}", sub):
                    per_lang[sub.split("-")[0]].append(entry["main_score"] * 100)
        per_task[tr.task_name] = float(np.mean(scores)) if scores else float("nan")
    by_type = defaultdict(list)
    for t in tasks:
        if t.metadata.name in per_task:
            by_type[t.metadata.type].append(per_task[t.metadata.name])
    type_means = {k: float(np.mean(v)) for k, v in by_type.items()}
    return {
        "per_task": per_task,
        "per_type": type_means,
        "macro": float(np.mean(list(type_means.values()))) if type_means else float("nan"),
        "per_language": {k: float(np.mean(v)) for k, v in sorted(per_lang.items())},
    }


def get_subset(names: list[str]):
    return mteb.get_tasks(tasks=names, languages=LANGS, exclusive_language_filter=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--prefetch", action="store_true", help="download task data and exit")
    parser.add_argument("--max-len", type=int, default=512, help="tokens per text (ours)")
    parser.add_argument("--dim", type=int, help="Matryoshka dimension (ours)")
    parser.add_argument("--prefix", default="", help="text prefix (ours)")
    parser.add_argument("--batch-size", type=int, default=64, help="baselines' batch size")
    parser.add_argument("--out", default="mmteb.json")
    args = parser.parse_args()

    tasks = get_subset(args.tasks)
    if args.prefetch:
        for t in tasks:
            t.load_data()
            print(f"downloaded {t.metadata.name} ({len(t.hf_subsets)} subsets)", flush=True)
        return

    cache = ResultCache()
    out_path = RUNS_DIR / "eval" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec in args.models:
        model = load_model(spec, args)
        results = mteb.evaluate(
            model, tasks, cache=cache, encode_kwargs={"batch_size": args.batch_size}
        )
        summary[spec] = summarize(results, tasks)
        out_path.write_text(json.dumps(summary, indent=1))
        s = summary[spec]
        print(
            f"\n{spec}: macro {s['macro']:.2f}  "
            + " ".join(f"{k} {v:.1f}" for k, v in s["per_type"].items())
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
