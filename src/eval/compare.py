"""Compare our model with the baselines on the MMTEB subset, mixing public and local scores.

    python -m src.eval.compare --ours ours__probe-a__latest.pt
    python -m src.eval.compare --ours ours__probe-a__latest.pt --public-json public_scores.json

Scores come from two places, per (model, task, language subset):
  * local: results our runner (src/eval/mmteb.py) wrote under MTEB_CACHE/results (always used for
    our own model, and for any baseline subset we ran ourselves);
  * public: the embeddings-benchmark/results repository (cloned under MTEB_CACHE/remote), read by
    mteb. A local score overrides a public one for the same subset.
Only language subsets whose languages are all among our 12 (and in the expected script) are used.
A task enters the headline average only if every model has every one of those subsets, so all
models are averaged over exactly the same data. Numbers are in percent.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.data.paths import RAW_DIR, RUNS_DIR, use_local_hf_cache

use_local_hf_cache()
from mteb.cache import ResultCache  # noqa: E402

from .mmteb import ALL_TASKS, ISO1, LANGS, SCRIPT, get_subset  # noqa: E402

ISO3_TO_1 = {v3: v1 for v3, v1 in ISO1.items()} | {"zho": "zh", "cmn": "zh", "arb": "ar"}
ONE_TO_3 = {v: k for k, v in ISO1.items()}
BASELINES = {
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-base": "intfloat/multilingual-e5-base",
    "granite-97m": "ibm-granite/granite-embedding-97m-multilingual-r2",
    "gemma-300m": "google/embeddinggemma-300m",
    "e5-small-v2": "intfloat/e5-small-v2",
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
}
DEFAULT_BASELINES = ["e5-small", "e5-base", "granite-97m", "gemma-300m"]


def side_lang(side: str) -> tuple[str | None, str | None]:
    """(iso1 language, script or None) of one side of a subset name, or (None, None)."""
    code, _, script = side.partition("_")
    lang = ISO3_TO_1.get(code, code if code in ISO3_TO_1.values() else None)
    return lang, script or None


def parse_subset(name: str) -> tuple[str, str] | None:
    """(language of side 1, language of side 2) if the subset is within our languages."""
    sides = name.split("-")
    if len(sides) == 3 and sides[1] in {"Latn", "Hans", "Hant", "Cyrl", "Arab"}:  # 'en-Latn'-style
        sides = [sides[0] + "_" + sides[1], sides[2]]
    sides = sides if len(sides) == 2 else [name, name]
    out = []
    for s in sides:
        lang, script = side_lang(s)
        if lang is None:
            return None
        if script and script not in (SCRIPT.get(ONE_TO_3.get(lang, ""), script), "Hant"):
            return None  # e.g. romanized Arabic (arb_Latn)
        out.append(lang)
    return out[0], out[1]


def load_local(model_dir: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in (RAW_DIR / ".mteb_cache" / "results" / model_dir).glob("*/*.json"):
        if path.name == "model_meta.json":
            continue
        data = json.loads(path.read_text())
        task = data.get("task_name", path.stem)
        for entries in data.get("scores", {}).values():
            for e in entries:
                out.setdefault(task, {})[e["hf_subset"]] = e["main_score"] * 100
    return out


def load_public(models: dict[str, str], tasks) -> dict[str, dict[str, dict[str, float]]]:
    br = ResultCache().load_results(models=list(models.values()), tasks=tasks)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for short, full in models.items():
        for m in br.model_results:
            if m.model_name != full:
                continue
            for tr in m.task_results:
                for entries in tr.scores.values():
                    for e in entries:
                        out.setdefault(short, {}).setdefault(tr.task_name, {})[e["hf_subset"]] = (
                            e["main_score"] * 100
                        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ours", nargs="+", required=True, help="our result dirs under results/")
    parser.add_argument("--out", default="mmteb_compare.md")
    parser.add_argument(
        "--baselines", nargs="+", choices=list(BASELINES), default=DEFAULT_BASELINES,
        help="baseline models to include (names from BASELINES)",
    )  # fmt: skip
    args = parser.parse_args()
    baselines = {k: BASELINES[k] for k in args.baselines}

    tasks = get_subset(ALL_TASKS)
    types = {t.metadata.name: t.metadata.type for t in tasks}
    wanted = {}
    for t in tasks:
        keep = {s for s in t.hf_subsets if parse_subset(s) is not None}
        wanted[t.metadata.name] = keep

    public = load_public(baselines, tasks)
    scores: dict[str, dict[str, dict[str, float]]] = {}
    source: dict[str, dict[str, set[str]]] = {}
    for short, full in baselines.items():
        merged = {t: dict(v) for t, v in public.get(short, {}).items()}
        local = load_local(full.replace("/", "__"))
        src: dict[str, set[str]] = {}
        for task, subs in local.items():
            for s, v in subs.items():
                if s not in merged.get(task, {}):
                    src.setdefault(task, set()).add("local")
                merged.setdefault(task, {})[s] = v
        scores[short] = merged
        source[short] = src
    for d in args.ours:
        scores[d.replace("ours__", "ours ").replace("__latest.pt", "")] = load_local(d)

    names = list(scores)
    complete = {t: all(wanted[t] <= set(scores[m].get(t, {})) for m in names) for t in wanted}
    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("Per task (mean over our language subsets; * = some subsets missing for that model)\n")
    emit(f"{'task':36s} {'type':18s} " + " ".join(f"{m[:11]:>11s}" for m in names))
    for t in wanted:
        row = []
        for m in names:
            sub = scores[m].get(t, {})
            vals = [sub[s] for s in wanted[t] if s in sub]
            if not vals:
                row.append(f"{'-':>11s}")
            else:
                mark = "" if len(vals) == len(wanted[t]) else "*"
                row.append(f"{np.mean(vals):10.1f}{mark or ' '}")
        emit(f"{t:36s} {types[t]:18s} " + " ".join(row))

    emit(f"\nHeadline: {sum(complete.values())} of {len(wanted)} tasks complete for all models")
    emit("Task-type mean and macro (mean of type means) over the complete tasks only\n")
    kinds = sorted({types[t] for t in wanted if complete[t]})
    emit(f"{'':14s} " + " ".join(f"{k[:12]:>12s}" for k in kinds) + f" {'macro':>8s}")
    for m in names:
        per_type = []
        for k in kinds:
            vals = [
                np.mean([scores[m][t][s] for s in wanted[t]])
                for t in wanted
                if complete[t] and types[t] == k
            ]
            per_type.append(float(np.mean(vals)))
        emit(
            f"{m[:14]:14s} "
            + " ".join(f"{v:12.1f}" for v in per_type)
            + f" {np.mean(per_type):8.1f}"
        )

    emit("\nPer language (monolingual subsets of the complete tasks)\n")
    codes = [ISO1[c] for c in LANGS]
    emit(f"{'':14s} " + " ".join(f"{c:>5s}" for c in codes))
    for m in names:
        row = []
        for code in codes:
            vals = [
                scores[m][t][s]
                for t in wanted
                if complete[t]
                for s in wanted[t]
                if parse_subset(s) == (code, code)
            ]
            row.append(f"{np.mean(vals):5.1f}" if vals else f"{'-':>5s}")
        emit(f"{m[:14]:14s} " + " ".join(row))
    (RUNS_DIR / "eval" / args.out).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
