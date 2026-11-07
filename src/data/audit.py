"""Audit the pretraining sources: licenses, plus per-language sizes for the 12 target languages.

Uses only the Hugging Face API (no data is downloaded). Writes data/audit/sources.json.

    python -m src.data.audit
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .paths import AUDIT_DIR, use_local_hf_cache

use_local_hf_cache()

from huggingface_hub import HfApi  # noqa: E402

from .languages import (  # noqa: E402
    FINEWEB2_REPO,
    FINEWEB_EDU_REPO,
    LANGUAGES,
    WIKIPEDIA_REPO,
    WIKIPEDIA_SNAPSHOT,
)

GIB = 2**30


def repo_license(api: HfApi, repo: str) -> str:
    info = api.dataset_info(repo)
    lic = info.card_data.get("license") if info.card_data else None
    return ", ".join(lic) if isinstance(lic, list) else str(lic)


def folder_size(api: HfApi, repo: str, path: str) -> tuple[int, int]:
    """(file count, total bytes) of all files under a repo path."""
    files = [
        f
        for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=path, recursive=True)
        if getattr(f, "size", None) is not None
    ]
    return len(files), sum(f.size for f in files)


def server_size(repo: str, config: str, split: str = "train") -> dict | None:
    """Row/byte counts from the datasets-server, when it has them (best effort)."""
    query = urllib.parse.urlencode({"dataset": repo, "config": config})
    url = f"https://datasets-server.huggingface.co/size?{query}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        for s in data.get("size", {}).get("splits", []):
            if s["split"] == split:
                return {"rows": s["num_rows"], "parquet_bytes": s["num_bytes_parquet_files"]}
    except Exception:
        return None
    return None


def main() -> None:
    api = HfApi()
    report: dict = {"licenses": {}, "fineweb2": {}, "fineweb_edu": {}, "wikipedia": {}}

    print("== licenses")
    for repo in (FINEWEB2_REPO, FINEWEB_EDU_REPO, WIKIPEDIA_REPO):
        report["licenses"][repo] = repo_license(api, repo)
        print(f"{repo:32s} {report['licenses'][repo]}")

    print("\n== FineWeb-2 (train split, parquet size on disk)")
    print(f"{'lang':5s} {'config':10s} {'files':>5s} {'GiB':>8s} {'rows':>14s}")
    for lang in LANGUAGES:
        if lang.fineweb2 is None:
            continue
        files, size = folder_size(api, FINEWEB2_REPO, f"data/{lang.fineweb2}/train")
        rows = server_size(FINEWEB2_REPO, lang.fineweb2)
        report["fineweb2"][lang.code] = {
            "config": lang.fineweb2,
            "files": files,
            "bytes": size,
            "rows": rows["rows"] if rows else None,
        }
        rows_str = f"{rows['rows']:,}" if rows else "n/a"
        print(f"{lang.code:5s} {lang.fineweb2:10s} {files:5d} {size / GIB:8.1f} {rows_str:>14s}")

    print("\n== FineWeb-Edu (English)")
    root = [
        f.path for f in api.list_repo_tree(FINEWEB_EDU_REPO, repo_type="dataset", recursive=False)
    ]
    print("top level:", root)
    sample_dirs = [p for p in root if p.startswith("sample")]
    for path in [*sample_dirs]:
        files, size = folder_size(api, FINEWEB_EDU_REPO, path)
        report["fineweb_edu"][path] = {"files": files, "bytes": size}
        print(f"{path:12s} {files:5d} files {size / GIB:8.1f} GiB")

    print(f"\n== Wikipedia ({WIKIPEDIA_SNAPSHOT} snapshot, parquet size)")
    for lang in LANGUAGES:
        config = f"{WIKIPEDIA_SNAPSHOT}.{lang.wikipedia}"
        try:
            files, size = folder_size(api, WIKIPEDIA_REPO, config)
        except Exception as err:
            print(f"{lang.code:5s} {config:14s} unavailable ({str(err)[:60]})")
            continue
        report["wikipedia"][lang.code] = {"config": config, "files": files, "bytes": size}
        print(f"{lang.code:5s} {config:14s} {files:5d} files {size / GIB:8.2f} GiB")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "sources.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
