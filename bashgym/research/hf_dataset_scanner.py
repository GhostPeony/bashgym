"""CLI: scan HuggingFace Hub for candidate training datasets and write a ranked report.

Usage:
    python -m bashgym.research.hf_dataset_scanner
    python bashgym/research/hf_dataset_scanner.py --limit 50 --no-cache

Output:
    ~/.bashgym/research/hf_datasets_report.md
    ~/.bashgym/research/hf_datasets_cache.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from bashgym.research.hf_client import SEARCH_QUERIES, HFResearchClient
from bashgym.research.report import render_report
from bashgym.research.scoring import DatasetMetadata, ScoredDataset, score_dataset

logger = logging.getLogger("hf_dataset_scanner")

OUTPUT_DIR = Path(os.path.expanduser("~/.bashgym/research"))
REPORT_PATH = OUTPUT_DIR / "hf_datasets_report.md"
CACHE_PATH = OUTPUT_DIR / "hf_datasets_cache.json"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("cache load failed (%s) — starting fresh", e)
        return {}


def _save_cache(cache: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))


def _metadata_to_cache(meta: DatasetMetadata) -> dict:
    return {
        "repo_id": meta.repo_id,
        "tags": meta.tags,
        "license": meta.license,
        "num_rows": meta.num_rows,
        "download_size_bytes": meta.download_size_bytes,
        "features": meta.features,
        "last_modified": meta.last_modified.isoformat() if meta.last_modified else None,
        "downloads": meta.downloads,
        "gated": meta.gated,
        "description": meta.description,
    }


def _metadata_from_cache(d: dict) -> DatasetMetadata:
    lm = d.get("last_modified")
    last_modified = None
    if lm:
        try:
            last_modified = datetime.fromisoformat(lm)
        except ValueError:
            last_modified = None
    return DatasetMetadata(
        repo_id=d["repo_id"],
        tags=d.get("tags") or [],
        license=d.get("license"),
        num_rows=d.get("num_rows"),
        download_size_bytes=d.get("download_size_bytes"),
        features=d.get("features") or {},
        last_modified=last_modified,
        downloads=d.get("downloads", 0),
        gated=d.get("gated", False),
        description=d.get("description", ""),
    )


def run(
    limit_per_query: int | None = None, use_cache: bool = True, max_candidates: int = 500
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logger.info(
        "Starting HF dataset scan — cache=%s, limit_per_query=%s", use_cache, limit_per_query
    )

    token = os.environ.get("HF_TOKEN")
    client = HFResearchClient(token=token)

    queries = SEARCH_QUERIES
    if limit_per_query is not None:
        queries = [
            {**q, "limit": min(q.get("limit", limit_per_query), limit_per_query)} for q in queries
        ]

    logger.info("Running %d discovery queries…", len(queries))
    repo_ids = client.discover_candidates(queries)
    logger.info("Discovered %d unique candidates", len(repo_ids))

    if len(repo_ids) > max_candidates:
        logger.info("Capping at max_candidates=%d", max_candidates)
        repo_ids = repo_ids[:max_candidates]

    cache = _load_cache() if use_cache else {}
    accepted: list[ScoredDataset] = []
    rejected: list[ScoredDataset] = []

    for i, repo_id in enumerate(repo_ids, start=1):
        meta: DatasetMetadata | None = None

        if use_cache and repo_id in cache:
            try:
                meta = _metadata_from_cache(cache[repo_id])
                logger.debug("[%d/%d] cache hit: %s", i, len(repo_ids), repo_id)
            except (KeyError, ValueError):
                meta = None

        if meta is None:
            logger.info("[%d/%d] enriching %s", i, len(repo_ids), repo_id)
            meta = client.enrich(repo_id)
            if meta is None:
                continue
            cache[repo_id] = _metadata_to_cache(meta)

        scored = score_dataset(meta)
        if scored.rejected:
            rejected.append(scored)
        else:
            accepted.append(scored)

    if use_cache:
        _save_cache(cache)
        logger.info("Saved cache with %d entries to %s", len(cache), CACHE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_md = render_report(accepted, rejected)
    REPORT_PATH.write_text(report_md)
    logger.info(
        "Wrote report to %s (%d accepted, %d rejected)", REPORT_PATH, len(accepted), len(rejected)
    )

    top = sorted(accepted, key=lambda s: s.score, reverse=True)[:20]
    print()
    print("Top 20 candidates:")
    print(f"{'rank':>4}  {'score':>5}  {'format':<6}  repo_id")
    for rank, s in enumerate(top, start=1):
        fmt = s.bashgym_format or "—"
        print(f"{rank:>4}  {s.score:>5.2f}  {fmt:<6}  {s.repo_id}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan HF Hub for candidate training datasets.")
    ap.add_argument("--limit", type=int, default=None, help="Max results per discovery query")
    ap.add_argument(
        "--no-cache", action="store_true", help="Ignore the JSON cache and re-enrich everything"
    )
    ap.add_argument(
        "--max-candidates", type=int, default=500, help="Hard cap on candidates to enrich"
    )
    args = ap.parse_args()
    return run(
        limit_per_query=args.limit, use_cache=not args.no_cache, max_candidates=args.max_candidates
    )


if __name__ == "__main__":
    sys.exit(main())
