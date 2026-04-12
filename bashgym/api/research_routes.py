"""Research API routes — surface the W2 HF dataset scanner + empirical ranking."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

RESEARCH_DIR = Path("~/.bashgym/research").expanduser()
REPORT_PATH = RESEARCH_DIR / "hf_datasets_report.md"
EMPIRICAL_PATH = RESEARCH_DIR / "dataset_empirical_ranking.md"
CACHE_PATH = RESEARCH_DIR / "hf_datasets_cache.json"


@router.get("/report")
async def get_report():
    """Return the latest HF dataset scanner report as markdown."""
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No scanner report found. Run: python -m bashgym.research.hf_dataset_scanner",
        )
    return {"content": REPORT_PATH.read_text(), "path": str(REPORT_PATH)}


@router.get("/empirical")
async def get_empirical_ranking():
    """Return the latest empirical dataset ranking report."""
    if not EMPIRICAL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No empirical ranking found. Run: python -m bashgym.research.dataset_research_runner",
        )
    return {"content": EMPIRICAL_PATH.read_text(), "path": str(EMPIRICAL_PATH)}


@router.get("/cache/stats")
async def get_cache_stats():
    """Return basic stats about the scanner cache."""
    if not CACHE_PATH.exists():
        return {"cached_datasets": 0, "path": str(CACHE_PATH)}
    import json
    try:
        data = json.loads(CACHE_PATH.read_text())
        return {
            "cached_datasets": len(data),
            "path": str(CACHE_PATH),
            "sample_ids": list(data.keys())[:5],
        }
    except (json.JSONDecodeError, OSError):
        return {"cached_datasets": 0, "path": str(CACHE_PATH), "error": "corrupt cache"}


class ScanRequest(BaseModel):
    limit: int | None = Field(default=None, description="Max results per query")
    max_candidates: int = Field(default=200, description="Hard cap on candidates to enrich")


@router.post("/scan")
async def trigger_scan(body: ScanRequest):
    """Trigger a fresh HF dataset scan. Runs synchronously (may take 30-120s)."""
    from bashgym.research.hf_dataset_scanner import run
    try:
        exit_code = run(
            limit_per_query=body.limit,
            use_cache=True,
            max_candidates=body.max_candidates,
        )
        if exit_code != 0:
            raise HTTPException(status_code=500, detail="Scanner returned non-zero exit code")
        report_content = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
        return {
            "status": "completed",
            "report_path": str(REPORT_PATH),
            "report_preview": report_content[:2000],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
