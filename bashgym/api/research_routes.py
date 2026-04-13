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


class EmpiricalRequest(BaseModel):
    top_n: int = Field(default=5, ge=1, le=50, description="Number of top-scored SFT candidates")
    mode: str = Field(default="simulate", description="'simulate' (fast, no GPU) or 'real' (trains)")
    num_records: int = Field(default=500, ge=10, le=10000, description="Records per dataset")
    train_steps: int = Field(default=100, ge=10, le=1000, description="SFT steps per candidate")
    base_model: str = Field(default="unsloth/gemma-4-E4B-it", description="Base model for SFT runs")


@router.post("/empirical")
async def run_empirical_ranking(body: EmpiricalRequest):
    """Run an empirical dataset ranking. Evaluate top-N SFT datasets by short training runs.

    In 'simulate' mode this completes instantly. In 'real' mode it trains
    briefly on each candidate (may take 5-60 minutes depending on top_n).
    """
    import asyncio
    from bashgym.research.dataset_research_runner import _run_async

    try:
        exit_code = await _run_async(
            top_n=body.top_n,
            mode=body.mode,
            num_records=body.num_records,
            train_steps=body.train_steps,
            base_model=body.base_model,
            cache_path=CACHE_PATH,
        )
        if exit_code != 0:
            raise HTTPException(status_code=500, detail="Empirical runner returned non-zero exit code")
        content = EMPIRICAL_PATH.read_text() if EMPIRICAL_PATH.exists() else ""
        return {
            "status": "completed",
            "report_path": str(EMPIRICAL_PATH),
            "content": content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
