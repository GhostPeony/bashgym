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
    mode: str = Field(
        default="simulate", description="'simulate' (fast, no GPU) or 'real' (trains)"
    )
    num_records: int = Field(default=500, ge=10, le=10000, description="Records per dataset")
    train_steps: int = Field(default=100, ge=10, le=1000, description="SFT steps per candidate")
    base_model: str = Field(default="unsloth/gemma-4-E4B-it", description="Base model for SFT runs")


@router.post("/empirical")
async def run_empirical_ranking(body: EmpiricalRequest):
    """Run an empirical dataset ranking. Evaluate top-N SFT datasets by short training runs.

    In 'simulate' mode this completes instantly. In 'real' mode it trains
    briefly on each candidate (may take 5-60 minutes depending on top_n).
    """
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
            raise HTTPException(
                status_code=500, detail="Empirical runner returned non-zero exit code"
            )
        content = EMPIRICAL_PATH.read_text() if EMPIRICAL_PATH.exists() else ""
        return {
            "status": "completed",
            "report_path": str(EMPIRICAL_PATH),
            "content": content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Firecrawl research index — news feed + AutoResearch advice
# =============================================================================


class AdviseRequest(BaseModel):
    base_model: str = Field("", description="Base model (used to resolve the family)")
    strategy: str = Field("sft", description="sft | dpo | grpo | rlvr | distillation")
    family: str = Field("", description="Override family; else resolved from base_model")
    domain: str = Field("", description="Optional weak capability/domain to target")


def _build_advisor():
    """A ResearchAdvisor over a fresh Firecrawl client (key from FIRECRAWL_API_KEY)."""
    from bashgym.research.advisor import ResearchAdvisor
    from bashgym.research.firecrawl_client import FirecrawlResearchClient

    client = FirecrawlResearchClient()
    return ResearchAdvisor(client), client


@router.get("/news")
async def research_news(k: int = 20, since: str | None = None):
    """Platform news feed: latest training-method changes (tracked GitHub repos +
    recent papers). Advisory — returns ``configured: false`` (no items) when
    ``FIRECRAWL_API_KEY`` is unset rather than erroring."""
    advisor, client = _build_advisor()
    try:
        items = await advisor.news_feed(k=k, since=since)
    finally:
        await client.close()
    return {
        "configured": client.configured,
        "count": len(items),
        "items": [i.to_dict() for i in items],
    }


@router.post("/advise")
async def research_advise(req: AdviseRequest):
    """Literature/GitHub-grounded advice for a training context: recommended
    techniques, relevant issues, and a hyperparameter prior for AutoResearch."""
    from bashgym.research.advisor import TrainingContext

    family = req.family
    if not family and req.base_model:
        try:
            from bashgym.families import resolve_family_profile

            family = resolve_family_profile(req.base_model).family
        except Exception:  # noqa: BLE001 - family is best-effort
            family = ""

    advisor, client = _build_advisor()
    try:
        report = await advisor.advise(
            TrainingContext(
                base_model=req.base_model,
                strategy=req.strategy,
                family=family,
                domain=req.domain,
            )
        )
    finally:
        await client.close()
    return {"configured": client.configured, **report.to_dict()}
