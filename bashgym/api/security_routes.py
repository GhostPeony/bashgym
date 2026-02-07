"""API routes for security dataset ingestion."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from bashgym.factory.security_ingester import (
    ConversionMode,
    DatasetType,
    IngestionConfig,
    IngestionResult,
    SecurityDomain,
    SecurityIngester,
    DATASET_DOMAINS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security", tags=["security"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SecurityIngestRequest(BaseModel):
    """Request to start a security dataset ingestion job."""
    dataset_type: str = Field(..., description="Dataset type: ember, phishtank, urlhaus, malwarebazaar, cic_ids")
    input_path: str = Field(..., description="Path to the dataset file")
    mode: str = Field("direct", description="Conversion mode: direct or enriched")
    max_samples: Optional[int] = Field(None, description="Maximum samples to process")
    balance_classes: bool = Field(True, description="Balance malicious/benign classes")
    benign_ratio: float = Field(0.3, ge=0.0, le=1.0, description="Target ratio of benign samples")
    output_dir: Optional[str] = Field(None, description="Output directory for JSONL files")
    train_split: float = Field(0.9, ge=0.5, le=1.0, description="Training set proportion")
    # Enrichment settings
    enrichment_provider: str = Field("anthropic", description="LLM provider for enriched mode")
    enrichment_model: str = Field("claude-sonnet-4-5-20250929", description="Model for enrichment")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_type": "phishtank",
                "input_path": "/data/phishtank/online-valid.json",
                "mode": "direct",
                "max_samples": 1000,
                "balance_classes": True
            }
        }


class SecurityIngestResponse(BaseModel):
    """Response from starting an ingestion job."""
    job_id: str
    status: str
    message: str


class SecurityJobStatus(BaseModel):
    """Status of an ingestion job."""
    job_id: str
    status: str  # queued, running, completed, failed
    dataset_type: str
    mode: str
    created_at: str
    completed_at: Optional[str] = None
    # Result fields (populated on completion)
    total_samples_read: int = 0
    examples_generated: int = 0
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    error: Optional[str] = None


class DatasetInfo(BaseModel):
    """Metadata about a supported dataset."""
    dataset_type: str
    name: str
    domain: str
    description: str
    input_formats: List[str]
    example_sources: List[str]


# =============================================================================
# In-Memory Job Storage
# =============================================================================

ingestion_jobs: Dict[str, Dict] = {}


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all supported security datasets with metadata."""
    return [
        DatasetInfo(
            dataset_type="ember",
            name="EMBER",
            domain="malware",
            description="Endgame Malware BEnchmark for Research — PE file features",
            input_formats=["jsonl"],
            example_sources=["https://github.com/elastic/ember"],
        ),
        DatasetInfo(
            dataset_type="phishtank",
            name="PhishTank",
            domain="phishing",
            description="Community-verified phishing URL database",
            input_formats=["json", "json.bz2"],
            example_sources=["https://phishtank.org/developer_info.php"],
        ),
        DatasetInfo(
            dataset_type="urlhaus",
            name="URLhaus",
            domain="phishing",
            description="Malware URL exchange from abuse.ch",
            input_formats=["csv", "json"],
            example_sources=["https://urlhaus.abuse.ch/downloads/"],
        ),
        DatasetInfo(
            dataset_type="malwarebazaar",
            name="MalwareBazaar",
            domain="malware",
            description="Malware sample sharing from abuse.ch",
            input_formats=["json", "jsonl"],
            example_sources=["https://bazaar.abuse.ch/export/"],
        ),
        DatasetInfo(
            dataset_type="cic_ids",
            name="CIC-IDS",
            domain="network",
            description="Canadian Institute for Cybersecurity IDS dataset — network flows",
            input_formats=["csv"],
            example_sources=["https://www.unb.ca/cic/datasets/"],
        ),
    ]


@router.post("/ingest", response_model=SecurityIngestResponse)
async def start_ingestion(
    request: SecurityIngestRequest,
    background_tasks: BackgroundTasks,
):
    """Start a background ingestion job for a security dataset."""
    # Validate dataset type
    try:
        dataset_type = DatasetType(request.dataset_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset type: {request.dataset_type}. "
            f"Supported: {[dt.value for dt in DatasetType]}",
        )

    # Validate input path
    input_path = Path(request.input_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset file not found: {request.input_path}",
        )

    # Validate mode
    try:
        mode = ConversionMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mode: {request.mode}. Supported: direct, enriched",
        )

    job_id = f"sec_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    ingestion_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "dataset_type": dataset_type.value,
        "mode": mode.value,
        "created_at": now,
        "completed_at": None,
        "total_samples_read": 0,
        "examples_generated": 0,
        "train_path": None,
        "val_path": None,
        "train_count": 0,
        "val_count": 0,
        "error": None,
    }

    # Build ingestion config
    output_dir = request.output_dir or f"data/security_training/{job_id}"
    config = IngestionConfig(
        dataset_type=dataset_type,
        input_path=request.input_path,
        mode=mode,
        max_samples=request.max_samples,
        balance_classes=request.balance_classes,
        benign_ratio=request.benign_ratio,
        output_dir=output_dir,
        train_split=request.train_split,
        enrichment_provider=request.enrichment_provider,
        enrichment_model=request.enrichment_model,
    )

    background_tasks.add_task(_run_ingestion, job_id, config)

    return SecurityIngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion job {job_id} queued for {dataset_type.value} dataset",
    )


@router.get("/jobs/{job_id}", response_model=SecurityJobStatus)
async def get_job_status(job_id: str):
    """Get the status of an ingestion job."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return SecurityJobStatus(**ingestion_jobs[job_id])


@router.get("/jobs", response_model=List[SecurityJobStatus])
async def list_jobs():
    """List all ingestion jobs."""
    return [SecurityJobStatus(**job) for job in ingestion_jobs.values()]


# =============================================================================
# Background Task
# =============================================================================

async def _run_ingestion(job_id: str, config: IngestionConfig):
    """Execute ingestion in the background."""
    try:
        ingestion_jobs[job_id]["status"] = "running"
        logger.info(f"[Security Ingestion] Starting job {job_id} for {config.dataset_type.value}")

        ingester = SecurityIngester(config)

        if config.mode == ConversionMode.ENRICHED:
            result = await ingester.ingest_enriched()
        else:
            # Run sync method in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, ingester.ingest_direct)

        ingestion_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_samples_read": result.total_samples_read,
            "examples_generated": result.examples_generated,
            "train_path": result.train_path,
            "val_path": result.val_path,
            "train_count": result.train_count,
            "val_count": result.val_count,
        })

        logger.info(
            f"[Security Ingestion] Job {job_id} completed: "
            f"{result.examples_generated} examples from {result.total_samples_read} samples"
        )

    except Exception as e:
        logger.error(f"[Security Ingestion] Job {job_id} failed: {e}")
        ingestion_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
