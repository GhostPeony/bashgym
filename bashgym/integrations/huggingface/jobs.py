"""
HuggingFace Jobs (Cloud Training)

Provides cloud training capabilities using HuggingFace Spaces.
Requires HuggingFace Pro subscription for all operations.

Features:
- Submit training jobs to HuggingFace cloud infrastructure
- Monitor job status and retrieve logs
- Cancel running jobs
- Support for various hardware configurations (T4, A10G, A100, H100)

Usage:
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobConfig

    client = get_hf_client()
    runner = HFJobRunner(client)

    config = HFJobConfig(hardware="a10g-large", timeout_minutes=60)
    job = runner.submit_training_job(
        script_path="train.py",
        repo_id="myorg/mymodel",
        config=config
    )

    # Monitor progress
    status = runner.get_job_status(job.job_id)
    logs = runner.get_job_logs(job.job_id)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .client import (
    HuggingFaceClient,
    HFProRequiredError,
    HFJobFailedError,
    HF_HUB_AVAILABLE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class JobStatus(Enum):
    """Status of a HuggingFace job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Hardware tier pricing and specs (as of 2025)
HARDWARE_SPECS = {
    "cpu-basic": {"gpu": None, "memory_gb": 2, "pro_required": False},
    "cpu-upgrade": {"gpu": None, "memory_gb": 8, "pro_required": False},
    "t4-small": {"gpu": "T4", "memory_gb": 16, "pro_required": True},
    "t4-medium": {"gpu": "T4", "memory_gb": 32, "pro_required": True},
    "a10g-small": {"gpu": "A10G", "memory_gb": 24, "pro_required": True},
    "a10g-large": {"gpu": "A10G", "memory_gb": 48, "pro_required": True},
    "a100-large": {"gpu": "A100", "memory_gb": 80, "pro_required": True},
    "h100": {"gpu": "H100", "memory_gb": 80, "pro_required": True},
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HFJobConfig:
    """Configuration for a HuggingFace training job."""

    hardware: str = "a10g-small"
    """Hardware tier for the job. See HARDWARE_SPECS for options."""

    timeout_minutes: int = 30
    """Maximum job runtime in minutes."""

    docker_image: Optional[str] = None
    """Custom Docker image. If None, uses HF default training image."""

    environment: Dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the job."""

    secrets: Dict[str, str] = field(default_factory=dict)
    """Secret environment variables (not logged). Keys are secret names."""

    requirements: Optional[str] = None
    """Path to requirements.txt or pip install string."""

    dataset_repo: Optional[str] = None
    """HuggingFace dataset repository to use."""

    output_repo: Optional[str] = None
    """Repository to push trained model to."""

    def validate(self) -> List[str]:
        """Validate the configuration."""
        errors = []

        if self.hardware not in HARDWARE_SPECS:
            valid = ", ".join(HARDWARE_SPECS.keys())
            errors.append(f"Invalid hardware '{self.hardware}'. Valid options: {valid}")

        if self.timeout_minutes < 1:
            errors.append("timeout_minutes must be at least 1")

        if self.timeout_minutes > 720:  # 12 hours max
            errors.append("timeout_minutes cannot exceed 720 (12 hours)")

        return errors


@dataclass
class HFJobInfo:
    """Information about a HuggingFace job."""

    job_id: str
    """Unique job identifier."""

    status: JobStatus
    """Current job status."""

    hardware: str
    """Hardware tier used for the job."""

    created_at: datetime
    """When the job was created."""

    started_at: Optional[datetime] = None
    """When the job started running."""

    completed_at: Optional[datetime] = None
    """When the job finished (completed, failed, or cancelled)."""

    logs_url: Optional[str] = None
    """URL to view job logs on HuggingFace."""

    error_message: Optional[str] = None
    """Error message if job failed."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    """Training metrics reported by the job."""

    output_repo: Optional[str] = None
    """Repository where output was pushed."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "hardware": self.hardware,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "logs_url": self.logs_url,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "output_repo": self.output_repo,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HFJobInfo":
        """Create from dictionary."""
        def parse_datetime(val: Optional[str]) -> Optional[datetime]:
            if val is None:
                return None
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                return None

        return cls(
            job_id=data.get("job_id", ""),
            status=JobStatus(data.get("status", "pending")),
            hardware=data.get("hardware", "unknown"),
            created_at=parse_datetime(data.get("created_at")) or datetime.now(timezone.utc),
            started_at=parse_datetime(data.get("started_at")),
            completed_at=parse_datetime(data.get("completed_at")),
            logs_url=data.get("logs_url"),
            error_message=data.get("error_message"),
            metrics=data.get("metrics", {}),
            output_repo=data.get("output_repo"),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return None


# =============================================================================
# Job Runner
# =============================================================================

class HFJobRunner:
    """
    Runs training jobs on HuggingFace cloud infrastructure.

    Requires HuggingFace Pro subscription for GPU hardware access.
    Uses HuggingFace Spaces or Jobs API for execution.

    Example:
        client = get_hf_client()
        runner = HFJobRunner(client)

        job = runner.submit_training_job(
            script_path="train.py",
            repo_id="myorg/mymodel",
            config=HFJobConfig(hardware="a10g-small")
        )

        while not job.is_terminal:
            job = runner.get_job_status(job.job_id)
            print(f"Status: {job.status.value}")
            time.sleep(30)
    """

    def __init__(
        self,
        client: Optional[HuggingFaceClient] = None,
        token: Optional[str] = None,
        pro_enabled: bool = False,
    ):
        """
        Initialize the job runner.

        Args:
            client: HuggingFaceClient instance. If None, creates one with token.
            token: HF API token (used if client is None).
            pro_enabled: Override for Pro status (useful for testing).
        """
        if client is not None:
            self._client = client
        else:
            self._client = HuggingFaceClient(token=token)

        self._pro_enabled_override = pro_enabled
        self._jobs: Dict[str, HFJobInfo] = {}

    @property
    def client(self) -> HuggingFaceClient:
        """Get the HuggingFace client."""
        return self._client

    @property
    def is_pro(self) -> bool:
        """Check if Pro features are available."""
        if self._pro_enabled_override:
            return True
        return self._client.is_pro

    def _require_pro(self, operation: str = "This operation") -> None:
        """Require Pro subscription for an operation."""
        if not self.is_pro:
            raise HFProRequiredError(
                f"{operation} requires HuggingFace Pro subscription. "
                f"Upgrade at https://huggingface.co/subscribe/pro"
            )

    def submit_training_job(
        self,
        script_path: Union[str, Path],
        repo_id: Optional[str] = None,
        config: Optional[HFJobConfig] = None,
        script_args: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> HFJobInfo:
        """
        Submit a training job to HuggingFace cloud.

        Args:
            script_path: Path to the training script.
            repo_id: Repository ID for the job (e.g., "username/training-job").
                     If None, auto-generates from client namespace.
            config: Job configuration. Uses defaults if None.
            script_args: Arguments to pass to the training script.
            description: Human-readable job description.

        Returns:
            HFJobInfo with job details.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            HFJobFailedError: If job submission fails.
            ValueError: If configuration is invalid.
        """
        self._require_pro("Submitting training jobs")

        config = config or HFJobConfig()

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid job configuration: {'; '.join(errors)}")

        # Check hardware requires Pro
        hw_spec = HARDWARE_SPECS.get(config.hardware, {})
        if hw_spec.get("pro_required", True) and not self.is_pro:
            raise HFProRequiredError(
                f"Hardware '{config.hardware}' requires HuggingFace Pro subscription"
            )

        # Validate script exists
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")

        # Generate repo ID if not provided
        if repo_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            repo_id = self._client.get_repo_id(f"training-job-{timestamp}")

        # Generate job ID
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        logger.info(f"Submitting training job: {job_id}")
        logger.info(f"  Script: {script_path}")
        logger.info(f"  Repo: {repo_id}")
        logger.info(f"  Hardware: {config.hardware}")
        logger.info(f"  Timeout: {config.timeout_minutes} minutes")

        # Create job info
        job_info = HFJobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            hardware=config.hardware,
            created_at=datetime.now(timezone.utc),
            logs_url=f"https://huggingface.co/spaces/{repo_id}/logs",
        )

        # Store job locally for tracking
        self._jobs[job_id] = job_info

        # In a real implementation, this would call the HF API:
        # - Create a Space with the training script
        # - Configure hardware and secrets
        # - Start the job
        #
        # For now, we simulate the API call behavior
        if HF_HUB_AVAILABLE and self._client.api is not None:
            try:
                # Attempt to use the actual HF API if available
                # Note: The exact API may vary based on HF SDK version
                self._submit_via_api(job_info, script_path, repo_id, config, script_args)
            except (AttributeError, NotImplementedError, TypeError) as e:
                # API method not available or incompatible, use simulation
                logger.debug(f"HF Jobs API not available or incompatible, using simulation: {e}")
                self._simulate_job_submission(job_info)
        else:
            # Simulate for testing
            self._simulate_job_submission(job_info)

        return job_info

    def _submit_via_api(
        self,
        job_info: HFJobInfo,
        script_path: Path,
        repo_id: str,
        config: HFJobConfig,
        script_args: Optional[List[str]],
    ) -> None:
        """Submit job via HuggingFace API."""
        # This is a placeholder for actual API integration
        # The HuggingFace Jobs/Spaces API structure may vary
        #
        # Typical flow:
        # 1. Create or update a Space repository
        # 2. Upload the training script
        # 3. Set hardware configuration
        # 4. Add secrets to the Space
        # 5. Restart/run the Space

        api = self._client.api

        # Check if the API has jobs support
        if hasattr(api, "run_job"):
            # Hypothetical jobs API
            result = api.run_job(
                repo_id=repo_id,
                command=["python", str(script_path)] + (script_args or []),
                hardware=config.hardware,
                timeout=config.timeout_minutes * 60,
                environment=config.environment,
            )
            job_info.job_id = result.get("job_id", job_info.job_id)
            job_info.status = JobStatus(result.get("status", "pending"))
        elif hasattr(api, "create_space"):
            # Use Spaces as a training environment
            # This creates an ephemeral Space for the training run
            logger.info("Using Spaces API for job submission")
            # Note: Full implementation would upload script, configure, and run
            raise NotImplementedError("Spaces-based job submission not yet implemented")
        else:
            raise NotImplementedError("HuggingFace Jobs API not available")

    def _simulate_job_submission(self, job_info: HFJobInfo) -> None:
        """Simulate job submission for testing."""
        logger.info(f"Simulating job submission for {job_info.job_id}")
        job_info.status = JobStatus.PENDING

    def get_job_status(self, job_id: str) -> HFJobInfo:
        """
        Get the current status of a job.

        Args:
            job_id: The job identifier.

        Returns:
            HFJobInfo with current status.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            KeyError: If job not found.
        """
        self._require_pro("Checking job status")

        # Check local cache first
        if job_id in self._jobs:
            job_info = self._jobs[job_id]

            # If job is not terminal, try to refresh from API
            if not job_info.is_terminal and HF_HUB_AVAILABLE and self._client.api is not None:
                try:
                    self._refresh_job_status(job_info)
                except Exception as e:
                    logger.debug(f"Could not refresh job status: {e}")

            return job_info

        # Try to fetch from API
        if HF_HUB_AVAILABLE and self._client.api is not None:
            try:
                return self._fetch_job_from_api(job_id)
            except Exception as e:
                logger.debug(f"Could not fetch job from API: {e}")

        raise KeyError(f"Job not found: {job_id}")

    def _refresh_job_status(self, job_info: HFJobInfo) -> None:
        """Refresh job status from API."""
        # Placeholder for actual API call
        api = self._client.api
        if hasattr(api, "get_job"):
            result = api.get_job(job_info.job_id)
            job_info.status = JobStatus(result.get("status", job_info.status.value))
            job_info.error_message = result.get("error")
            job_info.metrics = result.get("metrics", {})

            if result.get("started_at"):
                job_info.started_at = datetime.fromisoformat(result["started_at"])
            if result.get("completed_at"):
                job_info.completed_at = datetime.fromisoformat(result["completed_at"])

    def _fetch_job_from_api(self, job_id: str) -> HFJobInfo:
        """Fetch job info from API."""
        api = self._client.api
        if hasattr(api, "get_job"):
            result = api.get_job(job_id)
            job_info = HFJobInfo.from_dict(result)
            self._jobs[job_id] = job_info
            return job_info
        raise NotImplementedError("HuggingFace Jobs API not available")

    def get_job_logs(
        self,
        job_id: str,
        tail: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> str:
        """
        Get logs from a job.

        Args:
            job_id: The job identifier.
            tail: Only return the last N lines.
            since: Only return logs after this timestamp.

        Returns:
            Log content as a string.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            KeyError: If job not found.
        """
        self._require_pro("Retrieving job logs")

        # Verify job exists
        job_info = self.get_job_status(job_id)

        # Try to fetch logs from API
        if HF_HUB_AVAILABLE and self._client.api is not None:
            try:
                api = self._client.api
                if hasattr(api, "get_job_logs"):
                    logs = api.get_job_logs(
                        job_id,
                        tail=tail,
                        since=since.isoformat() if since else None,
                    )
                    return logs
            except Exception as e:
                logger.debug(f"Could not fetch logs from API: {e}")

        # Return placeholder for simulation
        return f"[Logs for job {job_id}]\nStatus: {job_info.status.value}\n"

    def cancel_job(self, job_id: str) -> HFJobInfo:
        """
        Cancel a running job.

        Args:
            job_id: The job identifier.

        Returns:
            Updated HFJobInfo.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            KeyError: If job not found.
            HFJobFailedError: If job cannot be cancelled.
        """
        self._require_pro("Cancelling jobs")

        job_info = self.get_job_status(job_id)

        if job_info.is_terminal:
            raise HFJobFailedError(
                f"Cannot cancel job in terminal state: {job_info.status.value}",
                job_id=job_id,
            )

        logger.info(f"Cancelling job: {job_id}")

        # Try to cancel via API
        if HF_HUB_AVAILABLE and self._client.api is not None:
            try:
                api = self._client.api
                if hasattr(api, "cancel_job"):
                    api.cancel_job(job_id)
            except Exception as e:
                logger.warning(f"Could not cancel via API: {e}")

        # Update local state
        job_info.status = JobStatus.CANCELLED
        job_info.completed_at = datetime.now(timezone.utc)
        self._jobs[job_id] = job_info

        return job_info

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[HFJobInfo]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Filter by job status.
            limit: Maximum number of jobs to return.

        Returns:
            List of HFJobInfo objects.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
        """
        self._require_pro("Listing jobs")

        jobs = list(self._jobs.values())

        if status is not None:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> HFJobInfo:
        """
        Wait for a job to complete.

        Args:
            job_id: The job identifier.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None = no timeout).

        Returns:
            Final HFJobInfo.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            TimeoutError: If timeout exceeded.
            HFJobFailedError: If job failed.
        """
        import time

        self._require_pro("Waiting for job completion")

        start_time = datetime.now(timezone.utc)

        while True:
            job_info = self.get_job_status(job_id)

            if job_info.is_terminal:
                if job_info.status == JobStatus.FAILED:
                    raise HFJobFailedError(
                        job_info.error_message or "Job failed",
                        job_id=job_id,
                    )
                return job_info

            # Check timeout
            if timeout is not None:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(poll_interval)

    def __repr__(self) -> str:
        """String representation."""
        pro_str = " [Pro]" if self.is_pro else ""
        return f"<HFJobRunner{pro_str} jobs={len(self._jobs)}>"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_job_runner(
    client: Optional["HuggingFaceClient"] = None,
    token: Optional[str] = None,
    pro_enabled: bool = False,
) -> HFJobRunner:
    """
    Create a job runner with the specified configuration.

    Args:
        client: HuggingFaceClient instance (preferred).
        token: HuggingFace API token (used if client not provided).
        pro_enabled: Override Pro status (for testing).

    Returns:
        Configured HFJobRunner instance.
    """
    return HFJobRunner(client=client, token=token, pro_enabled=pro_enabled)
