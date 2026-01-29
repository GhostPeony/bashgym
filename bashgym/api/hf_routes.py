"""
HuggingFace API routes for Bash Gym.

Provides REST endpoints for HuggingFace Pro features:
- Status and Pro detection
- Cloud training jobs
- Inference API
- Spaces management
- Dataset management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hf", tags=["huggingface"])


# =============================================================================
# Schemas
# =============================================================================

class HFStatusResponse(BaseModel):
    """HuggingFace integration status."""
    enabled: bool = Field(description="Whether HF integration is enabled")
    pro_enabled: bool = Field(description="Whether user has Pro subscription")
    username: str = Field(default="", description="HF username")
    namespace: str = Field(default="", description="Default namespace (org or username)")
    token_configured: bool = Field(default=False, description="Whether a token is configured")
    token_source: str = Field(default="", description="Where token comes from: 'env', 'stored', or ''")


class HFConfigureRequest(BaseModel):
    """Request to configure HuggingFace token."""
    token: str = Field(description="HuggingFace API token")


class JobSubmitRequest(BaseModel):
    """Request to submit a training job."""
    dataset_repo: str = Field(description="HF repo containing training data")
    output_repo: str = Field(description="HF repo to push trained model")
    hardware: str = Field(default="a10g-small", description="Hardware tier")
    base_model: str = Field(default="Qwen/Qwen2.5-Coder-1.5B-Instruct", description="Base model")
    num_epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2e-5, gt=0)


class JobResponse(BaseModel):
    """Response for job operations."""
    job_id: str
    status: str
    hardware: str
    created_at: str
    logs_url: Optional[str] = None
    error_message: Optional[str] = None


class SpaceCreateRequest(BaseModel):
    """Request to create a Space."""
    model_repo: str = Field(description="HF repo containing the model")
    space_name: str = Field(description="Name for the Space")
    private: bool = Field(default=True)
    gpu_duration: int = Field(default=60, ge=30, le=300, description="GPU duration in seconds")


class SpaceResponse(BaseModel):
    """Response for Space operations."""
    space_name: str
    url: str
    status: str


class DatasetUploadRequest(BaseModel):
    """Request to upload dataset."""
    local_path: str = Field(description="Local path to dataset directory")
    repo_name: str = Field(description="Name for the dataset repo")
    private: bool = Field(default=True)
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    """Response for dataset operations."""
    repo_id: str
    url: str
    train_count: int = 0
    val_count: int = 0


class InferenceGenerateRequest(BaseModel):
    """Request for text generation."""
    model: str = Field(description="Model ID (e.g., meta-llama/Llama-3.3-70B-Instruct)")
    prompt: str = Field(description="Input prompt")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class InferenceGenerateResponse(BaseModel):
    """Response for text generation."""
    text: str
    finish_reason: str = "unknown"
    model: str
    provider: Optional[str] = None


class InferenceEmbedRequest(BaseModel):
    """Request for embeddings."""
    model: str = Field(description="Embedding model ID")
    texts: List[str] = Field(description="Texts to embed")


class InferenceEmbedResponse(BaseModel):
    """Response for embeddings."""
    embeddings: List[List[float]]
    model: str
    dimension: int


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get("/status", response_model=HFStatusResponse)
async def get_hf_status():
    """Get HuggingFace integration status."""
    import os
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.secrets import get_secret

    # Determine token source
    token_source = ""
    token_configured = False
    if os.environ.get("HF_TOKEN"):
        token_source = "env"
        token_configured = True
    elif get_secret("HF_TOKEN"):
        token_source = "stored"
        token_configured = True

    client = get_hf_client()
    return HFStatusResponse(
        enabled=client.is_enabled,
        pro_enabled=client.is_pro,
        username=client.username or "",
        namespace=client.namespace or "",
        token_configured=token_configured,
        token_source=token_source,
    )


@router.post("/configure")
async def configure_hf_token(request: HFConfigureRequest):
    """Configure HuggingFace token."""
    from bashgym.secrets import set_secret
    from bashgym.integrations.huggingface import reset_hf_client, get_hf_client
    from bashgym.config import reload_settings

    if not request.token:
        raise HTTPException(status_code=400, detail="Token is required")

    if not request.token.startswith("hf_"):
        raise HTTPException(status_code=400, detail="Invalid token format. HuggingFace tokens start with 'hf_'")

    # Save the token
    set_secret("HF_TOKEN", request.token)

    # Reset the settings cache so it picks up the new token
    reload_settings()

    # Reset the HF client to pick up the new token
    reset_hf_client()

    # Validate the token by getting the client (pass token directly to avoid any caching issues)
    client = get_hf_client(token=request.token, force_new=True)

    if not client.is_enabled:
        # Token is invalid - remove it
        from bashgym.secrets import delete_secret
        delete_secret("HF_TOKEN")
        reset_hf_client()
        raise HTTPException(status_code=401, detail="Invalid token. Please check your token and try again.")

    return {
        "success": True,
        "username": client.username,
        "pro_enabled": client.is_pro,
    }


@router.delete("/configure")
async def remove_hf_token():
    """Remove stored HuggingFace token."""
    import os
    from bashgym.secrets import delete_secret, get_secret
    from bashgym.integrations.huggingface import reset_hf_client

    # Check if there's an env var (can't remove that)
    if os.environ.get("HF_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="Token is set via HF_TOKEN environment variable. Remove it from your environment to disable."
        )

    # Delete stored secret
    deleted = delete_secret("HF_TOKEN")

    # Reset client
    reset_hf_client()

    return {"success": True, "deleted": deleted}


# =============================================================================
# Jobs Endpoints
# =============================================================================

@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all HF training jobs."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import create_job_runner

    client = get_hf_client()

    try:
        client.require_pro()
        runner = create_job_runner(client)
        jobs = runner.list_jobs()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    return [
        JobResponse(
            job_id=job.job_id,
            status=job.status.value,
            hardware=job.hardware,
            created_at=job.created_at.isoformat() if job.created_at else "",
            logs_url=job.logs_url,
            error_message=job.error_message,
        )
        for job in jobs
    ]


@router.post("/jobs", response_model=JobResponse)
async def submit_job(request: JobSubmitRequest, background_tasks: BackgroundTasks):
    """Submit a new training job to HuggingFace."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import create_job_runner, HFJobConfig

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    runner = create_job_runner(client)

    # Generate a basic training script
    script_content = f'''"""
Auto-generated training script for HuggingFace Jobs.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Load model and tokenizer
model_id = "{request.base_model}"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load dataset
dataset = load_dataset("{request.dataset_repo}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs={request.num_epochs},
    learning_rate={request.learning_rate},
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="{request.output_repo}",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=training_args,
)

trainer.train()
trainer.push_to_hub()
'''

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        job = runner.submit_training_job(
            script_path=script_path,
            repo_id=request.output_repo,
            config=HFJobConfig(
                hardware=request.hardware,
                timeout_minutes=120,
            ),
        )

        return JobResponse(
            job_id=job.job_id,
            status=job.status.value,
            hardware=job.hardware,
            created_at=job.created_at.isoformat() if job.created_at else "",
            logs_url=job.logs_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {e}")
    finally:
        import os
        os.unlink(script_path)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import create_job_runner

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    runner = create_job_runner(client)

    try:
        job = runner.get_job_status(job_id)
        return JobResponse(
            job_id=job.job_id,
            status=job.status.value,
            hardware=job.hardware,
            created_at=job.created_at.isoformat() if job.created_at else "",
            logs_url=job.logs_url,
            error_message=job.error_message,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get job logs."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import create_job_runner

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    runner = create_job_runner(client)

    try:
        logs = runner.get_job_logs(job_id)
        return {"logs": logs}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import create_job_runner

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    runner = create_job_runner(client)

    try:
        success = runner.cancel_job(job_id)
        if success:
            return {"status": "cancelled", "job_id": job_id}
        else:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled (may already be completed)")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


# =============================================================================
# Inference Endpoints
# =============================================================================

@router.post("/inference/generate", response_model=InferenceGenerateResponse)
async def generate_text(request: InferenceGenerateRequest):
    """Generate text via HF Inference Providers."""
    from bashgym.integrations.huggingface import get_hf_client, HFError, HFAuthError, HFQuotaExceededError
    from bashgym.integrations.huggingface.inference import get_inference_client

    client = get_hf_client()
    if not client.is_enabled:
        raise HTTPException(status_code=403, detail="HuggingFace integration not configured. Set HF_TOKEN.")

    inference = get_inference_client()

    try:
        response = inference.generate(
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return InferenceGenerateResponse(
            text=response.text,
            finish_reason=response.finish_reason or "unknown",
            model=response.usage.model if response.usage else request.model,
            provider=response.usage.provider if response.usage else None,
        )
    except HFAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except HFQuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/embed", response_model=InferenceEmbedResponse)
async def embed_texts(request: InferenceEmbedRequest):
    """Generate embeddings via HF Inference Providers."""
    from bashgym.integrations.huggingface import get_hf_client, HFError, HFAuthError, HFQuotaExceededError
    from bashgym.integrations.huggingface.inference import get_inference_client

    client = get_hf_client()
    if not client.is_enabled:
        raise HTTPException(status_code=403, detail="HuggingFace integration not configured. Set HF_TOKEN.")

    inference = get_inference_client()

    try:
        response = inference.embed(
            model=request.model,
            texts=request.texts,
        )

        return InferenceEmbedResponse(
            embeddings=response.embeddings,
            model=response.usage.model if response.usage else request.model,
            dimension=len(response.embeddings[0]) if response.embeddings else 0,
        )
    except HFAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except HFQuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Spaces Endpoints
# =============================================================================

@router.get("/spaces", response_model=List[SpaceResponse])
async def list_spaces():
    """List all Bash Gym Spaces."""
    # TODO: Implement space listing when HF API supports it
    return []


@router.post("/spaces", response_model=SpaceResponse)
async def create_space(request: SpaceCreateRequest):
    """Create a new inference Space."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError, HFError
    from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    manager = HFSpaceManager(client=client)

    try:
        url = manager.create_inference_space(
            model_repo=request.model_repo,
            space_name=request.space_name,
            config=SpaceConfig(
                name=request.space_name,
                private=request.private,
            ),
            gpu_duration=request.gpu_duration,
        )

        return SpaceResponse(
            space_name=request.space_name,
            url=url,
            status="building",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spaces/{space_name}/status", response_model=SpaceResponse)
async def get_space_status(space_name: str):
    """Get Space status."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError, HFError
    from bashgym.integrations.huggingface.spaces import HFSpaceManager

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    manager = HFSpaceManager(client=client)

    try:
        status = manager.get_space_status(space_name)
        return SpaceResponse(
            space_name=space_name,
            url=f"https://huggingface.co/spaces/{space_name}",
            status=status.value,
        )
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/spaces/{space_name}")
async def delete_space(space_name: str):
    """Delete a Space."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError, HFError
    from bashgym.integrations.huggingface.spaces import HFSpaceManager

    client = get_hf_client()

    try:
        client.require_pro()
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    manager = HFSpaceManager(client=client)

    try:
        manager.delete_space(space_name)
        return {"status": "deleted", "space_name": space_name}
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Datasets Endpoints
# =============================================================================

@router.get("/datasets", response_model=List[str])
async def list_datasets(prefix: str = "bashgym"):
    """List Bash Gym datasets."""
    from bashgym.integrations.huggingface import get_hf_client, HFError
    from bashgym.integrations.huggingface.datasets import HFDatasetManager

    client = get_hf_client()
    if not client.is_enabled:
        return []

    manager = HFDatasetManager(client=client)

    try:
        return manager.list_datasets(prefix=prefix)
    except HFError as e:
        logger.error(f"Failed to list datasets: {e}")
        return []


@router.post("/datasets", response_model=DatasetResponse)
async def upload_dataset(request: DatasetUploadRequest):
    """Upload a dataset to HuggingFace Hub."""
    from bashgym.integrations.huggingface import get_hf_client, HFError
    from bashgym.integrations.huggingface.datasets import HFDatasetManager, DatasetConfig

    client = get_hf_client()
    if not client.is_enabled:
        raise HTTPException(status_code=403, detail="HuggingFace integration not configured. Set HF_TOKEN.")

    manager = HFDatasetManager(client=client)

    local_path = Path(request.local_path)
    if not local_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {request.local_path}")

    try:
        url = manager.upload_training_data(
            local_path=local_path,
            repo_name=request.repo_name,
            config=DatasetConfig(
                repo_name=request.repo_name,
                private=request.private,
            ),
            metadata=request.metadata,
        )

        # Count examples
        train_count = 0
        val_count = 0
        train_file = local_path / "train.jsonl"
        val_file = local_path / "val.jsonl"

        if train_file.exists():
            train_count = sum(1 for _ in train_file.open())
        if val_file.exists():
            val_count = sum(1 for _ in val_file.open())

        return DatasetResponse(
            repo_id=f"{client.namespace}/{request.repo_name}",
            url=url,
            train_count=train_count,
            val_count=val_count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{repo_name}")
async def delete_dataset(repo_name: str):
    """Delete a dataset."""
    from bashgym.integrations.huggingface import get_hf_client, HFError
    from bashgym.integrations.huggingface.datasets import HFDatasetManager

    client = get_hf_client()
    if not client.is_enabled:
        raise HTTPException(status_code=403, detail="HuggingFace integration not configured. Set HF_TOKEN.")

    manager = HFDatasetManager(client=client)

    try:
        success = manager.delete_dataset(repo_name)
        if success:
            return {"status": "deleted", "repo_name": repo_name}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")
    except HFError as e:
        raise HTTPException(status_code=500, detail=str(e))
