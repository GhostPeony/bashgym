"""
HuggingFace Spaces (ZeroGPU) Manager

Provides deployment capabilities for trained models to HuggingFace Spaces
with ZeroGPU support. Requires HuggingFace Pro subscription.

Features:
- Create ZeroGPU Spaces with auto-generated Gradio apps
- Monitor Space status (building, running, stopped, error)
- Update models in existing Spaces
- Delete Spaces

Usage:
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig

    client = get_hf_client()
    manager = HFSpaceManager(client)

    # Create a new Space
    config = SpaceConfig(name="my-inference-space")
    url = manager.create_inference_space(
        model_repo="myorg/my-model",
        space_name="my-inference-space",
        config=config,
        gpu_duration=60
    )

    # Check status
    status = manager.get_space_status("myorg/my-inference-space")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

from .client import (
    HuggingFaceClient,
    HFProRequiredError,
    HF_HUB_AVAILABLE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class SpaceStatus(Enum):
    """Status of a HuggingFace Space."""
    BUILDING = "building"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpaceConfig:
    """Configuration for a HuggingFace Space."""

    name: str
    """Name of the Space."""

    hardware: str = "zero-gpu"
    """Hardware configuration. Use 'zero-gpu' for ZeroGPU."""

    private: bool = True
    """Whether the Space should be private."""

    sdk: str = "gradio"
    """SDK to use (gradio or streamlit)."""

    python_version: str = "3.10"
    """Python version for the Space."""

    dev_mode: bool = False
    """Enable development mode with SSH access."""

    def validate(self) -> List[str]:
        """Validate the configuration."""
        errors = []

        if not self.name:
            errors.append("name is required")

        valid_sdks = ("gradio", "streamlit", "docker", "static")
        if self.sdk not in valid_sdks:
            errors.append(f"Invalid sdk '{self.sdk}'. Valid options: {', '.join(valid_sdks)}")

        return errors


@dataclass
class SSHCredentials:
    """SSH credentials for Space dev mode."""

    host: str
    """SSH host address."""

    port: int
    """SSH port."""

    username: str
    """SSH username."""

    key: str
    """SSH private key."""


# =============================================================================
# Gradio App Template
# =============================================================================

GRADIO_APP_TEMPLATE = '''"""
Auto-generated Gradio app for model inference.
Deployed to HuggingFace Spaces with ZeroGPU support.
"""

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "{model_id}"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = None

def load_model():
    global model
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model

@spaces.GPU(duration={gpu_duration})
def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    """Generate text from the model."""
    model = load_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response

# Create Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3),
        gr.Slider(minimum=1, maximum=2048, value=256, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Model Inference",
    description=f"Generate text using {{MODEL_ID}}",
    examples=[
        ["Write a short story about a robot:", 256, 0.7],
        ["Explain quantum computing:", 512, 0.5],
    ],
)

if __name__ == "__main__":
    demo.launch()
'''


# =============================================================================
# Space Manager
# =============================================================================

class HFSpaceManager:
    """
    Manages HuggingFace Spaces for model deployment.

    Requires HuggingFace Pro subscription for ZeroGPU hardware access.

    Example:
        client = get_hf_client()
        manager = HFSpaceManager(client)

        # Create Space with ZeroGPU
        url = manager.create_inference_space(
            model_repo="myorg/my-model",
            space_name="my-space",
            config=SpaceConfig(name="my-space"),
            gpu_duration=60
        )

        # Check status
        status = manager.get_space_status("myorg/my-space")

        # Update model
        manager.update_space_model("myorg/my-space", "myorg/new-model")

        # Delete Space
        manager.delete_space("myorg/my-space")
    """

    def __init__(
        self,
        client: Optional[HuggingFaceClient] = None,
        token: Optional[str] = None,
        pro_enabled: bool = False,
    ):
        """
        Initialize the Space manager.

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

    def create_inference_space(
        self,
        model_repo: str,
        space_name: str,
        config: Optional[SpaceConfig] = None,
        gpu_duration: int = 60,
    ) -> str:
        """
        Create a ZeroGPU Space with an auto-generated Gradio app.

        Args:
            model_repo: Repository ID of the model to deploy (e.g., "myorg/my-model").
            space_name: Name for the Space (without namespace).
            config: Space configuration. If None, uses defaults.
            gpu_duration: GPU duration in seconds for @spaces.GPU decorator.

        Returns:
            URL of the created Space.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
            ValueError: If configuration is invalid.
        """
        self._require_pro("Creating ZeroGPU Spaces")

        # Use default config if none provided
        if config is None:
            config = SpaceConfig(name=space_name)

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid Space configuration: {'; '.join(errors)}")

        # Construct full repo ID
        namespace = self._client.namespace
        repo_id = f"{namespace}/{space_name}" if namespace else space_name

        logger.info(f"Creating ZeroGPU Space: {repo_id}")
        logger.info(f"  Model: {model_repo}")
        logger.info(f"  Hardware: {config.hardware}")
        logger.info(f"  GPU Duration: {gpu_duration}s")

        # Generate the Gradio app code
        app_code = GRADIO_APP_TEMPLATE.format(
            model_id=model_repo,
            gpu_duration=gpu_duration,
        )

        # Generate requirements.txt
        requirements = """transformers>=4.40.0
torch>=2.0.0
accelerate>=0.25.0
gradio>=4.0.0
spaces
sentencepiece
"""

        # Create the Space using HF API
        api = self._client.api
        if api is not None and HF_HUB_AVAILABLE:
            try:
                # Create the Space repository
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="space",
                    space_sdk=config.sdk,
                    space_hardware=config.hardware if config.hardware != "zero-gpu" else None,
                    private=config.private,
                    exist_ok=True,
                )

                # Upload app.py
                api.upload_file(
                    path_or_fileobj=app_code.encode("utf-8"),
                    path_in_repo="app.py",
                    repo_id=repo_id,
                    repo_type="space",
                )

                # Upload requirements.txt
                api.upload_file(
                    path_or_fileobj=requirements.encode("utf-8"),
                    path_in_repo="requirements.txt",
                    repo_id=repo_id,
                    repo_type="space",
                )

                # If using ZeroGPU, we need to enable it via Space settings
                # The hardware config for ZeroGPU is set differently
                if config.hardware == "zero-gpu":
                    try:
                        api.request_space_hardware(repo_id=repo_id, hardware="zero-a10g")
                    except Exception as e:
                        logger.warning(f"Could not request ZeroGPU hardware: {e}")

                logger.info(f"Space created successfully: {repo_id}")
                return f"https://huggingface.co/spaces/{repo_id}"

            except Exception as e:
                logger.error(f"Failed to create Space: {e}")
                raise
        else:
            # Simulation mode
            logger.info(f"Simulating Space creation for {repo_id}")
            return f"https://huggingface.co/spaces/{repo_id}"

    def get_space_status(self, space_name: str) -> SpaceStatus:
        """
        Get the current status of a Space.

        Args:
            space_name: Full Space repo ID (e.g., "myorg/my-space").

        Returns:
            SpaceStatus enum value.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
        """
        self._require_pro("Checking Space status")

        api = self._client.api
        if api is not None and HF_HUB_AVAILABLE:
            try:
                runtime = api.get_space_runtime(repo_id=space_name)
                stage = getattr(runtime, "stage", "").upper()

                # Map HF stage to our SpaceStatus
                status_map = {
                    "BUILDING": SpaceStatus.BUILDING,
                    "RUNNING": SpaceStatus.RUNNING,
                    "STOPPED": SpaceStatus.STOPPED,
                    "PAUSED": SpaceStatus.STOPPED,
                    "ERROR": SpaceStatus.ERROR,
                    "RUNTIME_ERROR": SpaceStatus.ERROR,
                    "BUILD_ERROR": SpaceStatus.ERROR,
                }

                return status_map.get(stage, SpaceStatus.BUILDING)

            except Exception as e:
                logger.error(f"Failed to get Space status: {e}")
                return SpaceStatus.ERROR
        else:
            # Simulation mode
            return SpaceStatus.RUNNING

    def delete_space(self, space_name: str) -> None:
        """
        Delete a Space.

        Args:
            space_name: Full Space repo ID (e.g., "myorg/my-space").

        Raises:
            HFProRequiredError: If Pro subscription is not available.
        """
        self._require_pro("Deleting Spaces")

        logger.info(f"Deleting Space: {space_name}")

        api = self._client.api
        if api is not None and HF_HUB_AVAILABLE:
            try:
                api.delete_repo(repo_id=space_name, repo_type="space")
                logger.info(f"Space deleted: {space_name}")
            except Exception as e:
                logger.error(f"Failed to delete Space: {e}")
                raise
        else:
            # Simulation mode
            logger.info(f"Simulating Space deletion for {space_name}")

    def update_space_model(self, space_name: str, new_model_repo: str) -> None:
        """
        Update the model in an existing Space.

        Args:
            space_name: Full Space repo ID (e.g., "myorg/my-space").
            new_model_repo: Repository ID of the new model.

        Raises:
            HFProRequiredError: If Pro subscription is not available.
        """
        self._require_pro("Updating Space model")

        logger.info(f"Updating Space {space_name} with model {new_model_repo}")

        api = self._client.api
        if api is not None and HF_HUB_AVAILABLE:
            try:
                # Download existing app.py, update MODEL_ID, and re-upload
                # For simplicity, we'll regenerate the app.py with the new model
                app_code = GRADIO_APP_TEMPLATE.format(
                    model_id=new_model_repo,
                    gpu_duration=60,  # Use default duration
                )

                api.upload_file(
                    path_or_fileobj=app_code.encode("utf-8"),
                    path_in_repo="app.py",
                    repo_id=space_name,
                    repo_type="space",
                )

                # Restart the Space to pick up changes
                api.restart_space(repo_id=space_name)

                logger.info(f"Space model updated: {space_name} -> {new_model_repo}")

            except Exception as e:
                logger.error(f"Failed to update Space model: {e}")
                raise
        else:
            # Simulation mode
            logger.info(f"Simulating Space model update for {space_name}")

    def __repr__(self) -> str:
        """String representation."""
        pro_str = " [Pro]" if self.is_pro else ""
        return f"<HFSpaceManager{pro_str}>"
