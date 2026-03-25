"""
HuggingFace Model Manager

Push trained models to HuggingFace Hub, manage model repos,
and generate model cards from training metadata.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import (
    HF_HUB_AVAILABLE,
    HFError,
    HuggingFaceClient,
)

logger = logging.getLogger(__name__)


@dataclass
class HFModelInfo:
    """Information about a model on HuggingFace Hub."""

    id: str
    url: str = ""
    downloads: int = 0
    likes: int = 0
    private: bool = False
    last_modified: str = ""
    pipeline_tag: str | None = None
    tags: list[str] = field(default_factory=list)


MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
tags:
{tags_yaml}
---

# {display_name}

{description}

## Training Details

| Parameter | Value |
|-----------|-------|
| Strategy | {strategy} |
| Base Model | {base_model} |
| Training Duration | {duration} |
| Final Loss | {final_loss} |
{eval_loss_row}| Traces Used | {trace_count} |
| Training Repos | {repos} |

## Training Configuration

| Parameter | Value |
|-----------|-------|
{config_rows}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

{gguf_section}

---

*Trained with [BashGym](https://github.com/GhostPeony/bashgym) \u2014 A Self-Improving Agentic Development Gym*
"""


class HFModelManager:
    """Manage trained models on HuggingFace Hub."""

    def __init__(self, client: HuggingFaceClient):
        self._client = client

    @property
    def client(self) -> HuggingFaceClient:
        return self._client

    def push_model(
        self,
        local_path: Path,
        repo_name: str,
        private: bool = True,
        commit_message: str = "Push from BashGym",
    ) -> str:
        """Push a model directory to HuggingFace Hub.

        Returns the repo URL.
        """
        self.client.require_enabled()

        if not HF_HUB_AVAILABLE:
            raise HFError("huggingface_hub is not installed")

        repo_id = self.client.get_repo_id(repo_name)
        api = self.client.api

        # Create repo (ok if exists)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        # Upload folder
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.py", "*.sh", "checkpoint-*", "*.log", "run_state.json"],
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Pushed model to {url}")
        return url

    def push_gguf(
        self,
        gguf_path: Path,
        repo_name: str,
        private: bool = True,
    ) -> str:
        """Push a GGUF file to a separate -GGUF repo (HF convention).

        Returns the repo URL.
        """
        self.client.require_enabled()

        if not HF_HUB_AVAILABLE:
            raise HFError("huggingface_hub is not installed")

        # Convention: GGUF goes in {repo}-GGUF
        gguf_repo_name = f"{repo_name}-GGUF" if not repo_name.endswith("-GGUF") else repo_name
        repo_id = self.client.get_repo_id(gguf_repo_name)
        api = self.client.api

        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {gguf_path.name}",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Pushed GGUF to {url}")
        return url

    def list_my_models(self, limit: int = 50) -> list[HFModelInfo]:
        """List models owned by the authenticated user."""
        self.client.require_enabled()

        if not HF_HUB_AVAILABLE:
            raise HFError("huggingface_hub is not installed")

        api = self.client.api
        username = self.client.username
        models = []

        try:
            for model in api.list_models(author=username, limit=limit):
                models.append(
                    HFModelInfo(
                        id=model.id,
                        url=f"https://huggingface.co/{model.id}",
                        downloads=getattr(model, "downloads", 0) or 0,
                        likes=getattr(model, "likes", 0) or 0,
                        private=getattr(model, "private", False),
                        last_modified=str(getattr(model, "last_modified", "")),
                        pipeline_tag=getattr(model, "pipeline_tag", None),
                        tags=list(getattr(model, "tags", []) or []),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")

        return models

    def generate_model_card(
        self,
        repo_id: str,
        profile_data: dict[str, Any],
    ) -> str:
        """Generate a model card README from training profile data."""
        base_model = profile_data.get("base_model", "unknown")
        strategy = profile_data.get("training_strategy", "sft").upper()
        display_name = profile_data.get("display_name", repo_id.split("/")[-1])
        description = profile_data.get("description", f"Fine-tuned {base_model} with {strategy}")

        # Tags
        tags = ["fine-tuned", "bashgym", strategy.lower()]
        if "code" in base_model.lower() or "coder" in base_model.lower():
            tags.append("code-generation")
        tags_yaml = "\n".join(f"  - {t}" for t in tags)

        # Metrics
        metrics = profile_data.get("final_metrics", {})
        final_loss = f"{metrics.get('final_loss', 'N/A')}"
        eval_loss = metrics.get("eval_loss")
        eval_loss_row = f"| Eval Loss | {eval_loss} |\n" if eval_loss is not None else ""

        # Duration
        duration_seconds = profile_data.get("duration_seconds", 0)
        if duration_seconds > 0:
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            duration = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        else:
            duration = "N/A"

        # Training info
        traces = profile_data.get("training_traces", [])
        trace_count = str(len(traces)) if traces else "N/A"
        repos = ", ".join(profile_data.get("training_repos", [])) or "N/A"

        # Config table
        config = profile_data.get("config", {})
        config_keys = [
            "learning_rate",
            "batch_size",
            "num_epochs",
            "max_seq_length",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "gradient_accumulation_steps",
        ]
        config_rows = ""
        for key in config_keys:
            if key in config:
                label = key.replace("_", " ").title()
                config_rows += f"| {label} | {config[key]} |\n"

        # GGUF section
        gguf_section = ""
        artifacts = profile_data.get("artifacts", {})
        gguf_exports = artifacts.get("gguf_exports", [])
        if gguf_exports:
            gguf_repo = f"{repo_id}-GGUF"
            gguf_section = f"""## GGUF Quantizations

Available on [{gguf_repo}](https://huggingface.co/{gguf_repo}):

```bash
ollama run {repo_id.split('/')[-1].lower()}
```
"""

        return MODEL_CARD_TEMPLATE.format(
            base_model=base_model,
            display_name=display_name,
            description=description,
            strategy=strategy,
            duration=duration,
            final_loss=final_loss,
            eval_loss_row=eval_loss_row,
            trace_count=trace_count,
            repos=repos,
            config_rows=config_rows,
            tags_yaml=tags_yaml,
            repo_id=repo_id,
            gguf_section=gguf_section,
        )

    def push_model_card(self, repo_id: str, card_content: str) -> None:
        """Upload a model card README to a repo."""
        self.client.require_enabled()

        if not HF_HUB_AVAILABLE:
            raise HFError("huggingface_hub is not installed")

        self.client.api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card",
        )

    def delete_model(self, repo_id: str) -> bool:
        """Delete a model repo from Hub."""
        self.client.require_enabled()

        if not HF_HUB_AVAILABLE:
            raise HFError("huggingface_hub is not installed")

        try:
            self.client.api.delete_repo(repo_id=repo_id, repo_type="model")
            logger.info(f"Deleted model repo: {repo_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete model {repo_id}: {e}")
            return False


# Singleton
_model_manager: HFModelManager | None = None


def get_model_manager(client: HuggingFaceClient | None = None) -> HFModelManager:
    """Get or create the global HFModelManager."""
    global _model_manager
    if _model_manager is None:
        if client is None:
            from .client import get_hf_client

            client = get_hf_client()
        _model_manager = HFModelManager(client)
    return _model_manager
