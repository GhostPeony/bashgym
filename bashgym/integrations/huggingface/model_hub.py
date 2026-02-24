"""
HuggingFace Model Hub integration.

Wraps huggingface_hub.HfApi for model search and card management.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Simplified model info returned from HF Hub search."""
    id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None
    last_modified: Optional[str] = None
    private: bool = False
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None


class HFModelHub:
    """Wraps HuggingFace Hub API for model discovery and card management."""

    def __init__(self, token: Optional[str] = None):
        self.token = token

    def _get_api(self):
        try:
            from huggingface_hub import HfApi
            return HfApi(token=self.token)
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    async def search_models(
        self,
        task: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 10,
        framework: Optional[str] = None,
    ) -> List[ModelInfo]:
        """Search HuggingFace Hub for models.

        Args:
            task: Pipeline task filter (e.g. 'text-generation', 'code-generation')
            sort: Sort order — 'downloads', 'likes', or 'lastModified'
            limit: Max number of results
            framework: Framework filter (e.g. 'pytorch', 'transformers')

        Returns:
            List of ModelInfo objects
        """
        import asyncio

        def _search():
            api = self._get_api()
            kwargs: Dict[str, Any] = {
                "sort": sort,
                "limit": limit,
                "full": False,
            }
            if task:
                kwargs["filter"] = task
            if framework:
                kwargs["library"] = framework

            try:
                models = list(api.list_models(**kwargs))
            except Exception as e:
                logger.error(f"HF model search failed: {e}")
                return []

            results = []
            for m in models:
                results.append(ModelInfo(
                    id=getattr(m, "modelId", "") or getattr(m, "id", ""),
                    downloads=getattr(m, "downloads", 0) or 0,
                    likes=getattr(m, "likes", 0) or 0,
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    last_modified=str(getattr(m, "lastModified", "") or ""),
                    private=getattr(m, "private", False),
                    tags=list(getattr(m, "tags", []) or []),
                    author=getattr(m, "author", None),
                ))
            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info for a specific model.

        Args:
            model_id: HuggingFace model ID (e.g. 'meta-llama/Llama-3.3-70B-Instruct')

        Returns:
            ModelInfo or None if not found
        """
        import asyncio

        def _get():
            api = self._get_api()
            try:
                m = api.model_info(model_id)
                return ModelInfo(
                    id=getattr(m, "modelId", model_id),
                    downloads=getattr(m, "downloads", 0) or 0,
                    likes=getattr(m, "likes", 0) or 0,
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    last_modified=str(getattr(m, "lastModified", "") or ""),
                    private=getattr(m, "private", False),
                    tags=list(getattr(m, "tags", []) or []),
                    author=getattr(m, "author", None),
                )
            except Exception as e:
                logger.error(f"Failed to get model info for {model_id}: {e}")
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def update_model_card(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Update a model card's metadata.

        Args:
            model_id: HuggingFace model ID
            metadata: Metadata dict to merge into the model card
        """
        import asyncio

        def _update():
            try:
                from huggingface_hub import metadata_update
                metadata_update(model_id, metadata, token=self.token)
            except Exception as e:
                logger.error(f"Failed to update model card for {model_id}: {e}")
                raise

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)


def get_model_hub(token: Optional[str] = None) -> HFModelHub:
    """Get an HFModelHub instance, optionally with a token."""
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from bashgym.secrets import get_secret
                token = get_secret("HF_TOKEN")
            except Exception:
                pass
    return HFModelHub(token=token)
