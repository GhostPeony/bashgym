"""Upload bashgym agent traces to HuggingFace Hub as datasets.

HF Hub (April 2026) auto-detects agent trace formats (Claude Code, Codex, Pi)
and provides a specialized trace viewer. This module packages gold/failed
traces from ~/.bashgym/ into JSONL and uploads them as HF datasets.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


class TraceUploader:
    """Packages and uploads bashgym traces to HuggingFace Hub."""

    def __init__(self, token: str | None = None):
        self.api = HfApi(token=token)

    def upload_traces(
        self,
        trace_dir: Path,
        repo_id: str,
        private: bool = True,
        split_name: str = "train",
    ) -> dict[str, Any]:
        """Upload all .json trace files from trace_dir as an HF dataset.

        The traces are concatenated into a single JSONL file and uploaded
        with HF's auto-detection for agent traces enabled.

        Args:
            trace_dir: Path to a directory of .json trace files
            repo_id: HF dataset repo (e.g. 'user/bashgym-gold-traces')
            private: Whether the dataset should be private
            split_name: Name of the split (default: 'train')

        Returns:
            Dict with repo_id, url, num_traces, total_size_bytes
        """
        trace_files = sorted(trace_dir.glob("*.json"))
        if not trace_files:
            raise ValueError(f"No .json files found in {trace_dir}")

        self.api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

        total_size = 0
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="traces_"
        ) as f:
            for tf in trace_files:
                try:
                    data = json.loads(tf.read_text())
                    if isinstance(data, dict):
                        data["_source_file"] = tf.name
                    f.write(json.dumps(data) + "\n")
                    total_size += tf.stat().st_size
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Skipping %s: %s", tf.name, e)
            jsonl_path = f.name

        self.api.upload_file(
            path_or_fileobj=jsonl_path,
            path_in_repo=f"{split_name}.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Clean up temp file
        Path(jsonl_path).unlink(missing_ok=True)

        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(
            "Uploaded %d traces (%d bytes) to %s",
            len(trace_files), total_size, url,
        )

        return {
            "repo_id": repo_id,
            "url": url,
            "num_traces": len(trace_files),
            "total_size_bytes": total_size,
            "split": split_name,
        }

    def list_trace_datasets(self, prefix: str = "bashgym") -> list[dict[str, Any]]:
        """List the authenticated user's trace datasets on HF Hub."""
        try:
            whoami = self.api.whoami()
            username = whoami.get("name", "")
        except Exception:
            return []

        datasets = []
        for ds in self.api.list_datasets(author=username, search=prefix):
            datasets.append({
                "id": ds.id,
                "private": getattr(ds, "private", None),
                "downloads": getattr(ds, "downloads", 0),
                "last_modified": str(getattr(ds, "lastModified", None) or getattr(ds, "last_modified", None)),
            })
        return datasets
