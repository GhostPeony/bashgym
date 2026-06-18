"""HuggingFace Storage Buckets integration.

Wraps the huggingface_hub bucket API (v1.10+) for creating, listing,
browsing, syncing, and deleting storage buckets. Buckets are mutable,
non-versioned, S3-like object storage — designed for training checkpoints,
logs, and processed data shards that change frequently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, copy_files

logger = logging.getLogger(__name__)


class BucketManager:
    """Manages HuggingFace Storage Buckets."""

    def __init__(self, token: str | None = None):
        self.api = HfApi(token=token)

    def create(
        self,
        bucket_id: str,
        private: bool = True,
        exist_ok: bool = True,
    ) -> dict[str, Any]:
        """Create a storage bucket. Returns bucket URL info."""
        result = self.api.create_bucket(
            bucket_id=bucket_id,
            private=private,
            exist_ok=exist_ok,
        )
        logger.info("Created bucket: %s", bucket_id)
        return {"bucket_id": bucket_id, "url": str(result)}

    def list_buckets(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """List all buckets for the authenticated user or a namespace."""
        buckets = []
        for b in self.api.list_buckets(namespace=namespace):
            buckets.append(
                {
                    "id": b.id,
                    "private": b.private,
                    "created_at": str(getattr(b, "created_at", None)),
                    "updated_at": str(
                        getattr(b, "last_modified", None) or getattr(b, "updated_at", None)
                    ),
                }
            )
        return buckets

    def list_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """List files and folders in a bucket."""
        items = []
        for item in self.api.list_bucket_tree(
            bucket_id=bucket_id,
            prefix=prefix,
            recursive=recursive,
        ):
            entry = {"name": item.path, "type": item.type}
            if hasattr(item, "size"):
                entry["size"] = item.size
            if hasattr(item, "last_modified"):
                entry["last_modified"] = str(item.last_modified)
            items.append(entry)
        return items

    def info(self, bucket_id: str) -> dict[str, Any]:
        """Get bucket metadata."""
        b = self.api.bucket_info(bucket_id=bucket_id)
        return {
            "id": b.id,
            "private": b.private,
            "created_at": str(getattr(b, "created_at", None)),
            "updated_at": str(getattr(b, "last_modified", None) or getattr(b, "updated_at", None)),
        }

    def sync(
        self,
        source: str,
        dest: str,
        delete: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Sync files between local dir ↔ bucket or bucket ↔ bucket.

        source/dest can be local paths or hf:// URIs like
        'hf://buckets/user/bucket-name/subdir/'.
        """
        result = self.api.sync_bucket(
            source=source,
            dest=dest,
            delete=delete,
            dry_run=dry_run,
        )
        logger.info("Synced %s → %s (dry_run=%s)", source, dest, dry_run)
        return {
            "source": source,
            "dest": dest,
            "dry_run": dry_run,
            "plan": str(result) if result else None,
        }

    def delete(self, bucket_id: str, missing_ok: bool = True) -> None:
        """Delete a storage bucket."""
        self.api.delete_bucket(bucket_id=bucket_id, missing_ok=missing_ok)
        logger.info("Deleted bucket: %s", bucket_id)

    def download_files(
        self,
        bucket_id: str,
        files: list[tuple[str, str]],
    ) -> None:
        """Download specific files from a bucket.

        files: list of (remote_path, local_path) tuples.
        """
        self.api.download_bucket_files(
            bucket_id=bucket_id,
            files=[(src, Path(dst)) for src, dst in files],
        )

    @staticmethod
    def copy(source_uri: str, dest_uri: str, token: str | None = None) -> dict[str, Any]:
        """Server-side instant copy between buckets/repos (zero bandwidth for Xet files).

        URIs: 'hf://buckets/user/bucket/path' or 'hf://datasets/user/repo/path'
        Note: bucket-to-repo copy is not yet supported by HF (April 2026).
        """
        copy_files(source=source_uri, destination=dest_uri, token=token)
        logger.info("Copied %s → %s", source_uri, dest_uri)
        return {"source": source_uri, "destination": dest_uri, "status": "completed"}
