"""Thin wrapper around huggingface_hub for dataset discovery and enrichment.

Verified against huggingface_hub 1.8.0 on 2026-04-10. Key findings:

- ``HfApi().list_datasets(filter=..., search=..., limit=..., full=..., sort=...)``
  returns an iterator of ``DatasetInfo``. ``search="bash agent"`` returns zero
  results; narrower keyword combos (``"swe-bench"``, ``"code generation"``) work.
  Filter tokens like ``"code-generation"`` do work.
- ``HfApi().dataset_info(repo_id)`` returns a ``DatasetInfo`` with both
  ``cardData`` and ``card_data`` attributes pointing at the same
  ``DatasetCardData`` instance (attribute access + ``card["license"]`` + ``.to_dict()``).
- ``card.dataset_info`` is a plain dict with keys ``features``, ``splits``,
  ``download_size``, ``dataset_size``. ``features`` is a list of dicts like
  ``{"name": "prompt", "dtype": "string"}`` (sometimes ``{"list": [...]}`` for
  nested structs). ``splits`` is a list of dicts with ``num_examples``.
- ``info.lastModified`` is a datetime. ``info.gated`` is a bool. ``info.tags``
  contains structured strings like ``"license:mit"`` — a useful fallback when
  ``card.license`` is None.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
from huggingface_hub import HfApi

from bashgym.research.scoring import DatasetMetadata

logger = logging.getLogger(__name__)

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/info"

# Curated search queries. Each produces up to ``limit`` results; the caller
# dedupes by repo_id. Tuned based on the 2026-04-10 probe — queries that
# returned zero results are excluded.
SEARCH_QUERIES: list[dict] = [
    {"filter": "code-generation", "limit": 200},
    {"search": "code generation", "limit": 100},
    {"search": "swe-bench", "limit": 50},
    {"search": "humaneval", "limit": 50},
    {"search": "mbpp", "limit": 50},
    {"search": "tool use", "limit": 100},
    {"search": "coding agent", "limit": 100},
    {"search": "python code", "limit": 100},
    {"search": "bash", "limit": 50},
    {"search": "shell command", "limit": 50},
]


def _extract_license_from_tags(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def _extract_features(dataset_info_dict: dict[str, Any]) -> dict[str, str]:
    """Flatten a dataset_info['features'] value into {name: dtype}.

    Handles both shapes:
    - ``card_data.dataset_info`` (YAML): ``features`` is a list of
      ``{"name": ..., "dtype": ...}`` dicts.
    - ``datasets-server /info`` endpoint: ``features`` is a dict
      ``{name: {"dtype": ..., ...}}``.
    """
    features: dict[str, str] = {}
    feats = dataset_info_dict.get("features")
    if isinstance(feats, list):
        for f in feats:
            if not isinstance(f, dict) or "name" not in f:
                continue
            name = f["name"]
            if "dtype" in f:
                features[name] = str(f["dtype"])
            elif "list" in f:
                features[name] = "list"
            elif "sequence" in f:
                features[name] = "sequence"
            else:
                features[name] = "unknown"
    elif isinstance(feats, dict):
        for name, spec in feats.items():
            if isinstance(spec, dict):
                features[name] = str(spec.get("dtype") or spec.get("_type") or "unknown")
            elif isinstance(spec, list):
                features[name] = "list"
            else:
                features[name] = str(type(spec).__name__)
    return features


def _extract_num_rows(dataset_info_dict: dict[str, Any]) -> int | None:
    """Sum num_examples across splits. Handles list and dict shapes."""
    splits = dataset_info_dict.get("splits")
    total = 0
    saw_any = False
    if isinstance(splits, list):
        for s in splits:
            if isinstance(s, dict):
                n = s.get("num_examples")
                if isinstance(n, int):
                    total += n
                    saw_any = True
    elif isinstance(splits, dict):
        for s in splits.values():
            if isinstance(s, dict):
                n = s.get("num_examples")
                if isinstance(n, int):
                    total += n
                    saw_any = True
    return total if saw_any else None


def _fetch_datasets_server_info(repo_id: str, token: str | None = None) -> dict[str, Any] | None:
    """Fall back to HF datasets-server for schema + row counts.

    Returns the first config's dataset_info dict, or None on any failure.
    This endpoint powers HF's dataset viewer and is populated for most
    public datasets that have been auto-converted to parquet.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(
            DATASETS_SERVER_URL,
            params={"dataset": repo_id},
            headers=headers,
            timeout=20.0,
        )
        if r.status_code != 200:
            return None
        payload = r.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("datasets-server lookup failed for %s: %s", repo_id, e)
        return None

    configs = payload.get("dataset_info")
    if not isinstance(configs, dict) or not configs:
        return None
    # Prefer 'default' config if present, otherwise the first one.
    if "default" in configs:
        return configs["default"] if isinstance(configs["default"], dict) else None
    first = next(iter(configs.values()))
    return first if isinstance(first, dict) else None


class HFResearchClient:
    """Wraps HfApi for dataset discovery. All network calls go through here."""

    def __init__(self, token: str | None = None):
        self.api = HfApi(token=token)

    def discover_candidates(self, queries: list[dict] | None = None) -> list[str]:
        """Run all search queries; return deduped sorted list of repo_ids."""
        queries = queries or SEARCH_QUERIES
        seen: set[str] = set()
        for q in queries:
            try:
                for ds in self.api.list_datasets(**q, full=False):
                    rid = getattr(ds, "id", None)
                    if rid:
                        seen.add(rid)
            except Exception as e:
                logger.warning("discover query %s failed: %s", q, e)
        logger.info("discovery found %d unique datasets across %d queries", len(seen), len(queries))
        return sorted(seen)

    def enrich(self, repo_id: str) -> DatasetMetadata | None:
        """Fetch full metadata for one dataset. Returns None on failure."""
        try:
            info = self.api.dataset_info(repo_id)
        except Exception as e:
            logger.warning("dataset_info(%s) failed: %s", repo_id, e)
            return None

        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)

        license: str | None = None
        card_dataset_info: dict[str, Any] = {}
        if card is not None:
            license = getattr(card, "license", None)
            if isinstance(license, list):
                license = license[0] if license else None
            raw_ds_info = getattr(card, "dataset_info", None)
            if isinstance(raw_ds_info, dict):
                card_dataset_info = raw_ds_info
            elif isinstance(raw_ds_info, list) and raw_ds_info:
                first = raw_ds_info[0]
                if isinstance(first, dict):
                    card_dataset_info = first

        tags = list(getattr(info, "tags", []) or [])
        if license is None:
            license = _extract_license_from_tags(tags)

        num_rows = _extract_num_rows(card_dataset_info)
        download_size = card_dataset_info.get("download_size") if card_dataset_info else None
        features = _extract_features(card_dataset_info)

        # Fall back to HF datasets-server when the card lacks the info we need.
        # This is the same API that powers the HF dataset viewer; it's populated
        # for most public datasets that have been auto-converted to parquet.
        if num_rows is None or not features:
            logger.debug("falling back to datasets-server for %s", repo_id)
            server_info = _fetch_datasets_server_info(repo_id, token=self.api.token)
            if server_info is not None:
                if num_rows is None:
                    num_rows = _extract_num_rows(server_info)
                if not features:
                    features = _extract_features(server_info)
                if download_size is None:
                    download_size = server_info.get("download_size")

        last_modified = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
        if isinstance(last_modified, str):
            try:
                last_modified = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            except ValueError:
                last_modified = None

        downloads = int(getattr(info, "downloads", 0) or 0)
        gated = bool(getattr(info, "gated", False))
        description = str(getattr(info, "description", "") or "")[:500]

        return DatasetMetadata(
            repo_id=repo_id,
            tags=tags,
            license=license,
            num_rows=num_rows,
            download_size_bytes=download_size,
            features=features,
            last_modified=last_modified,
            downloads=downloads,
            gated=gated,
            description=description,
        )
