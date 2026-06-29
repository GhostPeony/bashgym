"""Curated public source catalog for BashGym training and evaluation."""

from typing import Any

from bashgym.sources.catalog import (
    SourceArtifactType,
    SourceCard,
    SourceRisk,
    SourceUse,
    get_source,
    list_sources,
    prepare_source_manifest,
    recommend_sources,
    validate_source_use,
)
from bashgym.sources.fetch import (
    DEFAULT_SOURCE_FETCH_LIMIT,
    SOURCE_FETCH_APPROVAL_LIMIT,
    SOURCE_FETCH_SCHEMA_VERSION,
    fetch_source_records,
    source_fetch_approval_policy,
)

SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION = "bashgym.source_artifact_prepare.v1"


def prepare_source_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import local source adapters to avoid preference/source cycles."""

    from bashgym.sources.adapters import prepare_source_artifacts as _prepare_source_artifacts

    return _prepare_source_artifacts(*args, **kwargs)

__all__ = [
    "SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION",
    "SOURCE_FETCH_SCHEMA_VERSION",
    "DEFAULT_SOURCE_FETCH_LIMIT",
    "SOURCE_FETCH_APPROVAL_LIMIT",
    "SourceArtifactType",
    "SourceCard",
    "SourceRisk",
    "SourceUse",
    "get_source",
    "list_sources",
    "prepare_source_manifest",
    "prepare_source_artifacts",
    "fetch_source_records",
    "source_fetch_approval_policy",
    "recommend_sources",
    "validate_source_use",
]
