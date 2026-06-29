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

SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION = "bashgym.source_artifact_prepare.v1"


def prepare_source_artifacts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import local source adapters to avoid preference/source cycles."""

    from bashgym.sources.adapters import prepare_source_artifacts as _prepare_source_artifacts

    return _prepare_source_artifacts(*args, **kwargs)

__all__ = [
    "SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION",
    "SourceArtifactType",
    "SourceCard",
    "SourceRisk",
    "SourceUse",
    "get_source",
    "list_sources",
    "prepare_source_manifest",
    "prepare_source_artifacts",
    "recommend_sources",
    "validate_source_use",
]
