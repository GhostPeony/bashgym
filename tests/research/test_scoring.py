"""Unit tests for bashgym.research.scoring. No network, all inputs mocked."""
from datetime import datetime, timedelta

import pytest

from bashgym.research.scoring import DatasetMetadata, score_dataset


def _fresh_meta(**overrides) -> DatasetMetadata:
    """Build a 'nominal good' DatasetMetadata, override specific fields per test."""
    base = DatasetMetadata(
        repo_id="test-org/test-dataset",
        tags=["code-generation", "task_categories:text-generation"],
        license="apache-2.0",
        num_rows=5_000,
        download_size_bytes=50_000_000,
        features={"messages": "list<struct>"},
        last_modified=datetime.now() - timedelta(days=30),
        downloads=5_000,
        gated=False,
        description="A high quality code generation dataset.",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestHardFilters:
    def test_rejects_gated_dataset(self):
        meta = _fresh_meta(gated=True)
        result = score_dataset(meta)
        assert result.rejected is True
        assert result.rejection_reason is not None
        assert "gated" in result.rejection_reason.lower()

    def test_rejects_non_commercial_license(self):
        meta = _fresh_meta(license="cc-by-nc-4.0")
        result = score_dataset(meta)
        assert result.rejected is True
        assert "license" in result.rejection_reason.lower()

    def test_rejects_too_small(self):
        meta = _fresh_meta(num_rows=5)
        result = score_dataset(meta)
        assert result.rejected is True
        assert "small" in result.rejection_reason.lower() or "size" in result.rejection_reason.lower()

    def test_rejects_too_large(self):
        meta = _fresh_meta(num_rows=5_000_000)
        result = score_dataset(meta)
        assert result.rejected is True
        assert "large" in result.rejection_reason.lower() or "size" in result.rejection_reason.lower()
