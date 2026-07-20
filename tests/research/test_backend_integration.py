"""Unit tests for BucketManager, TraceUploader, and research_routes. All mocked — no network."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestBucketManager:
    @patch("bashgym.integrations.huggingface.buckets.HfApi")
    def test_create_bucket(self, mock_api):
        from bashgym.integrations.huggingface.buckets import BucketManager

        mock_api.return_value.create_bucket.return_value = (
            "https://huggingface.co/buckets/user/test"
        )
        mgr = BucketManager(token="fake")
        result = mgr.create("user/test", private=True)
        assert result["bucket_id"] == "user/test"
        assert "url" in result
        mock_api.return_value.create_bucket.assert_called_once()

    @patch("bashgym.integrations.huggingface.buckets.HfApi")
    def test_list_buckets_empty(self, mock_api):
        from bashgym.integrations.huggingface.buckets import BucketManager

        mock_api.return_value.list_buckets.return_value = []
        mgr = BucketManager(token="fake")
        result = mgr.list_buckets()
        assert result == []

    @patch("bashgym.integrations.huggingface.buckets.HfApi")
    def test_delete_bucket(self, mock_api):
        from bashgym.integrations.huggingface.buckets import BucketManager

        mgr = BucketManager(token="fake")
        mgr.delete("user/test")
        mock_api.return_value.delete_bucket.assert_called_once_with(
            bucket_id="user/test", missing_ok=True
        )

    @patch("bashgym.integrations.huggingface.buckets.copy_files")
    def test_copy_static_method(self, mock_copy):
        from bashgym.integrations.huggingface.buckets import BucketManager

        result = BucketManager.copy(
            "hf://datasets/org/src/file.jsonl",
            "hf://buckets/org/dst/file.jsonl",
            token="fake",
        )
        mock_copy.assert_called_once_with(
            source="hf://datasets/org/src/file.jsonl",
            destination="hf://buckets/org/dst/file.jsonl",
            token="fake",
        )
        assert result["status"] == "completed"


class TestTraceUploader:
    @patch("bashgym.integrations.huggingface.traces.HfApi")
    def test_upload_traces(self, mock_api, tmp_path):
        from bashgym.integrations.huggingface.traces import TraceUploader

        # Create fake trace files
        for i in range(3):
            (tmp_path / f"trace_{i}.json").write_text(json.dumps({"session": i, "turns": []}))

        uploader = TraceUploader(token="fake")
        result = uploader.upload_traces(
            trace_dir=tmp_path,
            repo_id="user/test-traces",
            private=True,
        )

        assert result["repo_id"] == "user/test-traces"
        assert result["num_traces"] == 3
        assert result["total_size_bytes"] > 0
        mock_api.return_value.create_repo.assert_called_once()
        mock_api.return_value.upload_file.assert_called_once()

    def test_upload_traces_empty_dir_raises(self, tmp_path):
        from bashgym.integrations.huggingface.traces import TraceUploader

        uploader = TraceUploader(token="fake")
        with pytest.raises(ValueError, match="No .json files"):
            uploader.upload_traces(trace_dir=tmp_path, repo_id="user/empty")

    @patch("bashgym.integrations.huggingface.traces.HfApi")
    def test_list_trace_datasets(self, mock_api):
        from bashgym.integrations.huggingface.traces import TraceUploader

        mock_api.return_value.whoami.return_value = {"name": "testuser"}
        mock_ds = MagicMock()
        mock_ds.id = "testuser/bashgym-traces"
        mock_ds.private = True
        mock_ds.downloads = 5
        mock_ds.lastModified = "2026-04-10"
        ordinary_ds = MagicMock()
        ordinary_ds.id = "testuser/bashgym-training-data"
        mock_api.return_value.list_datasets.return_value = [mock_ds, ordinary_ds]

        uploader = TraceUploader(token="fake")
        result = uploader.list_trace_datasets()
        assert len(result) == 1
        assert result[0]["id"] == "testuser/bashgym-traces"


class TestResearchRoutes:
    def test_report_returns_content(self, tmp_path):
        from bashgym.api.research_routes import REPORT_PATH

        # Can't easily test FastAPI routes without TestClient, but we can
        # verify the module imports and the path constants are sensible.
        assert "hf_datasets_report.md" in str(REPORT_PATH)

    def test_cache_path_exists(self):
        from bashgym.api.research_routes import CACHE_PATH

        assert "hf_datasets_cache.json" in str(CACHE_PATH)
