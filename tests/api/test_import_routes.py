"""
Tests for unified trace import API endpoints.

Covers:
  - POST /api/traces/import/{source} for each valid source
  - POST /api/traces/import/all aggregated import
  - Invalid source returns 400
  - Response shape validation
  - Error handling when importers raise exceptions
  - GET /api/traces with source_tool filter
  - GET /api/traces/analytics with source_breakdown, cost_total_usd, avg_quality_score
"""

import json
import pytest
import tempfile
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from bashgym.api.routes import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


@dataclass
class FakeImportResult:
    """Mimics Claude's ImportResult dataclass."""
    session_id: str
    source_file: Path
    steps_imported: int
    destination_file: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


def _make_claude_results():
    """Return a list that mimics import_recent() output."""
    return [
        FakeImportResult(
            session_id="session_001",
            source_file=Path("/fake/source/001.jsonl"),
            steps_imported=10,
            destination_file=Path("/fake/dest/session_001.json"),
        ),
        FakeImportResult(
            session_id="session_002",
            source_file=Path("/fake/source/002.jsonl"),
            steps_imported=0,
            skipped=True,
            skip_reason="Already imported",
        ),
    ]


def _make_dict_results(source_tool: str):
    """Return a list of dicts mimicking Gemini/Copilot/OpenCode output."""
    return [
        {
            "session_id": f"{source_tool}_session_001",
            "source_file": f"/fake/{source_tool}/001.json",
            "steps_imported": 5,
            "destination_file": f"/fake/dest/{source_tool}_session_001.json",
            "error": None,
            "skipped": False,
            "skip_reason": None,
            "source_tool": source_tool,
        },
        {
            "session_id": f"{source_tool}_session_002",
            "source_file": f"/fake/{source_tool}/002.json",
            "steps_imported": 0,
            "destination_file": None,
            "error": None,
            "skipped": True,
            "skip_reason": "Already imported",
            "source_tool": source_tool,
        },
        {
            "session_id": f"{source_tool}_session_003",
            "source_file": f"/fake/{source_tool}/003.json",
            "steps_imported": 0,
            "destination_file": None,
            "error": "Parse error",
            "skipped": False,
            "source_tool": source_tool,
        },
    ]


def _make_codex_results():
    """Return a list of dicts mimicking import_codex_sessions() output."""
    return [
        {
            "session_file": "/fake/codex/transcript1.json",
            "trace_file": "/fake/dest/codex_transcript1.json",
            "steps_imported": 3,
            "error": None,
        },
        {
            "session_file": "/fake/codex/transcript2.json",
            "trace_file": None,
            "steps_imported": 0,
            "error": "already imported",
        },
    ]


# The import functions are resolved lazily inside _get_import_handlers(),
# so we patch them at their canonical module paths.
CLAUDE_PATCH = "bashgym.trace_capture.importers.import_recent"
GEMINI_PATCH = "bashgym.trace_capture.importers.import_gemini_sessions"
COPILOT_PATCH = "bashgym.trace_capture.importers.import_copilot_sessions"
OPENCODE_PATCH = "bashgym.trace_capture.importers.import_opencode_sessions"
CODEX_PATCH = "bashgym.trace_capture.adapters.codex.import_codex_sessions"
CHATGPT_PATCH = "bashgym.trace_capture.importers.import_chatgpt_sessions"
MCP_PATCH = "bashgym.trace_capture.importers.import_mcp_logs"


def _validate_response_shape(data: dict):
    """Assert that a per-source response has the required fields."""
    assert "source" in data
    assert "imported" in data
    assert isinstance(data["imported"], int)
    assert "skipped" in data
    assert isinstance(data["skipped"], int)
    assert "errors" in data
    assert isinstance(data["errors"], int)
    assert "total" in data
    assert isinstance(data["total"], int)
    assert "new_trace_ids" in data
    assert isinstance(data["new_trace_ids"], list)


# ---------------------------------------------------------------------------
# 1. POST /api/traces/import/{source} -- per-source
# ---------------------------------------------------------------------------

class TestImportBySource:
    """Tests for the /api/traces/import/{source} endpoint."""

    def test_import_claude(self, client):
        with patch(CLAUDE_PATCH, return_value=_make_claude_results()) as mock_fn:
            resp = client.post("/api/traces/import/claude", json={"days": 30, "limit": 50, "force": True})

        assert resp.status_code == 200
        data = resp.json()
        _validate_response_shape(data)
        assert data["source"] == "claude"
        assert data["imported"] == 1
        assert data["skipped"] == 1
        assert data["errors"] == 0
        assert data["total"] == 2
        assert len(data["new_trace_ids"]) == 1

        # Verify the handler was called with correct params
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        # Called via lambda: import_recent(days=req.days, verbose=False, force=req.force)
        assert call_kwargs[1]["days"] == 30 or call_kwargs.kwargs.get("days") == 30

    def test_import_gemini(self, client):
        with patch(GEMINI_PATCH, return_value=_make_dict_results("gemini_cli")):
            resp = client.post("/api/traces/import/gemini", json={"days": 14})

        assert resp.status_code == 200
        data = resp.json()
        _validate_response_shape(data)
        assert data["source"] == "gemini"
        assert data["imported"] == 1
        assert data["skipped"] == 1
        assert data["errors"] == 1
        assert data["total"] == 3

    def test_import_copilot(self, client):
        with patch(COPILOT_PATCH, return_value=_make_dict_results("copilot_cli")):
            resp = client.post("/api/traces/import/copilot")

        assert resp.status_code == 200
        data = resp.json()
        _validate_response_shape(data)
        assert data["source"] == "copilot"
        assert data["imported"] == 1

    def test_import_opencode(self, client):
        with patch(OPENCODE_PATCH, return_value=_make_dict_results("opencode")):
            resp = client.post("/api/traces/import/opencode")

        assert resp.status_code == 200
        data = resp.json()
        _validate_response_shape(data)
        assert data["source"] == "opencode"

    def test_import_codex(self, client):
        with patch(CODEX_PATCH, return_value=_make_codex_results()):
            resp = client.post("/api/traces/import/codex", json={"limit": 10})

        assert resp.status_code == 200
        data = resp.json()
        _validate_response_shape(data)
        assert data["source"] == "codex"
        # First result: no error -> imported. Second: has error -> error.
        assert data["imported"] == 1
        assert data["errors"] == 1
        assert data["total"] == 2
        # Only the first result has a trace_file
        assert len(data["new_trace_ids"]) == 1

    def test_invalid_source_returns_400(self, client):
        resp = client.post("/api/traces/import/invalid_tool")
        assert resp.status_code == 400
        assert "Unknown source" in resp.json()["detail"]

    def test_default_request_body(self, client):
        """Omitting the request body should use defaults."""
        with patch(GEMINI_PATCH, return_value=[]) as mock_fn:
            resp = client.post("/api/traces/import/gemini")

        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 0
        assert data["total"] == 0

    def test_importer_exception_returns_500(self, client):
        with patch(GEMINI_PATCH, side_effect=RuntimeError("Disk full")):
            resp = client.post("/api/traces/import/gemini")

        assert resp.status_code == 500
        assert "Disk full" in resp.json()["detail"]

    def test_empty_results(self, client):
        """Source exists but returns no results."""
        with patch(CLAUDE_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/claude")

        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 0
        assert data["skipped"] == 0
        assert data["errors"] == 0
        assert data["total"] == 0
        assert data["new_trace_ids"] == []


# ---------------------------------------------------------------------------
# 2. POST /api/traces/import/all
# ---------------------------------------------------------------------------

class TestImportAll:
    """Tests for the /api/traces/import/all endpoint."""

    def test_import_all_basic(self, client):
        with patch(CLAUDE_PATCH, return_value=_make_claude_results()), \
             patch(GEMINI_PATCH, return_value=_make_dict_results("gemini_cli")), \
             patch(COPILOT_PATCH, return_value=_make_dict_results("copilot_cli")), \
             patch(OPENCODE_PATCH, return_value=_make_dict_results("opencode")), \
             patch(CODEX_PATCH, return_value=_make_codex_results()), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all", json={"days": 7})

        assert resp.status_code == 200
        data = resp.json()

        assert "results" in data
        assert "total_imported" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) >= 5  # one per source (at least 5; chatgpt and mcp added later)

        sources = [r["source"] for r in data["results"]]
        assert "claude" in sources
        assert "gemini" in sources
        assert "copilot" in sources
        assert "opencode" in sources
        assert "codex" in sources

        for result in data["results"]:
            _validate_response_shape(result)

        # Claude: 1, Gemini: 1, Copilot: 1, OpenCode: 1, Codex: 1 = 5 (chatgpt and mcp return empty)
        assert data["total_imported"] == 5

    def test_import_all_partial_failure(self, client):
        """If one source fails, others should still succeed."""
        with patch(CLAUDE_PATCH, return_value=_make_claude_results()), \
             patch(GEMINI_PATCH, side_effect=RuntimeError("Gemini dir missing")), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 5

        gemini_result = next(r for r in data["results"] if r["source"] == "gemini")
        assert gemini_result["errors"] == 1
        assert "error_detail" in gemini_result
        assert "Gemini dir missing" in gemini_result["error_detail"]

        claude_result = next(r for r in data["results"] if r["source"] == "claude")
        assert claude_result["imported"] == 1

    def test_import_all_default_request(self, client):
        """Omitting body should use default TraceImportRequest."""
        with patch(CLAUDE_PATCH, return_value=[]), \
             patch(GEMINI_PATCH, return_value=[]), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_imported"] == 0
        assert len(data["results"]) >= 5

    def test_import_all_every_source_fails(self, client):
        """All sources fail -- we still get 200 with error details."""
        with patch(CLAUDE_PATCH, side_effect=RuntimeError("fail")), \
             patch(GEMINI_PATCH, side_effect=RuntimeError("fail")), \
             patch(COPILOT_PATCH, side_effect=RuntimeError("fail")), \
             patch(OPENCODE_PATCH, side_effect=RuntimeError("fail")), \
             patch(CODEX_PATCH, side_effect=RuntimeError("fail")), \
             patch(CHATGPT_PATCH, side_effect=RuntimeError("fail")), \
             patch(MCP_PATCH, side_effect=RuntimeError("fail")):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_imported"] == 0
        for result in data["results"]:
            assert result["errors"] == 1
            assert "error_detail" in result


# ---------------------------------------------------------------------------
# 3. Route ordering: "all" must not be captured by {source}
# ---------------------------------------------------------------------------

class TestRouteOrdering:
    """Verify that /import/all is matched before /import/{source}."""

    def test_all_is_not_treated_as_source(self, client):
        """POST /api/traces/import/all should hit import_traces_all,
        not import_traces_by_source with source='all'."""
        with patch(CLAUDE_PATCH, return_value=[]), \
             patch(GEMINI_PATCH, return_value=[]), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()
        # The /all endpoint returns {"results": [...], "total_imported": N}
        # The /{source} endpoint would return {"source": "all", ...} and fail
        assert "results" in data
        assert "total_imported" in data


# ---------------------------------------------------------------------------
# Helpers for trace file creation (used by source_tool and analytics tests)
# ---------------------------------------------------------------------------

def _make_trace_dict(source_tool: str, steps: int = 5, cost: float = 0.0,
                     total_score: float = 0.0, repo_name: str = "test-repo"):
    """Create a realistic imported TraceSession dict."""
    trace_steps = []
    for i in range(steps):
        trace_steps.append({
            "tool_name": "bash" if i % 2 == 0 else "read",
            "success": True,
            "exit_code": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "timestamp": f"2026-01-01T00:0{i}:00Z",
        })
    return {
        "session_id": f"test_{source_tool}_{id(trace_steps)}",
        "source_tool": source_tool,
        "trace": trace_steps,
        "primary_repo": {"name": repo_name, "path": f"/home/user/{repo_name}", "is_git_repo": True},
        "repos": [{"name": repo_name}],
        "metadata": {
            "api_equivalent_cost_usd": cost,
            "total_score": total_score,
            "user_initial_prompt": f"Test task from {source_tool}",
        },
        "summary": {
            "success_rate": 1.0,
            "total_score": total_score,
        },
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _make_raw_trace_list(steps: int = 3):
    """Create a raw session trace (list of steps) -- always claude_code."""
    return [
        {
            "tool_name": "bash",
            "success": True,
            "exit_code": 0,
            "input_tokens": 200,
            "output_tokens": 100,
            "timestamp": f"2026-01-01T00:0{i}:00Z",
            "repo": {"name": "raw-repo", "path": "/home/user/raw-repo", "is_git_repo": True},
        }
        for i in range(steps)
    ]


@dataclass
class _FakeSettings:
    """Minimal settings mock for trace endpoints."""
    @dataclass
    class _Data:
        data_dir: str = ""
    data: _Data = dc_field(default_factory=_Data)


@pytest.fixture
def trace_dirs(tmp_path):
    """Create a temporary directory structure with sample traces.

    Returns (data_dir, bashgym_dir) paths.
    """
    data_dir = tmp_path / "data"
    bashgym_dir = tmp_path / "bashgym"

    # Create tier directories
    gold_dir = data_dir / "gold_traces"
    gold_dir.mkdir(parents=True)
    failed_dir = data_dir / "failed_traces"
    failed_dir.mkdir(parents=True)

    # Create pending traces directory (global)
    pending_dir = bashgym_dir / "traces"
    pending_dir.mkdir(parents=True)

    # Gold trace from claude_code
    (gold_dir / "gold_claude_001.json").write_text(
        json.dumps(_make_trace_dict("claude_code", steps=4, cost=1.50, total_score=0.85)),
        encoding="utf-8"
    )

    # Gold trace from gemini_cli
    (gold_dir / "gold_gemini_001.json").write_text(
        json.dumps(_make_trace_dict("gemini_cli", steps=6, cost=0.80, total_score=0.78)),
        encoding="utf-8"
    )

    # Failed trace from copilot_cli
    (failed_dir / "failed_copilot_001.json").write_text(
        json.dumps(_make_trace_dict("copilot_cli", steps=3, cost=0.25, total_score=0.30)),
        encoding="utf-8"
    )

    # Pending imported trace from opencode
    (pending_dir / "imported_opencode_001.json").write_text(
        json.dumps(_make_trace_dict("opencode", steps=5, cost=0.0, total_score=0.60)),
        encoding="utf-8"
    )

    # Pending raw trace (claude_code by default)
    (pending_dir / "session_raw_001.json").write_text(
        json.dumps(_make_raw_trace_list(steps=3)),
        encoding="utf-8"
    )

    return data_dir, bashgym_dir


@pytest.fixture
def patched_client(trace_dirs):
    """TestClient with data_dir/bashgym_dir pointed at temp dirs.

    Uses create_app() with real settings (overriding data paths) so
    startup events fire and the trace cache indexes the temp files.
    """
    data_dir, bashgym_dir = trace_dirs

    from bashgym.config import get_settings
    real_settings = get_settings()
    original_data_dir = real_settings.data.data_dir

    # Override data paths to use temp dirs
    real_settings.data.data_dir = str(data_dir)

    with patch("bashgym.config.get_bashgym_dir", return_value=bashgym_dir):
        from bashgym.api.routes import create_app
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=False) as client:
            yield client

    # Restore
    real_settings.data.data_dir = original_data_dir


# ---------------------------------------------------------------------------
# 4. GET /api/traces -- source_tool filter
# ---------------------------------------------------------------------------

class TestSourceToolFilter:
    """Tests for the source_tool query parameter on GET /api/traces."""

    def test_list_traces_accepts_source_filter(self, patched_client):
        """Filtering by source_tool=claude_code should only return claude_code traces."""
        resp = patched_client.get("/api/traces?source_tool=claude_code")
        assert resp.status_code == 200
        data = resp.json()
        # We have gold_claude_001 + session_raw_001 as claude_code
        for trace in data["traces"]:
            assert trace["source_tool"] == "claude_code"
        assert data["total"] >= 1

    def test_list_traces_filter_gemini(self, patched_client):
        """Filtering by source_tool=gemini_cli should return only gemini traces."""
        resp = patched_client.get("/api/traces?source_tool=gemini_cli")
        assert resp.status_code == 200
        data = resp.json()
        for trace in data["traces"]:
            assert trace["source_tool"] == "gemini_cli"
        assert data["total"] >= 1

    def test_list_traces_filter_opencode(self, patched_client):
        """Filtering by opencode should return the pending imported trace."""
        resp = patched_client.get("/api/traces?source_tool=opencode")
        assert resp.status_code == 200
        data = resp.json()
        for trace in data["traces"]:
            assert trace["source_tool"] == "opencode"

    def test_list_traces_no_source_returns_all(self, patched_client):
        """Omitting source_tool should return all traces."""
        resp = patched_client.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        # We have 5 total traces across all directories
        assert data["total"] >= 4  # At least gold(2) + failed(1) + pending(1+)

    def test_list_traces_nonexistent_source_returns_empty(self, patched_client):
        """Filtering by a source that doesn't exist should return no traces."""
        resp = patched_client.get("/api/traces?source_tool=aider")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["traces"] == []

    def test_list_traces_source_combined_with_status(self, patched_client):
        """source_tool and status filters should work together."""
        resp = patched_client.get("/api/traces?source_tool=copilot_cli&status=failed")
        assert resp.status_code == 200
        data = resp.json()
        for trace in data["traces"]:
            assert trace["source_tool"] == "copilot_cli"
            assert trace["status"] == "failed"

    def test_list_traces_source_tool_in_response(self, patched_client):
        """Every trace in the response should have a source_tool field."""
        resp = patched_client.get("/api/traces")
        assert resp.status_code == 200
        for trace in resp.json()["traces"]:
            assert "source_tool" in trace
            assert isinstance(trace["source_tool"], str)


# ---------------------------------------------------------------------------
# 5. GET /api/traces/analytics -- source_breakdown, cost, avg_quality
# ---------------------------------------------------------------------------

class TestAnalyticsEnhanced:
    """Tests for the enhanced analytics endpoint."""

    def test_analytics_has_source_breakdown(self, patched_client):
        resp = patched_client.get("/api/traces/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert "source_breakdown" in data
        assert isinstance(data["source_breakdown"], list)

    def test_analytics_source_breakdown_structure(self, patched_client):
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()
        for entry in data["source_breakdown"]:
            assert "source" in entry
            assert "traces" in entry
            assert isinstance(entry["traces"], int)
            assert "steps" in entry
            assert isinstance(entry["steps"], int)
            assert "tokens" in entry
            assert isinstance(entry["tokens"], int)

    def test_analytics_source_breakdown_covers_sources(self, patched_client):
        """Analytics scans tiered dirs (gold/failed), so should see claude_code, gemini_cli, copilot_cli."""
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()
        sources = {e["source"] for e in data["source_breakdown"]}
        # gold has claude_code and gemini_cli; failed has copilot_cli
        assert "claude_code" in sources
        assert "gemini_cli" in sources
        assert "copilot_cli" in sources

    def test_analytics_has_cost_total(self, patched_client):
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()
        assert "cost_total_usd" in data
        assert isinstance(data["cost_total_usd"], (int, float))
        # gold_claude=1.50, gold_gemini=0.80, failed_copilot=0.25 => 2.55
        assert data["cost_total_usd"] == pytest.approx(2.55, abs=0.01)

    def test_analytics_has_avg_quality(self, patched_client):
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()
        assert "avg_quality_score" in data
        assert isinstance(data["avg_quality_score"], (int, float))
        # Quality score is averaged across all indexed traces (gold + failed + pending)
        assert data["avg_quality_score"] > 0
        assert data["avg_quality_score"] < 1

    def test_analytics_zero_cost_when_no_traces(self, tmp_path):
        """Empty data directory should yield zero cost and empty source_breakdown."""
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        bashgym_dir = tmp_path / "empty_bashgym"
        bashgym_dir.mkdir()
        fake = _FakeSettings()
        fake.data.data_dir = str(data_dir)

        with patch("bashgym.config.get_settings", return_value=fake), \
             patch("bashgym.config.get_bashgym_dir", return_value=bashgym_dir):
            c = TestClient(app)
            resp = c.get("/api/traces/analytics")

        assert resp.status_code == 200
        data = resp.json()
        assert data["cost_total_usd"] == 0.0
        assert data["avg_quality_score"] == 0.0
        assert data["source_breakdown"] == []

    def test_analytics_still_has_existing_fields(self, patched_client):
        """Ensure new fields didn't break existing response structure."""
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()
        assert "tool_stats" in data
        assert "quality_distribution" in data
        assert "totals" in data
        assert "training_readiness" in data
