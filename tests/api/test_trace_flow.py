"""
Integration tests for the full import -> filter -> analytics flow.

These tests verify cross-endpoint flows: importing traces from one endpoint,
then querying them via another, then checking analytics aggregates.

Uses the same fixture patterns as test_import_routes.py: temp directories for
trace storage, mocked importers, and a patched TestClient.
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
# Helpers — reuse the same trace-building functions from test_import_routes
# ---------------------------------------------------------------------------

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


def _make_trace_dict(source_tool: str, steps: int = 5, cost: float = 0.0,
                     total_score: float = 0.0, repo_name: str = "test-repo",
                     session_id: str = None):
    """Create a realistic imported TraceSession dict."""
    trace_steps = []
    for i in range(steps):
        trace_steps.append({
            "tool_name": "bash" if i % 2 == 0 else "read",
            "success": True,
            "exit_code": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "timestamp": f"2026-01-01T00:0{min(i, 9)}:00Z",
        })
    sid = session_id or f"test_{source_tool}_{id(trace_steps)}"
    return {
        "session_id": sid,
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
            "timestamp": f"2026-01-01T00:0{min(i, 9)}:00Z",
            "repo": {"name": "raw-repo", "path": "/home/user/raw-repo", "is_git_repo": True},
        }
        for i in range(steps)
    ]


# Patch targets — same as test_import_routes.py
CLAUDE_PATCH = "bashgym.trace_capture.importers.import_recent"
GEMINI_PATCH = "bashgym.trace_capture.importers.import_gemini_sessions"
COPILOT_PATCH = "bashgym.trace_capture.importers.import_copilot_sessions"
OPENCODE_PATCH = "bashgym.trace_capture.importers.import_opencode_sessions"
CODEX_PATCH = "bashgym.trace_capture.adapters.codex.import_codex_sessions"
CHATGPT_PATCH = "bashgym.trace_capture.importers.import_chatgpt_sessions"
MCP_PATCH = "bashgym.trace_capture.importers.import_mcp_logs"


# ---------------------------------------------------------------------------
# Minimal settings mock
# ---------------------------------------------------------------------------

@dataclass
class _FakeSettings:
    """Minimal settings mock for trace endpoints."""
    @dataclass
    class _Data:
        data_dir: str = ""
    data: _Data = dc_field(default_factory=_Data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trace_dirs(tmp_path):
    """Create a temporary directory structure with sample traces.

    Populates gold, failed, and pending directories with traces from
    multiple source tools (claude_code, gemini_cli, copilot_cli, opencode).
    """
    data_dir = tmp_path / "data"
    bashgym_dir = tmp_path / "bashgym"

    # Tier directories
    gold_dir = data_dir / "gold_traces"
    gold_dir.mkdir(parents=True)
    failed_dir = data_dir / "failed_traces"
    failed_dir.mkdir(parents=True)

    # Pending traces directory (global)
    pending_dir = bashgym_dir / "traces"
    pending_dir.mkdir(parents=True)

    # Gold trace from claude_code
    (gold_dir / "gold_claude_001.json").write_text(
        json.dumps(_make_trace_dict("claude_code", steps=4, cost=1.50, total_score=0.85)),
        encoding="utf-8",
    )

    # Gold trace from gemini_cli
    (gold_dir / "gold_gemini_001.json").write_text(
        json.dumps(_make_trace_dict("gemini_cli", steps=6, cost=0.80, total_score=0.78)),
        encoding="utf-8",
    )

    # Failed trace from copilot_cli
    (failed_dir / "failed_copilot_001.json").write_text(
        json.dumps(_make_trace_dict("copilot_cli", steps=3, cost=0.25, total_score=0.30)),
        encoding="utf-8",
    )

    # Pending imported trace from opencode
    (pending_dir / "imported_opencode_001.json").write_text(
        json.dumps(_make_trace_dict("opencode", steps=5, cost=0.0, total_score=0.60)),
        encoding="utf-8",
    )

    # Pending raw trace (claude_code by default)
    (pending_dir / "session_raw_001.json").write_text(
        json.dumps(_make_raw_trace_list(steps=3)),
        encoding="utf-8",
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

    real_settings.data.data_dir = str(data_dir)

    with patch("bashgym.config.get_bashgym_dir", return_value=bashgym_dir):
        from bashgym.api.routes import create_app
        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=False) as client:
            yield client

    real_settings.data.data_dir = original_data_dir


@pytest.fixture
def client():
    """Plain TestClient without patched directories."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Import then list by source
# ---------------------------------------------------------------------------

class TestImportThenListBySource:
    """Import via /import/claude, then GET /api/traces?source_tool=claude_code."""

    def test_import_then_list_by_source(self, trace_dirs):
        """Import via /import/claude, then list with ?source_tool=claude_code
        and verify the imported traces appear."""
        data_dir, bashgym_dir = trace_dirs

        # The import endpoint calls import_recent() which writes files.
        # We mock import_recent to write a real trace file into the pending dir
        # so list_traces can pick it up.
        dest_file = bashgym_dir / "traces" / "imported_claude_new.json"
        trace_data = _make_trace_dict(
            "claude_code", steps=7, cost=2.00, total_score=0.92,
            session_id="claude_new_session",
        )

        def fake_import(**kwargs):
            """Write a trace file and return results."""
            dest_file.write_text(json.dumps(trace_data), encoding="utf-8")
            return [
                FakeImportResult(
                    session_id="claude_new_session",
                    source_file=Path("/fake/source/new.jsonl"),
                    steps_imported=7,
                    destination_file=dest_file,
                ),
            ]

        from bashgym.config import get_settings as _gs
        _orig = _gs().data.data_dir
        _gs().data.data_dir = str(data_dir)

        with patch("bashgym.config.get_bashgym_dir", return_value=bashgym_dir), \
             patch(CLAUDE_PATCH, side_effect=fake_import):
            from bashgym.api.routes import create_app
            test_app = create_app()
            with TestClient(test_app, raise_server_exceptions=False) as c:

                # Step 1: Import
                import_resp = c.post("/api/traces/import/claude", json={"days": 30})
                assert import_resp.status_code == 200
                import_data = import_resp.json()
                assert import_data["imported"] == 1

                # Force cache refresh by resetting scan interval guard
                # (import writes new files but doesn't invalidate the cache;
                # the refresh() guard prevents re-scan within 2s of startup)
                test_app.state.trace_cache._last_scan_time = 0.0

                # Step 2: List filtered by source_tool=claude_code
                list_resp = c.get("/api/traces?source_tool=claude_code")
                assert list_resp.status_code == 200
                list_data = list_resp.json()

                # Should include the newly imported trace + existing claude traces
                assert list_data["total"] >= 1
                source_tools = {t["source_tool"] for t in list_data["traces"]}
                assert source_tools == {"claude_code"}, f"Expected only claude_code, got {source_tools}"

                # Verify the newly imported trace is present (by checking trace count increased)
                # We had gold_claude_001 + session_raw_001 + imported_claude_new = at least 3
                claude_ids = [t["trace_id"] for t in list_data["traces"]]
                # The new trace file stem should appear
                assert any("claude_new" in tid for tid in claude_ids), \
                    f"New import not found in traces: {claude_ids}"

        _gs().data.data_dir = _orig


# ---------------------------------------------------------------------------
# 2. Import all then analytics
# ---------------------------------------------------------------------------

class TestImportAllThenAnalytics:
    """Import via /import/all, then call /traces/analytics and verify source_breakdown."""

    def test_import_all_then_analytics(self, trace_dirs):
        """Import via /import/all, then check analytics source_breakdown has entries."""
        data_dir, bashgym_dir = trace_dirs

        from bashgym.config import get_settings as _gs
        _orig = _gs().data.data_dir
        _gs().data.data_dir = str(data_dir)

        # Mock all importers to return empty (we already have traces on disk)
        with patch("bashgym.config.get_bashgym_dir", return_value=bashgym_dir), \
             patch(CLAUDE_PATCH, return_value=[]), \
             patch(GEMINI_PATCH, return_value=[]), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            from bashgym.api.routes import create_app
            test_app = create_app()
            with TestClient(test_app, raise_server_exceptions=False) as c:

                # Step 1: Import all (no new traces, but endpoint should succeed)
                import_resp = c.post("/api/traces/import/all")
                assert import_resp.status_code == 200
                import_data = import_resp.json()
                assert "results" in import_data
                assert len(import_data["results"]) >= 5

                # Step 2: Analytics should reflect the traces already on disk
                analytics_resp = c.get("/api/traces/analytics")
                assert analytics_resp.status_code == 200
                analytics = analytics_resp.json()

                assert "source_breakdown" in analytics
                assert isinstance(analytics["source_breakdown"], list)
                # We have traces from claude_code, gemini_cli, copilot_cli on disk
                sources = {e["source"] for e in analytics["source_breakdown"]}
                assert len(sources) >= 2, f"Expected at least 2 sources, got: {sources}"
                assert "claude_code" in sources
                assert "gemini_cli" in sources

        _gs().data.data_dir = _orig


# ---------------------------------------------------------------------------
# 3. Analytics fields complete
# ---------------------------------------------------------------------------

class TestAnalyticsFieldsComplete:
    """Verify ALL expected analytics fields exist."""

    def test_analytics_fields_complete(self, patched_client):
        """Call /traces/analytics and verify ALL expected fields exist."""
        resp = patched_client.get("/api/traces/analytics")
        assert resp.status_code == 200
        data = resp.json()

        required_fields = [
            "tool_stats",
            "quality_distribution",
            "totals",
            "training_readiness",
            "source_breakdown",
            "cost_total_usd",
            "avg_quality_score",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_analytics_field_types(self, patched_client):
        """Verify analytics field types are correct."""
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()

        assert isinstance(data["tool_stats"], list)
        assert isinstance(data["quality_distribution"], dict)
        assert isinstance(data["totals"], dict)
        assert isinstance(data["training_readiness"], dict)
        assert isinstance(data["source_breakdown"], list)
        assert isinstance(data["cost_total_usd"], (int, float))
        assert isinstance(data["avg_quality_score"], (int, float))

    def test_analytics_source_breakdown_entries_complete(self, patched_client):
        """Each source_breakdown entry should have source, traces, steps, tokens."""
        resp = patched_client.get("/api/traces/analytics")
        data = resp.json()

        for entry in data["source_breakdown"]:
            assert "source" in entry
            assert "traces" in entry
            assert "steps" in entry
            assert "tokens" in entry
            assert isinstance(entry["source"], str)
            assert isinstance(entry["traces"], int)
            assert isinstance(entry["steps"], int)
            assert isinstance(entry["tokens"], int)


# ---------------------------------------------------------------------------
# 4. Source filter excludes other tools
# ---------------------------------------------------------------------------

class TestSourceFilterExcludesOtherTools:
    """Create traces from multiple sources, filter by one, verify exclusion."""

    def test_source_filter_excludes_other_tools(self, patched_client):
        """List with ?source_tool=gemini_cli — verify only Gemini traces appear."""
        resp = patched_client.get("/api/traces?source_tool=gemini_cli")
        assert resp.status_code == 200
        data = resp.json()

        # We have gold_gemini_001 on disk — should get at least 1 result
        assert data["total"] >= 1

        # Every trace in the response must be gemini_cli
        for trace in data["traces"]:
            assert trace["source_tool"] == "gemini_cli", \
                f"Expected gemini_cli, got {trace['source_tool']}"

    def test_source_filter_excludes_claude_when_filtering_copilot(self, patched_client):
        """Filter by copilot_cli — no claude or gemini traces should appear."""
        resp = patched_client.get("/api/traces?source_tool=copilot_cli")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total"] >= 1
        for trace in data["traces"]:
            assert trace["source_tool"] == "copilot_cli"
            assert trace["source_tool"] != "claude_code"
            assert trace["source_tool"] != "gemini_cli"

    def test_filter_nonexistent_source_returns_empty(self, patched_client):
        """Filtering by a source that doesn't exist should return zero traces."""
        resp = patched_client.get("/api/traces?source_tool=cursor_ai")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["traces"] == []

    def test_no_filter_returns_all_sources(self, patched_client):
        """Omitting source_tool should return traces from all sources."""
        resp = patched_client.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()

        source_tools = {t["source_tool"] for t in data["traces"]}
        # We have claude_code, gemini_cli, copilot_cli, opencode on disk
        assert len(source_tools) >= 3, f"Expected at least 3 sources, got: {source_tools}"


# ---------------------------------------------------------------------------
# 5. Import invalid source returns 400
# ---------------------------------------------------------------------------

class TestImportInvalidSource:
    """Verify invalid source names are rejected."""

    def test_import_invalid_source_returns_400(self, client):
        """POST /api/traces/import/invalid should return 400."""
        resp = client.post("/api/traces/import/invalid")
        assert resp.status_code == 400
        data = resp.json()
        assert "Unknown source" in data["detail"]

    def test_import_invalid_source_nonsense(self, client):
        """POST /api/traces/import/foobar123 should return 400."""
        resp = client.post("/api/traces/import/foobar123")
        assert resp.status_code == 400

    def test_import_invalid_source_lists_valid_sources(self, client):
        """The 400 error detail should list valid source names."""
        resp = client.post("/api/traces/import/invalid_tool")
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Should mention at least claude and gemini as valid
        assert "claude" in detail
        assert "gemini" in detail


# ---------------------------------------------------------------------------
# 6. Import all with partial failure
# ---------------------------------------------------------------------------

class TestImportAllPartialFailure:
    """Mock one importer to throw, verify /import/all still returns 200."""

    def test_import_all_partial_failure(self, client):
        """One importer throws, others succeed — overall 200 with mixed results."""
        def _make_gemini_results():
            return [{
                "session_id": "gemini_ok_001",
                "source_file": "/fake/gemini/001.json",
                "steps_imported": 5,
                "destination_file": "/fake/dest/gemini_ok_001.json",
                "error": None,
                "skipped": False,
                "skip_reason": None,
                "source_tool": "gemini_cli",
            }]

        with patch(CLAUDE_PATCH, side_effect=RuntimeError("Claude dir not found")), \
             patch(GEMINI_PATCH, return_value=_make_gemini_results()), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()

        assert "results" in data
        assert len(data["results"]) >= 5

        # Claude should have failed
        claude_result = next(r for r in data["results"] if r["source"] == "claude")
        assert claude_result["errors"] == 1
        assert "error_detail" in claude_result
        assert "Claude dir not found" in claude_result["error_detail"]

        # Gemini should have succeeded
        gemini_result = next(r for r in data["results"] if r["source"] == "gemini")
        assert gemini_result["imported"] == 1
        assert gemini_result["errors"] == 0

        # Total imported should count only successful imports
        assert data["total_imported"] == 1

    def test_import_all_multiple_failures(self, client):
        """Multiple importers fail — still 200, error details for each."""
        with patch(CLAUDE_PATCH, side_effect=RuntimeError("fail claude")), \
             patch(GEMINI_PATCH, side_effect=OSError("fail gemini")), \
             patch(COPILOT_PATCH, side_effect=ValueError("fail copilot")), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_imported"] == 0

        failed_sources = [r for r in data["results"] if "error_detail" in r]
        assert len(failed_sources) >= 3

        for result in failed_sources:
            assert result["errors"] == 1

    def test_import_all_partial_failure_preserves_response_shape(self, client):
        """Even with failures, every result has the standard response fields."""
        with patch(CLAUDE_PATCH, side_effect=RuntimeError("boom")), \
             patch(GEMINI_PATCH, return_value=[]), \
             patch(COPILOT_PATCH, return_value=[]), \
             patch(OPENCODE_PATCH, return_value=[]), \
             patch(CODEX_PATCH, return_value=[]), \
             patch(CHATGPT_PATCH, return_value=[]), \
             patch(MCP_PATCH, return_value=[]):
            resp = client.post("/api/traces/import/all")

        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert "source" in result
            assert "imported" in result
            assert "skipped" in result
            assert "errors" in result
            assert "new_trace_ids" in result


# ---------------------------------------------------------------------------
# 7. Full end-to-end flow: import -> list -> analytics consistency
# ---------------------------------------------------------------------------

class TestEndToEndConsistency:
    """Verify that analytics numbers are consistent with list results."""

    def test_analytics_cost_matches_traces(self, patched_client):
        """Cost total in analytics should equal sum of trace costs on disk."""
        analytics = patched_client.get("/api/traces/analytics").json()

        # We set costs: claude=1.50, gemini=0.80, copilot=0.25
        # Analytics scans gold + failed dirs (the tiered dirs)
        assert analytics["cost_total_usd"] == pytest.approx(2.55, abs=0.01)

    def test_analytics_sources_match_list_sources(self, patched_client):
        """Sources in analytics breakdown should match sources from list endpoint."""
        # Get all traces
        list_resp = patched_client.get("/api/traces")
        list_data = list_resp.json()
        list_sources = {t["source_tool"] for t in list_data["traces"]}

        # Get analytics
        analytics_resp = patched_client.get("/api/traces/analytics")
        analytics_data = analytics_resp.json()
        analytics_sources = {e["source"] for e in analytics_data["source_breakdown"]}

        # Analytics scans only tiered dirs (gold/failed), not pending
        # So analytics_sources should be a subset of list_sources
        # (list also includes pending traces)
        for src in analytics_sources:
            assert src in list_sources, \
                f"Analytics source {src} not found in list sources: {list_sources}"

    def test_quality_score_in_valid_range(self, patched_client):
        """Average quality score should be between 0 and 1."""
        analytics = patched_client.get("/api/traces/analytics").json()
        assert 0.0 <= analytics["avg_quality_score"] <= 1.0

    def test_source_breakdown_trace_counts_positive(self, patched_client):
        """Each source in the breakdown should have at least 1 trace."""
        analytics = patched_client.get("/api/traces/analytics").json()
        for entry in analytics["source_breakdown"]:
            assert entry["traces"] >= 1, \
                f"Source {entry['source']} has {entry['traces']} traces"
