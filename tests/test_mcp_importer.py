"""Tests for MCP tool log importer."""

import json
import tempfile
from pathlib import Path

import pytest

MCP_JSONRPC_LOG = [
    {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "/src/main.py"}}, "id": 1, "timestamp": "2024-11-15T10:00:00Z"},
    {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": "def main(): pass"}]}, "id": 1, "timestamp": "2024-11-15T10:00:01Z"},
    {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "write_file", "arguments": {"path": "/src/main.py", "content": "def main():\n    print('hello')"}}, "id": 2, "timestamp": "2024-11-15T10:00:10Z"},
    {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": "File written successfully"}]}, "id": 2, "timestamp": "2024-11-15T10:00:11Z"},
]

MCP_SIMPLE_LOG = [
    {"tool": "read_file", "arguments": {"path": "/src/main.py"}, "result": "def main(): pass", "timestamp": "2024-11-15T10:00:00Z", "duration_ms": 50},
    {"tool": "write_file", "arguments": {"path": "/src/main.py", "content": "updated"}, "result": "OK", "timestamp": "2024-11-15T10:00:10Z", "duration_ms": 30},
]


def write_log_file(log_data, tmp_dir: Path, fmt: str = "jsonl") -> Path:
    path = tmp_dir / f"mcp_log.{fmt}"
    if fmt == "jsonl":
        path.write_text("\n".join(json.dumps(entry) for entry in log_data))
    else:
        path.write_text(json.dumps(log_data))
    return path


class TestMCPLogImporter:
    def test_parse_jsonrpc_log(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()
        steps, metadata = importer.parse_log(MCP_JSONRPC_LOG)

        assert len(steps) == 2
        assert steps[0].tool_name == "read_file"
        assert steps[1].tool_name == "write_file"
        assert metadata["source"] == "mcp"

    def test_parse_simple_log(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()
        steps, metadata = importer.parse_log(MCP_SIMPLE_LOG)

        assert len(steps) == 2
        assert steps[0].tool_name == "read_file"
        assert steps[1].tool_name == "write_file"

    def test_import_from_jsonl_file(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()

        with tempfile.TemporaryDirectory() as tmp:
            log_file = write_log_file(MCP_JSONRPC_LOG, Path(tmp), "jsonl")
            result = importer.import_from_file(log_file, force=True)

            assert result.steps_imported == 2
            assert result.error is None

    def test_import_from_json_file(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()

        with tempfile.TemporaryDirectory() as tmp:
            log_file = write_log_file(MCP_SIMPLE_LOG, Path(tmp), "json")
            result = importer.import_from_file(log_file, force=True)

            assert result.steps_imported == 2

    def test_empty_log_skipped(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()
        steps, metadata = importer.parse_log([])
        assert len(steps) == 0

    def test_tools_extracted_to_metadata(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()
        _, metadata = importer.parse_log(MCP_JSONRPC_LOG)

        assert "read_file" in metadata.get("tools_used", [])
        assert "write_file" in metadata.get("tools_used", [])

    def test_jsonrpc_error_handling(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        error_log = [
            {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "bad_tool", "arguments": {}}, "id": 99, "timestamp": "2024-11-15T10:00:00Z"},
            {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid request"}, "id": 99, "timestamp": "2024-11-15T10:00:01Z"},
        ]

        importer = MCPLogImporter()
        steps, _ = importer.parse_log(error_log)

        assert len(steps) == 1
        assert steps[0].tool_name == "bad_tool"
        assert steps[0].success is False

    def test_skip_already_imported(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()

        with tempfile.TemporaryDirectory() as tmp:
            log_file = write_log_file(MCP_SIMPLE_LOG, Path(tmp), "json")

            # Force first import in case previous runs left state
            result1 = importer.import_from_file(log_file, force=True)
            assert result1.steps_imported == 2

            result2 = importer.import_from_file(log_file)
            assert result2.skipped is True
            assert result2.steps_imported == 0

    def test_force_reimport(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()

        with tempfile.TemporaryDirectory() as tmp:
            log_file = write_log_file(MCP_SIMPLE_LOG, Path(tmp), "json")

            result1 = importer.import_from_file(log_file, force=True)
            assert result1.steps_imported == 2

            result2 = importer.import_from_file(log_file, force=True)
            assert result2.skipped is False
            assert result2.steps_imported == 2

    def test_invalid_json_returns_error(self):
        from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

        importer = MCPLogImporter()

        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "bad.json"
            bad_file.write_text("this is not json {{{")

            result = importer.import_from_file(bad_file)
            assert result.error is not None
            assert result.steps_imported == 0
