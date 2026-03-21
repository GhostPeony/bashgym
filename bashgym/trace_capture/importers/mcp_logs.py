"""MCP (Model Context Protocol) tool log importer.

Parses MCP client logs and converts tool calls into TraceSession objects.
Supports two formats:
1. JSON-RPC (standard MCP wire format): request/response pairs matched by id
2. Simple (pre-paired): objects with tool, arguments, result fields
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import RepoInfo, TraceCapture, TraceSession, TraceStep


@dataclass
class MCPImportResult:
    """Result of importing MCP tool logs."""

    session_id: str
    steps_imported: int
    destination_file: Path | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class MCPLogImporter:
    """Import MCP tool call logs into Bash Gym trace format."""

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.imported_file = self.trace_capture.bashgym_dir / "imported_mcp.json"
        self._imported: set | None = None
        self._dummy_repo = RepoInfo(path="", name="mcp", is_git_repo=False)

    def _load_imported(self) -> set:
        if self._imported is None:
            if self.imported_file.exists():
                data = json.loads(self.imported_file.read_text())
                self._imported = set(data.get("imported_ids", []))
            else:
                self._imported = set()
        return self._imported

    def _mark_imported(self, log_id: str):
        imported = self._load_imported()
        imported.add(log_id)
        self.imported_file.parent.mkdir(parents=True, exist_ok=True)
        self.imported_file.write_text(json.dumps({"imported_ids": list(imported)}))

    def _is_jsonrpc(self, entries: list[dict]) -> bool:
        """Detect if log entries are JSON-RPC format."""
        return any(e.get("jsonrpc") == "2.0" for e in entries if isinstance(e, dict))

    def _parse_jsonrpc(self, entries: list[dict]) -> list[TraceStep]:
        """Parse JSON-RPC request/response pairs into steps."""
        requests = {}
        responses = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if entry_id is None:
                continue
            if "method" in entry:
                requests[entry_id] = entry
            elif "result" in entry or "error" in entry:
                responses[entry_id] = entry

        steps = []
        for req_id in sorted(requests.keys(), key=lambda x: requests[x].get("timestamp", "")):
            req = requests[req_id]
            params = req.get("params", {})
            tool_name = params.get("name", "unknown")
            arguments = params.get("arguments", {})
            timestamp = req.get("timestamp", datetime.now(timezone.utc).isoformat())

            resp = responses.get(req_id, {})
            result_content = resp.get("result", {})
            if isinstance(result_content, dict):
                content_parts = result_content.get("content", [])
                if isinstance(content_parts, list):
                    output = "\n".join(
                        p.get("text", str(p)) for p in content_parts if isinstance(p, dict)
                    )
                else:
                    output = str(result_content)
            else:
                output = str(result_content)

            error = resp.get("error")
            success = error is None

            step = TraceStep.create(
                tool_name=tool_name,
                command=json.dumps(arguments) if arguments else "",
                output=output,
                source_tool="mcp",
                repo_info=self._dummy_repo,
                format="jsonrpc",
                request_id=req_id,
            )
            step.success = success
            step.timestamp = timestamp
            steps.append(step)

        return steps

    def _parse_simple(self, entries: list[dict]) -> list[TraceStep]:
        """Parse simple pre-paired log entries into steps."""
        steps = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tool_name = entry.get("tool", entry.get("name", "unknown"))
            arguments = entry.get("arguments", entry.get("params", {}))
            result = entry.get("result", entry.get("output", ""))
            timestamp = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
            duration_ms = entry.get("duration_ms")

            step = TraceStep.create(
                tool_name=tool_name,
                command=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                output=str(result),
                source_tool="mcp",
                repo_info=self._dummy_repo,
                format="simple",
                duration_ms=duration_ms,
            )
            step.success = not entry.get("error")
            step.timestamp = timestamp
            steps.append(step)

        return steps

    def parse_log(self, entries: list[dict]) -> tuple[list[TraceStep], dict[str, Any]]:
        """Parse MCP log entries into TraceSteps + metadata."""
        if not entries:
            return [], {"source": "mcp"}

        if self._is_jsonrpc(entries):
            steps = self._parse_jsonrpc(entries)
        else:
            steps = self._parse_simple(entries)

        tools_used = list(set(s.tool_name for s in steps))

        metadata = {
            "source": "mcp",
            "tools_used": tools_used,
            "total_tool_calls": len(steps),
            "imported": True,
            "import_source": "mcp_log",
        }

        return steps, metadata

    def import_from_file(self, file_path: Path, force: bool = False) -> MCPImportResult:
        """Import MCP logs from a JSON or JSONL file."""
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]

        if not force and file_hash in self._load_imported():
            return MCPImportResult(
                session_id=file_hash,
                steps_imported=0,
                skipped=True,
                skip_reason="already_imported",
            )

        content = file_path.read_text()

        # Try JSON array first (most common for simple format)
        entries = []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]
        except json.JSONDecodeError:
            # Fall back to JSONL (one JSON object per line)
            try:
                for line in content.strip().splitlines():
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                return MCPImportResult(
                    session_id=file_hash,
                    steps_imported=0,
                    error=f"Failed to parse log file: {e}",
                )

        steps, metadata = self.parse_log(entries)
        if not steps:
            return MCPImportResult(
                session_id=file_hash,
                steps_imported=0,
                skipped=True,
                skip_reason="no_tool_calls_found",
            )

        session = TraceSession.from_steps(
            steps=steps,
            source_tool="mcp",
            **metadata,
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imported_mcp_{file_hash}_{timestamp_str}.json"
        dest = self.trace_capture.traces_dir / filename
        self.trace_capture.traces_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(asdict(session), default=str, indent=2))

        self._mark_imported(file_hash)

        return MCPImportResult(
            session_id=session.session_id,
            steps_imported=len(steps),
            destination_file=dest,
        )


# Module-level convenience
def import_mcp_logs(file_path=None, force=False, **kwargs):
    """Import MCP tool logs from a file."""
    if not file_path:
        return []
    importer = MCPLogImporter()
    result = importer.import_from_file(Path(file_path), force=force)
    return [result]
