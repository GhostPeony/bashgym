"""BashGym Sandbox MCP Server.

Exposes the agent execution sandbox as Model Context Protocol (MCP) tools over
stdio, so NeMo Data Designer's tool-use columns can drive *real* tool execution
while generating tool-use training trajectories.

Backends (selected by ``BASHGYM_MCP_BACKEND`` = ``auto`` | ``docker`` | ``local``;
default ``auto``):
  - **docker**: reuse ``SandboxManager`` (network-off container, dangerous-command
    guard, ``/workspace``). The real agent environment.
  - **local**: a confined temp-workspace fallback (path-confined file ops,
    dangerous-command guard, per-call timeout) for when Docker is unavailable.

Run as a subprocess by Data Designer via::

    LocalStdioMCPProvider(name="bashgym-sandbox", command=sys.executable,
                          args=["-m", "bashgym.mcp.sandbox_server"])

Tools: ``bash``, ``read_file``, ``write_file``, ``edit_file``, ``grep``,
``list_files``. File tools are path-confined to the workspace.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bashgym.arena.sandbox import is_dangerous_command

_MAX_OUTPUT = 8000  # cap tool output to keep transcripts manageable


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... (truncated, {len(text) - limit} more chars)"
    return text


# =========================================================================
# Workspace backends
# =========================================================================


class LocalWorkspace:
    """Confined temp-workspace executor (Docker-free fallback)."""

    def __init__(self, timeout_sec: int = 120):
        self.root = Path(tempfile.mkdtemp(prefix="bashgym-mcp-")).resolve()
        self.timeout_sec = timeout_sec

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path escapes workspace: {path!r}")
        return p

    def bash(self, command: str) -> str:
        if is_dangerous_command(command):
            return "BLOCKED: dangerous command refused by safety guard"
        # Minimal env: keep PATH, drop proxy/credential vars. (Network is not
        # hard-blocked in local mode; the docker backend is the isolated path.)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.root)}
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {self.timeout_sec}s"
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return _truncate(f"[exit {r.returncode}]\n{out}")

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: no such file: {path}"
        return _truncate(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            return f"ERROR: no such file: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return "ERROR: old string not found"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path}"

    def grep(self, pattern: str, path: str = ".") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        root = self._resolve(path)
        hits = []
        files = [root] if root.is_file() else [f for f in root.rglob("*") if f.is_file()]
        for f in files:
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(self.root)}:{i}:{line}")
            except OSError:
                continue
        return _truncate("\n".join(hits) or "(no matches)")

    def list_files(self, path: str = ".") -> str:
        root = self._resolve(path)
        if not root.exists():
            return f"ERROR: no such path: {path}"
        entries = [str(p.relative_to(self.root)) for p in sorted(root.rglob("*"))]
        return _truncate("\n".join(entries) or "(empty)")

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class DockerWorkspace:
    """Real Docker sandbox executor (SandboxManager-backed)."""

    def __init__(self, timeout_sec: int = 120):
        from bashgym.arena.sandbox import SandboxManager

        self.timeout_sec = timeout_sec
        self.mgr = SandboxManager()
        self.sid = self.mgr.create_sandbox()
        self.mgr.start_sandbox(self.sid)

    def _run(self, command: str) -> str:
        r = self.mgr.execute_command(self.sid, command, timeout=self.timeout_sec)
        if r.get("blocked"):
            return "BLOCKED: dangerous command refused by safety guard"
        out = (r.get("stdout") or "") + (("\n[stderr]\n" + r["stderr"]) if r.get("stderr") else "")
        return _truncate(f"[exit {r.get('exit_code')}]\n{out}")

    def bash(self, command: str) -> str:
        return self._run(command)

    def read_file(self, path: str) -> str:
        import shlex

        return self._run(f"cat {shlex.quote(path)}")

    def write_file(self, path: str, content: str) -> str:
        import base64
        import shlex

        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        q = shlex.quote(path)
        self._run(f"mkdir -p $(dirname {q}) && printf %s {shlex.quote(b64)} | base64 -d > {q}")
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        import shlex

        current = self.mgr.execute_command(
            self.sid, f"cat {shlex.quote(path)}", timeout=self.timeout_sec
        )
        text = current.get("stdout") or ""
        if old not in text:
            return "ERROR: old string not found"
        return self.write_file(path, text.replace(old, new, 1)) and f"edited {path}"

    def grep(self, pattern: str, path: str = ".") -> str:
        import shlex

        return self._run(f"grep -rn -- {shlex.quote(pattern)} {shlex.quote(path)} || true")

    def list_files(self, path: str = ".") -> str:
        return self._run(f"find {path} -type f 2>/dev/null | head -500")

    def close(self) -> None:
        try:
            self.mgr.cleanup_sandbox(self.sid, remove_workspace=True)
        except Exception:
            pass


def build_workspace(backend: str | None = None, timeout_sec: int = 120):
    """Build the workspace for the requested backend.

    ``auto`` (default) tries Docker and falls back to the local executor.
    """
    backend = (backend or os.environ.get("BASHGYM_MCP_BACKEND", "auto")).lower()
    if backend in ("docker", "auto"):
        try:
            return DockerWorkspace(timeout_sec=timeout_sec)
        except Exception:
            if backend == "docker":
                raise
    return LocalWorkspace(timeout_sec=timeout_sec)


# =========================================================================
# MCP server (FastMCP over stdio)
# =========================================================================


def build_server(workspace=None):
    """Build the FastMCP server with sandbox tools bound to a workspace."""
    from mcp.server.fastmcp import FastMCP

    ws = workspace or build_workspace()
    server = FastMCP("bashgym-sandbox")

    @server.tool()
    def bash(command: str) -> str:
        """Run a shell command in the workspace; returns exit code + combined output."""
        return ws.bash(command)

    @server.tool()
    def read_file(path: str) -> str:
        """Read a file from the workspace (path is relative to the workspace root)."""
        return ws.read_file(path)

    @server.tool()
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the workspace."""
        return ws.write_file(path, content)

    @server.tool()
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of ``old`` with ``new`` in a workspace file."""
        return ws.edit_file(path, old, new)

    @server.tool()
    def grep(pattern: str, path: str = ".") -> str:
        """Search workspace files for a regex pattern; returns file:line:match hits."""
        return ws.grep(pattern, path)

    @server.tool()
    def list_files(path: str = ".") -> str:
        """List files under a workspace path."""
        return ws.list_files(path)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
