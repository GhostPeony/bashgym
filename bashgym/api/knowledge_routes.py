"""Read-only knowledge source inspection for the canvas Knowledge Base node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
}
SKIP_NAMES = {
    ".git",
    ".env",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "coverage",
}
MAX_FILE_BYTES = 512 * 1024


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectInput(ApiModel):
    workspace_id: str
    provider: Literal["workspace", "gbrain"] = "workspace"
    root: str | None = Field(default=None, max_length=4096)
    max_depth: int = Field(default=4, ge=1, le=8)
    max_entries: int = Field(default=500, ge=10, le=2000)


class SearchInput(ApiModel):
    workspace_id: str
    provider: Literal["workspace", "gbrain"] = "workspace"
    root: str | None = Field(default=None, max_length=4096)
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=50)


class PreviewInput(ApiModel):
    workspace_id: str
    provider: Literal["workspace", "gbrain"] = "workspace"
    root: str | None = Field(default=None, max_length=4096)
    path: str = Field(min_length=1, max_length=4096)
    max_chars: int = Field(default=24_000, ge=500, le=100_000)


def _workspace_root(request: Request) -> Path:
    observer = getattr(request.app.state, "runtime_observer", None)
    configured = getattr(observer, "workspace_root", None)
    return Path(configured or Path.cwd()).expanduser().resolve()


def _gbrain_sources() -> list[dict[str, Any]]:
    executable = shutil.which("gbrain")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [executable, "sources", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode != 0:
            return []
        payload = json.loads(completed.stdout or "{}")
        rows = payload.get("sources", payload if isinstance(payload, list) else [])
        return [row for row in rows if isinstance(row, dict)]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def _allowed_roots(request: Request) -> list[Path]:
    roots = [_workspace_root(request)]
    gbrain_home = Path.home() / ".gbrain"
    if gbrain_home.exists():
        roots.append(gbrain_home.resolve())
    for source in _gbrain_sources():
        raw_path = source.get("path") or source.get("root")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path).expanduser()
            if candidate.exists():
                roots.append(candidate.resolve())
    return list(dict.fromkeys(roots))


def _resolve_root(request: Request, root: str | None) -> Path:
    candidate = Path(root).expanduser().resolve() if root else _workspace_root(request)
    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Knowledge source directory not found")
    for allowed in _allowed_roots(request):
        try:
            candidate.relative_to(allowed)
            return candidate
        except ValueError:
            continue
    raise HTTPException(
        status_code=403,
        detail="Knowledge sources must be inside the workspace or a discovered GBrain source",
    )


def _safe_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _walk_tree(
    root: Path, *, max_depth: int, max_entries: int
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    counts = {"files": 0, "folders": 0, "knowledge_files": 0}
    truncated = [False]

    def visit(directory: Path, depth: int, budget: int) -> list[dict[str, Any]]:
        if depth > max_depth or budget <= 0:
            truncated[0] = True
            return []
        try:
            children = sorted(
                (
                    item
                    for item in directory.iterdir()
                    if item.name not in SKIP_NAMES and not item.name.startswith(".")
                ),
                key=lambda item: (not item.is_dir(), item.name.casefold()),
            )
        except OSError:
            return []
        direct_limit = budget if depth >= max_depth else max(1, budget // 2)
        visible = children[:direct_limit]
        if len(children) > len(visible):
            truncated[0] = True
        folder_count = sum(1 for item in visible if item.is_dir())
        descendant_budget = max(0, budget - len(visible))
        per_folder_budget = descendant_budget // folder_count if folder_count else 0
        nodes: list[dict[str, Any]] = []
        for item in visible:
            relative = _safe_relative(root, item)
            if item.is_dir():
                counts["folders"] += 1
                nodes.append(
                    {
                        "name": item.name,
                        "path": relative,
                        "type": "folder",
                        "children": visit(item, depth + 1, per_folder_budget),
                    }
                )
            elif item.is_file():
                counts["files"] += 1
                if item.suffix.casefold() in TEXT_SUFFIXES:
                    counts["knowledge_files"] += 1
                nodes.append(
                    {
                        "name": item.name,
                        "path": relative,
                        "type": "file",
                        "knowledge": item.suffix.casefold() in TEXT_SUFFIXES,
                        "size_bytes": item.stat().st_size if item.exists() else 0,
                    }
                )
        return nodes

    return visit(root, 1, max_entries), counts, truncated[0]


def _read_text(path: Path, max_chars: int) -> str:
    if path.suffix.casefold() not in TEXT_SUFFIXES:
        raise HTTPException(
            status_code=415, detail="Preview is limited to safe text and source files"
        )
    if path.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File is too large to preview")
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Could not read this knowledge file") from exc


@router.get("/status")
def knowledge_status(request: Request, workspace_id: str = Query(...)) -> dict[str, Any]:
    del workspace_id
    executable = shutil.which("gbrain")
    sources = _gbrain_sources()
    return {
        "workspace_root": str(_workspace_root(request)),
        "gbrain": {
            "installed": bool(executable),
            "executable": executable,
            "configured": (Path.home() / ".gbrain" / "config.json").exists(),
            "sources": sources,
        },
        "adapters": [
            {"id": "workspace", "label": "Workspace files", "available": True, "mode": "local"},
            {
                "id": "gbrain",
                "label": "GBrain",
                "available": bool(executable or sources),
                "mode": "local-cli",
            },
        ],
    }


@router.post("/inspect")
def inspect_knowledge_source(request: Request, body: InspectInput) -> dict[str, Any]:
    root = _resolve_root(request, body.root)
    tree, counts, truncated = _walk_tree(
        root,
        max_depth=body.max_depth,
        max_entries=body.max_entries,
    )
    return {
        "provider": body.provider,
        "root": str(root),
        "label": root.name or str(root),
        "tree": tree,
        "counts": counts,
        "truncated": truncated,
        "capabilities": ["browse", "preview", "keyword_search", "canvas_context"],
    }


@router.post("/preview")
def preview_knowledge_file(request: Request, body: PreviewInput) -> dict[str, Any]:
    root = _resolve_root(request, body.root)
    candidate = (root / body.path).resolve()
    try:
        relative = _safe_relative(root, candidate)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path is outside the selected source") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Knowledge file not found")
    content = _read_text(candidate, body.max_chars)
    return {"path": relative, "content": content, "truncated": len(content) >= body.max_chars}


@router.post("/search")
def search_knowledge_source(request: Request, body: SearchInput) -> dict[str, Any]:
    root = _resolve_root(request, body.root)
    query = body.query.casefold()
    results: list[dict[str, Any]] = []
    scanned = 0
    for candidate in root.rglob("*"):
        if len(results) >= body.limit or scanned >= 2000:
            break
        if any(
            part in SKIP_NAMES or part.startswith(".") for part in candidate.relative_to(root).parts
        ):
            continue
        if not candidate.is_file() or candidate.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        lowered = text.casefold()
        position = lowered.find(query)
        if position < 0 and query not in candidate.name.casefold():
            continue
        start = max(0, position - 120) if position >= 0 else 0
        snippet = " ".join(text[start : start + 360].split())
        results.append({"path": _safe_relative(root, candidate), "snippet": snippet})
    return {
        "provider": body.provider,
        "root": str(root),
        "query": body.query,
        "results": results,
        "scanned_files": scanned,
        "truncated": len(results) >= body.limit or scanned >= 2000,
    }
