from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.knowledge_routes import router


def _client(workspace_root):
    app = FastAPI()
    app.state.runtime_observer = SimpleNamespace(workspace_root=workspace_root)
    app.include_router(router)
    return TestClient(app)


def test_inspect_search_and_preview_workspace_knowledge(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text(
        "# Architecture\n\nKnowledge nodes publish cited context to agents.",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("SECRET=never-read", encoding="utf-8")
    (tmp_path / ".internal").mkdir()
    (tmp_path / ".internal" / "hidden.md").write_text("hidden", encoding="utf-8")
    client = _client(tmp_path)

    inspected = client.post(
        "/api/knowledge/inspect",
        json={"workspace_id": "main", "provider": "workspace"},
    )
    assert inspected.status_code == 200
    payload = inspected.json()
    assert payload["counts"]["knowledge_files"] == 1
    assert payload["tree"][0]["name"] == "docs"
    assert all(node["name"] != ".env" for node in payload["tree"])

    searched = client.post(
        "/api/knowledge/search",
        json={"workspace_id": "main", "provider": "workspace", "query": "cited context"},
    )
    assert searched.status_code == 200
    assert searched.json()["results"][0]["path"] == "docs/architecture.md"

    previewed = client.post(
        "/api/knowledge/preview",
        json={
            "workspace_id": "main",
            "provider": "workspace",
            "path": "docs/architecture.md",
        },
    )
    assert previewed.status_code == 200
    assert "Knowledge nodes" in previewed.json()["content"]


def test_preview_refuses_paths_outside_selected_source(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    client = _client(workspace)

    response = client.post(
        "/api/knowledge/preview",
        json={
            "workspace_id": "main",
            "provider": "workspace",
            "path": "../outside.md",
        },
    )
    assert response.status_code == 403
