import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.hf_context_routes import router
from bashgym.api.workspace_routes import router as workspace_router
from bashgym.integrations.huggingface.context_persistence import HFContextRepository
from bashgym.integrations.huggingface.context_service import HFContextService

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hf_context"


class FakeSources:
    def discover_models(self, query: str, *, limit: int):
        return [json.loads((FIXTURES / "rich_model.json").read_text(encoding="utf-8"))]

    def discover_datasets(self, query: str, *, limit: int):
        return [json.loads((FIXTURES / "multi_config_dataset.json").read_text(encoding="utf-8"))]


def client(tmp_path):
    repository = HFContextRepository(tmp_path / "hf-context.sqlite3")
    repository.initialize()
    service = HFContextService(repository, sources=FakeSources())
    app = FastAPI()
    app.state.hf_context_service = service
    app.include_router(router)
    app.include_router(workspace_router)
    return TestClient(app)


def discover_ready(http: TestClient, *, workspace_id: str = "workspace-a"):
    accepted = http.post(
        "/api/hf/context/discover",
        json={"workspace_id": workspace_id, "intent": "code generation"},
    )
    assert accepted.status_code == 202
    collecting = accepted.json()
    assert collecting["lifecycle"] == "collecting"
    ready = http.get(
        f"/api/hf/context/bundles/{collecting['bundle_id']}/versions/1",
        params={"workspace_id": workspace_id},
    )
    assert ready.status_code == 200
    assert ready.json()["lifecycle"] == "ready"
    return ready.json()


def test_context_route_flow_is_workspace_scoped_and_returns_structured_markdown(tmp_path):
    http = client(tmp_path)
    bundle = discover_ready(http)

    history = http.get("/api/hf/context/bundles", params={"workspace_id": "workspace-a"})
    assert history.status_code == 200
    assert len(history.json()["bundles"]) == 1

    markdown = http.get(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1/markdown",
        params={"workspace_id": "workspace-a"},
    )
    assert markdown.status_code == 200
    assert markdown.json()["renderer_version"] == "hf-context-markdown-v1"
    assert "UNTRUSTED EXTERNAL EVIDENCE" not in markdown.json()["markdown"]  # no raw excerpts

    denied = http.get(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1",
        params={"workspace_id": "workspace-b"},
    )
    assert denied.status_code == 404
    assert denied.json()["detail"]["code"] == "hf_bundle_not_found"


def test_pin_conflict_uses_stable_error_code_and_activation_is_exact(tmp_path):
    http = client(tmp_path)
    bundle = discover_ready(http)
    selected = bundle["evidence"][0]["evidence_id"]

    pinned = http.post(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1/pin",
        json={
            "workspace_id": "workspace-a",
            "expected_version": 1,
            "selected_evidence_ids": [selected],
        },
    )
    assert pinned.status_code == 201
    assert pinned.json()["version"] == 2

    conflict = http.post(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1/pin",
        json={
            "workspace_id": "workspace-a",
            "expected_version": 1,
            "selected_evidence_ids": [selected],
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "hf_bundle_conflict"

    activated = http.post(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/2/activate",
        json={"workspace_id": "workspace-a"},
    )
    assert activated.status_code == 200
    assert activated.json()["version"] == 2

    deactivated = http.request(
        "DELETE", "/api/hf/context/active", json={"workspace_id": "workspace-a"}
    )
    assert deactivated.status_code == 200


def test_eval_action_is_preview_only_and_idempotent(tmp_path):
    http = client(tmp_path)
    bundle = discover_ready(http)
    path = f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1/actions/eval"

    first = http.post(path, json={"workspace_id": "workspace-a"})
    second = http.post(path, json={"workspace_id": "workspace-a"})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["execute"] is False


def test_workspace_context_exposes_only_active_visibility_safe_summary(tmp_path):
    http = client(tmp_path)
    bundle = discover_ready(http)
    http.post(
        f"/api/hf/context/bundles/{bundle['bundle_id']}/versions/1/activate",
        json={"workspace_id": "workspace-a"},
    )

    context = http.get("/api/workspace/context", params={"workspace_id": "workspace-a"}).json()
    summary = context["huggingface_context"]
    assert summary["bundle_id"] == bundle["bundle_id"]
    assert summary["evidence_counts"] == {"dataset": 1, "evaluation": 1, "model": 1}
    assert "evidence" not in summary
    assert "excerpt" not in json.dumps(summary)


def test_refresh_creates_collecting_successor_and_cancel_finalizes_collecting(tmp_path):
    http = client(tmp_path)
    first = discover_ready(http)

    refreshed = http.post(
        f"/api/hf/context/bundles/{first['bundle_id']}/versions/1/refresh",
        json={"workspace_id": "workspace-a", "expected_version": 1},
    )
    assert refreshed.status_code == 202
    assert refreshed.json()["version"] == 2
    assert refreshed.json()["lifecycle"] == "collecting"
    ready = http.get(
        f"/api/hf/context/bundles/{first['bundle_id']}/versions/2",
        params={"workspace_id": "workspace-a"},
    ).json()
    assert ready["lifecycle"] == "ready"

    service = http.app.state.hf_context_service
    collecting, _ = service.begin_refresh("workspace-a", first["bundle_id"], 2, expected_version=2)
    cancelled = http.post(
        f"/api/hf/context/bundles/{first['bundle_id']}/versions/{collecting.version}/cancel",
        json={"workspace_id": "workspace-a"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["completion_outcome"] == "cancelled"
