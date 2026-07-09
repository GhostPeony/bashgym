from fastapi.testclient import TestClient

from bashgym.api import workspace_routes
from bashgym.api.routes import app


def _client():
    return TestClient(app)


def _reset_state():
    app.state.workspace_canvas_snapshots = {}
    app.state.workspace_events = {}


def test_workspace_snapshot_redacts_secret_shaped_config(monkeypatch):
    captured = []

    async def fake_context_updated(payload):
        captured.append(payload)

    monkeypatch.setattr(
        workspace_routes, "broadcast_workspace_context_updated", fake_context_updated
    )
    _reset_state()

    response = _client().put(
        "/api/workspace/canvas/snapshot",
        json={
            "panels": [
                {
                    "panel_id": "agent-1",
                    "type": "agent",
                    "title": "Hermes",
                    "adapter_config": {
                        "api_key": "sk-secret",
                        "nested": {"token": "hf_secret"},
                        "model": "hermes-agent",
                    },
                }
            ],
            "edges": [],
            "terminals": [],
            "data_summaries": {},
            "allowed_actions": ["workspace.context.read"],
        },
    )

    assert response.status_code == 200
    assert captured[-1]["panels"] == 1
    assert captured[-1]["workspace_id"] == "default"

    context = _client().get("/api/workspace/context").json()
    config = context["canvas"]["panels"][0]["adapter_config"]
    assert config["api_key"] == "[redacted]"
    assert config["nested"]["token"] == "[redacted]"
    assert config["model"] == "hermes-agent"
    assert "sk-secret" not in str(context)


def test_workspace_event_is_stored_and_broadcast(monkeypatch):
    captured = []

    async def fake_canvas_intent(payload):
        captured.append(payload)

    monkeypatch.setattr(workspace_routes, "broadcast_workspace_canvas_intent", fake_canvas_intent)
    _reset_state()

    response = _client().post(
        "/api/workspace/events",
        json={
            "type": "training.prep.started",
            "source": {"kind": "terminal", "terminal_id": "term-1", "agent": "codex"},
            "title": "Prepare SFT run",
            "summary": "Agent is preparing a run",
            "entity": {"kind": "training_run", "strategy": "sft"},
            "suggested_nodes": [
                {"recipe": "training.run", "title": "SFT Run", "config": {"strategy": "sft"}}
            ],
        },
    )

    assert response.status_code == 200
    event = response.json()["event"]
    assert event["event_id"].startswith("workspace_evt_")
    assert event["correlation_id"].startswith("intent_")
    assert captured[-1]["type"] == "training.prep.started"
    assert app.state.workspace_events["default"][-1]["event_id"] == event["event_id"]


def test_workspace_event_broadcast_carries_workspace_id(monkeypatch):
    captured = []

    async def fake_canvas_intent(payload):
        captured.append(payload)

    monkeypatch.setattr(workspace_routes, "broadcast_workspace_canvas_intent", fake_canvas_intent)
    _reset_state()

    response = _client().post(
        "/api/workspace/events",
        json={"type": "node.suggest", "workspace_id": "ws-frontend"},
    )

    assert response.status_code == 200
    assert captured[-1]["workspace_id"] == "ws-frontend"
    assert app.state.workspace_events["ws-frontend"][-1]["type"] == "node.suggest"
    assert "default" not in app.state.workspace_events


def test_workspace_snapshots_are_isolated_per_workspace(monkeypatch):
    async def fake_context_updated(payload):
        pass

    monkeypatch.setattr(
        workspace_routes, "broadcast_workspace_context_updated", fake_context_updated
    )
    _reset_state()
    client = _client()

    for ws_id, title in [("ws-a", "Training"), ("ws-b", "Frontend")]:
        response = client.put(
            "/api/workspace/canvas/snapshot",
            json={
                "workspace_id": ws_id,
                "workspace_name": title,
                "panels": [{"panel_id": f"{ws_id}-panel", "type": "terminal", "title": title}],
                "edges": [],
                "terminals": [],
            },
        )
        assert response.status_code == 200

    ctx_a = client.get("/api/workspace/context?workspace_id=ws-a").json()
    ctx_b = client.get("/api/workspace/context?workspace_id=ws-b").json()
    assert ctx_a["workspace_id"] == "ws-a"
    assert ctx_a["canvas"]["panels"][0]["title"] == "Training"
    assert ctx_b["workspace_id"] == "ws-b"
    assert ctx_b["canvas"]["panels"][0]["title"] == "Frontend"

    # Without an id the most recently updated workspace is returned
    latest = client.get("/api/workspace/context").json()
    assert latest["workspace_id"] == "ws-b"

    # Unknown id returns an empty snapshot for that id, not another workspace's
    ctx_missing = client.get("/api/workspace/context?workspace_id=nope").json()
    assert ctx_missing["workspace_id"] == "nope"
    assert ctx_missing["canvas"]["panels"] == []


def test_workspace_context_markdown_mentions_canvas_and_runs():
    _reset_state()
    app.state.workspace_canvas_snapshots = {
        "default": workspace_routes.WorkspaceCanvasSnapshot(
            panels=[
                workspace_routes.WorkspacePanel(
                    panel_id="training-1", type="training", title="SFT Run"
                )
            ]
        )
    }
    app.state.training_runs = {
        "run-x": {
            "run_id": "run-x",
            "status": "running",
            "strategy": "sft",
            "compute_target": "local",
            "config": {},
        }
    }

    response = _client().get("/api/workspace/context?format=markdown")

    assert response.status_code == 200
    text = response.text
    assert "# BashGym Workspace Context" in text
    assert "SFT Run" in text
    assert "run-x" in text
