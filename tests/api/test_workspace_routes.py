from fastapi.testclient import TestClient

from bashgym.api import workspace_routes
from bashgym.api.routes import app
from bashgym.api.runtime_observer import RuntimeObserver
from bashgym.campaigns.contracts import (
    Campaign,
    CampaignKind,
    CampaignManifest,
    CredentialKind,
    ManifestRevision,
    TargetModelContract,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository


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
            "allowed_actions": ["workspace.context.read", "hf_context.search"],
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
    assert "hf_context.search" in context["allowed_actions"]


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
    assert "## Evidence Authority" in text
    assert "live runtime > durable ledger" in text
    assert "SFT Run" in text
    assert "run-x" in text


def test_workspace_context_exposes_provenance_and_conflicts(monkeypatch, tmp_path):
    _reset_state()
    monkeypatch.setattr("bashgym.api.training_state.list_run_states", lambda *_args: [])
    app.state.workspace_canvas_snapshots = {
        "default": workspace_routes.WorkspaceCanvasSnapshot(
            updated_at="2026-07-14T20:00:00+00:00"
        )
    }
    app.state.training_runs = {
        "run-x": {
            "run_id": "run-x",
            "status": "running",
            "strategy": "sft",
            "config": {},
        }
    }
    observer = RuntimeObserver(tmp_path)
    monkeypatch.setattr(
        observer,
        "list_jobs",
        lambda: [{"job_id": "run-x", "status": "completed", "kind": "training"}],
    )
    app.state.runtime_observer = observer

    context = _client().get("/api/workspace/context").json()

    assert context["schema_version"] == "bashgym.workspace.context.v2"
    assert context["authority"]["source_precedence"][0]["source_id"] == "live_runtime"
    assert context["authority"]["conflicts"][0]["code"] == "run_status_mismatch"


def test_workspace_context_projects_scoped_runs_campaigns_runtime_and_reports(
    tmp_path, monkeypatch
):
    _reset_state()
    monkeypatch.setattr("bashgym.api.training_state.list_run_states", lambda *_args: [])
    app.state.workspace_canvas_snapshots = {
        "workspace-a": workspace_routes.WorkspaceCanvasSnapshot(
            workspace_id="workspace-a", workspace_name="Research"
        )
    }
    app.state.training_runs = {
        "run-a": {
            "run_id": "run-a",
            "status": "running",
            "strategy": "sft",
            "started_at": "2026-07-13T10:00:00Z",
            "origin": {"workspace_id": "workspace-a"},
            "config": {},
        },
        "run-b": {
            "run_id": "run-b",
            "status": "running",
            "strategy": "dpo",
            "started_at": "2026-07-13T11:00:00Z",
            "origin": {"workspace_id": "workspace-b"},
            "config": {},
        },
    }

    observer = RuntimeObserver(tmp_path)
    monkeypatch.setattr(
        observer,
        "list_jobs",
        lambda: [
            {
                "job_id": "runtime-1",
                "kind": "training",
                "status": "running",
                "title": "SFT Training",
                "script": str(tmp_path / "train.py"),
                "cwd": str(tmp_path),
                "started_at": "2026-07-13T10:00:00Z",
                "log_path": str(tmp_path / "private.log"),
                "output_dir": str(tmp_path / "output"),
                "progress": {"current": 12, "total": 100, "unit": "steps"},
                "artifacts": [
                    {
                        "name": "metrics.jsonl",
                        "path": str(tmp_path / "output" / "metrics.jsonl"),
                        "size": 42,
                        "modified_at": "2026-07-13T10:01:00Z",
                    }
                ],
                "source": "process_observer",
            }
        ],
    )
    app.state.runtime_observer = observer

    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    campaign = Campaign(
        campaign_id="general-session",
        workspace_id="workspace-a",
        title="General SFT iteration",
        kind=CampaignKind.GENERAL,
        objective="Improve instruction following without regressing held-out quality.",
        target_model=TargetModelContract(
            target_contract_key="general-sft-v1",
            base_model_ref="org/base-model",
            task="instruction-following",
        ),
        owner_actor_id="hermes-agent",
    )
    manifest = CampaignManifest(
        approved_data_scopes=("approved-sft-data",),
        compute_profile_id="ssh-gpu-lab",
        budget_limits={"gpu_hours": 4.0, "study_count": 2.0},
        evaluation_plan={"suite": "heldout-v1"},
        promotion_gates={"quality_delta_min": 0.0},
    )
    created = repository.create_campaign(
        campaign,
        ManifestRevision(
            workspace_id="workspace-a",
            campaign_id="general-session",
            revision=1,
            manifest=manifest,
            actor_id="hermes-agent",
            correlation_id="session-create",
        ),
        actor_id="hermes-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="session-create",
        idempotency_key="session-create",
    )
    repository.record_export(
        "workspace-a",
        "general-session",
        "export-general-session",
        ("markdown", "pdf"),
        {
            "source_digest": "a" * 64,
            "quality_findings_available": True,
            "files": [
                {"name": "campaign_report.pdf", "sha256": "b" * 64, "size_bytes": 1024}
            ],
        },
        expected_version=created.campaign.version,
        actor_id="hermes-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="session-export",
        idempotency_key="session-export",
    )
    app.state.campaign_repository = repository

    from bashgym.ledger.contracts import ExperimentSpec, ProjectSpec, RunSpec
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    ledger = ExperimentLedgerRepository(repository.db_path)
    ledger.initialize()
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="general-llm",
            display_name="General LLM work",
            owner_actor_id="hermes-agent",
        )
    )
    ledger.register_experiment(
        ExperimentSpec(
            workspace_id="workspace-a",
            project_id="general-llm",
            experiment_id="sft-experiment",
            campaign_id="general-session",
            name="Instruction following",
            objective="Improve instruction following.",
        )
    )
    ledger.register_run(
        RunSpec(
            workspace_id="workspace-a",
            project_id="general-llm",
            experiment_id="sft-experiment",
            campaign_id="general-session",
            run_id="ledger-run-a",
            source_system="bashgym",
            source_run_id="run-a",
            run_kind="training",
            task_type="instruction-following",
            training_method="sft",
            recipe_digest="f" * 64,
            correlation_id="workspace-context-test",
        )
    )

    response = _client().get("/api/workspace/context?workspace_id=workspace-a")

    assert response.status_code == 200
    context = response.json()
    assert [run["run_id"] for run in context["training_runs"]] == ["run-a"]
    assert context["runtime_jobs"][0]["script"] == "train.py"
    assert "private.log" not in str(context["runtime_jobs"])
    assert context["campaigns"][0]["kind"] == "general"
    assert context["campaigns"][0]["objective"].startswith("Improve instruction")
    assert context["campaigns"][0]["latest_event_cursor"] == 2
    assert context["experiment_projects"][0]["project_id"] == "general-llm"
    assert context["experiment_projects"][0]["recent_runs"][0]["run_id"] == "ledger-run-a"
    assert context["report_refs"][0]["files"][0]["name"] == "campaign_report.pdf"
    assert "campaign.export" in context["allowed_actions"]
    assert all("masked_value" not in item for item in context["settings_readiness"])
