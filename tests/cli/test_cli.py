import json
import sys
from pathlib import Path

from bashgym.cli import main
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    RolloutAttempt,
)
from bashgym.eval.dppo_replay import build_dppo_replay_records, write_dppo_records_jsonl


def test_manifest_json_is_agent_readable(capsys):
    assert main(["manifest", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "training plan" in payload["commands"]
    assert "campaign list" in payload["commands"]
    assert "campaign status" in payload["commands"]
    assert "campaign study status" in payload["commands"]
    assert "campaign attempts" in payload["commands"]
    assert "campaign comparisons" in payload["commands"]
    assert "campaign events" in payload["commands"]
    assert "campaign metrics" in payload["commands"]
    assert "campaign start" in payload["commands"]
    assert "campaign pause" in payload["commands"]
    assert "campaign resume" in payload["commands"]
    assert "campaign cancel" in payload["commands"]
    assert "ledger context" in payload["commands"]
    assert "ledger trend" in payload["commands"]
    assert "ledger compare" in payload["commands"]
    assert "ledger events" in payload["commands"]
    assert "training start" in payload["commands"]
    assert "designer start" in payload["commands"]
    assert "designer status" in payload["commands"]
    assert "training capabilities" in payload["commands"]
    assert "training analyze" in payload["commands"]
    assert "training dpo-pairs" in payload["commands"]
    assert "training reward-examples" in payload["commands"]
    assert "training reward-model" in payload["commands"]
    assert "training reward-eval" in payload["commands"]
    assert "sources list" in payload["commands"]
    assert "compute targets" in payload["commands"]
    assert "replay scrub" in payload["commands"]
    assert any(doc["topic"] == "capabilities" for doc in payload["docs"])
    assert any(doc["topic"] == "methods-reference" for doc in payload["docs"])
    assert any(doc["topic"] == "external-review" for doc in payload["docs"])
    assert any(doc["topic"] == "rlhf-handbook-comparison" for doc in payload["docs"])
    assert any(doc["topic"] == "terminal-rl-recipe" for doc in payload["docs"])
    assert any(doc["topic"] == "private-compute-checklist" for doc in payload["docs"])
    assert all("experiment-ledger.md" not in doc["path"] for doc in payload["docs"])
    assert any(doc["topic"] == "session-distillation" for doc in payload["docs"])
    assert any(doc["topic"] == "artifacts" for doc in payload["docs"])
    assert any(doc["topic"] == "world-models" for doc in payload["docs"])
    assert all(isinstance(doc["exists"], bool) for doc in payload["docs"])
    assert payload["next"][0]["command"].startswith("bashgym ")


def test_canvas_action_commands_send_origin_and_runtime_metadata(monkeypatch, capsys):
    captured = []

    def fake_request(args, path, *, method="GET", payload=None):
        captured.append({"path": path, "method": method, "payload": payload})
        return {"title": "started", "status": "queued", "run_id": "job-1"}

    monkeypatch.setattr("bashgym.cli._workspace_http_json", fake_request)
    monkeypatch.setenv("BASHGYM_TERMINAL_ID", "terminal-42")
    monkeypatch.setenv("BASHGYM_AGENT_KIND", "codex")

    assert (
        main(
            [
                "training",
                "start",
                "--strategy",
                "sft",
                "--model",
                "Qwen/Test",
                "--dataset-path",
                "data/train.jsonl",
                "--compute-target",
                "private",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert captured[-1] == {
        "path": "/training/start",
        "method": "POST",
        "payload": {
            "strategy": "sft",
            "base_model": "Qwen/Test",
            "dataset_path": "data/train.jsonl",
            "compute_target": "ssh:remote",
            "use_remote_ssh": True,
            "origin": {
                "kind": "terminal",
                "terminal_id": "terminal-42",
                "agent": "codex",
            },
        },
    }

    assert (
        main(
            [
                "designer",
                "start",
                "--pipeline",
                "coding_agent_sft",
                "--num-records",
                "20",
                "--seed-source",
                "data/gold.jsonl",
                "--model",
                "Hermes/Test",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert captured[-1]["path"] == "/factory/designer/create"
    assert captured[-1]["payload"]["origin"]["terminal_id"] == "terminal-42"
    assert captured[-1]["payload"]["text_model"] == "Hermes/Test"


def test_training_start_accepts_full_strategy_and_storage_config(monkeypatch, capsys, tmp_path):
    captured = []

    def fake_request(args, path, *, method="GET", payload=None):
        captured.append({"path": path, "method": method, "payload": payload})
        return {"title": "started", "status": "queued", "run_id": "job-storage"}

    config_path = tmp_path / "distillation.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": "Teacher/Model",
                "distillation_alpha": 0.4,
                "num_epochs": 2,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("bashgym.cli._workspace_http_json", fake_request)

    assert (
        main(
            [
                "training",
                "start",
                "--strategy",
                "distillation",
                "--model",
                "Student/Model",
                "--dataset-path",
                "data/distill.jsonl",
                "--config",
                str(config_path),
                "--checkpoint-limit",
                "2",
                "--artifact-retention",
                "adapter_checkpoint",
                "--auto-push-hf",
                "--hf-repo-name",
                "student-distilled",
                "--hf-private",
                "--hf-upload-artifact",
                "adapter",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)

    payload = captured[-1]["payload"]
    assert payload["strategy"] == "distillation"
    assert payload["teacher_model"] == "Teacher/Model"
    assert payload["num_epochs"] == 2
    assert payload["checkpoint_limit"] == 2
    assert payload["artifact_retention"] == "adapter_checkpoint"
    assert payload["auto_push_hf"] is True
    assert payload["hf_private"] is True
    assert payload["hf_upload_artifact"] == "adapter"


def test_training_start_normalizes_session_distillation_alias(monkeypatch, capsys):
    captured = []

    def fake_request(args, path, *, method="GET", payload=None):
        captured.append(payload)
        return {"status": "queued", "run_id": "job-session-distill"}

    monkeypatch.setattr("bashgym.cli._workspace_http_json", fake_request)

    assert (
        main(
            [
                "training",
                "start",
                "--strategy",
                "session-distillation",
                "--model",
                "Qwen/Test",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert captured[-1]["strategy"] == "session_distillation"


def test_training_start_forwards_official_tracking_context(monkeypatch, capsys, tmp_path):
    captured = []

    def fake_request(args, path, *, method="GET", payload=None):
        captured.append(payload)
        return {"status": "queued", "run_id": "job-tracked"}

    tracking = {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "project_display_name": "Project A",
        "experiment_id": "experiment-a",
        "experiment_name": "Experiment A",
        "objective": "Improve held-out quality.",
        "task_type": "terminal-agent",
        "model_id": "model-a",
        "model_version_id": "model-a-v1",
        "model_source_uri": "hf://example/model-a",
        "model_config_digest": "a" * 64,
        "dataset_id": "dataset-a",
        "dataset_version_id": "dataset-a-v1",
        "dataset_source_uri": "file://data/dataset-a.manifest.json",
        "dataset_content_digest": "b" * 64,
        "environment_id": "environment-a",
        "environment_runtime_digest": "c" * 64,
    }
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps(tracking), encoding="utf-8")
    monkeypatch.setattr("bashgym.cli._workspace_http_json", fake_request)

    assert (
        main(
            [
                "training",
                "start",
                "--strategy",
                "sft",
                "--model",
                "example/model-a",
                "--tracking-context",
                str(path),
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert captured[-1]["tracking"] == tracking


def test_training_start_activates_ssh_target(monkeypatch, capsys):
    captured = []

    def fake_request(args, path, *, method="GET", payload=None):
        captured.append(payload)
        return {"status": "queued", "run_id": "job-ssh"}

    monkeypatch.setattr("bashgym.cli._workspace_http_json", fake_request)

    assert (
        main(
            [
                "training",
                "start",
                "--strategy",
                "sft",
                "--model",
                "Test/Model",
                "--compute-target",
                "ssh:lab-box",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert captured[-1]["use_remote_ssh"] is True
    assert captured[-1]["device_id"] == "lab-box"


def test_campaign_read_commands_preserve_workspace_and_pagination(monkeypatch, capsys):
    class FakeCampaignClient:
        def __init__(self):
            self.calls = []

        def request_json(
            self,
            method,
            path,
            *,
            query=None,
            payload=None,
            headers=None,
        ):
            self.calls.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "payload": payload,
                    "headers": headers,
                }
            )
            return {"ok": True}

    client = FakeCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    connection = [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "BASHGYM_CAMPAIGN_CODEX_REFRESH",
        "--json",
    ]

    assert main(["campaign", "list", *connection, "--status", "active"]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1] == {
        "method": "GET",
        "path": "/campaigns",
        "query": {
            "workspace_id": "workspace-a",
            "status": "active",
            "kind": None,
        },
        "payload": None,
        "headers": None,
    }

    assert main(["campaign", "templates", *connection]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/templates"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert main(["campaign", "status", *connection, "--campaign", "campaign:1"]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign%3A1"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert (
        main(
            [
                "campaign",
                "autoresearch",
                *connection,
                "--campaign",
                "campaign:1",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign%3A1/autoresearch"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert (
        main(
            [
                "campaign",
                "manifest",
                *connection,
                "--campaign",
                "campaign-1",
                "--revision",
                "3",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/manifest/3"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    for command in ("evidence", "artifacts", "proposals", "studies"):
        assert main(["campaign", command, *connection, "--campaign", "campaign-1"]) == 0
        json.loads(capsys.readouterr().out)
        assert client.calls[-1]["path"] == f"/campaigns/campaign-1/{command}"
        assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert (
        main(
            [
                "campaign",
                "study",
                "status",
                *connection,
                "--campaign",
                "campaign-1",
                "--study",
                "study:2",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/studies/study%3A2"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    for command in ("attempts", "comparisons"):
        assert main(["campaign", command, *connection, "--campaign", "campaign-1"]) == 0
        json.loads(capsys.readouterr().out)
        assert client.calls[-1]["path"] == f"/campaigns/campaign-1/{command}"
        assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert (
        main(
            [
                "campaign",
                "events",
                *connection,
                "--campaign",
                "campaign-1",
                "--after-cursor",
                "19",
                "--limit",
                "40",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/events"
    assert client.calls[-1]["query"] == {
        "workspace_id": "workspace-a",
        "after_cursor": 19,
        "limit": 40,
    }

    assert (
        main(
            [
                "campaign",
                "metrics",
                *connection,
                "--campaign",
                "campaign-1",
                "--attempt",
                "attempt-2",
                "--source",
                "training_metrics.jsonl",
                "--metric",
                "loss",
                "--after-step",
                "50",
                "--limit",
                "500",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/attempts/attempt-2/metrics"
    assert client.calls[-1]["query"] == {
        "workspace_id": "workspace-a",
        "source": "training_metrics.jsonl",
        "metric_name": "loss",
        "after_step": 50,
        "limit": 500,
    }


def test_ledger_read_commands_require_project_and_preserve_sync_cursor(monkeypatch, capsys):
    class FakeCampaignClient:
        def __init__(self):
            self.calls = []

        def request_json(self, method, path, *, query=None, payload=None, headers=None):
            self.calls.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "payload": payload,
                    "headers": headers,
                }
            )
            return {"ok": True}

    client = FakeCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    connection = [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "BASHGYM_CAMPAIGN_CODEX_REFRESH",
        "--json",
    ]

    assert main(["ledger", "projects", *connection]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/ledger/projects"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert main(["ledger", "context", *connection, "--project", "project:one"]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/ledger/projects/project%3Aone/context"
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a", "recent_limit": 20}

    assert (
        main(
            [
                "ledger",
                "trend",
                *connection,
                "--project",
                "project-one",
                "--run",
                "run:1",
                "--metric",
                "train.loss",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == (
        "/ledger/projects/project-one/metrics/train.loss/trend"
    )
    assert client.calls[-1]["query"]["run_id"] == "run:1"

    assert (
        main(
            [
                "ledger",
                "events",
                *connection,
                "--project",
                "project-one",
                "--after-cursor",
                "42",
                "--limit",
                "50",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["query"] == {
        "workspace_id": "workspace-a",
        "after_cursor": 42,
        "limit": 50,
    }

    assert (
        main(
            [
                "ledger",
                "compare",
                *connection,
                "--project",
                "project-one",
                "--run",
                "run-1",
                "--run",
                "run-2",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/ledger/projects/project-one/compare"
    assert client.calls[-1]["query"] == {
        "workspace_id": "workspace-a",
        "run_id": ["run-1", "run-2"],
    }


def test_campaign_mutations_use_strict_body_and_audit_headers(monkeypatch, capsys):
    class FakeCampaignClient:
        def __init__(self):
            self.calls = []

        def request_json(
            self,
            method,
            path,
            *,
            query=None,
            payload=None,
            headers=None,
        ):
            self.calls.append((method, path, query, payload, headers))
            return {"ok": True, "version": payload["expected_version"] + 1}

    client = FakeCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    connection = [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "CODEX_REFRESH",
        "--json",
    ]

    commands = [
        ("start", []),
        ("pause", ["--reason", "operator-pause"]),
        ("resume", []),
        ("cancel", ["--reason", "operator-cancel"]),
    ]
    for index, (name, extra) in enumerate(commands, start=1):
        assert (
            main(
                [
                    "campaign",
                    name,
                    *connection,
                    "--campaign",
                    "campaign-1",
                    "--expected-version",
                    str(index),
                    "--idempotency-key",
                    f"{name}-key",
                    "--correlation-id",
                    "workflow-1",
                    *extra,
                ]
            )
            == 0
        )
        json.loads(capsys.readouterr().out)

    for index, (name, extra) in enumerate(commands, start=1):
        method, path, query, payload, headers = client.calls[index - 1]
        assert method == "POST"
        assert path == f"/campaigns/campaign-1/{name}"
        assert query is None
        assert payload == {
            "workspace_id": "workspace-a",
            "expected_version": index,
            **({"stop_reason": extra[-1]} if name in {"pause", "cancel"} else {}),
        }
        assert headers == {
            "Idempotency-Key": f"{name}-key",
            "X-Correlation-ID": "workflow-1",
        }


def test_campaign_extended_cli_uses_typed_json_and_strict_authority_fields(
    monkeypatch, capsys, tmp_path
):
    class FakeCampaignClient:
        def __init__(self):
            self.calls = []

        def request_json(self, method, path, *, query=None, payload=None, headers=None):
            self.calls.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "payload": payload,
                    "headers": headers,
                }
            )
            return {"ok": True, "version": 2}

    client = FakeCampaignClient()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    connection = [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "CAMPAIGN_REFRESH",
        "--json",
    ]
    proposal = {
        "proposal_id": "proposal-1",
        "hypothesis": "Positive-aware mining improves retrieval.",
        "study_family": "negative-mining",
        "primary_variable": "negative_filter",
        "expected_outcome": "Higher development nDCG.",
        "falsification_criterion": "No improvement over champion.",
        "estimated_cost": 2.5,
        "dataset_recipe": {},
        "training_recipe": {},
        "evaluation_recipe": {},
        "stage_plan": {"items": []},
        "rationale": "Candidate A identifies false-negative pressure.",
    }
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    identity = {
        "compute_profile_id": "ssh-gpu-lab",
        "remote_run_id": "run-4",
        "pid": 812,
        "process_start_time": "2026-07-13T09:00:00Z",
        "command_hash": "a" * 64,
    }
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    protected_result = {
        "protected_epoch_id": "protected-1",
        "candidate_digest": "c" * 64,
        "passed": True,
        "metrics": {"recall_at_10": 0.84},
        "artifact_sha256": "d" * 64,
    }
    protected_result_path = tmp_path / "protected-result.json"
    protected_result_path.write_text(json.dumps(protected_result), encoding="utf-8")

    assert (
        main(
            [
                "campaign",
                "create",
                *connection,
                "--template",
                "embed-v1",
                "--campaign",
                "campaign-2",
                "--title",
                "Embedding cycle",
                "--idempotency-key",
                "idem-create",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "campaign",
                "propose",
                *connection,
                "--campaign",
                "campaign-2",
                "--expected-version",
                "2",
                "--proposal",
                str(proposal_path),
                "--idempotency-key",
                "idem-propose",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "campaign",
                "action",
                "force-stop",
                *connection,
                "--campaign",
                "campaign-2",
                "--action",
                "action-7",
                "--expected-version",
                "3",
                "--expected-process",
                str(identity_path),
                "--reason",
                "Worker remained alive after cancellation.",
                "--confirm",
                "--idempotency-key",
                "idem-force",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "campaign",
                "protected-result",
                *connection,
                "--campaign",
                "campaign-2",
                "--expected-version",
                "4",
                "--result-file",
                str(protected_result_path),
                "--idempotency-key",
                "idem-protected-result",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "campaign",
                "export",
                *connection,
                "--campaign",
                "campaign-2",
                "--expected-version",
                "5",
                "--formats",
                "markdown,csv,pdf",
                "--idempotency-key",
                "idem-export",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)

    create_call, proposal_call, force_call, protected_call, export_call = client.calls
    assert create_call["path"] == "/campaigns/from-template"
    assert create_call["payload"] == {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-2",
        "title": "Embedding cycle",
        "template_id": "embed-v1",
    }
    assert proposal_call["path"] == "/campaigns/campaign-2/proposals"
    assert proposal_call["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 2,
        **proposal,
    }
    assert force_call["payload"]["expected_remote_process_identity"] == identity
    assert force_call["payload"]["confirmed"] is True
    assert "pid" not in force_call["payload"]
    assert "command" not in force_call["payload"]
    assert protected_call["path"] == "/campaigns/campaign-2/protected-result"
    assert protected_call["payload"]["result"] == protected_result
    assert export_call["payload"]["formats"] == ["markdown", "csv", "pdf"]


def test_campaign_setup_autoresearch_installs_explicit_binding_without_credentials(
    tmp_path, capsys
):
    install_dir = tmp_path / "autoresearch-templates"
    revision = "b" * 40

    assert main([
        "campaign",
        "setup-autoresearch",
        "--template",
        "autoresearch-installed-v1",
        "--objective",
        "Improve a fixed held-out metric with one controlled change.",
        "--model-ref",
        f"hf://example/operator-selected-trainable-model@{revision}",
        "--target-contract",
        "operator-selected-model-v1",
        "--task",
        "heldout-task-autoresearch",
        "--dataset-version",
        "dataset-version-1",
        "--compute-profile",
        "private-training-1",
        "--source-repository-profile",
        "bashgym-source-1",
        "--project",
        "project-1",
        "--evaluation-suite",
        "evaluation-suite-1",
        "--primary-metric",
        "heldout_pass_at_1",
        "--metric-direction",
        "maximize",
        "--budget-unit",
        "gpu_hours",
        "--budget-limit",
        "4",
        "--max-attempts",
        "4",
        "--install-dir",
        str(install_dir),
        "--json",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True
    assert payload["binding_plan"]["model_ref"].endswith(f"@{revision}")
    assert payload["binding_plan"]["compute_profile_id"] == "private-training-1"
    assert payload["binding_plan"]["source_repository_profile_id"] == "bashgym-source-1"
    assert payload["binding_plan"]["required_training_stages"] == [
        "smoke_training",
        "full_training",
    ]
    assert (install_dir / "autoresearch-installed-v1.json").is_file()


def test_campaign_inspect_model_artifact_reports_secret_free_training_plan(tmp_path, capsys):
    artifact_dir = tmp_path / "operator-selected-snapshot"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model.safetensors").write_bytes(b"local-test-weights")
    revision = "c" * 40

    assert (
        main(
            [
                "campaign",
                "inspect-model-artifact",
                "--artifact-dir",
                str(artifact_dir),
                "--model-id",
                "example/modern-open-model",
                "--model-revision",
                revision,
                "--json",
            ]
        )
        == 0
    )

    raw = capsys.readouterr().out
    payload = json.loads(raw)
    plan = payload["model_onboarding_plan"]
    assert payload["ok"] is True
    assert plan["model_ref"] == f"hf://example/modern-open-model@{revision}"
    assert plan["task"] == "causal_lm"
    assert plan["artifact_role"] == "trainable_base"
    assert plan["ready_for_binding"] is True
    assert str(artifact_dir) not in raw


def test_campaign_setup_autoresearch_can_bind_inspected_trainable_artifact(tmp_path, capsys):
    artifact_dir = tmp_path / "selected-model"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model.safetensors").write_bytes(b"local-test-weights")
    install_dir = tmp_path / "autoresearch-templates"
    revision = "d" * 40

    assert (
        main(
            [
                "campaign",
                "setup-autoresearch",
                "--template",
                "autoresearch-inspected-v1",
                "--objective",
                "Improve a fixed held-out metric with one controlled change.",
                "--model-ref",
                f"hf://example/modern-open-model@{revision}",
                "--model-artifact-dir",
                str(artifact_dir),
                "--target-contract",
                "modern-open-model-v1",
                "--task",
                "causal_lm",
                "--dataset-version",
                "dataset-version-1",
                "--compute-profile",
                "private-training-1",
                "--source-repository-profile",
                "bashgym-source-1",
                "--project",
                "project-1",
                "--evaluation-suite",
                "evaluation-suite-1",
                "--primary-metric",
                "heldout_pass_at_1",
                "--metric-direction",
                "maximize",
                "--budget-unit",
                "gpu_hours",
                "--budget-limit",
                "4",
                "--max-attempts",
                "4",
                "--install-dir",
                str(install_dir),
                "--json",
            ]
        )
        == 0
    )

    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert payload["model_onboarding_plan"]["ready_for_binding"] is True
    assert payload["binding_plan"]["model_ref"] == f"hf://example/modern-open-model@{revision}"
    assert str(artifact_dir) not in raw
    assert (install_dir / "autoresearch-inspected-v1.json").is_file()


def test_campaign_setup_autoresearch_rejects_inspected_task_mismatch(tmp_path, capsys):
    artifact_dir = tmp_path / "selected-model"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model.safetensors").write_bytes(b"local-test-weights")
    install_dir = tmp_path / "autoresearch-templates"
    revision = "e" * 40

    exit_code = main(
        [
            "campaign",
            "setup-autoresearch",
            "--template",
            "autoresearch-mismatch-v1",
            "--objective",
            "Improve a fixed metric.",
            "--model-ref",
            f"hf://example/modern-open-model@{revision}",
            "--model-artifact-dir",
            str(artifact_dir),
            "--target-contract",
            "modern-open-model-v1",
            "--task",
            "vision_language",
            "--dataset-version",
            "dataset-version-1",
            "--compute-profile",
            "private-training-1",
            "--source-repository-profile",
            "bashgym-source-1",
            "--project",
            "project-1",
            "--evaluation-suite",
            "evaluation-suite-1",
            "--primary-metric",
            "heldout_pass_at_1",
            "--metric-direction",
            "maximize",
            "--budget-unit",
            "gpu_hours",
            "--budget-limit",
            "4",
            "--max-attempts",
            "4",
            "--install-dir",
            str(install_dir),
            "--json",
        ]
    )

    assert exit_code != 0
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "campaign_cli_invalid"
    assert not install_dir.exists()


def test_campaign_code_lineage_cli_routes_prepare_and_capture(monkeypatch, capsys):
    class Client:
        def __init__(self):
            self.calls = []

        def request_json(self, method, path, *, query=None, payload=None, headers=None):
            self.calls.append((method, path, payload, headers))
            return {"record": {"state": path.rsplit("/", 1)[-1]}}

    client = Client()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    common = [
        "--campaign",
        "campaign-1",
        "--proposal",
        "proposal-1",
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "CAMPAIGN_REFRESH",
        "--json",
    ]
    for operation in ("prepare", "capture"):
        assert main(
            [
                "campaign",
                "proposal",
                f"lineage-{operation}",
                *common,
                "--idempotency-key",
                f"lineage-{operation}-1",
            ]
        ) == 0
        json.loads(capsys.readouterr().out)

    assert [call[1] for call in client.calls] == [
        "/campaigns/campaign-1/proposals/proposal-1/code-lineage/prepare",
        "/campaigns/campaign-1/proposals/proposal-1/code-lineage/capture",
    ]
    assert all(call[2] == {"workspace_id": "workspace-a"} for call in client.calls)


def test_campaign_autoresearch_cli_routes_explicit_roles_and_result_identity(
    monkeypatch, capsys, tmp_path
):
    class Client:
        def __init__(self):
            self.calls = []

        def request_json(self, method, path, *, query=None, payload=None, headers=None):
            self.calls.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "payload": payload,
                    "headers": headers,
                }
            )
            return {"ok": True}

    client = Client()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    connection = [
        "--workspace-id",
        "workspace-a",
        "--credential-ref",
        "CAMPAIGN_REFRESH",
        "--json",
    ]
    proposal = {
        "proposal_id": "proposal-1",
        "hypothesis": "One bounded variable improves pass rate.",
        "study_family": "terminal-agent",
        "primary_variable": "learning_rate",
        "controlled_variables": ["dataset", "seed"],
        "expected_outcome": "Higher pass rate.",
        "falsification_criterion": "No held-out improvement.",
        "estimated_cost": 0.01,
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": {"schema_version": "recipe.v1"},
        "evaluation_recipe": {"schema_version": "recipe.v1"},
        "stage_plan": {"items": []},
        "rationale": "Test one controlled hypothesis.",
    }
    proposal_path = tmp_path / "autoresearch-proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    assert main([
        "campaign", "doctor", *connection,
        "--template", "autoresearch-installed-v1",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == (
        "/campaigns/templates/autoresearch-installed-v1/doctor"
    )
    assert client.calls[-1]["query"] == {"workspace_id": "workspace-a"}

    assert main([
        "campaign", "propose", *connection,
        "--campaign", "campaign-1",
        "--expected-version", "4",
        "--proposal", str(proposal_path),
        "--autoresearch-role", "baseline",
        "--idempotency-key", "baseline-1",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/autoresearch/baseline"

    assert main([
        "campaign", "propose", *connection,
        "--campaign", "campaign-1",
        "--expected-version", "5",
        "--proposal", str(proposal_path),
        "--autoresearch-role", "candidate",
        "--parent-proposal", "baseline-1",
        "--idempotency-key", "candidate-1",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert client.calls[-1]["path"] == "/campaigns/campaign-1/autoresearch/candidates"
    assert client.calls[-1]["payload"]["parent_proposal_id"] == "baseline-1"

    assert main([
        "campaign", "autoresearch-result", *connection,
        "--campaign", "campaign-1",
        "--project", "project-a",
        "--evaluation-result", "evaluation-1",
        "--idempotency-key", "result-1",
    ]) == 0
    json.loads(capsys.readouterr().out)
    result_call = client.calls[-1]
    assert result_call["path"] == (
        "/campaigns/campaign-1/autoresearch/ingest-evaluation"
    )
    assert result_call["payload"] == {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "evaluation_result_id": "evaluation-1",
    }


def test_campaign_proposal_rejects_server_owned_scope_fields(monkeypatch, capsys, tmp_path):
    class Client:
        def __init__(self):
            self.calls = []

        def request_json(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True}

    client = Client()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    proposal_path = tmp_path / "unsafe-proposal.json"
    proposal_path.write_text(
        json.dumps({"proposal_id": "p-1", "actor": "injected"}), encoding="utf-8"
    )
    result = main(
        [
            "campaign",
            "propose",
            "--workspace-id",
            "workspace-a",
            "--credential-ref",
            "CAMPAIGN_REFRESH",
            "--campaign",
            "campaign-1",
            "--expected-version",
            "2",
            "--proposal",
            str(proposal_path),
            "--idempotency-key",
            "idem-propose",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out)["error"]["code"] == "campaign_cli_invalid"
    assert client.calls == []


def test_campaign_json_input_must_be_a_regular_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "bashgym.cli._campaign_client",
        lambda _args: (_ for _ in ()).throw(AssertionError("client must not be built")),
    )
    result = main(
        [
            "campaign",
            "create",
            "--workspace-id",
            "workspace-a",
            "--credential-ref",
            "CAMPAIGN_REFRESH",
            "--manifest",
            str(tmp_path),
            "--idempotency-key",
            "idem-direct-create",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out)["error"]["code"] == "campaign_cli_invalid"


def test_campaign_direct_create_requires_a_strict_typed_envelope(monkeypatch, capsys, tmp_path):
    class Client:
        def __init__(self):
            self.calls = []

        def request_json(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {"campaign": {"campaign_id": "campaign-3"}}

    client = Client()
    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: client)
    envelope = {
        "campaign_id": "campaign-3",
        "title": "Typed embedding cycle",
        "kind": "embedding_retrieval",
        "objective": "Improve MemexAI retrieval without opening protected data.",
        "target_model": {
            "target_contract_key": "memexai-embedding-v1",
            "base_model_ref": "Qwen/Qwen3-Embedding-0.6B",
            "task": "text-embedding",
            "representation_contract": {"dimensions": 768},
        },
        "manifest": {
            "approved_data_scopes": ["memexai-reviewed"],
            "compute_profile_id": "ssh-gpu-lab",
            "budget_limits": {"GPU_HOURS": 8},
            "evaluation_plan": {},
            "promotion_gates": {},
        },
    }
    envelope_path = tmp_path / "campaign.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")

    assert (
        main(
            [
                "campaign",
                "create",
                "--workspace-id",
                "workspace-a",
                "--credential-ref",
                "CAMPAIGN_REFRESH",
                "--manifest",
                str(envelope_path),
                "--idempotency-key",
                "idem-direct-create",
                "--json",
            ]
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    method, path, kwargs = client.calls[0]
    assert method == "POST" and path == "/campaigns"
    assert kwargs["payload"] == {"workspace_id": "workspace-a", **envelope}
    assert set(kwargs["headers"]) == {"Idempotency-Key"}


def test_campaign_cli_preserves_stable_error_and_exit_code(monkeypatch, capsys):
    from bashgym.campaigns.client import CampaignClientError

    class ForbiddenClient:
        def request_json(self, *_args, **_kwargs):
            raise CampaignClientError(
                "campaign_capability_required:campaign.start",
                "Actor lacks campaign.start.",
                status_code=403,
            )

    monkeypatch.setattr("bashgym.cli._campaign_client", lambda _args: ForbiddenClient())
    result = main(
        [
            "campaign",
            "start",
            "--workspace-id",
            "workspace-a",
            "--credential-ref",
            "HERMES_REFRESH",
            "--campaign",
            "campaign-1",
            "--expected-version",
            "2",
            "--idempotency-key",
            "start-2",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 4
    assert json.loads(captured.out)["error"]["code"] == (
        "campaign_capability_required:campaign.start"
    )
    assert "campaign.start" in captured.err


def test_replay_scrub_cli_redacts_and_writes_output(tmp_path, capsys):
    source = tmp_path / "trace.json"
    output = tmp_path / "trace.scrubbed.json"
    source.write_text(
        json.dumps(
            {
                "trace": [
                    {
                        "stdout": "OPENAI_API_KEY=sk-1234567890abcdefghijklmnop",
                        "stderr": "x" * 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "replay",
                "scrub",
                str(source),
                "--output",
                str(output),
                "--max-output-chars",
                "64",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    scrubbed = json.loads(output.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["stats"]["redactions"] == 1
    assert payload["stats"]["truncations"] == 1
    assert "sk-1234567890abcdefghijklmnop" not in output.read_text(encoding="utf-8")
    assert "[truncated" in scrubbed["trace"][0]["stderr"]


def test_sources_cli_lists_inspects_recommends_and_prepares(tmp_path, capsys):
    assert main(["sources", "list", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["ok"] is True
    assert catalog["schema_version"] == "bashgym.source_catalog.v1"
    assert any(source["id"] == "harbor_terminal_bench" for source in catalog["sources"])

    assert main(["sources", "inspect", "ultrafeedback_binarized", "--json"]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["ok"] is True
    assert card["source"]["training_eligible"] is True

    assert main(["sources", "recommend", "--goal", "dpo", "--json"]) == 0
    recommendations = json.loads(capsys.readouterr().out)
    ids = {item["source"]["id"] for item in recommendations["recommendations"]}
    assert "ultrafeedback_binarized" in ids
    assert "harbor_terminal_bench" not in ids

    assert (
        main(
            [
                "sources",
                "prepare",
                "helpsteer2",
                "--goal",
                "reward_model",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["ok"] is True
    assert manifest["source"]["id"] == "helpsteer2"
    assert tmp_path.joinpath("source_manifest.json").exists()


def test_sources_cli_blocks_eval_only_training_export(capsys):
    assert main(["sources", "prepare", "harbor_terminal_bench", "--goal", "sft", "--json"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "eval_only_source_for_training" in payload["use_verdict"]["blocking_codes"]


def test_sources_cli_prepares_local_input_artifacts(tmp_path, capsys):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "uf-1",
                "prompt": "Fix a failing test.",
                "chosen": "Run the test, inspect the traceback, patch the bug.",
                "rejected": "Ignore the traceback.",
                "metadata": {
                    "quality_score": 0.9,
                    "label_source": "fixture",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "sources",
                "prepare",
                "ultrafeedback_binarized",
                "--goal",
                "dpo",
                "--input",
                str(source_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["artifacts"][0]["artifact_type"] == "dpo_pairs"
    assert (tmp_path / "out" / "dpo_pairs.jsonl").exists()


def test_sources_cli_fetches_and_prepares_remote_source(tmp_path, capsys, monkeypatch):
    def fake_fetch(
        card,
        *,
        output_dir,
        split,
        subset=None,
        revision=None,
        limit=None,
        approval_reason=None,
        force_refresh=False,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records_path = output_path / "source_records.jsonl"
        records_path.write_text(
            json.dumps(
                {
                    "id": "uf-1",
                    "prompt": "Fix a failing test.",
                    "chosen": "Run pytest and patch the failing function.",
                    "rejected": "Claim success without running tests.",
                    "metadata": {"decontamination_status": "checked"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "schema_version": "bashgym.source_fetch.v1",
            "ok": True,
            "source_id": card.id,
            "source_name": card.name,
            "huggingface_id": card.huggingface_id,
            "split": split,
            "subset": subset,
            "revision": revision,
            "limit": limit,
            "output_dir": str(output_path),
            "records_path": str(records_path),
            "report_path": str(output_path / "source_fetch_report.json"),
            "record_count": 1,
            "truncated": False,
            "cache_hit": False,
            "force_refresh": force_refresh,
            "approval_required": limit is None or limit > 1000,
            "approval_granted": True,
            "approval_reason": approval_reason,
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr("bashgym.cli.fetch_source_records", fake_fetch)

    assert (
        main(
            [
                "sources",
                "fetch",
                "ultrafeedback_binarized",
                "--output-dir",
                str(tmp_path / "fetch"),
                "--split",
                "train_prefs",
                "--limit",
                "1",
                "--force-refresh",
                "--json",
            ]
        )
        == 0
    )
    fetch_payload = json.loads(capsys.readouterr().out)
    assert fetch_payload["ok"] is True
    assert fetch_payload["schema_version"] == "bashgym.source_fetch.v1"
    assert fetch_payload["force_refresh"] is True

    assert (
        main(
            [
                "sources",
                "prepare",
                "ultrafeedback_binarized",
                "--goal",
                "dpo",
                "--fetch",
                "--output-dir",
                str(tmp_path / "prepare"),
                "--limit",
                "1",
                "--fetch-approval-reason",
                "fixture fetch",
                "--json",
            ]
        )
        == 0
    )
    prepare_payload = json.loads(capsys.readouterr().out)
    assert prepare_payload["ok"] is True
    assert prepare_payload["fetch_report"]["schema_version"] == "bashgym.source_fetch.v1"
    assert prepare_payload["fetch_report"]["approval_reason"] == "fixture fetch"
    assert prepare_payload["artifacts"][0]["artifact_type"] == "dpo_pairs"
    assert (tmp_path / "prepare" / "dpo_pairs.jsonl").exists()


def test_sources_cli_fetch_requires_approval_for_large_limit(tmp_path, capsys):
    assert (
        main(
            [
                "sources",
                "fetch",
                "ultrafeedback_binarized",
                "--output-dir",
                str(tmp_path / "fetch"),
                "--limit",
                "1001",
                "--json",
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["approval_required"] is True
    assert "remote_fetch_approval_required" in payload["errors"]
    assert not tmp_path.joinpath("fetch", "source_records.jsonl").exists()


def test_compute_cli_lists_preflights_and_dry_runs_launch(capsys):
    assert main(["compute", "targets", "--json"]) == 0
    targets = json.loads(capsys.readouterr().out)
    ids = {target["id"] for target in targets["targets"]}
    assert {"local_cpu_or_gpu", "private_gpu", "skypilot_a10g", "dstack_a10g"} <= ids

    assert main(["compute", "preflight", "--target", "private_gpu", "--json"]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["schema_version"] == "bashgym.compute_preflight.v1"
    assert any(
        check["code"] == "private_compute_target_configured" for check in preflight["checks"]
    )

    assert (
        main(
            [
                "compute",
                "launch",
                "--target",
                "skypilot_a10g",
                "--plan",
                "runs/demo/plan.json",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    launch = json.loads(capsys.readouterr().out)
    assert launch["schema_version"] == "bashgym.compute_launch_plan.v1"
    assert launch["provider_config"]["filename"] == "sky.yaml"
    assert launch["approval_required"] is True


def test_training_runcard_cli_create_validate_and_attach(tmp_path, capsys):
    path = tmp_path / "run_card.json"
    assert (
        main(
            [
                "training",
                "runcard",
                "create",
                "--run-id",
                "run-cli",
                "--training-method",
                "sft",
                "--base-model",
                "Qwen/Qwen3-Coder",
                "--compute-target",
                "local_cpu_or_gpu",
                "--training-plan",
                "plans/sft.json",
                "--source-manifest",
                "data/source_manifest.json",
                "--output",
                str(path),
                "--no-git",
                "--json",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["ok"] is True
    assert created["run_card"]["run_id"] == "run-cli"

    assert main(["training", "runcard", "validate", str(path), "--promotion", "--json"]) == 2
    missing = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in missing["findings"]}
    assert {"missing_metrics_path", "missing_release_evidence_path"} <= codes

    tmp_path.joinpath("plans").mkdir()
    tmp_path.joinpath("data").mkdir()
    tmp_path.joinpath("runs", "cli").mkdir(parents=True)
    tmp_path.joinpath("plans", "sft.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("data", "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.source_manifest.v1",
                "source": {"id": "helpsteer2"},
                "goal": "reward_model",
                "use_verdict": {
                    "ok": True,
                    "blocking_codes": [],
                    "warnings": [],
                    "requires_override_reason": False,
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("runs", "cli", "metrics.jsonl").write_text(
        '{"step": 1, "eval_loss": 0.4}\n',
        encoding="utf-8",
    )
    tmp_path.joinpath("runs", "cli", "release_evidence.json").write_text(
        json.dumps({"ship": True, "reasons": [], "release_gate": {"ship": True}}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "runcard",
                "attach-evidence",
                str(path),
                "--metrics",
                "runs/cli/metrics.jsonl",
                "--release-evidence",
                "runs/cli/release_evidence.json",
                "--promotion",
                "--json",
            ]
        )
        == 0
    )
    attached = json.loads(capsys.readouterr().out)
    assert attached["ok"] is True


def test_training_dpo_pairs_cli_validates_strict_metadata(tmp_path, capsys):
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps(
            {
                "id": "pair-1",
                "prompt": "Fix the failing test",
                "chosen_response": "Run pytest, inspect the failure, patch the function.",
                "rejected_response": "Ignore the failure and claim success.",
                "metadata": {
                    "pair_id": "pair-1",
                    "prompt_hash": "abc123",
                    "chosen_trace_id": "gold-1",
                    "rejected_trace_id": "failed-1",
                    "pair_generation_method": "trace_pair",
                    "label_strength": "verified_success_vs_failure",
                    "label_source": "trace_verifier",
                    "chosen_quality_score": 0.95,
                    "rejected_quality_score": 0.25,
                    "domain": "terminal_agent",
                    "task_family": "test_fix",
                    "split": "train",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["training", "dpo-pairs", "validate", str(pairs_path), "--strict", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["strict"] is True
    assert payload["total_records"] == 1


def test_training_reward_examples_cli_validates_strict_metadata(tmp_path, capsys):
    rewards_path = tmp_path / "reward_examples.jsonl"
    rewards_path.write_text(
        json.dumps(
            {
                "id": "reward-1",
                "reward_type": "outcome_reward",
                "prompt": "Fix the failing test",
                "response": "Run pytest, inspect the failure, patch the function.",
                "score": 0.9,
                "metadata": {
                    "reward_example_id": "reward-1",
                    "reward_scale": "0_to_1",
                    "label_source": "trace_verifier",
                    "source_id": "helpsteer2",
                    "quality_score": 0.95,
                    "domain": "terminal_agent",
                    "task_family": "test_fix",
                    "split": "train",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(["training", "reward-examples", "validate", str(rewards_path), "--strict", "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["strict"] is True
    assert payload["total_records"] == 1


def test_training_reward_examples_cli_fails_strict_missing_metadata(tmp_path, capsys):
    rewards_path = tmp_path / "reward_examples.jsonl"
    rewards_path.write_text(
        json.dumps(
            {
                "id": "reward-1",
                "reward_type": "process_reward",
                "prompt": "Fix it",
                "response": "Good fix",
                "score": 1.0,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(["training", "reward-examples", "validate", str(rewards_path), "--strict", "--json"])
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["ok"] is False
    assert "missing_process_reward_steps" in codes
    assert "missing_decontamination_metadata" in codes


def test_training_reward_eval_cli_writes_reward_metrics(tmp_path, capsys):
    rewards_path = tmp_path / "reward_predictions.jsonl"
    output_path = tmp_path / "reward_eval.json"
    records = [
        {
            "id": "reward-1a",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "Run pytest and patch the bug.",
            "score": 1.0,
            "predicted_reward": 0.9,
            "metadata": {
                "reward_example_id": "reward-1a",
                "pair_id": "pair-1",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.95,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "eval",
                "decontamination_status": "checked",
            },
        },
        {
            "id": "reward-1b",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "Ignore the failure.",
            "score": 0.0,
            "predicted_reward": 0.2,
            "metadata": {
                "reward_example_id": "reward-1b",
                "pair_id": "pair-1",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.25,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "eval",
                "decontamination_status": "checked",
            },
        },
    ]
    rewards_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "reward-eval",
                "evaluate",
                str(rewards_path),
                "--output",
                str(output_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["metrics"]["heldout_pair_accuracy"] == 1.0
    assert payload["metrics"]["pair_count"] == 1
    assert payload["output_path"] == str(output_path)
    assert output_path.exists()


def test_training_reward_model_smoke_cli_writes_fixture_artifacts(tmp_path, capsys):
    rewards_path = tmp_path / "reward_examples.jsonl"
    output_dir = tmp_path / "reward-smoke"
    records = [
        {
            "id": "train-good",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "patch fix test success",
            "score": 1.0,
            "metadata": {
                "reward_example_id": "train-good",
                "pair_id": "train-pair",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.95,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "train",
                "decontamination_status": "checked",
            },
        },
        {
            "id": "train-bad",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "ignore failure broken",
            "score": 0.0,
            "metadata": {
                "reward_example_id": "train-bad",
                "pair_id": "train-pair",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.2,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "train",
                "decontamination_status": "checked",
            },
        },
        {
            "id": "eval-good",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "patch the failing test",
            "score": 1.0,
            "metadata": {
                "reward_example_id": "eval-good",
                "pair_id": "eval-pair",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.95,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "eval",
                "decontamination_status": "checked",
            },
        },
        {
            "id": "eval-bad",
            "reward_type": "outcome_reward",
            "prompt": "Fix the failing test",
            "response": "ignore the failing test",
            "score": 0.0,
            "metadata": {
                "reward_example_id": "eval-bad",
                "pair_id": "eval-pair",
                "reward_scale": "0_to_1",
                "label_source": "trace_verifier",
                "source_id": "helpsteer2",
                "quality_score": 0.2,
                "domain": "terminal_agent",
                "task_family": "test_fix",
                "split": "eval",
                "decontamination_status": "checked",
            },
        },
    ]
    rewards_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "reward-model",
                "smoke",
                str(rewards_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "6",
                "--learning-rate",
                "0.8",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["reward_eval"]["metrics"]["pair_count"] == 1
    assert output_dir.joinpath("reward_model_fixture.json").exists()
    assert output_dir.joinpath("metrics.jsonl").exists()
    assert output_dir.joinpath("reward_eval.json").exists()


def test_training_dpo_pairs_cli_fails_strict_missing_metadata(tmp_path, capsys):
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        json.dumps(
            {
                "id": "pair-1",
                "prompt": "Fix it",
                "chosen_response": "Same",
                "rejected_response": "Same",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["training", "dpo-pairs", "validate", str(pairs_path), "--strict", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["ok"] is False
    assert "identical_chosen_rejected" in codes
    assert "missing_decontamination_metadata" in codes


def test_training_runcard_cli_requires_strict_dpo_pair_evidence(tmp_path, capsys):
    path = tmp_path / "run_card.json"
    tmp_path.joinpath("plan.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.source_manifest.v1",
                "source": {"id": "ultrafeedback_binarized"},
                "goal": "dpo",
                "use_verdict": {
                    "ok": True,
                    "blocking_codes": [],
                    "warnings": [],
                    "requires_override_reason": False,
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps({"ship": True, "reasons": [], "release_gate": {"ship": True}}),
        encoding="utf-8",
    )
    tmp_path.joinpath("dpo_pairs.jsonl").write_text(
        json.dumps(
            {
                "id": "pair-1",
                "prompt": "Fix the failing test",
                "chosen_response": "Run pytest, inspect the failure, patch the function.",
                "rejected_response": "Ignore the failure and claim success.",
                "metadata": {
                    "pair_id": "pair-1",
                    "prompt_hash": "abc123",
                    "chosen_trace_id": "gold-1",
                    "rejected_trace_id": "failed-1",
                    "pair_generation_method": "trace_pair",
                    "label_strength": "verified_success_vs_failure",
                    "label_source": "trace_verifier",
                    "chosen_quality_score": 0.95,
                    "rejected_quality_score": 0.25,
                    "domain": "terminal_agent",
                    "task_family": "test_fix",
                    "split": "train",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "runcard",
                "create",
                "--run-id",
                "run-dpo",
                "--training-method",
                "dpo",
                "--base-model",
                "Qwen/Qwen3-Coder",
                "--compute-target",
                "local_cpu_or_gpu",
                "--training-plan",
                "plan.json",
                "--source-manifest",
                "source_manifest.json",
                "--preference-pairs",
                "dpo_pairs.jsonl",
                "--metrics",
                "metrics.jsonl",
                "--release-evidence",
                "release_evidence.json",
                "--output",
                str(path),
                "--promotion",
                "--no-git",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["run_card"]["preference_pairs_path"] == "dpo_pairs.jsonl"


def test_training_runcard_cli_requires_strict_reward_example_evidence(tmp_path, capsys):
    path = tmp_path / "run_card.json"
    tmp_path.joinpath("plan.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.source_manifest.v1",
                "source": {"id": "helpsteer2"},
                "goal": "reward_model",
                "use_verdict": {
                    "ok": True,
                    "blocking_codes": [],
                    "warnings": [],
                    "requires_override_reason": False,
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps({"ship": True, "reasons": [], "release_gate": {"ship": True}}),
        encoding="utf-8",
    )
    tmp_path.joinpath("reward_examples.jsonl").write_text(
        json.dumps(
            {
                "id": "reward-1",
                "reward_type": "outcome_reward",
                "prompt": "Fix the failing test",
                "response": "Run pytest, inspect the failure, patch the function.",
                "score": 0.9,
                "metadata": {
                    "reward_example_id": "reward-1",
                    "reward_scale": "0_to_1",
                    "label_source": "trace_verifier",
                    "source_id": "helpsteer2",
                    "quality_score": 0.95,
                    "domain": "terminal_agent",
                    "task_family": "test_fix",
                    "split": "train",
                    "decontamination_status": "checked",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("reward_eval.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.reward_model_eval.v1",
                "ok": True,
                "split": "eval",
                "total_records": 2,
                "evaluated_records": 2,
                "prediction_records": 2,
                "metrics": {
                    "heldout_pair_accuracy": 1.0,
                    "pair_count": 1,
                    "calibration_error": 0.1,
                    "reward_margin": 0.6,
                    "length_bias": 0.0,
                    "reward_variance": 0.08,
                    "eval_only_leakage_count": 0,
                    "eval_only_leakage_rate": 0.0,
                },
                "task_family_breakdown": [],
                "eval_only_source_ids": [],
                "findings": [],
                "fail_count": 0,
                "warn_count": 0,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "runcard",
                "create",
                "--run-id",
                "run-reward",
                "--training-method",
                "reward_model",
                "--base-model",
                "Qwen/Qwen3-Coder",
                "--compute-target",
                "local_cpu_or_gpu",
                "--training-plan",
                "plan.json",
                "--source-manifest",
                "source_manifest.json",
                "--reward-examples",
                "reward_examples.jsonl",
                "--reward-eval",
                "reward_eval.json",
                "--metrics",
                "metrics.jsonl",
                "--release-evidence",
                "release_evidence.json",
                "--output",
                str(path),
                "--promotion",
                "--no-git",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["run_card"]["reward_examples_path"] == "reward_examples.jsonl"
    assert payload["run_card"]["reward_eval_path"] == "reward_eval.json"


def test_training_runcard_cli_validates_claim_tier(tmp_path, capsys):
    path = tmp_path / "run_card.json"
    tmp_path.joinpath("plan.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.source_manifest.v1",
                "source": {"id": "helpsteer2"},
                "goal": "reward_model",
                "use_verdict": {
                    "ok": True,
                    "blocking_codes": [],
                    "warnings": [],
                    "requires_override_reason": False,
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "release_gate": {
                    "ship": True,
                    "trace_ship": True,
                    "environment_ship": True,
                    "environment_sections": ["holdout_gate"],
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "runcard",
                "create",
                "--run-id",
                "run-narrow",
                "--training-method",
                "sft",
                "--base-model",
                "Qwen/Qwen3-Coder",
                "--compute-target",
                "local_cpu_or_gpu",
                "--training-plan",
                "plan.json",
                "--source-manifest",
                "source_manifest.json",
                "--metrics",
                "metrics.jsonl",
                "--release-evidence",
                "release_evidence.json",
                "--claim-tier",
                "narrow_routing",
                "--output",
                str(path),
                "--promotion",
                "--no-git",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["run_card"]["claim_tier"] == "narrow_routing"


def test_training_docs_topic_returns_content_in_json(capsys):
    assert main(["training", "docs", "--topic", "glossary", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["topic"] == "glossary"
    assert "Training Glossary" in payload["content"]


def test_training_docs_capabilities_topic_maps_full_spread(capsys):
    assert main(["training", "docs", "--topic", "capabilities", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["topic"] == "capabilities"
    assert "Training Capability Map" in payload["content"]
    assert "DPPO replay" in payload["content"]
    assert "ECHO" in payload["content"]


def test_training_docs_methods_and_review_packet_are_readable(capsys):
    assert main(["training", "docs", "--topic", "methods-reference", "--json"]) == 0
    methods = json.loads(capsys.readouterr().out)
    assert methods["ok"] is True
    assert "Training Methods Reference" in methods["content"]
    assert "DPPO Replay / Backend Path" in methods["content"]
    assert "JEPA-Style World Models" in methods["content"]

    assert main(["training", "docs", "--topic", "external-review", "--json"]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["ok"] is True
    assert "BashGym External AI/ML Review Packet" in review["content"]
    assert "Claims We Are Not Making Yet" in review["content"]
    assert "Reviewer Questions" in review["content"]

    assert main(["training", "docs", "--topic", "rlhf-handbook-comparison", "--json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["ok"] is True
    assert "RLHF Handbook Comparison And BashGym Gap Plan" in comparison["content"]
    assert "Reward models are a real missing lane" in comparison["content"]
    assert "Reviewer Questions Resolved" in comparison["content"]


def test_training_capabilities_matrix_maps_full_training_and_eval_spread(capsys):
    assert main(["training", "capabilities", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    training_ids = {item["id"] for item in payload["training"]}
    eval_ids = {item["id"] for item in payload["evaluation"]}
    backend_ids = {item["id"] for item in payload["backend_stacks"]}
    path_ids = {item["id"] for item in payload["recommended_paths"]}
    ecosystem_ids = {item["id"] for item in payload["ecosystem_methods"]}
    source_ids = {item["id"] for item in payload["source_refs"]}
    data_source_ids = {item["id"] for item in payload["data_sources"]}
    artifact_ids = {item["id"] for item in payload["artifact_contracts"]}
    family_ids = {item["id"] for item in payload["model_family_support"]}
    hardware_ids = {item["id"] for item in payload["hardware_profiles"]}
    config_axis_ids = {item["id"] for item in payload["config_axes"]}
    surface_ids = {item["id"] for item in payload["platform_surfaces"]}
    metric_catalog_ids = {item["id"] for item in payload["metric_catalog"]}
    recipe_stage_ids = {item["id"] for item in payload["recipe_stages"]}
    education_ids = {item["id"] for item in payload["education_path"]}

    assert payload["ok"] is True
    assert {"ready", "ready_with_evidence", "backend_dependent", "diagnostic"} <= set(
        payload["status_key"]
    )
    assert {"sft", "dpo", "reward_modeling", "grpo_rlvr", "dppo_replay", "echo_rwml"} <= (
        training_ids
    )
    assert {
        "heldout_trace",
        "environment_passk",
        "environment_holdout_gate",
        "spurious_reward_and_tamper",
        "external_benchmark_ingest",
    } <= eval_ids
    assert {"trl", "unsloth", "verl", "skyrl"} <= backend_ids
    assert {
        "first_student",
        "terminal_rl",
        "reward_model_lane",
        "dppo_backend",
        "jepa_world_model",
    } <= path_ids
    assert {"ppo", "orpo_kto_ipo_simpo", "gdpo_ebft", "multimodal_rl"} <= ecosystem_ids
    assert {"trl", "unsloth", "verl", "skyrl", "openrlhf", "axolotl"} <= source_ids
    assert "qwen" in source_ids
    assert {
        "gold_traces",
        "silver_bronze_failed_traces",
        "custom_jsonl",
        "synthetic_data_designer",
        "terminal_environments",
        "public_preference_reward_sources",
    } <= data_source_ids
    assert {
        "training_examples_jsonl",
        "dpo_pairs_jsonl",
        "reward_examples_jsonl",
        "reward_eval_json",
        "environment_spec",
        "metrics_jsonl",
        "dppo_replay_jsonl",
        "backend_smoke_bundle",
        "release_evidence_json",
    } <= artifact_ids
    assert {"gemma4", "qwen3", "qwen2.5", "llama3", "generic_hf_causal_lm"} <= family_ids
    assert {"local_12gb", "local_24gb", "private_compute_target", "cloud_backend"} <= hardware_ids
    assert {
        "data_scope",
        "adapter_mode",
        "sequence_length",
        "terminal_rl_sampling",
        "world_model_objectives",
        "promotion_thresholds",
    } <= config_axis_ids
    assert {
        "agent_cli",
        "training_api",
        "environment_api",
        "eval_api",
        "device_and_hardware_api",
        "ui_surfaces",
    } <= surface_ids
    assert {
        "setup_contracts",
        "optimization_health",
        "preference_health",
        "reward_model_health",
        "rl_signal_quality",
        "behavior_evidence",
        "safety_release",
        "world_model_diagnostics",
        "hardware_efficiency",
    } <= metric_catalog_ids
    assert {
        "orient",
        "data_contract",
        "local_smoke",
        "behavior_baseline",
        "training_run",
        "release_evidence",
        "backend_smoke",
        "route_or_iterate",
    } <= recipe_stage_ids
    assert {
        "mental_model",
        "settings",
        "metrics",
        "terminal_rl_recipe",
        "session_distillation",
        "private_compute_finalization",
    } <= (education_ids)
    qwen3 = next(item for item in payload["model_family_support"] if item["id"] == "qwen3")
    assert "Qwen3.6" in qwen3["display_name"]
    assert "newest compatible Qwen3.6" in qwen3["checkpoint_guidance"]
    assert any(
        "POST /api/eval/environments/dppo-replay/smoke-plan" in item.get("endpoints", [])
        for item in payload["platform_surfaces"]
    )
    assert any(
        "world-model quality remains diagnostic" in item
        for item in payload["minimum_promotion_evidence"]
    )
    assert payload["next"][0]["command"].startswith("bashgym training plan")


def test_training_docs_terminal_recipe_and_private_compute_checklist_are_readable(capsys):
    assert main(["training", "docs", "--topic", "terminal-rl-recipe", "--json"]) == 0
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["ok"] is True
    assert "TMax-Style Terminal RL Recipe" in recipe["content"]
    assert "smoke bundle" in recipe["content"]

    assert main(["training", "docs", "--topic", "private-compute-checklist", "--json"]) == 0
    checklist = json.loads(capsys.readouterr().out)
    assert checklist["ok"] is True
    assert "Private Compute Eval And Backend Smoke Checklist" in checklist["content"]
    assert "backend_smoke_readiness.json" in checklist["content"]

    assert main(["training", "docs", "--topic", "gx10-checklist", "--json"]) == 0
    alias = json.loads(capsys.readouterr().out)
    assert alias["ok"] is True
    assert alias["topic"] == "private-compute-checklist"
    assert alias["requested_topic"] == "gx10-checklist"


def test_training_plan_session_distillation_defaults(capsys):
    assert main(["training", "plan", "--strategy", "session-distillation", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    settings = payload["starting_settings"]
    assert payload["strategy"] == "session-distillation"
    assert settings["session_distillation_alpha"] == 0.7
    assert settings["session_distillation_mask_policy"] == "target_span_only"
    assert settings["artifact_retention"] == "adapter_only"
    assert settings["checkpoint_limit"] == 1
    assert settings["hf_private"] is True
    assert "session_distillation_loss" in payload["watch"]
    assert any(doc["topic"] == "session-distillation" for doc in payload["docs"])


def test_training_plan_distillation_is_launch_ready(capsys):
    assert main(["training", "plan", "--strategy", "distillation", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    settings = payload["starting_settings"]
    ladder_stages = {step["stage"] for step in payload["readiness_ladder"]}
    assert payload["strategy"] == "distillation"
    assert settings["teacher_model"] == "<teacher-model-id>"
    assert settings["num_epochs"] == 1
    assert settings["artifact_retention"] == "adapter_only"
    assert settings["auto_push_hf"] is False
    assert {"teacher_data_contract", "distillation_smoke", "behavior_gate"} <= ladder_stages


def test_all_direct_training_plans_surface_storage_defaults(capsys):
    for strategy in (
        "sft",
        "dpo",
        "grpo",
        "rlvr",
        "distillation",
        "session-distillation",
    ):
        assert main(["training", "plan", "--strategy", strategy, "--json"]) == 0
        settings = json.loads(capsys.readouterr().out)["starting_settings"]
        assert settings["checkpoint_limit"] == 1
        assert settings["artifact_retention"] == "adapter_only"
        assert settings["auto_push_hf"] is False
        assert settings["hf_private"] is True
        assert settings["hf_upload_artifact"] == "auto"


def test_training_plan_world_model_contains_echo_rwml_defaults(capsys):
    assert main(["training", "plan", "--strategy", "world-model", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    settings = payload["starting_settings"]
    ladder_stages = {step["stage"] for step in payload["readiness_ladder"]}
    rule_signals = {rule["signal"] for rule in payload["adjustment_rules"]}
    assert payload["strategy"] == "world-model"
    assert settings["echo_aux_lambda"] == 0.05
    assert settings["rwml_distance_threshold"] == 0.2
    assert settings["rwml_history_window"] == 4
    assert "world_model_records" in payload["watch"]
    assert (
        "Export replay with include_world_model_replay=true." in payload["recommended_next_steps"]
    )
    assert {"replay_coverage", "backend_probe", "diagnostic_quality"} <= ladder_stages
    assert "world-model coverage exists but quality is unclear" in rule_signals


def test_training_plan_reward_model_contains_reward_defaults(capsys):
    assert main(["training", "plan", "--strategy", "reward-model", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    settings = payload["starting_settings"]
    ladder_stages = {step["stage"] for step in payload["readiness_ladder"]}
    rule_signals = {rule["signal"] for rule in payload["adjustment_rules"]}
    settings_help = {item["setting"]: item for item in payload["settings_help"]}
    metric_guide = {item["metric"]: item for item in payload["metric_guide"]}

    assert payload["strategy"] == "reward-model"
    assert settings["reward_artifact"] == "reward_examples.jsonl"
    assert settings["reward_loss"] == "pairwise_or_regression"
    assert settings["eval_split_required"] is True
    assert "heldout_pair_accuracy" in payload["watch"]
    assert "eval_only_leakage" in payload["watch"]
    assert "reward_artifact" in settings_help
    assert metric_guide["heldout_pair_accuracy"]["role"] == "reward_model_health"
    assert metric_guide["eval_only_leakage"]["role"] == "safety_release"
    assert {"reward_artifact_contract", "heldout_reward_eval", "use_site_gate"} <= ladder_stages
    assert "length bias or task-family skew appears" in rule_signals
    assert payload["next"][0]["command"].startswith("bashgym training reward-examples validate")


def test_training_plan_reward_model_aliases_are_canonical(capsys):
    assert main(["training", "plan", "--strategy", "prm", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "reward-model"
    assert "process_reward" in payload["starting_settings"]["reward_type"]


def test_training_plan_grpo_adjusts_group_size_for_hardware(capsys):
    assert main(["training", "plan", "--strategy", "grpo", "--hardware", "dgx", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    ladder_stages = {step["stage"] for step in payload["readiness_ladder"]}
    rule_signals = {rule["signal"] for rule in payload["adjustment_rules"]}
    assert payload["starting_settings"]["training_profile"] == "terminal_rl_tmax_like"
    assert payload["starting_settings"]["grpo_group_size"] == 32
    assert "frac_reward_zero_std" in payload["watch"]
    assert "Confirm reward groups have non-zero variance." in payload["recommended_next_steps"]
    assert {"rollout_contrast", "replay_contract", "backend_smoke", "release_gate"} <= ladder_stages
    assert "reward_std is zero or frac_reward_zero_std is near 1.0" in rule_signals


def test_training_plan_sft_exposes_operator_next_steps(capsys):
    assert main(["training", "plan", "--strategy", "sft", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "sft"
    settings_help = {item["setting"]: item for item in payload["settings_help"]}
    metric_guide = {item["metric"]: item for item in payload["metric_guide"]}
    assert "learning_rate" in settings_help
    assert "adjust_when" in settings_help["max_seq_length"]
    assert metric_guide["train_loss"]["role"] == "training_health"
    assert metric_guide["heldout_pass@k"]["role"] == "behavior_evidence"
    assert payload["starting_settings"]["num_epochs"] == 3
    assert payload["starting_settings"]["artifact_retention"] == "adapter_only"
    assert payload["starting_settings"]["checkpoint_limit"] == 1
    assert payload["starting_settings"]["hf_upload_artifact"] == "auto"
    assert payload["recommended_next_steps"] == [
        "Inspect generated examples for truncation.",
        "Run heldout trace eval and executable environment pass@k before routing.",
    ]


def test_replay_summarize_reports_world_model_coverage(tmp_path, capsys):
    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_cli",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_cli",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-cli",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert main(["replay", "summarize", str(replay_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    summary = payload["summary"]
    assert summary["records"] == 1
    assert summary["world_model_records"] == 1
    assert summary["world_model"]["rwml_transitions"] == 1
    assert summary["world_model"]["echo_observation_chars"] == len("answer.txt\n")


def test_training_smoke_bundle_reports_backend_readiness(tmp_path, capsys):
    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_cli_smoke",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_cli_smoke",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={
                "response_logprobs": [{"tokens": ["ls"], "token_logprobs": [-0.1]}],
                "behavior_logprob_tokens": 1,
                "train_logprob_tokens": 1,
            },
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    output_dir = tmp_path / "bundle"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-cli-smoke",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert (
        main(
            [
                "training",
                "smoke-bundle",
                "--replay",
                str(replay_path),
                "--output-dir",
                str(output_dir),
                "--base-model",
                "example/operator-selected-trainable-model",
                "--backend",
                "grpo_fallback",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "bashgym.backend_smoke_bundle.v1"
    assert payload["contract_ready"] is True
    assert payload["optimizer_ready"] is True
    assert payload["backend_launch_ready"] is False
    assert payload["verdict"]["level"] == "needs_backend"
    assert payload["world_model_probe"]["batch"]["rwml_transitions"] == 1
    assert output_dir.joinpath("backend_smoke_readiness.json").exists()


def test_training_analyze_combines_metrics_replay_and_release_evidence(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 2.0, "reward_std": 0.1}),
                json.dumps(
                    {
                        "step": 2,
                        "loss": 1.5,
                        "reward_std": 0.0,
                        "frac_reward_zero_std": 0.75,
                        "kl": 0.05,
                        "entropy": 2.1,
                        "preference_accuracy": 0.62,
                        "reward_margin": 0.3,
                        "verifier_error_rate": 0.03,
                        "tokens_per_second": 123.0,
                        "gpu_memory_peak_gb": 14.0,
                        "oom_count": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps({"ship": False, "reasons": ["environment holdout: pass@1 too low"]}),
        encoding="utf-8",
    )

    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_analyze",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_analyze",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-analyze",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--replay",
                str(replay_path),
                "--release-evidence",
                str(release_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["schema_version"] == "bashgym.run_analysis.v1"
    assert payload["verdict"]["level"] == "blocked"
    assert payload["training_metrics"]["loss"]["last"] == 1.5
    assert payload["training_metrics"]["kl"]["last"] == 0.05
    assert payload["training_metrics"]["entropy"]["last"] == 2.1
    assert payload["training_metrics"]["preference_accuracy"]["last"] == 0.62
    assert payload["training_metrics"]["reward_margin"]["last"] == 0.3
    assert payload["training_metrics"]["verifier_error_rate"]["last"] == 0.03
    assert payload["training_metrics"]["tokens_per_second"]["last"] == 123.0
    assert payload["training_metrics"]["gpu_memory_peak_gb"]["last"] == 14.0
    assert payload["training_metrics"]["oom_count"]["last"] == 1.0
    assert payload["replay_summary"]["world_model_records"] == 1
    assert "zero_reward_variance" in codes
    assert "many_zero_std_groups" in codes
    assert "verifier_errors_elevated" in codes
    assert "oom_seen" in codes
    assert "world_model_quality_missing" in codes
    assert "release_gate_blocked" in codes


def test_training_analyze_reports_session_distillation_metrics(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "session_distillation_loss": 0.4,
                        "session_distillation_kl": 0.2,
                        "session_distillation_ce": 0.6,
                        "session_distillation_masked_tokens": 0,
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["training", "analyze", "--metrics", str(metrics_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    doc_topics = {doc["topic"] for doc in payload["docs"]}
    assert payload["training_metrics"]["session_distillation_loss"]["last"] == 0.4
    assert payload["training_metrics"]["session_distillation_kl"]["last"] == 0.2
    assert payload["training_metrics"]["session_distillation_ce"]["last"] == 0.6
    assert payload["training_metrics"]["session_distillation_masked_tokens"]["last"] == 0.0
    assert "session_distillation_zero_masked_tokens" in codes
    assert "session-distillation" in doc_topics


def test_training_analyze_reads_run_id_from_models_dir(tmp_path, capsys):
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "analyze",
                "--run-id",
                "run-001",
                "--models-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-001"
    assert payload["training_metrics"]["points"] == 1
    assert payload["verdict"]["summary"]["has_heldout_signal"] is True


def test_training_analyze_accepts_release_world_model_quality(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "release_gate": {
                    "ship": True,
                    "world_model_quality": {
                        "present": True,
                        "diagnostic_only": True,
                        "signal": "improving",
                        "metrics": {"echo_loss": 0.7, "echo_loss_delta": -0.2},
                        "findings": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_quality",
            "instruction": "List files",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_quality",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={},
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    replay_path = tmp_path / "replay.jsonl"
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-quality",
        include_world_model=True,
    )
    write_dppo_records_jsonl(replay_path, records)

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--replay",
                str(replay_path),
                "--release-evidence",
                str(release_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert "world_model_quality_missing" not in codes
    assert payload["release_evidence"]["world_model_quality"]["signal"] == "improving"
    assert payload["verdict"]["summary"]["has_world_model_quality"] is True


def test_training_analyze_reports_blocked_smoke_bundle(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )
    smoke_path = tmp_path / "backend_smoke_readiness.json"
    smoke_path.write_text(
        json.dumps(
            {
                "schema_version": "bashgym.backend_smoke_bundle.v1",
                "contract_ready": False,
                "optimizer_ready": False,
                "backend_launch_ready": False,
                "verdict": {
                    "level": "blocked",
                    "summary": "Replay is not ready.",
                    "blocking_codes": ["missing_world_model_payloads"],
                },
                "checks": [
                    {
                        "code": "missing_world_model_payloads",
                        "status": "fail",
                        "message": "0/1 records include world_model payloads.",
                        "next_step": "Export replay with include_world_model_replay=true.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--smoke-bundle",
                str(smoke_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["smoke_bundle"]["present"] is True
    assert payload["verdict"]["level"] == "blocked"
    assert payload["verdict"]["summary"]["backend_contract_ready"] is False
    assert "smoke_bundle_contract_blocked" in codes


def test_training_analyze_reports_smoke_bundle_needs_backend(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "loss": 2.0, "heldout_pass_at_1": 0.2}) + "\n",
        encoding="utf-8",
    )
    smoke_path = tmp_path / "backend_smoke_readiness.json"
    smoke_path.write_text(
        json.dumps(
            {
                "schema_version": "bashgym.backend_smoke_bundle.v1",
                "contract_ready": True,
                "optimizer_ready": True,
                "backend_launch_ready": False,
                "verdict": {
                    "level": "needs_backend",
                    "summary": "Install/configure a backend.",
                    "blocking_codes": ["backend_launch_plan"],
                },
                "checks": [
                    {
                        "code": "backend_launch_plan",
                        "status": "warn",
                        "message": "No backend detected.",
                        "next_step": "Install/configure verl, SkyRL, or TMax/open-instruct.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "training",
                "analyze",
                "--metrics",
                str(metrics_path),
                "--smoke-bundle",
                str(smoke_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["smoke_bundle"]["present"] is True
    assert payload["verdict"]["level"] == "needs_attention"
    assert payload["verdict"]["summary"]["backend_contract_ready"] is True
    assert payload["verdict"]["summary"]["backend_optimizer_ready"] is True
    assert payload["verdict"]["summary"]["backend_launch_ready"] is False
    assert "smoke_bundle_needs_backend" in codes


def test_serve_forwards_server_arguments(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_serve_main() -> None:
        captured["argv"] = sys.argv[:]

    monkeypatch.setattr("bashgym.main.main", fake_serve_main)

    assert (
        main(
            [
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8002",
                "--log-level",
                "warning",
                "--",
                "--workers",
                "1",
            ]
        )
        == 0
    )

    assert captured["argv"] == [
        "bashgym serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8002",
        "--log-level",
        "warning",
        "--workers",
        "1",
    ]


def test_training_session_records_build_cli(tmp_path, capsys):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "failed.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "metadata": {"user_initial_prompt": "Do it."},
                "primary_repo": {"name": "ghostwork"},
                "trace": [
                    {
                        "tool_name": "Bash",
                        "command": "python x.py",
                        "output": "ModuleNotFoundError: No module named 'x'",
                        "success": False,
                        "exit_code": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "records.jsonl"

    assert (
        main(["training", "session-records", "build", str(traces), "--out", str(out), "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["record_count"] == 1
    assert payload["validation_errors"] == []
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "failed"


def test_training_runcard_cli_accepts_session_distillation_evidence(tmp_path, capsys):
    path = tmp_path / "sd_card.json"
    assert (
        main(
            [
                "training",
                "runcard",
                "create",
                "--run-id",
                "sd-cli",
                "--training-method",
                "session_distillation",
                "--base-model",
                "tiny-local-model",
                "--compute-target",
                "local_cpu_or_gpu",
                "--session-distillation-records",
                "data/session_distillation/records.jsonl",
                "--session-distillation-metrics",
                "data/models/sd/metrics.jsonl",
                "--reader-model",
                "heuristic-session-distillation-reader-v1",
                "--confidence-threshold",
                "0.6",
                "--hint-policy",
                "heuristic",
                "--mask-policy",
                "target_span_only",
                "--target-token-count",
                "128",
                "--output",
                str(path),
                "--no-git",
                "--promotion",
                "--json",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    card = created["run_card"]
    assert card["session_distillation_records_path"] == "data/session_distillation/records.jsonl"
    assert card["session_distillation_reader_model"] == "heuristic-session-distillation-reader-v1"
    assert card["session_distillation_mask_policy"] == "target_span_only"
    assert card["session_distillation_target_token_count"] == 128
    # The evidence flags plumbed through, so the fields are no longer reported as
    # missing-entirely (the remaining findings are only "file does not exist",
    # since these test paths are not created on disk).
    codes = {finding["code"] for finding in created["findings"]}
    assert "missing_session_distillation_records_path" not in codes
    assert "missing_session_distillation_reader_model" not in codes
    assert "missing_session_distillation_mask_policy" not in codes
