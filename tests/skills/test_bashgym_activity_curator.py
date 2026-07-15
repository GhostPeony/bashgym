import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "assistant"
    / "workspace"
    / "skills"
    / "bashgym-operator"
    / "scripts"
    / "curate_activity.py"
)
SPEC = importlib.util.spec_from_file_location("bashgym_activity_curator", SCRIPT)
assert SPEC and SPEC.loader
curator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curator)


def test_receipt_is_idempotent_and_strips_secrets_paths_and_raw_logs(tmp_path):
    receipt = {
        "schema_version": "bashgym.activity.v1",
        "kind": "evaluation",
        "workspace_id": "memexai",
        "entity_id": "eval-a",
        "status": "completed",
        "occurred_at": "2026-07-13T18:20:00Z",
        "objective": "Compare against the frozen champion.",
        "configuration": {
            "model": "Qwen/Qwen3",
            "dataset_path": "/home/trainer/private/dev.jsonl",
            "api_token": "secret-value",
        },
        "metrics": {"mrr_delta": -0.012, "raw_logs": ["too", "large"]},
        "artifact_references": [{"id": "export-a", "digest": "a" * 64}],
    }

    relative, changed = curator._write_receipt(tmp_path, receipt)
    second_relative, second_changed = curator._write_receipt(tmp_path, receipt)

    assert relative == "evaluations/memexai/eval-a.md"
    assert second_relative == relative
    assert changed is True
    assert second_changed is False
    content = (tmp_path / relative).read_text(encoding="utf-8")
    assert "secret-value" not in content
    assert "/home/trainer" not in content
    assert "dev.jsonl" in content
    assert "too" not in content
    assert "[retained in BashGym]" in content


def test_workspace_campaign_becomes_task_general_training_session_receipt():
    context = {
        "schema_version": "bashgym.workspace.context.v1",
        "generated_at": "2026-07-13T18:20:00Z",
        "workspace_id": "research",
        "campaign_activity": [
            {
                "campaign_id": "sft-session",
                "event_id": "evt-1",
                "cursor": 3,
            }
        ],
    }
    campaign = {
        "campaign_id": "sft-session",
        "title": "General SFT session",
        "kind": "general",
        "objective": "Improve instruction following.",
        "status": "active",
        "version": 4,
        "updated_at": "2026-07-13T18:19:00Z",
        "target_model": {"base_model_ref": "org/base", "task": "instruction-following"},
        "budget_remaining": {"gpu_hours": 2.5},
        "study_status_counts": {"full_training": 1},
        "attempt_status_counts": {"running": 1},
        "latest_event_cursor": 3,
    }

    receipt = curator._campaign_receipt(context, campaign)
    kind, workspace, entity, content = curator.render_receipt(receipt)

    assert (kind, workspace, entity) == ("training-session", "research", "sft-session")
    assert "Improve instruction following" in content
    assert '"gpu_hours": 2.5' in content
    assert "evt-1" in content
    assert "embedding" not in content.casefold()


def test_receipt_directories_use_stable_known_plurals():
    assert curator._kind_directory("training-session") == "training-sessions"
    assert curator._kind_directory("run-inventory") == "run-inventories"
    assert curator._kind_directory("evaluation") == "evaluations"
    assert curator._kind_directory("session-handoff") == "session-handoffs"


def test_session_handoff_is_bounded_and_project_identified(tmp_path):
    payload = {
        "schema_version": "bashgym.session-handoff.v1",
        "workspace_id": "research",
        "project_id": "general-llm",
        "session_id": "codex-20260714-context-canvas",
        "updated_at": "2026-07-14T20:00:00Z",
        "summary": "Added authority metadata and prepared a dry campaign proof.",
        "decisions": ["Live runtime outranks conversation memory."],
        "next_actions": ["Run the campaign dry proof."],
        "evidence_refs": [{"kind": "test", "id": "context-authority-tests"}],
    }

    receipt = curator._handoff_receipt(payload)
    relative, changed = curator._write_receipt(tmp_path, receipt)

    assert relative == "session-handoffs/research/codex-20260714-context-canvas.md"
    assert changed is True
    content = (tmp_path / relative).read_text(encoding="utf-8")
    assert 'project_id: "general-llm"' in content
    assert "Live runtime outranks conversation memory" in content
