"""Stateless agent-facing projections of one experiment snapshot."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from bashgym.campaigns.agent_brief import build_agent_brief
from bashgym.campaigns.contracts import CampaignControlRoomSnapshotV1

FIXTURE = Path(__file__).with_name("fixtures") / "control_room_snapshot_draft.json"


def _draft_snapshot() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _active_snapshot() -> dict[str, object]:
    payload = _draft_snapshot()
    payload.update(
        {
            "aggregate_version": 7,
            "manifest_revision": 2,
            "authorization_revision": 3,
            "latest_event_cursor": 42,
        }
    )
    payload["campaign"].update(
        {
            "status": "active",
            "aggregate_version": 7,
            "manifest_revision": 2,
            "active_study_id": "study-1",
            "active_action_id": "action-1",
            "champion_ref": "baseline-v1",
        }
    )
    for phase, state in zip(
        payload["journey"],
        ("complete", "complete", "active", "not_started", "not_started"),
        strict=True,
    ):
        phase.update(
            {
                "state": state,
                "primary_blocker": None,
                "attention_owner": "agent" if state == "active" else "none",
                "execution_owner": "bashgym" if state == "active" else "none",
            }
        )
    payload["active_work"] = {
        "schema_version": "active_work_summary.v1",
        "study_id": "study-1",
        "proposal_id": "proposal-1",
        "action_id": "action-1",
        "attempt_id": "attempt-1",
        "stage": "full_training",
        "hypothesis_summary": "A focused data recipe improves tool-call recovery.",
        "primary_variable_summary": "training data recipe",
        "controlled_variable_summary": ["model", "evaluator"],
        "progress_fraction": 0.5,
        "eta_seconds": 120,
        "executor_type": "ssh_remote",
        "process_identity": None,
    }
    payload["bindings"]["compute"] = {
        "schema_version": "safe_binding_identity.v1",
        "binding_id": "compute-private-1",
        "immutable_digest": "a" * 64,
        "display_label": "Research compute",
    }
    payload["champion"] = {
        "schema_version": "candidate_summary.v1",
        "candidate_ref": "baseline-v1",
        "source_attempt_ids": ["attempt-0"],
        "source_artifact_ids": [],
        "latest_comparable_evaluation_id": "evaluation-0",
        "comparison_verdict": "passed",
        "gate_state": "passed",
    }
    payload["candidate"] = {
        "schema_version": "candidate_summary.v1",
        "candidate_ref": "candidate-v2",
        "source_attempt_ids": ["attempt-1"],
        "source_artifact_ids": ["artifact-1"],
        "latest_comparable_evaluation_id": "evaluation-1",
        "comparison_verdict": "passed",
        "gate_state": "passed",
    }
    payload["metrics"] = [
        {
            "schema_version": "metric_descriptor.v1",
            "metric_id": "task-success",
            "display_name": "Task success",
            "unit": "%",
            "direction": "maximize",
            "target": 0.8,
            "tolerance": None,
            "evaluator_revision": "eval-r1",
            "sample_count": 50,
            "uncertainty_method": None,
            "comparability_key": "b" * 64,
        }
    ]
    payload["budget"] = {
        "schema_version": "budget_summary.v1",
        "resources": [
            {
                "schema_version": "budget_resource_summary.v1",
                "unit": "gpu_hours",
                "limit": 10,
                "reserved": 2,
                "settled": 3,
                "remaining": 5,
                "blocked": False,
                "blocker_code": None,
            }
        ],
        "blocked": False,
    }
    payload["decision_surface"] = {
        "schema_version": "decision_surface.v1",
        "execution_owner": "bashgym",
        "attention_owner": "agent",
        "blocker": None,
        "next_actions": [
            {
                "schema_version": "decision_action.v1",
                "action": "inspect-active-work",
                "capability": "campaign.read",
                "freshness_class": "read",
                "requires_human_work": False,
            }
        ],
        "recovery_actions": [],
        "promotion_eligible": False,
    }
    return payload


def test_builds_compact_goal_and_markdown_from_typed_snapshot():
    snapshot = CampaignControlRoomSnapshotV1.model_validate(_active_snapshot())

    brief = build_agent_brief(
        snapshot,
        control_room_url=(
            "http://localhost:3000/?view=training&tab=autoresearch"
            "&workspace_id=workspace-a&campaign_id=campaign-1"
        ),
        stop_conditions=(
            "At most 3 experiment attempts",
            "At most 3 gpu_hours",
        ),
    )

    assert brief["schema_version"] == "bashgym.agent_brief.v1"
    assert brief["goal"] == {
        "objective": "Establish a safe baseline",
        "success_criteria": ["Task success target 0.8 % (maximize)"],
        "stop_conditions": [
            "At most 3 experiment attempts",
            "At most 3 gpu_hours",
        ],
        "completed_work": [
            {"phase": "setup", "evidence_count": 0},
            {"phase": "baseline", "evidence_count": 0},
        ],
        "current_work": {
            "phase": "experiments",
            "state": "active",
            "stage": "full_training",
            "summary": "A focused data recipe improves tool-call recovery.",
            "progress_fraction": 0.5,
            "eta_seconds": 120,
        },
        "next_action": {
            "action": "inspect-active-work",
            "capability": "campaign.read",
            "freshness_class": "read",
            "requires_human_work": False,
            "source": "decision_surface",
        },
        "resume": {
            "reference": "workspace-a/campaign-1",
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "aggregate_version": 7,
            "manifest_revision": 2,
            "authorization_revision": 3,
            "latest_event_cursor": 42,
            "snapshot_at": "2026-07-16T12:00:00Z",
        },
        "control_room": {
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "url": (
                "http://localhost:3000/?view=training&tab=autoresearch"
                "&workspace_id=workspace-a&campaign_id=campaign-1"
            ),
        },
    }
    markdown = brief["markdown"]
    assert "**Timeline:** setup ✓ → baseline ✓ → experiments ●" in markdown
    assert "**Current stage:** full training · 50% · ETA 2m" in markdown
    assert "**Baseline/champion vs candidate:** baseline-v1" in markdown
    assert "candidate-v2 (passed; evaluation evaluation-1)" in markdown
    assert "**Compute:**" not in markdown
    assert "gpu_hours: 3 settled + 2 reserved / 10; 5 remaining" in markdown
    assert "**Latest finding:** Candidate comparison verdict: passed." in markdown
    assert "**Next action:** inspect-active-work" in markdown
    assert "[Open experiment view](http://localhost:3000/" in markdown


def test_missing_optional_display_data_stays_explicitly_unknown():
    payload = {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "aggregate_version": 1,
        "manifest_revision": 1,
        "authorization_revision": 1,
        "latest_event_cursor": 0,
        "campaign": {
            "objective": "Measure a pinned baseline.",
            "status": "ready",
            "stop_reason": None,
        },
    }

    brief = build_agent_brief(payload)

    assert brief["goal"]["success_criteria"] == []
    assert brief["goal"]["stop_conditions"] == []
    assert brief["goal"]["completed_work"] == []
    assert brief["goal"]["current_work"] is None
    assert brief["goal"]["next_action"] is None
    assert brief["goal"]["resume"]["snapshot_at"] is None
    assert brief["goal"]["control_room"]["url"] is None
    assert "**Current stage:** ready" in brief["markdown"]
    assert "**Baseline/champion vs candidate:** Not reported in this snapshot." in brief["markdown"]
    assert "**Compute:**" not in brief["markdown"]
    assert "**Budget:** No resource usage reported in this snapshot." in brief["markdown"]
    assert "**Latest finding:** Not reported in this snapshot." in brief["markdown"]
    assert "**Next action:** None reported." in brief["markdown"]
    assert "experiment view" not in brief["markdown"].casefold()


def test_uses_exact_server_blocker_and_recovery_action_without_mutating_input():
    payload = _draft_snapshot()
    original = deepcopy(payload)

    brief = build_agent_brief(payload)

    assert payload == original
    assert brief["goal"]["current_work"] == {
        "phase": "setup",
        "state": "blocked",
        "stage": None,
        "summary": "Campaign readiness checks have not passed.",
        "progress_fraction": None,
        "eta_seconds": None,
    }
    assert brief["goal"]["next_action"] == {
        "action": "inspect",
        "capability": None,
        "freshness_class": None,
        "requires_human_work": None,
        "source": "recovery",
    }
    assert "**Latest finding:** Campaign readiness checks have not passed." in brief["markdown"]
    assert "**Next action:** inspect (recovery)" in brief["markdown"]


def test_terminal_stop_reason_is_preserved_as_available_goal_context():
    payload = _draft_snapshot()
    payload["campaign"].update({"status": "completed", "stop_reason": "target_reached"})

    brief = build_agent_brief(payload)

    assert brief["goal"]["stop_conditions"] == ["target_reached"]
    assert "**Latest finding:** Campaign stop reason: target_reached." in brief["markdown"]


def test_configured_stop_conditions_and_terminal_reason_are_combined_without_duplicates():
    payload = _draft_snapshot()
    payload["campaign"].update({"status": "completed", "stop_reason": "target_reached"})

    brief = build_agent_brief(
        payload,
        stop_conditions=("At most 3 experiment attempts", "target_reached"),
    )

    assert brief["goal"]["stop_conditions"] == [
        "At most 3 experiment attempts",
        "target_reached",
    ]


def test_rejects_snapshot_without_durable_identity():
    with pytest.raises(ValueError, match="workspace_id"):
        build_agent_brief({"campaign_id": "campaign-1", "campaign": {"objective": "x"}})
