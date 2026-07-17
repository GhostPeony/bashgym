"""Exact public contracts for the campaign control-room snapshot."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.contracts import CampaignControlRoomSnapshotV1

FIXTURE_DIR = Path(__file__).with_name("fixtures")


def test_draft_snapshot_fixture_is_the_exact_frozen_public_contract():
    payload = json.loads(
        (FIXTURE_DIR / "control_room_snapshot_draft.json").read_text(encoding="utf-8")
    )

    snapshot = CampaignControlRoomSnapshotV1.model_validate(payload)

    assert json.loads(snapshot.model_dump_json()) == payload
    assert tuple(phase.phase_id for phase in snapshot.journey) == (
        "setup",
        "baseline",
        "experiments",
        "human_review",
        "decision",
    )
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.authorization_revision = 2


def test_public_snapshot_rejects_unbounded_nested_collections():
    payload = json.loads(
        (FIXTURE_DIR / "control_room_snapshot_draft.json").read_text(encoding="utf-8")
    )
    oversized = deepcopy(payload)
    oversized["budget"]["resources"] = [
        {
            "schema_version": "budget_resource_summary.v1",
            "unit": f"resource_{index}",
            "limit": 1,
            "reserved": 0,
            "settled": 0,
            "remaining": 1,
            "blocked": False,
            "blocker_code": None,
        }
        for index in range(65)
    ]

    with pytest.raises(ValidationError, match="too_long"):
        CampaignControlRoomSnapshotV1.model_validate(oversized)


def test_all_approved_state_fixtures_materialize_exact_complete_snapshots():
    base = json.loads(
        (FIXTURE_DIR / "control_room_snapshot_draft.json").read_text(encoding="utf-8")
    )
    cases = json.loads(
        (FIXTURE_DIR / "control_room_snapshot_cases.json").read_text(encoding="utf-8")
    )
    action_contracts = {
        "cancel": ("campaign.cancel", "lifecycle"),
        "pause": ("campaign.pause", "lifecycle"),
        "start": ("campaign.start", "lifecycle"),
    }
    names: set[str] = set()

    for case in cases:
        names.add(case["name"])
        payload = deepcopy(base)
        payload["campaign"]["status"] = case["campaign_status"]
        payload["controller"]["state"] = case["controller_state"]
        if case["controller_state"] == "online":
            payload["controller"].update(
                {
                    "controller_observation_version": 1,
                    "heartbeat_age_seconds": 0,
                    "lease_expires_at": "2026-07-16T12:00:10Z",
                    "controller_instance_id": "resident-worker",
                    "safe_guidance": None,
                }
            )
        elif case["controller_state"] == "stale":
            payload["controller"].update(
                {
                    "controller_observation_version": 2,
                    "heartbeat_age_seconds": 30,
                    "lease_expires_at": "2026-07-16T11:59:50Z",
                    "controller_instance_id": "resident-worker",
                    "safe_guidance": "Reconcile the resident controller.",
                }
            )
        if case["readiness"] == "ready":
            payload["readiness"].update(
                {"materializable": True, "launch_ready": True, "blocking_codes": []}
            )
        for phase, state in zip(payload["journey"], case["journey_states"], strict=True):
            phase["state"] = state
            phase["primary_blocker"] = None
            phase["attention_owner"] = "none"
        blocker = None
        if case["blocker"] is not None:
            blocker = {
                "schema_version": "decision_blocker.v1",
                "code": case["blocker"],
                "summary": "Server-derived control-room blocker.",
                "evidence_ids": [],
                "secondary_codes": case["secondary_codes"],
            }
            for phase in payload["journey"]:
                if phase["state"] == "blocked":
                    phase["primary_blocker"] = blocker
                    phase["attention_owner"] = "bashgym"
        payload["decision_surface"]["blocker"] = blocker
        payload["decision_surface"]["attention_owner"] = (
            "bashgym" if blocker is not None else "none"
        )
        payload["decision_surface"]["next_actions"] = [
            {
                "schema_version": "decision_action.v1",
                "action": action,
                "capability": action_contracts[action][0],
                "freshness_class": action_contracts[action][1],
                "requires_human_work": False,
            }
            for action in case["next_actions"]
        ]
        payload["decision_surface"]["recovery_actions"] = case["recovery_actions"]
        if case["name"] == "running":
            payload["active_work"] = {
                "schema_version": "active_work_summary.v1",
                "study_id": "study-1",
                "proposal_id": "proposal-1",
                "action_id": "action-1",
                "attempt_id": "attempt-1",
                "stage": "smoke_training",
                "hypothesis_summary": None,
                "primary_variable_summary": None,
                "controlled_variable_summary": [],
                "progress_fraction": None,
                "eta_seconds": None,
                "executor_type": "fake",
                "process_identity": None,
            }
            payload["decision_surface"]["execution_owner"] = "bashgym"
        if case["name"] == "completed":
            payload["campaign"]["champion_ref"] = "candidate-1"
            payload["champion"] = {
                "schema_version": "candidate_summary.v1",
                "candidate_ref": "candidate-1",
                "source_attempt_ids": ["attempt-1"],
                "source_artifact_ids": ["artifact-1"],
                "latest_comparable_evaluation_id": "evaluation-1",
                "comparison_verdict": "passed",
                "gate_state": "promoted",
            }

        snapshot = CampaignControlRoomSnapshotV1.model_validate(payload)
        rendered = json.loads(snapshot.model_dump_json())
        assert rendered == payload
        assert all(key in rendered for key in ("active_work", "champion", "candidate"))
        assert all(key in rendered["controller"] for key in ("heartbeat_age_seconds", "lease_expires_at"))

    assert names == {
        "empty_draft",
        "ready",
        "running",
        "failed",
        "exhausted",
        "cancelled",
        "completed",
        "stale_controller",
        "invariant_failure",
    }
