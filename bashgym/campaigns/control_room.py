"""Bounded control-room projections over the durable campaign ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bashgym.campaigns.contracts import (
    ActiveWorkSummaryV1,
    ActorPrincipal,
    BindingSummaryV1,
    BudgetResourceSummaryV1,
    BudgetSummaryV1,
    Campaign,
    CampaignControlRoomSnapshotV1,
    CampaignControlRoomStateV1,
    CampaignStatus,
    CampaignSummaryV1,
    CampaignTrigger,
    CandidateSummaryV1,
    Capability,
    CollectionCursorV1,
    CollectionSummaryV1,
    ControllerObservationV1,
    ControlRoomArtifactSummaryV1,
    ControlRoomCampaignV1,
    ControlRoomCollectionSummaryV1,
    ControlRoomStatusCountV1,
    DecisionActionV1,
    DecisionBlockerV1,
    DecisionSurfaceV1,
    HumanWorkSummaryV1,
    JourneyPhaseSummaryV1,
    ManifestRevision,
    MetricDescriptorV1,
    OpaqueProcessIdentityV1,
    ReadinessSummaryV1,
    SafeBindingIdentityV1,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.campaigns.transitions import allowed_triggers, evaluate_promotion_gate


@dataclass(frozen=True)
class ControlRoomDurableProjection:
    """Private one-transaction inputs for the public control-room projection."""

    campaign: Campaign
    manifest: ManifestRevision
    latest_event_cursor: int
    collection_counts: dict[str, int]
    budget_totals: dict[str, tuple[float, float, float]]
    active_studies: tuple[dict[str, Any], ...]
    active_actions: tuple[dict[str, Any], ...]
    active_attempts: tuple[dict[str, Any], ...]
    remote_runs: tuple[dict[str, Any], ...]
    latest_gate: dict[str, Any] | None
    latest_evaluation_id: str | None
    autoresearch_spec: dict[str, Any] | None
    baseline_outcome: dict[str, Any] | None
    candidate_outcome: dict[str, Any] | None
    protected_gate_passed: bool
    human_work: tuple[dict[str, Any], ...] = ()
    agents: tuple[dict[str, Any], ...] = ()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def read_control_room_projection(
    connection: sqlite3.Connection,
    workspace_id: str,
    campaign_id: str,
    *,
    preview_limit: int = 10,
) -> ControlRoomDurableProjection:
    """Read all campaign-owned projection inputs through the caller's transaction."""

    if preview_limit < 1 or preview_limit > 10:
        raise ValueError("control-room preview limit must be between 1 and 10")
    row = connection.execute(
        "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
        (workspace_id, campaign_id),
    ).fetchone()
    if row is None:
        raise RecordNotFoundError("campaign not found")
    campaign = Campaign.model_validate(
        {
            "campaign_id": row["campaign_id"],
            "workspace_id": row["workspace_id"],
            "title": row["title"],
            "kind": row["kind"],
            "objective": row["objective"],
            "target_model": json.loads(row["target_model_json"]),
            "owner_actor_id": row["owner_actor_id"],
            "manifest_revision": row["manifest_revision"],
            "status": row["status"],
            "prior_scheduling_status": row["prior_scheduling_status"],
            "active_study_id": row["active_study_id"],
            "active_action_id": row["active_action_id"],
            "champion_ref": row["champion_ref"],
            "best_development_candidate_ref": row["best_development_candidate_ref"],
            "stop_reason": row["stop_reason"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )
    manifest_row = connection.execute(
        """
        SELECT * FROM campaign_manifest_revisions
        WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
        """,
        (workspace_id, campaign_id, campaign.manifest_revision),
    ).fetchone()
    if manifest_row is None:
        raise RuntimeError("campaign_projection_invariant_failed")
    manifest = ManifestRevision.model_validate(
        {
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "revision": manifest_row["revision"],
            "manifest": json.loads(manifest_row["manifest_json"]),
            "manifest_hash": manifest_row["manifest_hash"],
            "actor_id": manifest_row["actor_id"],
            "correlation_id": manifest_row["correlation_id"],
            "created_at": manifest_row["created_at"],
        }
    )
    event_row = connection.execute(
        """
        SELECT COUNT(*) AS item_count, COALESCE(MAX(cursor), 0) AS latest_cursor
        FROM campaign_events WHERE workspace_id = ? AND campaign_id = ?
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    count_row = connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM campaign_proposals WHERE workspace_id = ? AND campaign_id = ?) AS proposals,
          (SELECT COUNT(*) FROM campaign_studies WHERE workspace_id = ? AND campaign_id = ?) AS studies,
          (SELECT COUNT(*) FROM campaign_attempts t JOIN campaign_actions a
             ON a.workspace_id = t.workspace_id AND a.action_id = t.action_id
             WHERE a.workspace_id = ? AND a.campaign_id = ?) AS attempts,
          (SELECT COUNT(*) FROM campaign_artifacts WHERE workspace_id = ? AND campaign_id = ?) AS artifacts,
          (SELECT COUNT(*) FROM campaign_gate_decisions WHERE workspace_id = ? AND campaign_id = ?) AS comparisons
        """,
        (workspace_id, campaign_id) * 5,
    ).fetchone()
    budget_rows = connection.execute(
        """
        SELECT unit, COALESCE(SUM(reserved_delta), 0) AS reserved,
               COALESCE(SUM(actual_delta), 0) AS settled,
               COALESCE(SUM(limit_delta), 0) AS limit_delta
        FROM campaign_budget_ledger
        WHERE workspace_id = ? AND campaign_id = ? GROUP BY unit ORDER BY unit
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    active_studies = connection.execute(
        """
        SELECT study_id, proposal_id, status, current_stage_index, stage_plan_json
        FROM campaign_studies WHERE workspace_id = ? AND campaign_id = ?
          AND status NOT IN ('completed','rejected','promoted','final_rejected',
                             'execution_failed','abandoned','cancelled')
        ORDER BY updated_at DESC, study_id LIMIT 2
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    active_actions = connection.execute(
        """
        SELECT action_id, study_id, stage_index, stage_kind, status, candidate_digest
        FROM campaign_actions WHERE workspace_id = ? AND campaign_id = ?
          AND status NOT IN ('completed','failed','force_stopped','cancelled')
        ORDER BY updated_at DESC, action_id LIMIT 2
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    active_attempts = connection.execute(
        """
        SELECT t.attempt_id, t.action_id, a.study_id, t.status, t.executor_json,
               a.stage_kind, a.candidate_digest
        FROM campaign_attempts t JOIN campaign_actions a
          ON a.workspace_id = t.workspace_id AND a.action_id = t.action_id
        WHERE a.workspace_id = ? AND a.campaign_id = ?
          AND t.status NOT IN ('completed','failed','force_stopped','cancelled')
        ORDER BY t.updated_at DESC, t.attempt_id LIMIT 2
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    remote_runs = connection.execute(
        """
        SELECT r.attempt_id, r.identity_json, r.state
        FROM campaign_remote_runs r JOIN campaign_attempts t
          ON t.workspace_id = r.workspace_id AND t.attempt_id = r.attempt_id
        JOIN campaign_actions a
          ON a.workspace_id = t.workspace_id AND a.action_id = t.action_id
        WHERE a.workspace_id = ? AND a.campaign_id = ?
          AND t.status NOT IN ('completed','failed','force_stopped','cancelled')
        ORDER BY r.updated_at DESC, r.attempt_id LIMIT 2
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    gate_row = connection.execute(
        """
        SELECT decision_id, decision_json FROM campaign_gate_decisions
        WHERE workspace_id = ? AND campaign_id = ?
        ORDER BY created_at DESC, decision_id DESC LIMIT 1
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    evaluation_row = connection.execute(
        """
        SELECT evaluation_id FROM campaign_evaluations
        WHERE workspace_id = ? AND campaign_id = ?
        ORDER BY created_at DESC, evaluation_id DESC LIMIT 1
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    latest_gate = (
        {"decision_id": gate_row["decision_id"], **json.loads(gate_row["decision_json"])}
        if gate_row
        else None
    )
    protected_row = connection.execute(
        """
        SELECT lease_state, candidate_lock_digest, result_json
        FROM campaign_protected_epochs
        WHERE workspace_id = ? AND campaign_id = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    protected_gate_passed = False
    candidate_digest = latest_gate.get("candidate_digest") if latest_gate else None
    if protected_row is not None and protected_row["result_json"] and candidate_digest:
        protected_result = json.loads(protected_row["result_json"])
        expected_lock = canonical_hash(
            {
                "campaign_id": campaign_id,
                "candidate_digest": candidate_digest,
                "manifest_revision": campaign.manifest_revision,
            }
        )
        protected_gate_passed = bool(
            protected_row["lease_state"] == "completed"
            and protected_row["candidate_lock_digest"] == expected_lock
            and protected_result.get("candidate_digest") == candidate_digest
            and protected_result.get("passed") is True
        )
    autoresearch_spec = baseline_outcome = candidate_outcome = None
    if _table_exists(connection, "autoresearch_campaign_specs"):
        spec_row = connection.execute(
            """
            SELECT spec_json FROM autoresearch_campaign_specs
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        autoresearch_spec = json.loads(spec_row["spec_json"]) if spec_row else None
        baseline_row = connection.execute(
            """
            SELECT result_json, decision_json FROM autoresearch_results
            WHERE workspace_id = ? AND campaign_id = ?
              AND json_extract(decision_json, '$.decision') = 'baseline'
            ORDER BY created_at DESC, result_id DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        candidate_row = connection.execute(
            """
            SELECT result_json, decision_json FROM autoresearch_results
            WHERE workspace_id = ? AND campaign_id = ?
              AND json_extract(decision_json, '$.decision') <> 'baseline'
            ORDER BY created_at DESC, result_id DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        baseline_outcome = (
            {"result": json.loads(baseline_row["result_json"]), "decision": json.loads(baseline_row["decision_json"])}
            if baseline_row
            else None
        )
        candidate_outcome = (
            {"result": json.loads(candidate_row["result_json"]), "decision": json.loads(candidate_row["decision_json"])}
            if candidate_row
            else None
        )
    return ControlRoomDurableProjection(
        campaign=campaign,
        manifest=manifest,
        latest_event_cursor=int(event_row["latest_cursor"]),
        collection_counts={
            "events": int(event_row["item_count"]),
            "proposals": int(count_row["proposals"]),
            "studies": int(count_row["studies"]),
            "attempts": int(count_row["attempts"]),
            "artifacts": int(count_row["artifacts"]),
            "comparisons": int(count_row["comparisons"]),
            "human_work": 0,
        },
        budget_totals={
            str(item["unit"]): (
                float(item["reserved"]),
                float(item["settled"]),
                float(item["limit_delta"]),
            )
            for item in budget_rows
        },
        active_studies=tuple(dict(item) for item in active_studies),
        active_actions=tuple(dict(item) for item in active_actions),
        active_attempts=tuple(dict(item) for item in active_attempts),
        remote_runs=tuple(dict(item) for item in remote_runs),
        latest_gate=latest_gate,
        latest_evaluation_id=str(evaluation_row["evaluation_id"]) if evaluation_row else None,
        autoresearch_spec=autoresearch_spec,
        baseline_outcome=baseline_outcome,
        candidate_outcome=candidate_outcome,
        protected_gate_passed=protected_gate_passed,
    )


def _status_summary(
    connection: sqlite3.Connection,
    *,
    table: str,
    workspace_id: str,
    campaign_id: str,
) -> ControlRoomCollectionSummaryV1:
    allowed_tables = {
        "campaign_proposals",
        "campaign_studies",
        "campaign_actions",
    }
    if table not in allowed_tables:
        raise ValueError("unsupported control-room summary table")
    rows = connection.execute(
        f"""
        SELECT status, COUNT(*) AS item_count
        FROM {table}
        WHERE workspace_id = ? AND campaign_id = ?
        GROUP BY status
        ORDER BY status
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    counts = tuple(
        ControlRoomStatusCountV1(status=row["status"], count=int(row["item_count"]))
        for row in rows
    )
    return ControlRoomCollectionSummaryV1(
        total=sum(item.count for item in counts),
        by_status=counts,
    )


def _attempt_summary(
    connection: sqlite3.Connection,
    *,
    workspace_id: str,
    campaign_id: str,
) -> ControlRoomCollectionSummaryV1:
    rows = connection.execute(
        """
        SELECT attempts.status, COUNT(*) AS item_count
        FROM campaign_attempts AS attempts
        JOIN campaign_actions AS actions
          ON actions.workspace_id = attempts.workspace_id
         AND actions.action_id = attempts.action_id
        WHERE actions.workspace_id = ? AND actions.campaign_id = ?
        GROUP BY attempts.status
        ORDER BY attempts.status
        """,
        (workspace_id, campaign_id),
    ).fetchall()
    counts = tuple(
        ControlRoomStatusCountV1(status=row["status"], count=int(row["item_count"]))
        for row in rows
    )
    return ControlRoomCollectionSummaryV1(
        total=sum(item.count for item in counts),
        by_status=counts,
    )


def read_control_room_state(
    connection: sqlite3.Connection,
    workspace_id: str,
    campaign_id: str,
) -> CampaignControlRoomStateV1:
    """Read the safe durable projection through the caller's open transaction."""

    row = connection.execute(
        "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
        (workspace_id, campaign_id),
    ).fetchone()
    if row is None:
        raise RecordNotFoundError("campaign not found")
    cursor_row = connection.execute(
        """
        SELECT COALESCE(MAX(cursor), 0) AS latest_event_cursor
        FROM campaign_events
        WHERE workspace_id = ? AND campaign_id = ?
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    artifact_row = connection.execute(
        """
        SELECT COUNT(*) AS total,
               COALESCE(SUM(sealed), 0) AS sealed,
               COALESCE(SUM(valid), 0) AS valid
        FROM campaign_artifacts
        WHERE workspace_id = ? AND campaign_id = ?
        """,
        (workspace_id, campaign_id),
    ).fetchone()
    return CampaignControlRoomStateV1(
        campaign=ControlRoomCampaignV1(
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            title=row["title"],
            kind=row["kind"],
            objective=row["objective"],
            manifest_revision=row["manifest_revision"],
            status=row["status"],
            active_study_id=row["active_study_id"],
            active_action_id=row["active_action_id"],
            stop_reason=row["stop_reason"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        ),
        latest_event_cursor=int(cursor_row["latest_event_cursor"]),
        proposals=_status_summary(
            connection,
            table="campaign_proposals",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
        ),
        studies=_status_summary(
            connection,
            table="campaign_studies",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
        ),
        actions=_status_summary(
            connection,
            table="campaign_actions",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
        ),
        attempts=_attempt_summary(
            connection,
            workspace_id=workspace_id,
            campaign_id=campaign_id,
        ),
        artifacts=ControlRoomArtifactSummaryV1(
            total=int(artifact_row["total"]),
            sealed=int(artifact_row["sealed"]),
            valid=int(artifact_row["valid"]),
        ),
        state_observed_at=utc_now(),
    )


def _safe_reference(value: str | None) -> str | None:
    if value is None:
        return None
    if value.replace("-", "").replace("_", "").replace(".", "").isalnum():
        return value
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _binding(
    binding_id: object,
    *,
    digest: str | None = None,
    label: str | None = None,
) -> SafeBindingIdentityV1 | None:
    if not isinstance(binding_id, str) or not binding_id:
        return None
    return SafeBindingIdentityV1(
        binding_id=binding_id,
        immutable_digest=digest,
        display_label=label or binding_id,
    )


def _invariant_failed(durable: ControlRoomDurableProjection) -> bool:
    if any(
        len(rows) > 1
        for rows in (durable.active_studies, durable.active_actions, durable.active_attempts)
    ):
        return True
    campaign = durable.campaign
    study = durable.active_studies[0] if durable.active_studies else None
    action = durable.active_actions[0] if durable.active_actions else None
    attempt = durable.active_attempts[0] if durable.active_attempts else None
    if campaign.active_study_id != (study["study_id"] if study else None):
        return True
    if campaign.active_action_id != (action["action_id"] if action else None):
        return True
    if action and study and action["study_id"] != study["study_id"]:
        return True
    if attempt and action and attempt["action_id"] != action["action_id"]:
        return True
    if study:
        plan = json.loads(study["stage_plan_json"])
        items = plan.get("items", []) if isinstance(plan, dict) else []
        if int(study["current_stage_index"]) >= len(items):
            return True
        if action and int(action["stage_index"]) != int(study["current_stage_index"]):
            return True
    return False


def _decision_actions(
    durable: ControlRoomDurableProjection,
    readiness: ReadinessSummaryV1,
    controller: ControllerObservationV1,
    principal: ActorPrincipal,
    *,
    invariant_failed: bool,
    promotion_eligible: bool,
) -> tuple[DecisionActionV1, ...]:
    if invariant_failed:
        return ()
    trigger_actions = {
        CampaignTrigger.START: ("start", "campaign.start", "lifecycle"),
        CampaignTrigger.PAUSE: ("pause", "campaign.pause", "lifecycle"),
        CampaignTrigger.RESUME: ("resume", "campaign.resume", "lifecycle"),
        CampaignTrigger.CANCEL: ("cancel", "campaign.cancel", "lifecycle"),
        CampaignTrigger.CONCLUDE: ("conclude", "campaign.complete", "lifecycle"),
        CampaignTrigger.PROMOTION_COMMITTED: (
            "promote",
            "promotion.decide",
            "privileged",
        ),
    }
    actions: list[DecisionActionV1] = []
    for trigger in allowed_triggers(durable.campaign.status):
        if trigger not in trigger_actions:
            continue
        action, capability_value, freshness = trigger_actions[trigger]
        capability = Capability(capability_value)
        if capability not in principal.capabilities:
            continue
        if trigger == CampaignTrigger.START and (
            not readiness.launch_ready or controller.state != "online"
        ):
            continue
        if trigger == CampaignTrigger.PROMOTION_COMMITTED and (
            not promotion_eligible or controller.state != "online"
        ):
            continue
        actions.append(
            DecisionActionV1(
                action=action,
                capability=capability,
                freshness_class=freshness,
                requires_human_work=False,
            )
        )
    return tuple(sorted(actions, key=lambda item: item.action))


def _blocker(
    durable: ControlRoomDurableProjection,
    readiness: ReadinessSummaryV1,
    controller: ControllerObservationV1,
    *,
    invariant_failed: bool,
    budget_blocked: bool,
) -> DecisionBlockerV1 | None:
    codes: list[str] = []
    if invariant_failed:
        codes.append("campaign_projection_invariant_failed")
    process_unknown = any(row["status"] == "unknown" for row in durable.active_attempts) or any(
        row["state"] == "unknown" for row in durable.remote_runs
    )
    if process_unknown:
        codes.append("campaign_process_identity_unknown")
    readiness_codes = [
        code
        for code in readiness.blocking_codes
        if "compute" not in code and "capacity" not in code
    ]
    codes.extend(readiness_codes)
    compute_codes = [
        code for code in readiness.blocking_codes if "compute" in code or "capacity" in code
    ]
    codes.extend(compute_codes)
    if budget_blocked:
        codes.append("campaign_budget_exceeded")
    if controller.state != "online":
        codes.append("controller_stale" if controller.state == "stale" else "controller_offline")
    codes = list(dict.fromkeys(codes))
    if not codes:
        return None
    summaries = {
        "campaign_projection_invariant_failed": "Campaign records violate projection invariants.",
        "campaign_process_identity_unknown": "The active process identity must be reconciled.",
        "campaign_budget_exceeded": "The campaign budget has no remaining capacity.",
        "controller_stale": "The resident controller observation is stale.",
        "controller_offline": "The resident controller is offline.",
        "campaign_not_ready": "Campaign readiness checks have not passed.",
    }
    return DecisionBlockerV1(
        code=codes[0],
        summary=summaries.get(codes[0], "A campaign prerequisite has not been satisfied."),
        evidence_ids=(),
        secondary_codes=tuple(codes[1:]),
    )


def _phase(
    phase_id: str,
    state: str,
    *,
    execution_owner: str = "none",
    attention_owner: str = "none",
    blocker: DecisionBlockerV1 | None = None,
    evidence_count: int = 0,
    actions: tuple[str, ...] = (),
) -> JourneyPhaseSummaryV1:
    return JourneyPhaseSummaryV1(
        phase_id=phase_id,
        state=state,
        execution_owner=execution_owner,
        attention_owner=attention_owner,
        primary_blocker=blocker,
        evidence_count=evidence_count,
        next_action_ids=actions,
    )


def build_control_room_snapshot(
    durable: ControlRoomDurableProjection,
    controller: ControllerObservationV1,
    readiness: ReadinessSummaryV1,
    *,
    principal: ActorPrincipal,
    snapshot_at: datetime,
) -> CampaignControlRoomSnapshotV1:
    """Purely build the allowlisted principal-specific public projection."""

    campaign = durable.campaign
    manifest = durable.manifest.manifest
    invariant_failed = _invariant_failed(durable)
    budget_resources: list[BudgetResourceSummaryV1] = []
    for unit, base_limit in sorted(manifest.budget_limits.items()):
        reserved, settled, limit_delta = durable.budget_totals.get(unit, (0.0, 0.0, 0.0))
        limit = float(base_limit) + limit_delta
        remaining = limit - reserved - settled
        budget_resources.append(
            BudgetResourceSummaryV1(
                unit=unit,
                limit=limit,
                reserved=reserved,
                settled=settled,
                remaining=remaining,
                blocked=remaining < 0,
                blocker_code="campaign_budget_exceeded" if remaining < 0 else None,
            )
        )
    budget_blocked = any(resource.blocked for resource in budget_resources)
    gate = durable.latest_gate or {}
    promotion_gate = evaluate_promotion_gate(
        active_action_id=campaign.active_action_id,
        comparison_verdict=gate.get("verdict"),
        candidate_digest=gate.get("candidate_digest")
        or campaign.best_development_candidate_ref,
        protected_required=bool(manifest.protected_artifact_refs),
        protected_passed=durable.protected_gate_passed,
        human_work_complete=True,
    )
    promotion_eligible = bool(
        promotion_gate.eligible
        and CampaignTrigger.PROMOTION_COMMITTED in allowed_triggers(campaign.status)
    )
    blocker = _blocker(
        durable,
        readiness,
        controller,
        invariant_failed=invariant_failed,
        budget_blocked=budget_blocked,
    )
    actions = _decision_actions(
        durable,
        readiness,
        controller,
        principal,
        invariant_failed=invariant_failed,
        promotion_eligible=promotion_eligible,
    )
    execution_owner = "bashgym" if (
        durable.active_actions or durable.active_attempts
    ) else "none"
    attention_owner = "bashgym" if blocker is not None else "none"
    action_ids = tuple(action.action for action in actions)

    setup_state = "ready" if readiness.launch_ready else (
        "blocked" if readiness.blocking_codes else "not_started"
    )
    has_autoresearch = durable.autoresearch_spec is not None
    baseline_complete = bool(
        durable.baseline_outcome
        and durable.baseline_outcome["decision"].get("decision") == "baseline"
        and durable.baseline_outcome["result"].get("outcome") == "completed"
    )
    active = bool(durable.active_studies or durable.active_actions or durable.active_attempts)
    terminal = campaign.status in {
        CampaignStatus.COMPLETED,
        CampaignStatus.EXHAUSTED,
        CampaignStatus.FAILED,
        CampaignStatus.CANCELLED,
    }
    if not has_autoresearch:
        baseline_state = experiments_state = "skipped"
    else:
        baseline_crashed = bool(
            durable.baseline_outcome
            and durable.baseline_outcome["result"].get("outcome") == "crashed"
        )
        baseline_state = (
            "complete"
            if baseline_complete
            else "active"
            if active
            else "failed"
            if baseline_crashed or (terminal and campaign.status != CampaignStatus.COMPLETED)
            else "ready"
            if setup_state == "ready"
            else "not_started"
        )
        experiments_state = (
            "not_started"
            if not baseline_complete
            else "active"
            if active
            else "failed"
            if campaign.status == CampaignStatus.FAILED
            else "complete"
            if terminal or durable.latest_gate is not None
            else "ready"
        )
    decision_state = (
        "complete"
        if campaign.status == CampaignStatus.COMPLETED or campaign.champion_ref is not None
        else "failed"
        if campaign.status == CampaignStatus.FAILED
        else "blocked"
        if durable.latest_gate is not None
        else "not_started"
    )
    journey = (
        _phase(
            "setup",
            setup_state,
            attention_owner=attention_owner if setup_state == "blocked" else "none",
            blocker=blocker if setup_state == "blocked" else None,
            actions=tuple(item for item in action_ids if item == "start"),
        ),
        _phase(
            "baseline",
            baseline_state,
            execution_owner=execution_owner if baseline_state == "active" else "none",
            evidence_count=1 if durable.baseline_outcome else 0,
        ),
        _phase(
            "experiments",
            experiments_state,
            execution_owner=execution_owner if experiments_state == "active" else "none",
            attention_owner=attention_owner if invariant_failed else "none",
            blocker=blocker if invariant_failed else None,
            evidence_count=durable.collection_counts["attempts"]
            + durable.collection_counts["comparisons"],
        ),
        _phase("human_review", "not_started"),
        _phase(
            "decision",
            decision_state,
            evidence_count=durable.collection_counts["comparisons"],
        ),
    )

    active_work = None
    if active:
        study = durable.active_studies[0] if durable.active_studies else None
        action = durable.active_actions[0] if durable.active_actions else None
        attempt = durable.active_attempts[0] if durable.active_attempts else None
        executor_type = None
        if attempt:
            executor = json.loads(attempt["executor_json"])
            candidate_executor = executor.get("executor_kind")
            if candidate_executor in {"fake", "ssh_remote", "development_evaluation"}:
                executor_type = candidate_executor
        process_identity = None
        if durable.remote_runs:
            remote = durable.remote_runs[0]
            identity = json.loads(remote["identity_json"])
            remote_state = remote["state"]
            process_identity = OpaqueProcessIdentityV1(
                run_id=identity["run_id"],
                compute_profile_id=identity["compute_profile_id"],
                state="running" if remote_state == "paused" else remote_state,
            )
        active_work = ActiveWorkSummaryV1(
            study_id=str(study["study_id"]) if study else None,
            proposal_id=str(study["proposal_id"]) if study else None,
            action_id=str(action["action_id"]) if action else None,
            attempt_id=str(attempt["attempt_id"]) if attempt else None,
            stage=(action["stage_kind"] if action else attempt["stage_kind"] if attempt else None),
            hypothesis_summary=None,
            primary_variable_summary=None,
            controlled_variable_summary=(),
            progress_fraction=None,
            eta_seconds=None,
            executor_type=executor_type,
            process_identity=process_identity,
        )

    candidate_ref = _safe_reference(
        campaign.best_development_candidate_ref or gate.get("candidate_digest")
    )
    candidate = None
    if candidate_ref is not None:
        outcome = durable.candidate_outcome["result"] if durable.candidate_outcome else {}
        verdict = gate.get("verdict")
        candidate = CandidateSummaryV1(
            candidate_ref=candidate_ref,
            source_attempt_ids=tuple(outcome.get("attempt_ids", ())),
            source_artifact_ids=tuple(outcome.get("evidence_references", ())),
            latest_comparable_evaluation_id=durable.latest_evaluation_id,
            comparison_verdict=verdict if verdict in {"passed", "failed", "insufficient_evidence"} else None,
            gate_state=(
                "passed" if verdict == "passed" else "failed" if verdict == "failed" else "blocked" if gate else "not_evaluated"
            ),
        )
    champion = None
    champion_ref = _safe_reference(campaign.champion_ref)
    if champion_ref is not None:
        champion = CandidateSummaryV1(
            candidate_ref=champion_ref,
            source_attempt_ids=(),
            source_artifact_ids=(),
            latest_comparable_evaluation_id=durable.latest_evaluation_id,
            comparison_verdict="passed",
            gate_state="promoted",
        )

    metrics: tuple[MetricDescriptorV1, ...] = ()
    if durable.autoresearch_spec:
        spec = durable.autoresearch_spec
        metric_id = str(spec["primary_metric"])
        stop_rules = spec["stop_rules"]
        evaluation_suite_id = spec.get("evaluation_suite_id")
        dataset_id = manifest.evaluation_plan.get("dataset_binding_id")
        metrics = (
            MetricDescriptorV1(
                metric_id=metric_id,
                display_name=metric_id.replace("_", " ").title(),
                unit=None,
                direction=spec["metric_direction"],
                target=stop_rules.get("target_metric"),
                tolerance=None,
                evaluator_revision=None,
                sample_count=None,
                uncertainty_method=None,
                comparability_key=canonical_hash(
                    {
                        "metric_id": metric_id,
                        "direction": spec["metric_direction"],
                        "manifest_hash": durable.manifest.manifest_hash,
                        "dataset_binding_id": dataset_id,
                        "evaluation_suite_id": evaluation_suite_id,
                        "target_contract_key": campaign.target_model.target_contract_key,
                    }
                ),
            ),
        )

    def cursor(name: str) -> CollectionCursorV1:
        return CollectionCursorV1(
            count=durable.collection_counts[name], next_cursor=None, has_more=False
        )

    evaluation_plan = manifest.evaluation_plan
    model_digest = canonical_hash(campaign.target_model.model_dump(mode="json"))
    data_id = evaluation_plan.get("dataset_binding_id") or (
        manifest.approved_data_scopes[0] if manifest.approved_data_scopes else None
    )
    evaluator_id = evaluation_plan.get("evaluation_suite_id")
    source_id = evaluation_plan.get("source_repository_binding_id")
    recovery = ["inspect"]
    if invariant_failed:
        recovery.extend(("reconcile_controller", "reconcile_attempt"))
    else:
        if controller.state != "online":
            recovery.append("reconcile_controller")
        if blocker and blocker.code == "campaign_process_identity_unknown":
            recovery.append("reconcile_attempt")

    return CampaignControlRoomSnapshotV1(
        workspace_id=campaign.workspace_id,
        campaign_id=campaign.campaign_id,
        aggregate_version=campaign.version,
        manifest_revision=campaign.manifest_revision,
        authorization_revision=principal.authorization_revision,
        snapshot_at=snapshot_at,
        latest_event_cursor=durable.latest_event_cursor,
        campaign=CampaignSummaryV1(
            campaign_id=campaign.campaign_id,
            title=campaign.title,
            objective=campaign.objective,
            kind=campaign.kind,
            status=campaign.status,
            aggregate_version=campaign.version,
            manifest_revision=campaign.manifest_revision,
            active_study_id=campaign.active_study_id,
            active_action_id=campaign.active_action_id,
            champion_ref=champion_ref,
            stop_reason=campaign.stop_reason,
        ),
        controller=controller,
        readiness=readiness,
        bindings=BindingSummaryV1(
            model=_binding(
                campaign.target_model.target_contract_key,
                digest=model_digest,
                label=campaign.target_model.task,
            ),
            data=_binding(data_id),
            evaluator=_binding(evaluator_id),
            source=_binding(source_id),
            compute=_binding(manifest.compute_profile_id),
        ),
        journey=journey,
        active_work=active_work,
        champion=champion,
        candidate=candidate,
        metrics=metrics,
        budget=BudgetSummaryV1(resources=tuple(budget_resources), blocked=budget_blocked),
        human_work=HumanWorkSummaryV1(blocking_count=0, open_count=0, newest=()),
        agents=(),
        collections=CollectionSummaryV1(
            events=cursor("events"),
            proposals=cursor("proposals"),
            studies=cursor("studies"),
            attempts=cursor("attempts"),
            artifacts=cursor("artifacts"),
            comparisons=cursor("comparisons"),
            human_work=cursor("human_work"),
        ),
        decision_surface=DecisionSurfaceV1(
            execution_owner=execution_owner,
            attention_owner=attention_owner,
            blocker=blocker,
            next_actions=actions,
            recovery_actions=tuple(dict.fromkeys(recovery)),
            promotion_eligible=promotion_eligible,
        ),
    )


def principal_control_room_etag(
    snapshot: CampaignControlRoomSnapshotV1,
    principal: ActorPrincipal,
) -> str:
    """Return a private validator that cannot be shared across principals."""

    digest = canonical_hash(
        {
            "workspace_id": snapshot.workspace_id,
            "campaign_id": snapshot.campaign_id,
            "aggregate_version": snapshot.aggregate_version,
            "latest_event_cursor": snapshot.latest_event_cursor,
            "controller_observation_version": snapshot.controller.controller_observation_version,
            "principal": {
                "actor_id": principal.actor_id,
                "credential_id": principal.credential_id,
            },
            "authorization_revision": snapshot.authorization_revision,
        }
    )
    return f'"{digest}"'


def if_none_match_matches(value: str | None, etag: str) -> bool:
    """Accept standard strong/weak validators and wildcard compatibility."""

    if value is None:
        return False
    for candidate in value.split(","):
        normalized = candidate.strip()
        if normalized == "*":
            return True
        if normalized.startswith("W/"):
            normalized = normalized[2:].strip()
        if normalized == etag:
            return True
    return False


__all__ = [
    "ControlRoomDurableProjection",
    "build_control_room_snapshot",
    "if_none_match_matches",
    "principal_control_room_etag",
    "read_control_room_projection",
    "read_control_room_state",
]
