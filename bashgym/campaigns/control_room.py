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
    HumanWorkItemSummaryV1,
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
from bashgym.campaigns.transitions import (
    allowed_triggers,
    evaluate_promotion_gate,
    human_promotion_authorized,
)


@dataclass(frozen=True)
class CandidateProvenance:
    """Candidate-keyed IDs proven against campaign-owned rows."""

    candidate_digest: str
    source_attempt_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    latest_comparable_evaluation_id: str | None


@dataclass(frozen=True)
class ChampionProvenance:
    """Persisted champion claim correlated to its candidate and gate decision."""

    candidate_digest: str
    decision_id: str
    comparison_verdict: str | None
    override: bool


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
    autoresearch_spec: dict[str, Any] | None
    baseline_outcome: dict[str, Any] | None
    candidate_provenance: CandidateProvenance | None
    champion_provenance: ChampionProvenance | None
    projection_invariant_codes: tuple[str, ...]
    protected_gate_passed: bool
    human_work: tuple[dict[str, Any], ...] = ()
    human_work_open_count: int = 0
    human_work_blocking_count: int = 0
    human_promotion_state: str | None = None
    has_current_blocking_human_work: bool = False
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
    projection_invariant_codes: list[str] = []
    if len(manifest.manifest.budget_limits) > 64:
        projection_invariant_codes.append("campaign_projection_budget_resources_exceeded")
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
    latest_gate = None
    if gate_row:
        try:
            gate_payload = json.loads(gate_row["decision_json"])
        except json.JSONDecodeError:
            gate_payload = None
        if isinstance(gate_payload, dict):
            latest_gate = {"decision_id": gate_row["decision_id"], **gate_payload}
        else:
            projection_invariant_codes.append("campaign_projection_gate_decision_malformed")
    champion_provenance = None
    if campaign.champion_ref and _table_exists(connection, "campaign_champions"):
        reference_body = campaign.champion_ref.removeprefix("champion:")
        parts = reference_body.rsplit(":", 2)
        if (
            not campaign.champion_ref.startswith("champion:")
            or len(parts) != 3
            or not all(parts)
            or not parts[1].isdigit()
        ):
            projection_invariant_codes.append("campaign_projection_champion_ref_malformed")
        else:
            target_key, revision_text, referenced_candidate = parts
            champion_row = connection.execute(
                """
                SELECT champion_json FROM campaign_champions
                WHERE workspace_id = ? AND target_contract_key = ? AND revision = ?
                """,
                (workspace_id, target_key, int(revision_text)),
            ).fetchone()
            if champion_row is None or target_key != campaign.target_model.target_contract_key:
                projection_invariant_codes.append("campaign_projection_champion_claim_missing")
            else:
                champion_payload = json.loads(champion_row["champion_json"])
                decision_id = champion_payload.get("development_decision_id")
                claimed_candidate = champion_payload.get("candidate_digest")
                if (
                    champion_payload.get("campaign_id") != campaign_id
                    or not isinstance(claimed_candidate, str)
                    or claimed_candidate != referenced_candidate
                    or not isinstance(decision_id, str)
                ):
                    projection_invariant_codes.append("campaign_projection_champion_claim_mismatch")
                else:
                    champion_decision_row = connection.execute(
                        """
                        SELECT decision_json FROM campaign_gate_decisions
                        WHERE workspace_id = ? AND campaign_id = ? AND decision_id = ?
                        """,
                        (workspace_id, campaign_id, decision_id),
                    ).fetchone()
                    if champion_decision_row is None:
                        projection_invariant_codes.append(
                            "campaign_projection_champion_decision_missing"
                        )
                    else:
                        try:
                            champion_decision = json.loads(champion_decision_row["decision_json"])
                        except json.JSONDecodeError:
                            champion_decision = None
                        candidate_verdict = (
                            champion_decision.get("verdict")
                            if isinstance(champion_decision, dict)
                            else None
                        )
                        if (
                            not isinstance(champion_decision, dict)
                            or champion_decision.get("candidate_digest") != claimed_candidate
                            or candidate_verdict
                            not in {"passed", "failed", "insufficient_evidence"}
                        ):
                            projection_invariant_codes.append(
                                "campaign_projection_champion_decision_mismatch"
                            )
                        else:
                            champion_provenance = ChampionProvenance(
                                candidate_digest=claimed_candidate,
                                decision_id=decision_id,
                                comparison_verdict=candidate_verdict,
                                override=bool(champion_payload.get("override")),
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
    autoresearch_spec = baseline_outcome = None
    candidate_provenance = None
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
        baseline_outcome = (
            {
                "result": json.loads(baseline_row["result_json"]),
                "decision": json.loads(baseline_row["decision_json"]),
            }
            if baseline_row
            else None
        )
        gate_candidate = latest_gate.get("candidate_digest") if latest_gate else None
        selected_candidate = (
            champion_provenance.candidate_digest
            if champion_provenance
            else campaign.best_development_candidate_ref or gate_candidate
        )
        if (
            campaign.best_development_candidate_ref
            and gate_candidate
            and campaign.best_development_candidate_ref != gate_candidate
        ):
            projection_invariant_codes.append("campaign_projection_candidate_identity_mismatch")
        if selected_candidate:
            candidate_rows = connection.execute(
                """
                SELECT r.result_json, r.decision_json, s.study_id
                FROM autoresearch_results r
                JOIN campaign_studies s
                  ON s.workspace_id = r.workspace_id
                 AND s.campaign_id = r.campaign_id
                 AND s.proposal_id = r.proposal_id
                WHERE r.workspace_id = ? AND r.campaign_id = ?
                  AND s.candidate_digest = ?
                  AND json_valid(r.result_json)
                  AND json_valid(r.decision_json)
                ORDER BY r.created_at DESC, r.result_id DESC LIMIT 2
                """,
                (workspace_id, campaign_id, selected_candidate),
            ).fetchall()
            if len(candidate_rows) > 1:
                projection_invariant_codes.append("campaign_projection_candidate_outcome_ambiguous")
            if candidate_rows:
                candidate_row = candidate_rows[0]
                result_payload = json.loads(candidate_row["result_json"])
                raw_attempts = result_payload.get("attempt_ids", [])
                raw_artifacts = result_payload.get("evidence_references", [])
                if not isinstance(raw_attempts, list) or not all(
                    isinstance(item, str) for item in raw_attempts
                ):
                    raw_attempts = []
                    projection_invariant_codes.append(
                        "campaign_projection_candidate_attempts_malformed"
                    )
                if not isinstance(raw_artifacts, list) or not all(
                    isinstance(item, str) for item in raw_artifacts
                ):
                    raw_artifacts = []
                    projection_invariant_codes.append(
                        "campaign_projection_candidate_artifacts_malformed"
                    )
                if len(raw_attempts) > 100 or len(raw_artifacts) > 100:
                    projection_invariant_codes.append(
                        "campaign_projection_candidate_references_exceeded"
                    )
                study_id = str(candidate_row["study_id"])
                owned_attempt_rows = connection.execute(
                    """
                    SELECT t.attempt_id FROM campaign_attempts t
                    JOIN campaign_actions a
                      ON a.workspace_id = t.workspace_id AND a.action_id = t.action_id
                    WHERE a.workspace_id = ? AND a.campaign_id = ? AND a.study_id = ?
                    ORDER BY t.attempt_id LIMIT 101
                    """,
                    (workspace_id, campaign_id, study_id),
                ).fetchall()
                owned_artifact_rows = connection.execute(
                    """
                    SELECT ar.artifact_id FROM campaign_artifacts ar
                    JOIN campaign_actions a
                      ON a.workspace_id = ar.workspace_id
                     AND a.action_id = ar.producer_action_id
                    WHERE ar.workspace_id = ? AND ar.campaign_id = ? AND a.study_id = ?
                    ORDER BY ar.artifact_id LIMIT 101
                    """,
                    (workspace_id, campaign_id, study_id),
                ).fetchall()
                if len(owned_attempt_rows) > 100 or len(owned_artifact_rows) > 100:
                    projection_invariant_codes.append(
                        "campaign_projection_candidate_owned_rows_exceeded"
                    )
                owned_attempts = {str(item["attempt_id"]) for item in owned_attempt_rows[:100]}
                owned_artifacts = {str(item["artifact_id"]) for item in owned_artifact_rows[:100]}
                candidate_evaluation_row = connection.execute(
                    """
                    SELECT evaluation_id FROM campaign_evaluations
                    WHERE workspace_id = ? AND campaign_id = ?
                      AND json_valid(evaluation_json)
                      AND json_extract(evaluation_json, '$.candidate_digest') = ?
                    ORDER BY created_at DESC, evaluation_id DESC LIMIT 1
                    """,
                    (workspace_id, campaign_id, selected_candidate),
                ).fetchone()
                candidate_provenance = CandidateProvenance(
                    candidate_digest=selected_candidate,
                    source_attempt_ids=tuple(
                        item for item in raw_attempts[:100] if item in owned_attempts
                    ),
                    source_artifact_ids=tuple(
                        item for item in raw_artifacts[:100] if item in owned_artifacts
                    ),
                    latest_comparable_evaluation_id=(
                        str(candidate_evaluation_row["evaluation_id"])
                        if candidate_evaluation_row
                        else None
                    ),
                )
    human_work_rows: tuple[dict[str, Any], ...] = ()
    human_work_count = 0
    human_work_open_count = 0
    human_work_blocking_count = 0
    human_promotion_state: str | None = None
    has_current_blocking_human_work = False
    if _table_exists(connection, "campaign_human_work"):
        human_count_row = connection.execute(
            """
            SELECT COUNT(*) AS item_count,
                   SUM(CASE WHEN campaign_revision != ? OR state NOT IN ('submitted', 'replaced') THEN 1 ELSE 0 END) AS open_count,
                   SUM(CASE WHEN blocking = 1 AND (campaign_revision != ? OR state NOT IN ('submitted', 'replaced')) THEN 1 ELSE 0 END) AS blocking_count
            FROM campaign_human_work
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (
                campaign.manifest_revision,
                campaign.manifest_revision,
                workspace_id,
                campaign_id,
            ),
        ).fetchone()
        human_work_count = int(human_count_row["item_count"])
        human_work_open_count = int(human_count_row["open_count"] or 0)
        human_work_blocking_count = int(human_count_row["blocking_count"] or 0)
        human_rows = connection.execute(
            """
            SELECT work_id, campaign_revision, item_version, state, blocking,
                   claimed_by_actor_id, lease_expires_at, updated_at
            FROM campaign_human_work
            WHERE workspace_id = ? AND campaign_id = ?
            ORDER BY (campaign_revision = ?) DESC, updated_at DESC, work_id ASC
            LIMIT 10
            """,
            (workspace_id, campaign_id, campaign.manifest_revision),
        ).fetchall()
        human_work_rows = tuple(dict(item) for item in human_rows)
        current_blocking_row = connection.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM campaign_human_work
                WHERE workspace_id = ? AND campaign_id = ?
                  AND campaign_revision = ? AND blocking = 1 AND state != 'replaced'
            ) AS present
            """,
            (workspace_id, campaign_id, campaign.manifest_revision),
        ).fetchone()
        has_current_blocking_human_work = bool(current_blocking_row["present"])
    if _table_exists(connection, "campaign_human_promotions"):
        human_promotion_row = connection.execute(
            """
            SELECT state FROM campaign_human_promotions
            WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
            """,
            (workspace_id, campaign_id, campaign.manifest_revision),
        ).fetchone()
        human_promotion_state = (
            str(human_promotion_row["state"]) if human_promotion_row is not None else None
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
            "human_work": human_work_count,
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
        autoresearch_spec=autoresearch_spec,
        baseline_outcome=baseline_outcome,
        candidate_provenance=candidate_provenance,
        champion_provenance=champion_provenance,
        projection_invariant_codes=tuple(dict.fromkeys(projection_invariant_codes)),
        protected_gate_passed=protected_gate_passed,
        human_work=human_work_rows,
        human_work_open_count=human_work_open_count,
        human_work_blocking_count=human_work_blocking_count,
        human_promotion_state=human_promotion_state,
        has_current_blocking_human_work=has_current_blocking_human_work,
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
        ControlRoomStatusCountV1(status=row["status"], count=int(row["item_count"])) for row in rows
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
        ControlRoomStatusCountV1(status=row["status"], count=int(row["item_count"])) for row in rows
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
    if durable.projection_invariant_codes:
        return True
    if len(durable.manifest.manifest.budget_limits) > 64:
        return True
    if durable.candidate_provenance and (
        len(durable.candidate_provenance.source_attempt_ids) > 100
        or len(durable.candidate_provenance.source_artifact_ids) > 100
    ):
        return True
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
        if trigger == CampaignTrigger.PROMOTION_COMMITTED and (not promotion_eligible):
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
    promotion_blocking_codes: tuple[str, ...],
) -> DecisionBlockerV1 | None:
    codes: list[str] = []
    if invariant_failed:
        codes.append("campaign_projection_invariant_failed")
        codes.extend(durable.projection_invariant_codes)
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
    codes.extend(promotion_blocking_codes)
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
        "campaign_active_action_present": "Active campaign work must finish before promotion.",
        "campaign_development_gate_not_passed": "The development comparison gate has not passed.",
        "campaign_protected_gate_not_passed": "The protected evaluation gate has not passed.",
        "campaign_human_work_incomplete": "Required human review is incomplete.",
        "campaign_promotion_transition_unavailable": (
            "Promotion is unavailable from the current campaign state."
        ),
    }
    return DecisionBlockerV1(
        code=codes[0],
        summary=summaries.get(codes[0], "A campaign prerequisite has not been satisfied."),
        evidence_ids=(),
        secondary_codes=tuple(codes[1:]),
    )


def _promotion_blocker(blocking_codes: tuple[str, ...]) -> DecisionBlockerV1 | None:
    if not blocking_codes:
        return None
    summaries = {
        "campaign_active_action_present": "Active campaign work must finish before promotion.",
        "campaign_development_gate_not_passed": "The development comparison gate has not passed.",
        "campaign_protected_gate_not_passed": "The protected evaluation gate has not passed.",
        "campaign_human_work_incomplete": "Required human review is incomplete.",
        "campaign_promotion_transition_unavailable": (
            "Promotion is unavailable from the current campaign state."
        ),
    }
    return DecisionBlockerV1(
        code=blocking_codes[0],
        summary=summaries[blocking_codes[0]],
        evidence_ids=(),
        secondary_codes=blocking_codes[1:],
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


def _human_work_summary(
    durable: ControlRoomDurableProjection, snapshot_at: datetime
) -> HumanWorkSummaryV1:
    items: list[HumanWorkItemSummaryV1] = []
    current_revision = durable.campaign.manifest_revision
    for row in durable.human_work[:10]:
        raw_state = str(row["state"])
        due_at = (
            datetime.fromisoformat(str(row["lease_expires_at"]))
            if row.get("lease_expires_at")
            else None
        )
        if int(row["campaign_revision"]) != current_revision:
            status = "revision_requested"
        elif raw_state == "pending":
            status = "open"
        elif raw_state == "claimed":
            status = "expired" if due_at is not None and due_at <= snapshot_at else "claimed"
        elif raw_state == "submitted":
            status = "accepted"
        elif raw_state == "expired":
            status = "expired"
        else:
            status = "cancelled"
        completed = status == "accepted"
        items.append(
            HumanWorkItemSummaryV1(
                work_item_id=str(row["work_id"]),
                kind="blinded_sample_evaluation",
                status=status,
                blocking_scope=(
                    "comparison_and_promotion" if bool(row["blocking"]) else "advisory"
                ),
                assigned_actor_id=(
                    str(row["claimed_by_actor_id"])
                    if status == "claimed" and row.get("claimed_by_actor_id")
                    else None
                ),
                required_count=1,
                completed_count=1 if completed else 0,
                due_at=due_at,
            )
        )
    return HumanWorkSummaryV1(
        blocking_count=durable.human_work_blocking_count,
        open_count=durable.human_work_open_count,
        newest=tuple(items),
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
    for unit, base_limit in sorted(manifest.budget_limits.items())[:64]:
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
    human_work = _human_work_summary(durable, snapshot_at)
    gate = durable.latest_gate or {}
    promotion_gate = evaluate_promotion_gate(
        active_action_id=campaign.active_action_id,
        comparison_verdict=gate.get("verdict"),
        candidate_digest=gate.get("candidate_digest") or campaign.best_development_candidate_ref,
        protected_required=bool(manifest.protected_artifact_refs),
        protected_passed=durable.protected_gate_passed,
        human_work_complete=human_promotion_authorized(
            promotion_state=durable.human_promotion_state,
            has_current_blocking_work=durable.has_current_blocking_human_work,
        ),
    )
    promotion_transition_allowed = CampaignTrigger.PROMOTION_COMMITTED in allowed_triggers(
        campaign.status
    )
    promotion_eligible = bool(
        not invariant_failed and promotion_gate.eligible and promotion_transition_allowed
    )
    promotion_blocking_codes = promotion_gate.blocking_codes
    if promotion_gate.eligible and not promotion_transition_allowed:
        promotion_blocking_codes = (
            *promotion_blocking_codes,
            "campaign_promotion_transition_unavailable",
        )
    promotion_blocker = _promotion_blocker(promotion_blocking_codes)
    top_level_promotion_codes = promotion_blocking_codes if durable.latest_gate is not None else ()
    blocker = _blocker(
        durable,
        readiness,
        controller,
        invariant_failed=invariant_failed,
        budget_blocked=budget_blocked,
        promotion_blocking_codes=top_level_promotion_codes,
    )
    actions = _decision_actions(
        durable,
        readiness,
        controller,
        principal,
        invariant_failed=invariant_failed,
        promotion_eligible=promotion_eligible,
    )
    execution_owner = "bashgym" if (durable.active_actions or durable.active_attempts) else "none"
    attention_owner = "bashgym" if blocker is not None else "none"
    action_ids = tuple(action.action for action in actions)

    setup_state = (
        "ready"
        if readiness.launch_ready
        else ("blocked" if readiness.blocking_codes else "not_started")
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
            else (
                "active"
                if active
                else (
                    "failed"
                    if baseline_crashed
                    or (terminal and campaign.status != CampaignStatus.COMPLETED)
                    else "ready" if setup_state == "ready" else "not_started"
                )
            )
        )
        experiments_state = (
            "not_started"
            if not baseline_complete
            else (
                "active"
                if active
                else (
                    "failed"
                    if campaign.status == CampaignStatus.FAILED
                    else "complete" if terminal or durable.latest_gate is not None else "ready"
                )
            )
        )
    decision_state = (
        "blocked"
        if invariant_failed
        else (
            "complete"
            if campaign.status == CampaignStatus.COMPLETED or campaign.champion_ref is not None
            else (
                "failed"
                if campaign.status == CampaignStatus.FAILED
                else (
                    "ready"
                    if promotion_eligible
                    else "blocked" if durable.latest_gate is not None else "not_started"
                )
            )
        )
    )
    human_statuses = {item.status for item in human_work.newest}
    human_review_state = (
        "not_started"
        if not human_work.newest
        else (
            "complete"
            if human_work.open_count == 0
            else (
                "active"
                if "claimed" in human_statuses
                else "blocked" if human_statuses & {"expired", "revision_requested"} else "ready"
            )
        )
    )
    human_blocker = (
        _promotion_blocker(("campaign_human_work_incomplete",))
        if human_work.blocking_count > 0
        else None
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
        _phase(
            "human_review",
            human_review_state,
            attention_owner=(
                "human" if human_review_state in {"ready", "active", "blocked"} else "none"
            ),
            blocker=human_blocker if human_review_state == "blocked" else None,
            evidence_count=len([item for item in human_work.newest if item.status == "accepted"]),
        ),
        _phase(
            "decision",
            decision_state,
            attention_owner="bashgym" if decision_state == "blocked" else "none",
            blocker=(
                blocker
                if invariant_failed
                else promotion_blocker if decision_state == "blocked" else None
            ),
            evidence_count=durable.collection_counts["comparisons"],
            actions=tuple(item for item in action_ids if item == "promote"),
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
            candidate_executor = executor.get("kind", executor.get("executor_kind"))
            if candidate_executor in {"fake", "ssh_remote", "development_evaluation"}:
                executor_type = candidate_executor
        process_identity = None
        if durable.remote_runs:
            remote = durable.remote_runs[0]
            identity = json.loads(remote["identity_json"])
            process_identity = OpaqueProcessIdentityV1(
                run_id=identity["run_id"],
                compute_profile_id=identity["compute_profile_id"],
                state=remote["state"],
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

    candidate_digest = campaign.best_development_candidate_ref or gate.get("candidate_digest")
    candidate_ref = _safe_reference(candidate_digest)
    candidate = None
    if candidate_ref is not None:
        provenance = durable.candidate_provenance
        if provenance is not None and provenance.candidate_digest != candidate_digest:
            provenance = None
        gate_matches_candidate = gate.get("candidate_digest") == candidate_digest
        verdict = gate.get("verdict") if gate_matches_candidate else None
        candidate = CandidateSummaryV1(
            candidate_ref=candidate_ref,
            source_attempt_ids=provenance.source_attempt_ids[:100] if provenance else (),
            source_artifact_ids=provenance.source_artifact_ids[:100] if provenance else (),
            latest_comparable_evaluation_id=(
                provenance.latest_comparable_evaluation_id if provenance else None
            ),
            comparison_verdict=(
                verdict if verdict in {"passed", "failed", "insufficient_evidence"} else None
            ),
            gate_state=(
                "passed"
                if verdict == "passed"
                else "failed" if verdict == "failed" else "blocked" if gate else "not_evaluated"
            ),
        )
    champion = None
    champion_ref = _safe_reference(campaign.champion_ref)
    if champion_ref is not None:
        champion_claim = durable.champion_provenance
        champion_candidate_provenance = durable.candidate_provenance
        if (
            champion_claim is None
            or champion_candidate_provenance is None
            or champion_candidate_provenance.candidate_digest != champion_claim.candidate_digest
        ):
            champion_candidate_provenance = None
        champion = CandidateSummaryV1(
            candidate_ref=champion_ref,
            source_attempt_ids=(
                champion_candidate_provenance.source_attempt_ids[:100]
                if champion_candidate_provenance
                else ()
            ),
            source_artifact_ids=(
                champion_candidate_provenance.source_artifact_ids[:100]
                if champion_candidate_provenance
                else ()
            ),
            latest_comparable_evaluation_id=(
                champion_candidate_provenance.latest_comparable_evaluation_id
                if champion_candidate_provenance
                else None
            ),
            comparison_verdict=(champion_claim.comparison_verdict if champion_claim else None),
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
        human_work=human_work,
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
            attention_owner=("human" if human_work.blocking_count > 0 else attention_owner),
            blocker=blocker or human_blocker,
            next_actions=actions,
            recovery_actions=tuple(dict.fromkeys(recovery)),
            promotion_eligible=promotion_eligible,
        ),
    )


def principal_control_room_etag(
    snapshot: CampaignControlRoomSnapshotV1,
    principal: ActorPrincipal,
    *,
    readiness_revision: str | None = None,
) -> str:
    """Return a private validator that cannot be shared across principals."""

    digest = canonical_hash(
        {
            "workspace_id": snapshot.workspace_id,
            "campaign_id": snapshot.campaign_id,
            "aggregate_version": snapshot.aggregate_version,
            "latest_event_cursor": snapshot.latest_event_cursor,
            "controller_observation_version": snapshot.controller.controller_observation_version,
            "readiness": snapshot.readiness.model_dump(mode="json", exclude={"checked_at"}),
            "readiness_revision": readiness_revision,
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
    "CandidateProvenance",
    "ChampionProvenance",
    "ControlRoomDurableProjection",
    "build_control_room_snapshot",
    "if_none_match_matches",
    "principal_control_room_etag",
    "read_control_room_projection",
    "read_control_room_state",
]
