"""Bounded control-room projections over the durable campaign ledger."""

from __future__ import annotations

import sqlite3

from bashgym.campaigns.contracts import (
    ActorPrincipal,
    CampaignControlRoomSnapshotV1,
    CampaignControlRoomStateV1,
    ControlRoomArtifactSummaryV1,
    ControlRoomCampaignV1,
    ControlRoomCollectionSummaryV1,
    ControlRoomControllerObservationV1,
    ControlRoomStatusCountV1,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.persistence import RecordNotFoundError


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


def build_control_room_snapshot(
    durable_state: CampaignControlRoomStateV1,
    controller_observation: ControlRoomControllerObservationV1,
) -> CampaignControlRoomSnapshotV1:
    """Compose transaction state with a separately timestamped observation."""

    return CampaignControlRoomSnapshotV1(
        durable_state=durable_state,
        controller_observation=controller_observation,
    )


def principal_control_room_etag(
    snapshot: CampaignControlRoomSnapshotV1,
    principal: ActorPrincipal,
) -> str:
    """Return a private validator that cannot be shared across principals."""

    payload = snapshot.model_dump(
        mode="json",
        exclude={
            "durable_state": {"state_observed_at"},
            "controller_observation": {"observed_at", "heartbeat_age_seconds"},
        },
    )
    digest = canonical_hash(
        {
            "snapshot": payload,
            "principal": {
                "actor_id": principal.actor_id,
                "credential_id": principal.credential_id,
                "autonomy_profile": principal.autonomy_profile.value,
            },
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
    "build_control_room_snapshot",
    "if_none_match_matches",
    "principal_control_room_etag",
    "read_control_room_state",
]
