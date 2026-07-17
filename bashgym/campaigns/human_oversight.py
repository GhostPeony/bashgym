"""Durable, blinded, human-only oversight for experiment campaigns."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    CampaignEvent,
    Capability,
    CredentialKind,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.evaluation import (
    DevelopmentComparison,
    RetrievalEvaluationArtifact,
)
from bashgym.campaigns.persistence import (
    CampaignRepository,
    IdempotencyConflictError,
    RecordNotFoundError,
)

HUMAN_WORK_QUEUE_SCHEMA_VERSION = "human_work_queue.v1"
HUMAN_WORK_MAX_ITEMS = 50
HUMAN_WORK_LEASE = timedelta(minutes=15)


class HumanOversightError(RuntimeError):
    """Base class for stable, bounded oversight failures."""

    code = "human_work_unavailable"


class HumanOversightConflictError(HumanOversightError):
    code = "human_work_conflict"


class HumanOversightIntegrityError(HumanOversightError):
    code = "human_work_integrity_failed"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Choice(_StrictModel):
    choice_id: Literal["left", "right", "tie"]
    label: str = Field(min_length=1, max_length=80)


class _Rubric(_StrictModel):
    rubric_id: str = Field(pattern=r"^rub_[A-Za-z0-9_-]{8,120}$")
    version: int = Field(ge=1)
    instructions: str = Field(min_length=1, max_length=2000)
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=3)

    @field_validator("choices")
    @classmethod
    def _unique_choices(cls, value: tuple[_Choice, ...]) -> tuple[_Choice, ...]:
        if len({choice.choice_id for choice in value}) != len(value):
            raise ValueError("rubric choices must be unique")
        return value


class _SampleValue(_StrictModel):
    label: str = Field(min_length=1, max_length=80)
    display: str = Field(min_length=1, max_length=12000)


class _BlindedSample(_StrictModel):
    prompt: str = Field(min_length=1, max_length=2000)
    left: _SampleValue
    right: _SampleValue

    @field_validator("right")
    @classmethod
    def _distinct_labels(cls, value: _SampleValue, info):
        left = info.data.get("left")
        if left is not None and left.label == value.label:
            raise ValueError("sample labels must be distinct")
        return value


@dataclass(frozen=True)
class HumanOversightMutation:
    queue: dict[str, Any]
    event: CampaignEvent
    replayed: bool = False


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _wire_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stored_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _new_idempotency_key() -> str:
    return f"idem_{secrets.token_urlsafe(24).replace('-', '_')}"


def _require_reviewer(principal: ActorPrincipal, workspace_id: str) -> None:
    principal.require(workspace_id, Capability.CAMPAIGN_READ)
    if principal.autonomy_profile != AutonomyProfile.DESKTOP_USER:
        raise PermissionError("human_reviewer_required")


def _validate_work_id(work_id: str) -> None:
    if not isinstance(work_id, str) or not re.fullmatch(r"hw_[A-Za-z0-9_-]{16,120}", work_id):
        raise ValueError("invalid human work id")


def _validate_idempotency_key(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"idem_[A-Za-z0-9_-]{16,120}", value):
        raise HumanOversightIntegrityError("human work authority key is invalid")


def _is_canonical_wire_time(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return _wire_time(parsed) == value


class HumanOversightRepository:
    """Transactional oversight repository sharing the campaign SQLite source of truth."""

    def __init__(
        self,
        repository: CampaignRepository,
        *,
        sealer: ArtifactSealer,
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", sealer.key_version):
            raise HumanOversightIntegrityError("human receipt seal version is invalid")
        self.repository = repository
        self.sealer = sealer
        self.seal_key_version = sealer.key_version

    def _seal_receipt(self, payload: dict[str, Any]) -> str:
        signature = self.sealer.sign_canonical_payload(
            payload,
            domain="bashgym.human-oversight-receipt.v1",
        )
        return f"sha256:{signature}"

    def _seal_mutation_response(self, payload: dict[str, Any]) -> str:
        signature = self.sealer.sign_canonical_payload(
            payload,
            domain="bashgym.human-oversight-mutation-replay.v1",
        )
        return f"sha256:{signature}"

    @staticmethod
    def _campaign_row(
        connection: sqlite3.Connection, workspace_id: str, campaign_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
            (workspace_id, campaign_id),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign not found")
        return row

    @staticmethod
    def _promotion_fallback(
        workspace_id: str, campaign_id: str, campaign_revision: int
    ) -> dict[str, Any]:
        digest = hashlib.sha256(
            f"{workspace_id}:{campaign_id}:{campaign_revision}:promotion".encode()
        ).hexdigest()[:24]
        return {
            "state": "blocked_by_review",
            "version": 1,
            "eligible_receipt_id": None,
            "idempotency_key": f"idem_{digest}",
        }

    def _receipt_projection(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any] | None:
        receipt_id = row["receipt_id"]
        if receipt_id is None:
            if row["state"] == "submitted":
                raise HumanOversightIntegrityError("submitted work is missing its receipt")
            return None
        receipt = connection.execute(
            """
            SELECT * FROM campaign_human_receipts
            WHERE workspace_id = ? AND campaign_id = ? AND receipt_id = ?
            """,
            (row["workspace_id"], row["campaign_id"], receipt_id),
        ).fetchone()
        if receipt is None:
            raise HumanOversightIntegrityError("receipt evidence is missing")
        if not re.fullmatch(r"hrc_[A-Za-z0-9_-]{16,120}", receipt["receipt_id"]):
            raise HumanOversightIntegrityError("receipt identity is invalid")
        try:
            payload = json.loads(receipt["sealed_payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise HumanOversightIntegrityError("receipt evidence is invalid") from exc
        expected_keys = {
            "receipt_id",
            "workspace_id",
            "campaign_id",
            "work_id",
            "campaign_revision",
            "item_version",
            "rubric_version",
            "decision",
            "sealed_at",
            "rationale_digest",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise HumanOversightIntegrityError("receipt evidence shape is invalid")
        expected = {
            "receipt_id": receipt["receipt_id"],
            "workspace_id": receipt["workspace_id"],
            "campaign_id": receipt["campaign_id"],
            "work_id": receipt["work_id"],
            "campaign_revision": int(receipt["campaign_revision"]),
            "item_version": int(receipt["item_version"]),
            "rubric_version": int(receipt["rubric_version"]),
            "decision": receipt["decision"],
            "sealed_at": receipt["sealed_at"],
            "rationale_digest": hashlib.sha256(receipt["rationale"].encode()).hexdigest(),
        }
        digest = self._seal_receipt(payload)
        if (
            payload != expected
            or receipt["seal_key_version"] != self.seal_key_version
            or not hmac.compare_digest(str(receipt["receipt_digest"]), digest)
        ):
            raise HumanOversightIntegrityError("receipt digest verification failed")
        if not _is_canonical_wire_time(receipt["sealed_at"]):
            raise HumanOversightIntegrityError("receipt time is invalid")
        if (
            receipt["workspace_id"] != row["workspace_id"]
            or receipt["campaign_id"] != row["campaign_id"]
            or receipt["work_id"] != row["work_id"]
            or int(receipt["campaign_revision"]) != int(row["campaign_revision"])
            or int(receipt["item_version"]) != int(row["item_version"])
        ):
            raise HumanOversightIntegrityError("receipt binding verification failed")
        return {
            key: payload[key]
            for key in (
                "receipt_id",
                "workspace_id",
                "campaign_id",
                "work_id",
                "campaign_revision",
                "item_version",
                "rubric_version",
                "decision",
                "sealed_at",
            )
        } | {"receipt_digest": digest}

    def _read_queue_connection(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        campaign_id: str,
        *,
        reviewer_actor_id: str,
        now: datetime,
        limit: int,
    ) -> dict[str, Any]:
        campaign = self._campaign_row(connection, workspace_id, campaign_id)
        campaign_revision = int(campaign["manifest_revision"])
        promotion = connection.execute(
            """
            SELECT * FROM campaign_human_promotions
            WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
            """,
            (workspace_id, campaign_id, campaign_revision),
        ).fetchone()
        promotion_projection = (
            {
                "state": promotion["state"],
                "version": int(promotion["version"]),
                "eligible_receipt_id": promotion["eligible_receipt_id"],
                "idempotency_key": promotion["idempotency_key"],
            }
            if promotion is not None
            else self._promotion_fallback(workspace_id, campaign_id, campaign_revision)
        )
        _validate_idempotency_key(promotion_projection["idempotency_key"])
        eligible_receipt_id = promotion_projection["eligible_receipt_id"]
        if eligible_receipt_id is not None and not re.fullmatch(
            r"hrc_[A-Za-z0-9_-]{16,120}", eligible_receipt_id
        ):
            raise HumanOversightIntegrityError("promotion receipt identity is invalid")
        rows = connection.execute(
            """
            SELECT * FROM campaign_human_work
            WHERE workspace_id = ? AND campaign_id = ?
            ORDER BY CASE WHEN receipt_id = ? THEN 0 ELSE 1 END,
                     (campaign_revision = ?) DESC, created_at DESC, work_id ASC
            LIMIT ?
            """,
            (workspace_id, campaign_id, eligible_receipt_id, campaign_revision, limit),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                _validate_work_id(row["work_id"])
                _validate_idempotency_key(row["claim_idempotency_key"])
                _validate_idempotency_key(row["submit_idempotency_key"])
                rubric = _Rubric.model_validate_json(row["rubric_json"]).model_dump(mode="json")
                sample = _BlindedSample.model_validate_json(row["public_sample_json"]).model_dump(
                    mode="json"
                )
                lease = _parse_time(row["lease_expires_at"])
            except (TypeError, ValueError, ValidationError) as exc:
                raise HumanOversightIntegrityError("human work public evidence is invalid") from exc
            state = row["state"]
            if state == "claimed" and lease is not None and lease <= now:
                state = "expired"
            receipt = self._receipt_projection(connection, row)
            items.append(
                {
                    "work_id": row["work_id"],
                    "campaign_revision": int(row["campaign_revision"]),
                    "version": int(row["item_version"]),
                    "state": state,
                    "blocking": bool(row["blocking"]),
                    "rubric": rubric,
                    "sample": sample,
                    "lease_expires_at": _wire_time(lease) if lease else None,
                    "claimed_by_current_reviewer": (
                        state == "claimed" and row["claimed_by_actor_id"] == reviewer_actor_id
                    ),
                    "claim_idempotency_key": row["claim_idempotency_key"],
                    "submit_idempotency_key": row["submit_idempotency_key"],
                    "receipt": receipt,
                }
            )
        if eligible_receipt_id is not None and not any(
            item["receipt"] is not None and item["receipt"]["receipt_id"] == eligible_receipt_id
            for item in items
        ):
            raise HumanOversightIntegrityError(
                "eligible promotion receipt is missing from the bounded queue"
            )
        return {
            "schema_version": HUMAN_WORK_QUEUE_SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "campaign_revision": campaign_revision,
            "reviewer": {"authenticated": True, "review_capability": True},
            "items": items,
            "promotion": promotion_projection,
        }

    def read_queue(
        self,
        workspace_id: str,
        campaign_id: str,
        principal: ActorPrincipal,
        *,
        now: datetime | None = None,
        limit: int = HUMAN_WORK_MAX_ITEMS,
    ) -> dict[str, Any]:
        _require_reviewer(principal, workspace_id)
        if limit < 1 or limit > HUMAN_WORK_MAX_ITEMS:
            raise ValueError("human work limit must be between 1 and 50")
        with self.repository._connection() as connection:
            return self._read_queue_connection(
                connection,
                workspace_id,
                campaign_id,
                reviewer_actor_id=principal.actor_id,
                now=now or utc_now(),
                limit=limit,
            )

    def enqueue(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        work_id: str,
        campaign_revision: int,
        blocking: bool,
        rubric: dict[str, Any],
        sample: dict[str, Any],
        protected_context: dict[str, Any],
        replacement_for_work_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert controller-prepared blinded work; never exposed as a public route."""

        _validate_work_id(work_id)
        parsed_rubric = _Rubric.model_validate(rubric).model_dump(mode="json")
        parsed_sample = _BlindedSample.model_validate(sample).model_dump(mode="json")
        if not isinstance(protected_context, dict):
            raise ValueError("protected context must be an object")
        serialized_protected_context = _json(protected_context)
        if len(serialized_protected_context.encode("utf-8")) > 65_536:
            raise ValueError("protected context exceeds the bounded storage limit")
        observed = now or utc_now()
        with self.repository._connection(immediate=True) as connection:
            campaign = self._campaign_row(connection, workspace_id, campaign_id)
            if int(campaign["manifest_revision"]) != campaign_revision:
                raise HumanOversightConflictError("campaign revision changed")
            existing = connection.execute(
                """
                SELECT * FROM campaign_human_work
                WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?
                """,
                (workspace_id, campaign_id, work_id),
            ).fetchone()
            if existing is not None:
                if (
                    int(existing["campaign_revision"]) != campaign_revision
                    or bool(existing["blocking"]) != blocking
                    or existing["rubric_json"] != _json(parsed_rubric)
                    or existing["public_sample_json"] != _json(parsed_sample)
                    or existing["protected_context_json"] != serialized_protected_context
                    or existing["replacement_for_work_id"] != replacement_for_work_id
                ):
                    raise HumanOversightConflictError("human work identity was reused")
                return self._read_queue_connection(
                    connection,
                    workspace_id,
                    campaign_id,
                    reviewer_actor_id="desktop-user",
                    now=observed,
                    limit=HUMAN_WORK_MAX_ITEMS,
                )
            if replacement_for_work_id is not None:
                previous = connection.execute(
                    """
                    SELECT * FROM campaign_human_work
                    WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?
                    """,
                    (workspace_id, campaign_id, replacement_for_work_id),
                ).fetchone()
                if previous is None or int(previous["campaign_revision"]) >= campaign_revision:
                    raise HumanOversightConflictError("replacement lineage is invalid")
                connection.execute(
                    """
                    UPDATE campaign_human_work
                    SET state = 'replaced', item_version = item_version + 1,
                        claimed_by_actor_id = NULL, lease_expires_at = NULL,
                        receipt_id = NULL, updated_at = ?
                    WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?
                    """,
                    (_stored_time(observed), workspace_id, campaign_id, replacement_for_work_id),
                )
            connection.execute(
                """
                INSERT INTO campaign_human_work(
                    workspace_id, campaign_id, work_id, campaign_revision, item_version,
                    state, blocking, rubric_json, public_sample_json, protected_context_json,
                    claimed_by_actor_id, lease_expires_at, claim_idempotency_key,
                    submit_idempotency_key, receipt_id, replacement_for_work_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, 'pending', ?, ?, ?, ?, NULL, NULL, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    work_id,
                    campaign_revision,
                    int(blocking),
                    _json(parsed_rubric),
                    _json(parsed_sample),
                    serialized_protected_context,
                    _new_idempotency_key(),
                    _new_idempotency_key(),
                    replacement_for_work_id,
                    _stored_time(observed),
                    _stored_time(observed),
                ),
            )
            promotion = connection.execute(
                """
                SELECT * FROM campaign_human_promotions
                WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                """,
                (workspace_id, campaign_id, campaign_revision),
            ).fetchone()
            if promotion is None:
                connection.execute(
                    """
                    INSERT INTO campaign_human_promotions(
                        workspace_id, campaign_id, campaign_revision, state, version,
                        eligible_receipt_id, idempotency_key, decided_by_actor_id, updated_at
                    ) VALUES (?, ?, ?, ?, 1, NULL, ?, NULL, ?)
                    """,
                    (
                        workspace_id,
                        campaign_id,
                        campaign_revision,
                        "blocked_by_review" if blocking else "not_required",
                        _new_idempotency_key(),
                        _stored_time(observed),
                    ),
                )
            elif blocking and promotion["state"] != "blocked_by_review":
                connection.execute(
                    """
                    UPDATE campaign_human_promotions
                    SET state = 'blocked_by_review', version = version + 1,
                        eligible_receipt_id = NULL, idempotency_key = ?,
                        decided_by_actor_id = NULL, updated_at = ?
                    WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                    """,
                    (
                        _new_idempotency_key(),
                        _stored_time(observed),
                        workspace_id,
                        campaign_id,
                        campaign_revision,
                    ),
                )
            enqueue_key = f"enqueue-{work_id}-{campaign_revision}"
            event = CampaignEvent(
                event_id=f"evt-human-{uuid4().hex}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self.repository._next_event_sequence(
                    connection, workspace_id, campaign_id
                ),
                aggregate_version=int(campaign["version"]),
                event_type="campaign:human-work-enqueued",
                payload={
                    "work_id": work_id,
                    "campaign_revision": campaign_revision,
                    "item_version": 1,
                    "state": "pending",
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id="human-work-enqueue",
                idempotency_key=enqueue_key,
                created_at=observed,
            )
            self.repository._insert_event(connection, event)
            return self._read_queue_connection(
                connection,
                workspace_id,
                campaign_id,
                reviewer_actor_id="desktop-user",
                now=observed,
                limit=HUMAN_WORK_MAX_ITEMS,
            )

    @staticmethod
    def _public_evaluation_summary(evaluation: RetrievalEvaluationArtifact) -> str:
        rows = evaluation.rows
        count = len(rows)
        summary = {
            "sample_count": count,
            "exact_mrr": sum(1.0 / row.exact_rank for row in rows) / count,
            "local_mrr": sum(1.0 / row.local_rank for row in rows) / count,
            "exact_recall_at_1": sum(row.exact_rank == 1 for row in rows) / count,
            "wrong_top_video_rate": sum(not row.top_video_correct for row in rows) / count,
            "median_latency_ms": evaluation.median_latency_ms,
            "model_footprint_bytes": evaluation.model_footprint_bytes,
        }
        return json.dumps(summary, sort_keys=True, indent=2)

    def enqueue_development_comparison(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        campaign_revision: int,
        comparison: DevelopmentComparison,
        champion: RetrievalEvaluationArtifact,
        candidate: RetrievalEvaluationArtifact,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Materialize one deterministic blinded review before comparison authority."""

        if (
            comparison.champion_digest != champion.candidate_digest
            or comparison.candidate_digest != candidate.candidate_digest
        ):
            raise HumanOversightIntegrityError("comparison evaluation identity is invalid")
        champion_summary = self._public_evaluation_summary(champion)
        candidate_summary = self._public_evaluation_summary(candidate)
        candidate_is_left = int(comparison.comparison_digest[:2], 16) % 2 == 0
        left_role, right_role = (
            ("candidate", "champion") if candidate_is_left else ("champion", "candidate")
        )
        summaries = {"champion": champion_summary, "candidate": candidate_summary}
        work_id = f"hw_{comparison.comparison_digest[:32]}"
        return self.enqueue(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            work_id=work_id,
            campaign_revision=campaign_revision,
            blocking=True,
            rubric={
                "rubric_id": f"rub_{comparison.comparison_digest[:24]}",
                "version": 1,
                "instructions": (
                    "Compare the blinded aggregate evaluation summaries. Choose the stronger "
                    "result under the campaign quality, regression, latency, and footprint criteria."
                ),
                "choices": [
                    {"choice_id": "left", "label": "Sample A is stronger"},
                    {"choice_id": "right", "label": "Sample B is stronger"},
                    {"choice_id": "tie", "label": "No material difference"},
                ],
            },
            sample={
                "prompt": "Review the blinded candidate comparison before BashGym records the development gate.",
                "left": {"label": "Sample A", "display": summaries[left_role]},
                "right": {"label": "Sample B", "display": summaries[right_role]},
            },
            protected_context={
                "kind": "development_comparison.v1",
                "left_role": left_role,
                "right_role": right_role,
                "comparison": comparison.model_dump(mode="json"),
            },
            now=now,
        )

    @staticmethod
    def _protected_development_comparison(
        row: sqlite3.Row,
    ) -> DevelopmentComparison | None:
        try:
            context = json.loads(row["protected_context_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise HumanOversightIntegrityError("protected review context is invalid") from exc
        if not isinstance(context, dict) or context.get("kind") != "development_comparison.v1":
            return None
        if set(context) != {"kind", "left_role", "right_role", "comparison"}:
            raise HumanOversightIntegrityError("protected comparison context shape is invalid")
        if {context.get("left_role"), context.get("right_role")} != {
            "champion",
            "candidate",
        }:
            raise HumanOversightIntegrityError("protected comparison blinding is invalid")
        try:
            comparison = DevelopmentComparison.model_validate(context["comparison"])
        except ValidationError as exc:
            raise HumanOversightIntegrityError("protected comparison evidence is invalid") from exc
        if row["work_id"] != f"hw_{comparison.comparison_digest[:32]}":
            raise HumanOversightIntegrityError("protected comparison identity is invalid")
        return comparison

    def _replay(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        actor_id: str,
        mutation_kind: str,
        idempotency_key: str,
        request_hash: str,
        now: datetime,
    ) -> HumanOversightMutation | None:
        row = connection.execute(
            """
            SELECT * FROM campaign_human_mutations
            WHERE workspace_id = ? AND actor_id = ? AND mutation_kind = ? AND idempotency_key = ?
            """,
            (workspace_id, actor_id, mutation_kind, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflictError()
        event_row = connection.execute(
            "SELECT * FROM campaign_events WHERE event_id = ?", (row["event_id"],)
        ).fetchone()
        if event_row is None:
            raise HumanOversightIntegrityError("mutation event is missing")
        expected_event_type = {
            "human_work.claim": "campaign:human-work-claimed",
            "human_work.submit": "campaign:human-work-submitted",
            "human_work.promotion": (
                "campaign:human-promotion-approved",
                "campaign:human-promotion-held",
            ),
        }.get(mutation_kind)
        allowed_event_types = (
            expected_event_type
            if isinstance(expected_event_type, tuple)
            else (expected_event_type,)
        )
        if (
            event_row["workspace_id"] != workspace_id
            or event_row["campaign_id"] != row["campaign_id"]
            or event_row["actor_id"] != actor_id
            or event_row["idempotency_key"] != idempotency_key
            or event_row["event_type"] not in allowed_event_types
        ):
            raise HumanOversightIntegrityError("mutation replay event binding is invalid")
        try:
            response = json.loads(row["response_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise HumanOversightIntegrityError("mutation replay is invalid") from exc
        if (
            not isinstance(response, dict)
            or set(response)
            != {
                "schema_version",
                "workspace_id",
                "campaign_id",
                "mutation_kind",
                "event_id",
            }
            or response.get("schema_version") != "human_work_mutation_receipt.v1"
            or response.get("workspace_id") != workspace_id
            or response.get("campaign_id") != row["campaign_id"]
            or response.get("mutation_kind") != mutation_kind
            or response.get("event_id") != row["event_id"]
            or row["response_seal_key_version"] != self.seal_key_version
            or not hmac.compare_digest(
                str(row["response_digest"]),
                self._seal_mutation_response(response),
            )
        ):
            raise HumanOversightIntegrityError("mutation replay digest verification failed")
        return HumanOversightMutation(
            queue=self._read_queue_connection(
                connection,
                workspace_id,
                str(row["campaign_id"]),
                reviewer_actor_id=actor_id,
                now=now,
                limit=HUMAN_WORK_MAX_ITEMS,
            ),
            event=CampaignRepository._event_from_row(event_row),
            replayed=True,
        )

    def _record_mutation(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        campaign_id: str,
        actor_id: str,
        credential_kind,
        mutation_kind: str,
        event_type: str,
        safe_payload: dict[str, Any],
        idempotency_key: str,
        correlation_id: str,
        request_hash: str,
        queue: dict[str, Any],
        now: datetime,
    ) -> HumanOversightMutation:
        campaign = self._campaign_row(connection, workspace_id, campaign_id)
        event = CampaignEvent(
            event_id=f"evt-human-{uuid4().hex}",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            sequence=self.repository._next_event_sequence(connection, workspace_id, campaign_id),
            aggregate_version=int(campaign["version"]),
            event_type=event_type,
            payload=safe_payload,
            actor_id=actor_id,
            credential_kind=credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            created_at=now,
        )
        self.repository._insert_event(connection, event)
        replay_receipt = {
            "schema_version": "human_work_mutation_receipt.v1",
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "mutation_kind": mutation_kind,
            "event_id": event.event_id,
        }
        connection.execute(
            """
            INSERT INTO campaign_human_mutations(
                workspace_id, actor_id, mutation_kind, idempotency_key, request_hash,
                campaign_id, event_id, response_json, created_at, response_digest,
                response_seal_key_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                actor_id,
                mutation_kind,
                idempotency_key,
                request_hash,
                campaign_id,
                event.event_id,
                _json(replay_receipt),
                _stored_time(now),
                self._seal_mutation_response(replay_receipt),
                self.seal_key_version,
            ),
        )
        return HumanOversightMutation(queue=queue, event=event)

    def claim(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        work_id: str,
        expected_campaign_revision: int,
        expected_version: int,
        expected_state: Literal["pending", "expired"],
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> HumanOversightMutation:
        _require_reviewer(principal, workspace_id)
        _validate_work_id(work_id)
        observed = now or utc_now()
        request_hash = canonical_hash(
            {
                "campaign_id": campaign_id,
                "work_id": work_id,
                "campaign_revision": expected_campaign_revision,
                "version": expected_version,
                "state": expected_state,
            }
        )
        with self.repository._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=workspace_id,
                actor_id=principal.actor_id,
                mutation_kind="human_work.claim",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                now=observed,
            )
            if replay is not None:
                return replay
            campaign = self._campaign_row(connection, workspace_id, campaign_id)
            row = connection.execute(
                "SELECT * FROM campaign_human_work WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?",
                (workspace_id, campaign_id, work_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("human work not found")
            lease = _parse_time(row["lease_expires_at"])
            effective_state = (
                "expired"
                if row["state"] == "claimed" and lease is not None and lease <= observed
                else row["state"]
            )
            if (
                int(campaign["manifest_revision"]) != expected_campaign_revision
                or int(row["campaign_revision"]) != expected_campaign_revision
                or int(row["item_version"]) != expected_version
                or effective_state != expected_state
                or row["claim_idempotency_key"] != idempotency_key
            ):
                raise HumanOversightConflictError("human work claim changed")
            next_version = expected_version + 1
            connection.execute(
                """
                UPDATE campaign_human_work
                SET state = 'claimed', item_version = ?, claimed_by_actor_id = ?,
                    lease_expires_at = ?, claim_idempotency_key = ?,
                    submit_idempotency_key = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?
                """,
                (
                    next_version,
                    principal.actor_id,
                    _stored_time(observed + HUMAN_WORK_LEASE),
                    _new_idempotency_key(),
                    _new_idempotency_key(),
                    _stored_time(observed),
                    workspace_id,
                    campaign_id,
                    work_id,
                ),
            )
            queue = self._read_queue_connection(
                connection,
                workspace_id,
                campaign_id,
                reviewer_actor_id=principal.actor_id,
                now=observed,
                limit=HUMAN_WORK_MAX_ITEMS,
            )
            return self._record_mutation(
                connection,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                actor_id=principal.actor_id,
                credential_kind=principal.credential_kind,
                mutation_kind="human_work.claim",
                event_type="campaign:human-work-claimed",
                safe_payload={
                    "work_id": work_id,
                    "campaign_revision": expected_campaign_revision,
                    "item_version": next_version,
                    "state": "claimed",
                },
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                request_hash=request_hash,
                queue=queue,
                now=observed,
            )

    def submit(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        work_id: str,
        expected_campaign_revision: int,
        expected_version: int,
        expected_rubric_version: int,
        decision: Literal["prefer_left", "prefer_right", "no_material_difference"],
        rationale: str,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> HumanOversightMutation:
        _require_reviewer(principal, workspace_id)
        _validate_work_id(work_id)
        if not rationale or len(rationale) > 2000:
            raise ValueError("human rationale must be between 1 and 2000 characters")
        observed = now or utc_now()
        request_hash = canonical_hash(
            {
                "campaign_id": campaign_id,
                "work_id": work_id,
                "campaign_revision": expected_campaign_revision,
                "version": expected_version,
                "rubric_version": expected_rubric_version,
                "decision": decision,
                "rationale": rationale,
            }
        )
        with self.repository._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=workspace_id,
                actor_id=principal.actor_id,
                mutation_kind="human_work.submit",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                now=observed,
            )
            if replay is not None:
                return replay
            campaign = self._campaign_row(connection, workspace_id, campaign_id)
            row = connection.execute(
                "SELECT * FROM campaign_human_work WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?",
                (workspace_id, campaign_id, work_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("human work not found")
            rubric = _Rubric.model_validate_json(row["rubric_json"])
            lease = _parse_time(row["lease_expires_at"])
            if (
                int(campaign["manifest_revision"]) != expected_campaign_revision
                or int(row["campaign_revision"]) != expected_campaign_revision
                or int(row["item_version"]) != expected_version
                or row["state"] != "claimed"
                or row["claimed_by_actor_id"] != principal.actor_id
                or lease is None
                or lease <= observed
                or rubric.version != expected_rubric_version
                or row["submit_idempotency_key"] != idempotency_key
            ):
                raise HumanOversightConflictError("human work submission changed")
            offered = {
                "left": "prefer_left",
                "right": "prefer_right",
                "tie": "no_material_difference",
            }
            if decision not in {offered[choice.choice_id] for choice in rubric.choices}:
                raise HumanOversightConflictError("decision is not in the current rubric")
            protected_comparison = self._protected_development_comparison(row)
            next_version = expected_version + 1
            receipt_id = f"hrc_{uuid4().hex}"
            sealed_at = _wire_time(observed)
            sealed_payload = {
                "receipt_id": receipt_id,
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "work_id": work_id,
                "campaign_revision": expected_campaign_revision,
                "item_version": next_version,
                "rubric_version": expected_rubric_version,
                "decision": decision,
                "sealed_at": sealed_at,
                "rationale_digest": hashlib.sha256(rationale.encode()).hexdigest(),
            }
            receipt_digest = self._seal_receipt(sealed_payload)
            connection.execute(
                """
                INSERT INTO campaign_human_receipts(
                    workspace_id, campaign_id, receipt_id, work_id, campaign_revision,
                    item_version, rubric_version, decision, rationale, sealed_payload_json,
                    receipt_digest, sealed_at, seal_key_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    receipt_id,
                    work_id,
                    expected_campaign_revision,
                    next_version,
                    expected_rubric_version,
                    decision,
                    rationale,
                    _json(sealed_payload),
                    receipt_digest,
                    sealed_at,
                    self.seal_key_version,
                ),
            )
            connection.execute(
                """
                UPDATE campaign_human_work
                SET state = 'submitted', item_version = ?, claimed_by_actor_id = NULL,
                    lease_expires_at = NULL, claim_idempotency_key = ?,
                    submit_idempotency_key = ?, receipt_id = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?
                """,
                (
                    next_version,
                    _new_idempotency_key(),
                    _new_idempotency_key(),
                    receipt_id,
                    _stored_time(observed),
                    workspace_id,
                    campaign_id,
                    work_id,
                ),
            )
            if protected_comparison is not None:
                decision_id = f"gate-{protected_comparison.comparison_digest[:24]}"
                comparison_payload = _json(protected_comparison.model_dump(mode="json"))
                existing_comparison = connection.execute(
                    """
                    SELECT decision_json FROM campaign_gate_decisions
                    WHERE workspace_id = ? AND decision_id = ?
                    """,
                    (workspace_id, decision_id),
                ).fetchone()
                if existing_comparison is not None:
                    if existing_comparison["decision_json"] != comparison_payload:
                        raise HumanOversightIntegrityError(
                            "human-reviewed comparison identity changed"
                        )
                else:
                    connection.execute(
                        """
                        INSERT INTO campaign_gate_decisions(
                            workspace_id, campaign_id, decision_id, decision_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            workspace_id,
                            campaign_id,
                            decision_id,
                            comparison_payload,
                            _stored_time(observed),
                        ),
                    )
            promotion = connection.execute(
                """
                SELECT * FROM campaign_human_promotions
                WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                """,
                (workspace_id, campaign_id, expected_campaign_revision),
            ).fetchone()
            if promotion is None:
                raise HumanOversightIntegrityError("promotion state is missing")
            blocking_counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN state != 'submitted' THEN 1 ELSE 0 END) AS incomplete
                FROM campaign_human_work
                WHERE workspace_id = ? AND campaign_id = ?
                  AND campaign_revision = ? AND blocking = 1
                """,
                (workspace_id, campaign_id, expected_campaign_revision),
            ).fetchone()
            blocker_total = int(blocking_counts["total"])
            blockers = int(blocking_counts["incomplete"] or 0)
            promotion_state = (
                "not_required"
                if blocker_total == 0
                else "awaiting_human_decision" if blockers == 0 else "blocked_by_review"
            )
            connection.execute(
                """
                UPDATE campaign_human_promotions
                SET state = ?, version = version + 1,
                    eligible_receipt_id = ?, idempotency_key = ?,
                    decided_by_actor_id = NULL, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                """,
                (
                    promotion_state,
                    receipt_id if promotion_state == "awaiting_human_decision" else None,
                    _new_idempotency_key(),
                    _stored_time(observed),
                    workspace_id,
                    campaign_id,
                    expected_campaign_revision,
                ),
            )
            queue = self._read_queue_connection(
                connection,
                workspace_id,
                campaign_id,
                reviewer_actor_id=principal.actor_id,
                now=observed,
                limit=HUMAN_WORK_MAX_ITEMS,
            )
            return self._record_mutation(
                connection,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                actor_id=principal.actor_id,
                credential_kind=principal.credential_kind,
                mutation_kind="human_work.submit",
                event_type="campaign:human-work-submitted",
                safe_payload={
                    "work_id": work_id,
                    "campaign_revision": expected_campaign_revision,
                    "item_version": next_version,
                    "state": "submitted",
                },
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                request_hash=request_hash,
                queue=queue,
                now=observed,
            )

    def decide_promotion(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        receipt_id: str,
        work_id: str,
        expected_campaign_revision: int,
        expected_item_version: int,
        expected_rubric_version: int,
        expected_promotion_version: int,
        expected_promotion_state: Literal["awaiting_human_decision"],
        decision: Literal["promote", "hold"],
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> HumanOversightMutation:
        _require_reviewer(principal, workspace_id)
        observed = now or utc_now()
        request_hash = canonical_hash(
            {
                "campaign_id": campaign_id,
                "receipt_id": receipt_id,
                "work_id": work_id,
                "campaign_revision": expected_campaign_revision,
                "item_version": expected_item_version,
                "rubric_version": expected_rubric_version,
                "promotion_version": expected_promotion_version,
                "promotion_state": expected_promotion_state,
                "decision": decision,
            }
        )
        with self.repository._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=workspace_id,
                actor_id=principal.actor_id,
                mutation_kind="human_work.promotion",
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                now=observed,
            )
            if replay is not None:
                return replay
            campaign = self._campaign_row(connection, workspace_id, campaign_id)
            work = connection.execute(
                "SELECT * FROM campaign_human_work WHERE workspace_id = ? AND campaign_id = ? AND work_id = ?",
                (workspace_id, campaign_id, work_id),
            ).fetchone()
            promotion = connection.execute(
                """
                SELECT * FROM campaign_human_promotions
                WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                """,
                (workspace_id, campaign_id, expected_campaign_revision),
            ).fetchone()
            if work is None or promotion is None:
                raise RecordNotFoundError("human promotion evidence not found")
            rubric = _Rubric.model_validate_json(work["rubric_json"])
            if (
                int(campaign["manifest_revision"]) != expected_campaign_revision
                or int(work["campaign_revision"]) != expected_campaign_revision
                or work["state"] != "submitted"
                or work["receipt_id"] != receipt_id
                or int(work["item_version"]) != expected_item_version
                or rubric.version != expected_rubric_version
                or promotion["state"] != expected_promotion_state
                or int(promotion["version"]) != expected_promotion_version
                or promotion["eligible_receipt_id"] != receipt_id
                or promotion["idempotency_key"] != idempotency_key
            ):
                raise HumanOversightConflictError("human promotion state changed")
            blocker = connection.execute(
                """
                SELECT 1 FROM campaign_human_work
                WHERE workspace_id = ? AND campaign_id = ?
                  AND campaign_revision = ? AND blocking = 1
                  AND state != 'submitted'
                LIMIT 1
                """,
                (workspace_id, campaign_id, expected_campaign_revision),
            ).fetchone()
            if blocker is not None:
                raise HumanOversightConflictError("human review still blocks promotion")
            # Receipt projection performs the full tamper-evident verification before authority changes.
            self._receipt_projection(connection, work)
            next_state = "promoted" if decision == "promote" else "blocked_by_review"
            connection.execute(
                """
                UPDATE campaign_human_promotions
                SET state = ?, version = version + 1, eligible_receipt_id = ?,
                    idempotency_key = ?, decided_by_actor_id = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND campaign_revision = ?
                """,
                (
                    next_state,
                    receipt_id if decision == "promote" else None,
                    _new_idempotency_key(),
                    principal.actor_id,
                    _stored_time(observed),
                    workspace_id,
                    campaign_id,
                    expected_campaign_revision,
                ),
            )
            queue = self._read_queue_connection(
                connection,
                workspace_id,
                campaign_id,
                reviewer_actor_id=principal.actor_id,
                now=observed,
                limit=HUMAN_WORK_MAX_ITEMS,
            )
            event_type = (
                "campaign:human-promotion-approved"
                if decision == "promote"
                else "campaign:human-promotion-held"
            )
            return self._record_mutation(
                connection,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                actor_id=principal.actor_id,
                credential_kind=principal.credential_kind,
                mutation_kind="human_work.promotion",
                event_type=event_type,
                safe_payload={
                    "work_id": work_id,
                    "campaign_revision": expected_campaign_revision,
                    "item_version": expected_item_version,
                    "promotion_state": next_state,
                },
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                request_hash=request_hash,
                queue=queue,
                now=observed,
            )


__all__ = [
    "HUMAN_WORK_MAX_ITEMS",
    "HUMAN_WORK_QUEUE_SCHEMA_VERSION",
    "HumanOversightConflictError",
    "HumanOversightError",
    "HumanOversightIntegrityError",
    "HumanOversightMutation",
    "HumanOversightRepository",
]
