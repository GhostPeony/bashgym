"""SQLite source of truth for experiment campaigns and campaign credentials."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bashgym.campaigns.contracts import (
    BudgetLedgerEntry,
    Campaign,
    CampaignArtifactReference,
    CampaignControlRoomStateV1,
    CampaignEvent,
    CampaignEvidenceSnapshot,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    CodeLineageRecord,
    CodeLineageState,
    CompletedHypothesisSummary,
    CredentialKind,
    ManifestRevision,
    NemoGymEvidenceReference,
    ProposalRecord,
    ProposalStatus,
    ProposalValidation,
    ProtectedEvaluationResult,
    Study,
    StudyProposal,
    StudyProposalSubmission,
    StudyStatus,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.nemo_gym_evidence import NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA
from bashgym.campaigns.transitions import evaluate_promotion_gate, transition_campaign

if TYPE_CHECKING:
    from bashgym.campaigns.control_room import ControlRoomDurableProjection


class CampaignPersistenceError(RuntimeError):
    """Base class for stable campaign persistence failures."""


class RecordNotFoundError(CampaignPersistenceError):
    code = "campaign_not_found"


class RecordAlreadyExistsError(CampaignPersistenceError):
    code = "campaign_already_exists"


class RevisionConflictError(CampaignPersistenceError):
    code = "campaign_revision_conflict"

    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(f"{self.code}: expected version {expected}, current version {current}")


class IdempotencyConflictError(CampaignPersistenceError):
    code = "campaign_idempotency_conflict"

    def __init__(self) -> None:
        super().__init__(self.code)


class MigrationChecksumError(CampaignPersistenceError):
    code = "campaign_migration_checksum_mismatch"


class LeaseBusyError(CampaignPersistenceError):
    code = "campaign_lease_busy"


class LeaseLostError(CampaignPersistenceError):
    code = "campaign_lease_lost"


class BudgetExceededError(CampaignPersistenceError):
    code = "campaign_budget_exceeded"


class BudgetInvariantError(CampaignPersistenceError):
    code = "campaign_budget_invariant"


class CampaignBudgetResourceLimitError(CampaignPersistenceError):
    code = "campaign_budget_resource_limit_exceeded"

    def __init__(self) -> None:
        super().__init__(self.code)


class InvalidProposalTransitionError(CampaignPersistenceError):
    code = "campaign_invalid_transition"

    def __init__(self, message: str = "proposal transition is not allowed") -> None:
        super().__init__(f"{self.code}: {message}")


class ProtectedLeaseDeniedError(CampaignPersistenceError):
    code = "campaign_protected_lease_denied"


class PromotionGateFailedError(CampaignPersistenceError):
    code = "campaign_gate_failed"


@dataclass(frozen=True)
class CampaignMutation:
    campaign: Campaign
    event: CampaignEvent
    replayed: bool = False


@dataclass(frozen=True)
class ProposalMutation:
    campaign: Campaign
    event: CampaignEvent
    record: ProposalRecord
    replayed: bool = False


@dataclass(frozen=True)
class ProposalSelection:
    campaign: Campaign
    event: CampaignEvent
    record: ProposalRecord
    study: Study
    replayed: bool = False


@dataclass(frozen=True)
class StoredCredential:
    credential_id: str
    actor_id: str
    autonomy_profile: str
    credential_kind: str
    workspace_ids: tuple[str, ...]
    authorization_revision: int
    token_salt: str
    token_hash: str
    issued_at: datetime
    expires_at: datetime
    token_not_before: datetime
    revoked_at: datetime | None


@dataclass(frozen=True)
class StoredAccessToken:
    access_token_id: str
    credential_id: str
    token_salt: str
    token_hash: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None


@dataclass(frozen=True)
class LeaseRecord:
    lease_key: str
    owner_id: str
    generation: int
    controller_observation_version: int
    expires_at: datetime
    heartbeat_at: datetime


@dataclass(frozen=True)
class BudgetMutation:
    campaign: Campaign
    event: CampaignEvent
    entry: BudgetLedgerEntry
    replayed: bool = False


@dataclass(frozen=True)
class OperationMutation:
    """Audited campaign mutation with a bounded, secret-free result projection."""

    campaign: Campaign
    event: CampaignEvent
    details: dict[str, Any]
    replayed: bool = False


_INITIAL_SCHEMA = (
    """
    CREATE TABLE campaigns (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        title TEXT NOT NULL,
        kind TEXT NOT NULL,
        objective TEXT NOT NULL,
        target_model_json TEXT NOT NULL,
        owner_actor_id TEXT NOT NULL,
        manifest_revision INTEGER NOT NULL CHECK(manifest_revision >= 1),
        status TEXT NOT NULL,
        prior_scheduling_status TEXT,
        active_study_id TEXT,
        active_action_id TEXT,
        champion_ref TEXT,
        best_development_candidate_ref TEXT,
        stop_reason TEXT,
        version INTEGER NOT NULL CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, campaign_id)
    )
    """,
    """
    CREATE TABLE campaign_manifest_revisions (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        manifest_json TEXT NOT NULL,
        manifest_hash TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        correlation_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, campaign_id, revision),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_proposals (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        proposal_id TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 50 CHECK(priority BETWEEN 0 AND 100),
        estimated_cost REAL NOT NULL DEFAULT 0 CHECK(estimated_cost >= 0),
        creation_sequence INTEGER NOT NULL,
        proposal_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, proposal_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_studies (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        study_id TEXT NOT NULL,
        proposal_id TEXT NOT NULL,
        status TEXT NOT NULL,
        current_stage_index INTEGER NOT NULL DEFAULT 0 CHECK(current_stage_index >= 0),
        stage_plan_json TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, study_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT,
        FOREIGN KEY(workspace_id, proposal_id)
            REFERENCES campaign_proposals(workspace_id, proposal_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_actions (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        study_id TEXT NOT NULL,
        action_id TEXT NOT NULL,
        stage_index INTEGER NOT NULL CHECK(stage_index >= 0),
        stage_kind TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, action_id),
        UNIQUE(workspace_id, campaign_id, study_id, stage_index, input_digest),
        FOREIGN KEY(workspace_id, study_id)
            REFERENCES campaign_studies(workspace_id, study_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_attempts (
        workspace_id TEXT NOT NULL,
        action_id TEXT NOT NULL,
        attempt_id TEXT NOT NULL,
        attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
        claim_generation INTEGER NOT NULL DEFAULT 0 CHECK(claim_generation >= 0),
        status TEXT NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        heartbeat_at TEXT,
        executor_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, attempt_id),
        UNIQUE(workspace_id, action_id, attempt_number),
        FOREIGN KEY(workspace_id, action_id)
            REFERENCES campaign_actions(workspace_id, action_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_artifacts (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        artifact_id TEXT NOT NULL,
        producer_action_id TEXT,
        uri TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        schema_name TEXT NOT NULL,
        sealed INTEGER NOT NULL CHECK(sealed IN (0, 1)),
        valid INTEGER NOT NULL CHECK(valid IN (0, 1)),
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, artifact_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_evaluations (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        evaluation_id TEXT NOT NULL,
        evaluation_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, evaluation_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_gate_decisions (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        decision_id TEXT NOT NULL,
        decision_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, decision_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_champions (
        workspace_id TEXT NOT NULL,
        target_contract_key TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision >= 1),
        champion_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, target_contract_key, revision)
    )
    """,
    """
    CREATE TABLE campaign_budget_ledger (
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        unit TEXT NOT NULL,
        entry_kind TEXT NOT NULL,
        reserved_delta REAL NOT NULL DEFAULT 0,
        actual_delta REAL NOT NULL DEFAULT 0,
        limit_delta REAL NOT NULL DEFAULT 0,
        action_id TEXT,
        evidence_json TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, entry_id),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_events (
        cursor INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        workspace_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK(sequence >= 1),
        aggregate_version INTEGER NOT NULL CHECK(aggregate_version >= 1),
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        credential_kind TEXT NOT NULL,
        correlation_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(workspace_id, campaign_id, sequence),
        FOREIGN KEY(workspace_id, campaign_id)
            REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_mutations (
        workspace_id TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        mutation_kind TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, actor_id, mutation_kind, idempotency_key),
        FOREIGN KEY(event_id) REFERENCES campaign_events(event_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_actor_credentials (
        credential_id TEXT PRIMARY KEY,
        actor_id TEXT NOT NULL,
        autonomy_profile TEXT NOT NULL,
        workspace_ids_json TEXT NOT NULL,
        token_salt TEXT NOT NULL,
        token_hash TEXT NOT NULL,
        issued_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        token_not_before TEXT NOT NULL,
        revoked_at TEXT
    )
    """,
    """
    CREATE TABLE campaign_access_tokens (
        access_token_id TEXT PRIMARY KEY,
        credential_id TEXT NOT NULL,
        token_salt TEXT NOT NULL,
        token_hash TEXT NOT NULL,
        issued_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT,
        FOREIGN KEY(credential_id)
            REFERENCES campaign_actor_credentials(credential_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE campaign_auth_audit_events (
        event_id TEXT PRIMARY KEY,
        credential_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        safe_payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE campaign_protected_epochs (
        workspace_id TEXT NOT NULL,
        protected_epoch_id TEXT NOT NULL,
        target_contract_key TEXT NOT NULL,
        protected_set_hash TEXT NOT NULL,
        candidate_lock_digest TEXT NOT NULL,
        lease_state TEXT NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
        result_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(workspace_id, protected_epoch_id),
        UNIQUE(workspace_id, target_contract_key, protected_set_hash)
    )
    """,
    """
    CREATE TABLE campaign_scheduler_leases (
        lease_key TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK(generation >= 1),
        expires_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX idx_campaigns_status ON campaigns(workspace_id, status)",
    "CREATE INDEX idx_campaign_events_cursor ON campaign_events(workspace_id, campaign_id, cursor)",
    "CREATE INDEX idx_campaign_attempts_status ON campaign_attempts(workspace_id, status, lease_expires_at)",
    "CREATE INDEX idx_campaign_access_parent ON campaign_access_tokens(credential_id, expires_at)",
)

MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "initial_campaign_control_plane", _INITIAL_SCHEMA),
    (
        2,
        "durable_action_runtime",
        (
            "ALTER TABLE campaign_studies ADD COLUMN candidate_digest TEXT NOT NULL DEFAULT '0000000000000000000000000000000000000000000000000000000000000000'",
            "ALTER TABLE campaign_actions ADD COLUMN candidate_digest TEXT NOT NULL DEFAULT '0000000000000000000000000000000000000000000000000000000000000000'",
            "ALTER TABLE campaign_actions ADD COLUMN manifest_revision INTEGER NOT NULL DEFAULT 1 CHECK(manifest_revision >= 1)",
            "ALTER TABLE campaign_actions ADD COLUMN action_key TEXT",
            "ALTER TABLE campaign_actions ADD COLUMN reservation_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE campaign_actions ADD COLUMN sealed_result_uri TEXT",
            "CREATE UNIQUE INDEX idx_campaign_actions_key ON campaign_actions(workspace_id, action_key) WHERE action_key IS NOT NULL",
        ),
    ),
    (
        3,
        "remote_run_runtime",
        (
            """
            CREATE TABLE campaign_remote_runs (
                workspace_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                claim_generation INTEGER NOT NULL CHECK(claim_generation >= 1),
                identity_json TEXT NOT NULL,
                state TEXT NOT NULL,
                metric_cursor_json TEXT NOT NULL,
                log_cursor_json TEXT NOT NULL,
                last_observation_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, attempt_id),
                UNIQUE(workspace_id, attempt_id, claim_generation),
                FOREIGN KEY(workspace_id, attempt_id)
                    REFERENCES campaign_attempts(workspace_id, attempt_id) ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_campaign_remote_runs_state ON campaign_remote_runs(workspace_id, state, updated_at)",
        ),
    ),
    (
        4,
        "training_metric_streams",
        (
            """
            CREATE TABLE campaign_metric_points (
                workspace_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                source TEXT NOT NULL,
                step INTEGER NOT NULL CHECK(step >= 0),
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                raw_sha256 TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, attempt_id, source, step, metric_name),
                FOREIGN KEY(workspace_id, attempt_id)
                    REFERENCES campaign_attempts(workspace_id, attempt_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE campaign_training_alerts (
                workspace_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                alert_code TEXT NOT NULL,
                step INTEGER NOT NULL CHECK(step >= 0),
                metric_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_value REAL NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, attempt_id, alert_code, step, metric_name),
                FOREIGN KEY(workspace_id, attempt_id)
                    REFERENCES campaign_attempts(workspace_id, attempt_id) ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_campaign_metric_series ON campaign_metric_points(workspace_id, attempt_id, metric_name, step)",
        ),
    ),
    (
        5,
        "desktop_bootstrap_credentials",
        (
            "ALTER TABLE campaign_actor_credentials ADD COLUMN credential_kind TEXT NOT NULL DEFAULT 'refresh'",
        ),
    ),
    (
        6,
        "durable_campaign_proposals",
        (
            "ALTER TABLE campaign_proposals ADD COLUMN validation_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE campaign_proposals ADD COLUMN study_id TEXT",
            "ALTER TABLE campaign_proposals ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
            "CREATE INDEX idx_campaign_proposal_queue ON campaign_proposals(workspace_id, campaign_id, status, priority DESC, estimated_cost ASC, creation_sequence ASC, proposal_id ASC)",
        ),
    ),
    (
        7,
        "campaign_operator_actions",
        (
            """
            CREATE TABLE campaign_source_approvals (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                evidence_digest TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, campaign_id, source_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE campaign_action_control_requests (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                control TEXT NOT NULL,
                expected_identity_digest TEXT NOT NULL,
                reason TEXT NOT NULL,
                state TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, request_id),
                FOREIGN KEY(workspace_id, action_id)
                    REFERENCES campaign_actions(workspace_id, action_id) ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_campaign_action_controls_pending ON campaign_action_control_requests(workspace_id, state, created_at)",
            """
            CREATE TABLE campaign_exports (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                export_id TEXT NOT NULL,
                formats_json TEXT NOT NULL,
                export_manifest_json TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, export_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
            )
            """,
            "ALTER TABLE campaign_protected_epochs ADD COLUMN campaign_id TEXT NOT NULL DEFAULT ''",
        ),
    ),
    (
        8,
        "project_isolated_experiment_ledger",
        (
            """
            CREATE TABLE ledger_projects (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                owner_actor_id TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id)
            )
            """,
            """
            CREATE TABLE ledger_experiments (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                name TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                campaign_id TEXT,
                parent_experiment_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, experiment_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_models (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                architecture TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, model_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_model_versions (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_revision TEXT NOT NULL DEFAULT '',
                parent_model_version_id TEXT,
                config_digest TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, model_version_id),
                FOREIGN KEY(workspace_id, project_id, model_id)
                    REFERENCES ledger_models(workspace_id, project_id, model_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_datasets (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, dataset_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_dataset_versions (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                dataset_version_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                split_manifest_json TEXT NOT NULL DEFAULT '{}',
                row_counts_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, dataset_version_id),
                FOREIGN KEY(workspace_id, project_id, dataset_id)
                    REFERENCES ledger_datasets(workspace_id, project_id, dataset_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_environments (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                environment_id TEXT NOT NULL,
                compute_target TEXT NOT NULL,
                runtime_digest TEXT NOT NULL,
                hardware_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, environment_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_runs (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source_system TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                campaign_id TEXT,
                study_id TEXT,
                action_id TEXT,
                run_kind TEXT NOT NULL,
                task_type TEXT NOT NULL,
                training_method TEXT NOT NULL,
                status TEXT NOT NULL,
                context_status TEXT NOT NULL,
                model_version_id TEXT,
                dataset_version_id TEXT,
                environment_id TEXT,
                recipe_digest TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                correlation_id TEXT NOT NULL,
                is_simulation INTEGER NOT NULL DEFAULT 0 CHECK(is_simulation IN (0, 1)),
                identity_digest TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, run_id),
                UNIQUE(workspace_id, source_system, source_run_id),
                FOREIGN KEY(workspace_id, project_id, experiment_id)
                    REFERENCES ledger_experiments(workspace_id, project_id, experiment_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, model_version_id)
                    REFERENCES ledger_model_versions(workspace_id, project_id, model_version_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, dataset_version_id)
                    REFERENCES ledger_dataset_versions(workspace_id, project_id, dataset_version_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, environment_id)
                    REFERENCES ledger_environments(workspace_id, project_id, environment_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_run_attempts (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
                source_attempt_id TEXT,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, attempt_id),
                UNIQUE(workspace_id, project_id, run_id, attempt_number),
                FOREIGN KEY(workspace_id, project_id, run_id)
                    REFERENCES ledger_runs(workspace_id, project_id, run_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_metric_points (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                source TEXT NOT NULL,
                step INTEGER NOT NULL CHECK(step >= 0),
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                raw_sha256 TEXT NOT NULL,
                context_json TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, attempt_id, source, step, metric_name),
                FOREIGN KEY(workspace_id, project_id, attempt_id)
                    REFERENCES ledger_run_attempts(workspace_id, project_id, attempt_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_artifacts (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                attempt_id TEXT,
                kind TEXT NOT NULL,
                uri TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                media_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, artifact_id),
                FOREIGN KEY(workspace_id, project_id, run_id)
                    REFERENCES ledger_runs(workspace_id, project_id, run_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, attempt_id)
                    REFERENCES ledger_run_attempts(workspace_id, project_id, attempt_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_evaluation_suites (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                evaluation_suite_id TEXT NOT NULL,
                name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                dataset_version_id TEXT,
                metric_contract_json TEXT NOT NULL,
                code_digest TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, evaluation_suite_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, dataset_version_id)
                    REFERENCES ledger_dataset_versions(workspace_id, project_id, dataset_version_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_evaluation_results (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                evaluation_result_id TEXT NOT NULL,
                evaluation_suite_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                attempt_id TEXT,
                model_version_id TEXT,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                slice_metrics_json TEXT NOT NULL DEFAULT '{}',
                artifact_id TEXT,
                compared_to_result_id TEXT,
                identity_digest TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, evaluation_result_id),
                FOREIGN KEY(workspace_id, project_id, evaluation_suite_id)
                    REFERENCES ledger_evaluation_suites(workspace_id, project_id, evaluation_suite_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, run_id)
                    REFERENCES ledger_runs(workspace_id, project_id, run_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, project_id, artifact_id)
                    REFERENCES ledger_artifacts(workspace_id, project_id, artifact_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_decisions (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                run_id TEXT,
                decision_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                rationale TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                actor_id TEXT NOT NULL,
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, project_id, decision_id),
                FOREIGN KEY(workspace_id, project_id, experiment_id)
                    REFERENCES ledger_experiments(workspace_id, project_id, experiment_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE ledger_events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                experiment_id TEXT,
                run_id TEXT,
                attempt_id TEXT,
                event_type TEXT NOT NULL,
                source_system TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                correlation_id TEXT NOT NULL,
                identity_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(workspace_id, source_system, source_event_id),
                FOREIGN KEY(workspace_id, project_id)
                    REFERENCES ledger_projects(workspace_id, project_id) ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_ledger_projects_status ON ledger_projects(workspace_id, status, updated_at)",
            "CREATE INDEX idx_ledger_experiments_project ON ledger_experiments(workspace_id, project_id, status, updated_at)",
            "CREATE INDEX idx_ledger_runs_project ON ledger_runs(workspace_id, project_id, status, updated_at)",
            "CREATE INDEX idx_ledger_runs_experiment ON ledger_runs(workspace_id, project_id, experiment_id, queued_at)",
            "CREATE INDEX idx_ledger_attempts_run ON ledger_run_attempts(workspace_id, project_id, run_id, attempt_number)",
            "CREATE INDEX idx_ledger_metric_series ON ledger_metric_points(workspace_id, project_id, run_id, metric_name, step)",
            "CREATE INDEX idx_ledger_eval_runs ON ledger_evaluation_results(workspace_id, project_id, run_id, created_at)",
            "CREATE INDEX idx_ledger_events_cursor ON ledger_events(workspace_id, project_id, cursor)",
            "CREATE INDEX idx_ledger_decisions_project ON ledger_decisions(workspace_id, project_id, experiment_id, created_at)",
        ),
    ),
    (
        9,
        "autoresearch_code_lineage",
        (
            """
            CREATE TABLE campaign_code_lineages (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                state TEXT NOT NULL,
                record_json TEXT NOT NULL,
                record_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, proposal_id),
                UNIQUE(workspace_id, lineage_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, proposal_id)
                    REFERENCES campaign_proposals(workspace_id, proposal_id) ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_campaign_code_lineages_campaign ON campaign_code_lineages(workspace_id, campaign_id, state, created_at)",
        ),
    ),
    (
        10,
        "control_room_cache_revisions",
        (
            "ALTER TABLE campaign_actor_credentials ADD COLUMN authorization_revision INTEGER NOT NULL DEFAULT 1 CHECK(authorization_revision >= 1)",
            "ALTER TABLE campaign_scheduler_leases ADD COLUMN controller_observation_version INTEGER NOT NULL DEFAULT 1 CHECK(controller_observation_version >= 1)",
        ),
    ),
)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CampaignRepository:
    """Workspace-scoped durable repository with optimistic mutations."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._initialized = False

    @contextmanager
    def _connection(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Apply checksum-pinned migrations atomically and enable SQLite guards."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS campaign_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
        for version, name, statements in MIGRATIONS:
            checksum = hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()
            with self._connection(immediate=True) as connection:
                existing = connection.execute(
                    "SELECT name, checksum FROM campaign_schema_migrations WHERE version = ?",
                    (version,),
                ).fetchone()
                if existing is not None:
                    if existing["name"] != name or existing["checksum"] != checksum:
                        raise MigrationChecksumError(f"migration {version} checksum changed")
                    continue
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO campaign_schema_migrations(version, name, checksum, applied_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (version, name, checksum, _iso(utc_now())),
                )
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise CampaignPersistenceError("campaign repository is not initialized")

    def schema_versions(self) -> list[int]:
        self._require_initialized()
        with self._connection() as connection:
            return [
                int(row["version"])
                for row in connection.execute(
                    "SELECT version FROM campaign_schema_migrations ORDER BY version"
                )
            ]

    def journal_mode(self) -> str:
        self._require_initialized()
        with self._connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def foreign_keys_enabled(self) -> bool:
        self._require_initialized()
        with self._connection() as connection:
            return bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])

    @staticmethod
    def _code_lineage_from_row(row: sqlite3.Row) -> CodeLineageRecord:
        record = CodeLineageRecord.model_validate_json(row["record_json"])
        if record.record_digest != row["record_digest"] or record.state.value != row["state"]:
            raise CampaignPersistenceError("campaign_code_lineage_record_corrupt")
        return record

    def register_code_lineage_requirement(
        self, value: CodeLineageRecord
    ) -> CodeLineageRecord:
        """Persist one required lineage record with exact replay semantics."""

        self._require_initialized()
        record = CodeLineageRecord.model_validate(value.model_dump(mode="python"))
        if record.state != CodeLineageState.REQUIRED:
            raise ValueError("campaign_code_lineage_requirement_state_invalid")
        with self._connection(immediate=True) as connection:
            proposal = connection.execute(
                """
                SELECT campaign_id FROM campaign_proposals
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (record.workspace_id, record.proposal_id),
            ).fetchone()
            if proposal is None:
                raise RecordNotFoundError("campaign proposal not found")
            if proposal["campaign_id"] != record.campaign_id:
                raise ValueError("campaign_code_lineage_proposal_campaign_mismatch")
            existing = connection.execute(
                """
                SELECT * FROM campaign_code_lineages
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (record.workspace_id, record.proposal_id),
            ).fetchone()
            if existing is not None:
                current = self._code_lineage_from_row(existing)
                if current == record:
                    return current
                raise ValueError("campaign_code_lineage_identity_conflict")
            connection.execute(
                """
                INSERT INTO campaign_code_lineages(
                    workspace_id, campaign_id, proposal_id, lineage_id, state,
                    record_json, record_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.workspace_id,
                    record.campaign_id,
                    record.proposal_id,
                    record.lineage_id,
                    record.state.value,
                    _json(record.model_dump(mode="json")),
                    record.record_digest,
                    _iso(record.created_at),
                    _iso(record.updated_at),
                ),
            )
        return record

    def advance_code_lineage(self, value: CodeLineageRecord) -> CodeLineageRecord:
        """Advance required -> prepared -> captured without mutable evidence."""

        self._require_initialized()
        record = CodeLineageRecord.model_validate(value.model_dump(mode="python"))
        ranks = {
            CodeLineageState.REQUIRED: 0,
            CodeLineageState.PREPARED: 1,
            CodeLineageState.CAPTURED: 2,
        }
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_code_lineages
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (record.workspace_id, record.proposal_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign code lineage not found")
            current = self._code_lineage_from_row(row)
            if current == record:
                return current
            immutable_fields = (
                "lineage_id",
                "workspace_id",
                "campaign_id",
                "proposal_id",
                "mutation_kind",
                "source_repository_profile_id",
                "created_at",
            )
            immutable_changed = any(
                getattr(current, field) != getattr(record, field)
                for field in immutable_fields
            )
            evidence_rewritten = (
                current.state != CodeLineageState.REQUIRED
                and (
                    current.base_commit != record.base_commit
                    or current.branch_name != record.branch_name
                )
            )
            if (
                immutable_changed
                or evidence_rewritten
                or ranks[record.state] != ranks[current.state] + 1
                or record.updated_at < current.updated_at
            ):
                raise ValueError("campaign_code_lineage_transition_invalid")
            connection.execute(
                """
                UPDATE campaign_code_lineages
                SET state = ?, record_json = ?, record_digest = ?, updated_at = ?
                WHERE workspace_id = ? AND proposal_id = ? AND record_digest = ?
                """,
                (
                    record.state.value,
                    _json(record.model_dump(mode="json")),
                    record.record_digest,
                    _iso(record.updated_at),
                    record.workspace_id,
                    record.proposal_id,
                    current.record_digest,
                ),
            )
        return record

    def get_code_lineage(self, workspace_id: str, proposal_id: str) -> CodeLineageRecord:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_code_lineages
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (workspace_id, proposal_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign code lineage not found")
        return self._code_lineage_from_row(row)

    def list_code_lineages(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[CodeLineageRecord, ...]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM campaign_code_lineages
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, lineage_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(self._code_lineage_from_row(row) for row in rows)

    @staticmethod
    def _campaign_from_row(row: sqlite3.Row) -> Campaign:
        return Campaign(
            campaign_id=row["campaign_id"],
            workspace_id=row["workspace_id"],
            title=row["title"],
            kind=row["kind"],
            objective=row["objective"],
            target_model=json.loads(row["target_model_json"]),
            owner_actor_id=row["owner_actor_id"],
            manifest_revision=row["manifest_revision"],
            status=row["status"],
            prior_scheduling_status=row["prior_scheduling_status"],
            active_study_id=row["active_study_id"],
            active_action_id=row["active_action_id"],
            champion_ref=row["champion_ref"],
            best_development_candidate_ref=row["best_development_candidate_ref"],
            stop_reason=row["stop_reason"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> CampaignEvent:
        return CampaignEvent(
            event_id=row["event_id"],
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            sequence=row["sequence"],
            aggregate_version=row["aggregate_version"],
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            actor_id=row["actor_id"],
            credential_kind=row["credential_kind"],
            correlation_id=row["correlation_id"],
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _proposal_from_row(row: sqlite3.Row) -> ProposalRecord:
        proposal = StudyProposal.model_validate_json(row["proposal_json"]).model_copy(
            update={"status": ProposalStatus(row["status"])}
        )
        raw_validation = json.loads(row["validation_json"] or "{}")
        if raw_validation:
            validation = ProposalValidation.model_validate(raw_validation)
        elif proposal.status == ProposalStatus.REJECTED:
            validation = ProposalValidation(
                valid=False, reason_codes=("proposal_legacy_rejection",)
            )
        else:
            validation = ProposalValidation(valid=True)
        return ProposalRecord(
            proposal=proposal,
            validation=validation,
            study_id=row["study_id"],
            updated_at=row["updated_at"] or row["created_at"],
        )

    @staticmethod
    def _study_from_row(row: sqlite3.Row) -> Study:
        return Study(
            study_id=row["study_id"],
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            proposal_id=row["proposal_id"],
            status=row["status"],
            stage_plan=json.loads(row["stage_plan_json"]),
            current_stage_index=row["current_stage_index"],
            candidate_digest=row["candidate_digest"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _replay(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        actor_id: str,
        mutation_kind: str,
        idempotency_key: str,
        request_hash: str,
    ) -> CampaignMutation | None:
        row = connection.execute(
            """
            SELECT * FROM campaign_mutations
            WHERE workspace_id = ? AND actor_id = ? AND mutation_kind = ? AND idempotency_key = ?
            """,
            (workspace_id, actor_id, mutation_kind, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflictError()
        campaign_row = connection.execute(
            "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
            (workspace_id, row["campaign_id"]),
        ).fetchone()
        event_row = connection.execute(
            "SELECT * FROM campaign_events WHERE event_id = ?",
            (row["event_id"],),
        ).fetchone()
        if campaign_row is None or event_row is None:
            raise CampaignPersistenceError("idempotency record references missing evidence")
        response = Campaign.model_validate_json(row["response_json"])
        return CampaignMutation(response, self._event_from_row(event_row), replayed=True)

    def _replay_proposal(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        actor_id: str,
        mutation_kind: str,
        idempotency_key: str,
        request_hash: str,
        selection: bool = False,
    ) -> ProposalMutation | ProposalSelection | None:
        row = connection.execute(
            """
            SELECT * FROM campaign_mutations
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
            raise CampaignPersistenceError("idempotency record references missing evidence")
        payload = json.loads(row["response_json"])
        campaign = Campaign.model_validate(payload["campaign"])
        record = ProposalRecord.model_validate(payload["record"])
        event = self._event_from_row(event_row)
        if selection:
            return ProposalSelection(
                campaign=campaign,
                event=event,
                record=record,
                study=Study.model_validate(payload["study"]),
                replayed=True,
            )
        return ProposalMutation(campaign, event, record, replayed=True)

    def _replay_operation(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        actor_id: str,
        mutation_kind: str,
        idempotency_key: str,
        request_hash: str,
    ) -> OperationMutation | None:
        row = connection.execute(
            """
            SELECT * FROM campaign_mutations
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
            raise CampaignPersistenceError("idempotency record references missing evidence")
        payload = json.loads(row["response_json"])
        return OperationMutation(
            campaign=Campaign.model_validate(payload["campaign"]),
            event=self._event_from_row(event_row),
            details=dict(payload.get("details", {})),
            replayed=True,
        )

    @staticmethod
    def _next_event_sequence(
        connection: sqlite3.Connection, workspace_id: str, campaign_id: str
    ) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM campaign_events WHERE workspace_id = ? AND campaign_id = ?
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        return int(row["next_sequence"])

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, event: CampaignEvent) -> None:
        connection.execute(
            """
            INSERT INTO campaign_events(
                event_id, workspace_id, campaign_id, sequence, aggregate_version,
                event_type, payload_json, actor_id, credential_kind, correlation_id,
                idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.workspace_id,
                event.campaign_id,
                event.sequence,
                event.aggregate_version,
                event.event_type,
                _json(event.payload),
                event.actor_id,
                event.credential_kind.value,
                event.correlation_id,
                event.idempotency_key,
                _iso(event.created_at),
            ),
        )

    @staticmethod
    def _insert_mutation(
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        actor_id: str,
        mutation_kind: str,
        idempotency_key: str,
        request_hash: str,
        campaign: Campaign,
        event: CampaignEvent,
        response_json: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO campaign_mutations(
                workspace_id, actor_id, mutation_kind, idempotency_key, request_hash,
                campaign_id, event_id, response_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                actor_id,
                mutation_kind,
                idempotency_key,
                request_hash,
                campaign.campaign_id,
                event.event_id,
                response_json or campaign.model_dump_json(),
                _iso(event.created_at),
            ),
        )

    def create_campaign(
        self,
        campaign: Campaign,
        manifest_revision: ManifestRevision,
        *,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> CampaignMutation:
        """Create the aggregate, immutable manifest, event, and replay record atomically."""

        self._require_initialized()
        if len(manifest_revision.manifest.budget_limits) > 64:
            raise CampaignBudgetResourceLimitError()
        if campaign.status != CampaignStatus.DRAFT or campaign.version != 1:
            raise ValueError("new campaigns must start at draft version 1")
        if (
            manifest_revision.workspace_id != campaign.workspace_id
            or manifest_revision.campaign_id != campaign.campaign_id
            or manifest_revision.revision != campaign.manifest_revision
        ):
            raise ValueError("manifest revision identity must match campaign")
        request_hash = canonical_hash(
            {
                "campaign": campaign.model_dump(mode="json", exclude={"created_at", "updated_at"}),
                "manifest": manifest_revision.manifest.model_dump(mode="json"),
            }
        )
        mutation_kind = "campaign.create"
        with self._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=campaign.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            existing = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (campaign.workspace_id, campaign.campaign_id),
            ).fetchone()
            if existing is not None:
                raise RecordAlreadyExistsError("campaign already exists")
            connection.execute(
                """
                INSERT INTO campaigns(
                    workspace_id, campaign_id, title, kind, objective, target_model_json,
                    owner_actor_id, manifest_revision, status, prior_scheduling_status,
                    active_study_id, active_action_id, champion_ref,
                    best_development_candidate_ref, stop_reason, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign.workspace_id,
                    campaign.campaign_id,
                    campaign.title,
                    campaign.kind.value,
                    campaign.objective,
                    _json(campaign.target_model.model_dump(mode="json")),
                    campaign.owner_actor_id,
                    campaign.manifest_revision,
                    campaign.status.value,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    campaign.version,
                    _iso(campaign.created_at),
                    _iso(campaign.updated_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_manifest_revisions(
                    workspace_id, campaign_id, revision, manifest_json, manifest_hash,
                    actor_id, correlation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest_revision.workspace_id,
                    manifest_revision.campaign_id,
                    manifest_revision.revision,
                    _json(manifest_revision.manifest.model_dump(mode="json")),
                    manifest_revision.manifest_hash,
                    manifest_revision.actor_id,
                    manifest_revision.correlation_id,
                    _iso(manifest_revision.created_at),
                ),
            )
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{campaign.campaign_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=campaign.workspace_id,
                campaign_id=campaign.campaign_id,
                sequence=1,
                aggregate_version=1,
                event_type="campaign:created",
                payload={"manifest_hash": manifest_revision.manifest_hash},
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=campaign.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=campaign,
                event=event,
            )
        return CampaignMutation(campaign, event)

    def get_campaign(self, workspace_id: str, campaign_id: str) -> Campaign:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign not found")
        return self._campaign_from_row(row)

    def read_control_room_snapshot(
        self, workspace_id: str, campaign_id: str
    ) -> CampaignControlRoomStateV1:
        """Read the bounded durable control-room state in one SQLite transaction."""

        self._require_initialized()
        from bashgym.campaigns.control_room import read_control_room_state

        with self._connection() as connection:
            connection.execute("BEGIN")
            return read_control_room_state(connection, workspace_id, campaign_id)

    def read_control_room_projection(
        self, workspace_id: str, campaign_id: str, *, preview_limit: int = 10
    ) -> ControlRoomDurableProjection:
        """Read every durable control-room input in one explicit read transaction."""

        self._require_initialized()
        from bashgym.campaigns.control_room import read_control_room_projection

        with self._connection() as connection:
            connection.execute("BEGIN")
            return read_control_room_projection(
                connection,
                workspace_id,
                campaign_id,
                preview_limit=preview_limit,
            )

    def list_campaigns(self, workspace_id: str) -> list[Campaign]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? ORDER BY created_at, campaign_id",
                (workspace_id,),
            ).fetchall()
        return [self._campaign_from_row(row) for row in rows]

    def get_manifest_revision(
        self, workspace_id: str, campaign_id: str, revision: int
    ) -> ManifestRevision:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, revision),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign manifest revision not found")
        return ManifestRevision(
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            revision=row["revision"],
            manifest=CampaignManifest.model_validate_json(row["manifest_json"]),
            manifest_hash=row["manifest_hash"],
            actor_id=row["actor_id"],
            correlation_id=row["correlation_id"],
            created_at=row["created_at"],
        )

    def study_ids(self, workspace_id: str, campaign_id: str) -> frozenset[str]:
        """Return same-campaign study prerequisites without leaking other workspaces."""

        self._require_initialized()
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT study_id FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ?
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return frozenset(str(row["study_id"]) for row in rows)

    def list_studies(self, workspace_id: str, campaign_id: str) -> tuple[Study, ...]:
        self._require_initialized()
        with self._connection() as connection:
            if connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone() is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, study_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(self._study_from_row(row) for row in rows)

    def get_study(self, workspace_id: str, campaign_id: str, study_id: str) -> Study:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (workspace_id, campaign_id, study_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign study not found")
        return self._study_from_row(row)

    def get_proposal(self, workspace_id: str, campaign_id: str, proposal_id: str) -> ProposalRecord:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (workspace_id, campaign_id, proposal_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign proposal not found")
        return self._proposal_from_row(row)

    def list_proposals(self, workspace_id: str, campaign_id: str) -> tuple[ProposalRecord, ...]:
        self._require_initialized()
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT * FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY creation_sequence, proposal_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(self._proposal_from_row(row) for row in rows)

    def submit_proposal(
        self,
        submission: StudyProposalSubmission,
        validation: ProposalValidation,
        *,
        normalized_priority: int,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        """Persist a submitted or policy-rejected proposal atomically."""

        self._require_initialized()
        mutation_kind = "campaign.proposal.submit"
        request_hash = canonical_hash(
            {
                "submission": submission.model_dump(mode="json"),
                "normalized_priority": normalized_priority,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_proposal(
                connection,
                workspace_id=submission.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                assert isinstance(replay, ProposalMutation)
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (submission.workspace_id, submission.campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(campaign_row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.status not in {
                CampaignStatus.ACTIVE,
                CampaignStatus.PAUSED,
                CampaignStatus.AWAITING_AUTHORITY,
            }:
                raise InvalidProposalTransitionError(
                    f"cannot submit while campaign is {current.status.value}"
                )
            existing = connection.execute(
                "SELECT 1 FROM campaign_proposals WHERE workspace_id = ? AND proposal_id = ?",
                (submission.workspace_id, submission.proposal_id),
            ).fetchone()
            if existing is not None:
                raise RecordAlreadyExistsError("campaign proposal already exists")
            sequence_row = connection.execute(
                """
                SELECT COALESCE(MAX(creation_sequence), 0) + 1 AS next_sequence
                FROM campaign_proposals WHERE workspace_id = ? AND campaign_id = ?
                """,
                (submission.workspace_id, submission.campaign_id),
            ).fetchone()
            sequence = int(sequence_row["next_sequence"])
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (
                    submission.workspace_id,
                    submission.campaign_id,
                    current.manifest_revision,
                ),
            ).fetchone()
            if manifest_row is None:
                raise CampaignPersistenceError("campaign manifest revision missing")
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            if sequence > manifest.max_proposal_rounds and validation.valid:
                validation = ProposalValidation(
                    valid=False, reason_codes=("proposal_round_limit_exceeded",)
                )
            now = utc_now()
            proposal_status = (
                ProposalStatus.SUBMITTED if validation.valid else ProposalStatus.REJECTED
            )
            proposal = StudyProposal(
                **submission.model_dump(exclude={"schema_version", "priority"}),
                priority=normalized_priority,
                planner_actor_id=actor_id,
                status=proposal_status,
                creation_sequence=sequence,
                created_at=now,
            )
            record = ProposalRecord(proposal=proposal, validation=validation, updated_at=now)
            connection.execute(
                """
                INSERT INTO campaign_proposals(
                    workspace_id, campaign_id, proposal_id, status, priority,
                    estimated_cost, creation_sequence, proposal_json, created_at,
                    validation_json, study_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    submission.workspace_id,
                    submission.campaign_id,
                    submission.proposal_id,
                    proposal.status.value,
                    proposal.priority,
                    proposal.estimated_cost,
                    proposal.creation_sequence,
                    proposal.model_dump_json(),
                    _iso(now),
                    validation.model_dump_json(),
                    _iso(now),
                ),
            )
            updated = current.model_copy(update={"version": current.version + 1, "updated_at": now})
            cursor = connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.version,
                    _iso(now),
                    submission.workspace_id,
                    submission.campaign_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError(expected_version, current.version)
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{submission.proposal_id}:submit:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=submission.workspace_id,
                campaign_id=submission.campaign_id,
                sequence=self._next_event_sequence(
                    connection, submission.workspace_id, submission.campaign_id
                ),
                aggregate_version=updated.version,
                event_type=(
                    "campaign:proposal-submitted"
                    if validation.valid
                    else "campaign:proposal-rejected"
                ),
                payload={
                    "proposal_id": proposal.proposal_id,
                    "status": proposal.status.value,
                    "priority": proposal.priority,
                    "estimated_cost": proposal.estimated_cost,
                    "reason_codes": list(validation.reason_codes),
                },
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            self._insert_event(connection, event)
            response_json = _json(
                {
                    "campaign": updated.model_dump(mode="json"),
                    "record": record.model_dump(mode="json"),
                }
            )
            self._insert_mutation(
                connection,
                workspace_id=submission.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=response_json,
            )
        return ProposalMutation(updated, event, record)

    def withdraw_proposal(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        """Withdraw only a still-submitted proposal with exact replay semantics."""

        self._require_initialized()
        mutation_kind = "campaign.proposal.withdraw"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "proposal_id": proposal_id,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_proposal(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                assert isinstance(replay, ProposalMutation)
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(campaign_row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            proposal_row = connection.execute(
                """
                SELECT * FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (workspace_id, campaign_id, proposal_id),
            ).fetchone()
            if proposal_row is None:
                raise RecordNotFoundError("campaign proposal not found")
            record = self._proposal_from_row(proposal_row)
            if record.proposal.status != ProposalStatus.SUBMITTED:
                raise InvalidProposalTransitionError(
                    f"cannot withdraw proposal in {record.proposal.status.value}"
                )
            now = utc_now()
            withdrawn = record.proposal.model_copy(update={"status": ProposalStatus.WITHDRAWN})
            updated_record = record.model_copy(update={"proposal": withdrawn, "updated_at": now})
            connection.execute(
                """
                UPDATE campaign_proposals SET status = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (
                    ProposalStatus.WITHDRAWN.value,
                    _iso(now),
                    workspace_id,
                    campaign_id,
                    proposal_id,
                ),
            )
            updated = current.model_copy(update={"version": current.version + 1, "updated_at": now})
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.version,
                    _iso(now),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{proposal_id}:withdraw:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:proposal-withdrawn",
                payload={"proposal_id": proposal_id, "status": ProposalStatus.WITHDRAWN.value},
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {
                        "campaign": updated.model_dump(mode="json"),
                        "record": updated_record.model_dump(mode="json"),
                    }
                ),
            )
        return ProposalMutation(updated, event, updated_record)

    def request_advance(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> CampaignMutation:
        """Persist a controller wake-up request without selecting or accepting work."""

        self._require_initialized()
        mutation_kind = "campaign.advance.request"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.status not in {
                CampaignStatus.ACTIVE,
                CampaignStatus.PAUSED,
                CampaignStatus.AWAITING_AUTHORITY,
            }:
                raise InvalidProposalTransitionError(
                    f"cannot request advance while campaign is {current.status.value}"
                )
            now = utc_now()
            updated = current.model_copy(update={"version": current.version + 1, "updated_at": now})
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (updated.version, _iso(now), workspace_id, campaign_id, expected_version),
            )
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{campaign_id}:advance:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:advance-requested",
                payload={"requested_by": actor_id},
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
            )
        return CampaignMutation(updated, event)

    def select_next_proposal_as_controller(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        controller_id: str,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalSelection | None:
        """Select the deterministic ready proposal; this is not an actor API boundary."""

        self._require_initialized()
        mutation_kind = "campaign.controller.select_proposal"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_proposal(
                connection,
                workspace_id=workspace_id,
                actor_id=controller_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                selection=True,
            )
            if replay is not None:
                assert isinstance(replay, ProposalSelection)
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(campaign_row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.status != CampaignStatus.ACTIVE:
                raise InvalidProposalTransitionError(
                    f"controller cannot select while campaign is {current.status.value}"
                )
            if current.active_study_id is not None or current.active_action_id is not None:
                raise InvalidProposalTransitionError("campaign already has active work")
            rows = connection.execute(
                """
                SELECT * FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? AND status = ?
                ORDER BY priority DESC, estimated_cost ASC, creation_sequence ASC, proposal_id ASC
                """,
                (workspace_id, campaign_id, ProposalStatus.SUBMITTED.value),
            ).fetchall()
            selected: ProposalRecord | None = None
            terminal_studies = {
                StudyStatus.COMPLETED.value,
                StudyStatus.DEVELOPMENT_PASSED.value,
                StudyStatus.REJECTED.value,
                StudyStatus.PROMOTED.value,
                StudyStatus.FINAL_REJECTED.value,
                StudyStatus.EXECUTION_FAILED.value,
                StudyStatus.ABANDONED.value,
                StudyStatus.CANCELLED.value,
            }
            for row in rows:
                candidate = self._proposal_from_row(row)
                prerequisite_ids = candidate.proposal.prerequisite_study_ids
                if prerequisite_ids:
                    placeholders = ",".join("?" for _ in prerequisite_ids)
                    prerequisite_rows = connection.execute(
                        f"""
                        SELECT study_id, status FROM campaign_studies
                        WHERE workspace_id = ? AND campaign_id = ?
                          AND study_id IN ({placeholders})
                        """,
                        (workspace_id, campaign_id, *prerequisite_ids),
                    ).fetchall()
                    states = {
                        str(item["study_id"]): str(item["status"]) for item in prerequisite_rows
                    }
                    if any(
                        states.get(study_id) not in terminal_studies
                        for study_id in prerequisite_ids
                    ):
                        continue
                selected = candidate
                break
            if selected is None:
                return None
            now = utc_now()
            proposal = selected.proposal
            candidate_digest = canonical_hash(
                {
                    "proposal_id": proposal.proposal_id,
                    "dataset_recipe": proposal.dataset_recipe,
                    "training_recipe": proposal.training_recipe,
                    "evaluation_recipe": proposal.evaluation_recipe,
                    "stage_plan": proposal.stage_plan.model_dump(mode="json"),
                }
            )
            study_id = f"study-{canonical_hash({'campaign_id': campaign_id, 'proposal_id': proposal.proposal_id})[:24]}"
            study = Study(
                study_id=study_id,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                proposal_id=proposal.proposal_id,
                status=StudyStatus.VALIDATED,
                stage_plan=proposal.stage_plan,
                candidate_digest=candidate_digest,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO campaign_studies(
                    workspace_id, campaign_id, study_id, proposal_id, status,
                    current_stage_index, stage_plan_json, version, created_at,
                    updated_at, candidate_digest
                ) VALUES (?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    study.study_id,
                    proposal.proposal_id,
                    study.status.value,
                    _json(study.stage_plan.model_dump(mode="json")),
                    _iso(now),
                    _iso(now),
                    candidate_digest,
                ),
            )
            accepted = proposal.model_copy(update={"status": ProposalStatus.ACCEPTED})
            accepted_record = selected.model_copy(
                update={"proposal": accepted, "study_id": study_id, "updated_at": now}
            )
            connection.execute(
                """
                UPDATE campaign_proposals SET status = ?, study_id = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ? AND status = ?
                """,
                (
                    ProposalStatus.ACCEPTED.value,
                    study_id,
                    _iso(now),
                    workspace_id,
                    campaign_id,
                    proposal.proposal_id,
                    ProposalStatus.SUBMITTED.value,
                ),
            )
            updated = current.model_copy(
                update={
                    "active_study_id": study_id,
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            cursor = connection.execute(
                """
                UPDATE campaigns SET active_study_id = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                  AND active_study_id IS NULL AND active_action_id IS NULL
                """,
                (
                    study_id,
                    updated.version,
                    _iso(now),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError(expected_version, current.version)
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{proposal.proposal_id}:accept:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:proposal-accepted",
                payload={"proposal_id": proposal.proposal_id, "study_id": study_id},
                actor_id=controller_id,
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=controller_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {
                        "campaign": updated.model_dump(mode="json"),
                        "record": accepted_record.model_dump(mode="json"),
                        "study": study.model_dump(mode="json"),
                    }
                ),
            )
        return ProposalSelection(updated, event, accepted_record, study)

    @staticmethod
    def _safe_evidence_reference(value: str | None) -> str | None:
        if value is None:
            return None
        if value.replace("-", "").replace("_", "").replace(".", "").isalnum():
            return value
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"

    def build_evidence_snapshot(
        self, workspace_id: str, campaign_id: str
    ) -> CampaignEvidenceSnapshot:
        """Build bounded, secret-free planner context from durable summary rows only."""

        self._require_initialized()
        with self._connection() as connection:
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            campaign = self._campaign_from_row(campaign_row)
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, campaign.manifest_revision),
            ).fetchone()
            if manifest_row is None:
                raise CampaignPersistenceError("campaign manifest revision missing")
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            budget_rows = connection.execute(
                """
                SELECT unit, COALESCE(SUM(reserved_delta), 0) AS reserved,
                       COALESCE(SUM(actual_delta), 0) AS actual,
                       COALESCE(SUM(limit_delta), 0) AS limit_delta
                FROM campaign_budget_ledger
                WHERE workspace_id = ? AND campaign_id = ? GROUP BY unit
                """,
                (workspace_id, campaign_id),
            ).fetchall()
            totals = {str(row["unit"]): row for row in budget_rows}
            budget_remaining = {
                unit: float(limit)
                + float(totals[unit]["limit_delta"] if unit in totals else 0)
                - float(totals[unit]["reserved"] if unit in totals else 0)
                - float(totals[unit]["actual"] if unit in totals else 0)
                for unit, limit in sorted(manifest.budget_limits.items())
            }
            count_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? GROUP BY status
                """,
                (workspace_id, campaign_id),
            ).fetchall()
            raw_counts = {ProposalStatus(row["status"]): int(row["count"]) for row in count_rows}
            proposal_counts = {status: raw_counts.get(status, 0) for status in ProposalStatus}
            summary_rows = connection.execute(
                """
                SELECT * FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? AND status != ?
                ORDER BY creation_sequence DESC, proposal_id DESC LIMIT 50
                """,
                (workspace_id, campaign_id, ProposalStatus.SUBMITTED.value),
            ).fetchall()
            summaries = tuple(
                CompletedHypothesisSummary(
                    proposal_id=record.proposal.proposal_id,
                    study_id=record.study_id,
                    study_family=record.proposal.study_family,
                    status=record.proposal.status,
                )
                for record in (self._proposal_from_row(row) for row in summary_rows)
            )
            artifact_rows = connection.execute(
                """
                SELECT artifact_id, sha256, size_bytes, schema_name, valid, metadata_json
                FROM campaign_artifacts
                WHERE workspace_id = ? AND campaign_id = ? AND sealed = 1
                ORDER BY created_at DESC, artifact_id DESC LIMIT 100
                """,
                (workspace_id, campaign_id),
            ).fetchall()
            artifacts = tuple(
                CampaignArtifactReference(
                    artifact_id=row["artifact_id"],
                    sha256=row["sha256"],
                    size_bytes=row["size_bytes"],
                    schema_name=row["schema_name"],
                    valid=bool(row["valid"]),
                )
                for row in artifact_rows
            )
            nemo_gym_references = []
            for row in artifact_rows:
                metadata = json.loads(row["metadata_json"])
                raw_reference = metadata.get("nemo_gym")
                if raw_reference is None:
                    continue
                if row["schema_name"] != NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA:
                    raise CampaignPersistenceError(
                        "campaign_nemo_gym_evidence_schema_mismatch"
                    )
                reference = NemoGymEvidenceReference.model_validate(raw_reference)
                if (
                    reference.artifact_id != row["artifact_id"]
                    or reference.artifact_sha256 != row["sha256"]
                    or not bool(row["valid"])
                ):
                    raise CampaignPersistenceError(
                        "campaign_nemo_gym_evidence_reference_mismatch"
                    )
                nemo_gym_references.append(reference)
        return CampaignEvidenceSnapshot(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            campaign_version=campaign.version,
            manifest_revision=campaign.manifest_revision,
            status=campaign.status,
            objective=campaign.objective,
            champion_ref=self._safe_evidence_reference(campaign.champion_ref),
            best_development_candidate_ref=self._safe_evidence_reference(
                campaign.best_development_candidate_ref
            ),
            approved_data_scopes=manifest.approved_data_scopes,
            compute_profile_id=manifest.compute_profile_id,
            budget_remaining=budget_remaining,
            proposal_counts=proposal_counts,
            completed_hypotheses=summaries,
            artifact_references=artifacts,
            nemo_gym_evidence_references=tuple(nemo_gym_references),
            available_executors=("fake", "registered_remote"),
            active_study_id=campaign.active_study_id,
            active_action_id=campaign.active_action_id,
        )

    def revise_manifest(
        self,
        workspace_id: str,
        campaign_id: str,
        manifest: CampaignManifest,
        *,
        reason: str,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Append an immutable manifest revision while no action is in flight."""

        self._require_initialized()
        if len(manifest.budget_limits) > 64:
            raise CampaignBudgetResourceLimitError()
        mutation_kind = "campaign.manifest.revise"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "manifest": manifest.model_dump(mode="json"),
                "reason": reason,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.active_action_id is not None:
                raise CampaignPersistenceError("campaign_manifest_revision_action_in_flight")
            created_at = utc_now()
            revision = ManifestRevision(
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                revision=current.manifest_revision + 1,
                manifest=manifest,
                actor_id=actor_id,
                correlation_id=correlation_id,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO campaign_manifest_revisions(
                    workspace_id, campaign_id, revision, manifest_json, manifest_hash,
                    actor_id, correlation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    revision.revision,
                    _json(manifest.model_dump(mode="json")),
                    revision.manifest_hash,
                    actor_id,
                    correlation_id,
                    _iso(created_at),
                ),
            )
            updated = current.model_copy(
                update={
                    "manifest_revision": revision.revision,
                    "version": current.version + 1,
                    "updated_at": created_at,
                }
            )
            connection.execute(
                """
                UPDATE campaigns SET manifest_revision = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.manifest_revision,
                    updated.version,
                    _iso(created_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {
                "revision": revision.revision,
                "manifest_hash": revision.manifest_hash,
                "reason": reason,
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'revise:{campaign_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:manifest-revised",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=created_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def approve_source(
        self,
        workspace_id: str,
        campaign_id: str,
        source_id: str,
        evidence: dict[str, Any],
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Persist typed source evidence; approval does not silently revise data scope."""

        self._require_initialized()
        evidence_digest = canonical_hash(evidence)
        mutation_kind = "campaign.source.approve"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "source_id": source_id,
                "evidence_digest": evidence_digest,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if connection.execute(
                """
                SELECT 1 FROM campaign_source_approvals
                WHERE workspace_id = ? AND campaign_id = ? AND source_id = ?
                """,
                (workspace_id, campaign_id, source_id),
            ).fetchone() is not None:
                raise RecordAlreadyExistsError("campaign source approval already exists")
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO campaign_source_approvals(
                    workspace_id, campaign_id, source_id, evidence_json,
                    evidence_digest, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    source_id,
                    _json(evidence),
                    evidence_digest,
                    actor_id,
                    _iso(created_at),
                ),
            )
            updated = current.model_copy(
                update={"version": current.version + 1, "updated_at": created_at}
            )
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (updated.version, _iso(created_at), workspace_id, campaign_id, expected_version),
            )
            details = {"source_id": source_id, "evidence_digest": evidence_digest}
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'source:{campaign_id}:{source_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:source-approved",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=created_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def abandon_study(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        *,
        reason: str,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Abandon a nonterminal study without deleting its evidence."""

        self._require_initialized()
        mutation_kind = "campaign.study.abandon"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "study_id": study_id,
                "reason": reason,
                "expected_version": expected_version,
            }
        )
        terminal = {
            StudyStatus.PROMOTED,
            StudyStatus.FINAL_REJECTED,
            StudyStatus.EXECUTION_FAILED,
            StudyStatus.ABANDONED,
            StudyStatus.CANCELLED,
        }
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(campaign_row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            study_row = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (workspace_id, campaign_id, study_id),
            ).fetchone()
            if study_row is None:
                raise RecordNotFoundError("campaign study not found")
            study = self._study_from_row(study_row)
            if study.status in terminal:
                raise InvalidProposalTransitionError("study is already terminal")
            if current.active_action_id is not None and current.active_study_id == study_id:
                raise InvalidProposalTransitionError("study has an active action")
            changed_at = utc_now()
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (
                    StudyStatus.ABANDONED.value,
                    _iso(changed_at),
                    workspace_id,
                    campaign_id,
                    study_id,
                ),
            )
            updated = current.model_copy(
                update={
                    "active_study_id": None
                    if current.active_study_id == study_id
                    else current.active_study_id,
                    "version": current.version + 1,
                    "updated_at": changed_at,
                }
            )
            connection.execute(
                """
                UPDATE campaigns SET active_study_id = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.active_study_id,
                    updated.version,
                    _iso(changed_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {"study_id": study_id, "reason": reason}
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'abandon:{study_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:study-abandoned",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=changed_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def acquire_protected_epoch(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Acquire the one-use protected-set epoch after a passing development gate."""

        self._require_initialized()
        mutation_kind = "campaign.protected.acquire"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.active_action_id is not None:
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, current.manifest_revision),
            ).fetchone()
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            if not manifest.protected_artifact_refs:
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            decision_row = connection.execute(
                """
                SELECT decision_id, decision_json FROM campaign_gate_decisions
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at DESC, decision_id DESC LIMIT 1
                """,
                (workspace_id, campaign_id),
            ).fetchone()
            if decision_row is None:
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            decision = json.loads(decision_row["decision_json"])
            if decision.get("verdict") != "passed" or not decision.get("candidate_digest"):
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            protected_set_hash = canonical_hash(sorted(manifest.protected_artifact_refs))
            candidate_digest = str(decision["candidate_digest"])
            lock_digest = canonical_hash(
                {
                    "campaign_id": campaign_id,
                    "candidate_digest": candidate_digest,
                    "manifest_revision": current.manifest_revision,
                }
            )
            existing = connection.execute(
                """
                SELECT 1 FROM campaign_protected_epochs
                WHERE workspace_id = ? AND target_contract_key = ? AND protected_set_hash = ?
                """,
                (workspace_id, current.target_model.target_contract_key, protected_set_hash),
            ).fetchone()
            if existing is not None:
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            acquired_at = utc_now()
            epoch_id = f"protected-{hashlib.sha256(f'{campaign_id}:{protected_set_hash}:{candidate_digest}'.encode()).hexdigest()[:24]}"
            connection.execute(
                """
                INSERT INTO campaign_protected_epochs(
                    workspace_id, protected_epoch_id, target_contract_key,
                    protected_set_hash, candidate_lock_digest, lease_state,
                    access_count, result_json, created_at, updated_at, campaign_id
                ) VALUES (?, ?, ?, ?, ?, 'acquired', 1, NULL, ?, ?, ?)
                """,
                (
                    workspace_id,
                    epoch_id,
                    current.target_model.target_contract_key,
                    protected_set_hash,
                    lock_digest,
                    _iso(acquired_at),
                    _iso(acquired_at),
                    campaign_id,
                ),
            )
            updated = current.model_copy(
                update={
                    "best_development_candidate_ref": candidate_digest,
                    "version": current.version + 1,
                    "updated_at": acquired_at,
                }
            )
            connection.execute(
                """
                UPDATE campaigns SET best_development_candidate_ref = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    candidate_digest,
                    updated.version,
                    _iso(acquired_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {
                "protected_epoch_id": epoch_id,
                "candidate_lock_digest": lock_digest,
                "development_decision_id": str(decision_row["decision_id"]),
                "access_count": 1,
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'protected:{epoch_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:protected-lease-acquired",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=acquired_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def record_protected_evaluation(
        self,
        workspace_id: str,
        campaign_id: str,
        result: ProtectedEvaluationResult,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Seal one candidate-locked result for an acquired protected epoch."""

        self._require_initialized()
        mutation_kind = "campaign.protected.complete"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "result": result.model_dump(mode="json"),
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            epoch_row = connection.execute(
                """
                SELECT * FROM campaign_protected_epochs
                WHERE workspace_id = ? AND campaign_id = ? AND protected_epoch_id = ?
                """,
                (workspace_id, campaign_id, result.protected_epoch_id),
            ).fetchone()
            expected_lock = canonical_hash(
                {
                    "campaign_id": campaign_id,
                    "candidate_digest": result.candidate_digest,
                    "manifest_revision": current.manifest_revision,
                }
            )
            if (
                epoch_row is None
                or epoch_row["lease_state"] != "acquired"
                or epoch_row["result_json"] is not None
                or epoch_row["candidate_lock_digest"] != expected_lock
                or current.best_development_candidate_ref != result.candidate_digest
            ):
                raise ProtectedLeaseDeniedError("campaign_protected_lease_denied")
            completed_at = utc_now()
            connection.execute(
                """
                UPDATE campaign_protected_epochs
                SET lease_state = 'completed', result_json = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND protected_epoch_id = ?
                  AND lease_state = 'acquired' AND result_json IS NULL
                """,
                (
                    _json(result.model_dump(mode="json")),
                    _iso(completed_at),
                    workspace_id,
                    campaign_id,
                    result.protected_epoch_id,
                ),
            )
            updated = current.model_copy(
                update={"version": current.version + 1, "updated_at": completed_at}
            )
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.version,
                    _iso(completed_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {
                "protected_epoch_id": result.protected_epoch_id,
                "result_digest": result.result_digest,
                "candidate_digest": result.candidate_digest,
                "passed": result.passed,
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'protected-result:{result.protected_epoch_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:protected-evaluation-completed",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=completed_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def promote_candidate(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
        override_reason: str | None = None,
    ) -> OperationMutation:
        """Commit a champion only from passing evidence or an explicit override."""

        self._require_initialized()
        mutation_kind = "campaign.promotion.commit"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "expected_version": expected_version,
                "override_reason": override_reason,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            if current.active_action_id is not None:
                raise PromotionGateFailedError("campaign_gate_failed")
            result = transition_campaign(
                current.status,
                CampaignTrigger.PROMOTION_COMMITTED,
                prior_scheduling_status=current.prior_scheduling_status,
            )
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, current.manifest_revision),
            ).fetchone()
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            decision_row = connection.execute(
                """
                SELECT decision_id, decision_json FROM campaign_gate_decisions
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at DESC, decision_id DESC LIMIT 1
                """,
                (workspace_id, campaign_id),
            ).fetchone()
            decision = json.loads(decision_row["decision_json"]) if decision_row else {}
            candidate_digest = decision.get("candidate_digest")
            gate_passed = decision.get("verdict") == "passed" and bool(candidate_digest)
            protected_passed = not manifest.protected_artifact_refs
            if manifest.protected_artifact_refs and candidate_digest:
                epoch_row = connection.execute(
                    """
                    SELECT result_json, lease_state, candidate_lock_digest
                    FROM campaign_protected_epochs
                    WHERE workspace_id = ? AND campaign_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (workspace_id, campaign_id),
                ).fetchone()
                if epoch_row is not None and epoch_row["result_json"]:
                    protected_result = ProtectedEvaluationResult.model_validate_json(
                        epoch_row["result_json"]
                    )
                    expected_lock = canonical_hash(
                        {
                            "campaign_id": campaign_id,
                            "candidate_digest": candidate_digest,
                            "manifest_revision": current.manifest_revision,
                        }
                    )
                    protected_passed = (
                        epoch_row["lease_state"] == "completed"
                        and epoch_row["candidate_lock_digest"] == expected_lock
                        and protected_result.candidate_digest == candidate_digest
                        and protected_result.passed is True
                    )
            if not candidate_digest:
                candidate_digest = current.best_development_candidate_ref
            promotion_gate = evaluate_promotion_gate(
                active_action_id=current.active_action_id,
                comparison_verdict=decision.get("verdict"),
                candidate_digest=candidate_digest,
                protected_required=bool(manifest.protected_artifact_refs),
                protected_passed=protected_passed,
                human_work_complete=True,
            )
            if not candidate_digest or (not override_reason and not promotion_gate.eligible):
                raise PromotionGateFailedError("campaign_gate_failed")
            promoted_at = utc_now()
            target_key = current.target_model.target_contract_key
            revision_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision
                FROM campaign_champions WHERE workspace_id = ? AND target_contract_key = ?
                """,
                (workspace_id, target_key),
            ).fetchone()
            champion_revision = int(revision_row["next_revision"])
            champion = {
                "schema_version": "campaign_champion.v1",
                "campaign_id": campaign_id,
                "candidate_digest": candidate_digest,
                "development_decision_id": str(decision_row["decision_id"])
                if decision_row
                else None,
                "protected_gate_passed": protected_passed,
                "override": bool(override_reason),
                "override_reason": override_reason,
                "actor_id": actor_id,
                "created_at": _iso(promoted_at),
            }
            connection.execute(
                """
                INSERT INTO campaign_champions(
                    workspace_id, target_contract_key, revision, champion_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, target_key, champion_revision, _json(champion), _iso(promoted_at)),
            )
            champion_ref = f"champion:{target_key}:{champion_revision}:{candidate_digest}"
            updated = current.model_copy(
                update={
                    "status": result.status,
                    "prior_scheduling_status": result.prior_scheduling_status,
                    "champion_ref": champion_ref,
                    "best_development_candidate_ref": candidate_digest,
                    "version": current.version + 1,
                    "updated_at": promoted_at,
                }
            )
            connection.execute(
                """
                UPDATE campaigns SET status = ?, prior_scheduling_status = ?, champion_ref = ?,
                    best_development_candidate_ref = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.status.value,
                    None,
                    champion_ref,
                    candidate_digest,
                    updated.version,
                    _iso(promoted_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {
                "champion_ref": champion_ref,
                "candidate_digest": candidate_digest,
                "development_gate_passed": gate_passed,
                "protected_gate_passed": protected_passed,
                "override": bool(override_reason),
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'promote:{campaign_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:promotion-committed",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=promoted_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def record_export(
        self,
        workspace_id: str,
        campaign_id: str,
        export_id: str,
        formats: tuple[str, ...],
        export_manifest: dict[str, Any],
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Record a completed server-managed export without exposing its local path."""

        self._require_initialized()
        mutation_kind = "campaign.export.record"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "export_id": export_id,
                "formats": formats,
                "export_manifest": export_manifest,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO campaign_exports(
                    workspace_id, campaign_id, export_id, formats_json,
                    export_manifest_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    export_id,
                    _json(formats),
                    _json(export_manifest),
                    actor_id,
                    _iso(created_at),
                ),
            )
            updated = current.model_copy(
                update={"version": current.version + 1, "updated_at": created_at}
            )
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (updated.version, _iso(created_at), workspace_id, campaign_id, expected_version),
            )
            details = {
                "export_id": export_id,
                "formats": list(formats),
                "source_digest": export_manifest.get("source_digest"),
                "files": export_manifest.get("files", []),
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'export:{export_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:export-completed",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=created_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def transition_campaign(
        self,
        workspace_id: str,
        campaign_id: str,
        trigger: CampaignTrigger,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        stop_reason: str | None = None,
    ) -> CampaignMutation:
        """Apply one explicit transition with CAS, event, and idempotent replay."""

        self._require_initialized()
        safe_payload = payload or {}
        mutation_kind = f"campaign.transition.{trigger.value}"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "trigger": trigger.value,
                "expected_version": expected_version,
                "payload": safe_payload,
                "stop_reason": stop_reason,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            result = transition_campaign(
                current.status,
                trigger,
                prior_scheduling_status=current.prior_scheduling_status,
            )
            now = utc_now()
            updated = current.model_copy(
                update={
                    "status": result.status,
                    "prior_scheduling_status": result.prior_scheduling_status,
                    "stop_reason": stop_reason if stop_reason is not None else current.stop_reason,
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            cursor = connection.execute(
                """
                UPDATE campaigns
                SET status = ?, prior_scheduling_status = ?, stop_reason = ?,
                    version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.status.value,
                    updated.prior_scheduling_status.value
                    if updated.prior_scheduling_status is not None
                    else None,
                    updated.stop_reason,
                    updated.version,
                    _iso(updated.updated_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                fresh = connection.execute(
                    "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                    (workspace_id, campaign_id),
                ).fetchone()
                raise RevisionConflictError(expected_version, int(fresh["version"]))
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{campaign_id}:{trigger.value}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type=result.event_type,
                payload={"trigger": trigger.value, **safe_payload},
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
            )
        return CampaignMutation(updated, event)

    def list_events(
        self, workspace_id: str, campaign_id: str, *, after_cursor: int = 0, limit: int = 200
    ) -> list[tuple[int, CampaignEvent]]:
        self._require_initialized()
        limit = max(1, min(limit, 1000))
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT * FROM campaign_events
                WHERE workspace_id = ? AND campaign_id = ? AND cursor > ?
                ORDER BY cursor LIMIT ?
                """,
                (workspace_id, campaign_id, after_cursor, limit),
            ).fetchall()
        return [(int(row["cursor"]), self._event_from_row(row)) for row in rows]

    def list_recent_events(
        self, workspace_id: str, campaign_id: str, *, limit: int = 50
    ) -> list[tuple[int, CampaignEvent]]:
        """Return the newest durable events in chronological order.

        This is the bounded read used by workspace/agent context. Cursor-based
        consumers should continue to use :meth:`list_events`.
        """

        self._require_initialized()
        limit = max(1, min(limit, 200))
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT * FROM campaign_events
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY cursor DESC LIMIT ?
                """,
                (workspace_id, campaign_id, limit),
            ).fetchall()
        rows.reverse()
        return [(int(row["cursor"]), self._event_from_row(row)) for row in rows]

    def list_exports(self, workspace_id: str, campaign_id: str, *, limit: int = 10):
        """Return recent path-free export manifests for agent/report discovery."""

        self._require_initialized()
        limit = max(1, min(limit, 50))
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT export_id, formats_json, export_manifest_json, actor_id, created_at
                FROM campaign_exports
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at DESC, export_id DESC LIMIT ?
                """,
                (workspace_id, campaign_id, limit),
            ).fetchall()
        return [
            {
                "export_id": row["export_id"],
                "formats": json.loads(row["formats_json"]),
                "manifest": json.loads(row["export_manifest_json"]),
                "actor_id": row["actor_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def budget_totals(self, workspace_id: str, campaign_id: str, unit: str) -> dict[str, float]:
        """Return append-only ledger totals without trusting cached counters."""

        self._require_initialized()
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            row = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_delta), 0) AS reserved,
                       COALESCE(SUM(actual_delta), 0) AS actual,
                       COALESCE(SUM(limit_delta), 0) AS limit_delta
                FROM campaign_budget_ledger
                WHERE workspace_id = ? AND campaign_id = ? AND unit = ?
                """,
                (workspace_id, campaign_id, unit),
            ).fetchone()
        return {name: float(row[name]) for name in ("reserved", "actual", "limit_delta")}

    @staticmethod
    def _budget_entry_from_row(row: sqlite3.Row) -> BudgetLedgerEntry:
        return BudgetLedgerEntry(
            entry_id=row["entry_id"],
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            unit=row["unit"],
            kind=row["entry_kind"],
            reserved_delta=row["reserved_delta"],
            actual_delta=row["actual_delta"],
            limit_delta=row["limit_delta"],
            action_id=row["action_id"],
            evidence=json.loads(row["evidence_json"]),
            actor_id=row["actor_id"],
            created_at=row["created_at"],
        )

    def record_budget_entry(
        self,
        entry: BudgetLedgerEntry,
        *,
        expected_version: int,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> BudgetMutation:
        """Append budget evidence and update aggregate/event atomically."""

        self._require_initialized()
        mutation_kind = f"campaign.budget.{entry.kind.value}"
        request_hash = canonical_hash(
            {
                "entry": entry.model_dump(mode="json", exclude={"created_at"}),
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay(
                connection,
                workspace_id=entry.workspace_id,
                actor_id=entry.actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                row = connection.execute(
                    """
                    SELECT * FROM campaign_budget_ledger
                    WHERE workspace_id = ? AND entry_id = ?
                    """,
                    (entry.workspace_id, entry.entry_id),
                ).fetchone()
                if row is None:
                    raise CampaignPersistenceError("budget replay references missing entry")
                return BudgetMutation(
                    replay.campaign,
                    replay.event,
                    self._budget_entry_from_row(row),
                    replayed=True,
                )
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (entry.workspace_id, entry.campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            current = self._campaign_from_row(campaign_row)
            if current.version != expected_version:
                raise RevisionConflictError(expected_version, current.version)
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (entry.workspace_id, entry.campaign_id, current.manifest_revision),
            ).fetchone()
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            base_limit = manifest.budget_limits.get(entry.unit)
            if base_limit is None:
                raise BudgetInvariantError(
                    f"{BudgetInvariantError.code}: unknown unit {entry.unit}"
                )
            totals = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_delta), 0) AS reserved,
                       COALESCE(SUM(actual_delta), 0) AS actual,
                       COALESCE(SUM(limit_delta), 0) AS limit_delta
                FROM campaign_budget_ledger
                WHERE workspace_id = ? AND campaign_id = ? AND unit = ?
                """,
                (entry.workspace_id, entry.campaign_id, entry.unit),
            ).fetchone()
            reserved = float(totals["reserved"]) + entry.reserved_delta
            actual = float(totals["actual"]) + entry.actual_delta
            effective_limit = float(base_limit) + float(totals["limit_delta"]) + entry.limit_delta
            if reserved < 0 or actual < 0 or effective_limit < 0:
                raise BudgetInvariantError(f"{BudgetInvariantError.code}: negative ledger total")
            overrun = reserved + actual > effective_limit
            if overrun and entry.kind.value not in {"settle", "correction"}:
                raise BudgetExceededError(BudgetExceededError.code)
            connection.execute(
                """
                INSERT INTO campaign_budget_ledger(
                    workspace_id, campaign_id, entry_id, unit, entry_kind,
                    reserved_delta, actual_delta, limit_delta, action_id,
                    evidence_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.workspace_id,
                    entry.campaign_id,
                    entry.entry_id,
                    entry.unit,
                    entry.kind.value,
                    entry.reserved_delta,
                    entry.actual_delta,
                    entry.limit_delta,
                    entry.action_id,
                    _json(entry.evidence),
                    entry.actor_id,
                    _iso(entry.created_at),
                ),
            )
            next_status = current.status
            prior_status = current.prior_scheduling_status
            if overrun and current.status in {CampaignStatus.ACTIVE, CampaignStatus.PAUSED}:
                next_status = CampaignStatus.AWAITING_AUTHORITY
                prior_status = current.status
            updated = current.model_copy(
                update={
                    "status": next_status,
                    "prior_scheduling_status": prior_status,
                    "version": current.version + 1,
                    "updated_at": entry.created_at,
                }
            )
            cursor = connection.execute(
                """
                UPDATE campaigns SET status = ?, prior_scheduling_status = ?, version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.status.value,
                    updated.prior_scheduling_status.value
                    if updated.prior_scheduling_status
                    else None,
                    updated.version,
                    _iso(updated.updated_at),
                    entry.workspace_id,
                    entry.campaign_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError(expected_version, current.version)
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'{entry.entry_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=entry.workspace_id,
                campaign_id=entry.campaign_id,
                sequence=self._next_event_sequence(
                    connection, entry.workspace_id, entry.campaign_id
                ),
                aggregate_version=updated.version,
                event_type="campaign:budget-overrun" if overrun else "campaign:budget-recorded",
                payload={
                    "entry_id": entry.entry_id,
                    "unit": entry.unit,
                    "kind": entry.kind.value,
                    "reserved": reserved,
                    "actual": actual,
                    "effective_limit": effective_limit,
                },
                actor_id=entry.actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=entry.created_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=entry.workspace_id,
                actor_id=entry.actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
            )
        return BudgetMutation(updated, event, entry)

    def acquire_lease(
        self,
        lease_key: str,
        owner_id: str,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> LeaseRecord:
        """Acquire or renew a fenced scheduler/action lease in one transaction."""

        self._require_initialized()
        if ttl <= timedelta(0):
            raise ValueError("lease ttl must be positive")
        heartbeat = now or utc_now()
        expires_at = heartbeat + ttl
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_scheduler_leases WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
            if row is None:
                generation = 1
                observation_version = 1
                connection.execute(
                    """
                    INSERT INTO campaign_scheduler_leases(
                        lease_key, owner_id, generation, controller_observation_version,
                        expires_at, heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease_key,
                        owner_id,
                        generation,
                        observation_version,
                        _iso(expires_at),
                        _iso(heartbeat),
                    ),
                )
            else:
                current_expiry = _dt(row["expires_at"])
                if row["owner_id"] == owner_id and current_expiry > heartbeat:
                    generation = int(row["generation"])
                elif current_expiry <= heartbeat:
                    generation = int(row["generation"]) + 1
                else:
                    raise LeaseBusyError(
                        f"{LeaseBusyError.code}: held by another owner until {current_expiry.isoformat()}"
                    )
                observation_version = int(row["controller_observation_version"]) + 1
                connection.execute(
                    """
                    UPDATE campaign_scheduler_leases
                    SET owner_id = ?, generation = ?, controller_observation_version = ?,
                        expires_at = ?, heartbeat_at = ?
                    WHERE lease_key = ?
                    """,
                    (
                        owner_id,
                        generation,
                        observation_version,
                        _iso(expires_at),
                        _iso(heartbeat),
                        lease_key,
                    ),
                )
        return LeaseRecord(
            lease_key,
            owner_id,
            generation,
            observation_version,
            expires_at,
            heartbeat,
        )

    def heartbeat_lease(
        self,
        lease_key: str,
        owner_id: str,
        generation: int,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> LeaseRecord:
        """Renew only the current unexpired owner/generation fencing token."""

        self._require_initialized()
        if ttl <= timedelta(0):
            raise ValueError("lease ttl must be positive")
        heartbeat = now or utc_now()
        expires_at = heartbeat + ttl
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                UPDATE campaign_scheduler_leases
                SET expires_at = ?, heartbeat_at = ?,
                    controller_observation_version = controller_observation_version + 1
                WHERE lease_key = ? AND owner_id = ? AND generation = ? AND expires_at > ?
                RETURNING controller_observation_version
                """,
                (
                    _iso(expires_at),
                    _iso(heartbeat),
                    lease_key,
                    owner_id,
                    generation,
                    _iso(heartbeat),
                ),
            ).fetchone()
            if row is None:
                raise LeaseLostError(LeaseLostError.code)
            observation_version = int(row["controller_observation_version"])
        return LeaseRecord(
            lease_key,
            owner_id,
            generation,
            observation_version,
            expires_at,
            heartbeat,
        )

    def get_lease(self, lease_key: str) -> LeaseRecord | None:
        """Read one scheduler lease for controller health projection."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_scheduler_leases WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
        if row is None:
            return None
        expires_at = _dt(row["expires_at"])
        heartbeat_at = _dt(row["heartbeat_at"])
        if expires_at is None or heartbeat_at is None:
            raise CampaignPersistenceError("campaign_scheduler_lease_invalid")
        return LeaseRecord(
            lease_key=row["lease_key"],
            owner_id=row["owner_id"],
            generation=int(row["generation"]),
            controller_observation_version=int(row["controller_observation_version"]),
            expires_at=expires_at,
            heartbeat_at=heartbeat_at,
        )

    def release_lease(
        self,
        lease_key: str,
        owner_id: str,
        generation: int,
        *,
        now: datetime | None = None,
    ) -> None:
        """Expire a lease without deleting its monotonic fencing generation."""

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            released_at = now or utc_now()
            cursor = connection.execute(
                """
                UPDATE campaign_scheduler_leases
                SET expires_at = ?, heartbeat_at = ?,
                    controller_observation_version = controller_observation_version + 1
                WHERE lease_key = ? AND owner_id = ? AND generation = ?
                """,
                (_iso(released_at), _iso(released_at), lease_key, owner_id, generation),
            )
            if cursor.rowcount != 1:
                raise LeaseLostError(LeaseLostError.code)

    # Auth storage is deliberately narrow; token derivation and policy live in auth.py.
    def insert_actor_credential(self, value: StoredCredential) -> None:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO campaign_actor_credentials(
                    credential_id, actor_id, autonomy_profile, credential_kind,
                    workspace_ids_json, authorization_revision,
                    token_salt, token_hash, issued_at, expires_at, token_not_before, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    value.credential_id,
                    value.actor_id,
                    value.autonomy_profile,
                    value.credential_kind,
                    _json(value.workspace_ids),
                    value.authorization_revision,
                    value.token_salt,
                    value.token_hash,
                    _iso(value.issued_at),
                    _iso(value.expires_at),
                    _iso(value.token_not_before),
                    _iso(value.revoked_at) if value.revoked_at else None,
                ),
            )

    def get_actor_credential(self, credential_id: str) -> StoredCredential | None:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_actor_credentials WHERE credential_id = ?",
                (credential_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredCredential(
            credential_id=row["credential_id"],
            actor_id=row["actor_id"],
            autonomy_profile=row["autonomy_profile"],
            credential_kind=row["credential_kind"],
            workspace_ids=tuple(json.loads(row["workspace_ids_json"])),
            authorization_revision=int(row["authorization_revision"]),
            token_salt=row["token_salt"],
            token_hash=row["token_hash"],
            issued_at=_dt(row["issued_at"]),
            expires_at=_dt(row["expires_at"]),
            token_not_before=_dt(row["token_not_before"]),
            revoked_at=_dt(row["revoked_at"]),
        )

    def insert_access_token(self, value: StoredAccessToken) -> None:
        """Issue a child only if its parent is still valid in the same transaction."""

        self._require_initialized()
        now = utc_now()
        with self._connection(immediate=True) as connection:
            parent = connection.execute(
                "SELECT * FROM campaign_actor_credentials WHERE credential_id = ?",
                (value.credential_id,),
            ).fetchone()
            if (
                parent is None
                or parent["revoked_at"] is not None
                or _dt(parent["expires_at"]) <= now
                or _dt(parent["token_not_before"]) > value.issued_at
            ):
                raise CampaignPersistenceError("campaign_refresh_credential_invalid")
            connection.execute(
                """
                INSERT INTO campaign_access_tokens(
                    access_token_id, credential_id, token_salt, token_hash,
                    issued_at, expires_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    value.access_token_id,
                    value.credential_id,
                    value.token_salt,
                    value.token_hash,
                    _iso(value.issued_at),
                    _iso(value.expires_at),
                    None,
                ),
            )

    def get_access_with_parent(
        self, access_token_id: str
    ) -> tuple[StoredAccessToken, StoredCredential] | None:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    a.access_token_id, a.credential_id, a.token_salt AS access_salt,
                    a.token_hash AS access_hash, a.issued_at AS access_issued_at,
                    a.expires_at AS access_expires_at, a.revoked_at AS access_revoked_at,
                    c.actor_id, c.autonomy_profile, c.workspace_ids_json,
                    c.credential_kind, c.authorization_revision,
                    c.token_salt AS credential_salt, c.token_hash AS credential_hash,
                    c.issued_at AS credential_issued_at, c.expires_at AS credential_expires_at,
                    c.token_not_before, c.revoked_at AS credential_revoked_at
                FROM campaign_access_tokens a
                JOIN campaign_actor_credentials c ON c.credential_id = a.credential_id
                WHERE a.access_token_id = ?
                """,
                (access_token_id,),
            ).fetchone()
        if row is None:
            return None
        access = StoredAccessToken(
            access_token_id=row["access_token_id"],
            credential_id=row["credential_id"],
            token_salt=row["access_salt"],
            token_hash=row["access_hash"],
            issued_at=_dt(row["access_issued_at"]),
            expires_at=_dt(row["access_expires_at"]),
            revoked_at=_dt(row["access_revoked_at"]),
        )
        parent = StoredCredential(
            credential_id=row["credential_id"],
            actor_id=row["actor_id"],
            autonomy_profile=row["autonomy_profile"],
            credential_kind=row["credential_kind"],
            workspace_ids=tuple(json.loads(row["workspace_ids_json"])),
            authorization_revision=int(row["authorization_revision"]),
            token_salt=row["credential_salt"],
            token_hash=row["credential_hash"],
            issued_at=_dt(row["credential_issued_at"]),
            expires_at=_dt(row["credential_expires_at"]),
            token_not_before=_dt(row["token_not_before"]),
            revoked_at=_dt(row["credential_revoked_at"]),
        )
        return access, parent

    def revise_actor_authorization(
        self,
        credential_id: str,
        *,
        autonomy_profile: str,
        workspace_ids: tuple[str, ...],
        audit_event_id: str,
    ) -> int:
        """Atomically revise durable authority and advance its cache revision."""

        self._require_initialized()
        now = utc_now()
        with self._connection(immediate=True) as connection:
            parent = connection.execute(
                """
                SELECT actor_id, autonomy_profile, workspace_ids_json,
                       authorization_revision, revoked_at
                FROM campaign_actor_credentials WHERE credential_id = ?
                """,
                (credential_id,),
            ).fetchone()
            if parent is None:
                raise RecordNotFoundError("campaign credential not found")
            if parent["revoked_at"] is not None:
                raise CampaignPersistenceError("campaign_refresh_credential_invalid")
            encoded_workspaces = _json(workspace_ids)
            current_revision = int(parent["authorization_revision"])
            if (
                parent["autonomy_profile"] == autonomy_profile
                and parent["workspace_ids_json"] == encoded_workspaces
            ):
                return current_revision
            next_revision = current_revision + 1
            connection.execute(
                """
                UPDATE campaign_actor_credentials
                SET autonomy_profile = ?, workspace_ids_json = ?, authorization_revision = ?
                WHERE credential_id = ?
                """,
                (autonomy_profile, encoded_workspaces, next_revision, credential_id),
            )
            connection.execute(
                """
                INSERT INTO campaign_auth_audit_events(
                    event_id, credential_id, event_type, actor_id, safe_payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_event_id,
                    credential_id,
                    "campaign-auth:authorization-revised",
                    parent["actor_id"],
                    _json(
                        {
                            "autonomy_profile": autonomy_profile,
                            "workspace_count": len(workspace_ids),
                            "authorization_revision": next_revision,
                        }
                    ),
                    _iso(now),
                ),
            )
        return next_revision

    def revoke_actor_credential(
        self, credential_id: str, *, audit_event_id: str, reason: str
    ) -> int:
        """Revoke a parent and all descendants atomically; return child count."""

        self._require_initialized()
        now = utc_now()
        with self._connection(immediate=True) as connection:
            parent = connection.execute(
                """
                SELECT actor_id, authorization_revision, revoked_at
                FROM campaign_actor_credentials WHERE credential_id = ?
                """,
                (credential_id,),
            ).fetchone()
            if parent is None:
                raise RecordNotFoundError("campaign credential not found")
            if parent["revoked_at"] is not None:
                return 0
            connection.execute(
                """
                UPDATE campaign_actor_credentials
                SET revoked_at = ?, authorization_revision = ?
                WHERE credential_id = ?
                """,
                (_iso(now), int(parent["authorization_revision"]) + 1, credential_id),
            )
            children = connection.execute(
                """
                UPDATE campaign_access_tokens SET revoked_at = ?
                WHERE credential_id = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (_iso(now), credential_id, _iso(now)),
            )
            connection.execute(
                """
                INSERT INTO campaign_auth_audit_events(
                    event_id, credential_id, event_type, actor_id, safe_payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_event_id,
                    credential_id,
                    "campaign-auth:credential-revoked",
                    parent["actor_id"],
                    _json(
                        {
                            "reason": reason,
                            "descendants_revoked": children.rowcount,
                            "authorization_revision": int(parent["authorization_revision"]) + 1,
                        }
                    ),
                    _iso(now),
                ),
            )
        return children.rowcount


__all__ = [
    "BudgetExceededError",
    "BudgetInvariantError",
    "BudgetMutation",
    "CampaignMutation",
    "CampaignPersistenceError",
    "CampaignRepository",
    "IdempotencyConflictError",
    "InvalidProposalTransitionError",
    "LeaseBusyError",
    "LeaseLostError",
    "LeaseRecord",
    "MigrationChecksumError",
    "OperationMutation",
    "PromotionGateFailedError",
    "ProtectedLeaseDeniedError",
    "RecordAlreadyExistsError",
    "RecordNotFoundError",
    "RevisionConflictError",
    "ProposalMutation",
    "ProposalSelection",
    "StoredAccessToken",
    "StoredCredential",
]
