"""Transport-neutral campaign commands with actor capability enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    BudgetEntryKind,
    BudgetLedgerEntry,
    Campaign,
    CampaignEvidenceSnapshot,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    ManifestRevision,
    ProposalRecord,
    ProtectedEvaluationResult,
    PublicCampaignArtifactV1,
    StudyProposalSubmission,
    canonical_hash,
)
from bashgym.campaigns.export import CampaignExportSnapshot, export_campaign_evidence
from bashgym.campaigns.persistence import (
    CampaignMutation,
    ProposalMutation,
    ProposalSelection,
)
from bashgym.campaigns.proposals import validate_proposal_submission
from bashgym.campaigns.remote import RemoteRunIdentity
from bashgym.campaigns.runtime import (
    CampaignRuntimeRepository,
)
from bashgym.campaigns.visibility import (
    project_public_campaign_artifact,
    project_public_campaign_event,
)

_TRIGGER_CAPABILITIES = {
    CampaignTrigger.START: Capability.CAMPAIGN_START,
    CampaignTrigger.PAUSE: Capability.CAMPAIGN_PAUSE,
    CampaignTrigger.RESUME: Capability.CAMPAIGN_RESUME,
    CampaignTrigger.CANCEL: Capability.CAMPAIGN_CANCEL,
    CampaignTrigger.CONCLUDE: Capability.CAMPAIGN_COMPLETE,
    CampaignTrigger.PROMOTION_COMMITTED: Capability.PROMOTION_DECIDE,
}


class CampaignService:
    """Apply the same authority rules for REST, CLI, MCP, and the worker."""

    def __init__(
        self,
        repository: CampaignRuntimeRepository,
        *,
        approved_template_hashes: dict[str, str] | None = None,
        export_root: Path | None = None,
    ):
        self.repository = repository
        self.approved_template_hashes = dict(approved_template_hashes or {})
        self.export_root = (export_root or repository.db_path.parent / "exports").resolve()

    def create(
        self,
        campaign: Campaign,
        manifest: CampaignManifest,
        *,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        approved_template_id: str | None = None,
    ) -> CampaignMutation:
        manifest_hash = canonical_hash(manifest.model_dump(mode="json"))
        template_is_approved = (
            approved_template_id is not None
            and self.approved_template_hashes.get(approved_template_id) == manifest_hash
        )
        required = Capability.CAMPAIGN_CREATE
        if template_is_approved:
            required = Capability.CAMPAIGN_CREATE_FROM_TEMPLATE
        principal.require(campaign.workspace_id, required)
        if principal.actor_id != campaign.owner_actor_id:
            raise PermissionError("campaign_owner_must_match_principal")
        revision = ManifestRevision(
            workspace_id=campaign.workspace_id,
            campaign_id=campaign.campaign_id,
            revision=1,
            manifest=manifest,
            actor_id=principal.actor_id,
            correlation_id=correlation_id,
        )
        return self.repository.create_campaign(
            campaign,
            revision,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def get(self, workspace_id: str, campaign_id: str, principal: ActorPrincipal) -> Campaign:
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        return self.repository.get_campaign(workspace_id, campaign_id)

    def manifest(
        self,
        workspace_id: str,
        campaign_id: str,
        revision: int,
        principal: ActorPrincipal,
    ) -> ManifestRevision:
        self.get(workspace_id, campaign_id, principal)
        return self.repository.get_manifest_revision(workspace_id, campaign_id, revision)

    def list(
        self,
        workspace_id: str,
        principal: ActorPrincipal,
        *,
        kind: CampaignKind | None = None,
        status: CampaignStatus | None = None,
    ) -> list[Campaign]:
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        campaigns = self.repository.list_campaigns(workspace_id)
        return [
            campaign
            for campaign in campaigns
            if (kind is None or campaign.kind == kind)
            and (status is None or campaign.status == status)
        ]

    def events(
        self,
        workspace_id: str,
        campaign_id: str,
        principal: ActorPrincipal,
        *,
        after_cursor: int = 0,
        limit: int = 200,
    ):
        self.get(workspace_id, campaign_id, principal)
        return tuple(
            (cursor, project_public_campaign_event(event))
            for cursor, event in self.repository.list_events(
                workspace_id, campaign_id, after_cursor=after_cursor, limit=limit
            )
        )

    def artifacts(
        self, workspace_id: str, campaign_id: str, principal: ActorPrincipal
    ) -> tuple[PublicCampaignArtifactV1, ...]:
        self.get(workspace_id, campaign_id, principal)
        return tuple(
            project_public_campaign_artifact(artifact)
            for artifact in self.repository.list_artifacts(workspace_id, campaign_id)
        )

    def attempts(self, workspace_id: str, campaign_id: str, principal: ActorPrincipal):
        self.get(workspace_id, campaign_id, principal)
        return self.repository.list_attempts(workspace_id, campaign_id)

    def comparisons(self, workspace_id: str, campaign_id: str, principal: ActorPrincipal):
        self.get(workspace_id, campaign_id, principal)
        return self.repository.list_development_comparisons(workspace_id, campaign_id)

    def proposals(
        self, workspace_id: str, campaign_id: str, principal: ActorPrincipal
    ) -> tuple[ProposalRecord, ...]:
        self.get(workspace_id, campaign_id, principal)
        return self.repository.list_proposals(workspace_id, campaign_id)

    def studies(self, workspace_id: str, campaign_id: str, principal: ActorPrincipal):
        self.get(workspace_id, campaign_id, principal)
        return self.repository.list_studies(workspace_id, campaign_id)

    def study(
        self, workspace_id: str, campaign_id: str, study_id: str, principal: ActorPrincipal
    ):
        self.get(workspace_id, campaign_id, principal)
        return self.repository.get_study(workspace_id, campaign_id, study_id)

    def submit_proposal(
        self,
        submission: StudyProposalSubmission,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        principal.require(submission.workspace_id, Capability.STUDY_PROPOSE)
        campaign = self.repository.get_campaign(submission.workspace_id, submission.campaign_id)
        manifest = self.repository.get_manifest_revision(
            submission.workspace_id,
            submission.campaign_id,
            campaign.manifest_revision,
        ).manifest
        validation = validate_proposal_submission(
            submission,
            manifest,
            principal,
            existing_prerequisite_ids=self.repository.study_ids(
                submission.workspace_id, submission.campaign_id
            ),
        )
        normalized_priority = (
            50
            if principal.autonomy_profile == AutonomyProfile.HERMES_BOUNDED
            else submission.priority
        )
        return self.repository.submit_proposal(
            submission,
            validation,
            normalized_priority=normalized_priority,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def withdraw_proposal(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        principal.require(workspace_id, Capability.STUDY_PROPOSE)
        return self.repository.withdraw_proposal(
            workspace_id,
            campaign_id,
            proposal_id,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def evidence(
        self, workspace_id: str, campaign_id: str, principal: ActorPrincipal
    ) -> CampaignEvidenceSnapshot:
        self.get(workspace_id, campaign_id, principal)
        return self.repository.build_evidence_snapshot(workspace_id, campaign_id)

    def revise_manifest(
        self,
        workspace_id: str,
        campaign_id: str,
        manifest: CampaignManifest,
        *,
        reason: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.CAMPAIGN_REVISE)
        return self.repository.revise_manifest(
            workspace_id,
            campaign_id,
            manifest,
            reason=reason,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def retry_action(
        self,
        workspace_id: str,
        campaign_id: str,
        action_id: str,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.STUDY_RETRY)
        return self.repository.retry_action(
            workspace_id,
            campaign_id,
            action_id,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def abandon_study(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        *,
        reason: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.STUDY_ABANDON)
        return self.repository.abandon_study(
            workspace_id,
            campaign_id,
            study_id,
            reason=reason,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def amend_budget(
        self,
        workspace_id: str,
        campaign_id: str,
        resource: str,
        delta: float,
        *,
        reason: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.COMPUTE_AMEND_BUDGET)
        digest = hashlib.sha256(
            f"{campaign_id}:{resource}:{idempotency_key}".encode()
        ).hexdigest()[:24]
        entry = BudgetLedgerEntry(
            entry_id=f"budget-amend-{digest}",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            unit=resource,
            kind=BudgetEntryKind.AMEND,
            limit_delta=delta,
            evidence={"reason": reason},
            actor_id=principal.actor_id,
        )
        return self.repository.record_budget_entry(
            entry,
            expected_version=expected_version,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def approve_source(
        self,
        workspace_id: str,
        campaign_id: str,
        source_id: str,
        evidence: dict[str, Any],
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.DATA_APPROVE_EXTERNAL)
        return self.repository.approve_source(
            workspace_id,
            campaign_id,
            source_id,
            evidence,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def request_force_stop(
        self,
        workspace_id: str,
        campaign_id: str,
        action_id: str,
        identity: RemoteRunIdentity,
        *,
        reason: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.COMPUTE_FORCE_STOP)
        return self.repository.request_force_stop(
            workspace_id,
            campaign_id,
            action_id,
            identity,
            reason=reason,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def acquire_protected_lease(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.EVAL_PROTECTED_ACQUIRE)
        return self.repository.acquire_protected_epoch(
            workspace_id,
            campaign_id,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def record_protected_evaluation(
        self,
        workspace_id: str,
        campaign_id: str,
        result: ProtectedEvaluationResult,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.EVAL_PROTECTED_EXECUTE)
        return self.repository.record_protected_evaluation(
            workspace_id,
            campaign_id,
            result,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def promote(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        override_reason: str | None = None,
    ):
        principal.require(workspace_id, Capability.PROMOTION_DECIDE)
        if override_reason is not None:
            principal.require(workspace_id, Capability.PROMOTION_OVERRIDE)
        return self.repository.promote_candidate(
            workspace_id,
            campaign_id,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            override_reason=override_reason,
        )

    def export(
        self,
        workspace_id: str,
        campaign_id: str,
        formats: tuple[str, ...],
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ):
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        campaign = self.repository.get_campaign(workspace_id, campaign_id)
        export_id = f"export-{hashlib.sha256(f'{campaign_id}:{idempotency_key}'.encode()).hexdigest()[:24]}"
        output_directory = self.export_root / workspace_id / campaign_id / export_id
        manifest_path = output_directory / "export_manifest.json"
        if manifest_path.is_file():
            export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            if campaign.version != expected_version:
                from bashgym.campaigns.persistence import RevisionConflictError

                raise RevisionConflictError(expected_version, campaign.version)
            attempts = self.repository.list_attempts(workspace_id, campaign_id)
            artifacts = self.repository.list_artifacts(workspace_id, campaign_id)
            comparisons = self.repository.list_development_comparisons(workspace_id, campaign_id)
            losses: dict[str, tuple[dict[str, Any], ...]] = {}
            for attempt in attempts:
                values = self.repository.get_metric_series(
                    workspace_id,
                    attempt.attempt_id,
                    "loss",
                    source="training_metrics.jsonl",
                    limit=5000,
                )
                if values:
                    losses[attempt.attempt_id] = tuple(
                        item.model_dump(mode="json") for item in values
                    )
            snapshot = CampaignExportSnapshot(
                campaign=campaign.model_dump(mode="json"),
                attempts=tuple(
                    item.model_dump(mode="json", exclude={"sealed_result_uri"})
                    for item in attempts
                ),
                artifacts=tuple(
                    item.model_dump(mode="json", exclude={"uri"}) for item in artifacts
                ),
                comparisons=tuple(item.model_dump(mode="json") for item in comparisons),
                loss_by_attempt=losses,
            )
            export_manifest = export_campaign_evidence(snapshot, output_directory)
        return self.repository.record_export(
            workspace_id,
            campaign_id,
            export_id,
            formats,
            export_manifest,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def request_advance(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> CampaignMutation:
        principal.require(workspace_id, Capability.CAMPAIGN_START)
        return self.repository.request_advance(
            workspace_id,
            campaign_id,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def metrics(
        self,
        workspace_id: str,
        campaign_id: str,
        attempt_id: str,
        metric_name: str,
        principal: ActorPrincipal,
        *,
        source: str,
        after_step: int = -1,
        limit: int = 2000,
    ):
        self.get(workspace_id, campaign_id, principal)
        attempt = self.repository.get_attempt(workspace_id, attempt_id)
        if attempt.campaign_id != campaign_id:
            raise PermissionError("campaign_attempt_scope_mismatch")
        return self.repository.get_metric_series(
            workspace_id,
            attempt_id,
            metric_name,
            source=source,
            after_step=after_step,
            limit=limit,
        )

    def transition(
        self,
        workspace_id: str,
        campaign_id: str,
        trigger: CampaignTrigger,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        stop_reason: str | None = None,
    ) -> CampaignMutation:
        required = _TRIGGER_CAPABILITIES.get(trigger)
        if required is not None:
            principal.require(workspace_id, required)
        else:
            # Deterministic controller triggers are not agent shortcuts. The first
            # runtime worker will call a separate internal service boundary.
            raise PermissionError("campaign_controller_transition_required")
        return self.repository.transition_campaign(
            workspace_id,
            campaign_id,
            trigger,
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload=payload,
            stop_reason=stop_reason,
        )


class CampaignControllerService:
    """Controller-only scheduling boundary; never constructed from actor credentials."""

    def __init__(self, repository: CampaignRuntimeRepository, *, controller_id: str):
        self.repository = repository
        self.controller_id = controller_id

    def select_next_proposal(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        expected_version: int,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalSelection | None:
        return self.repository.select_next_proposal_as_controller(
            workspace_id,
            campaign_id,
            expected_version=expected_version,
            controller_id=self.controller_id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )


__all__ = ["CampaignControllerService", "CampaignService"]
