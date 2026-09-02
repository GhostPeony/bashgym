"""One-effect resident coordination for durable AutoResearch campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchError,
    AutoResearchNextAction,
    AutoResearchRepository,
    AutoResearchResult,
    AutoResearchState,
    ExperimentOutcome,
    ExperimentProvenance,
)
from bashgym.campaigns.contracts import (
    TERMINAL_CAMPAIGN_STATES,
    CampaignStatus,
    CredentialKind,
    FailureClass,
    StudyStatus,
    canonical_hash,
)
from bashgym.campaigns.persistence import CampaignPersistenceError, RecordNotFoundError
from bashgym.ledger.persistence import LedgerPersistenceError

if TYPE_CHECKING:
    from bashgym.campaigns.autoresearch_evidence import CampaignEvaluationProjector


MAX_ATTEMPTS_PER_ACTION = 2
_AGENT_ACTIONS = frozenset(
    {
        AutoResearchNextAction.PREPARE_CAMPAIGN,
        AutoResearchNextAction.START_CAMPAIGN,
        AutoResearchNextAction.SUBMIT_BASELINE,
        AutoResearchNextAction.PROPOSE_CANDIDATE,
        AutoResearchNextAction.BLOCKED,
    }
)


@dataclass(frozen=True)
class AutoResearchWaitObservation:
    """One canonical state sample paired with its durable event watermark."""

    status: Literal["waiting", "changed", "agent_action_required", "terminal"]
    next_cursor: int
    state: AutoResearchState


def _state_requires_agent_action(state: AutoResearchState) -> bool:
    return state.next_action in _AGENT_ACTIONS


def _latest_event_cursor(
    repository: AutoResearchRepository,
    workspace_id: str,
    campaign_id: str,
) -> int:
    with repository._connection() as connection:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(cursor), 0) AS latest_cursor
            FROM campaign_events
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (workspace_id, campaign_id),
        ).fetchone()
    return int(row["latest_cursor"])


def observe_research_wait(
    repository: AutoResearchRepository,
    core: AutoResearchCampaignCore,
    *,
    workspace_id: str,
    campaign_id: str,
    after_cursor: int,
) -> AutoResearchWaitObservation:
    """Read restart-safe wait state without creating a wake or session record."""

    # Pair state with a cursor that was stable on both sides of the state read.
    # If writes remain continuously active, retain the earlier watermark so a
    # subsequent wait can never resume past a change the caller did not observe.
    for _sample in range(8):
        cursor_before = _latest_event_cursor(repository, workspace_id, campaign_id)
        state = core.state(workspace_id, campaign_id)
        cursor_after = _latest_event_cursor(repository, workspace_id, campaign_id)
        if cursor_before == cursor_after:
            next_cursor = cursor_after
            break
    else:
        next_cursor = cursor_before
    if state.campaign_status in TERMINAL_CAMPAIGN_STATES:
        status = "terminal"
    elif _state_requires_agent_action(state):
        status = "agent_action_required"
    elif next_cursor > after_cursor:
        status = "changed"
    else:
        status = "waiting"
    return AutoResearchWaitObservation(status=status, next_cursor=next_cursor, state=state)


@dataclass(frozen=True)
class AutoResearchLoopTickResult:
    """Compact result for a host agent and the resident worker."""

    status: str
    effect_performed: bool
    workspace_id: str | None = None
    campaign_id: str | None = None
    proposal_id: str | None = None
    action_id: str | None = None
    state: AutoResearchState | None = None
    agent_action_required: bool = False
    max_attempts_per_action: int = MAX_ATTEMPTS_PER_ACTION


class AutoResearchLoopCoordinator:
    """Advance one durable mechanical effect; leave scientific judgment to the host."""

    def __init__(
        self,
        repository: AutoResearchRepository,
        projector: CampaignEvaluationProjector,
        core: AutoResearchCampaignCore,
    ) -> None:
        self.repository = repository
        self.projector = projector
        self.core = core

    def _next_completed_evaluation(self):
        with self.repository._connection() as connection:
            return connection.execute(
                """
                SELECT a.workspace_id, a.campaign_id, s.study_id, s.proposal_id,
                       a.action_id, t.attempt_id
                FROM campaign_actions a
                JOIN campaign_studies s
                  ON s.workspace_id = a.workspace_id AND s.study_id = a.study_id
                JOIN campaign_attempts t
                  ON t.workspace_id = a.workspace_id AND t.action_id = a.action_id
                JOIN campaigns c
                  ON c.workspace_id = a.workspace_id AND c.campaign_id = a.campaign_id
                JOIN autoresearch_proposal_controls p
                  ON p.workspace_id = s.workspace_id
                 AND p.campaign_id = s.campaign_id
                 AND p.proposal_id = s.proposal_id
                LEFT JOIN autoresearch_results r
                  ON r.workspace_id = p.workspace_id
                 AND r.campaign_id = p.campaign_id
                 AND r.proposal_id = p.proposal_id
                LEFT JOIN autoresearch_diagnostic_results d
                  ON d.workspace_id = p.workspace_id
                 AND d.campaign_id = p.campaign_id
                 AND d.proposal_id = p.proposal_id
                WHERE c.status = ?
                  AND a.stage_kind = 'development_evaluation'
                  AND a.status = 'completed'
                  AND t.status = 'completed'
                  AND a.sealed_result_uri IS NOT NULL
                  AND a.sealed_result_uri != ''
                  AND r.proposal_id IS NULL
                  AND d.proposal_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM campaign_attempts newer
                      WHERE newer.workspace_id = t.workspace_id
                        AND newer.action_id = t.action_id
                        AND newer.attempt_number > t.attempt_number
                  )
                ORDER BY a.updated_at, a.workspace_id, a.campaign_id, a.action_id
                LIMIT 1
                """,
                (CampaignStatus.ACTIVE.value,),
            ).fetchone()

    def _next_completed_diagnostic(self):
        with self.repository._connection() as connection:
            return connection.execute(
                """
                SELECT a.workspace_id, a.campaign_id, s.study_id, s.proposal_id,
                       a.action_id, t.attempt_id
                FROM campaign_actions a
                JOIN campaign_studies s
                  ON s.workspace_id = a.workspace_id AND s.study_id = a.study_id
                JOIN campaign_attempts t
                  ON t.workspace_id = a.workspace_id AND t.action_id = a.action_id
                JOIN campaigns c
                  ON c.workspace_id = a.workspace_id AND c.campaign_id = a.campaign_id
                JOIN autoresearch_proposal_controls p
                  ON p.workspace_id = s.workspace_id
                 AND p.campaign_id = s.campaign_id
                 AND p.proposal_id = s.proposal_id
                LEFT JOIN autoresearch_diagnostic_results d
                  ON d.workspace_id = p.workspace_id
                 AND d.campaign_id = p.campaign_id
                 AND d.proposal_id = p.proposal_id
                WHERE c.status = ?
                  AND p.role = 'diagnostic'
                  AND a.stage_kind = 'contract_evaluation'
                  AND a.status = 'completed'
                  AND t.status = 'completed'
                  AND a.sealed_result_uri IS NOT NULL
                  AND a.sealed_result_uri != ''
                  AND d.proposal_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM campaign_attempts newer
                      WHERE newer.workspace_id = t.workspace_id
                        AND newer.action_id = t.action_id
                        AND newer.attempt_number > t.attempt_number
                  )
                ORDER BY a.updated_at, a.workspace_id, a.campaign_id, a.action_id
                LIMIT 1
                """,
                (CampaignStatus.ACTIVE.value,),
            ).fetchone()

    def _next_failed_action(self):
        terminal_statuses = ("failed", "cancelled", "force_stopped")
        with self.repository._connection() as connection:
            return connection.execute(
                """
                SELECT a.workspace_id, a.campaign_id, a.study_id, a.action_id,
                       s.proposal_id, p.role, t.attempt_id, t.attempt_number,
                       c.version AS campaign_version
                FROM campaign_actions a
                JOIN campaign_studies s
                  ON s.workspace_id = a.workspace_id AND s.study_id = a.study_id
                JOIN campaign_attempts t
                  ON t.workspace_id = a.workspace_id AND t.action_id = a.action_id
                JOIN campaigns c
                  ON c.workspace_id = a.workspace_id AND c.campaign_id = a.campaign_id
                JOIN autoresearch_proposal_controls p
                  ON p.workspace_id = s.workspace_id
                 AND p.campaign_id = s.campaign_id
                 AND p.proposal_id = s.proposal_id
                LEFT JOIN autoresearch_results r
                  ON r.workspace_id = p.workspace_id
                 AND r.campaign_id = p.campaign_id
                 AND r.proposal_id = p.proposal_id
                LEFT JOIN autoresearch_diagnostic_results d
                  ON d.workspace_id = p.workspace_id
                 AND d.campaign_id = p.campaign_id
                 AND d.proposal_id = p.proposal_id
                WHERE c.status = ?
                  AND c.active_study_id IS NULL
                  AND c.active_action_id IS NULL
                  AND a.status IN (?, ?, ?)
                  AND t.status IN (?, ?, ?)
                  AND r.proposal_id IS NULL
                  AND d.proposal_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM campaign_attempts newer
                      WHERE newer.workspace_id = t.workspace_id
                        AND newer.action_id = t.action_id
                        AND newer.attempt_number > t.attempt_number
                  )
                ORDER BY a.updated_at, a.workspace_id, a.campaign_id, a.action_id
                LIMIT 1
                """,
                (CampaignStatus.ACTIVE.value, *terminal_statuses, *terminal_statuses),
            ).fetchone()

    @staticmethod
    def _operation_key(kind: str, *identity: str) -> str:
        return f"autoresearch-loop-{kind}-{canonical_hash(identity)[:32]}"

    @staticmethod
    def _agent_action_required(state: AutoResearchState) -> bool:
        return _state_requires_agent_action(state)

    def _idle_state(self, *, now: datetime | None) -> AutoResearchLoopTickResult:
        with self.repository._connection() as connection:
            row = connection.execute(
                """
                SELECT c.workspace_id, c.campaign_id
                FROM campaigns c
                JOIN autoresearch_campaign_specs s
                  ON s.workspace_id = c.workspace_id AND s.campaign_id = c.campaign_id
                WHERE c.status = ?
                ORDER BY c.updated_at, c.workspace_id, c.campaign_id
                LIMIT 1
                """,
                (CampaignStatus.ACTIVE.value,),
            ).fetchone()
        if row is None:
            return AutoResearchLoopTickResult(
                status="no_autoresearch_campaign", effect_performed=False
            )
        state = self.core.state(row["workspace_id"], row["campaign_id"], now=now)
        return AutoResearchLoopTickResult(
            status="agent_action_required" if self._agent_action_required(state) else "waiting",
            effect_performed=False,
            workspace_id=state.workspace_id,
            campaign_id=state.campaign_id,
            proposal_id=state.pending_proposal_id,
            state=state,
            agent_action_required=self._agent_action_required(state),
        )

    def _record_crash(self, failed) -> None:
        control = self.repository.get_autoresearch_proposal(
            failed["workspace_id"], failed["campaign_id"], failed["proposal_id"]
        )
        proposal = self.repository.get_proposal(
            failed["workspace_id"], failed["campaign_id"], failed["proposal_id"]
        )
        spec = self.repository.get_autoresearch_spec(failed["workspace_id"], failed["campaign_id"])
        attempts = self.repository.list_study_attempts(
            failed["workspace_id"], failed["campaign_id"], failed["study_id"]
        )
        usage = self.repository.study_budget_usage(
            failed["workspace_id"],
            failed["campaign_id"],
            failed["study_id"],
            spec.stop_rules.budget_unit,
        )
        provenance = (
            ExperimentProvenance.SIMULATED
            if self.core._proposal_is_simulated(proposal.proposal)
            else ExperimentProvenance.REAL
        )
        terminal_failure_classes: list[FailureClass | None] = []
        for item in attempts:
            if item.status.value not in {"failed", "cancelled", "force_stopped"}:
                continue
            try:
                manifest = self.repository.get_attempt_result_manifest(
                    item.workspace_id, item.attempt_id
                )
            except (RecordNotFoundError, CampaignPersistenceError):
                continue
            terminal_failure_classes.append(manifest.failure_class)

        if any(cls is None or cls == FailureClass.EXECUTION for cls in terminal_failure_classes):
            failure_class = FailureClass.EXECUTION
        elif terminal_failure_classes:
            failure_class = terminal_failure_classes[-1]
        else:
            failure_class = None
        self.core.record_result(
            AutoResearchResult(
                result_id=self._operation_key(
                    "crash-result",
                    failed["workspace_id"],
                    failed["campaign_id"],
                    failed["proposal_id"],
                    failed["action_id"],
                ),
                workspace_id=failed["workspace_id"],
                campaign_id=failed["campaign_id"],
                proposal_id=failed["proposal_id"],
                study_id=failed["study_id"],
                role=control.role,
                provenance=provenance,
                outcome=ExperimentOutcome.CRASHED,
                metric_name=spec.primary_metric,
                metric_value=None,
                failure_class=failure_class,
                actual_cost=float(usage["actual"]),
                attempt_ids=tuple(item.attempt_id for item in attempts),
                recorded_at=max(item.updated_at for item in attempts),
            )
        )

    def _record_diagnostic_failure(
        self, failed, *, reason: str = "diagnostic_execution_failed"
    ) -> None:
        from bashgym.campaigns.autoresearch import AutoResearchDiagnosticResult
        from bashgym.campaigns.diagnostic_actions import AutoResearchDiagnosticRecipe

        proposal = self.repository.get_proposal(
            failed["workspace_id"], failed["campaign_id"], failed["proposal_id"]
        ).proposal
        recipe = AutoResearchDiagnosticRecipe.model_validate(
            {key: value for key, value in proposal.evaluation_recipe.items() if key != "runtime"}
        )
        spec = self.repository.get_autoresearch_spec(failed["workspace_id"], failed["campaign_id"])
        usage = self.repository.study_budget_usage(
            failed["workspace_id"],
            failed["campaign_id"],
            failed["study_id"],
            spec.stop_rules.budget_unit,
        )
        attempts = self.repository.list_study_attempts(
            failed["workspace_id"], failed["campaign_id"], failed["study_id"]
        )
        self.repository.record_autoresearch_diagnostic_result(
            AutoResearchDiagnosticResult(
                workspace_id=failed["workspace_id"],
                campaign_id=failed["campaign_id"],
                proposal_id=failed["proposal_id"],
                study_id=failed["study_id"],
                attempt_id=failed["attempt_id"],
                status="unsupported",
                projection={
                    "schema_version": "bashgym.research_diagnostic_result.v1",
                    "probe_family": recipe.probe_family,
                    "question": recipe.question,
                    "hypothesis": recipe.hypothesis,
                    "informs_methods": list(recipe.informs_methods),
                    "status": "unsupported",
                    "measurements": [],
                    "observations": [],
                    "resource_usage": [],
                    "unsupported_reason": reason,
                    "evidence_reference": {
                        "proposal_id": failed["proposal_id"],
                        "study_id": failed["study_id"],
                        "attempt_id": failed["attempt_id"],
                    },
                },
                actual_cost=float(usage["actual"]),
                recorded_at=max(item.updated_at for item in attempts),
            )
        )

    def _record_invalid_evaluation(self, completed) -> None:
        proposal = self.repository.get_proposal(
            completed["workspace_id"], completed["campaign_id"], completed["proposal_id"]
        )
        study_id = proposal.study_id
        if study_id is None:
            raise RuntimeError("autoresearch_completed_evaluation_study_missing")
        study = self.repository.get_study(
            completed["workspace_id"], completed["campaign_id"], study_id
        )
        if study.status not in {
            StudyStatus.EXECUTION_FAILED,
            StudyStatus.ABANDONED,
            StudyStatus.CANCELLED,
        }:
            campaign = self.repository.get_campaign(
                completed["workspace_id"], completed["campaign_id"]
            )
            operation_key = self._operation_key(
                "invalid-evaluation",
                completed["workspace_id"],
                completed["campaign_id"],
                completed["action_id"],
            )
            self.repository.abandon_study(
                completed["workspace_id"],
                completed["campaign_id"],
                study_id,
                reason="sealed_evaluation_invalid",
                expected_version=campaign.version,
                actor_id="autoresearch-loop",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=operation_key,
                idempotency_key=operation_key,
            )
        failed = {
            **dict(completed),
            "study_id": study_id,
        }
        attempts = self.repository.list_study_attempts(
            completed["workspace_id"], completed["campaign_id"], study_id
        )
        if not attempts:
            raise RuntimeError("autoresearch_completed_evaluation_attempt_missing")
        failed["attempt_id"] = attempts[-1].attempt_id
        self._record_crash(failed)

    def _next_stop_state(self, *, now: datetime | None) -> AutoResearchState | None:
        with self.repository._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.workspace_id, c.campaign_id
                FROM campaigns c
                JOIN autoresearch_campaign_specs s
                  ON s.workspace_id = c.workspace_id AND s.campaign_id = c.campaign_id
                WHERE c.status = ?
                  AND c.active_study_id IS NULL
                  AND c.active_action_id IS NULL
                ORDER BY c.updated_at, c.workspace_id, c.campaign_id
                """,
                (CampaignStatus.ACTIVE.value,),
            ).fetchall()
        for row in rows:
            state = self.core.state(row["workspace_id"], row["campaign_id"], now=now)
            if state.next_action == AutoResearchNextAction.STOP:
                return state
        return None

    def tick(self, *, now: datetime | None = None) -> AutoResearchLoopTickResult:
        diagnostic = self._next_completed_diagnostic()
        if diagnostic is not None:
            diagnostic_status = "diagnostic_ingested"
            try:
                self.projector.project_diagnostic_and_ingest(
                    diagnostic["workspace_id"],
                    diagnostic["campaign_id"],
                    diagnostic["proposal_id"],
                )
            except (ValueError, AutoResearchError, LedgerPersistenceError):
                self._record_diagnostic_failure(diagnostic, reason="diagnostic_evidence_invalid")
                diagnostic_status = "invalid_diagnostic_recorded"
            state = self.core.state(diagnostic["workspace_id"], diagnostic["campaign_id"], now=now)
            return AutoResearchLoopTickResult(
                status=diagnostic_status,
                effect_performed=True,
                workspace_id=diagnostic["workspace_id"],
                campaign_id=diagnostic["campaign_id"],
                proposal_id=diagnostic["proposal_id"],
                action_id=diagnostic["action_id"],
                state=state,
                agent_action_required=self._agent_action_required(state),
            )
        completed = self._next_completed_evaluation()
        if completed is not None:
            try:
                self.projector.project_and_ingest(
                    completed["workspace_id"],
                    completed["campaign_id"],
                    completed["proposal_id"],
                )
            except (ValueError, AutoResearchError, LedgerPersistenceError):
                self._record_invalid_evaluation(completed)
                state = self.core.state(
                    completed["workspace_id"], completed["campaign_id"], now=now
                )
                return AutoResearchLoopTickResult(
                    status="invalid_evaluation_recorded",
                    effect_performed=True,
                    workspace_id=completed["workspace_id"],
                    campaign_id=completed["campaign_id"],
                    proposal_id=completed["proposal_id"],
                    action_id=completed["action_id"],
                    state=state,
                    agent_action_required=self._agent_action_required(state),
                )
            state = self.core.state(completed["workspace_id"], completed["campaign_id"], now=now)
            return AutoResearchLoopTickResult(
                status="evaluation_ingested",
                effect_performed=True,
                workspace_id=completed["workspace_id"],
                campaign_id=completed["campaign_id"],
                proposal_id=completed["proposal_id"],
                action_id=completed["action_id"],
                state=state,
                agent_action_required=self._agent_action_required(state),
            )
        failed = self._next_failed_action()
        if failed is not None and int(failed["attempt_number"]) < MAX_ATTEMPTS_PER_ACTION:
            operation_key = self._operation_key(
                "retry",
                failed["workspace_id"],
                failed["campaign_id"],
                failed["action_id"],
                str(MAX_ATTEMPTS_PER_ACTION),
            )
            self.repository.retry_action(
                failed["workspace_id"],
                failed["campaign_id"],
                failed["action_id"],
                expected_version=int(failed["campaign_version"]),
                actor_id="autoresearch-loop",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=operation_key,
                idempotency_key=operation_key,
            )
            state = self.core.state(failed["workspace_id"], failed["campaign_id"], now=now)
            return AutoResearchLoopTickResult(
                status="retry_scheduled",
                effect_performed=True,
                workspace_id=failed["workspace_id"],
                campaign_id=failed["campaign_id"],
                proposal_id=failed["proposal_id"],
                action_id=failed["action_id"],
                state=state,
            )
        if failed is not None:
            if failed["role"] == "diagnostic":
                self._record_diagnostic_failure(failed)
            else:
                self._record_crash(failed)
            state = self.core.state(failed["workspace_id"], failed["campaign_id"], now=now)
            return AutoResearchLoopTickResult(
                status="crash_recorded",
                effect_performed=True,
                workspace_id=failed["workspace_id"],
                campaign_id=failed["campaign_id"],
                proposal_id=failed["proposal_id"],
                action_id=failed["action_id"],
                state=state,
                agent_action_required=self._agent_action_required(state),
            )
        stop_state = self._next_stop_state(now=now)
        if stop_state is not None:
            operation_key = self._operation_key(
                "stop",
                stop_state.workspace_id,
                stop_state.campaign_id,
                stop_state.reason_code,
            )
            self.core.enforce_stop(
                stop_state.workspace_id,
                stop_state.campaign_id,
                controller_id="autoresearch-loop",
                correlation_id=operation_key,
                idempotency_key=operation_key,
                now=now,
            )
            terminal_state = self.core.state(
                stop_state.workspace_id, stop_state.campaign_id, now=now
            )
            return AutoResearchLoopTickResult(
                status="stop_enforced",
                effect_performed=True,
                workspace_id=stop_state.workspace_id,
                campaign_id=stop_state.campaign_id,
                state=terminal_state,
            )
        return self._idle_state(now=now)


__all__ = [
    "MAX_ATTEMPTS_PER_ACTION",
    "AutoResearchLoopCoordinator",
    "AutoResearchLoopTickResult",
    "AutoResearchWaitObservation",
    "observe_research_wait",
]
