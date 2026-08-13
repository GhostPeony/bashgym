"""Resident AutoResearch loop coordination over durable campaign state."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchRepository,
    AutoResearchResult,
    ExperimentOutcome,
    ExperimentProvenance,
    ResultDecision,
)
from bashgym.campaigns.contracts import (
    ActionStatus,
    AttemptStatus,
    CampaignStatus,
    CampaignTrigger,
    CredentialKind,
    StageKind,
    StudyStatus,
)
from bashgym.campaigns.remote import RemoteRunState
from bashgym.campaigns.runtime import ActionSpec, CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignControllerService
from bashgym.campaigns.worker import CampaignWorker
from bashgym.campaigns.worker_service import build_worker
from tests.campaigns.test_autoresearch_campaign import NOW, activate, fresh_core
from tests.campaigns.test_proposals import principal, proposal
from tests.campaigns.test_worker import (
    START,
    FakeRemoteAdapter,
    active_repository,
    schedule_remote,
    seed_validated_study,
)
from tests.campaigns.test_worker_service import config_for


class _OutcomeWritingProjector:
    """Stand in only for the already-tested cryptographic projector boundary."""

    def __init__(self, core: AutoResearchCampaignCore) -> None:
        self.core = core

    def project_and_ingest(self, workspace_id: str, campaign_id: str, proposal_id: str):
        repository = self.core.repository
        control = repository.get_autoresearch_proposal(workspace_id, campaign_id, proposal_id)
        proposal_record = repository.get_proposal(workspace_id, campaign_id, proposal_id)
        attempts = repository.list_study_attempts(
            workspace_id, campaign_id, proposal_record.study_id
        )
        return self.core.record_result(
            AutoResearchResult(
                result_id=f"projected-{proposal_id}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                proposal_id=proposal_id,
                study_id=proposal_record.study_id,
                role=control.role,
                provenance=ExperimentProvenance.SIMULATED,
                outcome=ExperimentOutcome.COMPLETED,
                metric_name="mrr_at_10",
                metric_value=0.5,
                actual_cost=0.0,
                attempt_ids=tuple(item.attempt_id for item in attempts),
                recorded_at=max(item.updated_at for item in attempts),
            )
        )


class _RejectingProjector:
    def project_and_ingest(self, *_args, **_kwargs):
        raise AssertionError("durable outcome should suppress duplicate projection")


class _InvalidEvidenceProjector:
    def project_and_ingest(self, *_args, **_kwargs):
        raise ValueError("evaluation artifact digest does not match its action seal")


class _NeverTickCoordinator:
    def tick(self, **_kwargs):
        raise AssertionError("reconciliation must run before AutoResearch coordination")


class _BlockedCoordinator:
    def tick(self, **_kwargs):
        from bashgym.campaigns.autoresearch_loop import AutoResearchLoopTickResult

        return AutoResearchLoopTickResult(
            status="agent_action_required",
            effect_performed=False,
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            agent_action_required=True,
        )


def _active_proposal(tmp_path, *, max_attempts: int = 3):
    _path, repository, core = fresh_core(tmp_path, max_attempts=max_attempts, target=None)
    activate(core)
    core.submit_baseline(
        proposal("baseline-loop", estimated_cost=0.1),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="baseline-loop-submit",
        idempotency_key="baseline-loop-submit",
    )
    campaign = repository.get_campaign("workspace-a", "campaign-1")
    selected = CampaignControllerService(
        repository, controller_id="autoresearch-loop"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=campaign.version,
        correlation_id="baseline-loop-select",
        idempotency_key="baseline-loop-select",
    )
    assert selected is not None
    leader = repository.acquire_lease(
        "scheduler:autoresearch-loop-test",
        "loop-worker",
        ttl=timedelta(minutes=1),
        now=NOW,
    )
    campaign = repository.get_campaign("workspace-a", "campaign-1")
    attempt = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id=selected.study.study_id,
            stage_index=0,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract={},
            candidate_digest=selected.study.candidate_digest,
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.1,
            fake_steps=2,
        ),
        leader,
        expected_campaign_version=campaign.version,
        now=NOW,
    )
    return repository, core, attempt


def _terminalize(repository, attempt, status: AttemptStatus, *, sealed_uri: str | None = None):
    action_status = ActionStatus(status.value)
    study_status = (
        StudyStatus.COMPLETED if status == AttemptStatus.COMPLETED else StudyStatus.EXECUTION_FAILED
    )
    finished_at = NOW + timedelta(seconds=attempt.attempt_number)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_attempts SET status = ?, updated_at = ? WHERE attempt_id = ?",
            (status.value, finished_at.isoformat(), attempt.attempt_id),
        )
        connection.execute(
            """
            UPDATE campaign_actions SET status = ?, sealed_result_uri = ?, updated_at = ?
            WHERE action_id = ?
            """,
            (action_status.value, sealed_uri, finished_at.isoformat(), attempt.action_id),
        )
        connection.execute(
            """
            UPDATE campaign_studies SET status = ?, current_stage_index = 1,
                version = version + 1, updated_at = ?
            WHERE study_id = ?
            """,
            (study_status.value, finished_at.isoformat(), attempt.study_id),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL, active_action_id = NULL,
                version = version + 1, updated_at = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (finished_at.isoformat(), attempt.workspace_id, attempt.campaign_id),
        )
        connection.execute(
            """
            INSERT INTO campaign_budget_ledger(
                workspace_id, campaign_id, entry_id, unit, entry_kind,
                reserved_delta, actual_delta, limit_delta, action_id,
                evidence_json, actor_id, created_at
            ) VALUES (?, ?, ?, 'gpu_hours', 'settle', -0.1, ?, 0, ?, '{}', ?, ?)
            """,
            (
                attempt.workspace_id,
                attempt.campaign_id,
                f"settle-{attempt.attempt_id}",
                0.0 if status == AttemptStatus.COMPLETED else 0.1,
                attempt.action_id,
                "autoresearch-loop-test",
                finished_at.isoformat(),
            ),
        )


def _insert_failed_side_action(repository, attempt) -> None:
    failed_at = NOW + timedelta(seconds=10)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_actions(
                workspace_id, campaign_id, study_id, action_id, stage_index,
                stage_kind, input_digest, candidate_digest, manifest_revision,
                reservation_json, status, version, created_at, updated_at
            ) VALUES (?, ?, ?, 'side-failure', 0, 'full_training', ?, ?, 1,
                      ?, 'failed', 1, ?, ?)
            """,
            (
                attempt.workspace_id,
                attempt.campaign_id,
                attempt.study_id,
                "f" * 64,
                attempt.candidate_digest,
                json.dumps({"unit": "gpu_hours", "amount": 0.1}),
                failed_at.isoformat(),
                failed_at.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_attempts(
                workspace_id, action_id, attempt_id, attempt_number,
                claim_generation, status, executor_json, created_at, updated_at
            ) VALUES (?, 'side-failure', 'side-attempt-1', 1, 1, 'failed', ?, ?, ?)
            """,
            (
                attempt.workspace_id,
                json.dumps({"kind": "fake", "steps": 2}),
                failed_at.isoformat(),
                failed_at.isoformat(),
            ),
        )


def test_completed_sealed_evaluation_is_ingested_once_across_restart(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, attempt = _active_proposal(tmp_path)
    _terminalize(
        repository,
        attempt,
        AttemptStatus.COMPLETED,
        sealed_uri=str(tmp_path / "sealed-evaluation"),
    )

    first = AutoResearchLoopCoordinator(repository, _OutcomeWritingProjector(core), core).tick(
        now=NOW + timedelta(seconds=2)
    )
    restarted = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=3)
    )

    outcomes = repository.list_autoresearch_outcomes("workspace-a", "campaign-1")
    assert first.status == "evaluation_ingested"
    assert first.effect_performed is True
    assert len(outcomes) == 1
    assert outcomes[0].result.proposal_id == "baseline-loop"
    assert restarted.effect_performed is False


def test_invalid_completed_evaluation_records_one_crash_instead_of_restart_loop(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, attempt = _active_proposal(tmp_path)
    _terminalize(
        repository,
        attempt,
        AttemptStatus.COMPLETED,
        sealed_uri=str(tmp_path / "invalid-sealed-evaluation"),
    )

    rejected = AutoResearchLoopCoordinator(repository, _InvalidEvidenceProjector(), core).tick(
        now=NOW + timedelta(seconds=2)
    )
    restarted = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=3)
    )

    outcomes = repository.list_autoresearch_outcomes("workspace-a", "campaign-1")
    assert rejected.status == "invalid_evaluation_recorded"
    assert rejected.effect_performed is True
    assert len(outcomes) == 1
    assert outcomes[0].result.outcome == ExperimentOutcome.CRASHED
    assert outcomes[0].decision.decision == ResultDecision.CRASH
    assert outcomes[0].result.attempt_ids == (attempt.attempt_id,)
    assert restarted.effect_performed is False


@pytest.mark.parametrize(
    "terminal_status",
    (AttemptStatus.FAILED, AttemptStatus.CANCELLED, AttemptStatus.FORCE_STOPPED),
)
def test_first_definitive_failure_schedules_one_budgeted_retry(tmp_path, terminal_status):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, first_attempt = _active_proposal(tmp_path)
    _terminalize(repository, first_attempt, terminal_status)

    tick = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=2)
    )

    attempts = repository.list_study_attempts("workspace-a", "campaign-1", first_attempt.study_id)
    assert tick.status == "retry_scheduled"
    assert tick.effect_performed is True
    assert tick.max_attempts_per_action == 2
    assert [(item.attempt_number, item.status) for item in attempts] == [
        (1, terminal_status),
        (2, AttemptStatus.SCHEDULED),
    ]
    campaign = repository.get_campaign("workspace-a", "campaign-1")
    assert campaign.active_action_id == first_attempt.action_id


def test_second_definitive_failure_records_one_durable_crash(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, first_attempt = _active_proposal(tmp_path)
    _terminalize(repository, first_attempt, AttemptStatus.FAILED)
    AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=2)
    )
    retry = repository.list_study_attempts("workspace-a", "campaign-1", first_attempt.study_id)[-1]
    _terminalize(repository, retry, AttemptStatus.FAILED)

    crashed = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=3)
    )
    restarted = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=4)
    )

    outcomes = repository.list_autoresearch_outcomes("workspace-a", "campaign-1")
    attempts = repository.list_study_attempts("workspace-a", "campaign-1", first_attempt.study_id)
    assert crashed.status == "crash_recorded"
    assert crashed.effect_performed is True
    assert len(outcomes) == 1
    assert outcomes[0].result.outcome == ExperimentOutcome.CRASHED
    assert outcomes[0].decision.decision == ResultDecision.CRASH
    assert outcomes[0].result.attempt_ids == tuple(item.attempt_id for item in attempts)
    assert outcomes[0].result.actual_cost == 0.2
    assert len(attempts) == 2
    assert restarted.effect_performed is False


def test_unknown_attempt_is_left_for_remote_reconciliation(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, attempt = _active_proposal(tmp_path)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_attempts SET status = ? WHERE attempt_id = ?",
            (AttemptStatus.UNKNOWN.value, attempt.attempt_id),
        )
        connection.execute(
            "UPDATE campaign_actions SET status = ? WHERE action_id = ?",
            (ActionStatus.UNKNOWN.value, attempt.action_id),
        )

    tick = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=2)
    )

    attempts = repository.list_study_attempts("workspace-a", "campaign-1", attempt.study_id)
    assert tick.effect_performed is False
    assert [(item.attempt_number, item.status) for item in attempts] == [(1, AttemptStatus.UNKNOWN)]


def test_research_wait_reconnects_by_event_cursor_without_idle_tick_wakes(tmp_path):
    from bashgym.campaigns.autoresearch_loop import (
        AutoResearchLoopCoordinator,
        observe_research_wait,
    )

    repository, core, first_attempt = _active_proposal(tmp_path)
    before_cursor = repository.list_events("workspace-a", "campaign-1")[-1][0]

    initial = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=before_cursor,
    )
    idle_tick = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=1)
    )
    after_idle = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=before_cursor,
    )

    assert initial.status == "waiting"
    assert idle_tick.effect_performed is False
    assert after_idle.status == "waiting"
    assert after_idle.next_cursor == before_cursor

    _terminalize(repository, first_attempt, AttemptStatus.FAILED)
    scheduled = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=2)
    )
    changed = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=before_cursor,
    )
    reconnected = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=changed.next_cursor,
    )

    assert scheduled.status == "retry_scheduled"
    assert changed.status == "changed"
    assert changed.next_cursor > before_cursor
    assert reconnected.status == "waiting"
    assert reconnected.next_cursor == changed.next_cursor


def test_research_wait_never_advances_past_state_it_did_not_observe(tmp_path, monkeypatch):
    """A mutation racing the state read must be reflected before its cursor is returned."""

    from bashgym.campaigns import autoresearch_loop

    _path, repository, core = fresh_core(tmp_path, target=None)
    before_cursor = repository.list_events("workspace-a", "campaign-1")[-1][0]
    prior_cursor = max(0, before_cursor - 1)
    real_latest = autoresearch_loop._latest_event_cursor
    real_state = core.state
    state_reads = 0
    cursor_reads = 0

    def sampled_state(workspace_id, campaign_id):
        nonlocal state_reads
        state_reads += 1
        return real_state(workspace_id, campaign_id)

    def racing_cursor(repo, workspace_id, campaign_id):
        nonlocal cursor_reads
        cursor_reads += 1
        if cursor_reads == 2:
            repository.transition_campaign(
                workspace_id,
                campaign_id,
                CampaignTrigger.CANCEL,
                expected_version=repository.get_campaign(workspace_id, campaign_id).version,
                actor_id="wait-race-test",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id="wait-race-cancel",
                idempotency_key="wait-race-cancel",
            )
        return real_latest(repo, workspace_id, campaign_id)

    monkeypatch.setattr(core, "state", sampled_state)
    monkeypatch.setattr(autoresearch_loop, "_latest_event_cursor", racing_cursor)

    observed = autoresearch_loop.observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=prior_cursor,
    )

    assert observed.next_cursor > before_cursor
    assert observed.state.campaign_status == CampaignStatus.CANCELLING
    assert state_reads >= 2


def test_research_wait_prioritizes_agent_action_and_terminal_state(tmp_path):
    from bashgym.campaigns.autoresearch_loop import observe_research_wait

    _path, repository, core = fresh_core(tmp_path, target=None)
    current_cursor = repository.list_events("workspace-a", "campaign-1")[-1][0]

    action_required = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=current_cursor,
    )

    assert action_required.status == "agent_action_required"
    assert action_required.state.next_action.value == "prepare_campaign"

    cancelling = repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.CANCEL,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        actor_id="autoresearch-loop-test",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="wait-cancel",
        idempotency_key="wait-cancel",
    )
    repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.CANCELLATION_SETTLED,
        expected_version=cancelling.campaign.version,
        actor_id="autoresearch-loop-test",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="wait-cancel-settled",
        idempotency_key="wait-cancel-settled",
    )
    terminal = observe_research_wait(
        repository,
        core,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        after_cursor=current_cursor,
    )

    assert terminal.status == "terminal"
    assert terminal.state.campaign_status == CampaignStatus.CANCELLED


def test_already_derived_stop_is_enforced_on_the_next_tick(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, attempt = _active_proposal(tmp_path, max_attempts=1)
    _terminalize(
        repository,
        attempt,
        AttemptStatus.COMPLETED,
        sealed_uri=str(tmp_path / "sealed-evaluation"),
    )
    AutoResearchLoopCoordinator(repository, _OutcomeWritingProjector(core), core).tick(
        now=NOW + timedelta(seconds=2)
    )

    stopped = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core).tick(
        now=NOW + timedelta(seconds=3)
    )
    campaign = repository.get_campaign("workspace-a", "campaign-1")

    assert stopped.status == "stop_enforced"
    assert stopped.effect_performed is True
    assert campaign.status == CampaignStatus.EXHAUSTED
    assert campaign.stop_reason == "attempt_limit_reached"


def test_completed_evaluation_wins_priority_without_retrying_another_action(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, attempt = _active_proposal(tmp_path)
    _terminalize(
        repository,
        attempt,
        AttemptStatus.COMPLETED,
        sealed_uri=str(tmp_path / "sealed-evaluation"),
    )
    _insert_failed_side_action(repository, attempt)

    tick = AutoResearchLoopCoordinator(repository, _OutcomeWritingProjector(core), core).tick(
        now=NOW + timedelta(seconds=11)
    )

    side_attempts = [
        item
        for item in repository.list_study_attempts("workspace-a", "campaign-1", attempt.study_id)
        if item.action_id == "side-failure"
    ]
    assert tick.status == "evaluation_ingested"
    assert [(item.attempt_number, item.status) for item in side_attempts] == [
        (1, AttemptStatus.FAILED)
    ]


def test_worker_returns_after_coordinator_effect_before_claiming_retry(tmp_path):
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

    repository, core, first_attempt = _active_proposal(tmp_path)
    _terminalize(repository, first_attempt, AttemptStatus.FAILED)
    coordinator = AutoResearchLoopCoordinator(repository, _RejectingProjector(), core)
    worker = CampaignWorker(
        repository,
        tmp_path / "worker-artifacts",
        ArtifactSealer(b"w" * 32, key_version="loop-test-v1"),
        data_directory=tmp_path / "worker-data",
        worker_id="autoresearch-loop-worker",
        autoresearch_loop=coordinator,
    )

    status = worker.run_once(now=NOW + timedelta(seconds=2))
    retry = repository.list_study_attempts("workspace-a", "campaign-1", first_attempt.study_id)[-1]

    assert status == "autoresearch_retry_scheduled"
    assert retry.attempt_number == 2
    assert retry.status == AttemptStatus.SCHEDULED
    assert retry.claim_generation == 0


def test_worker_never_schedules_generic_work_when_autoresearch_is_blocked(tmp_path, monkeypatch):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    worker = CampaignWorker(
        repository,
        tmp_path / "worker-artifacts",
        ArtifactSealer(b"w" * 32, key_version="loop-test-v1"),
        data_directory=tmp_path / "worker-data",
        worker_id="autoresearch-loop-worker",
        autoresearch_loop=_BlockedCoordinator(),
    )

    def unexpected_controller(*_args, **_kwargs):
        raise AssertionError("blocked AutoResearch must not enter generic scheduling")

    monkeypatch.setattr(worker, "controller_once", unexpected_controller)

    assert worker.run_once(now=NOW + timedelta(seconds=2)) == "autoresearch_agent_action_required"


def test_worker_service_builds_one_shared_autoresearch_authority(tmp_path):
    worker = build_worker(config_for(tmp_path), secret_resolver=lambda _reference: "s" * 32)

    assert isinstance(worker.repository, AutoResearchRepository)
    assert worker.autoresearch_loop is not None
    assert worker.autoresearch_loop.repository is worker.repository
    assert worker.autoresearch_loop.core.repository is worker.repository
    assert worker.autoresearch_loop.projector.reader.sealer is worker.sealer


def test_remote_adoption_precedes_coordinator_and_never_relaunches(tmp_path):
    path = tmp_path / "remote-campaigns.sqlite3"
    repository = active_repository(path)
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    first = CampaignWorker(
        repository,
        tmp_path / "remote-artifacts",
        ArtifactSealer(b"w" * 32, key_version="loop-test-v1"),
        data_directory=tmp_path / "remote-data",
        worker_id="worker-before",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, first, plan, tmp_path)
    claimed = repository.claim_next_action(
        first.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=1),
    )
    assert claimed is not None
    asyncio.run(adapter.launch(first._remote_request(claimed)))

    restarted_repository = CampaignRuntimeRepository(path)
    restarted_repository.initialize()
    successor = CampaignWorker(
        restarted_repository,
        tmp_path / "remote-artifacts",
        ArtifactSealer(b"w" * 32, key_version="loop-test-v1"),
        data_directory=tmp_path / "remote-data",
        worker_id="worker-after",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles=first.remote_executor_profiles,
        autoresearch_loop=_NeverTickCoordinator(),
    )

    status = successor.run_once(now=START + timedelta(seconds=17))

    assert status == "remote_running"
    assert adapter.launch_count == 1
    assert (
        restarted_repository.get_attempt("workspace-a", scheduled.attempt_id).claim_generation == 2
    )
