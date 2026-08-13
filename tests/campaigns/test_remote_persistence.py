"""Transactional remote identity and cursor persistence tests."""

from datetime import timedelta

import pytest

from bashgym.campaigns.contracts import AttemptStatus
from bashgym.campaigns.persistence import CampaignPersistenceError
from bashgym.campaigns.remote import (
    RemoteObservation,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamCursor,
)
from bashgym.campaigns.runtime import ActionIdentityMismatchError
from tests.campaigns.test_worker import (
    START,
    active_repository,
    make_worker,
    schedule,
    seed_validated_study,
)


def _identity(attempt_id: str) -> RemoteRunIdentity:
    return RemoteRunIdentity(
        compute_profile_id="ssh-gpu-lab",
        run_id=attempt_id,
        remote_run_directory=f"/home/trainer/bashgym-training/{attempt_id}",
        remote_pid=4242,
        process_group_id=4242,
        process_start_ticks=9001,
        boot_id="boot-1",
        command_hash="a" * 64,
        launch_manifest_sha256="b" * 64,
        launched_at=START,
    )


def _claimed_attempt(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    scheduled = schedule(repository, worker, plan)
    claimed = repository.claim_next_action(
        worker.leader, ttl=timedelta(seconds=15), now=START + timedelta(seconds=1)
    )
    assert claimed is not None
    assert claimed.attempt_id == scheduled.attempt_id
    assert claimed.status == AttemptStatus.RUNNING
    return repository, claimed


def test_remote_identity_registration_is_transactional_idempotent_and_evented(tmp_path):
    repository, attempt = _claimed_attempt(tmp_path)
    identity = _identity(attempt.attempt_id)
    first = repository.register_remote_identity(attempt, identity, now=START + timedelta(seconds=2))
    replay = repository.register_remote_identity(
        attempt, identity, now=START + timedelta(seconds=3)
    )
    assert replay == first
    assert repository.get_remote_run("workspace-a", attempt.attempt_id) == first
    events = repository.list_events("workspace-a", "campaign-1")
    assert sum(event.event_type == "campaign:remote-run-registered" for _, event in events) == 1

    with pytest.raises(ActionIdentityMismatchError):
        repository.register_remote_identity(
            attempt,
            identity.model_copy(update={"remote_pid": 9999}),
            now=START + timedelta(seconds=4),
        )


def test_remote_observation_and_cursors_use_compare_and_swap(tmp_path):
    repository, attempt = _claimed_attempt(tmp_path)
    identity = _identity(attempt.attempt_id)
    record = repository.register_remote_identity(
        attempt, identity, now=START + timedelta(seconds=2)
    )
    observation = RemoteObservation(
        identity=identity,
        state=RemoteRunState.RUNNING,
        observed_at=START + timedelta(seconds=3),
        safe_reason="remote_process_alive",
    )
    updated = repository.update_remote_run(
        record,
        observation,
        metric_cursor=RemoteStreamCursor(byte_offset=120, partial_line="partial"),
        log_cursor=RemoteStreamCursor(byte_offset=240),
        now=START + timedelta(seconds=3),
    )
    assert updated.metric_cursor.byte_offset == 120
    assert updated.log_cursor.byte_offset == 240
    assert updated.last_observation == observation

    with pytest.raises(ActionIdentityMismatchError):
        repository.update_remote_run(record, observation, now=START + timedelta(seconds=4))
    with pytest.raises(CampaignPersistenceError, match="cursor_regression"):
        repository.update_remote_run(
            updated,
            observation,
            metric_cursor=RemoteStreamCursor(byte_offset=119),
            now=START + timedelta(seconds=4),
        )
