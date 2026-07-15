"""Fenced scheduler/action lease tests."""

from datetime import UTC, datetime, timedelta

import pytest

from bashgym.campaigns.persistence import (
    CampaignRepository,
    LeaseBusyError,
    LeaseLostError,
)


def test_lease_takeover_increments_generation_and_fences_stale_owner(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    first_repository = CampaignRepository(path)
    first_repository.initialize()
    second_repository = CampaignRepository(path)
    second_repository.initialize()
    start = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    first = first_repository.acquire_lease(
        "scheduler:data-root", "worker-a", ttl=timedelta(seconds=15), now=start
    )
    with pytest.raises(LeaseBusyError):
        second_repository.acquire_lease(
            "scheduler:data-root",
            "worker-b",
            ttl=timedelta(seconds=15),
            now=start + timedelta(seconds=5),
        )

    second = second_repository.acquire_lease(
        "scheduler:data-root",
        "worker-b",
        ttl=timedelta(seconds=15),
        now=start + timedelta(seconds=16),
    )

    assert first.generation == 1
    assert second.generation == 2
    with pytest.raises(LeaseLostError):
        first_repository.heartbeat_lease(
            first.lease_key,
            first.owner_id,
            first.generation,
            ttl=timedelta(seconds=15),
            now=start + timedelta(seconds=17),
        )
    with pytest.raises(LeaseLostError):
        first_repository.release_lease(first.lease_key, first.owner_id, first.generation)


def test_current_owner_renews_without_changing_fencing_generation(tmp_path):
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    start = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    acquired = repository.acquire_lease(
        "action:action-1", "worker-a", ttl=timedelta(seconds=15), now=start
    )
    renewed = repository.heartbeat_lease(
        acquired.lease_key,
        acquired.owner_id,
        acquired.generation,
        ttl=timedelta(seconds=15),
        now=start + timedelta(seconds=5),
    )

    assert renewed.generation == acquired.generation
    assert renewed.expires_at == start + timedelta(seconds=20)
    repository.release_lease(
        renewed.lease_key,
        renewed.owner_id,
        renewed.generation,
        now=start + timedelta(seconds=5),
    )
    replacement = repository.acquire_lease(
        renewed.lease_key,
        "worker-b",
        ttl=timedelta(seconds=15),
        now=start + timedelta(seconds=6),
    )
    assert replacement.generation == 2
