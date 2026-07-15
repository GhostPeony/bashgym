"""Append-only budget reservation, settlement, and overrun tests."""

import pytest

from bashgym.campaigns.contracts import (
    BudgetEntryKind,
    BudgetLedgerEntry,
    CampaignStatus,
    CampaignTrigger,
    CredentialKind,
)
from bashgym.campaigns.persistence import BudgetExceededError, CampaignRepository
from tests.campaigns.test_persistence import create, transition


def active_repository(tmp_path) -> CampaignRepository:
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    create(repository)
    transition(repository, CampaignTrigger.VALIDATE, 1, key="validate")
    transition(repository, CampaignTrigger.VALIDATION_PASSED, 2, key="ready")
    transition(repository, CampaignTrigger.START, 3, key="start")
    return repository


def record(repository, entry, version, key):
    return repository.record_budget_entry(
        entry,
        expected_version=version,
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"correlation-{key}",
        idempotency_key=key,
    )


def test_budget_reservation_is_atomic_bounded_and_idempotent(tmp_path):
    repository = active_repository(tmp_path)
    entry = BudgetLedgerEntry(
        entry_id="budget-reserve-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        unit="gpu_hours",
        kind=BudgetEntryKind.RESERVE,
        reserved_delta=4,
        action_id="action-1",
        actor_id="codex-agent",
    )

    reserved = record(repository, entry, 4, "reserve-1")
    replay = record(repository, entry, 4, "reserve-1")

    assert reserved.campaign.version == 5
    assert replay.replayed is True
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 4.0,
        "actual": 0.0,
        "limit_delta": 0.0,
    }

    too_large = entry.model_copy(update={"entry_id": "budget-reserve-2", "reserved_delta": 9})
    with pytest.raises(BudgetExceededError):
        record(repository, too_large, 5, "reserve-2")
    assert repository.get_campaign("workspace-a", "campaign-1").version == 5


def test_measured_overrun_is_preserved_and_freezes_active_scheduling(tmp_path):
    repository = active_repository(tmp_path)
    reservation = BudgetLedgerEntry(
        entry_id="budget-reserve-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        unit="gpu_hours",
        kind=BudgetEntryKind.RESERVE,
        reserved_delta=12,
        action_id="action-1",
        actor_id="codex-agent",
    )
    record(repository, reservation, 4, "reserve-1")
    settlement = BudgetLedgerEntry(
        entry_id="budget-settle-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        unit="gpu_hours",
        kind=BudgetEntryKind.SETTLE,
        reserved_delta=-12,
        actual_delta=13.25,
        action_id="action-1",
        evidence={"source": "remote-resource-trace", "confidence": "measured"},
        actor_id="codex-agent",
    )

    settled = record(repository, settlement, 5, "settle-1")

    assert settled.campaign.status == CampaignStatus.AWAITING_AUTHORITY
    assert settled.campaign.prior_scheduling_status == CampaignStatus.ACTIVE
    assert settled.event.event_type == "campaign:budget-overrun"
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": 13.25,
        "limit_delta": 0.0,
    }
