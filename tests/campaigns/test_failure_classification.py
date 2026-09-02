"""Exit-code classification and its effect on settlement and attempt counting."""

from datetime import datetime

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.contracts import FailureClass, ResourceUsage, SealedActionResult
from bashgym.campaigns.failure_classification import classify_exit_code
from bashgym.campaigns.runtime import _settlement_actual_cost

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [
        (126, FailureClass.CONFIGURATION),
        (127, FailureClass.CONFIGURATION),
        (137, FailureClass.INFRASTRUCTURE),
        (143, FailureClass.INFRASTRUCTURE),
        (77, FailureClass.PERMISSION),
        (1, FailureClass.EXECUTION),
        (7, FailureClass.EXECUTION),
        (None, FailureClass.EXECUTION),
    ],
)
def test_classify_exit_code_is_conservative(exit_code, expected) -> None:
    assert classify_exit_code(exit_code) is expected


def _manifest(*, outcome: str, usage: tuple[ResourceUsage, ...]) -> SealedActionResult:
    return SealedActionResult.model_construct(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-1",
        attempt_id="attempt-1",
        manifest_revision=1,
        candidate_digest="a" * 64,
        input_digest="b" * 64,
        claim_generation=1,
        executor_id="campaign-fake-executor",
        executor_version="1",
        compute_profile_id="fake-local",
        remote_process_identity={},
        started_at=NOW,
        ended_at=NOW,
        outcome=outcome,
        exit_code=None if outcome == "completed" else 137,
        exit_reason="test",
        resource_usage=usage,
        log_reference=None,
        outputs=(),
        failure_class=None,
    )


def test_terminal_manifest_without_measured_usage_settles_at_zero() -> None:
    manifest = _manifest(outcome="failed", usage=())

    assert (
        _settlement_actual_cost(unit="gpu_hours", reservation_amount=0.25, manifest=manifest) == 0.0
    )


def test_terminal_manifest_with_measured_usage_still_charges_it() -> None:
    manifest = _manifest(
        outcome="failed",
        usage=(
            ResourceUsage(
                unit="wall_clock_seconds", amount=2.0, source="adapter", confidence="measured"
            ),
        ),
    )

    assert _settlement_actual_cost(
        unit="gpu_hours", reservation_amount=0.25, manifest=manifest
    ) == pytest.approx(2 / 3600)


def test_completed_manifest_without_measured_usage_keeps_the_reservation() -> None:
    manifest = _manifest(outcome="completed", usage=())

    assert (
        _settlement_actual_cost(unit="gpu_hours", reservation_amount=0.25, manifest=manifest)
        == 0.25
    )
