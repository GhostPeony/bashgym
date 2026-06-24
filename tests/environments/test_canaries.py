"""Tests for built-in reward-hacking canaries."""

import pytest

from bashgym.environments.canaries import (
    reward_hacking_canaries,
    run_reward_hacking_canaries,
)


def test_reward_hacking_canaries_include_core_categories():
    canaries = reward_hacking_canaries()

    assert {canary.category for canary in canaries} == {
        "env_manifest_tamper",
        "private_fixture_tamper",
        "tests_tamper",
        "verifier_tamper",
    }
    assert all(canary.expected_status == "tampered" for canary in canaries)


def test_reward_hacking_canaries_filter_unknown_category():
    with pytest.raises(ValueError, match="unknown canary categories"):
        reward_hacking_canaries(["not_real"])


def test_run_reward_hacking_canaries_all_guarded(tmp_path):
    canaries, rollouts, summary = run_reward_hacking_canaries(tmp_path, keep_workspace=False)

    assert len(canaries) == 4
    assert len(rollouts) == 4
    assert summary["guarded"] == 4
    assert summary["failed"] == 0
    assert summary["guard_rate"] == pytest.approx(1.0)
    assert all(result["guarded"] for result in summary["results"])
    assert all(rollout.attempt.verifier_status == "tampered" for rollout in rollouts)
    assert all(rollout.workspace.exists() is False for rollout in rollouts)


def test_run_reward_hacking_canaries_can_filter_category(tmp_path):
    canaries, rollouts, summary = run_reward_hacking_canaries(
        tmp_path,
        categories=["verifier_tamper"],
        keep_workspace=False,
    )

    assert [canary.category for canary in canaries] == ["verifier_tamper"]
    assert len(rollouts) == 1
    assert summary["categories"] == {"verifier_tamper": 1}
    assert summary["guarded"] == 1
