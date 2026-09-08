"""Readiness must report an observed service, not a successful supervisor command."""

from unittest.mock import patch

import pytest

from bashgym import studio

PROFILE = {
    "schema_version": "bashgym.studio.v1",
    "workspace_id": "workspace",
    "agent_host": "hermes",
    "credential_ref": "STUDIO_TEST_REF",
    "human_credential_id": "credential-test",
    "api_base": "http://127.0.0.1:8003/api",
}


@pytest.mark.parametrize(
    ("health", "action"),
    [
        ({"healthy": False, "state_root_match": False}, "start_headless_service"),
        (
            {"healthy": True, "state_root_match": False, "studio_compatible": True},
            "connect_matching_state_root",
        ),
        (
            {"healthy": True, "state_root_match": True, "studio_compatible": False},
            "restart_updated_headless_service",
        ),
    ],
)
def test_doctor_prescribes_service_remediation(tmp_path, health, action):
    studio._write_profile(tmp_path, PROFILE)
    with patch("bashgym.studio._health", return_value=health):
        result = studio.doctor(tmp_path)
    assert result["next_action"] == action
    assert result["ready_for_preparation"] is False
    assert result["recipe_verified"] is False


def test_doctor_auth_failure_changes_next_action(tmp_path):
    from bashgym.campaigns.client import CampaignClientError

    studio._write_profile(tmp_path, PROFILE)
    with (
        patch(
            "bashgym.studio._health",
            return_value={"healthy": True, "state_root_match": True, "studio_compatible": True},
        ),
        patch(
            "bashgym.campaigns.client.CampaignApiClient.request_json",
            side_effect=CampaignClientError("campaign_auth_required", "Authentication required"),
        ),
    ):
        result = studio.doctor(tmp_path)
    assert result["next_action"] == "repair_studio_credentials"
    assert result["ready_for_preparation"] is False


def test_doctor_exposes_unselected_recipe_without_claiming_target_test(tmp_path):
    studio._write_profile(tmp_path, PROFILE)
    with (
        patch(
            "bashgym.studio._health",
            return_value={"healthy": True, "state_root_match": True, "studio_compatible": True},
        ),
        patch(
            "bashgym.campaigns.client.CampaignApiClient.request_json",
            return_value={"session": None, "reason_codes": ["setup_session_not_started"]},
        ),
    ):
        result = studio.doctor(tmp_path)
    assert result["ready_for_preparation"] is True
    assert result["recipe_readiness"]["status"] == "not_selected"
    assert result["recipe_readiness"]["execution_verified"] is False


def test_wait_for_service_rejects_wrong_state_without_retrying(tmp_path):
    with patch(
        "bashgym.studio._health",
        return_value={"healthy": True, "state_root_match": False, "studio_compatible": True},
    ) as health:
        with pytest.raises(ValueError, match="studio_api_state_root_mismatch"):
            studio._wait_for_service(tmp_path, timeout_seconds=0.01)
    assert health.call_count == 1


def test_wait_for_service_times_out_without_success(tmp_path):
    with patch("bashgym.studio._health", return_value={"healthy": False}):
        with pytest.raises(ValueError, match="studio_api_unavailable"):
            studio._wait_for_service(tmp_path, timeout_seconds=0)


@pytest.mark.parametrize("matching", [False, True])
def test_init_reuses_matching_service_and_never_replaces_other_service(
    tmp_path, monkeypatch, matching
):
    from bashgym.api import database

    stored = {}
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr("bashgym.secrets.get_secret", stored.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", stored.__setitem__)
    monkeypatch.setattr("bashgym.operator_skills.install_skills", lambda **_: {"verified": True})
    with (
        patch(
            "bashgym.studio._health",
            return_value={"healthy": True, "state_root_match": matching, "studio_compatible": True},
        ),
        patch("bashgym.studio._setup_context", return_value={}) as context,
        patch("bashgym.campaigns.worker_service.ApiServiceManager") as manager,
    ):
        if matching:
            result = studio.initialize(tmp_path)
            assert result["service_verified"] is True
            assert result["next_action"] == "research_prepare"
            context.assert_called_once()
        else:
            with pytest.raises(ValueError, match="studio_api_state_root_mismatch"):
                studio.initialize(tmp_path)
            context.assert_not_called()
        manager.assert_not_called()
