from unittest.mock import patch

from bashgym import cli


def test_init_and_doctor_are_discoverable():
    parser = cli.build_parser()
    args = parser.parse_args(["init", "--agent-host", "hermes", "--no-service", "--json"])
    assert args.agent_host == "hermes"
    assert args.no_service is True
    assert parser.parse_args(["doctor", "--json"]).func is not None
    assert (
        parser.parse_args(
            [
                "training",
                "compare-recipes",
                "--baseline",
                "base.json",
                "--candidate",
                "candidate.json",
            ]
        ).func
        == cli.cmd_training_compare_recipes
    )


def test_headless_health_does_not_claim_recipe_certification(tmp_path):
    from bashgym.studio import doctor

    with patch("bashgym.studio._health", return_value={"healthy": True, "state_root_match": True}):
        result = doctor(tmp_path)
    assert result["recipe_verified"] is False
    assert result["checked_at"]
    assert result["next_action"] == "restart_updated_headless_service"
    assert result["service_compatible"] is False


def test_studio_rejects_rebinding_workspace(tmp_path):
    import json

    import pytest

    from bashgym.studio import read_profile

    (tmp_path / "studio.v1.json").write_text(json.dumps({"schema_version": "invalid"}))
    with pytest.raises(ValueError, match="studio_profile_invalid"):
        read_profile(tmp_path)


def test_init_replays_without_reissuing_authority(tmp_path, monkeypatch):
    import sqlite3

    from bashgym import studio
    from bashgym.api import database
    from bashgym.campaigns.auth import CampaignAuthService
    from bashgym.campaigns.autoresearch import AutoResearchRepository

    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "unused.db")
    secrets = {}
    with (
        patch("bashgym.operator_skills.install_skills", return_value={"ok": True}),
        patch("bashgym.secrets.set_secret", side_effect=secrets.__setitem__),
        patch("bashgym.secrets.get_secret", side_effect=secrets.get),
    ):
        first = studio.initialize(
            tmp_path, workspace_id="workspace", agent_host="hermes", start_service=False
        )
        profile = studio.read_profile(tmp_path)
        original_secrets = dict(secrets)
        with sqlite3.connect(tmp_path / "campaigns" / "campaigns.sqlite3") as connection:
            connection.execute(
                "UPDATE campaign_recovery_installations SET controller_owner_id='configured-controller'"
            )
        second = studio.initialize(
            tmp_path, workspace_id="workspace", agent_host="hermes", start_service=False
        )
        assert secrets == original_secrets
        with sqlite3.connect(tmp_path / "campaigns" / "campaigns.sqlite3") as connection:
            assert connection.execute(
                "SELECT controller_owner_id FROM campaign_recovery_installations"
            ).fetchall() == [("configured-controller",)]
    assert first["replayed"] is False
    assert second["replayed"] is True
    assert first["training_started"] is False
    assert studio.read_profile(tmp_path) == profile
    public_state = (tmp_path / "studio.v1.json").read_text() + str(first)
    assert all(secret not in public_state for secret in secrets.values())
    session = database.consume_local_pairing(second["pairing_code"])
    grant = database.get_local_session_grant(session)
    repository = AutoResearchRepository(tmp_path / "campaigns" / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    principal = auth.authenticate_local_session(grant)
    assert principal.workspace_ids == ("workspace",)
    auth.revoke_credential(principal.credential_id, reason="test")
    import pytest

    with pytest.raises(PermissionError):
        auth.authenticate_local_session(grant)


def test_clean_init_can_prepare_and_resume_setup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from bashgym import studio
    from bashgym.api import database
    from bashgym.api.routes import create_app

    secrets = {}
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setenv("BASHGYM_MODE", "headless")
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "unused.db")
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", secrets.__setitem__)
    monkeypatch.setattr("bashgym.api.campaign_routes.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.operator_skills.install_skills", lambda **_: {"ok": True})
    result = studio.initialize(tmp_path, start_service=False)
    http = TestClient(create_app(), base_url="http://localhost")
    headers = {"X-Requested-With": "XMLHttpRequest"}
    paired = http.post(
        "/api/auth/local/pair", json={"code": result["pairing_code"]}, headers=headers
    )
    assert paired.status_code == 200, paired.text
    response = http.get("/api/campaigns/setup/context", params={"workspace_id": "personal"})
    assert response.status_code == 200, response.text
    context = response.json()
    assert len(context["installations"]) == 1
    installation = context["installations"][0]
    assert installation["ready"] is False
    assert all(not bindings for bindings in installation["bindings"].values())
    assert len(installation["reason_codes"]) == 4
    session_id = "setupsess_0123456789abcdef0123456789abcdef"
    for version, (step, selection) in enumerate(
        [
            ("template", context["templates"][0]["template_id"]),
            ("installation", installation["installation_id"]),
        ]
    ):
        response = http.post(
            "/api/campaigns/setup/session",
            json={
                "workspace_id": "personal",
                "session_id": session_id,
                "expected_version": version,
                "step": step,
                "selection_id": selection,
            },
            headers={**headers, "Idempotency-Key": f"init-step-{version}"},
        )
        assert response.status_code == 200, response.text
    resumed = http.get(
        "/api/campaigns/setup/context",
        params={
            "workspace_id": "personal",
            "session_id": session_id,
        },
    )
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["session"] == response.json()["session"]
    assert resumed.json()["session"]["ready_for_validation"] is False
    assert result["training_started"] is False
    assert not list(tmp_path.rglob("worker*.json"))
    # Even a pre-bootstrap profile cannot replace authority for persisted steps.
    profile = studio.read_profile(tmp_path)
    del profile["installation_id"]
    studio._write_profile(tmp_path, profile)
    del secrets["BASHGYM_CAMPAIGN_SEAL_KEY"]
    import pytest

    with pytest.raises(ValueError, match="studio_seal_authority_unavailable"):
        studio.initialize(tmp_path, start_service=False)
    assert "BASHGYM_CAMPAIGN_SEAL_KEY" not in secrets


def test_init_does_not_replace_missing_bootstrapped_seal(tmp_path, monkeypatch):
    import pytest

    from bashgym import studio
    from bashgym.api import database

    secrets = {}
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "unused.db")
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", secrets.__setitem__)
    monkeypatch.setattr("bashgym.operator_skills.install_skills", lambda **_: {"ok": True})
    studio.initialize(tmp_path, start_service=False)
    original_profile = studio.read_profile(tmp_path)
    del secrets["BASHGYM_CAMPAIGN_SEAL_KEY"]
    with pytest.raises(ValueError, match="studio_seal_authority_unavailable"):
        studio.initialize(tmp_path, start_service=False)
    assert studio.read_profile(tmp_path) == original_profile
    assert "BASHGYM_CAMPAIGN_SEAL_KEY" not in secrets
