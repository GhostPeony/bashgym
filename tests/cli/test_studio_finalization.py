import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from bashgym import studio


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from bashgym.api import database

    stored = {}
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr("bashgym.secrets.get_secret", stored.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", stored.__setitem__)
    monkeypatch.setattr("bashgym.operator_skills.install_skills", lambda **_: {"verified": True})
    studio.initialize(tmp_path, start_service=False)
    profile = studio.read_profile(tmp_path)
    contract = SimpleNamespace(
        data_directory=tmp_path,
        installation_id=profile["installation_id"],
        workspace_id=profile["workspace_id"],
        credential_ref=profile["credential_ref"],
        controller_owner_id="selected-controller",
    )
    return contract, stored, tmp_path / "campaigns" / "campaigns.sqlite3"


def test_finalize_once_and_replay_preserves_sealed_transition(prepared):
    contract, _, path = prepared
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        receipt = connection.execute("SELECT * FROM studio_installation_transitions").fetchall()
        assert connection.execute(
            "SELECT controller_owner_id FROM campaign_recovery_installations"
        ).fetchone() == ("selected-controller",)
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT * FROM studio_installation_transitions").fetchall()
            == receipt
        )
    assert json.loads(receipt[0][1])["actor_id"] == "studio-codex"


def test_finalize_conflicting_concurrent_owners_has_one_winner(prepared):
    contract, _, path = prepared
    other = SimpleNamespace(**{**vars(contract), "controller_owner_id": "another-controller"})

    def apply(value):
        try:
            studio.finalize_installation(value)
            return "applied"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(apply, [contract, other]))
    assert sorted(results) == ["applied", "studio_installation_authority_conflict"]
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM studio_installation_transitions").fetchone()[0]
            == 1
        )


def test_finalize_rejects_registered_provisional_installation(prepared):
    contract, _, path = prepared
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO campaign_recovery_bindings(installation_id,binding_kind,logical_id,availability) "
            "VALUES (?, 'model', 'model-binding', 'reachable')",
            (contract.installation_id,),
        )
    with pytest.raises(ValueError, match="studio_installation_already_in_use"):
        studio.finalize_installation(contract)


def test_finalize_requires_existing_agent_authority(prepared):
    contract, stored, path = prepared
    del stored[contract.credential_ref]
    with pytest.raises(ValueError, match="studio_preparation_authority_unavailable"):
        studio.finalize_installation(contract)
    from bashgym.campaigns.onboarding import _ensure_local_operator_credential

    with pytest.raises(ValueError, match="studio_preparation_authority_unavailable"):
        _ensure_local_operator_credential(contract)
    assert contract.credential_ref not in stored
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT controller_owner_id FROM campaign_recovery_installations")
            .fetchone()[0]
            .startswith("unconfigured:")
        )


def test_finalize_preserves_existing_worker_owner(prepared):
    from bashgym.campaigns.worker_service import WorkerRunConfig, write_worker_config

    contract, _, path = prepared
    config_path = contract.data_directory / "campaigns" / "worker-config.v1.json"
    config = WorkerRunConfig.for_data_directory(contract.data_directory).model_copy(
        update={"controller_owner_id": "existing-controller"}
    )
    write_worker_config(config_path, config)
    before = config_path.read_bytes()
    with pytest.raises(ValueError, match="studio_existing_controller_conflict"):
        studio.finalize_installation(contract)
    assert config_path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT controller_owner_id FROM campaign_recovery_installations")
            .fetchone()[0]
            .startswith("unconfigured:")
        )


def test_finalize_rejects_tampered_transition(prepared):
    contract, _, path = prepared
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE studio_installation_transitions SET seal=?", ("0" * 64,))
    with pytest.raises(ValueError, match="studio_installation_transition_conflict"):
        studio.finalize_installation(contract)


def test_finalize_rejects_deleted_transition(prepared):
    contract, _, path = prepared
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM studio_installation_transitions")
    with pytest.raises(ValueError, match="studio_installation_transition_missing"):
        studio.finalize_installation(contract)


def test_onboarding_replay_verifies_transition_before_physical_operations(prepared, monkeypatch):
    from bashgym.campaigns.onboarding import LocalAutoResearchOnboardingServices

    contract, _, path = prepared
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE studio_installation_transitions SET seal=?", ("0" * 64,))
    services = object.__new__(LocalAutoResearchOnboardingServices)
    calls = []
    monkeypatch.setattr(services, "_target_model", lambda _: calls.append("model"))
    monkeypatch.setattr(
        "bashgym.campaigns.onboarding._install_local_resident_services",
        lambda _: calls.append("services"),
    )
    with pytest.raises(ValueError, match="studio_installation_transition_conflict"):
        services.reconcile(contract, ("target_model", "activation", "resident_services"))
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("installation_id", "ins_11111111111111111111111111111111"),
        ("lease_key_digest", "f" * 64),
        ("old_owner", "another-provisional-owner"),
        ("schema_version", "wrong-schema"),
    ],
)
def test_finalize_rejects_valid_seal_for_wrong_transition_identity(prepared, field, value):
    from bashgym.campaigns.artifacts import ArtifactSealer

    contract, stored, path = prepared
    studio.finalize_installation(contract)
    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM studio_installation_transitions"
            ).fetchone()[0]
        )
        payload[field] = value
        seal = ArtifactSealer(
            stored["BASHGYM_CAMPAIGN_SEAL_KEY"].encode(), key_version="campaign-seal-v1"
        ).sign_canonical_payload(payload, domain="studio-installation-transition-v1")
        connection.execute(
            "UPDATE studio_installation_transitions SET payload_json=?, seal=?",
            (json.dumps(payload), seal),
        )
    with pytest.raises(ValueError, match="studio_installation_transition_conflict"):
        studio.finalize_installation(contract)
