from __future__ import annotations

import json

import pytest

from bashgym.campaigns.onboarding import (
    AutoResearchOnboardingConflict,
    AutoResearchOnboardingCoordinator,
)
from bashgym.campaigns.studio_preparation import build_registered_preparation
from tests.campaigns.test_onboarding import _valid_contract


def _registered(root):
    from bashgym.campaigns.activation import AutoResearchActivationRequest
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.auth import CampaignAuthService
    from bashgym.campaigns.autoresearch import (
        AutoResearchRepository,
        AutoResearchTemplateDefinition,
    )
    from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
    from bashgym.campaigns.contracts import AutonomyProfile
    from bashgym.campaigns.guided_setup import GuidedSetupRepository
    from bashgym.campaigns.installation import install_autoresearch_definition
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.campaigns.worker_service import WorkerRunConfig, write_worker_config
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    original = _valid_contract(root)
    definition = AutoResearchTemplateDefinition.model_validate_json(
        original.definition_file.read_text()
    )
    activation = AutoResearchActivationRequest.model_validate_json(
        original.activation_file.read_text()
    )
    database = root / "campaigns" / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()
    auth = CampaignAuthService(repository)
    agent = auth.issue_refresh_credential(
        actor_id="studio-codex",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=(original.workspace_id,),
    )
    secrets = {
        "test_agent": agent.raw_token,
        "BASHGYM_CAMPAIGN_SEAL_KEY": "test-seal-material" * 3,
        "test_lease": scheduler_lease_key(root),
    }
    profile = {
        "schema_version": "bashgym.studio.v1",
        "workspace_id": original.workspace_id,
        "agent_host": "codex",
        "credential_ref": "test_agent",
        "human_credential_id": "human-fixture",
        "installation_id": "ins_" + "b" * 32,
        "api_base": str(original.api_base),
    }
    (root / "studio.v1.json").write_text(json.dumps(profile))
    sealer = ArtifactSealer(
        secrets["BASHGYM_CAMPAIGN_SEAL_KEY"].encode(), key_version="campaign-seal-v1"
    )
    recovery = CampaignRecoveryRepository(database, sealer=sealer)
    recovery.initialize()
    recovery.register_installation(
        installation_id=original.installation_id,
        controller_owner_id=original.controller_owner_id,
        controller_lease_key=scheduler_lease_key(root),
    )
    recovery.register_installation(
        installation_id=profile["installation_id"],
        controller_owner_id="unconfigured:" + profile["installation_id"],
        controller_lease_key=scheduler_lease_key(root),
    )
    install_autoresearch_definition(
        definition, directory=root / "campaigns" / "autoresearch-templates"
    )
    worker = WorkerRunConfig.for_data_directory(
        root,
        approved_remote_profiles=(activation.executor_profile,),
        approved_source_profiles=(activation.source_profile,),
    ).model_copy(update={"controller_owner_id": original.controller_owner_id})
    write_worker_config(root / "campaigns" / "worker-config.v1.json", worker)
    ledger = ExperimentLedgerRepository(database)
    ledger.initialize()
    for method, spec in (
        (ledger.register_project, activation.project),
        (ledger.register_dataset, activation.dataset),
        (ledger.register_dataset_version, activation.dataset_version),
        (ledger.register_evaluation_suite, activation.evaluation_suite),
    ):
        method(spec)
    setup = GuidedSetupRepository(database, sealer=sealer)
    setup.initialize()
    session_id = "setupsess_" + "a" * 32
    setup.advance_session(
        workspace_id=original.workspace_id,
        actor_id="browser-operator",
        definitions={definition.template_id: definition},
        session_id=session_id,
        expected_version=0,
        step="template",
        selection_id=definition.template_id,
        idempotency_key="fixture-template",
        workspace_shared=True,
    )
    setup.advance_session(
        workspace_id=original.workspace_id,
        actor_id="browser-operator",
        definitions={definition.template_id: definition},
        session_id=session_id,
        expected_version=1,
        step="installation",
        selection_id=original.installation_id,
        idempotency_key="fixture-installation",
        workspace_shared=True,
    )
    kwargs = dict(
        workspace_id=original.workspace_id,
        template_id=definition.template_id,
        expected_definition_digest=definition.definition_digest,
        session_id=session_id,
        expected_version=2,
        onboarding_id=original.onboarding_id,
        campaign_id=original.campaign_id,
        campaign_title=original.campaign_title,
        stop_rules=original.stop_rules,
        controller_lease_key_ref="test_lease",
        secret_resolver=secrets.get,
    )
    return original, definition, activation, kwargs, secrets


def _snapshot(root):
    import sqlite3
    from contextlib import closing

    snapshot = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(("-shm", "-wal"))
    }
    database = root / "campaigns" / "campaigns.sqlite3"
    if database.exists():
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            snapshot["__logical_database__"] = tuple(connection.iterdump())
    return snapshot


def test_registered_compiler_exact_inputs_and_pure_plan(tmp_path):
    original, definition, activation, kwargs, _ = _registered(tmp_path)
    before = _snapshot(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before
    assert compiled.session_version == 2
    assert compiled.contract.guided_setup_session_id == kwargs["session_id"]
    assert json.loads(compiled.inputs["definition"]) == definition.model_dump(mode="json")
    assert json.loads(compiled.inputs["activation"]) == activation.model_dump(mode="json")
    assert json.loads(compiled.inputs["model_request"]) == json.loads(
        original.model_request_file.read_text()
    )
    result = compiled.write_inputs()
    assert result["compute_started"] is False
    assert result["onboarding"]["applied"] is False
    assert compiled.contract_file.exists()
    assert (
        AutoResearchOnboardingCoordinator.plan(compiled.contract).experiment_contract.stop_rules
        == original.stop_rules
    )
    assert compiled.write_inputs() == result
    compiled.contract.model_request_file.write_text("{}")
    with pytest.raises(AutoResearchOnboardingConflict, match="input_digest_conflict"):
        AutoResearchOnboardingCoordinator.plan(compiled.contract)


def test_registered_compiler_rejects_stale_draft_without_writes(tmp_path):
    _, _, _, kwargs, _ = _registered(tmp_path)
    before = _snapshot(tmp_path)
    kwargs["expected_version"] = 1
    with pytest.raises(AutoResearchOnboardingConflict, match="version_conflict"):
        build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before


def test_registered_compiler_rejects_revoked_agent_without_issuing_authority(tmp_path):
    import sqlite3

    _, _, _, kwargs, _ = _registered(tmp_path)
    with sqlite3.connect(tmp_path / "campaigns" / "campaigns.sqlite3") as connection:
        connection.execute("UPDATE campaign_actor_credentials SET revoked_at=expires_at")
    before = _snapshot(tmp_path)
    with pytest.raises(AutoResearchOnboardingConflict, match="agent_authority_conflict"):
        build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before


def test_guided_resume_rejects_different_saved_selection_before_remote_work(tmp_path, monkeypatch):
    from bashgym.campaigns import onboarding

    _, definition, _, kwargs, secrets = _registered(tmp_path)
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    compiled.write_inputs()

    class Client:
        def request_json(self, *args, **kwargs):
            return {
                "session": {
                    "workspace_id": compiled.contract.workspace_id,
                    "session_id": compiled.contract.guided_setup_session_id,
                    "version": 2,
                    "completed_steps": ["template", "installation"],
                    "selections": {
                        "template_id": definition.template_id,
                        "installation_id": "ins_" + "b" * 32,
                        "bindings": {},
                    },
                }
            }

    monkeypatch.setattr(onboarding, "_campaign_client", lambda _: Client())
    with pytest.raises(AutoResearchOnboardingConflict, match="guided_setup_digest_conflict"):
        onboarding.LocalAutoResearchOnboardingServices(compiled.contract)


def test_prepare_cli_is_wired_and_reports_missing_inputs(capsys):
    from bashgym import cli

    args = cli.build_parser().parse_args(
        [
            "research",
            "prepare",
            "--template-id",
            "selected",
            "--workspace-id",
            "workspace",
            "--credential-ref",
            "agent",
            "--json",
        ]
    )
    assert args.func is cli.cmd_research_prepare
    assert args.func(args) == 2
    assert "stop_rules" in json.loads(capsys.readouterr().out)["missing_inputs"]


def test_registered_compiler_requires_the_reviewed_template_digest(tmp_path):
    _, _, _, kwargs, _ = _registered(tmp_path)
    kwargs["expected_definition_digest"] = "f" * 64
    before = _snapshot(tmp_path)
    with pytest.raises(AutoResearchOnboardingConflict, match="installed_template_digest_conflict"):
        build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before


def test_prebridge_contract_digest_remains_compatible(tmp_path):
    import hashlib

    from bashgym.campaigns.contracts import canonical_hash

    original = _valid_contract(tmp_path)
    payload = original.model_dump(mode="json")
    for name in (
        "guided_setup_session_id",
        "guided_setup_expected_version",
        "guided_setup_session_digest",
        "expected_input_sha256",
    ):
        payload.pop(name)
    payload["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("definition", original.definition_file),
            ("activation", original.activation_file),
            ("model_request", original.model_request_file),
        )
    }
    assert original.contract_digest == canonical_hash(payload)


def test_prepare_cli_materializes_a_real_registered_contract(tmp_path, monkeypatch, capsys):
    from bashgym import cli

    original, definition, _, kwargs, secrets = _registered(tmp_path)
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    stop_file = tmp_path / "stop-rules.json"
    stop_file.write_text(original.stop_rules.model_dump_json())
    args = cli.build_parser().parse_args(
        [
            "research",
            "prepare",
            "--template-id",
            definition.template_id,
            "--definition-digest",
            definition.definition_digest,
            "--workspace-id",
            original.workspace_id,
            "--credential-ref",
            "test_agent",
            "--api-base",
            str(original.api_base),
            "--expected-version",
            "2",
            "--session-id",
            kwargs["session_id"],
            "--campaign-id",
            original.campaign_id,
            "--onboarding-id",
            original.onboarding_id,
            "--campaign-title",
            original.campaign_title,
            "--stop-rules",
            str(stop_file),
            "--controller-lease-key-ref",
            "test_lease",
            "--write-inputs",
            "--json",
        ]
    )
    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["onboarding"]["applied"] is False
    assert output["binding_plan"]["model_ref"] == definition.target_model.base_model_ref
    assert output["agent_host"] == "codex"
    assert output["compute_started"] is False


def test_saved_draft_can_select_an_existing_nonbootstrap_installation(tmp_path):
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
    from bashgym.campaigns.worker import scheduler_lease_key

    original, _, _, kwargs, secrets = _registered(tmp_path)
    profile_path = tmp_path / "studio.v1.json"
    profile = json.loads(profile_path.read_text())
    profile["installation_id"] = "ins_" + "b" * 32
    profile_path.write_text(json.dumps(profile))
    recovery = CampaignRecoveryRepository(
        tmp_path / "campaigns" / "campaigns.sqlite3",
        sealer=ArtifactSealer(
            secrets["BASHGYM_CAMPAIGN_SEAL_KEY"].encode(), key_version="campaign-seal-v1"
        ),
    )
    recovery.initialize()
    recovery.register_installation(
        installation_id=profile["installation_id"],
        controller_owner_id="unconfigured:" + profile["installation_id"],
        controller_lease_key=scheduler_lease_key(tmp_path),
    )
    compiled = build_registered_preparation(tmp_path, **kwargs)
    assert compiled.contract.installation_id == original.installation_id
    assert compiled.contract.installation_id != profile["installation_id"]


def test_init_owned_provisional_installation_is_only_planned(tmp_path):
    import sqlite3

    original, _, _, kwargs, _ = _registered(tmp_path)
    with sqlite3.connect(tmp_path / "campaigns" / "campaigns.sqlite3") as connection:
        connection.execute(
            "UPDATE campaign_recovery_installations SET controller_owner_id=? WHERE installation_id=?",
            ("unconfigured:" + original.installation_id, original.installation_id),
        )
    profile_path = tmp_path / "studio.v1.json"
    profile = json.loads(profile_path.read_text())
    profile["installation_id"] = original.installation_id
    profile_path.write_text(json.dumps(profile))
    before = _snapshot(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    assert compiled.contract.controller_owner_id == original.controller_owner_id
    assert _snapshot(tmp_path) == before


def test_compiled_contract_uses_existing_coordinator_and_stops_at_ready(tmp_path):
    from tests.campaigns.test_onboarding import _RecordingServices

    _, _, _, kwargs, _ = _registered(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    compiled.write_inputs()
    services = _RecordingServices()
    assert services.calls == []
    receipt = AutoResearchOnboardingCoordinator(services).apply(compiled.contract)
    assert receipt.campaign_status == "ready"
    assert receipt.next_action == "explicit_start_confirmation_required"
    assert all("start" not in call for call in services.calls)


def test_interrupted_input_publication_is_retryable(tmp_path, monkeypatch):
    from bashgym.campaigns import studio_preparation

    _, _, _, kwargs, _ = _registered(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    original_link = studio_preparation.os.link
    calls = []

    def interrupted_link(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected interruption before atomic publication")
        return original_link(source, destination)

    monkeypatch.setattr(studio_preparation.os, "link", interrupted_link)
    with pytest.raises(OSError, match="injected interruption"):
        compiled.write_inputs()
    assert compiled.contract.definition_file.read_bytes() == compiled.inputs["definition"]
    assert not compiled.contract.activation_file.exists()
    assert not compiled.contract_file.exists()
    result = compiled.write_inputs()
    assert result["onboarding"]["applied"] is False
    assert compiled.contract.activation_file.read_bytes() == compiled.inputs["activation"]
    assert compiled.contract_file.exists()
    assert not list(compiled.contract_file.parent.glob(".preparation-*"))


@pytest.mark.parametrize(
    "pins",
    [
        {"guided_setup_expected_version": 2},
        {"guided_setup_session_digest": "a" * 64},
        {"guided_setup_expected_version": 2, "guided_setup_session_digest": "a" * 64},
        {"expected_input_sha256": {"definition": "a" * 64}},
    ],
)
def test_optional_preparation_pins_must_be_coherent(tmp_path, pins):
    from pydantic import ValidationError

    from bashgym.campaigns.onboarding import AutoResearchOnboardingContract

    original = _valid_contract(tmp_path)
    with pytest.raises(ValidationError, match="onboarding_.*pins"):
        AutoResearchOnboardingContract.model_validate(
            {**original.model_dump(mode="python"), **pins}
        )


def _local_services(compiled, definition, secrets, monkeypatch):
    from bashgym.campaigns import onboarding
    from bashgym.campaigns.artifacts import ArtifactSealer
    from bashgym.campaigns.guided_setup import GuidedSetupRepository

    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", secrets.__setitem__)

    class Client:
        def request_json(self, method, path, **kwargs):
            assert (method, path) == ("GET", "/campaigns/setup/context")
            setup = GuidedSetupRepository.open_binding_registry(
                compiled.contract.data_directory / "campaigns" / "campaigns.sqlite3",
                sealer=ArtifactSealer(
                    secrets["BASHGYM_CAMPAIGN_SEAL_KEY"].encode(), key_version="campaign-seal-v1"
                ),
            )
            return setup.context(
                workspace_id=compiled.contract.workspace_id,
                actor_id="studio-codex",
                definitions={definition.template_id: definition},
                session_id=compiled.contract.guided_setup_session_id,
                workspace_shared=True,
            )

    monkeypatch.setattr(onboarding, "_campaign_client", lambda _: Client())
    return onboarding.LocalAutoResearchOnboardingServices(compiled.contract)


@pytest.mark.parametrize("removed", ["approved_remote_profiles", "approved_source_profiles"])
def test_removed_approval_blocks_before_any_physical_step(tmp_path, monkeypatch, removed):
    from bashgym.campaigns import onboarding
    from bashgym.campaigns.worker_service import read_worker_config, write_worker_config

    _, definition, _, kwargs, secrets = _registered(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    compiled.write_inputs()
    services = _local_services(compiled, definition, secrets, monkeypatch)
    physical = []
    monkeypatch.setattr(services, "_target_model", lambda _: physical.append("target_model"))
    worker_path = tmp_path / "campaigns" / "worker-config.v1.json"
    worker = read_worker_config(worker_path)
    write_worker_config(worker_path, worker.model_copy(update={removed: ()}))
    before = _snapshot(tmp_path)
    with pytest.raises(
        AutoResearchOnboardingConflict, match="current_preparation_approval_conflict"
    ):
        AutoResearchOnboardingCoordinator(services).apply(compiled.contract)
    with pytest.raises(
        AutoResearchOnboardingConflict, match="current_preparation_approval_conflict"
    ):
        onboarding.LocalAutoResearchOnboardingServices(compiled.contract)
    assert physical == []
    assert _snapshot(tmp_path) == before


def test_configured_bootstrap_without_transition_is_rejected_read_only(tmp_path):
    original, _, _, kwargs, _ = _registered(tmp_path)
    profile_path = tmp_path / "studio.v1.json"
    profile = json.loads(profile_path.read_text())
    profile["installation_id"] = original.installation_id
    profile_path.write_text(json.dumps(profile))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="studio_installation_transition_missing"):
        build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before


def test_compiled_contract_runs_real_local_activation_with_existing_installation(
    tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from bashgym.campaigns import onboarding, remote

    original, definition, activation, kwargs, secrets = _registered(tmp_path)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    compiled.write_inputs()
    services = _local_services(compiled, definition, secrets, monkeypatch)
    approved = activation.executor_profile.registered_base_model
    inspected = []

    class Session:
        async def run(self, command, **kwargs):
            inspected.append(command)
            return SimpleNamespace(
                exit_status=0, stdout=approved.artifact_receipt.model_dump_json()
            )

    @asynccontextmanager
    async def session(_adapter):
        yield Session()

    async def admitted(_adapter, _policy):
        return SimpleNamespace(admitted=True)

    async def heldout(_adapter, value):
        assert value == activation.executor_profile.registered_evaluation_dataset

    monkeypatch.setattr(remote.RemoteTrainingAdapter, "_session", session)
    monkeypatch.setattr(remote.RemoteTrainingAdapter, "capacity_preflight", admitted)
    monkeypatch.setattr(
        remote.RemoteTrainingAdapter, "verify_registered_evaluation_dataset", heldout
    )
    # Actual register_remote_model parses transport stdout and reconstructs the source;
    # only transport/capacity probes are mocked. No timestamp fields are fabricated.
    services.run_step("target_model", compiled.contract)
    assert len(inspected) == 1
    saved = onboarding._read_private_payload(
        onboarding._private_state_path(compiled.contract, "target-model")
    )
    assert saved["source"] == approved.model_dump(mode="json")
    result = services.run_step("activation", compiled.contract)
    assert result.step == "activation"
    assert result.reference == definition.template_id
    services._check_current_approvals(compiled.contract)
    assert compiled.contract.installation_id == original.installation_id
    assert not (tmp_path / "campaigns" / "services").exists()


def test_partial_compiler_flags_do_not_silently_fall_back_to_context(capsys):
    from bashgym import cli

    args = cli.build_parser().parse_args(
        [
            "research",
            "prepare",
            "--write-inputs",
            "--workspace-id",
            "workspace",
            "--credential-ref",
            "agent",
            "--json",
        ]
    )
    assert args.func(args) == 2
    assert "template_id" in json.loads(capsys.readouterr().out)["missing_inputs"]


@pytest.mark.parametrize("field", ["credential_ref", "controller_lease_key_ref"])
def test_invalid_secret_reference_rejected_before_model_or_client(tmp_path, monkeypatch, field):
    from bashgym.campaigns import onboarding

    contract = _valid_contract(tmp_path).model_copy(update={field: "invalid-ref"})
    calls = []
    monkeypatch.setattr(onboarding, "_campaign_client", lambda _: calls.append("client"))
    monkeypatch.setattr(
        onboarding, "_register_target_model_on_compute", lambda *_: calls.append("model")
    )
    with pytest.raises(
        onboarding.AutoResearchOnboardingError, match="onboarding_" + field + "_invalid"
    ):
        onboarding.LocalAutoResearchOnboardingServices(contract)
    assert calls == []
    assert not (tmp_path / "campaigns" / "onboarding").exists()


def test_compiler_rejects_invalid_secret_reference_before_materialization(tmp_path):
    from bashgym.campaigns.onboarding import AutoResearchOnboardingError

    _, _, _, kwargs, _ = _registered(tmp_path)
    kwargs["controller_lease_key_ref"] = "invalid-ref"
    before = _snapshot(tmp_path)
    with pytest.raises(AutoResearchOnboardingError, match="controller_lease_key_ref_invalid"):
        build_registered_preparation(tmp_path, **kwargs)
    assert _snapshot(tmp_path) == before
