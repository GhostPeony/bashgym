from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bashgym.campaigns.autoresearch import (
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
)
from bashgym.campaigns.onboarding import (
    AutoResearchOnboardingConflict,
    AutoResearchOnboardingContract,
    AutoResearchOnboardingCoordinator,
    AutoResearchOnboardingError,
    AutoResearchOnboardingStepResult,
    build_local_onboarding_services,
)
from bashgym.campaigns.remote import RemoteModelRegistrationRequest
from tests.campaigns.test_autoresearch_activation import _activation_fixture


def _contract(root: Path) -> AutoResearchOnboardingContract:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("definition.json", "activation.json", "model-request.json"):
        path = root / name
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    return AutoResearchOnboardingContract(
        onboarding_id="onboarding-demo-v1",
        data_directory=root,
        definition_file=root / "definition.json",
        activation_file=root / "activation.json",
        model_request_file=root / "model-request.json",
        workspace_id="workspace-demo",
        installation_id="ins_0123456789abcdef0123456789abcdef",
        controller_owner_id="autoresearch-controller",
        controller_lease_key_ref="campaign_controller_lease",
        api_base="http://127.0.0.1:8003/api",
        credential_ref="campaign_local_operator",
        campaign_id="campaign-demo",
        campaign_title="Demo experiment",
        stop_rules=AutoResearchStopRules(
            max_attempts=3,
            budget_unit="gpu_hours",
            max_total_cost=2.0,
            minimum_improvement=0.01,
        ),
    )


def _valid_contract(root: Path) -> AutoResearchOnboardingContract:
    definition, activation = _activation_fixture(root / "fixture")
    registered = activation.executor_profile.registered_base_model
    assert registered is not None and registered.artifact_receipt is not None
    request = RemoteModelRegistrationRequest(
        operation="register",
        source_id=registered.source_id,
        compute_profile_id=registered.compute_profile_id,
        target_contract_key=registered.target_contract_key,
        target_model_digest=registered.model_digest,
        model_id=registered.artifact_receipt.model_id,
        revision=registered.artifact_receipt.revision,
        remote_model_path=registered.remote_model_path,
    )
    contract = _contract(root).model_copy(update={"workspace_id": activation.workspace_id})
    contract.definition_file.write_text(definition.model_dump_json(), encoding="utf-8")
    contract.activation_file.write_text(activation.model_dump_json(), encoding="utf-8")
    contract.model_request_file.write_text(request.model_dump_json(), encoding="utf-8")
    return contract


class _RecordingServices:
    def __init__(self, *, final_status: str = "ready", fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.final_status = final_status
        self.fail_on = fail_on

    def run_step(
        self, step: str, contract: AutoResearchOnboardingContract
    ) -> AutoResearchOnboardingStepResult:
        del contract
        self.calls.append(step)
        if step == self.fail_on:
            raise RuntimeError(f"failed:{step}")
        return AutoResearchOnboardingStepResult(
            step=step,
            reference=f"receipt-{step}",
            disposition="completed",
        )

    def campaign_status(self, contract: AutoResearchOnboardingContract) -> str:
        del contract
        self.calls.append("campaign_status")
        return self.final_status

    def reconcile(
        self, contract: AutoResearchOnboardingContract, completed_steps: tuple[str, ...]
    ) -> None:
        del contract
        self.calls.append("reconcile:" + ",".join(completed_steps))


class _ServiceManager:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.calls: list[str] = []

    def status(self, *_args):
        self.calls.append("status")
        return {"supervisor_state": "available" if self.available else "unavailable"}

    def start(self, *_args):
        self.calls.append("start")

    def install(self, *_args):
        self.calls.append("install")

    def replace(self, *_args):
        self.calls.append("replace")


def test_contract_loads_one_bounded_private_json_file(tmp_path: Path) -> None:
    contract_path = tmp_path / "onboarding.json"
    contract_path.write_text(
        json.dumps(_contract(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )

    loaded = AutoResearchOnboardingContract.from_file(contract_path)

    assert loaded == _contract(tmp_path)


def test_contract_digest_binds_the_three_input_files(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    for path, payload in (
        (contract.definition_file, "{}"),
        (contract.activation_file, "{}"),
        (contract.model_request_file, "{}"),
    ):
        path.write_text(payload, encoding="utf-8")
    before = contract.contract_digest

    contract.definition_file.write_text('{"changed":true}', encoding="utf-8")

    assert contract.contract_digest != before


def test_plan_is_pure_and_names_the_existing_boundaries(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    definition = AutoResearchTemplateDefinition.model_validate_json(
        contract.definition_file.read_text(encoding="utf-8")
    )
    assert definition.policy is not None
    services = _RecordingServices()

    plan = AutoResearchOnboardingCoordinator(services).plan(contract)

    assert plan.applied is False
    assert plan.steps == (
        "target_model",
        "activation",
        "resident_services",
        "registry_sync",
        "guided_setup",
        "campaign_prepare",
    )
    assert plan.next_action == "apply_onboarding"
    assert plan.experiment_contract.primary_metric == definition.policy.primary_metric
    assert plan.experiment_contract.metric_direction == definition.policy.metric_direction
    assert plan.experiment_contract.stop_rules == contract.stop_rules
    assert services.calls == []
    assert not (tmp_path / "campaigns" / "onboarding").exists()


def test_plan_rejects_an_invalid_inner_contract_without_writing_state(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    contract.definition_file.write_text("{}", encoding="utf-8")

    with pytest.raises(AutoResearchOnboardingError, match="onboarding_definition_invalid"):
        AutoResearchOnboardingCoordinator.plan(contract)

    assert not (tmp_path / "campaigns" / "onboarding").exists()


def test_plan_rejects_a_model_request_bound_to_another_compute_profile(
    tmp_path: Path,
) -> None:
    contract = _valid_contract(tmp_path)
    request = RemoteModelRegistrationRequest.model_validate_json(
        contract.model_request_file.read_text(encoding="utf-8")
    )
    contract.model_request_file.write_text(
        request.model_copy(update={"compute_profile_id": "other-compute"}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(AutoResearchOnboardingError, match="onboarding_model_request_mismatch"):
        AutoResearchOnboardingCoordinator.plan(contract)


def test_matching_worker_service_restarts_for_the_wrong_controller_owner(
    tmp_path: Path,
) -> None:
    from bashgym.campaigns import onboarding

    definition = SimpleNamespace(
        definition_path=tmp_path / "worker.json",
        definition_payload=b'{"exact":true}\n',
    )
    definition.definition_path.write_bytes(definition.definition_payload)
    manager = _ServiceManager(available=True)

    onboarding._converge_worker_service(
        manager,
        definition,
        SimpleNamespace(controller_owner_id="expected-controller"),
        SimpleNamespace(online=True, owner_id="old-controller"),
    )

    assert manager.calls == ["status", "replace"]


@pytest.mark.parametrize("kind", ("api", "worker"))
def test_matching_service_definition_reinstalls_when_supervisor_is_unavailable(
    kind: str, tmp_path: Path
) -> None:
    from bashgym.campaigns import onboarding

    definition = SimpleNamespace(
        definition_path=tmp_path / f"{kind}.json",
        definition_payload=b'{"exact":true}\n',
    )
    definition.definition_path.write_bytes(definition.definition_payload)
    manager = _ServiceManager(available=False)

    if kind == "api":
        onboarding._converge_api_service(manager, definition)
    else:
        onboarding._converge_worker_service(
            manager,
            definition,
            SimpleNamespace(controller_owner_id="expected-controller"),
            SimpleNamespace(online=False, owner_id=None),
        )

    assert manager.calls == ["status", "replace"]


@pytest.mark.parametrize("kind", ("api", "worker"))
def test_matching_running_service_definition_is_left_undisturbed(kind: str, tmp_path: Path) -> None:
    from bashgym.campaigns import onboarding

    definition = SimpleNamespace(
        definition_path=tmp_path / f"{kind}.json",
        definition_payload=b'{"exact":true}\n',
    )
    definition.definition_path.write_bytes(definition.definition_payload)
    manager = _ServiceManager(available=True)

    if kind == "api":
        onboarding._converge_api_service(manager, definition)
    else:
        onboarding._converge_worker_service(
            manager,
            definition,
            SimpleNamespace(controller_owner_id="expected-controller"),
            SimpleNamespace(online=True, owner_id="expected-controller"),
        )

    assert manager.calls == ["status"]


def test_apply_stops_at_ready_without_start_or_training_steps(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    services = _RecordingServices()

    receipt = AutoResearchOnboardingCoordinator(services).apply(contract)

    assert receipt.applied is True
    assert receipt.campaign_status == "ready"
    assert receipt.next_action == "explicit_start_confirmation_required"
    assert receipt.experiment_contract.stop_rules == contract.stop_rules
    assert tuple(item.step for item in receipt.completed_steps) == (
        "target_model",
        "activation",
        "resident_services",
        "registry_sync",
        "guided_setup",
        "campaign_prepare",
    )
    assert services.calls == [
        "target_model",
        "activation",
        "resident_services",
        "registry_sync",
        "guided_setup",
        "campaign_prepare",
        "campaign_status",
    ]
    serialized = json.loads(
        (tmp_path / "campaigns" / "onboarding" / "onboarding-demo-v1.json").read_text()
    )
    assert serialized["campaign_status"] == "ready"
    assert serialized["experiment_contract"]["stop_rules"] == contract.stop_rules.model_dump(
        mode="json"
    )
    assert {item["step"] for item in serialized["completed_steps"]}.isdisjoint(
        {"start", "training", "launch"}
    )


def test_apply_resumes_after_the_last_completed_step(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    first = _RecordingServices(fail_on="registry_sync")

    with pytest.raises(RuntimeError, match="failed:registry_sync"):
        AutoResearchOnboardingCoordinator(first).apply(contract)

    assert first.calls == [
        "target_model",
        "activation",
        "resident_services",
        "registry_sync",
    ]
    second = _RecordingServices()
    receipt = AutoResearchOnboardingCoordinator(second).apply(contract)

    assert receipt.campaign_status == "ready"
    assert second.calls == [
        "reconcile:target_model,activation,resident_services",
        "registry_sync",
        "guided_setup",
        "campaign_prepare",
        "campaign_status",
    ]


def test_replay_is_idempotent_and_changed_contract_conflicts(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    AutoResearchOnboardingCoordinator(_RecordingServices()).apply(contract)
    replay_services = _RecordingServices()

    replay = AutoResearchOnboardingCoordinator(replay_services).apply(contract)

    assert replay.replayed is True
    assert replay.experiment_contract.stop_rules == contract.stop_rules
    assert replay_services.calls == [
        "reconcile:target_model,activation,resident_services,registry_sync,guided_setup,campaign_prepare",
        "campaign_status",
    ]
    changed = contract.model_copy(update={"campaign_title": "Different experiment"})
    with pytest.raises(AutoResearchOnboardingConflict, match="contract_changed"):
        AutoResearchOnboardingCoordinator(_RecordingServices()).apply(changed)


def test_replay_preserves_completed_onboarding_after_campaign_start(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    AutoResearchOnboardingCoordinator(_RecordingServices()).apply(contract)
    replay_services = _RecordingServices(final_status="active")

    replay = AutoResearchOnboardingCoordinator(replay_services).apply(contract)

    assert replay.applied is True
    assert replay.replayed is True
    assert replay.campaign_status == "active"
    assert replay.next_action == "research_state"
    assert replay_services.calls == [
        "reconcile:target_model,activation,resident_services,registry_sync,guided_setup,campaign_prepare",
        "campaign_status",
    ]


def test_apply_never_claims_success_for_a_non_ready_campaign(tmp_path: Path) -> None:
    services = _RecordingServices(final_status="validating")

    with pytest.raises(AutoResearchOnboardingConflict, match="campaign_not_ready"):
        AutoResearchOnboardingCoordinator(services).apply(_valid_contract(tmp_path))

    receipt = json.loads(
        (tmp_path / "campaigns" / "onboarding" / "onboarding-demo-v1.json").read_text()
    )
    assert receipt["applied"] is False
    assert receipt["campaign_status"] == "validating"


def test_apply_validates_campaign_limits_before_running_any_step(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path).model_copy(
        update={
            "stop_rules": _contract(tmp_path).stop_rules.model_copy(update={"max_attempts": 100})
        }
    )
    services = _RecordingServices()

    with pytest.raises(AutoResearchOnboardingError, match="onboarding_definition_binding_invalid"):
        AutoResearchOnboardingCoordinator(services).apply(contract)

    assert services.calls == []
    assert not (tmp_path / "campaigns" / "onboarding").exists()


def test_local_services_join_existing_boundaries_and_stop_at_ready(
    tmp_path: Path, monkeypatch
) -> None:
    """The production adapter must compose one preparation flow, not another planner."""

    from bashgym.campaigns import onboarding

    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))

    definition, activation = _activation_fixture(tmp_path)
    registered = activation.executor_profile.registered_base_model
    assert registered is not None and registered.artifact_receipt is not None
    request = RemoteModelRegistrationRequest(
        operation="register",
        source_id=registered.source_id,
        compute_profile_id=registered.compute_profile_id,
        target_contract_key=registered.target_contract_key,
        target_model_digest=registered.model_digest,
        model_id=registered.artifact_receipt.model_id,
        revision=registered.artifact_receipt.revision,
        remote_model_path=registered.remote_model_path,
    )
    definition_path = tmp_path / "definition.json"
    activation_path = tmp_path / "activation.json"
    model_request_path = tmp_path / "model-request.json"
    definition_path.write_text(definition.model_dump_json(), encoding="utf-8")
    activation_path.write_text(activation.model_dump_json(), encoding="utf-8")
    model_request_path.write_text(request.model_dump_json(), encoding="utf-8")
    contract = _contract(tmp_path).model_copy(
        update={
            "definition_file": definition_path,
            "activation_file": activation_path,
            "model_request_file": model_request_path,
            "workspace_id": activation.workspace_id,
        }
    )
    physical_calls: list[str] = []
    monkeypatch.setattr(
        onboarding,
        "_register_target_model_on_compute",
        lambda model_request, activation_request: (
            physical_calls.append("target_model") or registered
        ),
    )
    monkeypatch.setattr(
        onboarding,
        "_verify_target_bindings_on_compute",
        lambda *_args, **_kwargs: physical_calls.append("target_bindings_verified"),
    )
    monkeypatch.setattr(
        onboarding,
        "_apply_local_activation",
        lambda *args: physical_calls.append("activation") or {"applied": True},
    )
    monkeypatch.setattr(
        onboarding,
        "_install_local_resident_services",
        lambda *args: physical_calls.append("resident_services") or {"ready": True},
    )
    monkeypatch.setattr(
        onboarding,
        "_sync_local_registry",
        lambda *args: physical_calls.append("registry_sync") or {"ready": True},
    )

    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict | None]] = []

        def request_json(self, method, path, *, query=None, payload=None, headers=None):
            del query, headers
            self.calls.append((method, path, payload))
            if path == "/campaigns/setup/doctor":
                return {"ready": True}
            if path == "/campaigns/setup/validate":
                return {
                    "ready": True,
                    "receipt_id": "setuprcpt_0123456789abcdef0123456789abcdef",
                }
            if path == "/campaigns/setup/create":
                return {"campaign": {"campaign_id": contract.campaign_id, "status": "ready"}}
            if path == f"/campaigns/{contract.campaign_id}":
                return {"campaign_id": contract.campaign_id, "status": "ready"}
            return {"session": {"ready_for_validation": False}}

    client = _Client()
    monkeypatch.setattr(onboarding, "_campaign_client", lambda _: client)

    receipt = AutoResearchOnboardingCoordinator(build_local_onboarding_services(contract)).apply(
        contract
    )

    assert receipt.campaign_status == "ready"
    assert physical_calls == [
        "target_model",
        "target_bindings_verified",
        "activation",
        "resident_services",
        "registry_sync",
    ]
    paths = [path for _method, path, _payload in client.calls]
    assert paths.count("/campaigns/setup/session") == 6
    assert "/campaigns/setup/doctor" in paths
    assert "/campaigns/setup/validate" in paths
    assert "/campaigns/setup/create" in paths
    assert all("start" not in path and "launch" not in path for path in paths)
    private_state = tmp_path / "campaigns" / "onboarding" / "private"
    assert (private_state / "onboarding-demo-v1.target-model.json").is_file()
    assert (private_state / "onboarding-demo-v1.validation.json").is_file()


def test_acquire_never_relabels_an_occupied_target_as_the_requested_model(
    tmp_path: Path, monkeypatch
) -> None:
    """Acquire must execute the acquire request, never register-first at its destination."""

    from types import SimpleNamespace

    from bashgym.campaigns import onboarding, remote

    _definition, activation = _activation_fixture(tmp_path)
    source = activation.executor_profile.registered_base_model
    assert source is not None and source.artifact_receipt is not None
    request = RemoteModelRegistrationRequest(
        operation="acquire",
        source_id=source.source_id,
        compute_profile_id=source.compute_profile_id,
        target_contract_key=source.target_contract_key,
        target_model_digest=source.model_digest,
        model_id=source.artifact_receipt.model_id,
        revision=source.artifact_receipt.revision,
        remote_model_path=source.remote_model_path,
    )
    operations: list[str] = []

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def capacity_preflight(self, _policy):
            return SimpleNamespace(admitted=True)

        async def register_remote_model(self, value):
            operations.append(value.operation)
            return source

    monkeypatch.setattr(remote, "RemoteTrainingAdapter", Adapter)

    assert onboarding._register_target_model_on_compute(request, activation) == source
    assert operations == ["acquire"]


def test_saved_target_receipt_is_reverified_before_onboarding_replay(
    tmp_path: Path, monkeypatch
) -> None:
    from bashgym.campaigns import onboarding

    definition, activation = _activation_fixture(tmp_path)
    source = activation.executor_profile.registered_base_model
    assert source is not None and source.artifact_receipt is not None
    request = RemoteModelRegistrationRequest(
        operation="register",
        source_id=source.source_id,
        compute_profile_id=source.compute_profile_id,
        target_contract_key=source.target_contract_key,
        target_model_digest=source.model_digest,
        model_id=source.artifact_receipt.model_id,
        revision=source.artifact_receipt.revision,
        remote_model_path=source.remote_model_path,
    )
    contract = _contract(tmp_path).model_copy(update={"workspace_id": activation.workspace_id})
    contract.definition_file.write_text(definition.model_dump_json(), encoding="utf-8")
    contract.activation_file.write_text(activation.model_dump_json(), encoding="utf-8")
    contract.model_request_file.write_text(request.model_dump_json(), encoding="utf-8")
    onboarding._write_private_payload(
        onboarding._private_state_path(contract, "target-model"),
        {"request_digest": request.request_digest, "source": source.model_dump(mode="json")},
    )
    verified: list[bool] = []
    monkeypatch.setattr(
        onboarding,
        "_verify_target_bindings_on_compute",
        lambda observed, request, *, verify_model: verified.append(verify_model),
    )

    services = onboarding.LocalAutoResearchOnboardingServices(contract)
    result = services._target_model(contract)

    assert result.disposition == "replayed"
    assert verified == [True]
