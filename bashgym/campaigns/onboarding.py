"""Deterministic, resumable preparation of one AutoResearch campaign.

This module coordinates existing installation boundaries. It does not select a
model, invent an experiment, or start compute. Physical setup is supplied by a
small services adapter so the same coordinator can be exercised without SSH,
services, or an API in unit tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, HttpUrl

from bashgym.campaigns.autoresearch import AutoResearchStopRules, MetricDirection
from bashgym.campaigns.contracts import (
    CampaignStatus,
    FrozenContractModel,
    Identifier,
    canonical_hash,
)
from bashgym.campaigns.method_policy import AutoResearchMethodThresholds

_ONBOARDING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_MAX_RECEIPT_BYTES = 1024 * 1024
ONBOARDING_STEPS = (
    "target_model",
    "activation",
    "resident_services",
    "registry_sync",
    "guided_setup",
    "campaign_prepare",
)


class AutoResearchOnboardingError(ValueError):
    """Stable onboarding validation failure."""


class AutoResearchOnboardingConflict(AutoResearchOnboardingError):  # noqa: N818
    """An onboarding identity or persisted result conflicts."""


class AutoResearchOnboardingContract(FrozenContractModel):
    """Private local inputs for a deterministic preparation run.

    Paths and transport details remain in this ignored operator contract. The
    persisted public-facing receipt contains only its digest and opaque IDs.
    """

    schema_version: Literal["autoresearch_onboarding_contract.v1"] = (
        "autoresearch_onboarding_contract.v1"
    )
    onboarding_id: Identifier
    data_directory: Path
    definition_file: Path
    activation_file: Path
    model_request_file: Path
    workspace_id: Identifier
    installation_id: str = Field(pattern=r"^ins_[0-9a-f]{32}$")
    controller_owner_id: Identifier
    controller_lease_key_ref: Identifier
    api_base: HttpUrl
    credential_ref: Identifier
    campaign_id: Identifier
    campaign_title: str = Field(min_length=1, max_length=240)
    stop_rules: AutoResearchStopRules

    @classmethod
    def from_file(cls, path: Path) -> AutoResearchOnboardingContract:
        source = path.expanduser().resolve()
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size > _MAX_RECEIPT_BYTES
        ):
            raise AutoResearchOnboardingError("onboarding_contract_invalid")
        try:
            return cls.model_validate_json(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AutoResearchOnboardingError("onboarding_contract_invalid") from exc

    @property
    def contract_digest(self) -> str:
        payload = self.model_dump(mode="json")
        payload["input_sha256"] = {
            name: _bounded_file_sha256(path)
            for name, path in (
                ("definition", self.definition_file),
                ("activation", self.activation_file),
                ("model_request", self.model_request_file),
            )
        }
        return canonical_hash(payload)


class AutoResearchOnboardingStepResult(FrozenContractModel):
    schema_version: Literal["autoresearch_onboarding_step_result.v1"] = (
        "autoresearch_onboarding_step_result.v1"
    )
    step: Literal[
        "target_model",
        "activation",
        "resident_services",
        "registry_sync",
        "guided_setup",
        "campaign_prepare",
    ]
    reference: Identifier
    disposition: Literal["completed", "replayed"]


class AutoResearchOnboardingExperimentContract(FrozenContractModel):
    """The exact scientific and resource contract shown before Start."""

    schema_version: Literal["autoresearch_onboarding_experiment_contract.v1"] = (
        "autoresearch_onboarding_experiment_contract.v1"
    )
    primary_metric: Identifier
    metric_direction: MetricDirection
    stop_rules: AutoResearchStopRules
    method_thresholds: AutoResearchMethodThresholds = Field(
        default_factory=AutoResearchMethodThresholds
    )


class AutoResearchOnboardingPlan(FrozenContractModel):
    schema_version: Literal["autoresearch_onboarding_plan.v1"] = "autoresearch_onboarding_plan.v1"
    onboarding_id: Identifier
    contract_digest: str
    applied: Literal[False] = False
    steps: tuple[str, ...]
    experiment_contract: AutoResearchOnboardingExperimentContract
    next_action: Literal["apply_onboarding"] = "apply_onboarding"


class AutoResearchOnboardingReceipt(FrozenContractModel):
    schema_version: Literal["autoresearch_onboarding_receipt.v1"] = (
        "autoresearch_onboarding_receipt.v1"
    )
    onboarding_id: Identifier
    contract_digest: str
    applied: bool = False
    replayed: bool = False
    completed_steps: tuple[AutoResearchOnboardingStepResult, ...] = ()
    campaign_id: Identifier
    experiment_contract: AutoResearchOnboardingExperimentContract
    campaign_status: str = "preparing"
    next_action: Literal[
        "resume_onboarding",
        "explicit_start_confirmation_required",
        "research_state",
    ] = "resume_onboarding"


class AutoResearchOnboardingServices(Protocol):
    """Physical boundary implementations used by the coordinator."""

    def run_step(
        self, step: str, contract: AutoResearchOnboardingContract
    ) -> AutoResearchOnboardingStepResult: ...

    def campaign_status(self, contract: AutoResearchOnboardingContract) -> str: ...

    def reconcile(
        self,
        contract: AutoResearchOnboardingContract,
        completed_steps: tuple[str, ...],
    ) -> None: ...


def _bounded_file_sha256(path: Path) -> str:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_RECEIPT_BYTES:
        raise AutoResearchOnboardingError("onboarding_input_invalid")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_onboarding_model(path: Path, model: Any, *, error_code: str) -> Any:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_RECEIPT_BYTES:
        raise AutoResearchOnboardingError(error_code)
    try:
        payload = source.read_bytes()
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise AutoResearchOnboardingError(error_code)
        return model.model_validate_json(payload)
    except AutoResearchOnboardingError:
        raise
    except (OSError, ValueError) as exc:
        raise AutoResearchOnboardingError(error_code) from exc


def validate_local_onboarding_contract(
    contract: AutoResearchOnboardingContract,
) -> tuple[Any, Any, Any]:
    """Validate all local inputs and their exact logical bindings without I/O effects."""

    from bashgym.campaigns.activation import (
        AutoResearchActivationRequest,
        _validate_definition_bindings,
    )
    from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition
    from bashgym.campaigns.remote import RemoteModelRegistrationRequest

    definition = _read_onboarding_model(
        contract.definition_file,
        AutoResearchTemplateDefinition,
        error_code="onboarding_definition_invalid",
    )
    activation = _read_onboarding_model(
        contract.activation_file,
        AutoResearchActivationRequest,
        error_code="onboarding_activation_invalid",
    )
    model_request = _read_onboarding_model(
        contract.model_request_file,
        RemoteModelRegistrationRequest,
        error_code="onboarding_model_request_invalid",
    )
    if activation.workspace_id != contract.workspace_id:
        raise AutoResearchOnboardingError("onboarding_workspace_mismatch")
    try:
        _validate_definition_bindings(definition, activation)
        definition.validate_campaign_stop_rules(contract.stop_rules)
    except ValueError as exc:
        raise AutoResearchOnboardingError("onboarding_definition_binding_invalid") from exc

    executor = activation.executor_profile
    registered = executor.registered_base_model
    artifact = registered.artifact_receipt if registered is not None else None
    if (
        registered is None
        or artifact is None
        or model_request.source_id != registered.source_id
        or model_request.compute_profile_id != executor.compute_profile_id
        or model_request.compute_profile_id != registered.compute_profile_id
        or model_request.target_contract_key != executor.target_contract_key
        or model_request.target_contract_key != registered.target_contract_key
        or model_request.target_model_digest != executor.target_model_digest
        or model_request.target_model_digest != registered.model_digest
        or model_request.remote_model_path != registered.remote_model_path
        or model_request.model_id != artifact.model_id
        or model_request.revision != artifact.revision
    ):
        raise AutoResearchOnboardingError("onboarding_model_request_mismatch")
    return definition, activation, model_request


def _private_state_path(contract: AutoResearchOnboardingContract, label: str) -> Path:
    return (
        contract.data_directory.expanduser().resolve()
        / "campaigns"
        / "onboarding"
        / "private"
        / f"{contract.onboarding_id}.{label}.json"
    )


def _write_private_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise AutoResearchOnboardingError("onboarding_private_receipt_too_large")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise AutoResearchOnboardingConflict("onboarding_private_receipt_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AutoResearchOnboardingConflict("onboarding_private_receipt_invalid") from exc
    if not isinstance(value, dict):
        raise AutoResearchOnboardingConflict("onboarding_private_receipt_invalid")
    return value


def _register_target_model_on_compute(model_request, activation_request):
    from bashgym.campaigns.remote import RemoteTrainingAdapter
    from bashgym.gym.remote_trainer import SSHConfig

    profile = activation_request.executor_profile
    adapter = RemoteTrainingAdapter(
        SSHConfig(
            host=profile.host,
            username=profile.username,
            port=profile.port,
            key_path=profile.key_path,
            remote_work_dir=profile.remote_work_dir,
        ),
        compute_profile_id=profile.compute_profile_id,
    )

    async def register():
        from bashgym.campaigns.contracts import StageKind

        capacity = await adapter.capacity_preflight(
            profile.stage_profile(StageKind.FULL_TRAINING).capacity_policy
        )
        if not capacity.admitted:
            raise AutoResearchOnboardingError("onboarding_compute_preflight_blocked")
        return await adapter.register_remote_model(model_request)

    return asyncio.run(register())


def _verify_target_bindings_on_compute(
    source,
    activation_request,
    *,
    verify_model: bool,
) -> None:
    """Recheck target-resident model/data without moving their contents."""

    from bashgym.campaigns.remote import RemoteTrainingAdapter
    from bashgym.gym.remote_trainer import SSHConfig

    profile = activation_request.executor_profile
    heldout = profile.registered_evaluation_dataset
    if heldout is None:
        raise AutoResearchOnboardingError("onboarding_evaluation_dataset_missing")
    adapter = RemoteTrainingAdapter(
        SSHConfig(
            host=profile.host,
            username=profile.username,
            port=profile.port,
            key_path=profile.key_path,
            remote_work_dir=profile.remote_work_dir,
        ),
        compute_profile_id=profile.compute_profile_id,
    )

    async def verify() -> None:
        if verify_model:
            await adapter.verify_registered_base_model(source)
        await adapter.verify_registered_evaluation_dataset(heldout)

    asyncio.run(verify())


def _definition_matches(path: Path, expected: bytes) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise AutoResearchOnboardingConflict("resident_service_definition_invalid")
    try:
        return path.read_bytes() == expected
    except OSError as exc:
        raise AutoResearchOnboardingConflict("resident_service_definition_invalid") from exc


def _converge_api_service(manager, definition) -> None:
    if not _definition_matches(definition.definition_path, definition.definition_payload):
        if definition.definition_path.exists():
            manager.replace(definition)
        else:
            manager.install(definition)
        return
    if manager.status(definition).get("supervisor_state") != "available":
        manager.replace(definition)


def _worker_controller_ready(config, controller) -> bool:
    return bool(
        config.controller_owner_id is not None
        and controller.online
        and controller.owner_id == config.controller_owner_id
    )


def _converge_worker_service(manager, definition, config, controller) -> None:
    if not _definition_matches(definition.definition_path, definition.definition_payload):
        if definition.definition_path.exists():
            manager.replace(definition, config)
        else:
            manager.install(definition, config)
        return
    if manager.status(definition, controller).get(
        "supervisor_state"
    ) != "available" or not _worker_controller_ready(config, controller):
        manager.replace(definition, config)


def _apply_local_activation(definition, activation_request, contract):
    from bashgym import secrets as secret_store
    from bashgym.campaigns.activation import activate_autoresearch
    from bashgym.campaigns.installation import install_autoresearch_definition
    from bashgym.campaigns.worker_service import read_worker_config, write_worker_config

    root = contract.data_directory.expanduser().resolve()
    install_autoresearch_definition(
        definition,
        directory=root / "campaigns" / "autoresearch-templates",
    )
    receipt = activate_autoresearch(
        definition,
        activation_request,
        data_directory=root,
        apply=True,
        install_service=False,
        secret_resolver=secret_store.get_secret,
        secret_writer=secret_store.set_secret,
    )
    config_path = root / "campaigns" / "worker-config.v1.json"
    config = read_worker_config(config_path).model_copy(
        update={"controller_owner_id": contract.controller_owner_id}
    )
    write_worker_config(config_path, config)
    return receipt.model_dump(mode="json")


def _api_origin(contract: AutoResearchOnboardingContract) -> tuple[str, int, str]:
    parsed = urllib.parse.urlsplit(str(contract.api_base))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path.rstrip("/") != "/api"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AutoResearchOnboardingError("onboarding_api_base_invalid")
    return parsed.hostname, parsed.port or 80, f"{parsed.scheme}://{parsed.netloc}"


def _ensure_local_operator_credential(contract: AutoResearchOnboardingContract) -> None:
    from bashgym import secrets as secret_store
    from bashgym.campaigns.auth import CampaignAuthService
    from bashgym.campaigns.autoresearch import AutoResearchRepository
    from bashgym.campaigns.contracts import AutonomyProfile
    from bashgym.mcp.policy import validate_secret_ref_name

    validate_secret_ref_name(contract.credential_ref)
    if secret_store.get_secret(contract.credential_ref) is not None:
        return
    repository = AutoResearchRepository(
        contract.data_directory.expanduser().resolve() / "campaigns" / "campaigns.sqlite3"
    )
    repository.initialize()
    issued = CampaignAuthService(repository).issue_refresh_credential(
        actor_id=contract.controller_owner_id,
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=(contract.workspace_id,),
    )
    secret_store.set_secret(contract.credential_ref, issued.raw_token)


def _install_local_resident_services(contract: AutoResearchOnboardingContract):
    from bashgym import secrets as secret_store
    from bashgym.campaigns.persistence import CampaignRepository
    from bashgym.campaigns.worker import scheduler_lease_key
    from bashgym.campaigns.worker_service import (
        ApiServiceManager,
        WorkerServiceManager,
        build_api_service_definition,
        build_service_definition,
        project_controller_status,
        read_worker_config,
    )
    from bashgym.mcp.policy import validate_secret_ref_name

    root = contract.data_directory.expanduser().resolve()
    validate_secret_ref_name(contract.controller_lease_key_ref)
    expected_lease_key = scheduler_lease_key(root)
    existing_lease_key = secret_store.get_secret(contract.controller_lease_key_ref)
    if existing_lease_key is None:
        secret_store.set_secret(contract.controller_lease_key_ref, expected_lease_key)
    elif existing_lease_key != expected_lease_key:
        raise AutoResearchOnboardingConflict("controller_lease_authority_conflict")
    _ensure_local_operator_credential(contract)

    host, port, origin = _api_origin(contract)
    api_definition = build_api_service_definition(
        host=host,
        port=port,
        data_directory=root,
    )
    api_manager = ApiServiceManager()
    _converge_api_service(api_manager, api_definition)

    health_url = f"{origin}/api/health"
    deadline = time.monotonic() + 30
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                from bashgym.config import state_root_digest

                payload = json.loads(response.read(8192).decode("utf-8"))
                if (
                    response.status == 200
                    and isinstance(payload, dict)
                    and payload.get("state_root_digest") == state_root_digest(root)
                ):
                    break
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise AutoResearchOnboardingError("onboarding_api_not_ready")
        time.sleep(0.25)

    config_path = root / "campaigns" / "worker-config.v1.json"
    config = read_worker_config(config_path)
    if config.controller_owner_id != contract.controller_owner_id:
        raise AutoResearchOnboardingConflict("controller_owner_identity_conflict")
    worker_definition = build_service_definition(config_path)
    worker_manager = WorkerServiceManager()
    repository = CampaignRepository(config.database_path)
    repository.initialize()
    deadline = time.monotonic() + 30
    controller = project_controller_status(repository, root)
    _converge_worker_service(worker_manager, worker_definition, config, controller)
    while not _worker_controller_ready(config, controller) and time.monotonic() < deadline:
        time.sleep(0.25)
        controller = project_controller_status(repository, root)
    if not _worker_controller_ready(config, controller):
        raise AutoResearchOnboardingError("onboarding_worker_not_ready")
    return {"api": "ready", "worker": "ready"}


def _sync_local_registry(definition, contract):
    from bashgym import secrets as secret_store
    from bashgym.campaigns.registry_sync import (
        apply_autoresearch_registry_sync,
        plan_autoresearch_registry_sync,
    )
    from bashgym.campaigns.worker_service import read_worker_config
    from bashgym.ledger.persistence import ExperimentLedgerRepository

    root = contract.data_directory.expanduser().resolve()
    database_path = root / "campaigns" / "campaigns.sqlite3"
    lease_key = secret_store.get_secret(contract.controller_lease_key_ref)
    if not lease_key:
        raise AutoResearchOnboardingError("controller_lease_authority_unavailable")
    plan = plan_autoresearch_registry_sync(
        definitions=(definition,),
        workspace_id=contract.workspace_id,
        installation_id=contract.installation_id,
        ledger=ExperimentLedgerRepository.open_existing(database_path),
        worker_config=read_worker_config(root / "campaigns" / "worker-config.v1.json"),
    )
    if not plan.ready:
        raise AutoResearchOnboardingError("onboarding_registry_not_ready")
    return apply_autoresearch_registry_sync(
        plan,
        database_path=database_path,
        controller_owner_id=contract.controller_owner_id,
        controller_lease_key=lease_key,
    ).model_dump(mode="json")


def _campaign_client(contract: AutoResearchOnboardingContract):
    from bashgym.campaigns.client import CampaignApiClient

    return CampaignApiClient(
        api_base=str(contract.api_base).rstrip("/"),
        credential_ref=contract.credential_ref,
    )


def _receipt_path(contract: AutoResearchOnboardingContract) -> Path:
    if _ONBOARDING_ID.fullmatch(contract.onboarding_id) is None:
        raise AutoResearchOnboardingError("onboarding_id_invalid")
    return (
        contract.data_directory.expanduser().resolve()
        / "campaigns"
        / "onboarding"
        / f"{contract.onboarding_id}.json"
    )


def _read_receipt(path: Path) -> AutoResearchOnboardingReceipt | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise AutoResearchOnboardingConflict("receipt_invalid")
    try:
        return AutoResearchOnboardingReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AutoResearchOnboardingConflict("receipt_invalid") from exc


def _write_receipt(path: Path, receipt: AutoResearchOnboardingReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AutoResearchOnboardingCoordinator:
    """Run at most the declared preparation steps and stop at READY."""

    def __init__(self, services: AutoResearchOnboardingServices) -> None:
        self._services = services

    @staticmethod
    def plan(contract: AutoResearchOnboardingContract) -> AutoResearchOnboardingPlan:
        definition, _activation, _model_request = validate_local_onboarding_contract(contract)
        assert definition.policy is not None
        return AutoResearchOnboardingPlan(
            onboarding_id=contract.onboarding_id,
            contract_digest=contract.contract_digest,
            steps=ONBOARDING_STEPS,
            experiment_contract=AutoResearchOnboardingExperimentContract(
                primary_metric=definition.policy.primary_metric,
                metric_direction=definition.policy.metric_direction,
                stop_rules=contract.stop_rules,
                method_thresholds=definition.policy.method_thresholds,
            ),
        )

    def apply(self, contract: AutoResearchOnboardingContract) -> AutoResearchOnboardingReceipt:
        definition, _activation, _model_request = validate_local_onboarding_contract(contract)
        assert definition.policy is not None
        experiment_contract = AutoResearchOnboardingExperimentContract(
            primary_metric=definition.policy.primary_metric,
            metric_direction=definition.policy.metric_direction,
            stop_rules=contract.stop_rules,
            method_thresholds=definition.policy.method_thresholds,
        )
        path = _receipt_path(contract)
        contract_digest = contract.contract_digest
        existing = _read_receipt(path)
        if existing is not None and existing.contract_digest != contract_digest:
            raise AutoResearchOnboardingConflict("contract_changed")

        completed = list(existing.completed_steps if existing is not None else ())
        completed_names = tuple(item.step for item in completed)
        if completed_names != ONBOARDING_STEPS[: len(completed_names)]:
            raise AutoResearchOnboardingConflict("receipt_step_order_invalid")

        if existing is not None and completed_names:
            self._services.reconcile(contract, completed_names)
        if existing is not None and existing.applied:
            status = self._services.campaign_status(contract).lower()
            try:
                campaign_status = CampaignStatus(status)
            except ValueError as exc:
                raise AutoResearchOnboardingConflict("campaign_status_invalid") from exc
            if campaign_status in {CampaignStatus.DRAFT, CampaignStatus.VALIDATING}:
                raise AutoResearchOnboardingConflict("campaign_not_ready")
            next_action = (
                "explicit_start_confirmation_required"
                if campaign_status == CampaignStatus.READY
                else "research_state"
            )
            return existing.model_copy(
                update={
                    "replayed": True,
                    "campaign_status": status,
                    "next_action": next_action,
                }
            )

        for step in ONBOARDING_STEPS[len(completed) :]:
            result = self._services.run_step(step, contract)
            if result.step != step:
                raise AutoResearchOnboardingConflict("step_result_mismatch")
            completed.append(result)
            _write_receipt(
                path,
                AutoResearchOnboardingReceipt(
                    onboarding_id=contract.onboarding_id,
                    contract_digest=contract_digest,
                    completed_steps=tuple(completed),
                    campaign_id=contract.campaign_id,
                    experiment_contract=experiment_contract,
                ),
            )

        status = self._services.campaign_status(contract).lower()
        ready = status == "ready"
        receipt = AutoResearchOnboardingReceipt(
            onboarding_id=contract.onboarding_id,
            contract_digest=contract_digest,
            applied=ready,
            completed_steps=tuple(completed),
            campaign_id=contract.campaign_id,
            experiment_contract=experiment_contract,
            campaign_status=status,
            next_action=("explicit_start_confirmation_required" if ready else "resume_onboarding"),
        )
        _write_receipt(path, receipt)
        if not ready:
            raise AutoResearchOnboardingConflict("campaign_not_ready")
        return receipt


class LocalAutoResearchOnboardingServices:
    """Concrete adapter joining target registration, services, and canonical APIs."""

    def __init__(self, contract: AutoResearchOnboardingContract) -> None:
        self.definition, self.activation, self.model_request = validate_local_onboarding_contract(
            contract
        )
        self.client = _campaign_client(contract)

    @staticmethod
    def _result(step: str, reference: str, *, replayed: bool = False):
        return AutoResearchOnboardingStepResult(
            step=step,
            reference=reference,
            disposition="replayed" if replayed else "completed",
        )

    def _target_model(self, contract: AutoResearchOnboardingContract):
        from bashgym.campaigns.remote import RegisteredRemoteModelSource

        path = _private_state_path(contract, "target-model")
        existing = _read_private_payload(path)
        if existing is not None:
            if existing.get("request_digest") != self.model_request.request_digest:
                raise AutoResearchOnboardingConflict("target_model_request_changed")
            source = RegisteredRemoteModelSource.model_validate(existing.get("source"))
            _verify_target_bindings_on_compute(
                source,
                self.activation,
                verify_model=True,
            )
            return self._result("target_model", source.source_id, replayed=True)
        source = _register_target_model_on_compute(self.model_request, self.activation)
        _verify_target_bindings_on_compute(
            source,
            self.activation,
            verify_model=False,
        )
        _write_private_payload(
            path,
            {
                "request_digest": self.model_request.request_digest,
                "source": source.model_dump(mode="json"),
            },
        )
        return self._result("target_model", source.source_id)

    def reconcile(
        self,
        contract: AutoResearchOnboardingContract,
        completed_steps: tuple[str, ...],
    ) -> None:
        if "target_model" in completed_steps:
            self._target_model(contract)
        if "resident_services" in completed_steps:
            _install_local_resident_services(contract)

    def _activation(self, contract: AutoResearchOnboardingContract):
        from bashgym.campaigns.remote import (
            ApprovedRemoteExecutorProfile,
            RegisteredRemoteModelSource,
        )

        payload = _read_private_payload(_private_state_path(contract, "target-model"))
        if payload is None:
            raise AutoResearchOnboardingConflict("target_model_receipt_missing")
        source = RegisteredRemoteModelSource.model_validate(payload.get("source"))
        executor = ApprovedRemoteExecutorProfile.model_validate(
            {
                **self.activation.executor_profile.model_dump(
                    mode="python", exclude={"profile_digest", "registered_base_model"}
                ),
                "registered_base_model": source,
            }
        )
        request = self.activation.model_copy(update={"executor_profile": executor})
        _apply_local_activation(self.definition, request, contract)
        return self._result("activation", self.definition.template_id)

    def _guided_setup(self, contract: AutoResearchOnboardingContract):
        from bashgym.campaigns.installation import autoresearch_binding_plan

        binding = autoresearch_binding_plan(self.definition)
        session_id = f"setupsess_{canonical_hash(contract.onboarding_id)[:32]}"
        steps = (
            ("template", self.definition.template_id),
            ("installation", contract.installation_id),
            ("model", binding.target_contract_key),
            ("data", binding.dataset_version_id),
            ("compute", binding.compute_profile_id),
            ("evaluation", binding.evaluation_suite_id),
        )
        context = self.client.request_json(
            "GET",
            "/campaigns/setup/context",
            query={"workspace_id": contract.workspace_id, "session_id": session_id},
        )
        session = context.get("session") if isinstance(context, dict) else None
        version = int(session.get("version", 0)) if isinstance(session, dict) else 0
        if not 0 <= version <= len(steps):
            raise AutoResearchOnboardingConflict("guided_setup_session_invalid")
        for expected_version, (step, selection_id) in enumerate(steps[version:], start=version):
            self.client.request_json(
                "POST",
                "/campaigns/setup/session",
                payload={
                    "workspace_id": contract.workspace_id,
                    "session_id": session_id,
                    "expected_version": expected_version,
                    "step": step,
                    "selection_id": selection_id,
                },
                headers={"Idempotency-Key": (f"{contract.onboarding_id}-setup-{expected_version}")},
            )
        draft = {
            "workspace_id": contract.workspace_id,
            "template_id": self.definition.template_id,
            "installation_id": contract.installation_id,
            "bindings": {
                "model": binding.target_contract_key,
                "data": binding.dataset_version_id,
                "compute": binding.compute_profile_id,
                "evaluation": binding.evaluation_suite_id,
            },
            "stop_rules": contract.stop_rules.model_dump(mode="json"),
        }
        doctor = self.client.request_json("POST", "/campaigns/setup/doctor", payload=draft)
        if not isinstance(doctor, dict) or not doctor.get("ready"):
            raise AutoResearchOnboardingError("onboarding_doctor_not_ready")
        validated = self.client.request_json(
            "POST",
            "/campaigns/setup/validate",
            payload=draft,
            headers={"Idempotency-Key": f"{contract.onboarding_id}-validate"},
        )
        receipt_id = validated.get("receipt_id") if isinstance(validated, dict) else None
        if not isinstance(receipt_id, str) or not receipt_id.startswith("setuprcpt_"):
            raise AutoResearchOnboardingError("onboarding_validation_receipt_invalid")
        _write_private_payload(
            _private_state_path(contract, "validation"),
            {"receipt_id": receipt_id, "session_id": session_id},
        )
        return self._result("guided_setup", receipt_id)

    def _campaign_prepare(self, contract: AutoResearchOnboardingContract):
        validation = _read_private_payload(_private_state_path(contract, "validation"))
        receipt_id = validation.get("receipt_id") if validation else None
        if not isinstance(receipt_id, str):
            raise AutoResearchOnboardingConflict("validation_receipt_missing")
        response = self.client.request_json(
            "POST",
            "/campaigns/setup/create",
            payload={
                "workspace_id": contract.workspace_id,
                "campaign_id": contract.campaign_id,
                "title": contract.campaign_title,
                "validation_receipt_id": receipt_id,
            },
            headers={
                "Idempotency-Key": f"{contract.onboarding_id}-create",
                "X-Correlation-ID": contract.onboarding_id,
            },
        )
        campaign = response.get("campaign") if isinstance(response, dict) else None
        if not isinstance(campaign, dict) or str(campaign.get("status", "")).lower() != "ready":
            raise AutoResearchOnboardingError("onboarding_campaign_not_ready")
        return self._result("campaign_prepare", contract.campaign_id)

    def run_step(self, step: str, contract: AutoResearchOnboardingContract):
        if step == "target_model":
            return self._target_model(contract)
        if step == "activation":
            return self._activation(contract)
        if step == "resident_services":
            _install_local_resident_services(contract)
            return self._result(step, "resident-services")
        if step == "registry_sync":
            _sync_local_registry(self.definition, contract)
            return self._result(step, contract.installation_id)
        if step == "guided_setup":
            return self._guided_setup(contract)
        if step == "campaign_prepare":
            return self._campaign_prepare(contract)
        raise AutoResearchOnboardingError("onboarding_step_invalid")

    def campaign_status(self, contract: AutoResearchOnboardingContract) -> str:
        response = self.client.request_json(
            "GET",
            f"/campaigns/{urllib.parse.quote(contract.campaign_id, safe='')}",
            query={"workspace_id": contract.workspace_id},
        )
        if not isinstance(response, dict) or not isinstance(response.get("status"), str):
            raise AutoResearchOnboardingError("onboarding_campaign_status_invalid")
        return response["status"]


def build_local_onboarding_services(
    contract: AutoResearchOnboardingContract,
) -> AutoResearchOnboardingServices:
    """Build the concrete local/SSH/API adapter for an apply operation."""

    from bashgym.config import get_bashgym_dir

    if contract.data_directory.expanduser().resolve() != get_bashgym_dir().resolve():
        raise AutoResearchOnboardingError("onboarding_data_directory_mismatch")
    _ = contract.contract_digest
    return LocalAutoResearchOnboardingServices(contract)


__all__ = [
    "ONBOARDING_STEPS",
    "AutoResearchOnboardingConflict",
    "AutoResearchOnboardingContract",
    "AutoResearchOnboardingCoordinator",
    "AutoResearchOnboardingError",
    "AutoResearchOnboardingExperimentContract",
    "AutoResearchOnboardingPlan",
    "AutoResearchOnboardingReceipt",
    "AutoResearchOnboardingServices",
    "AutoResearchOnboardingStepResult",
    "LocalAutoResearchOnboardingServices",
    "build_local_onboarding_services",
    "validate_local_onboarding_contract",
]
