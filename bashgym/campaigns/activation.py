"""Idempotent installation activation for real AutoResearch campaigns."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition
from bashgym.campaigns.contracts import FrozenContractModel, Identifier, utc_now
from bashgym.campaigns.installation import autoresearch_binding_plan
from bashgym.campaigns.lineage import ApprovedSourceRepositoryProfile
from bashgym.campaigns.readiness import AutoResearchDoctorReport, doctor_autoresearch_template
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile
from bashgym.campaigns.worker_service import (
    ControllerStatusProjection,
    WorkerPlatform,
    WorkerRunConfig,
    WorkerServiceManager,
    build_service_definition,
    ensure_worker_bootstrap,
    load_approved_remote_profiles,
    load_approved_source_profiles,
    project_controller_status,
    read_worker_config,
    validate_code_lineage_execution_bindings,
    write_worker_config,
)
from bashgym.ledger.contracts import (
    DatasetSpec,
    DatasetVersionSpec,
    EvaluationSuiteSpec,
    ProjectSpec,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository, RecordNotFoundError


class AutoResearchActivationError(ValueError):
    """A stable activation validation or identity conflict."""


class AutoResearchActivationRequest(FrozenContractModel):
    """Fully validated installation records for one installed definition."""

    schema_version: Literal["autoresearch_activation_request.v1"] = (
        "autoresearch_activation_request.v1"
    )
    workspace_id: Identifier
    project: ProjectSpec
    dataset: DatasetSpec
    dataset_version: DatasetVersionSpec
    evaluation_suite: EvaluationSuiteSpec
    source_profile: ApprovedSourceRepositoryProfile
    executor_profile: ApprovedRemoteExecutorProfile

    @model_validator(mode="after")
    def consistent_ledger_scope(self) -> AutoResearchActivationRequest:
        records = (self.project, self.dataset, self.dataset_version, self.evaluation_suite)
        if any(record.workspace_id != self.workspace_id for record in records):
            raise ValueError("activation ledger records must share one workspace")
        if any(record.project_id != self.project.project_id for record in records):
            raise ValueError("activation ledger records must share one project")
        if self.dataset_version.dataset_id != self.dataset.dataset_id:
            raise ValueError("dataset version must belong to the activation dataset")
        if self.evaluation_suite.dataset_version_id != self.dataset_version.dataset_version_id:
            raise ValueError("evaluation suite must use the activation dataset version")
        return self


class ActivationRecordResult(FrozenContractModel):
    schema_version: Literal["autoresearch_activation_record.v1"] = (
        "autoresearch_activation_record.v1"
    )
    record_type: Identifier
    record_id: Identifier
    disposition: Literal["planned", "created", "replayed"]


class AutoResearchActivationReceipt(FrozenContractModel):
    """Secret-free activation outcome; private paths and SSH identity are omitted."""

    schema_version: Literal["autoresearch_activation_receipt.v1"] = (
        "autoresearch_activation_receipt.v1"
    )
    template_id: Identifier
    definition_digest: str
    applied: bool
    records: tuple[ActivationRecordResult, ...]
    worker_config: Literal["planned", "created", "updated", "replayed"]
    service: Literal["planned", "not_installed", "installed"]
    service_platform: str
    compute_profile_id: Identifier
    executor_profile_digest: str
    source_profile_id: Identifier
    source_profile_digest: str
    doctor: AutoResearchDoctorReport | None = None


_RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    "project": ("display_name", "description", "owner_actor_id", "tags"),
    "dataset": ("display_name", "task_type", "metadata"),
    "dataset_version": (
        "dataset_id",
        "source_uri",
        "content_digest",
        "split_manifest",
        "row_counts",
        "metadata",
    ),
    "evaluation_suite": (
        "name",
        "task_type",
        "dataset_version_id",
        "metric_contract",
        "code_digest",
        "metadata",
    ),
}


def _same_record(record_type: str, existing: dict[str, Any], expected: Any) -> bool:
    fields = _RECORD_FIELDS[record_type]
    wanted = expected.model_dump(mode="json")
    return all(existing.get(field) == wanted.get(field) for field in fields)


def _ledger_dispositions(
    repository: ExperimentLedgerRepository | None,
    request: AutoResearchActivationRequest,
) -> dict[str, Literal["planned", "replayed"]]:
    checks: tuple[tuple[str, str, Any, Callable[[], dict[str, Any]]], ...] = (
        (
            "project",
            request.project.project_id,
            request.project,
            lambda: repository.get_project(request.workspace_id, request.project.project_id),  # type: ignore[union-attr]
        ),
        (
            "dataset",
            request.dataset.dataset_id,
            request.dataset,
            lambda: repository.get_dataset(  # type: ignore[union-attr]
                request.workspace_id, request.project.project_id, request.dataset.dataset_id
            ),
        ),
        (
            "dataset_version",
            request.dataset_version.dataset_version_id,
            request.dataset_version,
            lambda: repository.get_dataset_version(  # type: ignore[union-attr]
                request.workspace_id,
                request.project.project_id,
                request.dataset_version.dataset_version_id,
            ),
        ),
        (
            "evaluation_suite",
            request.evaluation_suite.evaluation_suite_id,
            request.evaluation_suite,
            lambda: repository.get_evaluation_suite(  # type: ignore[union-attr]
                request.workspace_id,
                request.project.project_id,
                request.evaluation_suite.evaluation_suite_id,
            ),
        ),
    )
    states: dict[str, Literal["planned", "replayed"]] = {}
    for record_type, _record_id, expected, getter in checks:
        if repository is None:
            states[record_type] = "planned"
            continue
        try:
            existing = getter()
        except RecordNotFoundError:
            states[record_type] = "planned"
            continue
        if not _same_record(record_type, existing, expected):
            raise AutoResearchActivationError(
                f"{record_type} already exists with a different identity"
            )
        states[record_type] = "replayed"
    return states


def _validate_definition_bindings(
    definition: AutoResearchTemplateDefinition,
    request: AutoResearchActivationRequest,
) -> None:
    binding = autoresearch_binding_plan(definition)
    metric = request.evaluation_suite.metric_contract
    required_stages = {stage.value for stage in binding.required_training_stages}
    configured_stages = {stage.stage.value for stage in request.executor_profile.stages}
    if request.project.project_id != binding.ledger_project_id:
        raise AutoResearchActivationError("activation project does not match template binding")
    if request.dataset_version.dataset_version_id != binding.dataset_version_id:
        raise AutoResearchActivationError(
            "activation dataset version does not match template binding"
        )
    if request.evaluation_suite.evaluation_suite_id != binding.evaluation_suite_id:
        raise AutoResearchActivationError("activation evaluator does not match template binding")
    if (
        metric.get("primary_metric") != binding.primary_metric
        or metric.get("metric_direction") != binding.metric_direction.value
    ):
        raise AutoResearchActivationError("activation metric contract does not match template")
    if request.source_profile.profile_id != binding.source_repository_profile_id:
        raise AutoResearchActivationError("activation source profile does not match template")
    executor = request.executor_profile
    if (
        executor.compute_profile_id != binding.compute_profile_id
        or executor.target_contract_key != binding.target_contract_key
        or executor.target_model_digest != binding.target_model_digest
    ):
        raise AutoResearchActivationError("activation executor does not match template")
    if not required_stages.issubset(configured_stages):
        raise AutoResearchActivationError("activation executor is missing required training stages")
    for stage in executor.stages:
        if stage.stage.value not in required_stages:
            continue
        code = stage.code_lineage_binding
        if code is None or code.source_repository_profile_id != request.source_profile.profile_id:
            raise AutoResearchActivationError("required stage is missing its source binding")


def _merge_worker_config(
    current: WorkerRunConfig,
    request: AutoResearchActivationRequest,
) -> tuple[WorkerRunConfig, Literal["updated", "replayed"]]:
    sources = {profile.profile_id: profile for profile in current.approved_source_profiles}
    prior_source = sources.get(request.source_profile.profile_id)
    if (
        prior_source is not None
        and prior_source.profile_digest != request.source_profile.profile_digest
    ):
        raise AutoResearchActivationError(
            "source profile ID already exists with a different identity"
        )
    sources[request.source_profile.profile_id] = request.source_profile

    remotes = {
        (profile.compute_profile_id, profile.target_contract_key): profile
        for profile in current.approved_remote_profiles
    }
    key = (
        request.executor_profile.compute_profile_id,
        request.executor_profile.target_contract_key,
    )
    prior_remote = remotes.get(key)
    if (
        prior_remote is not None
        and prior_remote.profile_digest != request.executor_profile.profile_digest
    ):
        raise AutoResearchActivationError(
            "compute and target contract already resolve to a different executor"
        )
    remotes[key] = request.executor_profile
    merged = current.model_copy(
        update={
            "approved_source_profiles": tuple(
                sorted(sources.values(), key=lambda profile: profile.profile_id)
            ),
            "approved_remote_profiles": tuple(
                sorted(
                    remotes.values(),
                    key=lambda profile: (
                        profile.compute_profile_id,
                        profile.target_contract_key,
                        profile.profile_id,
                        profile.profile_revision,
                    ),
                )
            ),
        }
    )
    merged = WorkerRunConfig.model_validate(merged.model_dump())
    remote_registry = load_approved_remote_profiles(merged)
    source_registry = load_approved_source_profiles(merged)
    validate_code_lineage_execution_bindings(remote_registry, source_registry)
    state: Literal["updated", "replayed"] = "replayed" if merged == current else "updated"
    return merged, state


def activate_autoresearch(
    definition: AutoResearchTemplateDefinition,
    request: AutoResearchActivationRequest,
    *,
    data_directory: Path,
    apply: bool = False,
    install_service: bool = False,
    worker_config_path: Path | None = None,
    service_target: WorkerPlatform | None = None,
    service_home: Path | None = None,
    service_manager: WorkerServiceManager | None = None,
    controller: ControllerStatusProjection | None = None,
    secret_resolver: Callable[[str], str | None] | None = None,
    secret_writer: Callable[[str, str], None] | None = None,
) -> AutoResearchActivationReceipt:
    """Plan or apply exact ledger, profile, worker, and service bindings."""

    if install_service and not apply:
        raise AutoResearchActivationError("service installation requires apply=True")
    _validate_definition_bindings(definition, request)
    root = data_directory.expanduser().resolve()
    config_path = (
        worker_config_path.expanduser().resolve()
        if worker_config_path is not None
        else root / "campaigns" / "worker-config.v1.json"
    )
    database_path = root / "campaigns" / "campaigns.sqlite3"
    repository = ExperimentLedgerRepository(database_path) if database_path.is_file() else None
    if repository is not None:
        # Re-open the shared campaign/ledger repository before conflict checks.
        # This may apply a pending installation migration, but never creates or
        # changes activation records during a plan.
        repository.initialize()
    planned = _ledger_dispositions(repository, request)

    config_created = not config_path.exists()
    current = (
        read_worker_config(config_path)
        if not config_created
        else WorkerRunConfig.for_data_directory(root)
    )
    if current.data_directory != root:
        raise AutoResearchActivationError("worker config belongs to another data directory")
    merged, config_state = _merge_worker_config(current, request)
    definition_record = build_service_definition(
        config_path, target=service_target, home=service_home
    )

    record_states: dict[str, str] = dict(planned)
    doctor = None
    service_state: Literal["planned", "not_installed", "installed"] = "planned"
    if apply:
        repository = ExperimentLedgerRepository(database_path)
        repository.initialize()
        registrations = (
            ("project", request.project, repository.register_project),
            ("dataset", request.dataset, repository.register_dataset),
            ("dataset_version", request.dataset_version, repository.register_dataset_version),
            (
                "evaluation_suite",
                request.evaluation_suite,
                repository.register_evaluation_suite,
            ),
        )
        for record_type, spec, register in registrations:
            _record, replayed = register(spec)
            record_states[record_type] = "replayed" if replayed else "created"
        write_worker_config(config_path, merged)
        bootstrap_kwargs: dict[str, Any] = {"config_path": config_path}
        if secret_resolver is not None:
            bootstrap_kwargs["secret_resolver"] = secret_resolver
        if secret_writer is not None:
            bootstrap_kwargs["secret_writer"] = secret_writer
        ensure_worker_bootstrap(root, **bootstrap_kwargs)
        service_state = "not_installed"
        if install_service:
            (service_manager or WorkerServiceManager()).install(definition_record, merged)
            service_state = "installed"
        observed_controller = controller
        if observed_controller is None and install_service:
            observed_controller = project_controller_status(repository, root)
        if observed_controller is None:
            observed_controller = ControllerStatusProjection(
                online=False,
                state="offline",
                code="controller_offline",
                observed_at=utc_now(),
                guidance="Start the installed resident campaign worker.",
            )
        doctor = doctor_autoresearch_template(
            definition,
            workspace_id=request.workspace_id,
            ledger=repository,
            executor_profiles=load_approved_remote_profiles(merged),
            source_profiles=load_approved_source_profiles(merged),
            controller=observed_controller,
        )

    ids = {
        "project": request.project.project_id,
        "dataset": request.dataset.dataset_id,
        "dataset_version": request.dataset_version.dataset_version_id,
        "evaluation_suite": request.evaluation_suite.evaluation_suite_id,
    }
    return AutoResearchActivationReceipt(
        template_id=definition.template_id,
        definition_digest=definition.definition_digest,
        applied=apply,
        records=tuple(
            ActivationRecordResult(
                record_type=record_type,
                record_id=ids[record_type],
                disposition=record_states[record_type],
            )
            for record_type in ("project", "dataset", "dataset_version", "evaluation_suite")
        ),
        worker_config=("planned" if not apply else ("created" if config_created else config_state)),
        service=service_state,
        service_platform=definition_record.platform.value,
        compute_profile_id=request.executor_profile.compute_profile_id,
        executor_profile_digest=request.executor_profile.profile_digest,
        source_profile_id=request.source_profile.profile_id,
        source_profile_digest=request.source_profile.profile_digest,
        doctor=doctor,
    )


__all__ = [
    "ActivationRecordResult",
    "AutoResearchActivationError",
    "AutoResearchActivationReceipt",
    "AutoResearchActivationRequest",
    "activate_autoresearch",
]
