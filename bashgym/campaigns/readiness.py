"""Fail-closed readiness checks for installation-bound AutoResearch templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field

from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition
from bashgym.campaigns.contracts import FrozenContractModel, Identifier, StageKind, canonical_hash
from bashgym.campaigns.lineage import (
    ApprovedSourceRepositoryProfile,
    GitHypothesisLineageManager,
    GitLineageError,
)
from bashgym.campaigns.nemo_rl import NEMO_GYM_STAGE_OUTPUT_PATHS, NEMO_RL_STAGE_OUTPUT_PATHS
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile
from bashgym.campaigns.worker_service import ControllerStatusProjection
from bashgym.ledger.persistence import ExperimentLedgerRepository

_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64})$")
_DEFAULT_COMPUTE_STAGES = frozenset({StageKind.SMOKE_TRAINING, StageKind.FULL_TRAINING})
_REMOTE_COMPUTE_STAGES = frozenset({*_DEFAULT_COMPUTE_STAGES, StageKind.DEVELOPMENT_EVALUATION})


class AutoResearchDoctorCheck(FrozenContractModel):
    schema_version: Literal["autoresearch_doctor_check.v1"] = "autoresearch_doctor_check.v1"
    check_id: Identifier
    ready: bool
    code: Identifier
    guidance: str = Field(default="", max_length=1000)


class AutoResearchDoctorReport(FrozenContractModel):
    schema_version: Literal["autoresearch_doctor_report.v1"] = "autoresearch_doctor_report.v1"
    workspace_id: Identifier
    template_id: Identifier
    definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_claim_eligible: bool
    materializable: bool
    launch_ready: bool
    available: bool
    blocking_codes: tuple[Identifier, ...]
    checks: tuple[AutoResearchDoctorCheck, ...]


def _check(
    check_id: str,
    ready: bool,
    success_code: str,
    failure_code: str,
    guidance: str,
) -> AutoResearchDoctorCheck:
    return AutoResearchDoctorCheck(
        check_id=check_id,
        ready=ready,
        code=success_code if ready else failure_code,
        guidance="" if ready else guidance,
    )


def _source_entrypoint_ready(
    source_profile: ApprovedSourceRepositoryProfile, entrypoint_path: str
) -> bool:
    repository = source_profile.repository_path.expanduser().resolve()
    candidate = repository.joinpath(*PurePosixPath(entrypoint_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository)
    except (OSError, ValueError):
        return False
    return resolved.is_file() and not candidate.is_symlink()


def has_immutable_model_revision(model_ref: str) -> bool:
    """Return whether a model reference names an exact content revision."""

    if model_ref.startswith("unconfigured://") or "@" not in model_ref:
        return False
    _location, revision = model_ref.rsplit("@", 1)
    return bool(_IMMUTABLE_REVISION.fullmatch(revision.casefold()))


def _required_compute_stages(definition: AutoResearchTemplateDefinition) -> frozenset[StageKind]:
    evaluation_plan = definition.manifest.evaluation_plan
    configured = evaluation_plan.get(
        "required_compute_stages", evaluation_plan.get("required_training_stages")
    )
    if configured is None:
        return _DEFAULT_COMPUTE_STAGES
    if not isinstance(configured, (list, tuple)):
        return frozenset()
    try:
        stages = frozenset(StageKind(str(value)) for value in configured)
    except ValueError:
        return frozenset()
    return stages if stages and stages.issubset(_REMOTE_COMPUTE_STAGES) else frozenset()


def doctor_autoresearch_template(
    definition: AutoResearchTemplateDefinition,
    *,
    workspace_id: str,
    ledger: ExperimentLedgerRepository,
    executor_profiles: Mapping[tuple[str, str], ApprovedRemoteExecutorProfile],
    controller: ControllerStatusProjection,
    source_profiles: Mapping[str, ApprovedSourceRepositoryProfile] | None = None,
) -> AutoResearchDoctorReport:
    """Return a secret-free report; every quality binding must resolve exactly."""

    policy = definition.policy
    quality_claim_eligible = bool(
        (policy and policy.quality_claim_eligible)
        or definition.manifest.promotion_gates.get("quality_claim_eligible", False)
    )
    checks: list[AutoResearchDoctorCheck] = []
    checks.append(
        _check(
            "template_policy",
            policy is not None,
            "template_policy_ready",
            "template_policy_missing",
            "Install a schema-validated AutoResearch policy before materializing this template.",
        )
    )

    if quality_claim_eligible:
        model_ready = has_immutable_model_revision(definition.target_model.base_model_ref) and (
            definition.target_model.representation_contract.get("artifact_role") == "trainable_base"
        )
        checks.append(
            _check(
                "model_binding",
                model_ready,
                "model_binding_ready",
                "model_binding_unresolved",
                "Register an operator-selected trainable base at an immutable revision; inference quants are not training bases.",
            )
        )

        evaluation_plan = definition.manifest.evaluation_plan
        dataset_binding_id = evaluation_plan.get("dataset_binding_id")
        data_ready = (
            isinstance(dataset_binding_id, str)
            and dataset_binding_id in definition.manifest.approved_data_scopes
            and policy is not None
        )
        if data_ready:
            try:
                ledger.get_dataset_version(
                    workspace_id, policy.ledger_project_id, dataset_binding_id
                )
            except RecordNotFoundError:
                data_ready = False
        checks.append(
            _check(
                "data_binding",
                data_ready,
                "data_binding_ready",
                "data_binding_unresolved",
                "Register the exact approved dataset version in the experiment ledger.",
            )
        )

        evaluator_ready = policy is not None
        if evaluator_ready:
            try:
                ledger.get_project(workspace_id, policy.ledger_project_id)
                suite = ledger.get_evaluation_suite(
                    workspace_id,
                    policy.ledger_project_id,
                    policy.evaluation_suite_id,
                )
                metric_contract = suite.get("metric_contract", {})
                evaluator_ready = bool(
                    isinstance(metric_contract, dict)
                    and metric_contract.get("primary_metric") == policy.primary_metric
                    and metric_contract.get("metric_direction") == policy.metric_direction.value
                    and suite.get("dataset_version_id") == dataset_binding_id
                )
            except RecordNotFoundError:
                evaluator_ready = False
        checks.append(
            _check(
                "evaluator_binding",
                evaluator_ready,
                "evaluator_binding_ready",
                "evaluator_binding_unresolved",
                "Register a hash-pinned evaluation suite whose dataset and primary metric exactly match the campaign policy.",
            )
        )

        profile_key = (
            definition.manifest.compute_profile_id,
            definition.target_model.target_contract_key,
        )
        profile = executor_profiles.get(profile_key)
        compute_ready = profile is not None
        required_compute_stages = _required_compute_stages(definition)
        if profile is not None:
            expected_target_digest = canonical_hash(definition.target_model.model_dump(mode="json"))
            compute_ready = bool(
                required_compute_stages
                and profile.target_model_digest == expected_target_digest
                and required_compute_stages.issubset({stage.stage for stage in profile.stages})
            )
            if compute_ready:
                try:
                    profile.verify_materials()
                except (OSError, ValueError):
                    compute_ready = False
        checks.append(
            _check(
                "compute_binding",
                compute_ready,
                "compute_binding_ready",
                "compute_binding_unresolved",
                "Install an exact registered-training profile for this model contract and verify every pinned script, input, credential, and stage.",
            )
        )
        if profile is not None and profile.nemo_rl is not None:
            nemo = profile.nemo_rl
            receipt = nemo.runtime_receipt
            model_location, _, model_revision = definition.target_model.base_model_ref.rpartition(
                "@"
            )
            expected_model_id = (
                model_location.removeprefix("hf://") if model_location.startswith("hf://") else ""
            )
            checks.extend(
                (
                    _check(
                        "nemo_source",
                        bool(
                            receipt is not None
                            and receipt.source_ready
                            and receipt.source_revision == nemo.source_revision
                        ),
                        "nemo_source_ready",
                        "nemo_source_unresolved",
                        "Re-run setup-nemo-rl against the exact source revision embedded in the pinned image.",
                    ),
                    _check(
                        "nemo_image",
                        bool(
                            receipt is not None
                            and receipt.docker_ready
                            and receipt.nvidia_runtime_ready
                            and receipt.image_ready
                            and receipt.image_digest == nemo.image_digest
                            and receipt.platform == nemo.platform
                        ),
                        "nemo_image_ready",
                        "nemo_image_unresolved",
                        "Explicitly pull and verify the platform-specific image digest on registered private compute.",
                    ),
                    _check(
                        "nemo_model_support",
                        bool(
                            receipt is not None
                            and receipt.model_ready
                            and nemo.model_id == expected_model_id
                            and nemo.model_revision == model_revision
                            and nemo.model_support_level.value
                            in {"broad_api_compatible", "recipe_reproduced", "optimized"}
                        ),
                        "nemo_model_support_ready",
                        "nemo_model_support_unresolved",
                        "Bind the selected immutable trainable model and record its verified NeMo support level.",
                    ),
                    _check(
                        "nemo_recipe_data_verifier",
                        bool(
                            receipt is not None
                            and receipt.recipe_ready
                            and receipt.recipe_sha256 == nemo.recipe_sha256
                            and nemo.dataset_path.is_file()
                            and nemo.verifier_digest
                        ),
                        "nemo_recipe_data_verifier_ready",
                        "nemo_recipe_data_verifier_unresolved",
                        "Pin the exact recipe, dataset, deterministic verifier, and their content digests.",
                    ),
                    _check(
                        "nemo_runtime_capacity",
                        bool(
                            receipt is not None
                            and receipt.gpu_count >= nemo.gpu_count
                            and receipt.available_disk_gib >= nemo.minimum_available_disk_gib
                            and receipt.shared_memory_gib >= nemo.shared_memory_gib
                        ),
                        "nemo_runtime_capacity_ready",
                        "nemo_runtime_capacity_insufficient",
                        "Free the configured disk/shared memory capacity or select a smaller approved recipe.",
                    ),
                    _check(
                        "nemo_execution_contract",
                        all(
                            stage.output_paths
                            == (
                                NEMO_GYM_STAGE_OUTPUT_PATHS
                                if nemo.nemo_gym is not None
                                else NEMO_RL_STAGE_OUTPUT_PATHS
                            )
                            and stage.budget_reservation > 0
                            and stage.script_args
                            == (
                                "--contract-json",
                                nemo.container_contract(stage.stage).model_dump_json(),
                            )
                            for stage in profile.stages
                            if stage.stage in required_compute_stages
                        ),
                        "nemo_execution_contract_ready",
                        "nemo_execution_contract_unresolved",
                        "Re-run setup-nemo-rl so the bounded wrapper, outputs, stop rules, and budget are exact.",
                    ),
                )
            )
            if nemo.nemo_gym is not None:
                gym = nemo.nemo_gym
                checks.append(
                    _check(
                        "nemo_gym_execution_contract",
                        bool(
                            receipt is not None
                            and receipt.nemo_gym_source_ready is True
                            and receipt.nemo_gym_source_revision == gym.nemo_gym_source_revision
                            and nemo.entrypoint_path == "examples/nemo_gym/run_grpo_nemo_gym.py"
                            and all(
                                gym.bundle_archive_path in stage.input_files
                                and stage.input_sha256.get(gym.bundle_archive_path.name)
                                == gym.bundle_archive_sha256
                                and nemo.container_contract(stage.stage).nemo_gym is not None
                                for stage in profile.stages
                                if stage.stage in required_compute_stages
                            )
                        ),
                        "nemo_gym_execution_contract_ready",
                        "nemo_gym_execution_contract_unresolved",
                        "Re-run setup-nemo-rl with the exact Gym bundle and embedded source revision on a dedicated profile.",
                    )
                )
        source_profile_id = evaluation_plan.get("source_repository_binding_id")
        source_profile = (
            source_profiles.get(source_profile_id)
            if isinstance(source_profile_id, str) and source_profiles is not None
            else None
        )
        source_ready = source_profile is not None
        if source_profile is not None:
            try:
                GitHypothesisLineageManager.verify_profile(source_profile)
            except GitLineageError:
                source_ready = False
        checks.append(
            _check(
                "source_repository_binding",
                source_ready,
                "source_repository_binding_ready",
                "source_repository_binding_unresolved",
                "Register the logical source profile in the worker configuration with a private Git repository and operator-approved mutation paths.",
            )
        )
        code_execution_ready = bool(
            source_ready
            and profile is not None
            and source_profile is not None
            and required_compute_stages
            and all(
                stage.code_lineage_binding is not None
                and stage.code_lineage_binding.source_repository_profile_id
                == source_profile.profile_id
                and _source_entrypoint_ready(
                    source_profile, stage.code_lineage_binding.entrypoint_path
                )
                for stage in profile.stages
                if stage.stage in required_compute_stages
            )
            and required_compute_stages.issubset(
                {stage.stage for stage in profile.stages if stage.code_lineage_binding is not None}
            )
        )
        checks.append(
            _check(
                "code_lineage_execution_binding",
                code_execution_ready,
                "code_lineage_execution_binding_ready",
                "code_lineage_execution_binding_unresolved",
                "Bind each required training stage to the logical source profile and an in-repository Python entrypoint.",
            )
        )
    else:
        checks.extend(
            (
                _check(
                    "model_binding",
                    True,
                    "control_template_model_not_required",
                    "model_binding_unresolved",
                    "",
                ),
                _check(
                    "data_binding",
                    True,
                    "control_template_data_not_required",
                    "data_binding_unresolved",
                    "",
                ),
                _check(
                    "evaluator_binding",
                    True,
                    "control_template_evaluator_not_required",
                    "evaluator_binding_unresolved",
                    "",
                ),
                _check(
                    "compute_binding",
                    True,
                    "control_template_compute_not_required",
                    "compute_binding_unresolved",
                    "",
                ),
                _check(
                    "source_repository_binding",
                    True,
                    "control_template_source_repository_not_required",
                    "source_repository_binding_unresolved",
                    "",
                ),
                _check(
                    "code_lineage_execution_binding",
                    True,
                    "control_template_code_lineage_execution_not_required",
                    "code_lineage_execution_binding_unresolved",
                    "",
                ),
            )
        )

    checks.append(
        _check(
            "controller",
            controller.online,
            "controller_ready",
            controller.code,
            controller.guidance or "Start the resident campaign controller.",
        )
    )
    materializable = all(check.ready for check in checks if check.check_id != "controller")
    launch_ready = materializable and controller.online
    blocking_codes = tuple(check.code for check in checks if not check.ready)
    return AutoResearchDoctorReport(
        workspace_id=workspace_id,
        template_id=definition.template_id,
        definition_digest=definition.definition_digest,
        quality_claim_eligible=quality_claim_eligible,
        materializable=materializable,
        launch_ready=launch_ready,
        available=launch_ready,
        blocking_codes=blocking_codes,
        checks=tuple(checks),
    )


__all__ = [
    "AutoResearchDoctorCheck",
    "AutoResearchDoctorReport",
    "doctor_autoresearch_template",
    "has_immutable_model_revision",
]
