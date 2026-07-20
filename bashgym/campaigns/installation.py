"""Portable installation helpers for quality-claiming AutoResearch templates."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from bashgym.campaigns.autoresearch import (
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
    AutoResearchTemplatePolicy,
    MetricDirection,
)
from bashgym.campaigns.contracts import (
    CampaignManifest,
    FrozenContractModel,
    HexDigest,
    Identifier,
    StageKind,
    TargetModelContract,
    canonical_hash,
)
from bashgym.campaigns.readiness import has_immutable_model_revision

_MAX_INSTALLATION_TEMPLATE_BYTES = 64 * 1024
_REQUIRED_TRAINING_STAGES = (StageKind.SMOKE_TRAINING, StageKind.FULL_TRAINING)


class AutoResearchInstallationError(ValueError):
    """Base error for local AutoResearch installation changes."""


class AutoResearchInstallationConflictError(AutoResearchInstallationError):
    """An installed template ID already names a different definition."""


class AutoResearchBindingPlan(FrozenContractModel):
    """Secret-free exact identities an installation must register."""

    schema_version: Literal["autoresearch_binding_plan.v1"] = "autoresearch_binding_plan.v1"
    template_id: Identifier
    definition_digest: HexDigest
    model_ref: str = Field(min_length=1, max_length=1000)
    target_model_digest: HexDigest
    target_contract_key: Identifier
    dataset_version_id: Identifier
    ledger_project_id: Identifier
    evaluation_suite_id: Identifier
    primary_metric: Identifier
    metric_direction: MetricDirection
    compute_profile_id: Identifier
    source_repository_profile_id: Identifier
    required_training_stages: tuple[StageKind, ...] = _REQUIRED_TRAINING_STAGES


class AutoResearchInstallationReceipt(FrozenContractModel):
    """Result of an idempotent, atomic installation write."""

    schema_version: Literal["autoresearch_installation_receipt.v1"] = (
        "autoresearch_installation_receipt.v1"
    )
    path: str
    created: bool
    replaced: bool
    binding_plan: AutoResearchBindingPlan


def build_quality_autoresearch_definition(
    *,
    template_id: str,
    template_revision: str,
    objective: str,
    model_ref: str,
    target_contract_key: str,
    task: str,
    dataset_version_id: str,
    compute_profile_id: str,
    source_repository_profile_id: str,
    ledger_project_id: str,
    evaluation_suite_id: str,
    primary_metric: str,
    metric_direction: MetricDirection | str,
    budget_unit: str,
    budget_limit: float,
    max_attempts: int,
    minimum_improvement: float = 0.0,
    target_metric: float | None = None,
    deadline: datetime | None = None,
    retention_days_failed: int = 90,
) -> AutoResearchTemplateDefinition:
    """Build a quality template only from explicit installation-owned bindings.

    There is deliberately no default model. The selected base must be a trainable
    artifact at an immutable content revision; serving/inference quants belong in
    deployment contracts, not this training definition.
    """

    if not has_immutable_model_revision(model_ref):
        raise AutoResearchInstallationError(
            "model_ref must include an immutable 40/64-character revision or sha256 digest"
        )
    direction = MetricDirection(metric_direction)
    stop_rules = AutoResearchStopRules(
        max_attempts=max_attempts,
        budget_unit=budget_unit,
        max_total_cost=budget_limit,
        target_metric=target_metric,
        minimum_improvement=minimum_improvement,
        deadline=deadline,
    )
    policy = AutoResearchTemplatePolicy(
        template_revision=template_revision,
        primary_metric=primary_metric,
        metric_direction=direction,
        stop_rules=stop_rules,
        ledger_project_id=ledger_project_id,
        evaluation_suite_id=evaluation_suite_id,
        require_sealed_artifact=True,
        quality_claim_eligible=True,
    )
    return AutoResearchTemplateDefinition(
        template_id=template_id,
        objective=objective,
        target_model=TargetModelContract(
            target_contract_key=target_contract_key,
            base_model_ref=model_ref,
            task=task,
            representation_contract={
                "artifact_role": "trainable_base",
                "revision_binding": "immutable",
            },
        ),
        manifest=CampaignManifest(
            approved_data_scopes=(dataset_version_id,),
            compute_profile_id=compute_profile_id,
            budget_limits={budget_unit: budget_limit},
            evaluation_plan={
                "dataset_binding_id": dataset_version_id,
                "source_repository_binding_id": source_repository_profile_id,
                "ledger_project_id": ledger_project_id,
                "evaluation_suite_id": evaluation_suite_id,
                "primary_metric": primary_metric,
                "metric_direction": direction.value,
                "required_training_stages": [stage.value for stage in _REQUIRED_TRAINING_STAGES],
            },
            promotion_gates={
                "quality_claim_eligible": True,
                "requires_real_evidence": True,
                "requires_sealed_artifact": True,
            },
            max_proposal_rounds=max_attempts,
            retention_days_failed=retention_days_failed,
            allow_hf_publication=False,
            allow_memexai_handoff=False,
        ),
        policy=policy,
    )


def autoresearch_binding_plan(
    definition: AutoResearchTemplateDefinition,
) -> AutoResearchBindingPlan:
    """Project exact logical registrations without exposing transport or secrets."""

    policy = definition.policy
    if policy is None or not policy.quality_claim_eligible:
        raise AutoResearchInstallationError("binding plans require a quality-claiming policy")
    dataset_version_id = definition.manifest.evaluation_plan.get("dataset_binding_id")
    if not isinstance(dataset_version_id, str):
        raise AutoResearchInstallationError("definition is missing dataset_binding_id")
    source_repository_profile_id = definition.manifest.evaluation_plan.get(
        "source_repository_binding_id"
    )
    if not isinstance(source_repository_profile_id, str):
        raise AutoResearchInstallationError("definition is missing source_repository_binding_id")
    return AutoResearchBindingPlan(
        template_id=definition.template_id,
        definition_digest=definition.definition_digest,
        model_ref=definition.target_model.base_model_ref,
        target_model_digest=canonical_hash(definition.target_model.model_dump(mode="json")),
        target_contract_key=definition.target_model.target_contract_key,
        dataset_version_id=dataset_version_id,
        ledger_project_id=policy.ledger_project_id,
        evaluation_suite_id=policy.evaluation_suite_id,
        primary_metric=policy.primary_metric,
        metric_direction=policy.metric_direction,
        compute_profile_id=definition.manifest.compute_profile_id,
        source_repository_profile_id=source_repository_profile_id,
    )


def _read_definition(path: Path) -> AutoResearchTemplateDefinition:
    if path.is_symlink() or not path.is_file():
        raise AutoResearchInstallationError(f"unsafe installed template path: {path.name}")
    raw = path.read_bytes()
    if len(raw) > _MAX_INSTALLATION_TEMPLATE_BYTES:
        raise AutoResearchInstallationError(f"installed template is too large: {path.name}")
    try:
        return AutoResearchTemplateDefinition.model_validate_json(raw)
    except Exception as exc:
        raise AutoResearchInstallationError(f"installed template is invalid: {path.name}") from exc


def install_autoresearch_definition(
    definition: AutoResearchTemplateDefinition,
    *,
    directory: Path,
    replace: bool = False,
) -> AutoResearchInstallationReceipt:
    """Validate and atomically install one definition, idempotently by digest."""

    binding_plan = autoresearch_binding_plan(definition)
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not directory.is_dir():
        raise AutoResearchInstallationError("AutoResearch installation path is not a directory")
    target = directory / f"{definition.template_id}.json"
    if target.exists() or target.is_symlink():
        installed = _read_definition(target)
        if installed.definition_digest == definition.definition_digest:
            return AutoResearchInstallationReceipt(
                path=os.fspath(target),
                created=False,
                replaced=False,
                binding_plan=binding_plan,
            )
        if not replace:
            raise AutoResearchInstallationConflictError(
                f"template {definition.template_id!r} is already installed with a different digest"
            )

    encoded = (
        json.dumps(definition.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_INSTALLATION_TEMPLATE_BYTES:
        raise AutoResearchInstallationError("AutoResearch definition exceeds installation limit")
    temporary = directory / f".{definition.template_id}.{uuid.uuid4().hex}.tmp"
    target_existed = target.exists()
    created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
            created = not target_existed
        else:
            try:
                os.link(temporary, target)
                created = True
            except FileExistsError:
                installed = _read_definition(target)
                if installed.definition_digest != definition.definition_digest:
                    raise AutoResearchInstallationConflictError(
                        f"template {definition.template_id!r} changed during installation"
                    )
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return AutoResearchInstallationReceipt(
        path=os.fspath(target),
        created=created,
        replaced=replace and not created,
        binding_plan=binding_plan,
    )


__all__ = [
    "AutoResearchBindingPlan",
    "AutoResearchInstallationConflictError",
    "AutoResearchInstallationError",
    "AutoResearchInstallationReceipt",
    "autoresearch_binding_plan",
    "build_quality_autoresearch_definition",
    "install_autoresearch_definition",
]
