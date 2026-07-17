"""Project applied AutoResearch activation evidence into the guided-setup registry.

Planning is read-only and emits logical identifiers only. Applying requires the
installation controller authority explicitly and writes one atomic registry
transaction; it never discovers, downloads, or substitutes a model.
"""

from __future__ import annotations

import hmac
import re
import sqlite3
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition
from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
from bashgym.campaigns.contracts import FrozenContractModel, Identifier
from bashgym.campaigns.installation import autoresearch_binding_plan
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.campaigns.worker_service import (
    WorkerRunConfig,
    WorkerServiceError,
    load_approved_remote_profiles,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository

MAX_REGISTRY_SYNC_DEFINITIONS = 64
_INSTALLATION_ID = re.compile(r"^ins_[0-9a-f]{32}$")


class AutoResearchRegistrySyncError(ValueError):
    """Actionable, non-secret registry synchronization failure."""


class AutoResearchRegistryBinding(FrozenContractModel):
    schema_version: Literal["autoresearch_registry_binding.v1"] = "autoresearch_registry_binding.v1"
    kind: Literal["model", "data", "evaluator", "compute"]
    logical_id: Identifier
    availability: Literal["reachable", "unknown"]
    reason_codes: tuple[Identifier, ...] = ()
    display_label: str | None = Field(default=None, max_length=120)
    integration_label: Literal["NeMo"] | None = None

    @model_validator(mode="after")
    def consistent_projection(self):
        if self.availability == "reachable" and self.reason_codes:
            raise ValueError("reachable registry binding cannot carry blockers")
        if self.availability == "unknown" and not self.reason_codes:
            raise ValueError("unknown registry binding requires a reason code")
        if self.kind != "model" and self.display_label is not None:
            raise ValueError("only model registry bindings may have labels")
        if self.kind != "compute" and self.integration_label is not None:
            raise ValueError("only compute registry bindings may have integrations")
        return self


class AutoResearchRegistrySyncPlan(FrozenContractModel):
    schema_version: Literal["autoresearch_registry_sync_plan.v1"] = (
        "autoresearch_registry_sync_plan.v1"
    )
    workspace_id: Identifier
    installation_id: str = Field(pattern=r"^ins_[0-9a-f]{32}$")
    template_ids: tuple[Identifier, ...]
    bindings: tuple[AutoResearchRegistryBinding, ...]
    ready: bool
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def canonical_plan(self):
        if not self.template_ids or tuple(sorted(set(self.template_ids))) != self.template_ids:
            raise ValueError("registry sync template identities must be sorted and unique")
        keys = tuple((item.kind, item.logical_id) for item in self.bindings)
        if not keys or tuple(sorted(set(keys))) != keys:
            raise ValueError("registry sync bindings must be sorted and unique")
        expected_ready = all(item.availability == "reachable" for item in self.bindings)
        if self.ready != expected_ready:
            raise ValueError("registry sync readiness conflicts with binding evidence")
        return self


class AutoResearchRegistrySyncReceipt(FrozenContractModel):
    schema_version: Literal["autoresearch_registry_sync_receipt.v1"] = (
        "autoresearch_registry_sync_receipt.v1"
    )
    installation_id: str = Field(pattern=r"^ins_[0-9a-f]{32}$")
    disposition: Literal["created", "updated", "replayed"]
    created_bindings: int = Field(ge=0)
    updated_bindings: int = Field(ge=0)
    replayed_bindings: int = Field(ge=0)


def resolve_autoresearch_installation_id(
    requested: str | None,
    *,
    apply: bool,
) -> str:
    """Generate a reviewable plan identity, but require it explicitly for apply."""

    if requested is not None:
        if _INSTALLATION_ID.fullmatch(requested) is None:
            raise AutoResearchRegistrySyncError("installation_identity_invalid")
        return requested
    if apply:
        raise AutoResearchRegistrySyncError("registry_apply_installation_identity_required")
    return f"ins_{uuid4().hex}"


def _evidence_binding(
    *,
    kind: Literal["model", "data", "evaluator", "compute"],
    logical_id: str,
    reachable: bool,
    reason: str,
    integration_label: Literal["NeMo"] | None = None,
) -> AutoResearchRegistryBinding:
    return AutoResearchRegistryBinding(
        kind=kind,
        logical_id=logical_id,
        availability="reachable" if reachable else "unknown",
        reason_codes=() if reachable else (reason,),
        integration_label=integration_label if reachable else None,
    )


def plan_autoresearch_registry_sync(
    *,
    definitions: tuple[AutoResearchTemplateDefinition, ...],
    workspace_id: str,
    installation_id: str,
    ledger: ExperimentLedgerRepository,
    worker_config: WorkerRunConfig,
) -> AutoResearchRegistrySyncPlan:
    """Build a bounded secret-free plan from exact persisted activation records."""

    if not definitions:
        raise AutoResearchRegistrySyncError("installed_definitions_unavailable")
    if len(definitions) > MAX_REGISTRY_SYNC_DEFINITIONS:
        raise AutoResearchRegistrySyncError("installed_definitions_limit_exceeded")
    if _INSTALLATION_ID.fullmatch(installation_id) is None:
        raise AutoResearchRegistrySyncError("installation_identity_invalid")
    ordered = tuple(sorted(definitions, key=lambda item: item.template_id))
    template_ids = tuple(item.template_id for item in ordered)
    if len(set(template_ids)) != len(template_ids):
        raise AutoResearchRegistrySyncError("installed_definition_identity_conflict")
    try:
        executors = load_approved_remote_profiles(worker_config)
    except WorkerServiceError as exc:
        raise AutoResearchRegistrySyncError("training_executor_registry_invalid") from exc

    candidates: dict[tuple[str, str], list[AutoResearchRegistryBinding]] = {}
    for definition in ordered:
        binding = autoresearch_binding_plan(definition)
        executor = executors.get((binding.compute_profile_id, binding.target_contract_key))
        required_stages = {stage.value for stage in binding.required_training_stages}
        executor_exact = bool(
            executor is not None
            and executor.target_model_digest == binding.target_model_digest
            and required_stages.issubset({stage.stage.value for stage in executor.stages})
        )
        integration = "NeMo" if executor_exact and executor and executor.nemo_rl else None
        for item in (
            _evidence_binding(
                kind="model",
                logical_id=binding.target_contract_key,
                reachable=executor_exact,
                reason="exact_training_executor_not_registered",
            ),
            _evidence_binding(
                kind="compute",
                logical_id=binding.compute_profile_id,
                reachable=executor_exact,
                reason="exact_training_executor_not_registered",
                integration_label=integration,
            ),
        ):
            candidates.setdefault((item.kind, item.logical_id), []).append(item)

        try:
            dataset = ledger.get_dataset_version(
                workspace_id, binding.ledger_project_id, binding.dataset_version_id
            )
            dataset_exact = dataset.get("dataset_version_id") == binding.dataset_version_id
        except RecordNotFoundError:
            dataset_exact = False
        data_item = _evidence_binding(
            kind="data",
            logical_id=binding.dataset_version_id,
            reachable=dataset_exact,
            reason="dataset_version_not_registered",
        )
        candidates.setdefault((data_item.kind, data_item.logical_id), []).append(data_item)

        try:
            evaluator = ledger.get_evaluation_suite(
                workspace_id, binding.ledger_project_id, binding.evaluation_suite_id
            )
            metric = evaluator.get("metric_contract", {})
            evaluator_exact = bool(
                evaluator.get("evaluation_suite_id") == binding.evaluation_suite_id
                and evaluator.get("dataset_version_id") == binding.dataset_version_id
                and metric.get("primary_metric") == binding.primary_metric
                and metric.get("metric_direction") == binding.metric_direction.value
            )
        except RecordNotFoundError:
            evaluator_exact = False
        eval_item = _evidence_binding(
            kind="evaluator",
            logical_id=binding.evaluation_suite_id,
            reachable=evaluator_exact,
            reason="evaluation_suite_not_registered",
        )
        candidates.setdefault((eval_item.kind, eval_item.logical_id), []).append(eval_item)

    projected: list[AutoResearchRegistryBinding] = []
    for key in sorted(candidates):
        values = candidates[key]
        reachable = all(item.availability == "reachable" for item in values)
        projected.append(
            AutoResearchRegistryBinding(
                kind=values[0].kind,
                logical_id=values[0].logical_id,
                availability="reachable" if reachable else "unknown",
                reason_codes=(
                    ()
                    if reachable
                    else tuple(sorted({code for item in values for code in item.reason_codes}))
                ),
                integration_label=(
                    "NeMo"
                    if reachable
                    and values[0].kind == "compute"
                    and all(item.integration_label == "NeMo" for item in values)
                    else None
                ),
            )
        )
    ready = all(item.availability == "reachable" for item in projected)
    reasons = tuple(sorted({code for item in projected for code in item.reason_codes}))
    return AutoResearchRegistrySyncPlan(
        workspace_id=workspace_id,
        installation_id=installation_id,
        template_ids=template_ids,
        bindings=tuple(projected),
        ready=ready,
        reason_codes=reasons,
    )


def apply_autoresearch_registry_sync(
    plan: AutoResearchRegistrySyncPlan,
    *,
    database_path: Path,
    controller_owner_id: str,
    controller_lease_key: str,
) -> AutoResearchRegistrySyncReceipt:
    """Atomically register the explicit installation authority and planned bindings."""

    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", controller_owner_id) is None
        or not controller_lease_key
        or len(controller_lease_key) > 4096
        or any(character in "\x00\r\n" for character in controller_lease_key)
    ):
        raise AutoResearchRegistrySyncError("installation_authority_incomplete")
    repository = CampaignRecoveryRepository(database_path, sealer=None)
    repository.initialize()
    created = updated = replayed = 0
    installation_created = False
    connection = sqlite3.connect(str(database_path), timeout=10)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("BEGIN IMMEDIATE")
        existing_installation = connection.execute(
            """
            SELECT controller_owner_id, controller_lease_key
            FROM campaign_recovery_installations WHERE installation_id=?
            """,
            (plan.installation_id,),
        ).fetchone()
        if existing_installation is None:
            connection.execute(
                """
                INSERT INTO campaign_recovery_installations(
                    installation_id, controller_owner_id, controller_lease_key
                ) VALUES (?, ?, ?)
                """,
                (plan.installation_id, controller_owner_id, controller_lease_key),
            )
            installation_created = True
        elif not (
            hmac.compare_digest(str(existing_installation[0]), controller_owner_id)
            and hmac.compare_digest(str(existing_installation[1]), controller_lease_key)
        ):
            raise AutoResearchRegistrySyncError("installation_authority_conflict")

        for binding in plan.bindings:
            desired = (
                binding.availability,
                binding.display_label,
                binding.integration_label,
            )
            existing = connection.execute(
                """
                SELECT availability, display_label, integration_label
                FROM campaign_recovery_bindings
                WHERE installation_id=? AND binding_kind=? AND logical_id=?
                """,
                (plan.installation_id, binding.kind, binding.logical_id),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO campaign_recovery_bindings(
                        installation_id, binding_kind, logical_id, availability,
                        display_label, integration_label
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (plan.installation_id, binding.kind, binding.logical_id, *desired),
                )
                created += 1
            elif tuple(existing) == desired:
                replayed += 1
            else:
                connection.execute(
                    """
                    UPDATE campaign_recovery_bindings
                    SET availability=?, display_label=?, integration_label=?
                    WHERE installation_id=? AND binding_kind=? AND logical_id=?
                    """,
                    (*desired, plan.installation_id, binding.kind, binding.logical_id),
                )
                updated += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    disposition: Literal["created", "updated", "replayed"]
    if installation_created or created:
        disposition = "created"
    elif updated:
        disposition = "updated"
    else:
        disposition = "replayed"
    return AutoResearchRegistrySyncReceipt(
        installation_id=plan.installation_id,
        disposition=disposition,
        created_bindings=created,
        updated_bindings=updated,
        replayed_bindings=replayed,
    )


__all__ = [
    "AutoResearchRegistryBinding",
    "AutoResearchRegistrySyncError",
    "AutoResearchRegistrySyncPlan",
    "AutoResearchRegistrySyncReceipt",
    "MAX_REGISTRY_SYNC_DEFINITIONS",
    "apply_autoresearch_registry_sync",
    "plan_autoresearch_registry_sync",
    "resolve_autoresearch_installation_id",
]
