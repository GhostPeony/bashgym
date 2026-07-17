"""Deterministic validation for actor-authored campaign study proposals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from bashgym.campaigns.contracts import (
    ActorPrincipal,
    CampaignManifest,
    Capability,
    ProposalValidation,
    StageDisposition,
    StageKind,
    StudyProposalSubmission,
)

_FORBIDDEN_EXECUTION_KEYS = frozenset(
    {
        "host",
        "username",
        "port",
        "key_path",
        "work_dir",
        "remote_work_dir",
        "script_path",
        "script_args",
        "input_files",
        "output_paths",
        "python_executable",
        "pid",
        "remote_pid",
        "process_group_id",
    }
)
_FAKE_RUNTIME_KEYS = frozenset({"executor_kind", "budget_unit", "budget_reservation", "fake_steps"})
_LIVE_RUNTIME_KEYS = frozenset({"executor_kind"})
_REGISTERED_COMPUTE_KINDS = frozenset({"registered_compute", "registered_training", "ssh_remote"})


def _recipe_has_schema_version(recipe: Mapping[str, Any]) -> bool:
    value = recipe.get("schema_version")
    return isinstance(value, str) and bool(value.strip())


def _declared_data_scopes(recipes: Iterable[Mapping[str, Any]]) -> set[str]:
    scopes: set[str] = set()
    for recipe in recipes:
        single = recipe.get("data_scope_id")
        if isinstance(single, str):
            scopes.add(single)
        for key in ("data_scope_ids", "approved_data_scopes"):
            values = recipe.get(key)
            if isinstance(values, (list, tuple)):
                scopes.update(value for value in values if isinstance(value, str))
    return scopes


def _contains_forbidden_execution_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(str(key) in _FORBIDDEN_EXECUTION_KEYS for key in value):
            return True
        return any(_contains_forbidden_execution_material(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_execution_material(item) for item in value)
    return False


def _runtime_policy_reasons(recipe: Mapping[str, Any], *, live_compute_allowed: bool) -> set[str]:
    runtime = recipe.get("runtime")
    if runtime is None:
        return set()
    if not isinstance(runtime, Mapping):
        return {"proposal_recipe_runtime_invalid"}
    executor_kind = runtime.get("executor_kind", "fake")
    if executor_kind == "fake":
        allowed = _FAKE_RUNTIME_KEYS
    elif executor_kind in _REGISTERED_COMPUTE_KINDS and live_compute_allowed:
        allowed = _LIVE_RUNTIME_KEYS
    else:
        return {"proposal_executor_kind_not_allowed"}
    if not set(runtime).issubset(allowed):
        return {"proposal_runtime_keys_not_allowed"}
    if executor_kind == "fake":
        reservation = runtime.get("budget_reservation", 0.01)
        steps = runtime.get("fake_steps", 8)
        budget_unit = runtime.get("budget_unit")
        if (
            isinstance(reservation, bool)
            or not isinstance(reservation, (int, float))
            or not 0 < float(reservation)
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 2 <= steps <= 10_000
            or (budget_unit is not None and not isinstance(budget_unit, str))
        ):
            return {"proposal_recipe_runtime_invalid"}
    return set()


def validate_proposal_submission(
    submission: StudyProposalSubmission,
    manifest: CampaignManifest,
    principal: ActorPrincipal,
    *,
    existing_prerequisite_ids: frozenset[str],
) -> ProposalValidation:
    """Return stable policy reason codes without executing actor-authored content."""

    reasons: set[str] = set()
    controlled = tuple(value.strip() for value in submission.controlled_variables)
    if len(set(controlled)) != len(controlled):
        reasons.add("proposal_controlled_variables_not_unique")
    if submission.primary_variable.strip() in set(controlled):
        reasons.add("proposal_primary_variable_is_controlled")
    if len(set(submission.prerequisite_study_ids)) != len(submission.prerequisite_study_ids):
        reasons.add("proposal_prerequisites_not_unique")
    if not set(submission.prerequisite_study_ids).issubset(existing_prerequisite_ids):
        reasons.add("proposal_prerequisite_not_found")

    recipes = (
        submission.dataset_recipe,
        submission.training_recipe,
        submission.evaluation_recipe,
    )
    if any(not _recipe_has_schema_version(recipe) for recipe in recipes):
        reasons.add("proposal_recipe_schema_missing")
    if any(_contains_forbidden_execution_material(recipe) for recipe in recipes):
        reasons.add("proposal_executable_material_forbidden")
    reasons.update(_runtime_policy_reasons(submission.dataset_recipe, live_compute_allowed=False))
    reasons.update(_runtime_policy_reasons(submission.training_recipe, live_compute_allowed=True))
    reasons.update(_runtime_policy_reasons(submission.evaluation_recipe, live_compute_allowed=True))

    required = set(submission.required_capabilities)
    if not required.issubset(principal.capabilities):
        reasons.add("proposal_capability_not_authorized")

    declared_scopes = _declared_data_scopes(recipes)
    if not declared_scopes.issubset(manifest.approved_data_scopes):
        reasons.add("proposal_data_scope_not_approved")

    required_stages = {
        item.stage
        for item in submission.stage_plan.items
        if item.disposition == StageDisposition.REQUIRED
    }
    training_runtime = submission.training_recipe.get("runtime")
    live_training = (
        isinstance(training_runtime, Mapping)
        and training_runtime.get("executor_kind") in _REGISTERED_COMPUTE_KINDS
    )
    evaluation_runtime = submission.evaluation_recipe.get("runtime")
    live_evaluation = (
        isinstance(evaluation_runtime, Mapping)
        and evaluation_runtime.get("executor_kind") in _REGISTERED_COMPUTE_KINDS
    )
    if live_training:
        if StageKind.SMOKE_TRAINING in required_stages and Capability.COMPUTE_SMOKE not in required:
            reasons.add("proposal_compute_smoke_capability_missing")
        if (
            StageKind.FULL_TRAINING in required_stages
            and Capability.COMPUTE_TRAIN_WITHIN_BUDGET not in required
        ):
            reasons.add("proposal_compute_training_capability_missing")
    if (
        live_evaluation
        and StageKind.DEVELOPMENT_EVALUATION in required_stages
        and Capability.EVAL_DEVELOPMENT not in required
    ):
        reasons.add("proposal_development_evaluation_capability_missing")
    if StageKind.PROTECTED_EVALUATION in required_stages:
        if not manifest.protected_artifact_refs:
            reasons.add("proposal_protected_evaluation_not_configured")
        protected_caps = {
            Capability.EVAL_PROTECTED_ACQUIRE,
            Capability.EVAL_PROTECTED_EXECUTE,
        }
        if not protected_caps.issubset(required):
            reasons.add("proposal_protected_capabilities_missing")
    if StageKind.PROMOTION in required_stages and Capability.PROMOTION_DECIDE not in required:
        reasons.add("proposal_promotion_capability_missing")
    if Capability.ARTIFACT_PUBLISH_HF in required and not manifest.allow_hf_publication:
        reasons.add("proposal_hf_publication_not_approved")
    if Capability.HANDOFF_EXTERNAL_PREPARE in required and not manifest.allow_external_handoff:
        reasons.add("proposal_external_handoff_not_approved")
    if Capability.HANDOFF_MEMEXAI_PREPARE in required:
        reasons.add("proposal_legacy_handoff_read_only")

    reason_codes = tuple(sorted(reasons))
    return ProposalValidation(valid=not reason_codes, reason_codes=reason_codes)


__all__ = ["validate_proposal_submission"]
