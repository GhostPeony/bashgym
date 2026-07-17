"""Portable AutoResearch installation-definition tests."""

import pytest

from bashgym.campaigns.autoresearch import load_autoresearch_template_definitions
from bashgym.campaigns.installation import (
    AutoResearchInstallationConflictError,
    AutoResearchInstallationError,
    autoresearch_binding_plan,
    build_quality_autoresearch_definition,
    install_autoresearch_definition,
)

REVISION = "a" * 40


def definition(**overrides):
    values = {
        "template_id": "autoresearch-installed-v1",
        "template_revision": "v1",
        "objective": "Improve held-out task success with one controlled change per study.",
        "model_ref": f"hf://example/current-trainable-model@{REVISION}",
        "target_contract_key": "current-trainable-model-v1",
        "task": "heldout-task-autoresearch",
        "dataset_version_id": "dataset-version-1",
        "compute_profile_id": "private-training-1",
        "source_repository_profile_id": "bashgym-source-1",
        "ledger_project_id": "project-1",
        "evaluation_suite_id": "evaluation-suite-1",
        "primary_metric": "heldout_pass_at_1",
        "metric_direction": "maximize",
        "budget_unit": "gpu_hours",
        "budget_limit": 4.0,
        "max_attempts": 4,
        "minimum_improvement": 0.01,
    }
    values.update(overrides)
    return build_quality_autoresearch_definition(**values)


def test_builder_requires_an_explicit_immutable_trainable_base():
    with pytest.raises(AutoResearchInstallationError, match="immutable"):
        definition(model_ref="hf://example/current-trainable-model@main")

    built = definition()
    assert built.target_model.representation_contract == {
        "artifact_role": "trainable_base",
        "revision_binding": "immutable",
    }
    assert built.manifest.allow_hf_publication is False
    assert built.manifest.evaluation_plan["required_training_stages"] == [
        "smoke_training",
        "full_training",
    ]
    assert built.manifest.evaluation_plan["source_repository_binding_id"] == (
        "bashgym-source-1"
    )


def test_install_is_atomic_idempotent_and_emits_exact_binding_plan(tmp_path):
    built = definition()

    first = install_autoresearch_definition(built, directory=tmp_path)
    second = install_autoresearch_definition(built, directory=tmp_path)
    loaded = load_autoresearch_template_definitions(tmp_path)

    assert first.created is True
    assert first.replaced is False
    assert second.created is False
    assert second.replaced is False
    assert loaded == (built,)
    assert first.binding_plan == autoresearch_binding_plan(built)
    assert first.binding_plan.target_model_digest
    assert first.binding_plan.compute_profile_id == "private-training-1"
    assert first.binding_plan.source_repository_profile_id == "bashgym-source-1"
    assert tuple(stage.value for stage in first.binding_plan.required_training_stages) == (
        "smoke_training",
        "full_training",
    )


def test_install_refuses_silent_rebinding_and_requires_explicit_replace(tmp_path):
    original = definition()
    install_autoresearch_definition(original, directory=tmp_path)
    changed = original.model_copy(update={"objective": "A deliberately revised objective."})

    with pytest.raises(AutoResearchInstallationConflictError, match="different digest"):
        install_autoresearch_definition(changed, directory=tmp_path)

    receipt = install_autoresearch_definition(changed, directory=tmp_path, replace=True)
    assert receipt.created is False
    assert receipt.replaced is True
    assert load_autoresearch_template_definitions(tmp_path) == (changed,)
