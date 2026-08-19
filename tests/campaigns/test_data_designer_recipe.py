"""Typed AutoResearch data-design recipe and installed runner contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.data_designer_recipe import (
    AutoResearchDataDesignRecipe,
    DataDesignerPipelinePolicy,
    DataDesignerRunnerContract,
)


def policy(**overrides: object) -> DataDesignerPipelinePolicy:
    values: dict[str, object] = {
        "pipeline": "coding_agent_sft",
        "min_rows": 16,
        "max_rows": 256,
        "required_columns": ("messages", "label"),
        "allowed_labels": {"label": ("zero", "one", "many")},
    }
    values.update(overrides)
    return DataDesignerPipelinePolicy(**values)


def contract(tmp_path: Path, **overrides: object) -> DataDesignerRunnerContract:
    parent = (tmp_path / "parent.jsonl").resolve()
    fingerprints = (tmp_path / "heldout.sha256").resolve()
    values: dict[str, object] = {
        "parent_dataset_version_id": "dataset-version-1",
        "parent_dataset_path": parent,
        "parent_dataset_sha256": "a" * 64,
        "protected_fingerprints_path": fingerprints,
        "protected_fingerprints_sha256": "b" * 64,
        "provider_name": "local-generator",
        "provider_endpoint": "http://127.0.0.1:8001/v1",
        "text_model": "generator-model-v1",
        "code_model": "generator-model-v1",
        "judge_model": "generator-model-v1",
        "verifier_digest": "c" * 64,
        "pipeline_policies": (policy(),),
    }
    values.update(overrides)
    return DataDesignerRunnerContract(**values)


def test_agent_authors_a_novel_design_inside_the_execution_envelope() -> None:
    recipe = AutoResearchDataDesignRecipe(
        hypothesis="More stateful debugging tasks will improve repair success.",
        pipeline="coding_agent_sft",
        generation_brief=(
            "Generate stateful debugging and test-repair examples concentrated on "
            "multi-step diagnosis, with balanced success and recovery cases."
        ),
        target_rows=128,
        train_fraction=0.8,
        seed=17,
    )

    assert recipe.runtime == {"executor_kind": "registered_training"}
    assert recipe.script_args()[0] == "--recipe-json"
    assert AutoResearchDataDesignRecipe.model_validate_json(recipe.script_args()[1]) == recipe


def test_recipe_rejects_execution_material_and_unregistered_runtime() -> None:
    with pytest.raises(ValidationError):
        AutoResearchDataDesignRecipe(
            hypothesis="Try a new data mixture.",
            pipeline="coding_agent_sft",
            generation_brief="Generate targeted debugging examples.",
            target_rows=128,
            train_fraction=0.8,
            seed=17,
            script_args=["--provider-endpoint", "http://unreviewed.invalid/v1"],
        )
    with pytest.raises(ValidationError, match="registered training runtime"):
        AutoResearchDataDesignRecipe(
            runtime={"executor_kind": "fake"},
            hypothesis="Try a new data mixture.",
            pipeline="coding_agent_sft",
            generation_brief="Generate targeted debugging examples.",
            target_rows=128,
            train_fraction=0.8,
            seed=17,
        )


def test_runner_contract_owns_exact_generator_and_bounded_designs(tmp_path: Path) -> None:
    value = contract(tmp_path)

    assert value.policy("coding_agent_sft") == policy()
    assert str(value.provider_endpoint) == "http://127.0.0.1:8001/v1"
    with pytest.raises(KeyError, match="unknown-pipeline"):
        value.policy("unknown-pipeline")

    with pytest.raises(ValidationError, match="unique"):
        contract(tmp_path, pipeline_policies=(policy(), policy()))
    with pytest.raises(ValidationError):
        contract(tmp_path, text_model="")
    with pytest.raises(ValidationError, match="row bounds"):
        policy(min_rows=256, max_rows=16)
    with pytest.raises(ValidationError, match="absolute"):
        contract(tmp_path, parent_dataset_path=Path("relative.jsonl"))
    with pytest.raises(ValidationError, match="absolute"):
        contract(tmp_path, protected_fingerprints_path=Path("relative.sha256"))


def test_contract_bounds_but_does_not_predeclare_agent_designs(tmp_path: Path) -> None:
    value = contract(tmp_path)
    first = AutoResearchDataDesignRecipe(
        hypothesis="Target stateful shell failures.",
        pipeline="coding_agent_sft",
        generation_brief="Generate stateful shell debugging tasks.",
        target_rows=64,
        train_fraction=0.75,
        seed=1,
    )
    second = first.model_copy(
        update={
            "hypothesis": "Target test-repair failures.",
            "generation_brief": "Generate test-repair tasks with misleading initial failures.",
            "target_rows": 192,
        }
    )

    assert value.validate_recipe(first) == first
    assert value.validate_recipe(second) == second
    with pytest.raises(ValueError, match="row count"):
        value.validate_recipe(second.model_copy(update={"target_rows": 512}))
