"""Typed experiment controls for the TMax runner."""

import pytest
from pydantic import ValidationError

from bashgym.campaigns.tmax_recipe import TMaxCompositeTrainingRecipe


def test_tmax_recipe_renders_one_canonical_grpo_rlvr_abi():
    recipe = TMaxCompositeTrainingRecipe(
        algorithm="grpo",
        sft_enabled=False,
        learning_rate=0.00002,
        max_steps=250,
        group_size=16,
        temperature=0.7,
        seed=7,
    )

    assert recipe.script_args() == (
        "--algorithm",
        "grpo",
        "--sft-enabled",
        "false",
        "--learning-rate",
        "2e-05",
        "--max-steps",
        "250",
        "--group-size",
        "16",
        "--temperature",
        "0.7",
        "--seed",
        "7",
    )


def test_tmax_recipe_defaults_to_grpo_without_sft_and_rejects_dppo_or_paths():
    recipe = TMaxCompositeTrainingRecipe()

    assert recipe.algorithm == "grpo"
    assert recipe.sft_enabled is False
    with pytest.raises(ValidationError):
        TMaxCompositeTrainingRecipe(algorithm="dppo")
    with pytest.raises(ValidationError):
        TMaxCompositeTrainingRecipe(algorithm="rlvr")
    with pytest.raises(ValidationError):
        TMaxCompositeTrainingRecipe(model_path="/private/models/model")
    with pytest.raises(ValidationError, match="False"):
        TMaxCompositeTrainingRecipe(sft_enabled=True)
