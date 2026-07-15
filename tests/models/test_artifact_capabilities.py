import pytest

from bashgym.models.artifact_capabilities import (
    GEMMA4_12B_TRAINING_MODEL,
    classify_model_artifact,
    require_trainable_base,
)


def test_nvfp4_is_a_vllm_inference_quant():
    result = classify_model_artifact("unsloth/gemma-4-12b-it-NVFP4")

    assert result.artifact_role == "inference_quant"
    assert result.quantization == "nvfp4"
    assert result.runtime == "vllm"
    assert result.trainable is False


def test_matching_12b_base_is_trainable():
    result = classify_model_artifact(GEMMA4_12B_TRAINING_MODEL)

    assert result.artifact_role == "trainable_base"
    assert result.trainable is True
    require_trainable_base(GEMMA4_12B_TRAINING_MODEL)


def test_nvfp4_training_error_names_the_matching_base():
    with pytest.raises(ValueError, match="unsloth/gemma-4-12b-it"):
        require_trainable_base("unsloth/gemma-4-12b-it-NVFP4")
