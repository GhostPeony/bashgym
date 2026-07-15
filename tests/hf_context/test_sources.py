import json
from pathlib import Path

from bashgym.integrations.huggingface.context_contracts import (
    Comparability,
    EvidenceKind,
    Visibility,
)
from bashgym.integrations.huggingface.context_sources import (
    normalize_dataset,
    normalize_model,
    normalize_model_card_evals,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "hf_context"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_model_normalization_preserves_revision_access_size_and_safe_config_facts():
    item = normalize_model(fixture("rich_model.json"), intent="code generation")

    assert item.kind is EvidenceKind.MODEL
    assert item.revision == "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
    assert item.facts["license"] == "apache-2.0"
    assert item.facts["base_model"] == "Qwen/Qwen2.5-Coder-0.5B"
    assert item.facts["parameter_count"] == 494032768
    assert item.facts["dominant_dtype"] == "BF16"
    assert item.facts["chat_template_present"] is True
    assert len(item.facts["chat_template_hash"]) == 64
    assert "{% for" not in item.model_dump_json()
    assert item.assessment.task_relevance == 3


def test_structured_model_card_evals_are_orientation_only_when_harness_details_are_missing():
    results = normalize_model_card_evals(fixture("rich_model.json"))

    assert len(results) == 1
    result = results[0]
    assert result.kind is EvidenceKind.EVALUATION
    assert result.facts["score"] == 0.6203
    assert result.facts["raw_score"] == 62.03
    assert result.facts["verified"] is True
    assert result.eval_settings is not None
    assert result.eval_settings.few_shot == 0
    assert result.assessment.comparability is Comparability.ORIENTATION_ONLY
    assert "harness" in " ".join(result.cautions).lower()


def test_dataset_normalization_preserves_every_config_split_and_license_fallback():
    item = normalize_dataset(fixture("multi_config_dataset.json"), intent="software engineering")

    assert item.kind is EvidenceKind.DATASET
    assert item.revision == "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
    assert item.facts["license"] == "mit"
    assert [config["name"] for config in item.facts["configs"]] == ["default", "verified"]
    assert item.facts["configs"][0]["splits"][1]["num_examples"] == 300
    assert item.facts["total_rows"] == 373


def test_gated_or_private_resources_are_workspace_private():
    model = fixture("rich_model.json")
    model["gated"] = "manual"
    assert normalize_model(model).visibility is Visibility.WORKSPACE_PRIVATE

    dataset = fixture("multi_config_dataset.json")
    dataset["private"] = True
    assert normalize_dataset(dataset).visibility is Visibility.WORKSPACE_PRIVATE


def test_sparse_eval_metadata_never_upgrades_comparability():
    model = fixture("rich_model.json")
    result = model["card_data"]["eval_results"][0]
    result.update(
        {
            "dataset_config": None,
            "dataset_split": None,
            "dataset_args": None,
            "source_url": None,
            "verified": None,
        }
    )

    normalized = normalize_model_card_evals(model)[0]
    assert normalized.assessment.comparability is Comparability.ORIENTATION_ONLY
    assert normalized.eval_settings is not None
    assert normalized.eval_settings.few_shot is None
