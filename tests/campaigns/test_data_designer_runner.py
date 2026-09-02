"""Durable Data Designer runner for AutoResearch DATA_BUILD stages."""

import hashlib
import json
from pathlib import Path

from bashgym.campaigns.autoresearch_dataset import AutoResearchDatasetGeneration
from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.data_designer_recipe import (
    AutoResearchDataDesignRecipe,
    DataDesignerPipelinePolicy,
    DataDesignerRunnerContract,
)
from bashgym.campaigns.data_designer_runner import canonical_row_digest, run_contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingPipeline:
    configs = []
    sources = []

    def __init__(self, config):
        self.configs.append(config)

    def from_dataset(self, source: str, *, num_records: int):
        self.sources.append((source, num_records))
        protected = {"messages": [{"role": "user", "content": "protected"}], "label": "one"}
        return RecordingFrame(
            (
                {"messages": [{"role": "user", "content": "keep-a"}], "label": "zero"},
                {"messages": [{"role": "user", "content": "keep-a"}], "label": "zero"},
                protected,
                {"messages": [{"role": "user", "content": "missing-label"}]},
                {"messages": [{"role": "user", "content": "bad-label"}], "label": "other"},
                {"messages": [{"role": "user", "content": "keep-b"}], "label": "many"},
            )
        )


class RecordingFrame:
    def __init__(self, records):
        self.records = records

    def to_dict(self, *, orient: str):
        assert orient == "records"
        return list(self.records)


def _inputs(tmp_path: Path):
    parent = tmp_path / "parent.jsonl"
    parent.write_text('{"messages":[],"label":"zero"}\n', encoding="utf-8")
    protected_row = {
        "messages": [{"role": "user", "content": "protected"}],
        "label": "one",
    }
    fingerprints = tmp_path / "heldout.sha256"
    fingerprints.write_text(canonical_row_digest(protected_row) + "\n", encoding="ascii")
    policy = DataDesignerPipelinePolicy(
        pipeline="coding_agent_sft",
        min_rows=2,
        max_rows=10,
        required_columns=("messages", "label"),
        allowed_labels={"label": ("zero", "one", "many")},
    )
    contract = DataDesignerRunnerContract(
        parent_dataset_version_id="dataset-version-1",
        parent_dataset_path=parent.resolve(),
        parent_dataset_sha256=_sha256(parent),
        protected_fingerprints_path=fingerprints.resolve(),
        protected_fingerprints_sha256=_sha256(fingerprints),
        provider_name="local-generator",
        provider_endpoint="http://127.0.0.1:8001/v1",
        text_model="generator-model-v1",
        code_model="generator-model-v1",
        judge_model="judge-model-v1",
        verifier_digest="c" * 64,
        pipeline_policies=(policy,),
    )
    recipe = AutoResearchDataDesignRecipe(
        hypothesis="Target difficult repair failures.",
        pipeline="coding_agent_sft",
        generation_brief="Generate difficult multi-step debugging and repair examples.",
        target_rows=6,
        train_fraction=0.5,
        seed=17,
    )
    return contract, recipe


def test_runner_uses_exact_models_and_emits_validated_immutable_dataset(tmp_path: Path) -> None:
    RecordingPipeline.configs.clear()
    RecordingPipeline.sources.clear()
    contract, recipe = _inputs(tmp_path)

    receipt = run_contract(
        contract,
        recipe,
        tmp_path / "run",
        pipeline_factory=RecordingPipeline,
    )

    config = RecordingPipeline.configs[0]
    assert config.provider == "local-generator"
    assert config.provider_endpoint == "http://127.0.0.1:8001/v1"
    assert (config.text_model, config.code_model, config.judge_model) == (
        "generator-model-v1",
        "generator-model-v1",
        "judge-model-v1",
    )
    assert config.experiment_brief == recipe.generation_brief
    assert RecordingPipeline.sources == [(str(contract.parent_dataset_path), 6)]
    assert receipt.row_counts == {"train": 1, "validation": 1}
    assert receipt.quality is not None
    assert receipt.quality.model_dump(mode="json") | {} == {
        "schema_version": "autoresearch_dataset_quality.v1",
        "generated_rows": 6,
        "accepted_rows": 2,
        "deterministic_verified_rows": 4,
        "verification_failed_rows": 2,
        "duplicate_rows_removed": 1,
        "contamination_rows_removed": 1,
        "verifier_digest": "c" * 64,
    }
    assert isinstance(receipt.generator, AutoResearchDatasetGeneration)
    assert receipt.generator.parent_dataset_version_id == "dataset-version-1"
    assert receipt.generator.hypothesis == recipe.hypothesis
    assert receipt.generator.generation_brief == recipe.generation_brief
    assert receipt.generator.target_rows == 6
    assert receipt.generator.train_fraction == 0.5
    assert receipt.generator.recipe_digest == canonical_hash(recipe.model_dump(mode="json"))
    assert receipt.generator.split_seed == 17
    assert receipt.generator.generation_seed is None
    assert receipt.generator.generation_determinism == "provider_unseeded"
    assert receipt.generator.generation_config_digest == canonical_hash(
        {
            "pipeline": "coding_agent_sft",
            "provider": "local-generator",
            "models": {
                "text": "generator-model-v1",
                "code": "generator-model-v1",
                "judge": "judge-model-v1",
            },
            "num_records": 6,
            "buffer_size": 100,
            "max_parallel_requests": 4,
            "experiment_brief": recipe.generation_brief,
            "train_val_split": 0.5,
            "temperatures": {"text": 0.85, "code": 0.2, "judge": 0.1},
            "seed_source": None,
            "tools": {
                "enabled": False,
                "alias": "sandbox",
                "backend": "auto",
                "max_turns": 8,
                "timeout_seconds": 120,
            },
        }
    )
    assert len(receipt.generator.generator_implementation_digest) == 64
    assert receipt.generator.models == {
        "code": "generator-model-v1",
        "judge": "judge-model-v1",
        "text": "generator-model-v1",
    }
    assert "127.0.0.1" not in receipt.model_dump_json()

    train = (tmp_path / "run" / "dataset" / "train.jsonl").read_text(encoding="utf-8")
    validation = (tmp_path / "run" / "dataset" / "validation.jsonl").read_text(encoding="utf-8")
    assert "protected" not in train + validation
    assert "bad-label" not in train + validation
    assert "missing-label" not in train + validation
    assert {json.loads(train)["label"], json.loads(validation)["label"]} == {"zero", "many"}


def test_runner_replay_is_deterministic(tmp_path: Path) -> None:
    contract, recipe = _inputs(tmp_path)

    first = run_contract(
        contract,
        recipe,
        tmp_path / "run-a",
        pipeline_factory=RecordingPipeline,
    )
    second = run_contract(
        contract,
        recipe,
        tmp_path / "run-b",
        pipeline_factory=RecordingPipeline,
    )

    assert first.content_digest == second.content_digest
    assert first.files == second.files
    assert (
        first.generator.generator_implementation_digest
        == second.generator.generator_implementation_digest
    )
