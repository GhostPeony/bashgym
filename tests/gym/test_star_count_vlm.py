"""Star-count VLM recipe, collation, and metric tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from bashgym.environments.star_count import (
    create_star_count_archive,
    generate_star_count_dataset,
)
from bashgym.gym.star_count_vlm import (
    StarCountVLMDataCollator,
    StarCountVLMRecipe,
    extract_star_count_archive,
    load_star_count_records,
    main,
    summarize_star_count_failures,
    summarize_star_count_predictions,
)


def test_recipe_requires_an_exact_model_revision():
    recipe = StarCountVLMRecipe(
        model_id="unsloth/gemma-4-E2B-it",
        model_revision="d91d4cb1ad34506d670ac82a69c460fb1e207492",
        max_steps=20,
    )
    assert recipe.local_files_only is True
    assert recipe.max_steps == 20

    with pytest.raises(ValueError, match="immutable revision"):
        StarCountVLMRecipe(
            model_id="unsloth/gemma-4-E2B-it",
            model_revision="main",
        )


def test_collator_keeps_pixels_and_masks_the_prompt(tmp_path):
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        pad_token_id = 0

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **_kwargs):
            return "prompt" if len(messages) == 1 else "full"

        def __call__(self, *, text, images, **_kwargs):
            texts = [text] if isinstance(text, str) else text
            length = 3 if texts == ["prompt"] else 5
            return {
                "input_ids": torch.tensor([[index + 1 for index in range(length)]] * len(texts)),
                "attention_mask": torch.ones((len(texts), length), dtype=torch.long),
                "pixel_values": torch.ones((len(images), 3, 2, 2)),
            }

    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    record = {
        "image_path": image_path,
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "count"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "red=1"}]},
        ],
    }

    batch = StarCountVLMDataCollator(FakeProcessor(), max_length=128)([record])

    assert batch["pixel_values"].shape == (1, 3, 2, 2)
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]


def test_loader_rejects_images_outside_the_dataset_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), "white").save(outside)
    record = {
        "example_id": "heldout-000000",
        "image": "../outside.png",
        "counts": {"red": 1, "blue": 0, "green": 0, "yellow": 0},
        "messages": [],
    }
    (root / "heldout.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes dataset root"):
        load_star_count_records(root / "heldout.jsonl")


def test_prediction_summary_uses_fixed_exact_count_metric():
    rows = [
        {
            "prediction": "red=1, blue=0, green=2, yellow=0",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
        {
            "prediction": "red=1, blue=1, green=2, yellow=0",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
    ]

    summary = summarize_star_count_predictions(rows)

    assert summary["primary_metric"] == "exact_count_accuracy"
    assert summary["exact_count_accuracy"] == pytest.approx(0.5)
    assert summary["count_accuracy"] == pytest.approx(0.875)
    assert summary["format_accuracy"] == pytest.approx(1.0)
    assert summary["example_count"] == 2


def test_failure_summary_emits_mutually_exclusive_aggregate_categories():
    rows = [
        {
            "example_id": "exact",
            "prediction": "red=1, blue=0, green=2, yellow=0",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
        {
            "example_id": "count",
            "prediction": "red=1, blue=1, green=2, yellow=0",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
        {
            "example_id": "format",
            "prediction": "red: 1 blue: 0 green: 2 yellow: 0",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
        {
            "example_id": "both",
            "prediction": "I see several stars.",
            "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
        },
    ]

    summaries = [item.model_dump(mode="json") for item in summarize_star_count_failures(rows)]

    assert [(item["category"], item["count"]) for item in summaries] == [
        ("count_and_format_error", 1),
        ("count_error", 1),
        ("format_error", 1),
    ]
    serialized = json.dumps(summaries, sort_keys=True)
    assert "example_id" not in serialized
    assert "prediction" not in serialized
    assert '"counts"' not in serialized


def test_archive_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.json", "{}")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_star_count_archive(archive, tmp_path / "output")


def test_archive_extraction_verifies_manifest_and_member_hashes(tmp_path):
    dataset = tmp_path / "dataset"
    generate_star_count_dataset(dataset, train_size=1, validation_size=1, heldout_size=1)
    archive = tmp_path / "dataset.zip"
    create_star_count_archive(dataset, archive)

    extracted = extract_star_count_archive(archive, tmp_path / "extracted")
    assert len(load_star_count_records(extracted / "heldout.jsonl")) == 1

    corrupt = tmp_path / "corrupt.zip"
    image_path = next((dataset / "images").rglob("*.png"))
    with zipfile.ZipFile(corrupt, "w") as bundle:
        for path in sorted(item for item in dataset.rglob("*") if item.is_file()):
            payload = b"corrupt" if path == image_path else path.read_bytes()
            bundle.writestr(path.relative_to(dataset).as_posix(), payload)

    with pytest.raises(ValueError, match="manifest verification failed"):
        extract_star_count_archive(corrupt, tmp_path / "corrupt-output")


def test_train_command_can_evaluate_the_sealed_candidate(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("train", "validation", "heldout"):
        (dataset / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "candidate"
    calls = {}

    monkeypatch.setattr(
        "bashgym.gym.star_count_vlm.extract_star_count_archive",
        lambda *_args: dataset,
    )

    def fake_train(_recipe, **kwargs):
        calls["training"] = kwargs
        return {"train_loss": 1.0}

    monkeypatch.setattr("bashgym.gym.star_count_vlm.train_star_count_lora", fake_train)

    def fake_evaluate(_recipe, **kwargs):
        calls["evaluation"] = kwargs
        return {"metrics": {"exact_count_accuracy": 0.5}}

    monkeypatch.setattr("bashgym.gym.star_count_vlm.evaluate_star_count_model", fake_evaluate)

    assert (
        main(
            [
                "train",
                "--model-id",
                "example/model",
                "--model-revision",
                "a" * 40,
                "--dataset-archive",
                str(tmp_path / "dataset.zip"),
                "--output",
                str(output),
                "--evaluate-heldout",
            ]
        )
        == 0
    )
    assert calls["evaluation"]["heldout_jsonl"] == dataset / "heldout.jsonl"
    assert calls["evaluation"]["adapter_path"] == output / "final_adapter"
    assert calls["evaluation"]["output_path"] == str(output / "evaluation_result.json")


def test_autoresearch_train_publishes_final_and_bounded_retained_checkpoints(monkeypatch, tmp_path):
    model_dir = tmp_path / "base-model"
    model_dir.mkdir()
    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    train_jsonl.write_text("{}\n", encoding="utf-8")
    validation_jsonl.write_text("{}\n", encoding="utf-8")
    recipe_file = tmp_path / "star-count-autoresearch-recipe.json"
    recipe_file.write_text(
        json.dumps(
            {
                "schema_version": "star_count_autoresearch_training_recipe.v1",
                "model_revision": "a" * 40,
                "train_jsonl": str(train_jsonl),
                "validation_jsonl": str(validation_jsonl),
                "max_steps": 80,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_train(_recipe, **kwargs):
        output = Path(kwargs["output_dir"])
        adapter = output / "final_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(model_dir)}), encoding="utf-8"
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        for step in range(10, 100, 10):
            checkpoint = output / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": str(model_dir)}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(f"step-{step}".encode())
        return {"train_loss": 0.25}

    monkeypatch.setattr("bashgym.gym.star_count_vlm.train_star_count_lora", fake_train)

    assert (
        main(
            [
                "autoresearch-train",
                "--model-dir",
                str(model_dir),
                "--recipe-file",
                str(recipe_file),
            ]
        )
        == 0
    )
    assert (tmp_path / "final" / "adapter_config.json").is_file()
    assert (tmp_path / "final" / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert not (tmp_path / "final" / "final_adapter").exists()
    retained = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
    assert retained == [
        "step-20",
        "step-30",
        "step-40",
        "step-50",
        "step-60",
        "step-70",
        "step-80",
        "step-90",
    ]
    assert not (tmp_path / "training" / "checkpoint-10").exists()


def test_autoresearch_evaluate_emits_context_bound_standard_evidence(monkeypatch, tmp_path):
    context = {
        "schema_version": "autoresearch_evaluation_context.v1",
        "workspace_id": "workspace-demo",
        "campaign_id": "campaign-demo",
        "study_id": "study-demo",
        "action_id": "action-demo",
        "attempt_id": "attempt-demo",
        "candidate_digest": "1" * 64,
        "evaluation_suite_id": "suite-demo",
        "evaluation_code_digest": "2" * 64,
        "dataset_version_id": "dataset-version-demo",
        "dataset_content_digest": "3" * 64,
        "evaluated_model_manifest_digest": "4" * 64,
    }
    context_path = tmp_path / "autoresearch_evaluation_context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    dataset_path = tmp_path / "heldout.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    adapter = tmp_path / "candidate-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model)}), encoding="utf-8"
    )
    output_path = tmp_path / "autoresearch_evaluation.json"

    def fake_evaluate(recipe, **kwargs):
        assert recipe.model_id == str(base_model)
        assert kwargs["adapter_path"] == adapter
        return {
            "adapter_evaluated": True,
            "metrics": {
                "primary_metric": "exact_count_accuracy",
                "exact_count_accuracy": 0.625,
                "count_accuracy": 0.875,
                "format_accuracy": 1.0,
                "mean_reward": 0.88125,
                "example_count": 64,
            },
            "predictions": [
                {
                    "example_id": "heldout-1",
                    "prediction": "red=1, blue=0, green=2, yellow=0",
                    "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
                }
            ],
        }

    monkeypatch.setattr("bashgym.gym.star_count_vlm.evaluate_star_count_model", fake_evaluate)

    assert (
        main(
            [
                "autoresearch-evaluate",
                "--model-revision",
                "a" * 40,
                "--context",
                str(context_path),
                "--model-dir",
                str(adapter),
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "autoresearch_evaluation_evidence.v1"
    assert evidence["campaign_id"] == "campaign-demo"
    assert evidence["attempt_id"] == "attempt-demo"
    assert evidence["evaluated_model_manifest_digest"] == "4" * 64
    assert evidence["metrics"] == {
        "count_accuracy": 0.875,
        "exact_count_accuracy": 0.625,
        "format_accuracy": 1.0,
        "mean_reward": 0.88125,
    }
    assert evidence["slice_metrics"] == {
        "adapter_evaluated": True,
        "example_count": 64,
    }


def test_autoresearch_evaluate_records_bounded_checkpoint_trajectory(monkeypatch, tmp_path):
    context = {
        "schema_version": "autoresearch_evaluation_context.v1",
        "workspace_id": "workspace-demo",
        "campaign_id": "campaign-demo",
        "study_id": "study-demo",
        "action_id": "action-demo",
        "attempt_id": "attempt-demo",
        "candidate_digest": "1" * 64,
        "evaluation_suite_id": "suite-demo",
        "evaluation_code_digest": "2" * 64,
        "dataset_version_id": "dataset-version-demo",
        "dataset_content_digest": "3" * 64,
        "evaluated_model_manifest_digest": "4" * 64,
    }
    context_path = tmp_path / "autoresearch_evaluation_context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    dataset_path = tmp_path / "heldout.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    checkpoints = tmp_path / "run" / "checkpoints"
    for step in (20, 40, 60):
        checkpoint = checkpoints / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(base_model)}), encoding="utf-8"
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(f"step-{step}".encode())
    (final / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model)}), encoding="utf-8"
    )
    output_path = tmp_path / "autoresearch_evaluation.json"

    def fake_evaluate(_recipe, **kwargs):
        adapter = Path(kwargs["adapter_path"])
        step = int(adapter.name.removeprefix("step-")) if adapter.name.startswith("step-") else 80
        return {
            "adapter_evaluated": True,
            "metrics": {
                "exact_count_accuracy": step / 100,
                "example_count": 64,
            },
            "predictions": [
                {
                    "example_id": "heldout-1",
                    "prediction": "red=1, blue=0, green=2, yellow=0",
                    "counts": {"red": 1, "blue": 0, "green": 2, "yellow": 0},
                }
            ],
        }

    monkeypatch.setattr("bashgym.gym.star_count_vlm.evaluate_star_count_model", fake_evaluate)

    assert (
        main(
            [
                "autoresearch-evaluate",
                "--model-revision",
                "a" * 40,
                "--context",
                str(context_path),
                "--model-dir",
                str(final),
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_path),
                "--checkpoint-limit",
                "2",
            ]
        )
        == 0
    )

    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    observations = evidence["checkpoint_observations"]
    assert [item["checkpoint_step"] for item in observations] == [40, 60]
    assert [item["metrics"]["exact_count_accuracy"] for item in observations] == [0.4, 0.6]
    assert all(len(item["evaluated_model_manifest_digest"]) == 64 for item in observations)
