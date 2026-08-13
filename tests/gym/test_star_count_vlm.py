"""Star-count VLM recipe, collation, and metric tests."""

from __future__ import annotations

import json
import zipfile

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
