"""Deterministic star-count environment and verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from bashgym.environments.star_count import (
    STAR_COUNT_COLORS,
    canonical_star_count_answer,
    create_star_count_archive,
    generate_star_count_dataset,
    score_star_count_prediction,
    star_count_environment_spec,
)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_star_count_verifier_separates_count_accuracy_from_formatting():
    expected = {"red": 2, "blue": 1, "green": 0, "yellow": 3}
    canonical = canonical_star_count_answer(expected)

    exact = score_star_count_prediction(canonical, expected)
    assert exact.exact is True
    assert exact.count_accuracy == 1.0
    assert exact.format_accuracy == 1.0

    flexible = score_star_count_prediction("blue: 1; yellow: 3; red: 2; green: 0", expected)
    assert flexible.exact is True
    assert flexible.count_accuracy == 1.0
    assert flexible.format_accuracy == 0.0

    near_miss = score_star_count_prediction("red=2, blue=1, green=1, yellow=3", expected)
    assert near_miss.exact is False
    assert near_miss.count_accuracy == 0.75
    assert near_miss.format_accuracy == 1.0

    invalid = score_star_count_prediction("there are several stars", expected)
    assert invalid.exact is False
    assert invalid.count_accuracy == 0.0
    assert invalid.format_accuracy == 0.0


def test_star_count_dataset_is_reproducible_and_secret_free(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {"train_size": 4, "validation_size": 2, "heldout_size": 3, "seed": 17}

    manifest_a = generate_star_count_dataset(first, **kwargs)
    manifest_b = generate_star_count_dataset(second, **kwargs)

    assert manifest_a == manifest_b
    assert manifest_a["split_sizes"] == {"train": 4, "validation": 2, "heldout": 3}
    assert manifest_a["colors"] == list(STAR_COUNT_COLORS)
    assert len(manifest_a["dataset_digest"]) == 64

    all_records: list[dict] = []
    for split in ("train", "validation", "heldout"):
        first_jsonl = first / f"{split}.jsonl"
        second_jsonl = second / f"{split}.jsonl"
        assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
        records = _records(first_jsonl)
        all_records.extend(records)
        for record in records:
            assert not Path(record["image"]).is_absolute()
            assert ".." not in Path(record["image"]).parts
            assert record["answer"] == canonical_star_count_answer(record["counts"])
            with Image.open(first / record["image"]) as image:
                assert image.mode == "RGB"
                assert image.width != 0 and image.height != 0
            assert (first / record["image"]).read_bytes() == (second / record["image"]).read_bytes()

    assert len({record["example_id"] for record in all_records}) == 9
    serialized = json.dumps(manifest_a, sort_keys=True)
    assert str(tmp_path) not in serialized

    archive_a = create_star_count_archive(first, tmp_path / "first.zip")
    archive_b = create_star_count_archive(second, tmp_path / "second.zip")
    assert archive_a == archive_b
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()


def test_star_count_environment_declares_exact_and_partial_rewards():
    environment = star_count_environment_spec()

    assert environment.validation_errors() == []
    assert environment.domain == "vision_language"
    assert environment.verifier.reward_type == "components"
    assert [item.name for item in environment.verifier.reward_components] == [
        "count_accuracy",
        "format_accuracy",
    ]
    assert (
        environment.verifier.combine_reward_components(
            {"count_accuracy": 1.0, "format_accuracy": 1.0}
        )
        == 1.0
    )
