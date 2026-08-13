"""Deterministic visual star-count fixtures and verifier rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw

from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentAxis,
    EnvironmentSpec,
    RewardComponentSpec,
    RolloutSpec,
    VerifierSpec,
)

STAR_COUNT_COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
_RGB = {
    "red": (214, 48, 49),
    "blue": (42, 103, 210),
    "green": (42, 154, 84),
    "yellow": (232, 181, 25),
}
_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (224, 224),
    (336, 224),
    (224, 336),
    (336, 336),
    (448, 336),
)
STAR_COUNT_PROMPT = (
    "Count the red, blue, green, and yellow stars in the image. "
    "Reply exactly as: red=N, blue=N, green=N, yellow=N"
)
_PAIR_PATTERN = re.compile(r"\b(red|blue|green|yellow)\s*[:=]\s*(\d+)\b", re.I)


@dataclass(frozen=True)
class StarCountScore:
    exact: bool
    count_accuracy: float
    format_accuracy: float
    predicted_counts: dict[str, int] | None

    def reward_components(self) -> dict[str, float]:
        return {
            "count_accuracy": self.count_accuracy,
            "format_accuracy": self.format_accuracy,
        }


def _validated_counts(counts: dict[str, Any]) -> dict[str, int]:
    if set(counts) != set(STAR_COUNT_COLORS):
        raise ValueError("star counts must contain exactly the declared colors")
    normalized = {color: int(counts[color]) for color in STAR_COUNT_COLORS}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("star counts cannot be negative")
    return normalized


def canonical_star_count_answer(counts: dict[str, Any]) -> str:
    normalized = _validated_counts(counts)
    return ", ".join(f"{color}={normalized[color]}" for color in STAR_COUNT_COLORS)


def parse_star_count_prediction(prediction: str) -> dict[str, int] | None:
    matches = _PAIR_PATTERN.findall(prediction)
    if len(matches) != len(STAR_COUNT_COLORS):
        return None
    parsed: dict[str, int] = {}
    for raw_color, raw_value in matches:
        color = raw_color.casefold()
        if color in parsed:
            return None
        parsed[color] = int(raw_value)
    return parsed if set(parsed) == set(STAR_COUNT_COLORS) else None


def score_star_count_prediction(prediction: str, expected_counts: dict[str, Any]) -> StarCountScore:
    expected = _validated_counts(expected_counts)
    predicted = parse_star_count_prediction(prediction)
    if predicted is None:
        return StarCountScore(False, 0.0, 0.0, None)
    correct = sum(predicted[color] == expected[color] for color in STAR_COUNT_COLORS)
    return StarCountScore(
        exact=correct == len(STAR_COUNT_COLORS),
        count_accuracy=correct / len(STAR_COUNT_COLORS),
        format_accuracy=float(prediction.strip() == canonical_star_count_answer(predicted)),
        predicted_counts=predicted,
    )


def star_count_environment_spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        id="star-count-v1",
        instruction=STAR_COUNT_PROMPT,
        source="bashgym",
        domain="vision_language",
        skills=["visual_counting", "structured_output"],
        axes=[
            EnvironmentAxis("canvas_resolution", "variable", "synthetic"),
            EnvironmentAxis("star_color", "four_color", "synthetic"),
            EnvironmentAxis("star_density", "variable", "synthetic"),
        ],
        verifier=VerifierSpec(
            kind="star_count_exact",
            command=(
                "python -m bashgym.environments.star_count verify "
                "--expected expected.json --prediction prediction.txt"
            ),
            path=None,
            reward_type="components",
            success_threshold=1.0,
            reward_components=[
                RewardComponentSpec(
                    "count_accuracy", 0.95, "Fraction of colors counted correctly."
                ),
                RewardComponentSpec("format_accuracy", 0.05, "Exact canonical response format."),
            ],
        ),
        build=BuildSpec(dockerfile=None, network_disabled=True),
        rollout=RolloutSpec(
            harness="vision-language-generation",
            max_steps=1,
            max_tool_calls=1,
            timeout_sec=300,
            max_prompt_tokens=256,
            max_response_tokens=64,
        ),
        metadata={
            "schema_version": "star_count_environment.v1",
            "primary_metric": "exact_count_accuracy",
            "colors": list(STAR_COUNT_COLORS),
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _star_points(
    center_x: float, center_y: float, outer_radius: float, rotation: float
) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        radius = outer_radius if index % 2 == 0 else outer_radius * 0.43
        angle = rotation - math.pi / 2 + index * math.pi / 5
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return points


def _sample_counts(rng: random.Random) -> dict[str, int]:
    while True:
        counts = {color: rng.randint(0, 5) for color in STAR_COUNT_COLORS}
        if 3 <= sum(counts.values()) <= 18:
            return counts


def _render_example(path: Path, counts: dict[str, int], rng: random.Random) -> tuple[int, int]:
    width, height = rng.choice(_RESOLUTIONS)
    image = Image.new("RGB", (width, height), (248, 247, 243))
    draw = ImageDraw.Draw(image)
    spacing = 36
    positions = [
        (x, y) for y in range(24, height - 20, spacing) for x in range(24, width - 20, spacing)
    ]
    rng.shuffle(positions)
    cursor = 0
    for color in STAR_COUNT_COLORS:
        for _ in range(counts[color]):
            x, y = positions[cursor]
            cursor += 1
            radius = rng.uniform(10.0, 15.0)
            jitter_x = rng.uniform(-4.0, 4.0)
            jitter_y = rng.uniform(-4.0, 4.0)
            rotation = rng.uniform(0.0, math.tau)
            draw.polygon(
                _star_points(x + jitter_x, y + jitter_y, radius, rotation),
                fill=_RGB[color],
                outline=(45, 45, 45),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return width, height


def _write_split(root: Path, split: str, size: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"star-count-v1:{seed}:{split}")
    records: list[dict[str, Any]] = []
    for index in range(size):
        example_id = f"{split}-{index:06d}"
        relative_image = PurePosixPath("images") / split / f"{example_id}.png"
        counts = _sample_counts(rng)
        width, height = _render_example(root / Path(relative_image), counts, rng)
        records.append(
            {
                "schema_version": "star_count_example.v1",
                "example_id": example_id,
                "split": split,
                "image": str(relative_image),
                "width": width,
                "height": height,
                "prompt": STAR_COUNT_PROMPT,
                "answer": canonical_star_count_answer(counts),
                "counts": counts,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": STAR_COUNT_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": canonical_star_count_answer(counts)}],
                    },
                ],
            }
        )
    jsonl = root / f"{split}.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return records


def generate_star_count_dataset(
    output_dir: str | Path,
    *,
    train_size: int = 512,
    validation_size: int = 64,
    heldout_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate disjoint deterministic fixtures with a content-addressed manifest."""

    sizes = {"train": train_size, "validation": validation_size, "heldout": heldout_size}
    if any(not isinstance(value, int) or value <= 0 for value in sizes.values()):
        raise ValueError("star-count split sizes must be positive integers")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"star-count output directory is not empty: {root.name}")
    root.mkdir(parents=True, exist_ok=True)

    for split, size in sizes.items():
        _write_split(root, split, size, seed=seed)

    file_records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_records.append(
            {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    identity = {
        "schema_version": "star_count_dataset.v1",
        "generator_version": "star-count-v1",
        "seed": seed,
        "colors": list(STAR_COUNT_COLORS),
        "split_sizes": sizes,
        "files": file_records,
    }
    dataset_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {**identity, "dataset_digest": dataset_digest}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def create_star_count_archive(dataset_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a deterministic, single-file transport artifact for private compute."""

    root = Path(dataset_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"star-count archive already exists: {destination.name}")
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    return {
        "schema_version": "star_count_archive.v1",
        "dataset_digest": manifest["dataset_digest"],
        "archive_sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BashGym star-count environment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--train-size", type=int, default=512)
    generate.add_argument("--validation-size", type=int, default=64)
    generate.add_argument("--heldout-size", type=int, default=64)
    generate.add_argument("--seed", type=int, default=42)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--expected", required=True)
    verify.add_argument("--prediction", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_star_count_dataset(
            args.output,
            train_size=args.train_size,
            validation_size=args.validation_size,
            heldout_size=args.heldout_size,
            seed=args.seed,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    score = score_star_count_prediction(Path(args.prediction).read_text(encoding="utf-8"), expected)
    components = score.reward_components()
    print(
        json.dumps(
            {
                "reward_components": components,
                "total_reward": 0.95 * components["count_accuracy"]
                + 0.05 * components["format_accuracy"],
                "exact": score.exact,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "STAR_COUNT_COLORS",
    "STAR_COUNT_PROMPT",
    "StarCountScore",
    "canonical_star_count_answer",
    "create_star_count_archive",
    "generate_star_count_dataset",
    "parse_star_count_prediction",
    "score_star_count_prediction",
    "star_count_environment_spec",
]
