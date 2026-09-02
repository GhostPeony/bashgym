"""Installation-owned Data Designer runner for durable AutoResearch DATA_BUILD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bashgym.campaigns.autoresearch_dataset import (
    AUTORESEARCH_DATASET_RECEIPT_FILENAME,
    AutoResearchDatasetFile,
    AutoResearchDatasetGeneration,
    AutoResearchDatasetQuality,
    AutoResearchDatasetReceipt,
)
from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.data_designer_recipe import (
    AutoResearchDataDesignRecipe,
    DataDesignerRunnerContract,
)
from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

_MAX_FINGERPRINT_BYTES = 16 * 1024 * 1024


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"dataset row contains a non-JSON value: {type(value).__name__}")


def _canonical_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    encoded = json.dumps(
        dict(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return json.loads(encoded), encoded


def canonical_row_digest(row: Mapping[str, Any]) -> str:
    """Return the content identity used for deduplication and leakage filtering."""

    _normalized, encoded = _canonical_row(row)
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_config(config: PipelineConfig) -> dict[str, Any]:
    """Project the effective, secret-free settings that can change generated rows."""

    return {
        "pipeline": config.pipeline,
        "provider": config.provider,
        "models": {
            "text": config.text_model,
            "code": config.code_model,
            "judge": config.judge_model,
        },
        "num_records": config.num_records,
        "buffer_size": config.buffer_size,
        "max_parallel_requests": config.max_parallel_requests,
        "experiment_brief": config.experiment_brief,
        "train_val_split": config.train_val_split,
        "temperatures": {
            "text": config.temperature_text,
            "code": config.temperature_code,
            "judge": config.temperature_judge,
        },
        "seed_source": config.seed_source,
        "tools": {
            "enabled": config.enable_tools,
            "alias": config.mcp_tool_alias,
            "backend": config.mcp_backend,
            "max_turns": config.mcp_max_tool_turns,
            "timeout_seconds": config.mcp_tool_timeout_sec,
        },
    }


def _generator_implementation_digest(pipeline_factory: Callable[[PipelineConfig], Any]) -> str:
    pipeline_source = inspect.getsourcefile(pipeline_factory)
    if not pipeline_source:
        raise RuntimeError("data_designer_runner_implementation_unverifiable")
    try:
        data_designer_version = importlib.metadata.version("data-designer")
    except importlib.metadata.PackageNotFoundError:
        data_designer_version = "not-installed"
    return canonical_hash(
        {
            "runner_sha256": _sha256_file(Path(__file__)),
            "pipeline_sha256": _sha256_file(Path(pipeline_source)),
            "data_designer_version": data_designer_version,
        }
    )


def _verify_file(path: Path, expected_sha256: str, *, maximum_bytes: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("data_designer_runner_input_missing")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise RuntimeError("data_designer_runner_input_too_large")
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("data_designer_runner_input_digest_mismatch")


def _fingerprints(contract: DataDesignerRunnerContract) -> set[str]:
    path = contract.protected_fingerprints_path
    _verify_file(
        path,
        contract.protected_fingerprints_sha256,
        maximum_bytes=_MAX_FINGERPRINT_BYTES,
    )
    values = {
        line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()
    }
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise RuntimeError("data_designer_runner_fingerprint_invalid")
    return values


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            normalized, encoded = _canonical_row(row)
            del normalized
            handle.write(encoded.decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _dataset_file(
    path: Path, *, run_directory: Path, split: str, rows: int
) -> AutoResearchDatasetFile:
    return AutoResearchDatasetFile(
        path=path.relative_to(run_directory).as_posix(),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        split=split,
        row_count=rows,
    )


def run_contract(
    contract: DataDesignerRunnerContract,
    recipe: AutoResearchDataDesignRecipe,
    run_directory: Path,
    *,
    pipeline_factory: Callable[[PipelineConfig], Any] = DataDesignerPipeline,
) -> AutoResearchDatasetReceipt:
    """Generate, validate, split, and describe one compute-resident dataset."""

    contract.validate_recipe(recipe)
    policy = contract.policy(recipe.pipeline)
    _verify_file(contract.parent_dataset_path, contract.parent_dataset_sha256)
    protected = _fingerprints(contract)
    config = PipelineConfig(
        pipeline=recipe.pipeline,
        provider=contract.provider_name,
        provider_endpoint=contract.provider_endpoint,
        text_model=contract.text_model,
        code_model=contract.code_model,
        judge_model=contract.judge_model,
        num_records=recipe.target_rows,
        train_val_split=recipe.train_fraction,
        experiment_brief=recipe.generation_brief,
    )
    config.provider_api_key = None
    frame = pipeline_factory(config).from_dataset(
        str(contract.parent_dataset_path),
        num_records=recipe.target_rows,
    )
    raw_records = frame.to_dict(orient="records")
    if not isinstance(raw_records, list) or len(raw_records) != recipe.target_rows:
        raise RuntimeError("data_designer_runner_row_count_mismatch")

    verified_rows = 0
    verification_failed_rows = 0
    duplicate_rows_removed = 0
    contamination_rows_removed = 0
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            verification_failed_rows += 1
            continue
        row, encoded = _canonical_row(raw)
        if any(column not in row or row[column] is None for column in policy.required_columns):
            verification_failed_rows += 1
            continue
        if any(row.get(column) not in allowed for column, allowed in policy.allowed_labels.items()):
            verification_failed_rows += 1
            continue
        verified_rows += 1
        digest = hashlib.sha256(encoded).hexdigest()
        if digest in seen:
            duplicate_rows_removed += 1
            continue
        seen.add(digest)
        if digest in protected:
            contamination_rows_removed += 1
            continue
        accepted.append(row)

    if len(accepted) < 2:
        raise RuntimeError("data_designer_runner_insufficient_accepted_rows")
    random.Random(recipe.seed).shuffle(accepted)
    split_index = min(
        max(int(len(accepted) * recipe.train_fraction), 1),
        len(accepted) - 1,
    )
    train_rows = accepted[:split_index]
    validation_rows = accepted[split_index:]

    run_directory = run_directory.resolve()
    dataset_directory = run_directory / "dataset"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    train_path = dataset_directory / "train.jsonl"
    validation_path = dataset_directory / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    provider_config_digest = canonical_hash(
        {
            "provider_name": contract.provider_name,
            "provider_endpoint": contract.provider_endpoint,
            "models": {
                "text": contract.text_model,
                "code": contract.code_model,
                "judge": contract.judge_model,
            },
        }
    )
    generation_config_digest = canonical_hash(_generation_config(config))
    generator_implementation_digest = _generator_implementation_digest(pipeline_factory)
    receipt = AutoResearchDatasetReceipt(
        files=tuple(
            sorted(
                (
                    _dataset_file(
                        train_path,
                        run_directory=run_directory,
                        split="train",
                        rows=len(train_rows),
                    ),
                    _dataset_file(
                        validation_path,
                        run_directory=run_directory,
                        split="validation",
                        rows=len(validation_rows),
                    ),
                ),
                key=lambda item: item.path,
            )
        ),
        generator=AutoResearchDatasetGeneration(
            parent_dataset_version_id=contract.parent_dataset_version_id,
            hypothesis=recipe.hypothesis,
            generation_brief=recipe.generation_brief,
            pipeline=recipe.pipeline,
            target_rows=recipe.target_rows,
            train_fraction=recipe.train_fraction,
            recipe_digest=canonical_hash(recipe.model_dump(mode="json")),
            provider_config_digest=provider_config_digest,
            generation_config_digest=generation_config_digest,
            generator_implementation_digest=generator_implementation_digest,
            models={
                "text": contract.text_model,
                "code": contract.code_model,
                "judge": contract.judge_model,
            },
            split_seed=recipe.seed,
        ),
        quality=AutoResearchDatasetQuality(
            generated_rows=len(raw_records),
            accepted_rows=len(accepted),
            deterministic_verified_rows=verified_rows,
            verification_failed_rows=verification_failed_rows,
            duplicate_rows_removed=duplicate_rows_removed,
            contamination_rows_removed=contamination_rows_removed,
            verifier_digest=contract.verifier_digest,
        ),
    )
    receipt_path = run_directory / AUTORESEARCH_DATASET_RECEIPT_FILENAME
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--recipe-json", required=True)
    args = parser.parse_args(argv)
    contract = DataDesignerRunnerContract.model_validate_json(args.contract_json)
    recipe = AutoResearchDataDesignRecipe.model_validate_json(args.recipe_json)
    run_contract(contract, recipe, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["canonical_row_digest", "main", "run_contract"]
