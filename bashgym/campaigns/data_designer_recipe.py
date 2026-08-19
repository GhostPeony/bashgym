"""Bounded experiment inputs for an installation-owned Data Designer runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    FrozenContractModel,
    HexDigest,
    Identifier,
)

AUTORESEARCH_DATA_DESIGN_RECIPE_SCHEMA = "bashgym.autoresearch_data_design_recipe.v1"
DATA_DESIGNER_RUNNER_CONTRACT_SCHEMA = "bashgym.data_designer_runner_contract.v1"


class AutoResearchDataDesignRecipe(FrozenContractModel):
    """One agent-authored data hypothesis inside the installed execution envelope."""

    schema_version: Literal["bashgym.autoresearch_data_design_recipe.v1"] = (
        AUTORESEARCH_DATA_DESIGN_RECIPE_SCHEMA
    )
    runtime: dict[str, str] = Field(
        default_factory=lambda: {"executor_kind": "registered_training"}
    )
    hypothesis: str = Field(min_length=10, max_length=2000)
    pipeline: Identifier
    generation_brief: str = Field(min_length=10, max_length=4000)
    target_rows: int = Field(ge=2, le=1_000_000)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("runtime")
    @classmethod
    def registered_runtime_only(cls, value: dict[str, str]) -> dict[str, str]:
        if value != {"executor_kind": "registered_training"}:
            raise ValueError("data design requires the registered training runtime")
        return value

    def script_args(self) -> tuple[str, ...]:
        """Render the complete actor-controlled runner ABI."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return ("--recipe-json", payload)


class DataDesignerPipelinePolicy(FrozenContractModel):
    """Installation-owned limits for one executable Data Designer pipeline."""

    pipeline: Identifier
    min_rows: int = Field(ge=2, le=1_000_000)
    max_rows: int = Field(ge=2, le=1_000_000)
    required_columns: tuple[Identifier, ...] = Field(min_length=1, max_length=100)
    allowed_labels: dict[Identifier, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonical_fields(self) -> DataDesignerPipelinePolicy:
        if self.min_rows > self.max_rows:
            raise ValueError("pipeline row bounds are inconsistent")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required columns must be unique")
        if not set(self.allowed_labels).issubset(self.required_columns):
            raise ValueError("allowed label fields must also be required columns")
        if any(
            not labels or len(set(labels)) != len(labels) for labels in self.allowed_labels.values()
        ):
            raise ValueError("allowed labels must be non-empty and unique")
        return self


class DataDesignerRunnerContract(FrozenContractModel):
    """Installation-owned data, generator, validation, and budget bounds."""

    schema_version: Literal["bashgym.data_designer_runner_contract.v1"] = (
        DATA_DESIGNER_RUNNER_CONTRACT_SCHEMA
    )
    parent_dataset_version_id: Identifier
    parent_dataset_path: Path
    parent_dataset_sha256: HexDigest
    protected_fingerprints_path: Path
    protected_fingerprints_sha256: HexDigest
    provider_name: Identifier
    provider_endpoint: str = Field(min_length=1, max_length=2048)
    text_model: str = Field(min_length=1, max_length=500)
    code_model: str = Field(min_length=1, max_length=500)
    judge_model: str = Field(min_length=1, max_length=500)
    verifier_digest: HexDigest
    pipeline_policies: tuple[DataDesignerPipelinePolicy, ...] = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator("parent_dataset_path", "protected_fingerprints_path")
    @classmethod
    def absolute_paths_only(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("runner data paths must be absolute")
        return value

    @field_validator("provider_endpoint")
    @classmethod
    def http_endpoint_only(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("provider endpoint must be an HTTP(S) origin or API path")
        return value.rstrip("/")

    @model_validator(mode="after")
    def bounded_unique_pipelines(self) -> DataDesignerRunnerContract:
        identifiers = tuple(item.pipeline for item in self.pipeline_policies)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("data design pipeline policies must be unique")
        return self

    def policy(self, pipeline: str) -> DataDesignerPipelinePolicy:
        """Return the execution limits for one supported pipeline."""

        for item in self.pipeline_policies:
            if item.pipeline == pipeline:
                return item
        raise KeyError(pipeline)

    def validate_recipe(
        self,
        recipe: AutoResearchDataDesignRecipe,
    ) -> AutoResearchDataDesignRecipe:
        """Validate an agent-authored design without preapproving its hypothesis."""

        policy = self.policy(recipe.pipeline)
        if not policy.min_rows <= recipe.target_rows <= policy.max_rows:
            raise ValueError("agent-authored data design row count is outside policy")
        return recipe


__all__ = [
    "AUTORESEARCH_DATA_DESIGN_RECIPE_SCHEMA",
    "DATA_DESIGNER_RUNNER_CONTRACT_SCHEMA",
    "AutoResearchDataDesignRecipe",
    "DataDesignerPipelinePolicy",
    "DataDesignerRunnerContract",
]
