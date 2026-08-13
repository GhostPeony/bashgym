"""Small agent-authored recipe for an installed TMax training runner."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from bashgym.campaigns.contracts import FrozenContractModel

TMAX_COMPOSITE_TRAINING_RECIPE_SCHEMA = "bashgym.tmax_composite_training_recipe.v1"


class TMaxCompositeTrainingRecipe(FrozenContractModel):
    """The bounded experiment variables; installation owns paths and executables."""

    schema_version: Literal["bashgym.tmax_composite_training_recipe.v1"] = (
        TMAX_COMPOSITE_TRAINING_RECIPE_SCHEMA
    )
    runtime: dict[str, str] = Field(
        default_factory=lambda: {"executor_kind": "registered_training"}
    )
    algorithm: Literal["grpo"] = "grpo"
    sft_enabled: Literal[False] = False
    learning_rate: float = Field(default=2e-5, gt=0, le=1.0)
    max_steps: int = Field(default=100, ge=1, le=100_000)
    group_size: int = Field(default=8, ge=2, le=128)
    temperature: float = Field(default=0.8, gt=0, le=5.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("runtime")
    @classmethod
    def registered_runtime_only(cls, value: dict[str, str]) -> dict[str, str]:
        if value != {"executor_kind": "registered_training"}:
            raise ValueError("TMax recipes require the registered training runtime")
        return value

    @staticmethod
    def _number(value: float) -> str:
        return format(value, ".12g")

    def script_args(self) -> tuple[str, ...]:
        """Render one canonical ABI for the installation-owned runner."""

        return (
            "--algorithm",
            self.algorithm,
            "--sft-enabled",
            "true" if self.sft_enabled else "false",
            "--learning-rate",
            self._number(self.learning_rate),
            "--max-steps",
            str(self.max_steps),
            "--group-size",
            str(self.group_size),
            "--temperature",
            self._number(self.temperature),
            "--seed",
            str(self.seed),
        )


__all__ = [
    "TMAX_COMPOSITE_TRAINING_RECIPE_SCHEMA",
    "TMaxCompositeTrainingRecipe",
]
