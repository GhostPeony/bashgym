"""AutoResearch search space for public-source data recipe proposals."""

from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

from bashgym.gym.autoresearch import SearchSpace
from bashgym.sources import SourceCard, SourceUse
from bashgym.sources.catalog import TRAINING_USES, supports_goal

DATA_RECIPE_SCHEMA_VERSION = "bashgym.data_recipe_proposal.v1"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_weights(values: dict[str, Any], keys: list[str]) -> dict[str, float]:
    normalized = {key: max(0.0, float(values.get(key, 1.0))) for key in keys}
    total = sum(normalized.values())
    if total <= 0:
        return {key: 0.0 for key in keys}
    rounded = {key: round(value / total, 4) for key, value in normalized.items()}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift and rounded:
        largest = max(rounded, key=rounded.get)
        rounded[largest] = round(max(0.0, rounded[largest] + drift), 4)
    return rounded


class DataRecipeSearchSpace(SearchSpace):
    """Mutate and evaluate data-source mix recipes.

    A genome describes which public/internal sources to use, how heavily to
    weight domains, and which quality/decontamination controls to apply. Lower
    metric is better to match ``AutoResearcher``.
    """

    def __init__(
        self,
        sources: list[SourceCard],
        *,
        goal: SourceUse | str = SourceUse.SFT,
        mutation_rate: float = 0.35,
        mutation_scale: float = 0.2,
    ) -> None:
        if not sources:
            raise ValueError("DataRecipeSearchSpace requires at least one source")
        self.sources = sources
        self.goal = SourceUse(goal)
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.source_ids = [source.id for source in sources]
        self.domains = sorted({source.domain for source in sources if source.domain})

    @staticmethod
    def create_default_genome(
        source_ids: list[str],
        domains: list[str],
        *,
        sample_size: int = 1000,
        quality_threshold: float = 0.7,
        synthetic_multiplier: float = 1.0,
        decontam_jaccard_threshold: float = 0.7,
        cost_budget_usd: float = 25.0,
        eval_target: str = "heldout_pass@k",
        seed: int = 0,
    ) -> dict[str, Any]:
        return {
            "sample_size": sample_size,
            "seed": seed,
            "quality_threshold": quality_threshold,
            "synthetic_multiplier": synthetic_multiplier,
            "decontam_jaccard_threshold": decontam_jaccard_threshold,
            "cost_budget_usd": cost_budget_usd,
            "eval_target": eval_target,
            "source_weights": _normalize_weights({}, source_ids),
            "domain_weights": _normalize_weights({}, domains),
        }

    def mutate(self, genome: dict[str, Any]) -> dict[str, Any]:
        mutated = copy.deepcopy(self._normalize_genome(genome))
        if random.random() < self.mutation_rate:
            delta = random.gauss(0, max(1.0, mutated["sample_size"] * self.mutation_scale))
            mutated["sample_size"] = int(_clamp(mutated["sample_size"] + delta, 1, 1_000_000))
        if random.random() < self.mutation_rate:
            mutated["quality_threshold"] = round(
                _clamp(mutated["quality_threshold"] + random.gauss(0, 0.1), 0.0, 1.0),
                4,
            )
        if random.random() < self.mutation_rate:
            mutated["synthetic_multiplier"] = round(
                _clamp(mutated["synthetic_multiplier"] + random.gauss(0, 0.5), 0.0, 10.0),
                4,
            )
        if random.random() < self.mutation_rate:
            mutated["decontam_jaccard_threshold"] = round(
                _clamp(
                    mutated["decontam_jaccard_threshold"] + random.gauss(0, 0.1),
                    0.0,
                    1.0,
                ),
                4,
            )
        if random.random() < self.mutation_rate:
            mutated["cost_budget_usd"] = round(
                _clamp(mutated["cost_budget_usd"] + random.gauss(0, 10.0), 0.0, 10_000.0),
                2,
            )
        if random.random() < self.mutation_rate:
            mutated["seed"] = int(mutated["seed"]) + random.randint(1, 997)
        self._mutate_weight_map(mutated["source_weights"])
        self._mutate_weight_map(mutated["domain_weights"])
        mutated["source_weights"] = _normalize_weights(mutated["source_weights"], self.source_ids)
        mutated["domain_weights"] = _normalize_weights(mutated["domain_weights"], self.domains)
        return mutated

    def evaluate(
        self,
        genome: dict[str, Any],
        experiment_number: int,
        total_experiments: int,
    ) -> float:
        normalized = self._normalize_genome(genome)
        selected = self.selected_sources(normalized)
        if not selected:
            return 10.0

        unsupported = [source for source in selected if not supports_goal(source, self.goal)]
        eval_only_training = [
            source for source in selected if self.goal in TRAINING_USES and source.eval_only
        ]
        risk_count = sum(len(source.known_risks) for source in selected)
        domains = [source.domain for source in selected if source.domain]
        domain_counts = Counter(domains)
        domain_loss = 1.0
        if domain_counts:
            desired = normalized["domain_weights"]
            domain_loss = sum(
                abs((domain_counts.get(domain, 0) / len(selected)) - desired.get(domain, 0.0))
                for domain in self.domains
            ) / max(1, len(self.domains))

        quality_loss = abs(normalized["quality_threshold"] - 0.75)
        synthetic_loss = max(0.0, normalized["synthetic_multiplier"] - 3.0) * 0.05
        decontam_loss = max(0.0, normalized["decontam_jaccard_threshold"] - 0.85)
        unsupported_loss = len(unsupported) / len(selected)
        eval_only_loss = len(eval_only_training) / len(selected)
        risk_loss = min(1.0, risk_count / max(1, len(selected) * 3))
        return float(
            (0.25 * domain_loss)
            + (0.2 * unsupported_loss)
            + (0.2 * eval_only_loss)
            + (0.15 * risk_loss)
            + (0.1 * quality_loss)
            + (0.05 * synthetic_loss)
            + (0.05 * decontam_loss)
        )

    def get_config_snapshot(self, genome: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_genome(genome)

    def selected_sources(self, genome: dict[str, Any]) -> list[SourceCard]:
        normalized = self._normalize_genome(genome)
        weights = normalized["source_weights"]
        return [source for source in self.sources if weights.get(source.id, 0.0) > 0.0]

    def proposal_for(self, genome: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_genome(genome)
        selected = self.selected_sources(normalized)
        return {
            "schema_version": DATA_RECIPE_SCHEMA_VERSION,
            "goal": self.goal.value,
            "genome": normalized,
            "sources": [
                {
                    "id": source.id,
                    "domain": source.domain,
                    "weight": normalized["source_weights"].get(source.id, 0.0),
                    "adapter": source.adapter,
                    "artifact_types": [artifact.value for artifact in source.artifact_types],
                    "training_eligible": source.training_eligible,
                    "eval_only": source.eval_only,
                }
                for source in selected
            ],
            "guardrails": {
                "quality_threshold": normalized["quality_threshold"],
                "decontam_jaccard_threshold": normalized["decontam_jaccard_threshold"],
                "eval_only_sources_block_training": True,
            },
            "data_designer": {
                "pipeline": "from_source",
                "synthetic_multiplier": normalized["synthetic_multiplier"],
                "sample_size": normalized["sample_size"],
            },
        }

    def _normalize_genome(self, genome: dict[str, Any]) -> dict[str, Any]:
        source_weights = _normalize_weights(genome.get("source_weights", {}), self.source_ids)
        domain_weights = _normalize_weights(genome.get("domain_weights", {}), self.domains)
        return {
            "sample_size": int(_clamp(float(genome.get("sample_size", 1000)), 1, 1_000_000)),
            "seed": int(genome.get("seed", 0)),
            "quality_threshold": round(
                _clamp(float(genome.get("quality_threshold", 0.7)), 0.0, 1.0),
                4,
            ),
            "synthetic_multiplier": round(
                _clamp(float(genome.get("synthetic_multiplier", 1.0)), 0.0, 10.0),
                4,
            ),
            "decontam_jaccard_threshold": round(
                _clamp(float(genome.get("decontam_jaccard_threshold", 0.7)), 0.0, 1.0),
                4,
            ),
            "cost_budget_usd": round(
                _clamp(float(genome.get("cost_budget_usd", 25.0)), 0.0, 10_000.0),
                2,
            ),
            "eval_target": str(genome.get("eval_target", "heldout_pass@k")),
            "source_weights": source_weights,
            "domain_weights": domain_weights,
        }

    def _mutate_weight_map(self, weights: dict[str, float]) -> None:
        for key in list(weights):
            if random.random() < self.mutation_rate:
                weights[key] = max(
                    0.0,
                    weights[key] + random.gauss(0, max(0.01, self.mutation_scale)),
                )
