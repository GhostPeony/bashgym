"""AutoResearch search space for executable environment recipe proposals."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.metrics import balance_score, summarize_environment_mix
from bashgym.gym.autoresearch import SearchSpace

RECIPE_AXES = (
    "domain",
    "skill",
    "task_complexity",
    "command_complexity",
    "language",
    "fixture_kind",
)

PROPOSAL_SCHEMA_VERSION = "bashgym.environment_recipe_proposal.v1"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _stable_jitter(seed: int, env_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{env_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _metadata_rate(env: EnvironmentSpec, key: str) -> float | None:
    candidates = (key, key.replace("_", "@"), key.replace("@", "_"))
    for candidate in candidates:
        value = env.metadata.get(candidate)
        if value is None and isinstance(env.metadata.get("raw_record"), dict):
            value = env.metadata["raw_record"].get(candidate)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        return numeric / 100 if numeric > 1.0 else numeric
    return None


def _axis_values(env: EnvironmentSpec, axis: str) -> list[str]:
    if axis == "domain":
        return [env.domain] if env.domain else []

    if axis == "skill":
        return [skill for skill in env.skills if skill]

    values: list[str] = []
    value = env.axis_value(axis)
    if value:
        values.extend(part.strip() for part in str(value).split(",") if part.strip())

    if axis == "fixture_kind":
        values.extend(fixture.kind for fixture in env.fixtures if fixture.kind)

    return list(dict.fromkeys(values))


def _normalize_weight_map(
    incoming: dict[str, Any] | None,
    keys: list[str] | tuple[str, ...],
    *,
    default: float = 1.0,
) -> dict[str, float]:
    values = dict(incoming or {})
    return {key: round(float(_clamp(float(values.get(key, default)), 0.0, 5.0)), 4) for key in keys}


class EnvironmentRecipeSearchSpace(SearchSpace):
    """Mutate and evaluate TMax-style environment mix recipes.

    A genome describes how to sample a reproducible subset from a source pool:
    axis weights, verifier/fixture preferences, target learnability, sample size,
    and seed. Evaluation favors balanced, verifier-backed, moderately learnable
    mixes. Lower metric is better to match ``AutoResearcher``.
    """

    def __init__(
        self,
        environments: list[EnvironmentSpec],
        *,
        mutation_rate: float = 0.35,
        mutation_scale: float = 0.2,
    ) -> None:
        if not environments:
            raise ValueError("EnvironmentRecipeSearchSpace requires at least one environment")

        self.environments = environments
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.possible_domains = sorted({env.domain for env in environments if env.domain})
        self.possible_skills = sorted({skill for env in environments for skill in env.skills})
        self.possible_verifier_kinds = sorted({env.verifier.kind for env in environments})
        self.possible_fixture_kinds = sorted(
            {fixture.kind for env in environments for fixture in env.fixtures if fixture.kind}
        )
        self._value_counts = self._build_axis_counts()

    @staticmethod
    def create_default_genome(
        *,
        sample_size: int = 64,
        pass_at_1_target: float = 0.35,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Create a reproducible starting recipe genome."""
        return {
            "sample_size": sample_size,
            "seed": seed,
            "pass_at_1_target": pass_at_1_target,
            "axis_weights": {axis: 1.0 for axis in RECIPE_AXES},
            "verifier_kind_weights": {
                "exact_success": 1.0,
                "pytest": 1.0,
                "script": 1.0,
                "unit": 1.0,
            },
            "fixture_kind_weights": {
                "file": 1.0,
                "files": 1.0,
                "repo": 1.0,
                "service": 1.0,
                "archive": 1.0,
            },
        }

    def mutate(self, genome: dict[str, Any]) -> dict[str, Any]:
        """Create a mutated environment recipe genome."""
        mutated = copy.deepcopy(self._normalize_genome(genome))

        if random.random() < self.mutation_rate:
            delta = random.gauss(0, max(1.0, mutated["sample_size"] * self.mutation_scale))
            mutated["sample_size"] = int(
                _clamp(round(mutated["sample_size"] + delta), 1, len(self.environments))
            )

        if random.random() < self.mutation_rate:
            mutated["pass_at_1_target"] = round(
                _clamp(
                    mutated["pass_at_1_target"] + random.gauss(0, self.mutation_scale * 0.2),
                    0.0,
                    1.0,
                ),
                4,
            )

        if random.random() < self.mutation_rate:
            mutated["seed"] = int(mutated["seed"]) + random.randint(1, 997)

        self._mutate_weights(mutated["axis_weights"])
        self._mutate_weights(mutated["verifier_kind_weights"])
        self._mutate_weights(mutated["fixture_kind_weights"])
        return mutated

    def evaluate(
        self,
        genome: dict[str, Any],
        experiment_number: int,
        total_experiments: int,
    ) -> float:
        """Evaluate a recipe proposal. Lower is better."""
        normalized = self._normalize_genome(genome)
        selected = self.select_environments(normalized)
        if not selected:
            return 10.0

        report = summarize_environment_mix(
            selected,
            possible_domains=self.possible_domains,
            possible_skills=self.possible_skills,
        )

        axis_weights = normalized["axis_weights"]
        weight_total = sum(weight for weight in axis_weights.values() if weight > 0) or 1.0
        axis_loss = (
            sum(
                weight * (1.0 - _clamp(report.axis_balance.get(axis, 0.0), 0.0, 1.0))
                for axis, weight in axis_weights.items()
                if weight > 0
            )
            / weight_total
        )

        verifier_values = [env.verifier.kind for env in selected]
        verifier_loss = 1.0 - balance_score(verifier_values, self.possible_verifier_kinds)

        fixture_values = [
            value for env in selected for value in _axis_values(env, "fixture_kind") if value
        ]
        fixture_loss = (
            1.0 - balance_score(fixture_values, self.possible_fixture_kinds)
            if self.possible_fixture_kinds
            else 0.0
        )

        pass_at_1 = report.mean_pass_rates.get("pass@1")
        pass_loss = (
            abs(pass_at_1 - normalized["pass_at_1_target"]) if pass_at_1 is not None else 0.0
        )
        size_loss = 1.0 - (len(selected) / max(1, normalized["sample_size"]))

        return float(
            (0.55 * axis_loss)
            + (0.2 * pass_loss)
            + (0.15 * verifier_loss)
            + (0.05 * fixture_loss)
            + (0.05 * size_loss)
        )

    def get_config_snapshot(self, genome: dict[str, Any]) -> dict[str, Any]:
        """Extract a serializable recipe genome."""
        return self._normalize_genome(genome)

    def select_environments(self, genome: dict[str, Any]) -> list[EnvironmentSpec]:
        """Select a deterministic environment subset from a recipe genome."""
        normalized = self._normalize_genome(genome)
        sample_size = normalized["sample_size"]
        scored = [
            (self._environment_score(env, normalized), env.id, env) for env in self.environments
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [env for _, _, env in scored[:sample_size]]

    def proposal_for(
        self,
        genome: dict[str, Any],
        *,
        metric: float | None = None,
    ) -> dict[str, Any]:
        """Build an exportable environment recipe proposal."""
        normalized = self._normalize_genome(genome)
        selected = self.select_environments(normalized)
        metric_value = self.evaluate(normalized, 0, 0) if metric is None else float(metric)
        mix_report = summarize_environment_mix(
            selected,
            possible_domains=self.possible_domains,
            possible_skills=self.possible_skills,
        )
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "source_count": len(self.environments),
            "selected_count": len(selected),
            "selected_environment_ids": [env.id for env in selected],
            "recipe": normalized,
            "metric": round(metric_value, 6),
            "mix_report": mix_report.to_dict(),
            "selected_environments": [
                {
                    "id": env.id,
                    "domain": env.domain,
                    "skills": env.skills,
                    "verifier_kind": env.verifier.kind,
                    "fixture_kinds": _axis_values(env, "fixture_kind"),
                    "pass@1": _metadata_rate(env, "pass@1"),
                }
                for env in selected
            ],
        }

    def write_proposal(
        self,
        genome: dict[str, Any],
        output_path: str | Path,
        *,
        metric: float | None = None,
    ) -> Path:
        """Write a proposal JSON file and return the resolved path."""
        proposal = self.proposal_for(genome, metric=metric)
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _normalize_genome(self, genome: dict[str, Any]) -> dict[str, Any]:
        defaults = self.create_default_genome(
            sample_size=min(64, len(self.environments)),
            pass_at_1_target=0.35,
            seed=0,
        )
        merged = copy.deepcopy(defaults)
        merged.update(genome or {})
        merged["sample_size"] = int(
            _clamp(int(merged.get("sample_size") or 1), 1, len(self.environments))
        )
        merged["seed"] = int(merged.get("seed") or 0)
        merged["pass_at_1_target"] = round(
            _clamp(float(merged.get("pass_at_1_target", 0.35)), 0.0, 1.0),
            4,
        )

        verifier_keys = sorted(
            set(defaults["verifier_kind_weights"]) | set(self.possible_verifier_kinds)
        )
        fixture_keys = sorted(
            set(defaults["fixture_kind_weights"]) | set(self.possible_fixture_kinds)
        )

        merged["axis_weights"] = _normalize_weight_map(merged.get("axis_weights"), RECIPE_AXES)
        merged["verifier_kind_weights"] = _normalize_weight_map(
            merged.get("verifier_kind_weights"), verifier_keys
        )
        merged["fixture_kind_weights"] = _normalize_weight_map(
            merged.get("fixture_kind_weights"), fixture_keys
        )
        return merged

    def _build_axis_counts(self) -> dict[str, Counter[str]]:
        counts: dict[str, Counter[str]] = {axis: Counter() for axis in RECIPE_AXES}
        for env in self.environments:
            for axis in RECIPE_AXES:
                counts[axis].update(_axis_values(env, axis))
        return counts

    def _environment_score(self, env: EnvironmentSpec, genome: dict[str, Any]) -> float:
        score = 0.0
        for axis, weight in genome["axis_weights"].items():
            if weight <= 0:
                continue
            for value in _axis_values(env, axis):
                score += weight / max(1, self._value_counts[axis].get(value, 1))

        score += 0.2 * genome["verifier_kind_weights"].get(env.verifier.kind, 1.0)
        for fixture_kind in _axis_values(env, "fixture_kind"):
            score += 0.1 * genome["fixture_kind_weights"].get(fixture_kind, 1.0)

        pass_at_1 = _metadata_rate(env, "pass@1")
        if pass_at_1 is not None:
            score -= 0.5 * abs(pass_at_1 - genome["pass_at_1_target"])

        return score + (_stable_jitter(genome["seed"], env.id) * 0.01)

    def _mutate_weights(self, weights: dict[str, float]) -> None:
        for key, value in list(weights.items()):
            if random.random() > self.mutation_rate:
                continue
            mutated = value + random.gauss(0, max(0.05, value * self.mutation_scale))
            weights[key] = round(_clamp(mutated, 0.05, 5.0), 4)
