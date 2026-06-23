"""Metrics for executable terminal-environment mixes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec


def axis_distribution(values: list[str]) -> dict[str, int]:
    """Count non-empty axis values in stable sorted-key order."""
    counts = Counter(v for v in values if v)
    return {k: counts[k] for k in sorted(counts)}


def balance_score(values: list[str], possible_values: list[str] | None = None) -> float:
    """Per-uniform diversity score from TMax: ``exp(H) / N``.

    ``N`` is the number of expected buckets when ``possible_values`` is supplied,
    otherwise the number of observed buckets. The score is in [1/N, 1] when at
    least one value is present.
    """
    counts = Counter(v for v in values if v)
    if not counts:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    n_buckets = len(possible_values) if possible_values else len(counts)
    if n_buckets <= 0:
        return 0.0
    return math.exp(entropy) / n_buckets


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


@dataclass
class EnvironmentMixReport:
    total: int
    domain_distribution: dict[str, int]
    skill_distribution: dict[str, int]
    axis_balance: dict[str, float]
    verifier_distribution: dict[str, int]
    mean_pass_rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "domain_distribution": self.domain_distribution,
            "skill_distribution": self.skill_distribution,
            "axis_balance": self.axis_balance,
            "verifier_distribution": self.verifier_distribution,
            "mean_pass_rates": self.mean_pass_rates,
        }


def summarize_environment_mix(
    envs: list[EnvironmentSpec],
    *,
    possible_domains: list[str] | None = None,
    possible_skills: list[str] | None = None,
) -> EnvironmentMixReport:
    """Summarize diversity and difficulty signals for an environment set."""
    domains = [env.domain for env in envs]
    skills = [skill for env in envs for skill in env.skills]
    verifier_kinds = [env.verifier.kind for env in envs]
    axis_balance = {
        "domain": balance_score(domains, possible_domains),
        "skill": balance_score(skills, possible_skills),
    }
    for axis_name in ("task_complexity", "command_complexity", "language", "fixture_kind"):
        axis_values = [env.axis_value(axis_name) or "" for env in envs]
        if any(axis_values):
            axis_balance[axis_name] = balance_score(axis_values)

    mean_pass_rates: dict[str, float] = {}
    for key in ("pass@1", "pass@4", "pass@8"):
        rates = [r for env in envs if (r := _metadata_rate(env, key)) is not None]
        if rates:
            mean_pass_rates[key] = sum(rates) / len(rates)

    return EnvironmentMixReport(
        total=len(envs),
        domain_distribution=axis_distribution(domains),
        skill_distribution=axis_distribution(skills),
        axis_balance=axis_balance,
        verifier_distribution=axis_distribution(verifier_kinds),
        mean_pass_rates=mean_pass_rates,
    )
