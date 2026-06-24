"""Contamination-aware holdout gates for executable environment evals."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_passk import (
    EnvironmentAttempt,
    EnvironmentPassKReport,
    evaluate_environment_attempts,
)

ENVIRONMENT_HOLDOUT_SCHEMA_VERSION = "bashgym.environment_holdout.v1"
SPLIT_KEYS = {"domain", "source", "source_uri", "repo", "generator_seed", "task_family"}


def environment_hash(environment: EnvironmentSpec | dict[str, Any]) -> str:
    """Stable hash of executable environment content for leakage manifests."""

    spec = (
        environment
        if isinstance(environment, EnvironmentSpec)
        else EnvironmentSpec.from_dict(environment)
    )
    payload = {
        "instruction": spec.instruction,
        "verifier": spec.verifier.to_dict(),
        "files": spec.files,
        "fixtures": [fixture.to_dict() for fixture in spec.fixtures],
        "build": spec.build.to_dict(),
        "rollout": spec.rollout.to_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _metadata(spec: EnvironmentSpec) -> dict[str, Any]:
    return dict(spec.metadata or {})


def environment_group_key(spec: EnvironmentSpec, by: str) -> str:
    """Return the group key used to split train vs holdout environments."""

    if by not in SPLIT_KEYS:
        raise ValueError(f"split_by must be one of {sorted(SPLIT_KEYS)}")

    metadata = _metadata(spec)
    if by == "domain":
        return spec.domain or "_unknown"
    if by == "source":
        return spec.source or "_unknown"
    if by == "source_uri":
        return spec.source_uri or str(metadata.get("source_uri") or "_unknown")
    if by == "repo":
        repo = metadata.get("repo") or metadata.get("repository") or metadata.get("primary_repo")
        if isinstance(repo, dict):
            return str(repo.get("name") or repo.get("full_name") or "_unknown")
        return str(repo or "_unknown")
    if by == "generator_seed":
        return str(metadata.get("generator_seed") or metadata.get("seed") or "_unknown")
    return str(
        metadata.get("task_family")
        or metadata.get("family")
        or spec.axis_value("task_family")
        or "_unknown"
    )


@dataclass
class EnvironmentHoldoutSplit:
    """A deterministic grouped holdout split over environment specs."""

    train: list[EnvironmentSpec]
    holdout: list[EnvironmentSpec]
    by: str
    fraction: float
    seed: int
    holdout_hashes: set[str]
    train_hashes: set[str]
    holdout_group_keys: list[str]
    train_group_keys: list[str]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": ENVIRONMENT_HOLDOUT_SCHEMA_VERSION,
            "split_by": self.by,
            "fraction": self.fraction,
            "seed": self.seed,
            "n_train": len(self.train),
            "n_holdout": len(self.holdout),
            "train_ids": [env.id for env in self.train],
            "holdout_ids": [env.id for env in self.holdout],
            "train_hashes": sorted(self.train_hashes),
            "holdout_hashes": sorted(self.holdout_hashes),
            "train_group_keys": self.train_group_keys,
            "holdout_group_keys": self.holdout_group_keys,
        }


def make_environment_holdout_split(
    environments: list[EnvironmentSpec | dict[str, Any]],
    *,
    by: str = "task_family",
    fraction: float = 0.2,
    seed: int = 0,
) -> EnvironmentHoldoutSplit:
    """Freeze a grouped environment holdout split for release/eval gates."""

    if fraction <= 0 or fraction >= 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    specs = [
        (
            environment
            if isinstance(environment, EnvironmentSpec)
            else EnvironmentSpec.from_dict(environment)
        )
        for environment in environments
    ]
    groups: dict[str, list[EnvironmentSpec]] = {}
    for spec in specs:
        groups.setdefault(environment_group_key(spec, by), []).append(spec)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_holdout_groups = max(1, round(len(keys) * fraction)) if keys else 0
    holdout_keys = set(keys[:n_holdout_groups])

    train: list[EnvironmentSpec] = []
    holdout: list[EnvironmentSpec] = []
    for key in keys:
        target = holdout if key in holdout_keys else train
        target.extend(groups[key])

    train_hashes = {environment_hash(env) for env in train}
    holdout_hashes = {environment_hash(env) for env in holdout}
    return EnvironmentHoldoutSplit(
        train=train,
        holdout=holdout,
        by=by,
        fraction=fraction,
        seed=seed,
        holdout_hashes=holdout_hashes,
        train_hashes=train_hashes,
        train_group_keys=[key for key in keys if key not in holdout_keys],
        holdout_group_keys=[key for key in keys if key in holdout_keys],
    )


def environment_holdout_contamination(split: EnvironmentHoldoutSplit) -> list[str]:
    """Environment hashes that appear in both train and holdout splits."""

    return sorted(split.train_hashes & split.holdout_hashes)


def _attempt_summary_value(report: EnvironmentPassKReport, key: str) -> float:
    value = report.to_dict()["attempt_summary"].get(key)
    return float(value or 0.0)


def _attempt_environment_id(attempt: EnvironmentAttempt | dict[str, Any]) -> str:
    if isinstance(attempt, EnvironmentAttempt):
        return attempt.environment_id
    return str(attempt.get("environment_id") or "")


def evaluate_environment_holdout_gate(
    environments: list[EnvironmentSpec | dict[str, Any]],
    attempts: list[EnvironmentAttempt | dict[str, Any]],
    *,
    split_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    min_pass_at_1: float = 0.0,
    max_timeout_rate: float = 0.25,
    max_tamper_rate: float = 0.0,
    require_no_contamination: bool = True,
) -> dict[str, Any]:
    """Compute holdout pass@k and a release-gate verdict for environment attempts."""

    split = make_environment_holdout_split(
        environments,
        by=split_by,
        fraction=holdout_fraction,
        seed=seed,
    )
    holdout_ids = {env.id for env in split.holdout}
    holdout_attempts = [
        (
            attempt
            if isinstance(attempt, EnvironmentAttempt)
            else EnvironmentAttempt.from_dict(attempt)
        )
        for attempt in attempts
        if _attempt_environment_id(attempt) in holdout_ids
    ]
    report = evaluate_environment_attempts(split.holdout, holdout_attempts, k_values=k_values)
    report_data = report.to_dict()
    contamination = environment_holdout_contamination(split)
    status_dist = report_data["attempt_summary"]["verifier_status_distribution"]
    tamper_count = int(status_dist.get("tampered", 0))
    tamper_rate = tamper_count / report.n_attempts if report.n_attempts else 0.0
    timeout_rate = _attempt_summary_value(report, "timeout_rate")
    pass_at_1 = float(report.pass_at_k.get("pass@1", 0.0))

    reasons: list[str] = []
    if pass_at_1 < min_pass_at_1:
        reasons.append(f"pass@1 {pass_at_1:.3f} < required {min_pass_at_1:.3f}")
    if timeout_rate > max_timeout_rate:
        reasons.append(f"timeout rate {timeout_rate:.3f} > allowed {max_timeout_rate:.3f}")
    if tamper_rate > max_tamper_rate:
        reasons.append(f"tamper rate {tamper_rate:.3f} > allowed {max_tamper_rate:.3f}")
    if require_no_contamination and contamination:
        reasons.append(f"holdout contamination detected: {len(contamination)} hashes")

    return {
        "schema_version": ENVIRONMENT_HOLDOUT_SCHEMA_VERSION,
        "split": split.manifest(),
        "contamination": contamination,
        "report": report_data,
        "gate": {
            "ship": not reasons,
            "reasons": reasons,
            "thresholds": {
                "min_pass_at_1": min_pass_at_1,
                "max_timeout_rate": max_timeout_rate,
                "max_tamper_rate": max_tamper_rate,
                "require_no_contamination": require_no_contamination,
            },
            "observed": {
                "pass_at_1": pass_at_1,
                "timeout_rate": timeout_rate,
                "tamper_rate": tamper_rate,
            },
        },
    }
