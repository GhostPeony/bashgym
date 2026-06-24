"""Environment-level pass@k reports for executable terminal tasks.

The existing ``passk`` module owns the unbiased estimator. This module adds the
terminal-environment bookkeeping around it: explicit attempt records, telemetry
summaries, and per-environment reports that can be stored on model profiles or
shown in the dashboard. Attempt records deliberately include action and
observation token counters so ECHO-style training can reuse the same rollout
logs later instead of throwing terminal feedback away.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.passk import EpisodeResult, PassKReport, compute_pass_at_k


@dataclass
class EnvironmentAttempt:
    """One sampled attempt for one executable environment."""

    environment_id: str
    attempt_index: int
    passed: bool
    reward: float | None = None
    verifier_status: str | None = None
    timeout: bool = False
    tool_calls: int | None = None
    tokens: int | None = None
    action_tokens: int | None = None
    observation_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "attempt_index": self.attempt_index,
            "passed": self.passed,
            "reward": self.reward,
            "verifier_status": self.verifier_status,
            "timeout": self.timeout,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "action_tokens": self.action_tokens,
            "observation_tokens": self.observation_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentAttempt:
        return cls(
            environment_id=str(data.get("environment_id") or data.get("env_id") or ""),
            attempt_index=int(data.get("attempt_index", data.get("sample_index", 0))),
            passed=bool(data.get("passed", data.get("success", False))),
            reward=_optional_float(data.get("reward")),
            verifier_status=(
                str(data["verifier_status"]) if data.get("verifier_status") is not None else None
            ),
            timeout=bool(data.get("timeout", False)),
            tool_calls=_optional_int(data.get("tool_calls")),
            tokens=_optional_int(data.get("tokens")),
            action_tokens=_optional_int(data.get("action_tokens")),
            observation_tokens=_optional_int(data.get("observation_tokens")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EnvironmentPassKReport:
    """Environment pass@k plus rollout telemetry."""

    k_values: list[int]
    reports: dict[int, PassKReport]
    episodes: list[EpisodeResult]
    attempts: list[EnvironmentAttempt]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_environments(self) -> int:
        return len(self.episodes)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def pass_at_k(self) -> dict[str, float]:
        return {f"pass@{k}": self.reports[k].mean_pass_at_k for k in self.k_values}

    @property
    def mean_success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(
            (episode.c / episode.n) if episode.n else 0.0 for episode in self.episodes
        ) / len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_values": self.k_values,
            "n_environments": self.n_environments,
            "n_attempts": self.n_attempts,
            "pass_at_k": self.pass_at_k,
            "mean_success_rate": self.mean_success_rate,
            "per_environment": self._per_environment(),
            "attempt_summary": self._attempt_summary(),
            "warnings": self.warnings,
        }

    def _per_environment(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for episode in self.episodes:
            row: dict[str, Any] = {
                "attempts": episode.n,
                "passes": episode.c,
                "success_rate": (episode.c / episode.n) if episode.n else 0.0,
            }
            for k in self.k_values:
                row[f"pass@{k}"] = self.reports[k].per_task.get(episode.task_id, 0.0)
            out[episode.task_id] = row
        return out

    def _attempt_summary(self) -> dict[str, Any]:
        status_counts = Counter(
            attempt.verifier_status or ("passed" if attempt.passed else "failed")
            for attempt in self.attempts
        )
        return {
            "timeout_rate": _mean_bool([attempt.timeout for attempt in self.attempts]),
            "mean_tool_calls": _mean_optional([attempt.tool_calls for attempt in self.attempts]),
            "mean_tokens": _mean_optional([attempt.tokens for attempt in self.attempts]),
            "mean_action_tokens": _mean_optional(
                [attempt.action_tokens for attempt in self.attempts]
            ),
            "mean_observation_tokens": _mean_optional(
                [attempt.observation_tokens for attempt in self.attempts]
            ),
            "verifier_status_distribution": dict(sorted(status_counts.items())),
        }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean_optional(values: list[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _environment_id(env: EnvironmentSpec | dict[str, Any] | str) -> str:
    if isinstance(env, EnvironmentSpec):
        return env.id
    if isinstance(env, dict):
        return str(env.get("id") or env.get("task_id") or env.get("name") or "")
    return str(env)


def _normalize_attempt(value: bool | EnvironmentAttempt | dict[str, Any], env_id: str, i: int):
    if isinstance(value, EnvironmentAttempt):
        return value
    if isinstance(value, dict):
        attempt = EnvironmentAttempt.from_dict(value)
        if not attempt.environment_id:
            attempt.environment_id = env_id
        return attempt
    return EnvironmentAttempt(environment_id=env_id, attempt_index=i, passed=bool(value))


def evaluate_environment_attempts(
    environments: list[EnvironmentSpec | dict[str, Any] | str],
    attempts: list[EnvironmentAttempt | dict[str, Any]],
    *,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    require_complete: bool = True,
) -> EnvironmentPassKReport:
    """Compute environment pass@k from already-collected attempt outcomes."""
    env_ids = [_environment_id(env) for env in environments]
    if not env_ids:
        env_ids = sorted(
            {_normalize_attempt(attempt, "", 0).environment_id for attempt in attempts}
        )
    if not all(env_ids):
        raise ValueError("all environments must have an id")

    normalized = [
        (
            attempt
            if isinstance(attempt, EnvironmentAttempt)
            else EnvironmentAttempt.from_dict(attempt)
        )
        for attempt in attempts
    ]
    known = set(env_ids)
    unknown = sorted(
        {attempt.environment_id for attempt in normalized if attempt.environment_id not in known}
    )
    if unknown:
        raise ValueError(f"attempts reference unknown environments: {unknown}")

    ks = sorted({int(k) for k in k_values})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")
    max_k = max(ks)

    by_env: dict[str, list[EnvironmentAttempt]] = {env_id: [] for env_id in env_ids}
    seen: set[tuple[str, int]] = set()
    warnings: list[str] = []
    for attempt in normalized:
        key = (attempt.environment_id, attempt.attempt_index)
        if key in seen:
            raise ValueError(
                f"duplicate attempt index for {attempt.environment_id}: {attempt.attempt_index}"
            )
        seen.add(key)
        by_env[attempt.environment_id].append(attempt)

    episodes: list[EpisodeResult] = []
    ordered_attempts: list[EnvironmentAttempt] = []
    for env_id in env_ids:
        env_attempts = sorted(by_env[env_id], key=lambda attempt: attempt.attempt_index)
        if require_complete and len(env_attempts) < max_k:
            raise ValueError(
                f"environment {env_id!r} has {len(env_attempts)} attempts, needs at least {max_k}"
            )
        if len(env_attempts) < max_k:
            warnings.append(
                f"{env_id} has fewer attempts ({len(env_attempts)}) than max k ({max_k})"
            )
        episodes.append(EpisodeResult(env_id, [attempt.passed for attempt in env_attempts]))
        ordered_attempts.extend(env_attempts)

    reports = {k: compute_pass_at_k(episodes, k) for k in ks}
    return EnvironmentPassKReport(
        k_values=ks,
        reports=reports,
        episodes=episodes,
        attempts=ordered_attempts,
        warnings=warnings,
    )


def evaluate_environment_pass_at_k(
    environments: list[EnvironmentSpec | dict[str, Any] | str],
    run_episode: Callable[
        [EnvironmentSpec | dict[str, Any] | str, int], bool | EnvironmentAttempt | dict[str, Any]
    ],
    *,
    n_samples: int,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
) -> EnvironmentPassKReport:
    """Collect attempts with ``run_episode`` and compute environment pass@k."""
    if n_samples < max(k_values):
        raise ValueError(f"n_samples ({n_samples}) must be >= max k ({max(k_values)})")
    attempts: list[EnvironmentAttempt] = []
    for env in environments:
        env_id = _environment_id(env)
        for i in range(n_samples):
            attempts.append(_normalize_attempt(run_episode(env, i), env_id, i))
    return evaluate_environment_attempts(environments, attempts, k_values=k_values)
