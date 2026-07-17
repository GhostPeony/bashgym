"""DPPO replay-batch artifacts for terminal environment rollouts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.gym.dppo import DPPODivergence, DPPOToken, analyze_dppo_batch
from bashgym.gym.echo import ACTION_ROLE, OBSERVATION_ROLE
from bashgym.gym.rwml import RWML_DEFAULT_HISTORY_WINDOW, extract_transitions

if TYPE_CHECKING:
    from bashgym.environments.rollout import CommandObservation, EnvironmentRolloutResult

DPPO_REPLAY_SCHEMA_VERSION = "bashgym.dppo_replay.v1"
TrainLogprobScorer = Callable[[dict[str, Any]], dict[str, Any]]

# Note about the ``world_model`` enrichment key built by ``_build_world_model``.
# ECHO masks (``bashgym.gym.echo.build_echo_masks``) require model *token ids*,
# which a text-only replay record cannot provide without a tokenizer. Rather than
# fabricate token masks, the enrichment records role-tagged TEXT spans plus char
# counts. The terminal-RL trainer backend tokenizes these spans with its own
# model tokenizer and calls ``build_echo_masks`` downstream.
ECHO_DOWNSTREAM_NOTE = (
    "Text-only ECHO summary: action/observation spans are role-tagged text, not "
    "token ids. The terminal-RL trainer tokenizes these spans with the model "
    "tokenizer and builds bashgym.gym.echo.build_echo_masks downstream."
)


def _observation_state(observation: CommandObservation) -> str:
    """Map one command's terminal output (stdout/stderr) to a world-model state."""

    parts = [part for part in (observation.stdout, observation.stderr) if part]
    return "\n".join(parts)


def _build_world_model(
    rollout: EnvironmentRolloutResult,
    instruction: str,
    *,
    history_window: int,
) -> dict[str, Any]:
    """Build the optional RWML/ECHO enrichment for one replay record.

    ``rwml_transitions`` are ``WorldModelTransition.to_dict()`` triplets built from
    the rollout's (command -> observation output) steps via
    ``bashgym.gym.rwml.extract_transitions``; each ``CommandObservation`` maps to
    ``{action: command, state: stdout/stderr output}``.

    ``echo`` is a coarse, tokenizer-free summary: role-tagged TEXT spans
    (``action`` per command, ``observation`` per command output) plus char counts.
    Real ECHO token masks are built downstream by the trainer (see
    ``ECHO_DOWNSTREAM_NOTE``).
    """

    steps = [
        {"action": observation.command, "state": _observation_state(observation)}
        for observation in rollout.observations
        if observation.command
    ]
    transitions = extract_transitions(
        steps,
        instruction=instruction,
        history_window=history_window,
    )

    segments: list[dict[str, str]] = []
    n_action_chars = 0
    n_observation_chars = 0
    for step in steps:
        action_text = step["action"]
        observation_text = step["state"]
        segments.append({"role": ACTION_ROLE, "text": action_text})
        segments.append({"role": OBSERVATION_ROLE, "text": observation_text})
        n_action_chars += len(action_text)
        n_observation_chars += len(observation_text)

    return {
        "rwml_transitions": [transition.to_dict() for transition in transitions],
        "echo": {
            "segments": segments,
            "n_action_chars": n_action_chars,
            "n_observation_chars": n_observation_chars,
            "note": ECHO_DOWNSTREAM_NOTE,
        },
    }


def _behavior_policy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": metadata.get("model"),
        "base_url": metadata.get("base_url"),
        "response_logprobs": metadata.get("response_logprobs", []),
        "behavior_logprob_tokens": int(metadata.get("behavior_logprob_tokens") or 0),
        "behavior_logprob_sum": metadata.get("behavior_logprob_sum"),
        "behavior_mean_logprob": metadata.get("behavior_mean_logprob"),
    }


def _flatten_response_logprobs(
    response_logprobs: list[dict[str, Any]],
) -> tuple[list[str], list[float]]:
    tokens: list[str] = []
    logprobs: list[float] = []
    for response_logprob in response_logprobs:
        raw_tokens = response_logprob.get("tokens") or []
        raw_logprobs = response_logprob.get("token_logprobs") or []
        if not isinstance(raw_tokens, list) or not isinstance(raw_logprobs, list):
            continue
        for index, raw_logprob in enumerate(raw_logprobs):
            try:
                logprob = float(raw_logprob)
            except (TypeError, ValueError):
                continue
            token = str(raw_tokens[index]) if index < len(raw_tokens) else ""
            tokens.append(token)
            logprobs.append(logprob)
    return tokens, logprobs


def _behavior_tokens(record: dict[str, Any]) -> tuple[list[str], list[float]]:
    behavior = record.get("behavior_policy") or {}
    response_logprobs = behavior.get("response_logprobs") or []
    if not isinstance(response_logprobs, list):
        return [], []
    return _flatten_response_logprobs(response_logprobs)


def _group_relative_advantages(records: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    rewards_by_env: dict[str, list[float]] = {}
    for record in records:
        environment_id = str(record.get("environment_id", ""))
        try:
            reward = float(record.get("reward") or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        rewards_by_env.setdefault(environment_id, []).append(reward)

    mean_by_env = {
        environment_id: sum(rewards) / len(rewards)
        for environment_id, rewards in rewards_by_env.items()
        if rewards
    }
    advantages: dict[tuple[str, int], float] = {}
    for record in records:
        environment_id = str(record.get("environment_id", ""))
        attempt_index = int(record.get("attempt_index") or 0)
        try:
            reward = float(record.get("reward") or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        advantages[(environment_id, attempt_index)] = reward - mean_by_env.get(environment_id, 0.0)
    return advantages


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def summarize_world_model_payloads(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize optional ECHO/RWML replay payloads without scoring predictions.

    This is replay-coverage telemetry, not model-quality telemetry. Quality
    metrics such as embedding-distance distributions require predicted next
    states from a trainer/backend and should be layered on top of this summary.
    """

    world_model_records = 0
    rwml_transitions = 0
    rwml_prior_pairs = 0
    rwml_max_prior_pairs = 0
    echo_segments = 0
    echo_action_chars = 0
    echo_observation_chars = 0

    for record in records:
        world_model = record.get("world_model")
        if not isinstance(world_model, dict):
            continue
        world_model_records += 1

        transitions = world_model.get("rwml_transitions") or []
        if isinstance(transitions, list):
            rwml_transitions += len(transitions)
            for transition in transitions:
                if not isinstance(transition, dict):
                    continue
                prior = transition.get("prior") or []
                prior_len = len(prior) if isinstance(prior, list) else 0
                rwml_prior_pairs += prior_len
                rwml_max_prior_pairs = max(rwml_max_prior_pairs, prior_len)

        echo = world_model.get("echo") or {}
        if not isinstance(echo, dict):
            continue
        segments = echo.get("segments") or []
        if isinstance(segments, list):
            echo_segments += len(segments)
        echo_action_chars += _as_int(echo.get("n_action_chars"))
        echo_observation_chars += _as_int(echo.get("n_observation_chars"))

    total_echo_chars = echo_action_chars + echo_observation_chars
    return {
        "records": world_model_records,
        "records_missing_world_model": len(records) - world_model_records,
        "rwml_transitions": rwml_transitions,
        "rwml_mean_transitions_per_record": (
            rwml_transitions / world_model_records if world_model_records else 0.0
        ),
        "rwml_mean_prior_pairs": (rwml_prior_pairs / rwml_transitions if rwml_transitions else 0.0),
        "rwml_max_prior_pairs": rwml_max_prior_pairs,
        "echo_segments": echo_segments,
        "echo_action_chars": echo_action_chars,
        "echo_observation_chars": echo_observation_chars,
        "echo_observation_char_fraction": (
            echo_observation_chars / total_echo_chars if total_echo_chars else 0.0
        ),
    }


def read_dppo_replay_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a DPPO replay JSONL artifact."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def write_dppo_records_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write DPPO replay records to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_dppo_replay_records(
    environments: list[EnvironmentSpec],
    rollouts: list[EnvironmentRolloutResult],
    *,
    batch_id: str | None = None,
    include_world_model: bool = False,
    history_window: int = RWML_DEFAULT_HISTORY_WINDOW,
) -> list[dict[str, Any]]:
    """Build JSONL-ready records for sampled terminal RL optimizer replay.

    When ``include_world_model`` is set, each record additionally carries an
    optional ``world_model`` key with RWML ``rwml_transitions`` and a tokenizer-free
    ``echo`` text summary for the (pending) terminal-RL backend. The enrichment is
    purely additive: the ``bashgym.dppo_replay.v1`` schema semantics of every
    existing field are unchanged, and the key is omitted by default.
    """

    active_batch_id = batch_id or f"dppo_replay_{uuid.uuid4().hex[:12]}"
    env_by_id = {environment.id: environment for environment in environments}
    records: list[dict[str, Any]] = []
    for rollout in rollouts:
        environment = env_by_id.get(rollout.attempt.environment_id)
        if environment is None:
            continue
        metadata = rollout.attempt.metadata or {}
        behavior_tokens = int(metadata.get("behavior_logprob_tokens") or 0)
        train_tokens = int(metadata.get("train_logprob_tokens") or 0)
        record = {
            "schema_version": DPPO_REPLAY_SCHEMA_VERSION,
            "batch_id": active_batch_id,
            "environment_id": rollout.attempt.environment_id,
            "attempt_index": rollout.attempt.attempt_index,
            "prompt": environment.instruction,
            "environment": environment.to_dict(),
            "reward": rollout.attempt.reward if rollout.attempt.reward is not None else 0.0,
            "passed": rollout.attempt.passed,
            "verifier_status": rollout.attempt.verifier_status,
            "active_sampling_selected": bool(metadata.get("active_sampling_selected")),
            "reward_group_std": metadata.get("reward_group_std"),
            "trajectory": {
                "commands": [
                    observation.command
                    for observation in rollout.observations
                    if observation.command
                ],
                "observations": [observation.to_dict() for observation in rollout.observations],
                "verifier_observation": (
                    rollout.verifier_observation.to_dict() if rollout.verifier_observation else None
                ),
            },
            "behavior_policy": _behavior_policy_metadata(metadata),
            "optimizer": {
                "behavior_logprobs_ready": behavior_tokens > 0,
                "train_logprobs_ready": train_tokens > 0,
                "train_logprob_replay_required": behavior_tokens > 0 and train_tokens == 0,
                "train_logprob_tokens": train_tokens,
            },
        }
        if include_world_model:
            record["world_model"] = _build_world_model(
                rollout,
                environment.instruction,
                history_window=history_window,
            )
        records.append(record)
    return records


def summarize_dppo_replay_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact telemetry for a replay-batch artifact."""

    behavior_ready = sum(1 for record in records if record["optimizer"]["behavior_logprobs_ready"])
    train_ready = sum(1 for record in records if record["optimizer"].get("train_logprobs_ready"))
    replay_required = sum(
        1 for record in records if record["optimizer"]["train_logprob_replay_required"]
    )
    world_model = summarize_world_model_payloads(records)
    environment_ids = sorted({str(record["environment_id"]) for record in records})
    return {
        "schema_version": DPPO_REPLAY_SCHEMA_VERSION,
        "records": len(records),
        "environments": len(environment_ids),
        "environment_ids": environment_ids,
        "behavior_logprobs_ready_records": behavior_ready,
        "train_logprobs_ready_records": train_ready,
        "train_logprob_replay_required_records": replay_required,
        "world_model_records": world_model["records"],
        "world_model": world_model,
    }


def enrich_dppo_replay_records(
    records: list[dict[str, Any]],
    train_logprob_scorer: TrainLogprobScorer,
    *,
    divergence: DPPODivergence = "binary_tv",
    threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach train-policy logprobs and compute DPPO trust-region telemetry.

    ``train_logprob_scorer`` receives one replay record and must return at least
    ``{"token_logprobs": [...]}`` for the same sampled behavior tokens.
    """

    if not records:
        raise ValueError("records must not be empty")
    advantages = _group_relative_advantages(records)
    enriched: list[dict[str, Any]] = []
    dppo_tokens: list[DPPOToken] = []
    mismatched_records = 0
    missing_behavior_records = 0

    for record in records:
        behavior_tokens, behavior_logprobs = _behavior_tokens(record)
        if not behavior_logprobs:
            missing_behavior_records += 1
            enriched.append(record)
            continue

        scored = train_logprob_scorer(record)
        raw_train_logprobs = scored.get("token_logprobs") or []
        if not isinstance(raw_train_logprobs, list):
            raise ValueError("train scorer must return token_logprobs as a list")
        train_logprobs = [float(logprob) for logprob in raw_train_logprobs]
        if len(train_logprobs) != len(behavior_logprobs):
            mismatched_records += 1
            raise ValueError(
                "train scorer returned "
                f"{len(train_logprobs)} logprobs for {len(behavior_logprobs)} behavior tokens"
            )

        environment_id = str(record.get("environment_id", ""))
        attempt_index = int(record.get("attempt_index") or 0)
        advantage = advantages[(environment_id, attempt_index)]
        token_decisions_start = len(dppo_tokens)
        for token, behavior_logprob, train_logprob in zip(
            behavior_tokens, behavior_logprobs, train_logprobs, strict=True
        ):
            dppo_tokens.append(
                DPPOToken(
                    behavior_logprob=behavior_logprob,
                    train_logprob=train_logprob,
                    advantage=advantage,
                    token=token,
                )
            )

        updated = dict(record)
        optimizer = dict(updated.get("optimizer") or {})
        train_sum = sum(train_logprobs)
        updated["train_policy"] = {
            "model": scored.get("model"),
            "base_url": scored.get("base_url"),
            "tokens": scored.get("tokens") or behavior_tokens,
            "token_logprobs": train_logprobs,
            "train_logprob_tokens": len(train_logprobs),
            "train_logprob_sum": train_sum,
            "train_mean_logprob": train_sum / len(train_logprobs) if train_logprobs else None,
        }
        optimizer.update(
            {
                "train_logprobs_ready": True,
                "train_logprob_replay_required": False,
                "train_logprob_tokens": len(train_logprobs),
                "advantage": advantage,
                "dppo_token_offset": token_decisions_start,
                "dppo_token_count": len(train_logprobs),
            }
        )
        updated["optimizer"] = optimizer
        enriched.append(updated)

    if not dppo_tokens:
        raise ValueError("no records had behavior logprobs to replay")

    decisions, telemetry = analyze_dppo_batch(
        dppo_tokens,
        divergence=divergence,
        threshold=threshold,
    )
    summary = summarize_dppo_replay_records(enriched)
    summary.update(
        {
            "missing_behavior_logprob_records": missing_behavior_records,
            "mismatched_train_logprob_records": mismatched_records,
            "dppo": telemetry.to_dict(),
            "masked_update_tokens": [decision.token for decision in decisions if decision.masked][
                :20
            ],
        }
    )
    return enriched, summary


def enrich_dppo_replay_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    train_logprob_scorer: TrainLogprobScorer,
    *,
    divergence: DPPODivergence = "binary_tv",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Read replay JSONL, attach train logprobs, write enriched JSONL."""

    records = read_dppo_replay_jsonl(input_path)
    enriched, summary = enrich_dppo_replay_records(
        records,
        train_logprob_scorer,
        divergence=divergence,
        threshold=threshold,
    )
    write_dppo_records_jsonl(output_path, enriched)
    summary["input_path"] = str(input_path)
    summary["path"] = str(output_path)
    summary["batch_id"] = enriched[0].get("batch_id") if enriched else None
    return summary


def write_dppo_replay_jsonl(
    path: str | Path,
    environments: list[EnvironmentSpec],
    rollouts: list[EnvironmentRolloutResult],
    *,
    batch_id: str | None = None,
    include_world_model: bool = False,
    history_window: int = RWML_DEFAULT_HISTORY_WINDOW,
) -> dict[str, Any]:
    """Write sampled rollout records to JSONL and return artifact metadata."""

    output_path = Path(path)
    records = build_dppo_replay_records(
        environments,
        rollouts,
        batch_id=batch_id,
        include_world_model=include_world_model,
        history_window=history_window,
    )
    write_dppo_records_jsonl(output_path, records)

    summary = summarize_dppo_replay_records(records)
    summary["path"] = str(output_path)
    summary["batch_id"] = records[0]["batch_id"] if records else batch_id
    return summary
