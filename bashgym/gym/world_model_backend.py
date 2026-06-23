"""Backend adapter helpers for ECHO/RWML replay payloads.

The DPPO replay schema stores world-model data as provider-neutral JSON:
role-tagged ECHO text spans and RWML transition triplets. A trainer backend
still needs to tokenize those spans, build ECHO masks, and align predicted next
states with RWML targets. This module is that narrow adapter layer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA, EchoMasks, EchoSegment, build_echo_masks
from bashgym.gym.echo_trainer import echo_augmented_loss
from bashgym.gym.rwml import (
    RWML_DEFAULT_DISTANCE_THRESHOLD,
    RWML_DEFAULT_EASY_KEEP_PROBABILITY,
    RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
    RWML_DEFAULT_HISTORY_WINDOW,
    EmbedFn,
    WorldModelTransition,
    build_world_model_reward_fn,
    cosine_similarity,
)

TokenizerLike = Callable[[str], Sequence[int] | dict[str, Any]]
BatchEmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]
PredictionTextFn = Callable[[Any], str]


@dataclass(frozen=True)
class WorldModelBackendBatch:
    """Tokenized ECHO masks plus RWML targets extracted from replay records."""

    echo_masks: tuple[EchoMasks, ...]
    rwml_transitions: tuple[WorldModelTransition, ...]
    records_total: int
    records_with_world_model: int

    @property
    def actual_next_states(self) -> tuple[str, ...]:
        return tuple(transition.next_state for transition in self.rwml_transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_total": self.records_total,
            "records_with_world_model": self.records_with_world_model,
            "echo_sequences": len(self.echo_masks),
            "echo_tokens": sum(len(mask.input_ids) for mask in self.echo_masks),
            "echo_observation_tokens": sum(
                mask.total_observation_tokens for mask in self.echo_masks
            ),
            "rwml_transitions": len(self.rwml_transitions),
        }


@dataclass(frozen=True)
class WorldModelTrainerSettings:
    """Resolved world-model settings consumed by external trainer hooks."""

    echo_enabled: bool = False
    echo_aux_lambda: float = ECHO_DEFAULT_LAMBDA
    rwml_enabled: bool = False
    rwml_distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD
    rwml_easy_pass_rate_threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD
    rwml_easy_keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY
    rwml_history_window: int = RWML_DEFAULT_HISTORY_WINDOW
    rwml_embedding_model: str = ""
    rwml_kl_beta: float = 0.0

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        prefix: str = "BASHGYM_DPPO_",
    ) -> WorldModelTrainerSettings:
        """Build settings from the DPPO launch environment contract."""

        values = env or os.environ
        return cls(
            echo_enabled=_env_bool(values, f"{prefix}ECHO_ENABLED", False),
            echo_aux_lambda=_env_float(values, f"{prefix}ECHO_LAMBDA", ECHO_DEFAULT_LAMBDA),
            rwml_enabled=_env_bool(values, f"{prefix}RWML_ENABLED", False),
            rwml_distance_threshold=_env_float(
                values, f"{prefix}RWML_DISTANCE_THRESHOLD", RWML_DEFAULT_DISTANCE_THRESHOLD
            ),
            rwml_easy_pass_rate_threshold=_env_float(
                values,
                f"{prefix}RWML_EASY_PASS_RATE_THRESHOLD",
                RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
            ),
            rwml_easy_keep_probability=_env_float(
                values,
                f"{prefix}RWML_EASY_KEEP_PROBABILITY",
                RWML_DEFAULT_EASY_KEEP_PROBABILITY,
            ),
            rwml_history_window=_env_int(
                values, f"{prefix}RWML_HISTORY_WINDOW", RWML_DEFAULT_HISTORY_WINDOW
            ),
            rwml_embedding_model=values.get(f"{prefix}RWML_EMBEDDING_MODEL", ""),
            rwml_kl_beta=_env_float(values, f"{prefix}RWML_KL_BETA", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "echo_enabled": self.echo_enabled,
            "echo_aux_lambda": self.echo_aux_lambda,
            "rwml_enabled": self.rwml_enabled,
            "rwml_distance_threshold": self.rwml_distance_threshold,
            "rwml_easy_pass_rate_threshold": self.rwml_easy_pass_rate_threshold,
            "rwml_easy_keep_probability": self.rwml_easy_keep_probability,
            "rwml_history_window": self.rwml_history_window,
            "rwml_embedding_model": self.rwml_embedding_model,
            "rwml_kl_beta": self.rwml_kl_beta,
        }


@dataclass
class CachedEmbeddingProvider:
    """Small cached/batched embedding adapter for RWML scoring."""

    embed_fn: EmbedFn | None = None
    batch_embed_fn: BatchEmbedFn | None = None
    cache: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        missing: list[str] = []
        seen_missing: set[str] = set()
        for text in texts:
            if text not in self.cache and text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)

        if missing:
            if self.batch_embed_fn is not None:
                vectors = self.batch_embed_fn(missing)
            elif self.embed_fn is not None:
                vectors = [self.embed_fn(text) for text in missing]
            else:
                raise RuntimeError("RWML scoring requires embed_fn or batch_embed_fn")
            if len(vectors) != len(missing):
                raise ValueError("embedding provider returned the wrong number of vectors")
            for text, vector in zip(missing, vectors, strict=True):
                self.cache[text] = tuple(float(value) for value in vector)

        return [self.cache[text] for text in texts]


@dataclass(frozen=True)
class RWMLRewardResult:
    """RWML prediction rewards plus quality metrics for logging/release evidence."""

    predicted_next_states: tuple[str, ...]
    actual_next_states: tuple[str, ...]
    rewards: tuple[float, ...]
    distances: tuple[float, ...]
    distance_threshold: float

    @property
    def pass_rate(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def embedding_distance_mean(self) -> float:
        return sum(self.distances) / len(self.distances) if self.distances else 0.0

    @property
    def embedding_distance_p95(self) -> float:
        return _percentile(self.distances, 0.95)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.rewards),
            "rwml_pass_rate": self.pass_rate,
            "embedding_distance_mean": self.embedding_distance_mean,
            "embedding_distance_p95": self.embedding_distance_p95,
            "distance_threshold": self.distance_threshold,
            "rewards": list(self.rewards),
            "distances": list(self.distances),
        }


@dataclass
class WorldModelTrainerAdapter:
    """Trainer-facing ECHO/RWML hook object for TRL, verl, and DPPO adapters."""

    batch: WorldModelBackendBatch
    settings: WorldModelTrainerSettings = field(default_factory=WorldModelTrainerSettings)
    embedder: CachedEmbeddingProvider | None = None

    @classmethod
    def from_records(
        cls,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        settings: WorldModelTrainerSettings | None = None,
        embed_fn: EmbedFn | None = None,
        batch_embed_fn: BatchEmbedFn | None = None,
        exclude_token_ids: Iterable[int] = (),
    ) -> WorldModelTrainerAdapter:
        batch = build_world_model_backend_batch(
            records,
            tokenizer,
            exclude_token_ids=exclude_token_ids,
        )
        embedder = (
            CachedEmbeddingProvider(embed_fn=embed_fn, batch_embed_fn=batch_embed_fn)
            if embed_fn or batch_embed_fn
            else None
        )
        return cls(
            batch=batch,
            settings=settings or WorldModelTrainerSettings(),
            embedder=embedder,
        )

    @classmethod
    def from_replay_path(
        cls,
        replay_path: str | Path,
        tokenizer: Any,
        *,
        settings: WorldModelTrainerSettings | None = None,
        embed_fn: EmbedFn | None = None,
        batch_embed_fn: BatchEmbedFn | None = None,
        exclude_token_ids: Iterable[int] = (),
    ) -> WorldModelTrainerAdapter:
        return cls.from_records(
            read_replay_jsonl(replay_path),
            tokenizer,
            settings=settings,
            embed_fn=embed_fn,
            batch_embed_fn=batch_embed_fn,
            exclude_token_ids=exclude_token_ids,
        )

    @classmethod
    def from_env(
        cls,
        tokenizer: Any,
        *,
        env: Mapping[str, str] | None = None,
        embed_fn: EmbedFn | None = None,
        batch_embed_fn: BatchEmbedFn | None = None,
        exclude_token_ids: Iterable[int] = (),
    ) -> WorldModelTrainerAdapter:
        values = env or os.environ
        replay_path = values.get("BASHGYM_DPPO_REPLAY_PATH")
        if not replay_path:
            raise RuntimeError("BASHGYM_DPPO_REPLAY_PATH is required")
        return cls.from_replay_path(
            replay_path,
            tokenizer,
            settings=WorldModelTrainerSettings.from_env(values),
            embed_fn=embed_fn,
            batch_embed_fn=batch_embed_fn,
            exclude_token_ids=exclude_token_ids,
        )

    def apply_echo_loss(
        self,
        base_loss: Any,
        logits: Any,
        *,
        input_ids: Any | None = None,
        observation_mask: Any | None = None,
        total_observation_tokens: int | None = None,
        sequence_index: int = 0,
    ) -> Any:
        """Apply ECHO auxiliary loss inside a backend ``compute_loss`` hook.

        ``logits`` can be either ``(seq, vocab)`` or ``(batch, seq, vocab)``.
        If ``input_ids``/``observation_mask`` are omitted, the adapter uses the
        tokenized replay masks for ``sequence_index``.
        """

        if not self.settings.echo_enabled or not self.batch.echo_masks:
            return base_loss
        masks = self.batch.echo_masks[sequence_index]
        logits_for_loss = _select_sequence(logits, sequence_index)
        input_ids_for_loss = _select_sequence(
            input_ids if input_ids is not None else masks.input_ids,
            sequence_index,
            like=logits_for_loss,
            dtype="long",
        )
        observation_mask_for_loss = _select_sequence(
            observation_mask if observation_mask is not None else masks.observation_mask,
            sequence_index,
            like=logits_for_loss,
            dtype="bool",
        )
        total_tokens = (
            int(total_observation_tokens)
            if total_observation_tokens is not None
            else masks.total_observation_tokens
        )
        return echo_augmented_loss(
            base_loss,
            logits_for_loss,
            input_ids_for_loss,
            observation_mask_for_loss,
            total_tokens,
            aux_lambda=self.settings.echo_aux_lambda,
        )

    def score_rwml_predictions(
        self,
        predicted_next_states: Sequence[str],
        *,
        actual_next_states: Sequence[str] | None = None,
        embedder: CachedEmbeddingProvider | None = None,
        embed_fn: EmbedFn | None = None,
        batch_embed_fn: BatchEmbedFn | None = None,
    ) -> RWMLRewardResult:
        """Score predicted next states against replay/explicit RWML targets."""

        if not self.settings.rwml_enabled:
            return RWMLRewardResult(
                predicted_next_states=tuple(str(value) for value in predicted_next_states),
                actual_next_states=tuple(str(value) for value in (actual_next_states or ())),
                rewards=tuple(),
                distances=tuple(),
                distance_threshold=self.settings.rwml_distance_threshold,
            )
        active_embedder = (
            embedder
            or self.embedder
            or CachedEmbeddingProvider(embed_fn=embed_fn, batch_embed_fn=batch_embed_fn)
        )
        actual_states = actual_next_states or self.batch.actual_next_states
        return score_rwml_prediction_pairs(
            predicted_next_states,
            actual_states,
            active_embedder,
            distance_threshold=self.settings.rwml_distance_threshold,
        )

    def build_trl_rwml_reward_func(
        self,
        *,
        actual_states_key: str = "actual_next_state",
        prediction_text_fn: PredictionTextFn | None = None,
    ) -> Callable[..., list[float | None]]:
        """Return a TRL ``GRPOTrainer`` custom reward function for RWML.

        TRL passes completions plus dataset columns into custom reward functions.
        If ``actual_states_key`` is present, it is used as the target next state;
        otherwise the adapter consumes replay transitions in order.
        """

        def rwml_reward_func(completions: Sequence[Any], **kwargs: Any) -> list[float | None]:
            if not self.settings.rwml_enabled:
                return [None for _ in completions]
            predicted = [
                (prediction_text_fn or completion_to_text)(completion)
                for completion in completions
            ]
            actual = _coerce_string_sequence(kwargs.get(actual_states_key))
            if actual is None:
                actual = list(self.batch.actual_next_states[: len(predicted)])
            result = self.score_rwml_predictions(predicted, actual_next_states=actual)
            rwml_reward_func.last_rwml_result = result.to_dict()  # type: ignore[attr-defined]
            return list(result.rewards)

        rwml_reward_func.__name__ = "bashgym_rwml_reward"
        rwml_reward_func.last_rwml_result = {}  # type: ignore[attr-defined]
        return rwml_reward_func


def _world_model_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("world_model")
    return payload if isinstance(payload, dict) else None


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
    elif callable(tokenizer):
        encoded = tokenizer(text)
    else:
        raise TypeError("tokenizer must be callable or expose encode(text)")

    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    return [int(token_id) for token_id in encoded]


def echo_segments_from_replay_record(
    record: dict[str, Any],
    tokenizer: Any,
) -> list[EchoSegment]:
    """Tokenize the replay record's ECHO text spans into ``EchoSegment`` values."""

    payload = _world_model_payload(record)
    if not payload:
        return []
    echo = payload.get("echo") or {}
    if not isinstance(echo, dict):
        return []

    segments: list[EchoSegment] = []
    for raw_segment in echo.get("segments") or []:
        if not isinstance(raw_segment, dict):
            continue
        role = str(raw_segment.get("role") or "")
        text = str(raw_segment.get("text") or "")
        segments.append(EchoSegment(role=role, token_ids=_encode_text(tokenizer, text)))
    return segments


def echo_masks_from_replay_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    exclude_token_ids: Iterable[int] = (),
) -> EchoMasks:
    """Build ECHO action/observation masks from one replay record."""

    return build_echo_masks(
        echo_segments_from_replay_record(record, tokenizer),
        exclude_token_ids=exclude_token_ids,
    )


def rwml_transitions_from_replay_record(record: dict[str, Any]) -> list[WorldModelTransition]:
    """Parse RWML transition triplets from one replay record."""

    payload = _world_model_payload(record)
    if not payload:
        return []
    raw_transitions = payload.get("rwml_transitions") or []
    if not isinstance(raw_transitions, list):
        return []

    transitions: list[WorldModelTransition] = []
    for raw_transition in raw_transitions:
        if not isinstance(raw_transition, dict):
            continue
        prior_pairs: list[tuple[str, str]] = []
        raw_prior = raw_transition.get("prior") or []
        if isinstance(raw_prior, list):
            for raw_pair in raw_prior:
                if isinstance(raw_pair, (list, tuple)) and len(raw_pair) == 2:
                    prior_pairs.append((str(raw_pair[0]), str(raw_pair[1])))
        transitions.append(
            WorldModelTransition(
                instruction=str(raw_transition.get("instruction") or ""),
                prior=tuple(prior_pairs),
                action=str(raw_transition.get("action") or ""),
                next_state=str(raw_transition.get("next_state") or ""),
            )
        )
    return transitions


def build_world_model_backend_batch(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    exclude_token_ids: Iterable[int] = (),
) -> WorldModelBackendBatch:
    """Build a backend-ready batch from DPPO replay records."""

    echo_masks: list[EchoMasks] = []
    rwml_transitions: list[WorldModelTransition] = []
    records_with_world_model = 0

    for record in records:
        if not _world_model_payload(record):
            continue
        records_with_world_model += 1
        masks = echo_masks_from_replay_record(
            record,
            tokenizer,
            exclude_token_ids=exclude_token_ids,
        )
        if masks.input_ids:
            echo_masks.append(masks)
        rwml_transitions.extend(rwml_transitions_from_replay_record(record))

    return WorldModelBackendBatch(
        echo_masks=tuple(echo_masks),
        rwml_transitions=tuple(rwml_transitions),
        records_total=len(records),
        records_with_world_model=records_with_world_model,
    )


def read_replay_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read DPPO replay records from JSONL for trainer adapters."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def rwml_rewards_from_predictions(
    predicted_next_states: Sequence[str],
    transitions: Sequence[WorldModelTransition],
    embed_fn: EmbedFn,
    *,
    distance_threshold: float,
) -> list[float]:
    """Score predicted next states against RWML transition targets."""

    actual_next_states = [transition.next_state for transition in transitions]
    reward_fn = build_world_model_reward_fn(embed_fn, distance_threshold=distance_threshold)
    return reward_fn(predicted_next_states, actual_next_states)


def score_rwml_prediction_pairs(
    predicted_next_states: Sequence[str],
    actual_next_states: Sequence[str],
    embedder: CachedEmbeddingProvider,
    *,
    distance_threshold: float,
) -> RWMLRewardResult:
    """Score predicted/actual pairs with cached embeddings and return metrics."""

    predicted = tuple(str(value) for value in predicted_next_states)
    actual = tuple(str(value) for value in actual_next_states)
    if len(predicted) != len(actual):
        raise ValueError("predicted_next_states and actual_next_states must have equal length")

    vectors = embedder.embed_many([*predicted, *actual])
    predicted_vectors = vectors[: len(predicted)]
    actual_vectors = vectors[len(predicted) :]
    distances = tuple(
        1.0 - cosine_similarity(predicted_vector, actual_vector)
        for predicted_vector, actual_vector in zip(predicted_vectors, actual_vectors, strict=True)
    )
    rewards = tuple(1.0 if distance < distance_threshold else 0.0 for distance in distances)
    return RWMLRewardResult(
        predicted_next_states=predicted,
        actual_next_states=actual,
        rewards=rewards,
        distances=distances,
        distance_threshold=distance_threshold,
    )


def build_trl_rwml_reward_func(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    embed_fn: EmbedFn | None = None,
    batch_embed_fn: BatchEmbedFn | None = None,
    settings: WorldModelTrainerSettings | None = None,
    actual_states_key: str = "actual_next_state",
    prediction_text_fn: PredictionTextFn | None = None,
) -> Callable[..., list[float | None]]:
    """Convenience factory for TRL ``GRPOTrainer(reward_funcs=[...])``."""

    adapter = WorldModelTrainerAdapter.from_records(
        records,
        tokenizer,
        settings=settings,
        embed_fn=embed_fn,
        batch_embed_fn=batch_embed_fn,
    )
    return adapter.build_trl_rwml_reward_func(
        actual_states_key=actual_states_key,
        prediction_text_fn=prediction_text_fn,
    )


def build_verl_rwml_reward_fn(
    embed_fn: EmbedFn | None = None,
    *,
    batch_embed_fn: BatchEmbedFn | None = None,
    settings: WorldModelTrainerSettings | None = None,
) -> Callable[[str, str, str, dict[str, Any] | None], float]:
    """Return a verl-compatible custom reward function for RWML.

    The returned callable matches verl's documented signature:
    ``(data_source, solution_str, ground_truth, extra_info=None)``.
    """

    active_settings = settings or WorldModelTrainerSettings(rwml_enabled=True)
    embedder = CachedEmbeddingProvider(embed_fn=embed_fn, batch_embed_fn=batch_embed_fn)

    def bashgym_rwml_reward(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, Any] | None = None,
    ) -> float:
        del data_source
        target = _target_next_state(ground_truth, extra_info)
        result = score_rwml_prediction_pairs(
            [solution_str],
            [target],
            embedder,
            distance_threshold=active_settings.rwml_distance_threshold,
        )
        bashgym_rwml_reward.last_rwml_result = result.to_dict()  # type: ignore[attr-defined]
        return result.rewards[0]

    bashgym_rwml_reward.__name__ = "bashgym_rwml_reward"
    bashgym_rwml_reward.last_rwml_result = {}  # type: ignore[attr-defined]
    return bashgym_rwml_reward


def completion_to_text(completion: Any) -> str:
    """Normalize TRL standard/chat completions to plain prediction text."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    if isinstance(completion, Sequence):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def _env_bool(
    env: Mapping[str, str],
    key: str,
    default: bool,
) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(
    env: Mapping[str, str],
    key: str,
    default: float,
) -> float:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    return float(raw)


def _env_int(
    env: Mapping[str, str],
    key: str,
    default: int,
) -> int:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    return int(raw)


def _select_sequence(
    values: Any,
    sequence_index: int,
    *,
    like: Any | None = None,
    dtype: str | None = None,
) -> Any:
    if _looks_like_tensor(values):
        rank = getattr(values, "ndim", 0)
        selected = values[sequence_index] if rank > (1 if dtype is not None else 2) else values
        return selected

    if like is None:
        return values

    import torch

    torch_dtype = {"long": torch.long, "bool": torch.bool}.get(dtype or "")
    return torch.tensor(values, device=like.device, dtype=torch_dtype)


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "ndim") and hasattr(value, "to") and hasattr(value, "device")


def _coerce_string_sequence(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _target_next_state(ground_truth: str, extra_info: dict[str, Any] | None) -> str:
    if extra_info:
        for key in ("actual_next_state", "next_state", "ground_truth"):
            value = extra_info.get(key)
            if value not in (None, ""):
                return str(value)
    return str(ground_truth)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]
