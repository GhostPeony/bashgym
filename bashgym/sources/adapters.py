"""Fixture/local source adapters for BashGym source cards.

The first adapter layer is intentionally dependency-light: it converts local
JSON/JSONL records into BashGym artifact contracts while preserving source-card
provenance. Network download/Hugging Face orchestration can sit above this
contract later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bashgym.preferences import (
    validate_preference_pair_records,
    validate_reward_example_records,
)
from bashgym.sources.catalog import (
    SourceArtifactType,
    SourceCard,
    SourceUse,
    prepare_source_manifest,
)

SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION = "bashgym.source_artifact_prepare.v1"
SOURCE_SCHEMA_MAPPING_VERSION = "bashgym.source_schema_mapping.v1"
HELPSTEER2_SCORE_AXES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def _text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _score_axes(record: dict[str, Any], *, prefix: str = "") -> dict[str, float]:
    axes: dict[str, float] = {}
    for axis in HELPSTEER2_SCORE_AXES:
        value = _numeric(record.get(f"{prefix}{axis}"))
        if value is not None:
            axes[axis] = value
    return axes


def _aggregate_score(record: dict[str, Any], *, prefix: str = "") -> float | None:
    axes = _score_axes(record, prefix=prefix)
    if axes:
        return sum(axes.values()) / len(axes)
    for key in (
        f"{prefix}score",
        f"{prefix}reward",
        f"{prefix}rating",
        f"{prefix}quality",
        f"{prefix}preference_score",
    ):
        value = _numeric(record.get(key))
        if value is not None:
            return value
    return None


def _quality_from_score(score: float | int | str | None, *, scale_max: float = 4.0) -> float:
    value = _numeric(score)
    if value is None:
        return 1.0
    return max(0.0, min(1.0, value / scale_max))


def _record_id(record: dict[str, Any], index: int) -> str:
    metadata = _metadata(record)
    raw = _text(record, "source_record_id", "record_id", "id", "example_id", "pair_id") or str(
        _metadata_value(metadata, "source_record_id", "record_id", "id", "example_id") or ""
    )
    return raw or f"record-{index + 1:06d}"


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return ""


def _last_message_text(messages: list[dict[str, Any]], *roles: str) -> str:
    allowed = set(roles)
    for message in reversed(messages):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in allowed:
            text = _message_text(message)
            if text:
                return text
    return ""


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"}:
            text = _message_text(message)
            if text:
                return text
    return ""


def _stringify_response(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    messages = _messages(value)
    if messages:
        return _last_message_text(messages, "assistant", "gpt", "model") or _message_text(
            messages[-1]
        )
    if isinstance(value, dict):
        for key in ("content", "text", "response", "answer"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _prompt(record: dict[str, Any]) -> str:
    prompt = _text(record, "prompt", "input", "question", "instruction")
    if prompt:
        return prompt
    for key in ("messages", "conversations", "trajectory", "chosen", "rejected"):
        prompt = _first_user_text(_messages(record.get(key)))
        if prompt:
            return prompt
    return ""


def _response(record: dict[str, Any]) -> str:
    response = _text(record, "response", "completion", "answer", "output", "text")
    if response:
        return response
    for key in ("messages", "conversations", "trajectory"):
        messages = _messages(record.get(key))
        response = _last_message_text(messages, "assistant", "gpt", "model")
        if response:
            return response
    return ""


def _pair_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        text = _stringify_response(value)
        if text:
            return text
    return ""


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _score(record: dict[str, Any], default: float | None = None) -> float | int | str | None:
    for key in ("score", "reward", "rating", "label", "target", "preference_score"):
        value = record.get(key)
        if value not in (None, ""):
            return value
    numeric_axes = [
        value
        for key in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
        if isinstance((value := record.get(key)), int | float)
    ]
    if numeric_axes:
        return sum(numeric_axes) / len(numeric_axes)
    return default


def _with_mapping_metadata(
    record: dict[str, Any],
    *,
    source_schema: str,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(record)
    metadata = _metadata(payload)
    metadata.setdefault("source_schema", source_schema)
    if updates:
        metadata.update(
            {key: value for key, value in updates.items() if value not in (None, "", [], {})}
        )
    payload["metadata"] = metadata
    return payload


def _normalize_ultrafeedback_record(record: dict[str, Any]) -> dict[str, Any]:
    chosen_score = _numeric(record.get("chosen_score"))
    rejected_score = _numeric(record.get("rejected_score"))
    metadata_updates = {
        "source_schema": "ultrafeedback_binarized",
        "preference_source": "ultrafeedback_binarized",
        "pair_generation_method": "ultrafeedback_binarized_preference",
    }
    if chosen_score is not None:
        metadata_updates["chosen_quality_score"] = _quality_from_score(chosen_score)
    if rejected_score is not None:
        metadata_updates["rejected_quality_score"] = _quality_from_score(rejected_score)
    if chosen_score is not None and rejected_score is not None:
        metadata_updates["score_delta"] = chosen_score - rejected_score
    return _with_mapping_metadata(
        record,
        source_schema="ultrafeedback_binarized",
        updates=metadata_updates,
    )


def _helpsteer2_axis_metadata(record: dict[str, Any]) -> dict[str, Any]:
    axes = _score_axes(record)
    aggregate = _aggregate_score(record)
    updates: dict[str, Any] = {
        "source_schema": "helpsteer2_scored_response",
        "score_axes": axes,
        "reward_scale": "likert_0_to_4",
        "score_scale": "likert_0_to_4",
        "label_schema": "helpfulness_correctness_coherence_complexity_verbosity_mean",
    }
    if aggregate is not None:
        updates["aggregate_score"] = aggregate
        updates["quality_score"] = _quality_from_score(aggregate)
    return updates


def _normalize_helpsteer2_reward_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append(
            _with_mapping_metadata(
                record,
                source_schema="helpsteer2_scored_response",
                updates=_helpsteer2_axis_metadata(record),
            )
        )
    return normalized


def _helpsteer2_preference_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = _prompt(record)
        response_1 = _stringify_response(record.get("response_1"))
        response_2 = _stringify_response(record.get("response_2"))
        strength = _numeric(record.get("preference_strength"))
        if not (prompt and response_1 and response_2) or strength in (None, 0.0, -100.0):
            continue
        if strength < 0:
            chosen = response_1
            rejected = response_2
            preferred = "response_1"
        else:
            chosen = response_2
            rejected = response_1
            preferred = "response_2"
        pair_id = _text(record, "pair_id", "id", "example_id") or f"helpsteer2-pref-{index + 1:06d}"
        prompt_hash = _text(record, "prompt_hash") or _prompt_hash(prompt)
        metadata = _metadata(record)
        metadata.update(
            {
                "source_schema": "helpsteer2_preference",
                "pair_id": pair_id,
                "prompt_hash": prompt_hash,
                "preference_strength": strength,
                "preference_outcome": preferred,
                "pair_generation_method": "helpsteer2_human_preference",
                "label_strength": f"{preferred}_better_by_{abs(int(strength))}",
                "chosen_quality_score": 1.0,
                "rejected_quality_score": 0.0,
                "score_delta": abs(strength),
                "reward_scale": "preference_strength_-3_to_3",
                "label_schema": "negative=response_1_preferred,positive=response_2_preferred,-100=invalid",
            }
        )
        pairs.append(
            {
                "id": pair_id,
                "pair_id": pair_id,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": 1.0,
                "rejected_score": 0.0,
                "metadata": metadata,
            }
        )
    return pairs


def _helpsteer2_scored_response_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any], float]]] = {}
    for index, record in enumerate(records):
        prompt = _prompt(record)
        response = _response(record)
        score = _aggregate_score(record)
        if prompt and response and score is not None:
            grouped.setdefault(prompt, []).append((index, record, score))

    pairs: list[dict[str, Any]] = []
    for prompt, candidates in grouped.items():
        candidates = sorted(candidates, key=lambda item: item[0])
        for pair_index in range(0, len(candidates) - 1, 2):
            left_index, left, left_score = candidates[pair_index]
            right_index, right, right_score = candidates[pair_index + 1]
            if left_score == right_score:
                continue
            if left_score > right_score:
                chosen_record, chosen_score = left, left_score
                rejected_record, rejected_score = right, right_score
                chosen_index, rejected_index = left_index, right_index
            else:
                chosen_record, chosen_score = right, right_score
                rejected_record, rejected_score = left, left_score
                chosen_index, rejected_index = right_index, left_index
            prompt_hash = _prompt_hash(prompt)
            pair_id = f"helpsteer2-score-{prompt_hash}-{pair_index // 2 + 1:03d}"
            metadata = dict(_metadata(chosen_record))
            metadata.update(
                {
                    "source_schema": "helpsteer2_scored_response_pair",
                    "pair_id": pair_id,
                    "prompt_hash": prompt_hash,
                    "pair_generation_method": "helpsteer2_score_delta_pair",
                    "label_strength": "higher_aggregate_score_preferred",
                    "chosen_quality_score": _quality_from_score(chosen_score),
                    "rejected_quality_score": _quality_from_score(rejected_score),
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "score_delta": chosen_score - rejected_score,
                    "chosen_source_record_index": chosen_index,
                    "rejected_source_record_index": rejected_index,
                    "reward_scale": "likert_0_to_4",
                    "label_schema": "higher_mean_axis_score_preferred",
                }
            )
            pairs.append(
                {
                    "id": pair_id,
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "prompt_hash": prompt_hash,
                    "chosen": _response(chosen_record),
                    "rejected": _response(rejected_record),
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "metadata": metadata,
                }
            )
    return pairs


def _normalize_helpsteer2_records(
    records: list[dict[str, Any]],
    *,
    goal: SourceUse,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    has_preference_rows = any(
        record.get("response_1") not in (None, "")
        and record.get("response_2") not in (None, "")
        and record.get("preference_strength") not in (None, "")
        for record in records
    )
    if goal == SourceUse.DPO:
        normalized = (
            _helpsteer2_preference_pairs(records)
            if has_preference_rows
            else _helpsteer2_scored_response_pairs(records)
        )
        mapper = (
            "helpsteer2_preference_pairs"
            if has_preference_rows
            else "helpsteer2_scored_response_pairs"
        )
        if has_preference_rows:
            consumed_records = len(normalized)
        else:
            consumed_indexes: set[int] = set()
            for pair in normalized:
                metadata = _metadata(pair)
                for key in ("chosen_source_record_index", "rejected_source_record_index"):
                    value = metadata.get(key)
                    if isinstance(value, int):
                        consumed_indexes.add(value)
            consumed_records = len(consumed_indexes)
    else:
        normalized = _normalize_helpsteer2_reward_records(records)
        mapper = "helpsteer2_scored_response"
        consumed_records = len(normalized)
    return normalized, {
        "schema_version": SOURCE_SCHEMA_MAPPING_VERSION,
        "source_id": "helpsteer2",
        "mapper": mapper,
        "input_records": len(records),
        "consumed_records": consumed_records,
        "normalized_records": len(normalized),
        "dropped_records": max(0, len(records) - consumed_records),
    }


def _normalize_source_records(
    card: SourceCard,
    records: list[dict[str, Any]],
    *,
    goal: SourceUse,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if card.id == "helpsteer2":
        return _normalize_helpsteer2_records(records, goal=goal)
    if card.id == "ultrafeedback_binarized":
        normalized = [_normalize_ultrafeedback_record(record) for record in records]
        return normalized, {
            "schema_version": SOURCE_SCHEMA_MAPPING_VERSION,
            "source_id": card.id,
            "mapper": "ultrafeedback_binarized",
            "input_records": len(records),
            "consumed_records": len(normalized),
            "normalized_records": len(normalized),
            "dropped_records": 0,
        }
    return records, {
        "schema_version": SOURCE_SCHEMA_MAPPING_VERSION,
        "source_id": card.id,
        "mapper": "generic_json_or_jsonl",
        "input_records": len(records),
        "consumed_records": len(records),
        "normalized_records": len(records),
        "dropped_records": 0,
    }


def _base_metadata(
    card: SourceCard,
    record: dict[str, Any],
    *,
    index: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    metadata = _metadata(record)
    source_record_id = _record_id(record, index)
    base = dict(metadata)
    base.update(
        {
            "source_id": card.id,
            "source_name": card.name,
            "source_record_id": source_record_id,
            "source_adapter": card.adapter,
            "source_manifest_path": manifest.get("manifest_path"),
            "domain": metadata.get("domain") or card.domain,
            "task_family": metadata.get("task_family") or card.task_family,
            "split": metadata.get("split") or record.get("split") or "train",
            "split_policy": metadata.get("split_policy") or card.split_policy,
            "decontamination_status": metadata.get("decontamination_status")
            or record.get("decontamination_status")
            or "source_card_policy_recorded_not_checked",
            "quality_score": (
                metadata.get("quality_score")
                if metadata.get("quality_score") is not None
                else record.get("quality_score", 1.0)
            ),
            "label_source": metadata.get("label_source")
            or record.get("label_source")
            or card.adapter,
        }
    )
    return base


def _load_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    records: list[dict[str, Any]]
    if input_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("records", "data", "examples", "pairs", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    records = [item for item in value if isinstance(item, dict)]
                    break
            else:
                records = [payload]
        elif isinstance(payload, list):
            records = [item for item in payload if isinstance(item, dict)]
        else:
            raise ValueError("JSON source input must be an object, list, or records container")
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            records.append(payload)
    return records[:limit] if limit is not None else records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _to_sft_examples(
    card: SourceCard,
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        messages = _messages(record.get("messages")) or _messages(record.get("conversations"))
        if not messages:
            prompt = _prompt(record)
            response = _response(record)
            if prompt and response:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
        if not messages:
            continue
        metadata = _base_metadata(card, record, index=index, manifest=manifest)
        examples.append({"messages": messages, "metadata": metadata})
    return examples


def _to_dpo_pairs(
    card: SourceCard,
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = _prompt(record)
        chosen = _pair_text(record, "chosen", "chosen_response", "chosen_text", "accepted")
        rejected = _pair_text(record, "rejected", "rejected_response", "rejected_text", "declined")
        if not (prompt and chosen and rejected):
            continue
        metadata = _base_metadata(card, record, index=index, manifest=manifest)
        pair_id = _text(record, "pair_id", "id", "example_id") or f"{card.id}-pair-{index + 1:06d}"
        prompt_hash = _text(record, "prompt_hash") or str(
            metadata.get("prompt_hash") or _prompt_hash(prompt)
        )
        source_record_id = str(metadata["source_record_id"])
        metadata.update(
            {
                "pair_id": pair_id,
                "prompt_hash": prompt_hash,
                "chosen_trace_id": metadata.get("chosen_trace_id") or f"{source_record_id}:chosen",
                "rejected_trace_id": metadata.get("rejected_trace_id")
                or f"{source_record_id}:rejected",
                "pair_generation_method": metadata.get("pair_generation_method")
                or "source_preference_pair",
                "label_strength": metadata.get("label_strength")
                or "source_preferred_over_rejected",
                "chosen_quality_score": metadata.get("chosen_quality_score", 1.0),
                "rejected_quality_score": metadata.get("rejected_quality_score", 0.0),
            }
        )
        pairs.append(
            {
                "id": pair_id,
                "pair_id": pair_id,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "metadata": metadata,
            }
        )
    return pairs


def _to_reward_examples(
    card: SourceCard,
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    process: bool = False,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = _prompt(record)
        metadata = _base_metadata(card, record, index=index, manifest=manifest)
        pair_chosen = _pair_text(record, "chosen", "chosen_response", "chosen_text", "accepted")
        pair_rejected = _pair_text(
            record, "rejected", "rejected_response", "rejected_text", "declined"
        )
        pair_id = _text(record, "pair_id", "id", "example_id") or f"{card.id}-pair-{index + 1:06d}"
        reward_type = (
            "process_reward" if process else str(record.get("reward_type") or "preference_reward")
        )

        if prompt and pair_chosen and pair_rejected and not process:
            chosen_score = _numeric(record.get("chosen_score"))
            rejected_score = _numeric(record.get("rejected_score"))
            pair_items = (
                ("chosen", pair_chosen, 1.0 if chosen_score is None else chosen_score),
                ("rejected", pair_rejected, 0.0 if rejected_score is None else rejected_score),
            )
            for label, response, score in pair_items:
                example_id = f"{pair_id}-{label}"
                example_metadata = dict(metadata)
                example_metadata.update(
                    {
                        "reward_example_id": example_id,
                        "pair_id": pair_id,
                        "prompt_hash": str(metadata.get("prompt_hash") or _prompt_hash(prompt)),
                        "reward_type": "preference_reward",
                        "reward_scale": metadata.get("reward_scale") or "0_to_1",
                        "label_schema": metadata.get("label_schema") or "chosen=1,rejected=0",
                        "preference_outcome": label,
                    }
                )
                examples.append(
                    {
                        "id": example_id,
                        "reward_example_id": example_id,
                        "reward_type": "preference_reward",
                        "prompt": prompt,
                        "response": response,
                        "score": score,
                        "metadata": example_metadata,
                    }
                )
            continue

        response = _response(record)
        score = _score(record, default=1.0 if response else None)
        if not (prompt and response and score is not None):
            continue
        example_id = _text(record, "reward_example_id", "example_id", "id") or (
            f"{card.id}-reward-{index + 1:06d}"
        )
        metadata.update(
            {
                "reward_example_id": example_id,
                "reward_type": reward_type,
                "reward_scale": metadata.get("reward_scale") or "source_native",
                "label_schema": metadata.get("label_schema") or "source_native_score",
            }
        )
        example: dict[str, Any] = {
            "id": example_id,
            "reward_example_id": example_id,
            "reward_type": reward_type,
            "prompt": prompt,
            "response": response,
            "score": score,
            "metadata": metadata,
        }
        for key in ("steps", "step_rewards", "process_rewards", "step_scores"):
            if key in record:
                example[key] = record[key]
        examples.append(example)
    return examples


def _eval_manifest(
    card: SourceCard, records: list[dict[str, Any]], *, input_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": "bashgym.source_eval_manifest.v1",
        "source_id": card.id,
        "source_name": card.name,
        "adapter": card.adapter,
        "input_path": str(input_path),
        "eval_only": card.eval_only,
        "record_count": len(records),
        "record_ids": [_record_id(record, index) for index, record in enumerate(records)],
        "split_policy": card.split_policy,
        "decontam_notes": card.decontam_notes,
    }


def _to_environment_specs(
    card: SourceCard,
    records: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from bashgym.environments.loader import environment_from_record

    specs: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        metadata = _base_metadata(card, record, index=index, manifest=manifest)
        env_record = dict(record)
        env_record["metadata"] = metadata
        env_record.setdefault("domain", card.domain)
        env_record.setdefault("skills", [card.task_family])
        env_record.setdefault("license", card.license)
        env_record.setdefault("source_uri", card.homepage)
        spec = environment_from_record(env_record, source=card.id, source_uri=card.homepage)
        errors = spec.validation_errors()
        findings.append(
            {
                "index": index,
                "source_record_id": metadata["source_record_id"],
                "environment_id": spec.id,
                "ok": not errors,
                "errors": errors,
            }
        )
        if not errors:
            specs.append(spec.to_dict())
    invalid_count = sum(1 for finding in findings if not finding["ok"])
    return specs, {
        "schema_version": "bashgym.environment_source_validation.v1",
        "ok": invalid_count == 0,
        "total_records": len(records),
        "valid_count": len(specs),
        "invalid_count": invalid_count,
        "findings": findings,
    }


def prepare_source_artifacts(
    card: SourceCard,
    *,
    goal: SourceUse | str,
    input_path: str | Path,
    output_dir: str | Path,
    allow_eval_only: bool = False,
    override_reason: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert local source records into BashGym artifacts for a source goal."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_use = goal if isinstance(goal, SourceUse) else SourceUse(goal)
    manifest = prepare_source_manifest(
        card,
        goal=source_use,
        output_dir=output_path,
        allow_eval_only=allow_eval_only,
        override_reason=override_reason,
    )
    report: dict[str, Any] = {
        "schema_version": SOURCE_ARTIFACT_PREPARE_SCHEMA_VERSION,
        "ok": manifest["use_verdict"]["ok"],
        "source_id": card.id,
        "goal": source_use.value,
        "input_path": str(input_path),
        "output_dir": str(output_path),
        "source_manifest": manifest,
        "record_count": 0,
        "converted_count": 0,
        "artifacts": [],
        "warnings": [],
        "errors": [],
    }
    if not manifest["use_verdict"]["ok"]:
        report["errors"].extend(manifest["use_verdict"]["blocking_codes"])
        return report

    loaded_records = _load_records(input_path, limit=limit)
    report["record_count"] = len(loaded_records)
    records, mapping_report = _normalize_source_records(card, loaded_records, goal=source_use)
    report["source_schema_mapping"] = mapping_report
    if mapping_report.get("dropped_records", 0):
        report["warnings"].append("source_schema_mapping_dropped_records")
    input_source = Path(input_path)

    def add_artifact(
        *,
        artifact_type: str,
        path: Path,
        records_written: int,
        validation: dict[str, Any] | None = None,
    ) -> None:
        report["converted_count"] += records_written
        item: dict[str, Any] = {
            "artifact_type": artifact_type,
            "path": str(path),
            "record_count": records_written,
        }
        if validation is not None:
            item["validation"] = validation
            if not validation.get("ok", False):
                report["ok"] = False
        report["artifacts"].append(item)

    if source_use == SourceUse.SFT:
        examples = _to_sft_examples(card, records, manifest=manifest)
        artifact_path = output_path / "training_examples.jsonl"
        _write_jsonl(artifact_path, examples)
        add_artifact(
            artifact_type="sft_examples", path=artifact_path, records_written=len(examples)
        )
    elif source_use == SourceUse.DPO:
        pairs = _to_dpo_pairs(card, records, manifest=manifest)
        artifact_path = output_path / "dpo_pairs.jsonl"
        _write_jsonl(artifact_path, pairs)
        validation = validate_preference_pair_records(pairs, strict=True)
        add_artifact(
            artifact_type="dpo_pairs",
            path=artifact_path,
            records_written=len(pairs),
            validation=validation,
        )
    elif source_use == SourceUse.REWARD_MODEL:
        examples = _to_reward_examples(card, records, manifest=manifest)
        artifact_path = output_path / "reward_examples.jsonl"
        _write_jsonl(artifact_path, examples)
        validation = validate_reward_example_records(examples, strict=True)
        add_artifact(
            artifact_type="reward_examples",
            path=artifact_path,
            records_written=len(examples),
            validation=validation,
        )
    elif source_use == SourceUse.PROCESS_REWARD:
        examples = _to_reward_examples(card, records, manifest=manifest, process=True)
        artifact_path = output_path / "process_reward_examples.jsonl"
        _write_jsonl(artifact_path, examples)
        validation = validate_reward_example_records(examples, strict=True)
        add_artifact(
            artifact_type="process_reward_examples",
            path=artifact_path,
            records_written=len(examples),
            validation=validation,
        )
    elif source_use == SourceUse.TERMINAL_RL:
        specs, validation = _to_environment_specs(card, records, manifest=manifest)
        artifact_path = output_path / "environment_specs.jsonl"
        _write_jsonl(artifact_path, specs)
        add_artifact(
            artifact_type="environment_specs",
            path=artifact_path,
            records_written=len(specs),
            validation=validation,
        )
    elif source_use == SourceUse.EVALUATION:
        manifest_payload = _eval_manifest(card, records, input_path=input_source)
        artifact_path = output_path / "eval_manifest.json"
        artifact_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        add_artifact(
            artifact_type="eval_manifest",
            path=artifact_path,
            records_written=len(records),
        )
        if SourceArtifactType.ENVIRONMENT_SPECS in card.artifact_types:
            specs, validation = _to_environment_specs(card, records, manifest=manifest)
            env_path = output_path / "environment_specs.jsonl"
            _write_jsonl(env_path, specs)
            add_artifact(
                artifact_type="environment_specs",
                path=env_path,
                records_written=len(specs),
                validation=validation,
            )
    else:
        report["ok"] = False
        report["errors"].append(f"local adapter for {source_use.value!r} is not implemented yet")

    if report["converted_count"] == 0:
        report["ok"] = False
        report["errors"].append("no source records converted for requested goal")

    report_path = output_path / "source_adapter_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
