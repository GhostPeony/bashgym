"""Reward-model evaluation metrics for BashGym artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from bashgym.sources import get_source

REWARD_MODEL_EVAL_SCHEMA_VERSION = "bashgym.reward_model_eval.v1"


def _finding(
    code: str,
    level: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    payload = {"code": code, "level": level, "message": message}
    payload.update(
        {key: value for key, value in details.items() if value not in (None, "", [], {})}
    )
    return payload


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def _first(record: dict[str, Any], metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _text(record: dict[str, Any], metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first(record, metadata, keys)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        label_values = {
            "chosen": 1.0,
            "preferred": 1.0,
            "win": 1.0,
            "winner": 1.0,
            "success": 1.0,
            "passed": 1.0,
            "positive": 1.0,
            "good": 1.0,
            "rejected": 0.0,
            "not_preferred": 0.0,
            "loss": 0.0,
            "loser": 0.0,
            "failure": 0.0,
            "failed": 0.0,
            "negative": 0.0,
            "bad": 0.0,
        }
        if normalized in label_values:
            return label_values[normalized]
        try:
            result = float(normalized)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _source_ids(record: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    raw = _first(record, metadata, ("source_id", "source_ids", "sources"))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item not in (None, "")]
    return []


def _split(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = _first(record, metadata, ("split", "split_id", "dataset_split"))
    return str(value).strip().lower() if value is not None else ""


def _include_split(record: dict[str, Any], split: str) -> bool:
    if split == "all":
        return True
    metadata = _metadata(record)
    record_split = _split(record, metadata)
    if split == "eval":
        return record_split in {"eval", "evaluation", "valid", "validation", "test", "heldout"}
    return record_split == split


def _group_key(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = _first(
        record,
        metadata,
        ("pair_id", "preference_group_id", "group_id", "prompt_hash", "prompt_id"),
    )
    if value not in (None, ""):
        return str(value)
    prompt = _text(record, metadata, ("prompt", "input", "question", "instruction"))
    return f"prompt:{prompt}" if prompt else ""


def _task_family(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    value = _first(record, metadata, ("task_family", "domain"))
    return str(value) if value not in (None, "") else "unknown"


def _response_length(record: dict[str, Any], metadata: dict[str, Any]) -> int:
    text = _text(
        record,
        metadata,
        ("response", "completion", "answer", "chosen", "text", "trajectory", "messages", "steps"),
    )
    return len(text)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("examples", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        raise ValueError(
            "JSON reward eval artifact must be a list or contain examples/records/data"
        )

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        records.append(payload)
    return records


def _variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = _metadata(record)
        key = _group_key(record, metadata)
        if key:
            groups[key].append(record)

    pair_count = 0
    correct = 0.0
    margins: list[float] = []
    for group_records in groups.values():
        if len(group_records) < 2:
            continue
        scored: list[tuple[float, float]] = []
        for record in group_records:
            metadata = _metadata(record)
            true_reward = _coerce_float(
                _first(
                    record,
                    metadata,
                    ("reward", "score", "rating", "label", "target", "preference_score"),
                )
            )
            predicted = _coerce_float(
                _first(
                    record,
                    metadata,
                    (
                        "predicted_reward",
                        "predicted_score",
                        "model_score",
                        "reward_model_score",
                        "prediction",
                    ),
                )
            )
            if true_reward is not None and predicted is not None:
                scored.append((true_reward, predicted))
        for left_index, left in enumerate(scored):
            for right in scored[left_index + 1 :]:
                true_delta = left[0] - right[0]
                if true_delta == 0:
                    continue
                pred_delta = left[1] - right[1]
                pair_count += 1
                if pred_delta == 0:
                    correct += 0.5
                elif (pred_delta > 0) == (true_delta > 0):
                    correct += 1.0
                margins.append(pred_delta if true_delta > 0 else -pred_delta)

    return {
        "pair_count": pair_count,
        "heldout_pair_accuracy": correct / pair_count if pair_count else None,
        "reward_margin": sum(margins) / len(margins) if margins else None,
    }


def _calibration_error(
    true_rewards: list[float], predictions: list[float], bins: int
) -> float | None:
    paired = [
        (true_reward, prediction)
        for true_reward, prediction in zip(true_rewards, predictions, strict=True)
        if 0.0 <= true_reward <= 1.0
    ]
    if not paired:
        return None
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for true_reward, prediction in paired:
        clamped = max(0.0, min(1.0, prediction))
        index = min(bins - 1, int(clamped * bins))
        buckets[index].append((true_reward, clamped))
    total = len(paired)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_true = sum(item[0] for item in bucket) / len(bucket)
        mean_pred = sum(item[1] for item in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_pred - mean_true)
    return error


def _eval_only_leakage(records: list[dict[str, Any]]) -> dict[str, Any]:
    leaked_sources: set[str] = set()
    leaked_records: list[str] = []
    for index, record in enumerate(records):
        metadata = _metadata(record)
        explicit_eval_only = bool(_first(record, metadata, ("eval_only", "benchmark_eval_only")))
        source_use = str(_first(record, metadata, ("source_use", "use", "goal")) or "").lower()
        source_ids = _source_ids(record, metadata)
        source_eval_only = False
        for source_id in source_ids:
            try:
                card = get_source(source_id)
            except KeyError:
                continue
            if card.eval_only:
                source_eval_only = True
                leaked_sources.add(source_id)
        if (
            explicit_eval_only
            or source_use in {"eval", "evaluation", "benchmark"}
            or source_eval_only
        ):
            example_id = str(
                _first(record, metadata, ("reward_example_id", "example_id", "id")) or index
            )
            leaked_records.append(example_id)
            leaked_sources.update(source_ids)
    return {
        "eval_only_leakage_count": len(leaked_records),
        "eval_only_leakage_rate": len(leaked_records) / len(records) if records else 0.0,
        "eval_only_source_ids": sorted(leaked_sources),
        "eval_only_record_ids": leaked_records,
    }


def _task_family_breakdown(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[_task_family(record, _metadata(record))].append(record)

    rows: list[dict[str, Any]] = []
    for family, family_records in sorted(by_family.items()):
        true_rewards: list[float] = []
        predictions: list[float] = []
        for record in family_records:
            metadata = _metadata(record)
            true_reward = _coerce_float(
                _first(
                    record,
                    metadata,
                    ("reward", "score", "rating", "label", "target", "preference_score"),
                )
            )
            predicted = _coerce_float(
                _first(
                    record,
                    metadata,
                    (
                        "predicted_reward",
                        "predicted_score",
                        "model_score",
                        "reward_model_score",
                        "prediction",
                    ),
                )
            )
            if true_reward is not None:
                true_rewards.append(true_reward)
            if predicted is not None:
                predictions.append(predicted)
        pair_metrics = _pair_metrics(family_records)
        rows.append(
            {
                "task_family": family,
                "records": len(family_records),
                "predicted_records": len(predictions),
                "heldout_pair_accuracy": pair_metrics["heldout_pair_accuracy"],
                "mean_true_reward": sum(true_rewards) / len(true_rewards) if true_rewards else None,
                "mean_predicted_reward": (
                    sum(predictions) / len(predictions) if predictions else None
                ),
                "reward_variance": _variance(predictions),
            }
        )
    return rows


def evaluate_reward_model_records(
    records: list[dict[str, Any]],
    *,
    split: str = "eval",
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate reward-model predictions embedded in reward example records."""

    split = split.lower().strip() or "eval"
    selected = [record for record in records if _include_split(record, split)]
    findings: list[dict[str, Any]] = []
    true_rewards: list[float] = []
    predictions: list[float] = []
    paired_true_rewards: list[float] = []
    paired_predictions: list[float] = []
    response_lengths: list[float] = []

    for record in selected:
        metadata = _metadata(record)
        true_reward = _coerce_float(
            _first(
                record,
                metadata,
                ("reward", "score", "rating", "label", "target", "preference_score"),
            )
        )
        predicted = _coerce_float(
            _first(
                record,
                metadata,
                (
                    "predicted_reward",
                    "predicted_score",
                    "model_score",
                    "reward_model_score",
                    "prediction",
                ),
            )
        )
        if true_reward is not None:
            true_rewards.append(true_reward)
        if predicted is not None:
            predictions.append(predicted)
            response_lengths.append(float(_response_length(record, metadata)))
        if true_reward is not None and predicted is not None:
            paired_true_rewards.append(true_reward)
            paired_predictions.append(predicted)

    if not selected:
        findings.append(
            _finding(
                "no_records_for_split",
                "fail",
                f"no reward examples matched split {split!r}",
            )
        )
    if selected and not predictions:
        findings.append(
            _finding(
                "missing_reward_model_predictions",
                "fail",
                "records need predicted_reward, predicted_score, model_score, reward_model_score, or prediction",
            )
        )

    pair_metrics = _pair_metrics(selected)
    if selected and pair_metrics["pair_count"] == 0:
        findings.append(
            _finding(
                "no_pairwise_comparisons",
                "warn",
                "heldout pair accuracy needs at least two scored records per prompt/pair group",
            )
        )

    calibration_error = _calibration_error(
        paired_true_rewards, paired_predictions, calibration_bins
    )
    if selected and calibration_error is None:
        findings.append(
            _finding(
                "calibration_unavailable",
                "warn",
                "calibration requires numeric true rewards on a 0..1 scale and predictions",
            )
        )

    leakage = _eval_only_leakage(selected)
    if leakage["eval_only_leakage_count"]:
        findings.append(
            _finding(
                "eval_only_leakage",
                "fail",
                "eval-only sources or benchmark records appear in the reward-model eval artifact",
                source_ids=leakage["eval_only_source_ids"],
                record_ids=leakage["eval_only_record_ids"],
            )
        )

    reward_variance = _variance(predictions)
    if predictions and (reward_variance is None or reward_variance == 0.0):
        findings.append(
            _finding(
                "zero_reward_variance",
                "warn",
                "reward-model predictions have no variance",
            )
        )

    fail_count = sum(1 for finding in findings if finding["level"] == "fail")
    warn_count = sum(1 for finding in findings if finding["level"] == "warn")
    metrics = {
        "heldout_pair_accuracy": pair_metrics["heldout_pair_accuracy"],
        "pair_count": pair_metrics["pair_count"],
        "calibration_error": calibration_error,
        "reward_margin": pair_metrics["reward_margin"],
        "length_bias": _pearson(response_lengths, predictions),
        "reward_variance": reward_variance,
        "eval_only_leakage_count": leakage["eval_only_leakage_count"],
        "eval_only_leakage_rate": leakage["eval_only_leakage_rate"],
    }
    return {
        "schema_version": REWARD_MODEL_EVAL_SCHEMA_VERSION,
        "ok": fail_count == 0,
        "split": split,
        "total_records": len(records),
        "evaluated_records": len(selected),
        "prediction_records": len(predictions),
        "metrics": metrics,
        "task_family_breakdown": _task_family_breakdown(selected),
        "eval_only_source_ids": leakage["eval_only_source_ids"],
        "findings": findings,
        "fail_count": fail_count,
        "warn_count": warn_count,
    }


def evaluate_reward_model_file(
    path: str | Path,
    *,
    split: str = "eval",
    calibration_bins: int = 10,
) -> dict[str, Any]:
    records = _load_records(path)
    result = evaluate_reward_model_records(
        records,
        split=split,
        calibration_bins=calibration_bins,
    )
    result["path"] = str(path)
    return result
