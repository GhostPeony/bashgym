"""Dependency-free reward-model fixture training.

This is not a replacement for TRL/OpenRLHF reward-model training. It is a tiny
contract smoke: given strict reward examples, learn a deterministic bag-of-words
scorer, emit predictions, and produce the same reward-eval artifact serious
runs must attach.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bashgym.sources import get_source

from .reward_evaluation import evaluate_reward_model_records
from .reward_validation import validate_reward_example_records

REWARD_MODEL_FIXTURE_SCHEMA_VERSION = "bashgym.reward_model_fixture_train.v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


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
        raise ValueError("JSON reward artifact must be a list or contain examples/records/data")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        records.append(payload)
    return records


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


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


def _split(record: dict[str, Any]) -> str:
    metadata = _metadata(record)
    value = _first(record, metadata, ("split", "split_id", "dataset_split"))
    return str(value).strip().lower() if value is not None else ""


def _include_split(record: dict[str, Any], split: str) -> bool:
    split = split.lower().strip()
    if split == "all":
        return True
    record_split = _split(record)
    if split == "eval":
        return record_split in {"eval", "evaluation", "valid", "validation", "test", "heldout"}
    return record_split == split


def _source_ids(record: dict[str, Any]) -> list[str]:
    metadata = _metadata(record)
    raw = _first(record, metadata, ("source_id", "source_ids", "sources"))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item not in (None, "")]
    return []


def _eval_only_training_sources(records: list[dict[str, Any]]) -> list[str]:
    leaked: set[str] = set()
    for record in records:
        metadata = _metadata(record)
        if bool(_first(record, metadata, ("eval_only", "benchmark_eval_only"))):
            leaked.update(_source_ids(record) or ["unknown_eval_only_source"])
        source_use = str(_first(record, metadata, ("source_use", "use", "goal")) or "").lower()
        if source_use in {"eval", "evaluation", "benchmark"}:
            leaked.update(_source_ids(record) or ["unknown_eval_only_source"])
        for source_id in _source_ids(record):
            try:
                card = get_source(source_id)
            except KeyError:
                continue
            if card.eval_only:
                leaked.add(source_id)
    return sorted(leaked)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        labels = {
            "chosen": 1.0,
            "preferred": 1.0,
            "win": 1.0,
            "success": 1.0,
            "passed": 1.0,
            "positive": 1.0,
            "good": 1.0,
            "rejected": 0.0,
            "loser": 0.0,
            "failure": 0.0,
            "failed": 0.0,
            "negative": 0.0,
            "bad": 0.0,
        }
        if normalized in labels:
            return labels[normalized]
        try:
            parsed = float(normalized)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _target(record: dict[str, Any]) -> float | None:
    metadata = _metadata(record)
    value = _first(
        record,
        metadata,
        ("reward", "score", "rating", "label", "target", "preference_score"),
    )
    numeric = _coerce_float(value)
    if numeric is not None:
        return max(0.0, min(1.0, numeric))

    for key in ("step_rewards", "process_rewards", "step_scores"):
        raw_steps = _first(record, metadata, (key,))
        if isinstance(raw_steps, list):
            values = [_coerce_float(item) for item in raw_steps]
            numeric_values = [item for item in values if item is not None]
            if numeric_values:
                mean = sum(numeric_values) / len(numeric_values)
                return max(0.0, min(1.0, mean))
    return None


def _text(record: dict[str, Any]) -> str:
    metadata = _metadata(record)
    chunks: list[str] = []
    for key in (
        "prompt",
        "input",
        "question",
        "instruction",
        "response",
        "completion",
        "answer",
        "chosen",
        "text",
        "trajectory",
        "messages",
        "steps",
        "trace",
        "rollout",
    ):
        value = _first(record, metadata, (key,))
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            chunks.append(value)
        else:
            chunks.append(json.dumps(value, sort_keys=True))
    return "\n".join(chunks)


def _features(record: dict[str, Any], vocabulary: set[str] | None = None) -> dict[str, float]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(_text(record))]
    counts = Counter(tokens)
    if vocabulary is not None:
        counts = Counter({token: count for token, count in counts.items() if token in vocabulary})
    total = sum(counts.values()) or 1
    return {token: count / total for token, count in counts.items()}


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _predict(weights: dict[str, float], bias: float, features: dict[str, float]) -> float:
    score = bias + sum(weights.get(token, 0.0) * value for token, value in features.items())
    return _sigmoid(score)


def _loss(prediction: float, target: float) -> float:
    prediction = max(1e-6, min(1.0 - 1e-6, prediction))
    return -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))


def _mean_loss(
    records: list[dict[str, Any]],
    targets: list[float],
    weights: dict[str, float],
    bias: float,
    vocabulary: set[str],
) -> float:
    losses = [
        _loss(_predict(weights, bias, _features(record, vocabulary)), target)
        for record, target in zip(records, targets, strict=True)
    ]
    return sum(losses) / len(losses) if losses else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def train_reward_model_fixture_file(
    path: str | Path,
    *,
    output_dir: str | Path,
    train_split: str = "train",
    eval_split: str = "eval",
    epochs: int = 8,
    learning_rate: float = 0.5,
    l2: float = 0.001,
    max_features: int = 2048,
    calibration_bins: int = 10,
    strict: bool = True,
) -> dict[str, Any]:
    """Train a tiny reward scorer and write fixture evidence artifacts."""

    records = _load_records(path)
    validation = validate_reward_example_records(records, strict=strict)
    findings: list[dict[str, Any]] = []
    if not validation["ok"]:
        findings.append(
            {
                "code": "invalid_reward_examples",
                "level": "fail",
                "message": "reward examples failed validation",
            }
        )

    train_records = [record for record in records if _include_split(record, train_split)]
    eval_records = [record for record in records if _include_split(record, eval_split)]
    if not train_records:
        findings.append(
            {
                "code": "missing_train_split",
                "level": "fail",
                "message": f"no reward examples matched train split {train_split!r}",
            }
        )
    if not eval_records:
        findings.append(
            {
                "code": "missing_eval_split",
                "level": "fail",
                "message": f"no reward examples matched eval split {eval_split!r}",
            }
        )

    eval_only_sources = _eval_only_training_sources(train_records)
    if eval_only_sources:
        findings.append(
            {
                "code": "eval_only_source_in_training_split",
                "level": "fail",
                "message": "eval-only sources cannot be used by the reward-model fixture trainer",
                "source_ids": eval_only_sources,
            }
        )

    train_targets = [_target(record) for record in train_records]
    eval_targets = [_target(record) for record in eval_records]
    if any(target is None for target in train_targets):
        findings.append(
            {
                "code": "missing_train_targets",
                "level": "fail",
                "message": "all training records need numeric reward/score/label targets",
            }
        )
    if any(target is None for target in eval_targets):
        findings.append(
            {
                "code": "missing_eval_targets",
                "level": "fail",
                "message": "all eval records need numeric reward/score/label targets",
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if any(finding["level"] == "fail" for finding in findings):
        report = {
            "schema_version": REWARD_MODEL_FIXTURE_SCHEMA_VERSION,
            "ok": False,
            "path": str(path),
            "output_dir": str(output_path),
            "train_split": train_split,
            "eval_split": eval_split,
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "validation": validation,
            "findings": findings,
        }
        _write_json(output_path / "reward_model_fixture_report.json", report)
        return report

    typed_train_targets = [float(target) for target in train_targets if target is not None]
    typed_eval_targets = [float(target) for target in eval_targets if target is not None]
    token_counts: Counter[str] = Counter()
    for record in train_records:
        token_counts.update(_features(record).keys())
    vocabulary = {token for token, _ in token_counts.most_common(max(1, int(max_features)))}
    weights = {token: 0.0 for token in vocabulary}
    mean_target = sum(typed_train_targets) / len(typed_train_targets)
    bias = math.log((mean_target + 1e-3) / (1.0 - mean_target + 1e-3))

    metric_rows: list[dict[str, Any]] = []
    epochs = max(1, int(epochs))
    learning_rate = max(1e-6, float(learning_rate))
    for epoch in range(1, epochs + 1):
        for record, target in zip(train_records, typed_train_targets, strict=True):
            feats = _features(record, vocabulary)
            prediction = _predict(weights, bias, feats)
            error = prediction - target
            bias -= learning_rate * error
            for token, value in feats.items():
                weights[token] -= learning_rate * (error * value + l2 * weights[token])
        metric_rows.append(
            {
                "step": epoch,
                "train_loss": _mean_loss(
                    train_records, typed_train_targets, weights, bias, vocabulary
                ),
                "eval_loss": _mean_loss(
                    eval_records, typed_eval_targets, weights, bias, vocabulary
                ),
                "learning_rate": learning_rate,
                "train_records": len(train_records),
                "eval_records": len(eval_records),
            }
        )

    predicted_eval_records: list[dict[str, Any]] = []
    for record in eval_records:
        predicted = _predict(weights, bias, _features(record, vocabulary))
        enriched = json.loads(json.dumps(record))
        enriched["predicted_reward"] = predicted
        metadata = enriched.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["reward_model_fixture"] = True
            metadata["reward_model_fixture_schema_version"] = REWARD_MODEL_FIXTURE_SCHEMA_VERSION
        predicted_eval_records.append(enriched)

    reward_eval = evaluate_reward_model_records(
        predicted_eval_records,
        split=eval_split,
        calibration_bins=calibration_bins,
    )
    artifact_paths = {
        "model_path": str(output_path / "reward_model_fixture.json"),
        "metrics_path": str(output_path / "metrics.jsonl"),
        "predictions_path": str(output_path / "reward_predictions.jsonl"),
        "reward_eval_path": str(output_path / "reward_eval.json"),
        "report_path": str(output_path / "reward_model_fixture_report.json"),
    }
    model_artifact = {
        "schema_version": REWARD_MODEL_FIXTURE_SCHEMA_VERSION,
        "model_type": "bag_of_words_logistic_reward_fixture",
        "source_path": str(path),
        "train_split": train_split,
        "eval_split": eval_split,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "max_features": max_features,
        "bias": bias,
        "weights": dict(sorted(weights.items())),
        "vocabulary_size": len(vocabulary),
    }
    report = {
        "schema_version": REWARD_MODEL_FIXTURE_SCHEMA_VERSION,
        "ok": bool(reward_eval["ok"]),
        "path": str(path),
        "output_dir": str(output_path),
        "train_split": train_split,
        "eval_split": eval_split,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "validation": validation,
        "reward_eval": reward_eval,
        "metrics": metric_rows[-1] if metric_rows else {},
        "artifacts": artifact_paths,
        "findings": findings + reward_eval.get("findings", []),
    }

    _write_json(Path(artifact_paths["model_path"]), model_artifact)
    _write_jsonl(Path(artifact_paths["predictions_path"]), predicted_eval_records)
    _write_json(Path(artifact_paths["reward_eval_path"]), reward_eval)
    _write_json(Path(artifact_paths["report_path"]), report)
    Path(artifact_paths["metrics_path"]).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    return report
