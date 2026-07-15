"""Hugging Face source normalization for context packs.

Network clients stay thin; these pure normalizers are the canonical boundary so
Hub response drift degrades individual fields instead of corrupting bundles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .context_contracts import (
    Comparability,
    EvalSettings,
    EvidenceAssessment,
    EvidenceKind,
    EvidenceRecord,
    Provenance,
    Visibility,
    canonical_hash,
    utc_now,
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _card_data(info: Any) -> dict[str, Any]:
    return _mapping(_get(info, "card_data") or _get(info, "cardData"))


def _visibility(info: Any) -> Visibility:
    gated = _get(info, "gated", False)
    return (
        Visibility.WORKSPACE_PRIVATE
        if bool(_get(info, "private", False)) or gated not in (False, None, "false")
        else Visibility.PUBLIC
    )


def _license(card: Mapping[str, Any], tags: list[str]) -> str | None:
    value = card.get("license")
    if isinstance(value, list):
        value = value[0] if value else None
    if value:
        return str(value)
    return next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), None)


def _task_relevance(intent: str | None, haystack: list[str]) -> int:
    if not intent:
        return 0
    normalized_haystack = " ".join(haystack).lower().replace("-", "_").replace("/", "_")
    words = [word for word in intent.lower().replace("-", " ").split() if len(word) > 2]
    if words and all(word in normalized_haystack for word in words):
        return 3
    matches = sum(word in normalized_haystack for word in words)
    if matches >= 2:
        return 2
    return 1 if matches else 0


def _evidence_id(kind: EvidenceKind, resource_id: str, suffix: str = "") -> str:
    fingerprint = canonical_hash({"kind": kind.value, "id": resource_id, "suffix": suffix})[:20]
    return f"hf_{kind.value}_{fingerprint}"


def _provenance(kind: EvidenceKind, resource_id: str, revision: str | None) -> Provenance:
    if kind is EvidenceKind.DATASET:
        url = f"https://huggingface.co/datasets/{resource_id}"
    else:
        url = f"https://huggingface.co/{resource_id}"
    return Provenance(
        source="huggingface_hub",
        source_url=url,
        source_revision=revision,
        retrieved_at=utc_now(),
    )


def normalize_model(info: Any, *, intent: str | None = None) -> EvidenceRecord:
    resource_id = str(_get(info, "id") or _get(info, "modelId") or "")
    if not resource_id:
        raise ValueError("model record requires an id")
    revision = _get(info, "sha")
    tags = list(_get(info, "tags", []) or [])
    card = _card_data(info)
    config = _mapping(_get(info, "config"))
    safetensors = _mapping(_get(info, "safetensors"))
    parameters = _mapping(safetensors.get("parameters"))
    total_params = safetensors.get("total")
    if not isinstance(total_params, int):
        total_params = sum(int(value) for value in parameters.values() if isinstance(value, int))
    dominant_dtype = max(parameters, key=parameters.get) if parameters else None
    tokenizer_config = _mapping(config.get("tokenizer_config"))
    chat_template = tokenizer_config.get("chat_template")
    pipeline_tag = _get(info, "pipeline_tag") or card.get("pipeline_tag")
    license_name = _license(card, tags)
    base_model = card.get("base_model")

    present = [revision, pipeline_tag, license_name, base_model, total_params or None, config]
    confidence = sum(value is not None and value != {} for value in present) / len(present)
    facts = {
        "author": _get(info, "author"),
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "license": license_name,
        "base_model": base_model,
        "library_name": _get(info, "library_name") or card.get("library_name"),
        "parameter_count": total_params or None,
        "dominant_dtype": dominant_dtype,
        "architectures": config.get("architectures") or [],
        "model_type": config.get("model_type"),
        "chat_template_present": bool(chat_template),
        "chat_template_hash": canonical_hash(chat_template) if chat_template else None,
        "downloads": int(_get(info, "downloads", 0) or 0),
        "likes": int(_get(info, "likes", 0) or 0),
        "gated": _get(info, "gated", False),
        "private": bool(_get(info, "private", False)),
    }
    facts = {key: value for key, value in facts.items() if value is not None}
    relevance = _task_relevance(intent, [resource_id, str(pipeline_tag or ""), *tags])
    canonical_url = f"https://huggingface.co/{resource_id}"
    return EvidenceRecord(
        evidence_id=_evidence_id(EvidenceKind.MODEL, resource_id),
        kind=EvidenceKind.MODEL,
        resource_id=resource_id,
        revision=str(revision) if revision else None,
        canonical_url=canonical_url,
        summary=f"Hugging Face model for {pipeline_tag or 'an unspecified task'}.",
        facts=facts,
        visibility=_visibility(info),
        provenance=_provenance(EvidenceKind.MODEL, resource_id, revision),
        assessment=EvidenceAssessment(
            task_relevance=relevance,
            compatibility=1,
            confidence=round(confidence, 2),
            rationale=("Matches the requested task metadata." if relevance else None),
        ),
        cautions=(("Gated or private resource; access must be rechecked.",) if _visibility(info) is Visibility.WORKSPACE_PRIVATE else ()),
    )


def _eval_result_values(info: Any) -> list[Any]:
    card_object = _get(info, "card_data") or _get(info, "cardData")
    direct = _get(card_object, "eval_results") if card_object is not None else None
    if direct:
        return list(direct)
    card = _card_data(info)
    return list(card.get("eval_results") or [])


def normalize_model_card_evals(info: Any) -> list[EvidenceRecord]:
    resource_id = str(_get(info, "id") or _get(info, "modelId") or "")
    if not resource_id:
        raise ValueError("model record requires an id")
    revision = _get(info, "sha")
    output: list[EvidenceRecord] = []
    for index, raw in enumerate(_eval_result_values(info)):
        result = _mapping(raw)
        dataset_args = _mapping(result.get("dataset_args"))
        raw_score = result.get("metric_value")
        score = raw_score
        if isinstance(score, (int, float)) and 1 < score <= 100:
            score = score / 100
        benchmark_id = str(
            result.get("dataset_type")
            or result.get("dataset_name")
            or result.get("task_type")
            or "unknown"
        )
        metric = str(result.get("metric_type") or result.get("metric_name") or "unknown")
        source_url = result.get("source_url")
        if not isinstance(source_url, str) or not source_url.startswith("https://"):
            source_url = f"https://huggingface.co/{resource_id}"
        settings = EvalSettings(
            benchmark_id=benchmark_id,
            task_revision=result.get("dataset_revision"),
            metric=metric,
            prompt_template=None,
            few_shot=dataset_args.get("num_few_shot"),
            harness=None,
            harness_version=None,
            backend=None,
            sampling=None,
        )
        suffix = f"{benchmark_id}:{metric}:{index}"
        output.append(
            EvidenceRecord(
                evidence_id=_evidence_id(EvidenceKind.EVALUATION, resource_id, suffix),
                kind=EvidenceKind.EVALUATION,
                resource_id=f"{resource_id}#{benchmark_id}",
                revision=str(revision) if revision else None,
                canonical_url=source_url,
                summary=f"Published {metric} result for {benchmark_id}.",
                facts={
                    "model_id": resource_id,
                    "task_type": result.get("task_type"),
                    "task_name": result.get("task_name"),
                    "dataset_type": result.get("dataset_type"),
                    "dataset_name": result.get("dataset_name"),
                    "dataset_config": result.get("dataset_config"),
                    "dataset_split": result.get("dataset_split"),
                    "raw_score": raw_score,
                    "score": score,
                    "metric": metric,
                    "verified": result.get("verified"),
                    "source_name": result.get("source_name"),
                },
                visibility=_visibility(info),
                provenance=Provenance(
                    source="huggingface_model_card",
                    source_url=source_url,
                    source_revision=str(revision) if revision else None,
                ),
                assessment=EvidenceAssessment(
                    comparability=Comparability.ORIENTATION_ONLY,
                    confidence=0.5,
                    rationale="Published score is useful for orientation, not an apples-to-apples comparison.",
                ),
                eval_settings=settings,
                cautions=(
                    "Prompt template, harness version, backend, or sampling settings are missing; treat this score as orientation-only.",
                ),
            )
        )
    return output


def _dataset_configs(card: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_info = card.get("dataset_info")
    values = raw_info if isinstance(raw_info, list) else [raw_info] if isinstance(raw_info, Mapping) else []
    configs: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        info = _mapping(raw)
        features: list[dict[str, Any]] = []
        for feature in info.get("features") or []:
            value = _mapping(feature)
            if value.get("name"):
                features.append(
                    {
                        "name": value["name"],
                        "type": value.get("dtype")
                        or ("list" if "list" in value else "sequence" if "sequence" in value else "unknown"),
                    }
                )
        splits = []
        for split in info.get("splits") or []:
            value = _mapping(split)
            if value.get("name"):
                splits.append(
                    {
                        "name": value["name"],
                        "num_examples": value.get("num_examples"),
                        "num_bytes": value.get("num_bytes"),
                    }
                )
        configs.append(
            {
                "name": info.get("config_name") or ("default" if index == 0 else f"config-{index + 1}"),
                "features": features,
                "splits": splits,
                "download_size": info.get("download_size"),
                "dataset_size": info.get("dataset_size"),
            }
        )
    return configs


def normalize_dataset(info: Any, *, intent: str | None = None) -> EvidenceRecord:
    resource_id = str(_get(info, "id") or "")
    if not resource_id:
        raise ValueError("dataset record requires an id")
    revision = _get(info, "sha")
    tags = list(_get(info, "tags", []) or [])
    card = _card_data(info)
    configs = _dataset_configs(card)
    total_rows = sum(
        int(split["num_examples"])
        for config in configs
        for split in config["splits"]
        if isinstance(split.get("num_examples"), int)
    )
    license_name = _license(card, tags)
    present = [revision, license_name, configs or None, total_rows or None]
    confidence = sum(value is not None for value in present) / len(present)
    relevance = _task_relevance(intent, [resource_id, *tags])
    canonical_url = f"https://huggingface.co/datasets/{resource_id}"
    return EvidenceRecord(
        evidence_id=_evidence_id(EvidenceKind.DATASET, resource_id),
        kind=EvidenceKind.DATASET,
        resource_id=resource_id,
        revision=str(revision) if revision else None,
        canonical_url=canonical_url,
        summary="Hugging Face dataset with normalized schema and split metadata.",
        facts={
            "tags": tags,
            "license": license_name,
            "configs": configs,
            "total_rows": total_rows,
            "downloads": int(_get(info, "downloads", 0) or 0),
            "likes": int(_get(info, "likes", 0) or 0),
            "gated": _get(info, "gated", False),
            "private": bool(_get(info, "private", False)),
        },
        visibility=_visibility(info),
        provenance=_provenance(EvidenceKind.DATASET, resource_id, revision),
        assessment=EvidenceAssessment(
            task_relevance=relevance,
            compatibility=1,
            confidence=round(confidence, 2),
            rationale=("Matches the requested dataset metadata." if relevance else None),
        ),
        cautions=(("Gated or private resource; examples are not persisted in release one.",) if _visibility(info) is Visibility.WORKSPACE_PRIVATE else ()),
    )


class HFContextSourceClient:
    """Bounded network adapter over ``huggingface_hub.HfApi``."""

    def __init__(self, *, api: Any | None = None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api

    def discover_models(self, query: str, *, limit: int = 3) -> list[Any]:
        listed = list(self.api.list_models(search=query, sort="downloads", limit=max(limit, 3)))
        output: list[Any] = []
        for candidate in listed[: min(limit, 3)]:
            resource_id = _get(candidate, "id")
            if not resource_id:
                continue
            try:
                output.append(
                    self.api.model_info(
                        resource_id,
                        expand=[
                            "author",
                            "cardData",
                            "config",
                            "downloads",
                            "gated",
                            "lastModified",
                            "likes",
                            "model-index",
                            "pipeline_tag",
                            "private",
                            "safetensors",
                            "sha",
                            "tags",
                        ],
                    )
                )
            except Exception:  # noqa: BLE001 - one bad Hub record must not collapse discovery
                output.append(candidate)
        return output

    def discover_datasets(self, query: str, *, limit: int = 3) -> list[Any]:
        listed = list(self.api.list_datasets(search=query, sort="downloads", limit=max(limit, 3)))
        output: list[Any] = []
        for candidate in listed[: min(limit, 3)]:
            resource_id = _get(candidate, "id")
            if not resource_id:
                continue
            try:
                output.append(
                    self.api.dataset_info(
                        resource_id,
                        expand=[
                            "author",
                            "cardData",
                            "downloads",
                            "gated",
                            "lastModified",
                            "likes",
                            "private",
                            "sha",
                            "tags",
                        ],
                    )
                )
            except Exception:  # noqa: BLE001 - preserve partial discovery
                output.append(candidate)
        return output


__all__ = [
    "HFContextSourceClient",
    "normalize_dataset",
    "normalize_model",
    "normalize_model_card_evals",
]
