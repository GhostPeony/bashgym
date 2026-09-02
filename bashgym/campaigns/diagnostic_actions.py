"""Agent-authored, budget-enveloped diagnostic contracts for AutoResearch."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    FrozenContractModel,
    HexDigest,
    Identifier,
    ResourceUsage,
    canonical_hash,
)

AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME = "autoresearch_diagnostic_request.json"
AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME = "autoresearch_diagnostic.json"
AUTORESEARCH_DIAGNOSTIC_RECIPE_SCHEMA = "bashgym.autoresearch_diagnostic_recipe.v1"
AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA = "bashgym.autoresearch_diagnostic_evidence.v1"
AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN = "bashgym.autoresearch.normalized-diagnostic.v1"
MAX_AUTORESEARCH_DIAGNOSTIC_BYTES = 1024 * 1024

_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|file|ssh)://"),
    re.compile(r"(?i)(?:[a-z]:\\|/(?:users|home|var|tmp|etc)/)"),
    re.compile(r"(?i)\b(?:api[_-]?(?:key|token)|password|secret|access[_-]?token|token)\s*[:=]"),
)
_FORBIDDEN_PARAMETER_KEYS = frozenset(
    {
        "api_key",
        "argv",
        "command",
        "credentials",
        "env",
        "environment",
        "executable",
        "labels",
        "output",
        "output_path",
        "password",
        "path",
        "predictions",
        "prompts",
        "raw_rows",
        "rows",
        "script",
        "script_args",
        "script_path",
        "secret",
        "targets",
        "token",
        "uri",
        "url",
    }
)
_PARAMETER_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


def _safe_text(value: str) -> str:
    normalized = value.strip()
    if any(pattern.search(normalized) for pattern in _PRIVATE_TEXT_PATTERNS):
        raise ValueError("diagnostic text contains private or secret-like material")
    return normalized


def _parameter_value(value: Any) -> str | int | float | bool | tuple[str | int | float | bool, ...]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic parameters must be finite")
        return value
    if isinstance(value, str):
        if not 1 <= len(value) <= 1000:
            raise ValueError("diagnostic parameter text is out of bounds")
        return _safe_text(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not 1 <= len(value) <= 32:
            raise ValueError("diagnostic parameter lists are out of bounds")
        normalized = tuple(_parameter_value(item) for item in value)
        if any(isinstance(item, tuple) for item in normalized):
            raise ValueError("diagnostic parameters cannot contain nested lists")
        return normalized  # type: ignore[return-value]
    raise ValueError("diagnostic parameters must contain bounded JSON scalars")


class DiagnosticMeasurementRequest(FrozenContractModel):
    """One aggregate quantity selected by the host agent."""

    schema_version: Literal["bashgym.diagnostic_measurement_request.v1"] = (
        "bashgym.diagnostic_measurement_request.v1"
    )
    name: Identifier
    interpretation: Literal["maximize", "minimize", "observe"]
    unit: Identifier | None = None


class AutoResearchDiagnosticRecipe(FrozenContractModel):
    """Scientific diagnostic authored by an agent inside a fixed execution envelope."""

    schema_version: Literal["bashgym.autoresearch_diagnostic_recipe.v1"] = (
        AUTORESEARCH_DIAGNOSTIC_RECIPE_SCHEMA
    )
    probe_family: Identifier
    question: str = Field(min_length=1, max_length=1000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    informs_methods: tuple[Identifier, ...] = Field(min_length=1, max_length=6)
    measurements: tuple[DiagnosticMeasurementRequest, ...] = Field(
        min_length=1,
        max_length=16,
    )
    sample_limit: int = Field(ge=1, le=10_000)
    seed: int = Field(ge=0, le=2**63 - 1)
    data_scope_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)
    parameters: dict[Identifier, Any] = Field(default_factory=dict, max_length=32)

    @field_validator("question", "hypothesis")
    @classmethod
    def safe_scientific_text(cls, value: str) -> str:
        return _safe_text(value)

    @field_validator("informs_methods")
    @classmethod
    def unique_methods(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("informs_methods must be unique")
        return value

    @field_validator("data_scope_ids")
    @classmethod
    def canonical_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("data_scope_ids must be sorted and unique")
        return value

    @field_validator("parameters")
    @classmethod
    def bounded_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            canonical_key = key.casefold().replace("-", "_")
            if not _PARAMETER_KEY.fullmatch(key) or canonical_key in _FORBIDDEN_PARAMETER_KEYS:
                raise ValueError(
                    "diagnostic parameters contain forbidden execution or raw material"
                )
            normalized[key] = _parameter_value(item)
        return normalized

    @model_validator(mode="after")
    def unique_measurements(self) -> AutoResearchDiagnosticRecipe:
        names = tuple(item.name for item in self.measurements)
        if len(names) != len(set(names)):
            raise ValueError("diagnostic measurement names must be unique")
        if self.probe_family == "plasticity_probe":
            direction = self.parameters.get("metric_direction")
            step_budget = self.parameters.get("fixed_step_budget")
            efficiency_ratio = self.parameters.get("minimum_efficiency_ratio")
            retention_drop = self.parameters.get("maximum_retention_drop")
            if (
                direction not in {"maximize", "minimize"}
                or isinstance(step_budget, bool)
                or not isinstance(step_budget, int)
                or step_budget < 1
                or isinstance(efficiency_ratio, bool)
                or not isinstance(efficiency_ratio, (int, float))
                or not 0 <= float(efficiency_ratio) <= 1
                or isinstance(retention_drop, bool)
                or not isinstance(retention_drop, (int, float))
                or float(retention_drop) < 0
            ):
                raise ValueError("plasticity probe parameters are incomplete or invalid")
            required_measurements = {
                "initial_probe_metric",
                "final_probe_metric",
                "retention_delta",
                "cumulative_training_steps",
                "cumulative_training_tokens",
                "dataset_revision_count",
            }
            if not required_measurements.issubset(names):
                raise ValueError("plasticity probe measurements are incomplete")
        if self.probe_family == "reward_integrity_probe":
            reward_spec_digest = self.parameters.get("reward_spec_digest")
            canary_suite_id = self.parameters.get("canary_suite_id")
            if (
                not isinstance(reward_spec_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", reward_spec_digest) is None
                or not isinstance(canary_suite_id, str)
                or _PARAMETER_KEY.fullmatch(canary_suite_id) is None
            ):
                raise ValueError("reward integrity probe parameters are incomplete or invalid")
            required_measurements = {
                "reward_canary_cases",
                "reward_canary_failure_rate",
                "hard_constraint_violation_rate",
            }
            if not required_measurements.issubset(names):
                raise ValueError("reward integrity probe measurements are incomplete")
        if self.probe_family == "preference_integrity_probe":
            preference_dataset_digest = self.parameters.get("preference_dataset_digest")
            labeling_contract_digest = self.parameters.get("labeling_contract_digest")
            if (
                not isinstance(preference_dataset_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", preference_dataset_digest) is None
                or not isinstance(labeling_contract_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", labeling_contract_digest) is None
            ):
                raise ValueError("preference integrity probe parameters are incomplete or invalid")
            required_measurements = {
                "preference_pairs",
                "preference_agreement_lower_bound",
                "ambiguous_pair_rate",
                "preference_position_bias_rate",
                "preference_label_conflict_rate",
                "preference_contamination_rate",
            }
            if not required_measurements.issubset(names):
                raise ValueError("preference integrity probe measurements are incomplete")
        if self.probe_family == "teacher_gap_probe":
            evaluation_suite_id = self.parameters.get("evaluation_suite_id")
            metric_direction = self.parameters.get("metric_direction")
            teacher_model_digest = self.parameters.get("teacher_model_digest")
            student_model_digest = self.parameters.get("student_model_digest")
            validation_digest = self.parameters.get("output_validation_contract_digest")
            digests = (teacher_model_digest, student_model_digest, validation_digest)
            if (
                not isinstance(evaluation_suite_id, str)
                or _PARAMETER_KEY.fullmatch(evaluation_suite_id) is None
                or metric_direction not in {"maximize", "minimize"}
                or any(
                    not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in digests
                )
                or teacher_model_digest == student_model_digest
                or set(names) != {"teacher_metric_gap", "teacher_output_acceptance_rate"}
            ):
                raise ValueError("teacher gap probe parameters or measurements are invalid")
        if self.probe_family == "recovery_trace_probe":
            recovery_digest = self.parameters.get("recovery_dataset_digest")
            reader_digest = self.parameters.get("reader_contract_digest")
            confidence_level = self.parameters.get("confidence_level")
            if (
                not isinstance(recovery_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", recovery_digest) is None
                or not isinstance(reader_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", reader_digest) is None
                or confidence_level != 0.95
                or set(names) != {"recovery_traces", "recovery_lift_lower_bound"}
            ):
                raise ValueError("recovery trace probe parameters or measurements are invalid")
        return self


class DiagnosticMeasurementResult(FrozenContractModel):
    schema_version: Literal["bashgym.diagnostic_measurement_result.v1"] = (
        "bashgym.diagnostic_measurement_result.v1"
    )
    name: Identifier
    value: float
    sample_count: int = Field(ge=1, le=10_000_000)
    unit: Identifier | None = None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("diagnostic measurement value must be finite")
        return value


class AutoResearchDiagnosticRequest(FrozenContractModel):
    """Exact worker-owned request passed to the registered diagnostic runner."""

    schema_version: Literal["bashgym.autoresearch_diagnostic_request.v1"] = (
        "bashgym.autoresearch_diagnostic_request.v1"
    )
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    recipe: AutoResearchDiagnosticRecipe
    recipe_digest: HexDigest
    runner_id: Identifier
    runner_version: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def verify_recipe_digest(self) -> AutoResearchDiagnosticRequest:
        if self.recipe_digest != diagnostic_recipe_digest(self.recipe):
            raise ValueError("diagnostic request recipe digest mismatch")
        return self


class DiagnosticObservation(FrozenContractModel):
    schema_version: Literal["bashgym.diagnostic_observation.v1"] = (
        "bashgym.diagnostic_observation.v1"
    )
    observation_id: Identifier
    category: Identifier
    summary: str = Field(min_length=1, max_length=1000)
    count: int = Field(ge=1, le=1_000_000)

    @field_validator("summary")
    @classmethod
    def safe_summary(cls, value: str) -> str:
        return _safe_text(value)


class AutoResearchDiagnosticEvidence(FrozenContractModel):
    """Bounded aggregate evidence emitted by an installation-owned runner."""

    schema_version: Literal["bashgym.autoresearch_diagnostic_evidence.v1"] = (
        AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA
    )
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    recipe_digest: HexDigest
    runner_id: Identifier
    runner_version: str = Field(min_length=1, max_length=240)
    status: Literal["completed", "unsupported"]
    measurements: tuple[DiagnosticMeasurementResult, ...] = Field(default=(), max_length=16)
    observations: tuple[DiagnosticObservation, ...] = Field(default=(), max_length=12)
    resource_usage: tuple[ResourceUsage, ...] = Field(default=(), max_length=8)
    unsupported_reason: Identifier | None = None

    @model_validator(mode="after")
    def validate_status_and_uniqueness(self) -> AutoResearchDiagnosticEvidence:
        measurement_names = tuple(item.name for item in self.measurements)
        observation_ids = tuple(item.observation_id for item in self.observations)
        if len(measurement_names) != len(set(measurement_names)):
            raise ValueError("diagnostic evidence measurement names must be unique")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("diagnostic observation IDs must be unique")
        if any(item.confidence != "measured" for item in self.resource_usage):
            raise ValueError("diagnostic evidence resource usage must be measured")
        if self.status == "completed":
            if not self.measurements or self.unsupported_reason is not None:
                raise ValueError("completed diagnostic evidence requires measurements only")
        elif self.measurements:
            raise ValueError("unsupported evidence cannot contain measurements")
        elif self.observations:
            raise ValueError("unsupported evidence cannot contain observations")
        elif self.unsupported_reason is None:
            raise ValueError("unsupported evidence requires one reason")
        return self


def diagnostic_recipe_digest(recipe: AutoResearchDiagnosticRecipe) -> str:
    return canonical_hash(recipe.model_dump(mode="json"))


def diagnostic_request_bytes(request: AutoResearchDiagnosticRequest) -> bytes:
    return (
        json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def validate_diagnostic_envelope(
    recipe: AutoResearchDiagnosticRecipe,
    *,
    approved_data_scopes: frozenset[str],
    max_sample_limit: int,
    max_measurements: int,
) -> AutoResearchDiagnosticRecipe:
    """Check only operator-owned scope and resource ceilings."""

    if not set(recipe.data_scope_ids).issubset(approved_data_scopes):
        raise ValueError("diagnostic_data_scope_not_approved")
    if recipe.sample_limit > max_sample_limit:
        raise ValueError("diagnostic_sample_limit_exceeded")
    if len(recipe.measurements) > max_measurements:
        raise ValueError("diagnostic_measurement_limit_exceeded")
    return recipe


def validated_diagnostic_evidence(
    value: Any,
    *,
    recipe: AutoResearchDiagnosticRecipe,
    expected_identity: Mapping[str, str],
    expected_runner_id: str,
    expected_runner_version: str,
) -> AutoResearchDiagnosticEvidence:
    """Authenticate evidence against the exact request and attempt identities."""

    evidence = (
        AutoResearchDiagnosticEvidence.model_validate_json(value)
        if isinstance(value, (str, bytes, bytearray))
        else AutoResearchDiagnosticEvidence.model_validate(value)
    )
    if evidence.recipe_digest != diagnostic_recipe_digest(recipe):
        raise ValueError("diagnostic_recipe_digest_mismatch")
    for field in (
        "workspace_id",
        "campaign_id",
        "proposal_id",
        "study_id",
        "action_id",
        "attempt_id",
    ):
        if getattr(evidence, field) != expected_identity.get(field):
            raise ValueError("diagnostic_evidence_identity_mismatch")
    if (
        evidence.runner_id != expected_runner_id
        or evidence.runner_version != expected_runner_version
    ):
        raise ValueError("diagnostic_runner_identity_mismatch")
    if evidence.status == "completed":
        requested = tuple(item.name for item in recipe.measurements)
        observed = tuple(item.name for item in evidence.measurements)
        if observed != requested:
            raise ValueError("diagnostic_measurements_mismatch")
        requested_units = {item.name: item.unit for item in recipe.measurements}
        if any(
            requested_units[item.name] is not None and item.unit != requested_units[item.name]
            for item in evidence.measurements
        ):
            raise ValueError("diagnostic_measurement_unit_mismatch")
        if any(item.sample_count > recipe.sample_limit for item in evidence.measurements):
            raise ValueError("diagnostic_sample_limit_mismatch")
        if recipe.probe_family == "reward_integrity_probe":
            observed = {item.name: item.value for item in evidence.measurements}
            canary_cases = observed["reward_canary_cases"]
            rate_names = (
                "reward_canary_failure_rate",
                "hard_constraint_violation_rate",
            )
            if (
                canary_cases < 1
                or not canary_cases.is_integer()
                or any(not 0 <= observed[name] <= 1 for name in rate_names)
            ):
                raise ValueError("reward_integrity_measurement_out_of_range")
        if recipe.probe_family == "preference_integrity_probe":
            observed = {item.name: item.value for item in evidence.measurements}
            pairs = observed["preference_pairs"]
            rate_names = (
                "preference_agreement_lower_bound",
                "ambiguous_pair_rate",
                "preference_position_bias_rate",
                "preference_label_conflict_rate",
                "preference_contamination_rate",
            )
            if (
                pairs < 1
                or not pairs.is_integer()
                or any(not 0 <= observed[name] <= 1 for name in rate_names)
            ):
                raise ValueError("preference_integrity_measurement_out_of_range")
        if recipe.probe_family == "teacher_gap_probe":
            observed = {item.name: item.value for item in evidence.measurements}
            if not 0 <= observed["teacher_output_acceptance_rate"] <= 1:
                raise ValueError("teacher_gap_measurement_out_of_range")
        if recipe.probe_family == "recovery_trace_probe":
            observed = {item.name: item.value for item in evidence.measurements}
            traces = observed["recovery_traces"]
            if (
                traces < 1
                or not traces.is_integer()
                or not -1 <= observed["recovery_lift_lower_bound"] <= 1
            ):
                raise ValueError("recovery_trace_measurement_out_of_range")
    return evidence


def public_diagnostic_projection(
    recipe: AutoResearchDiagnosticRecipe,
    evidence: AutoResearchDiagnosticEvidence,
) -> dict[str, Any]:
    """Project only aggregate scientific context safe for agents and reports."""

    projection = {
        "schema_version": "bashgym.research_diagnostic_result.v1",
        "probe_family": recipe.probe_family,
        "question": recipe.question,
        "hypothesis": recipe.hypothesis,
        "informs_methods": list(recipe.informs_methods),
        "status": evidence.status,
        "measurements": [
            {
                "name": item.name,
                "value": item.value,
                "sample_count": item.sample_count,
                "unit": item.unit,
            }
            for item in evidence.measurements
        ],
        "observations": [
            {
                "observation_id": item.observation_id,
                "category": item.category,
                "summary": item.summary,
                "count": item.count,
            }
            for item in evidence.observations
        ],
        "resource_usage": [
            {
                "unit": item.unit,
                "amount": item.amount,
                "source": item.source,
                "confidence": item.confidence,
            }
            for item in evidence.resource_usage
        ],
        "unsupported_reason": evidence.unsupported_reason,
        "evidence_reference": {
            "proposal_id": evidence.proposal_id,
            "study_id": evidence.study_id,
            "attempt_id": evidence.attempt_id,
            "recipe_digest": evidence.recipe_digest,
        },
    }
    if recipe.probe_family == "plasticity_probe":
        projection["comparison_contract"] = {
            "metric_direction": recipe.parameters["metric_direction"],
            "fixed_step_budget": recipe.parameters["fixed_step_budget"],
            "minimum_efficiency_ratio": recipe.parameters["minimum_efficiency_ratio"],
            "maximum_retention_drop": recipe.parameters["maximum_retention_drop"],
            "sample_limit": recipe.sample_limit,
            "seed": recipe.seed,
            "data_scope_ids": list(recipe.data_scope_ids),
        }
    elif recipe.probe_family == "reward_integrity_probe":
        projection["comparison_contract"] = {
            "reward_spec_digest": recipe.parameters["reward_spec_digest"],
            "canary_suite_id": recipe.parameters["canary_suite_id"],
            "sample_limit": recipe.sample_limit,
            "seed": recipe.seed,
        }
    elif recipe.probe_family == "preference_integrity_probe":
        projection["comparison_contract"] = {
            "preference_dataset_digest": recipe.parameters["preference_dataset_digest"],
            "labeling_contract_digest": recipe.parameters["labeling_contract_digest"],
            "sample_limit": recipe.sample_limit,
            "seed": recipe.seed,
        }
    elif recipe.probe_family == "teacher_gap_probe":
        projection["comparison_contract"] = {
            "evaluation_suite_id": recipe.parameters["evaluation_suite_id"],
            "metric_direction": recipe.parameters["metric_direction"],
            "teacher_model_digest": recipe.parameters["teacher_model_digest"],
            "student_model_digest": recipe.parameters["student_model_digest"],
            "output_validation_contract_digest": recipe.parameters[
                "output_validation_contract_digest"
            ],
            "sample_limit": recipe.sample_limit,
            "seed": recipe.seed,
        }
    elif recipe.probe_family == "recovery_trace_probe":
        projection["comparison_contract"] = {
            "recovery_dataset_digest": recipe.parameters["recovery_dataset_digest"],
            "reader_contract_digest": recipe.parameters["reader_contract_digest"],
            "confidence_level": recipe.parameters["confidence_level"],
            "sample_limit": recipe.sample_limit,
            "seed": recipe.seed,
        }
    return projection


__all__ = [
    "AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME",
    "AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA",
    "AUTORESEARCH_DIAGNOSTIC_RECIPE_SCHEMA",
    "AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME",
    "AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN",
    "MAX_AUTORESEARCH_DIAGNOSTIC_BYTES",
    "AutoResearchDiagnosticEvidence",
    "AutoResearchDiagnosticRecipe",
    "AutoResearchDiagnosticRequest",
    "DiagnosticMeasurementRequest",
    "DiagnosticMeasurementResult",
    "DiagnosticObservation",
    "diagnostic_recipe_digest",
    "diagnostic_request_bytes",
    "public_diagnostic_projection",
    "validate_diagnostic_envelope",
    "validated_diagnostic_evidence",
]
