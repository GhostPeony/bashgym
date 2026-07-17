"""Run card artifacts for reproducible BashGym training runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bashgym._compat import UTC
from bashgym.factory.session_distillation import validate_session_distillation_records
from bashgym.preferences import (
    REWARD_MODEL_EVAL_SCHEMA_VERSION,
    validate_preference_pairs_file,
    validate_reward_examples_file,
)

SCHEMA_VERSION = "bashgym.run_card.v1"
VALID_DECISIONS = {"pending", "hold", "promote", "reject", "route_narrowly"}
VALID_CLAIM_TIERS = {"local_smoke", "narrow_routing", "broad_public_claim"}
REWARD_TRAINING_METHODS = {
    "reward_model",
    "rm",
    "preference_rm",
    "orm",
    "prm",
    "process_reward",
    "process_reward_model",
}
SESSION_DISTILLATION_METHODS = {"session_distillation", "opsd", "opd", "targeted_opsd"}


def _finding(
    code: str,
    level: str,
    message: str,
    *,
    field: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    finding = {"code": code, "level": level, "message": message}
    if field:
        finding["field"] = field
    if path:
        finding["path"] = path
    return finding


@dataclass
class RunCard:
    run_id: str
    training_method: str
    base_model: str
    compute_target_id: str
    training_plan_path: str | None = None
    source_manifest_path: str | None = None
    preference_pairs_path: str | None = None
    reward_examples_path: str | None = None
    reward_eval_path: str | None = None
    session_distillation_records_path: str | None = None
    session_distillation_metrics_path: str | None = None
    session_distillation_reader_model: str | None = None
    session_distillation_confidence_threshold: float | None = None
    session_distillation_hint_policy: str | None = None
    session_distillation_mask_policy: str | None = None
    session_distillation_target_token_count: int | None = None
    dataset_card_path: str | None = None
    backend: str | None = None
    git_commit: str | None = None
    branch: str | None = None
    metrics_path: str | None = None
    release_evidence_path: str | None = None
    smoke_bundle_path: str | None = None
    claim_tier: str = "local_smoke"
    thresholds: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    decision: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validation_findings(self, *, promotion: bool = False) -> list[dict[str, str]]:
        findings: list[dict[str, str]] = []
        required = {
            "run_id": self.run_id,
            "training_method": self.training_method,
            "base_model": self.base_model,
            "compute_target_id": self.compute_target_id,
            "training_plan_path": self.training_plan_path,
            "source_manifest_path": self.source_manifest_path,
        }
        if promotion:
            required.update(
                {
                    "metrics_path": self.metrics_path,
                    "release_evidence_path": self.release_evidence_path,
                }
            )
            if self.training_method.lower() == "dpo":
                required["preference_pairs_path"] = self.preference_pairs_path
            if self.training_method.lower() in REWARD_TRAINING_METHODS:
                required["reward_examples_path"] = self.reward_examples_path
                required["reward_eval_path"] = self.reward_eval_path
            if self.training_method.lower() in SESSION_DISTILLATION_METHODS:
                required["session_distillation_records_path"] = (
                    self.session_distillation_records_path
                )
                required["session_distillation_reader_model"] = (
                    self.session_distillation_reader_model
                )
                required["session_distillation_confidence_threshold"] = (
                    self.session_distillation_confidence_threshold
                )
                required["session_distillation_hint_policy"] = self.session_distillation_hint_policy
                required["session_distillation_mask_policy"] = self.session_distillation_mask_policy
                required["session_distillation_target_token_count"] = (
                    self.session_distillation_target_token_count
                )
        for field_name, value in required.items():
            if not value:
                findings.append(
                    _finding(
                        f"missing_{field_name}",
                        "fail",
                        f"{field_name} is required",
                        field=field_name,
                    )
                )
        if self.decision not in VALID_DECISIONS:
            findings.append(
                _finding(
                    "unknown_decision",
                    "warn",
                    "decision should be pending, hold, promote, reject, or route_narrowly",
                    field="decision",
                )
            )
        if self.claim_tier not in VALID_CLAIM_TIERS:
            findings.append(
                _finding(
                    "unknown_claim_tier",
                    "fail",
                    "claim_tier should be local_smoke, narrow_routing, or broad_public_claim",
                    field="claim_tier",
                )
            )
        if self.training_method.lower() in SESSION_DISTILLATION_METHODS:
            if (
                self.session_distillation_confidence_threshold is not None
                and not 0 <= float(self.session_distillation_confidence_threshold) <= 1
            ):
                findings.append(
                    _finding(
                        "invalid_session_distillation_confidence_threshold",
                        "fail",
                        "session_distillation_confidence_threshold must be between 0 and 1",
                        field="session_distillation_confidence_threshold",
                    )
                )
            if (
                self.session_distillation_mask_policy is not None
                and self.session_distillation_mask_policy != "target_span_only"
            ):
                findings.append(
                    _finding(
                        "invalid_session_distillation_mask_policy",
                        "fail",
                        "Session Distillation promotion currently requires target_span_only masking",
                        field="session_distillation_mask_policy",
                    )
                )
            if (
                self.session_distillation_target_token_count is not None
                and self.session_distillation_target_token_count <= 0
            ):
                findings.append(
                    _finding(
                        "invalid_session_distillation_target_token_count",
                        "fail",
                        "session_distillation_target_token_count must be positive",
                        field="session_distillation_target_token_count",
                    )
                )
        return findings


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def current_git_commit() -> str | None:
    return _git_value(["git", "rev-parse", "HEAD"])


def current_git_branch() -> str | None:
    return _git_value(["git", "branch", "--show-current"])


def parse_thresholds(items: list[str] | None) -> dict[str, str]:
    thresholds: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"threshold {item!r} must use key=value")
        key, value = item.split("=", 1)
        thresholds[key.strip()] = value.strip()
    return thresholds


def create_run_card(
    *,
    run_id: str,
    training_method: str,
    base_model: str,
    compute_target_id: str,
    training_plan_path: str | None = None,
    source_manifest_path: str | None = None,
    preference_pairs_path: str | None = None,
    reward_examples_path: str | None = None,
    reward_eval_path: str | None = None,
    session_distillation_records_path: str | None = None,
    session_distillation_metrics_path: str | None = None,
    session_distillation_reader_model: str | None = None,
    session_distillation_confidence_threshold: float | None = None,
    session_distillation_hint_policy: str | None = None,
    session_distillation_mask_policy: str | None = None,
    session_distillation_target_token_count: int | None = None,
    dataset_card_path: str | None = None,
    backend: str | None = None,
    metrics_path: str | None = None,
    release_evidence_path: str | None = None,
    smoke_bundle_path: str | None = None,
    claim_tier: str = "local_smoke",
    thresholds: dict[str, Any] | None = None,
    outputs: list[str] | None = None,
    known_limitations: list[str] | None = None,
    decision: str = "pending",
    include_git: bool = True,
) -> RunCard:
    return RunCard(
        run_id=run_id,
        training_method=training_method,
        base_model=base_model,
        compute_target_id=compute_target_id,
        training_plan_path=training_plan_path,
        source_manifest_path=source_manifest_path,
        preference_pairs_path=preference_pairs_path,
        reward_examples_path=reward_examples_path,
        reward_eval_path=reward_eval_path,
        session_distillation_records_path=session_distillation_records_path,
        session_distillation_metrics_path=session_distillation_metrics_path,
        session_distillation_reader_model=session_distillation_reader_model,
        session_distillation_confidence_threshold=session_distillation_confidence_threshold,
        session_distillation_hint_policy=session_distillation_hint_policy,
        session_distillation_mask_policy=session_distillation_mask_policy,
        session_distillation_target_token_count=session_distillation_target_token_count,
        dataset_card_path=dataset_card_path,
        backend=backend,
        git_commit=current_git_commit() if include_git else None,
        branch=current_git_branch() if include_git else None,
        metrics_path=metrics_path,
        release_evidence_path=release_evidence_path,
        smoke_bundle_path=smoke_bundle_path,
        claim_tier=claim_tier,
        thresholds=thresholds or {},
        outputs=outputs or [],
        known_limitations=known_limitations or [],
        decision=decision,
    )


def write_run_card(card: RunCard, path: str | Path) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(output_path), "run_card": card.to_dict()}


def read_run_card(path: str | Path) -> RunCard:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run card schema {payload.get('schema_version')!r}")
    payload = dict(payload)
    payload.pop("schema_version", None)
    return RunCard(schema_version=SCHEMA_VERSION, **payload)


def attach_run_card_evidence(
    path: str | Path,
    *,
    metrics_path: str | None = None,
    release_evidence_path: str | None = None,
    preference_pairs_path: str | None = None,
    reward_examples_path: str | None = None,
    reward_eval_path: str | None = None,
    session_distillation_records_path: str | None = None,
    session_distillation_metrics_path: str | None = None,
    smoke_bundle_path: str | None = None,
    claim_tier: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    card = read_run_card(path)
    if metrics_path:
        card.metrics_path = metrics_path
    if release_evidence_path:
        card.release_evidence_path = release_evidence_path
    if preference_pairs_path:
        card.preference_pairs_path = preference_pairs_path
    if reward_examples_path:
        card.reward_examples_path = reward_examples_path
    if reward_eval_path:
        card.reward_eval_path = reward_eval_path
    if session_distillation_records_path:
        card.session_distillation_records_path = session_distillation_records_path
    if session_distillation_metrics_path:
        card.session_distillation_metrics_path = session_distillation_metrics_path
    if smoke_bundle_path:
        card.smoke_bundle_path = smoke_bundle_path
    if claim_tier:
        card.claim_tier = claim_tier
    return write_run_card(card, output_path or path)


def _resolve_artifact_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _read_json_artifact(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "expected a JSON object"
    return payload, None


def _release_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    if isinstance(report, dict):
        return report
    return payload


def _reason_text(reasons: Any) -> str:
    if not isinstance(reasons, list):
        return ""
    clean = [str(reason) for reason in reasons if reason]
    return "; ".join(clean)


def _validate_source_manifest(
    source_manifest: dict[str, Any],
    *,
    path: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if source_manifest.get("schema_version") != "bashgym.source_manifest.v1":
        findings.append(
            _finding(
                "invalid_source_manifest_schema",
                "fail",
                "source manifest schema_version must be bashgym.source_manifest.v1",
                field="source_manifest_path",
                path=path,
            )
        )
    verdict = source_manifest.get("use_verdict")
    if not isinstance(verdict, dict):
        findings.append(
            _finding(
                "missing_source_manifest_use_verdict",
                "fail",
                "source manifest must include a use_verdict",
                field="source_manifest_path",
                path=path,
            )
        )
        return findings
    if verdict.get("ok") is not True:
        blocking = (
            verdict.get("blocking_codes") if isinstance(verdict.get("blocking_codes"), list) else []
        )
        suffix = f": {', '.join(str(code) for code in blocking)}" if blocking else ""
        findings.append(
            _finding(
                "source_manifest_use_blocked",
                "fail",
                f"source manifest use_verdict is not ok{suffix}",
                field="source_manifest_path",
                path=path,
            )
        )
    if verdict.get("requires_override_reason"):
        findings.append(
            _finding(
                "source_manifest_override_reason_missing",
                "fail",
                "eval-only source override requires a saved reason before promotion",
                field="source_manifest_path",
                path=path,
            )
        )
    return findings


def _validate_release_evidence(
    release_evidence: dict[str, Any],
    *,
    path: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    report = _release_report(release_evidence)
    ship = report.get("ship")
    if not isinstance(ship, bool):
        findings.append(
            _finding(
                "release_evidence_missing_ship_verdict",
                "fail",
                "release evidence must include a boolean ship verdict",
                field="release_evidence_path",
                path=path,
            )
        )
        return findings
    if not ship:
        reasons = _reason_text(report.get("reasons"))
        detail = f": {reasons}" if reasons else ""
        findings.append(
            _finding(
                "release_evidence_not_shippable",
                "fail",
                f"release evidence verdict is ship=false{detail}",
                field="release_evidence_path",
                path=path,
            )
        )

    release_gate = report.get("release_gate")
    if not isinstance(release_gate, dict):
        findings.append(
            _finding(
                "release_evidence_missing_release_gate",
                "warn",
                "release evidence has a ship verdict but no release_gate summary",
                field="release_evidence_path",
                path=path,
            )
        )
        return findings

    if release_gate.get("ship") is not ship:
        findings.append(
            _finding(
                "release_gate_ship_mismatch",
                "fail",
                "release_gate.ship does not match the top-level ship verdict",
                field="release_evidence_path",
                path=path,
            )
        )

    quality = release_gate.get("world_model_quality")
    if isinstance(quality, dict) and quality.get("present"):
        findings.append(
            _finding(
                "world_model_quality_diagnostic_only",
                "diagnostic",
                "world-model quality evidence is present but remains diagnostic-only",
                field="release_evidence_path",
                path=path,
            )
        )
        quality_findings = quality.get("findings")
        if isinstance(quality_findings, list) and quality_findings:
            findings.append(
                _finding(
                    "world_model_quality_needs_attention",
                    "warn",
                    "world-model quality diagnostics need attention: "
                    + "; ".join(str(item) for item in quality_findings if item),
                    field="release_evidence_path",
                    path=path,
                )
            )

    return findings


def _has_key(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(payload.get(key) not in (None, "", [], {}) for key in keys)


def _environment_evidence(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("environment_evidence")
    return evidence if isinstance(evidence, dict) else {}


def _validate_claim_tier(
    card: RunCard,
    release_evidence: dict[str, Any] | None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    tier = card.claim_tier
    if tier not in VALID_CLAIM_TIERS:
        return findings
    if release_evidence is None:
        findings.append(
            _finding(
                "claim_tier_missing_release_evidence",
                "fail",
                "claim-tier validation requires release evidence",
                field="claim_tier",
            )
        )
        return findings

    report = _release_report(release_evidence)
    if report.get("ship") is not True:
        findings.append(
            _finding(
                "claim_tier_release_not_shippable",
                "fail",
                f"{tier} requires ship=true release evidence",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )

    if tier == "local_smoke":
        return findings

    release_gate = report.get("release_gate")
    if not isinstance(release_gate, dict):
        findings.append(
            _finding(
                "claim_tier_missing_release_gate",
                "fail",
                f"{tier} requires a structured release_gate summary",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )
        return findings

    if release_gate.get("trace_ship") is not True:
        findings.append(
            _finding(
                "claim_tier_trace_gate_missing_or_failed",
                "fail",
                f"{tier} requires trace_ship=true",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )

    environment_sections = release_gate.get("environment_sections")
    if release_gate.get("environment_ship") is not True or not environment_sections:
        findings.append(
            _finding(
                "claim_tier_environment_evidence_missing_or_failed",
                "fail",
                f"{tier} requires passing executable environment evidence",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )

    if tier == "narrow_routing":
        return findings

    external_sections = release_gate.get("external_benchmark_sections")
    if release_gate.get("external_benchmark_ship") is not True or not external_sections:
        findings.append(
            _finding(
                "claim_tier_external_benchmark_missing_or_failed",
                "fail",
                "broad_public_claim requires passing external benchmark evidence",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )

    evidence = _environment_evidence(report)
    has_split = _has_key(report, ("split_manifest", "split_manifest_path")) or _has_key(
        evidence, ("split_manifest", "split_manifest_path")
    )
    has_decontam = _has_key(
        report,
        ("decontamination_manifest", "decontamination_manifest_path"),
    ) or _has_key(
        evidence,
        ("decontamination_manifest", "decontamination_manifest_path"),
    )
    if not has_split:
        findings.append(
            _finding(
                "claim_tier_split_manifest_missing",
                "fail",
                "broad_public_claim requires split manifest evidence",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )
    if not has_decontam:
        findings.append(
            _finding(
                "claim_tier_decontamination_manifest_missing",
                "fail",
                "broad_public_claim requires decontamination evidence",
                field="claim_tier",
                path=card.release_evidence_path,
            )
        )
    return findings


def _validate_preference_pairs(
    path: Path,
    *,
    raw_path: str,
) -> list[dict[str, str]]:
    try:
        validation = validate_preference_pairs_file(path, strict=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _finding(
                "invalid_preference_pairs_artifact",
                "fail",
                f"preference pairs could not be read: {exc}",
                field="preference_pairs_path",
                path=raw_path,
            )
        ]

    findings: list[dict[str, str]] = []
    for pair_finding in validation["findings"]:
        level = str(pair_finding.get("level", "warn"))
        code = str(pair_finding.get("code", "preference_pair_finding"))
        pair_id = pair_finding.get("pair_id")
        index = pair_finding.get("index")
        location = f"record {index}"
        if pair_id:
            location += f" ({pair_id})"
        findings.append(
            _finding(
                f"preference_pairs_{code}",
                level,
                f"{location}: {pair_finding.get('message', code)}",
                field="preference_pairs_path",
                path=raw_path,
            )
        )
    return findings


def _validate_reward_examples(
    path: Path,
    *,
    raw_path: str,
) -> list[dict[str, str]]:
    try:
        validation = validate_reward_examples_file(path, strict=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            _finding(
                "invalid_reward_examples_artifact",
                "fail",
                f"reward examples could not be read: {exc}",
                field="reward_examples_path",
                path=raw_path,
            )
        ]

    findings: list[dict[str, str]] = []
    for reward_finding in validation["findings"]:
        level = str(reward_finding.get("level", "warn"))
        code = str(reward_finding.get("code", "reward_example_finding"))
        example_id = reward_finding.get("example_id")
        index = reward_finding.get("index")
        location = f"record {index}"
        if example_id:
            location += f" ({example_id})"
        findings.append(
            _finding(
                f"reward_examples_{code}",
                level,
                f"{location}: {reward_finding.get('message', code)}",
                field="reward_examples_path",
                path=raw_path,
            )
        )
    return findings


def _validate_reward_eval(
    reward_eval: dict[str, Any],
    *,
    path: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if reward_eval.get("schema_version") != REWARD_MODEL_EVAL_SCHEMA_VERSION:
        findings.append(
            _finding(
                "invalid_reward_eval_schema",
                "fail",
                f"reward eval schema_version must be {REWARD_MODEL_EVAL_SCHEMA_VERSION}",
                field="reward_eval_path",
                path=path,
            )
        )
    if reward_eval.get("ok") is not True:
        findings.append(
            _finding(
                "reward_eval_not_ok",
                "fail",
                "reward-model eval artifact must have ok=true",
                field="reward_eval_path",
                path=path,
            )
        )
    metrics = reward_eval.get("metrics")
    if not isinstance(metrics, dict):
        findings.append(
            _finding(
                "reward_eval_missing_metrics",
                "fail",
                "reward-model eval artifact must include metrics",
                field="reward_eval_path",
                path=path,
            )
        )
        return findings
    if metrics.get("eval_only_leakage_count", 0):
        findings.append(
            _finding(
                "reward_eval_eval_only_leakage",
                "fail",
                "reward-model eval reports eval-only leakage",
                field="reward_eval_path",
                path=path,
            )
        )
    if metrics.get("heldout_pair_accuracy") is None and metrics.get("calibration_error") is None:
        findings.append(
            _finding(
                "reward_eval_no_heldout_quality_metric",
                "warn",
                "reward-model eval should include heldout pair accuracy or calibration",
                field="reward_eval_path",
                path=path,
            )
        )
    for reward_finding in reward_eval.get("findings", []):
        if not isinstance(reward_finding, dict):
            continue
        level = str(reward_finding.get("level", "warn"))
        code = str(reward_finding.get("code", "reward_eval_finding"))
        findings.append(
            _finding(
                f"reward_eval_{code}",
                level,
                str(reward_finding.get("message", code)),
                field="reward_eval_path",
                path=path,
            )
        )
    return findings


def _read_jsonl_artifact(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                return None, f"line {line_number}: expected a JSON object"
            records.append(payload)
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSONL: {exc}"
    return records, None


def _validate_session_distillation_records_file(
    path: Path,
    *,
    raw_path: str,
) -> list[dict[str, str]]:
    records, error = _read_jsonl_artifact(path)
    if error or records is None:
        return [
            _finding(
                "invalid_session_distillation_records_artifact",
                "fail",
                f"session distillation records could not be read: {error}",
                field="session_distillation_records_path",
                path=raw_path,
            )
        ]
    if not records:
        return [
            _finding(
                "session_distillation_records_empty",
                "fail",
                "session_distillation_records.jsonl must contain at least one record",
                field="session_distillation_records_path",
                path=raw_path,
            )
        ]

    findings: list[dict[str, str]] = []
    for error in validate_session_distillation_records(records):
        findings.append(
            _finding(
                "session_distillation_record_invalid",
                "fail",
                error,
                field="session_distillation_records_path",
                path=raw_path,
            )
        )
    return findings


def _validate_session_distillation_metrics(
    path: Path,
    *,
    raw_path: str,
) -> list[dict[str, str]]:
    records, error = _read_jsonl_artifact(path)
    if error or records is None:
        return [
            _finding(
                "invalid_session_distillation_metrics",
                "fail",
                f"session distillation metrics could not be read: {error}",
                field="metrics_path",
                path=raw_path,
            )
        ]
    has_loss = any(
        "session_distillation_loss" in record and "session_distillation_masked_tokens" in record
        for record in records
    )
    if not has_loss:
        return [
            _finding(
                "session_distillation_metrics_missing_masked_loss",
                "fail",
                "metrics must include session_distillation_loss and session_distillation_masked_tokens",
                field="metrics_path",
                path=raw_path,
            )
        ]
    return []


def _promotion_gate_for_finding(finding: dict[str, str]) -> str:
    code = finding.get("code", "")
    field = finding.get("field", "")
    if code.startswith("session_distillation_") or field.startswith("session_distillation_"):
        return "Session Distillation evidence"
    if code.startswith("preference_pairs_") or field == "preference_pairs_path":
        return "DPO preference-pair evidence"
    if code.startswith("reward_examples_") or field == "reward_examples_path":
        return "reward-example evidence"
    if code.startswith("reward_eval_") or field == "reward_eval_path":
        return "reward-model eval evidence"
    if code.startswith("source_manifest_") or field == "source_manifest_path":
        return "source policy evidence"
    if code.startswith("release_evidence_") or code.startswith("release_gate_"):
        return "release evidence"
    if code.startswith("claim_tier_") or field == "claim_tier":
        return "claim-tier evidence"
    if code.startswith("world_model_quality_"):
        return "world-model diagnostics"
    if "metrics" in code or field == "metrics_path":
        return "metrics evidence"
    if "training_plan" in code or field == "training_plan_path":
        return "training plan evidence"
    if "artifact" in code or "file" in code:
        return "artifact files"
    return "run-card metadata"


def _promotion_next_action(gate: str) -> str:
    actions = {
        "Session Distillation evidence": (
            "Attach valid session_distillation_records.jsonl, reader/mask metadata, "
            "masked-loss metrics, and heldout release evidence before promotion."
        ),
        "DPO preference-pair evidence": (
            "Attach strict DPO preference pairs with prompt hashes, chosen/rejected trace ids, "
            "quality scores, split, and decontamination metadata."
        ),
        "reward-example evidence": (
            "Attach strict reward examples with reward scale, label source, split, "
            "quality, and decontamination metadata."
        ),
        "reward-model eval evidence": (
            "Attach a passing reward_eval.json with heldout accuracy or calibration metrics "
            "and no eval-only leakage."
        ),
        "source policy evidence": (
            "Attach a source_manifest.json whose use_verdict is ok, or record a valid "
            "eval-only override reason when policy permits it."
        ),
        "release evidence": (
            "Run heldout/release gates and attach release_evidence.json with ship=true."
        ),
        "claim-tier evidence": (
            "Attach the behavior evidence required by the selected claim tier, or lower "
            "the claim tier to match the evidence that exists."
        ),
        "world-model diagnostics": (
            "Treat ECHO/RWML quality as diagnostic until it is correlated with behavior gates."
        ),
        "metrics evidence": "Attach a non-empty metrics artifact from the completed training run.",
        "training plan evidence": "Attach the training plan used for the run.",
        "artifact files": "Fix missing or unreadable artifact paths in the RunCard.",
        "run-card metadata": "Fix required RunCard metadata before promotion review.",
    }
    return actions.get(gate, "Resolve the listed findings before promotion review.")


def explain_run_card_promotion(
    card: RunCard,
    findings: list[dict[str, str]],
) -> dict[str, Any]:
    """Summarize why a RunCard can or cannot be promoted."""

    blockers = [finding for finding in findings if finding.get("level") == "fail"]
    warnings = [finding for finding in findings if finding.get("level") == "warn"]
    diagnostics = [finding for finding in findings if finding.get("level") == "diagnostic"]
    gates: dict[str, dict[str, Any]] = {}
    for finding in blockers:
        gate = _promotion_gate_for_finding(finding)
        item = gates.setdefault(
            gate,
            {
                "gate": gate,
                "blocker_count": 0,
                "codes": [],
                "summary": "",
                "next_action": _promotion_next_action(gate),
            },
        )
        item["blocker_count"] += 1
        code = finding.get("code", "unknown_finding")
        if code not in item["codes"]:
            item["codes"].append(code)

    failed_gates = sorted(
        gates.values(),
        key=lambda item: (-int(item["blocker_count"]), str(item["gate"])),
    )
    for gate in failed_gates:
        codes = ", ".join(str(code) for code in gate["codes"][:4])
        extra = "" if len(gate["codes"]) <= 4 else f" and {len(gate['codes']) - 4} more"
        gate["summary"] = f"{gate['gate']} has {gate['blocker_count']} blocker(s): {codes}{extra}."

    next_actions = [str(gate["next_action"]) for gate in failed_gates[:5]]
    if not blockers and warnings:
        next_actions.append("Review warnings before making a stronger public claim.")
    if not blockers and diagnostics:
        next_actions.append("Keep diagnostic evidence separate from promotion blockers.")

    if blockers:
        headline = (
            f"{card.run_id} is not promotable for {card.claim_tier}: "
            f"{len(blockers)} blocker(s) across {len(failed_gates)} gate(s)."
        )
    elif warnings:
        headline = (
            f"{card.run_id} clears promotion blockers for {card.claim_tier}, "
            f"with {len(warnings)} warning(s) to review."
        )
    else:
        headline = f"{card.run_id} clears promotion blockers for {card.claim_tier}."

    return {
        "schema_version": "bashgym.run_card_promotion_explanation.v1",
        "ok": not blockers,
        "headline": headline,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "diagnostic_count": len(diagnostics),
        "failed_gates": failed_gates,
        "next_actions": next_actions,
    }


def validate_run_card_file(
    path: str | Path,
    *,
    promotion: bool = False,
) -> dict[str, Any]:
    """Validate a run card and, for promotion, fail closed on artifact evidence."""

    card_path = Path(path)
    card = read_run_card(card_path)
    findings = card.validation_findings(promotion=promotion)
    artifact_status: list[dict[str, Any]] = []
    release_evidence_payload: dict[str, Any] | None = None

    if promotion:
        base_dir = card_path.resolve().parent
        required_artifacts = (
            "training_plan_path",
            "source_manifest_path",
            "metrics_path",
            "release_evidence_path",
        )
        if card.training_method.lower() == "dpo":
            required_artifacts = (*required_artifacts, "preference_pairs_path")
        if card.training_method.lower() in REWARD_TRAINING_METHODS:
            required_artifacts = (*required_artifacts, "reward_examples_path", "reward_eval_path")
        if card.training_method.lower() in SESSION_DISTILLATION_METHODS:
            required_artifacts = (*required_artifacts, "session_distillation_records_path")
            if card.session_distillation_metrics_path:
                required_artifacts = (*required_artifacts, "session_distillation_metrics_path")
        optional_artifacts = ("dataset_card_path", "smoke_bundle_path")
        for field_name in (*required_artifacts, *optional_artifacts):
            raw_path = getattr(card, field_name)
            if not raw_path:
                continue
            resolved = _resolve_artifact_path(raw_path, base_dir)
            present = resolved.exists()
            artifact_status.append(
                {
                    "field": field_name,
                    "path": raw_path,
                    "resolved_path": str(resolved),
                    "present": present,
                }
            )
            if not present:
                level = "fail" if field_name in required_artifacts else "warn"
                findings.append(
                    _finding(
                        f"missing_{field_name}_file",
                        level,
                        f"{field_name} file does not exist",
                        field=field_name,
                        path=raw_path,
                    )
                )
                continue
            if field_name == "metrics_path" and resolved.stat().st_size == 0:
                findings.append(
                    _finding(
                        "metrics_file_empty",
                        "fail",
                        "metrics file is empty",
                        field=field_name,
                        path=raw_path,
                    )
                )
            if (
                field_name == "metrics_path"
                and card.training_method.lower() in SESSION_DISTILLATION_METHODS
                and not card.session_distillation_metrics_path
            ):
                findings.extend(_validate_session_distillation_metrics(resolved, raw_path=raw_path))
            if field_name == "session_distillation_metrics_path":
                findings.extend(_validate_session_distillation_metrics(resolved, raw_path=raw_path))
            if field_name == "session_distillation_records_path":
                findings.extend(
                    _validate_session_distillation_records_file(resolved, raw_path=raw_path)
                )
            if field_name == "source_manifest_path":
                payload, error = _read_json_artifact(resolved)
                if error or payload is None:
                    findings.append(
                        _finding(
                            "invalid_source_manifest_json",
                            "fail",
                            f"source manifest could not be read: {error}",
                            field=field_name,
                            path=raw_path,
                        )
                    )
                else:
                    findings.extend(_validate_source_manifest(payload, path=raw_path))
            if field_name == "release_evidence_path":
                payload, error = _read_json_artifact(resolved)
                if error or payload is None:
                    findings.append(
                        _finding(
                            "invalid_release_evidence_json",
                            "fail",
                            f"release evidence could not be read: {error}",
                            field=field_name,
                            path=raw_path,
                        )
                    )
                else:
                    release_evidence_payload = payload
                    findings.extend(_validate_release_evidence(payload, path=raw_path))
            if field_name == "preference_pairs_path":
                findings.extend(_validate_preference_pairs(resolved, raw_path=raw_path))
            if field_name == "reward_examples_path":
                findings.extend(_validate_reward_examples(resolved, raw_path=raw_path))
            if field_name == "reward_eval_path":
                payload, error = _read_json_artifact(resolved)
                if error or payload is None:
                    findings.append(
                        _finding(
                            "invalid_reward_eval_json",
                            "fail",
                            f"reward eval could not be read: {error}",
                            field=field_name,
                            path=raw_path,
                        )
                    )
                else:
                    findings.extend(_validate_reward_eval(payload, path=raw_path))

        findings.extend(_validate_claim_tier(card, release_evidence_payload))

    return {
        "path": str(card_path),
        "run_card": card.to_dict(),
        "findings": findings,
        "artifact_status": artifact_status,
        "promotion_explanation": explain_run_card_promotion(card, findings),
        "ok": not any(finding["level"] == "fail" for finding in findings),
    }
