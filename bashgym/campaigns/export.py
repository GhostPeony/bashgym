"""Deterministic, hash-reconciled campaign evidence exports."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.reporting import (
    write_campaign_docx,
    write_campaign_pdf,
    write_loss_png,
)


class CampaignExportError(ValueError):
    """Stable export failure for invalid or unsafe evidence projections."""


_PUBLIC_ATTEMPT_FIELDS = (
    "attempt_id",
    "study_id",
    "action_id",
    "stage",
    "status",
    "candidate_digest",
    "recipe_digest",
    "compute_profile_id",
    "attempt_number",
    "created_at",
    "updated_at",
)
_PRIVATE_LOCATION_KEYS = {
    "path",
    "uri",
    "sealed_result_uri",
    "local_path",
    "remote_path",
    "remote_model_path",
    "remote_dataset_path",
    "working_directory",
    "workdir",
}
_ABSOLUTE_LOCATION = re.compile(
    r"(?i)(?:^[a-z]:[\\/]|^/(?:users|home|var|tmp|etc|opt|srv|mnt)/|\b(?:file|ssh)://)"
)


@dataclass(frozen=True)
class CampaignExportSnapshot:
    campaign: dict[str, Any]
    attempts: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    comparisons: tuple[dict[str, Any], ...] = ()
    loss_by_attempt: dict[str, tuple[dict[str, Any], ...]] | None = None
    flags: tuple[str, ...] = ()
    autoresearch_history: dict[str, Any] | None = None

    def safe_payload(self) -> dict[str, Any]:
        artifacts = []
        for item in self.artifacts:
            if "uri" in item or "path" in item or "sealed_result_uri" in item:
                raise CampaignExportError("campaign_export_contains_local_path")
            artifacts.append(
                {
                    key: item[key]
                    for key in (
                        "artifact_id",
                        "producer_action_id",
                        "sha256",
                        "size_bytes",
                        "schema_name",
                        "sealed",
                        "valid",
                        "created_at",
                    )
                    if key in item
                }
            )
        attempts = []
        for item in self.attempts:
            attempts.append({key: item[key] for key in _PUBLIC_ATTEMPT_FIELDS if key in item})
        history = self.autoresearch_history
        campaign = dict(self.campaign)
        report_summary = _autoresearch_report_summary(history)
        if report_summary is not None:
            campaign.update(report_summary)
        payload = {
            "schema_version": "campaign_export_snapshot.v2",
            "campaign": campaign,
            "attempts": attempts,
            "artifacts": artifacts,
            "comparisons": list(self.comparisons),
            "loss_by_attempt": {
                key: list(value) for key, value in sorted((self.loss_by_attempt or {}).items())
            },
            "flags": list(self.flags),
            "autoresearch_history": history,
        }
        _reject_private_location_fields(payload)
        return payload


def _reject_private_location_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        if _PRIVATE_LOCATION_KEYS.intersection(str(key).lower() for key in value):
            raise CampaignExportError("campaign_export_contains_local_path")
        for item in value.values():
            _reject_private_location_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_location_fields(item)
    elif isinstance(value, str) and _ABSOLUTE_LOCATION.search(value):
        raise CampaignExportError("campaign_export_contains_local_path")


def _autoresearch_report_summary(history: dict[str, Any] | None) -> dict[str, Any] | None:
    if not history:
        return None
    reference: dict[str, Any] | None = None
    baseline_cost = 0.0
    candidate_cost = 0.0
    has_cost = False
    for experiment in history.get("experiments") or ():
        result = experiment.get("result") or {}
        cost = result.get("actual_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost):
            has_cost = True
            if experiment.get("role") == "baseline":
                baseline_cost += float(cost)
            else:
                candidate_cost += float(cost)
        decision = (experiment.get("decision") or {}).get("decision")
        if decision not in {"baseline", "keep"}:
            continue
        primary = (experiment.get("performance") or {}).get("primary") or {}
        reference = {
            "proposal_id": experiment.get("proposal_id"),
            "metric_name": primary.get("metric_name"),
            "metric_value": primary.get("candidate_value"),
            "decision": decision,
        }
    summary: dict[str, Any] = {"autoresearch_reference": reference}
    if has_cost:
        summary["recorded_costs"] = {
            "baseline": baseline_cost,
            "candidate": candidate_cost,
            "total": baseline_cost + candidate_cost,
        }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _loss_svg(snapshot: dict[str, Any]) -> str:
    width, height = 960, 480
    left, right, top, bottom = 70, 30, 45, 65
    series = []
    attempt_by_id = {
        item.get("attempt_id"): item for item in snapshot["attempts"] if item.get("attempt_id")
    }
    for attempt_id, values in snapshot["loss_by_attempt"].items():
        points = [
            (int(item["step"]), float(item["value"]))
            for item in values
            if "step" in item and "value" in item
        ]
        if points:
            series.append(
                (attempt_id, attempt_by_id.get(attempt_id, {}).get("stage", "unknown"), points)
            )
    max_step = max((step for _id, _stage, points in series for step, _value in points), default=1)
    losses = [value for _id, _stage, points in series for _step, value in points]
    min_loss = min(losses, default=0.0)
    max_loss = max(losses, default=1.0)
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    def x(step: int) -> float:
        return left + (width - left - right) * (step / max_step)

    def y(loss: float) -> float:
        return top + (height - top - bottom) * ((max_loss - loss) / (max_loss - min_loss))

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">',
        '<rect width="960" height="480" fill="#fffdf7"/>',
        '<text x="70" y="28" font-family="sans-serif" font-size="18" font-weight="700">Campaign training loss</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#332f2a" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#332f2a" stroke-width="2"/>',
        f'<text x="{width/2:.0f}" y="455" text-anchor="middle" font-family="sans-serif" font-size="13">Training step</text>',
        f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})" font-family="sans-serif" font-size="13">Loss</text>',
    ]
    colors = ("#7c3aed", "#d97706", "#15803d", "#0369a1", "#be123c")
    for index, (attempt_id, stage, points) in enumerate(series):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{x(step):.2f},{y(value):.2f}" for step, value in points)
        dash = ' stroke-dasharray="7 5"' if stage == "smoke_training" else ""
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
        )
        label = html.escape(f"{attempt_id} · {stage}")
        lines.append(
            f'<text x="{left + 10}" y="{top + 20 + index * 18}" font-family="monospace" font-size="11" fill="{color}">{label}</text>'
        )
    if not series:
        lines.append(
            '<text x="480" y="240" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#6b645d">No persisted loss series</text>'
        )
    lines.append(
        '<text x="930" y="455" text-anchor="end" font-family="sans-serif" font-size="10" fill="#6b645d">Dashed = smoke engineering evidence</text>'
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


def _markdown(snapshot: dict[str, Any], source_digest: str) -> str:
    campaign = snapshot["campaign"]
    attempts = snapshot["attempts"]
    comparisons = snapshot["comparisons"]
    smoke = [item for item in attempts if item.get("stage") == "smoke_training"]
    full = [item for item in attempts if item.get("stage") == "full_training"]
    completed_full = [item for item in full if item.get("status") == "completed"]
    history = snapshot.get("autoresearch_history") or {}
    experiments = history.get("experiments") or []
    autoresearch_quality = any(
        item.get("result", {}).get("outcome") == "completed"
        and item.get("result", {}).get("provenance") == "real"
        for item in experiments
    )
    reference = campaign.get("autoresearch_reference") or {}
    reference_id = reference.get("proposal_id") or "not recorded"
    costs = campaign.get("recorded_costs") or {}
    lines = [
        "# Campaign Evidence Report",
        "",
        f"- Campaign: `{campaign.get('campaign_id', 'unknown')}`",
        f"- Objective: {campaign.get('objective', 'Not recorded')}",
        f"- Status: `{campaign.get('status', 'unknown')}`",
        f"- Current AutoResearch reference: `{reference_id}`",
        f"- Promoted campaign champion: `{campaign.get('champion_ref') or 'not promoted'}`",
        f"- Evidence digest: `{source_digest}`",
    ]
    if costs:
        lines.extend(
            [
                f"- Recorded baseline cost: `{_number(costs.get('baseline'))}`",
                f"- Recorded candidate cost: `{_number(costs.get('candidate'))}`",
                f"- Recorded campaign total cost: `{_number(costs.get('total'))}`",
            ]
        )
    lines.extend(["", "## Model-quality findings", ""])
    if completed_full and comparisons:
        latest = comparisons[-1]
        lines.extend(
            [
                f"The latest deterministic development gate verdict is **{latest.get('verdict', 'unknown')}**.",
                "",
                "This section is backed by at least one completed full-training attempt and a persisted comparison.",
            ]
        )
    elif autoresearch_quality:
        lines.extend(
            [
                "The experiment history below contains completed fixed-suite evaluation evidence.",
                "",
                "KEEP and DISCARD are reported from the persisted AutoResearch decisions.",
            ]
        )
    else:
        lines.append(
            "No model-quality findings are claimed. A completed full-training attempt and persisted comparison are both required."
        )
    if experiments:
        method_thresholds = {
            key: value
            for key, value in (history.get("method_thresholds") or {}).items()
            if key != "schema_version" and value is not None
        }
        lines.extend(
            [
                "",
                "## AutoResearch experiment history",
                "",
                f"Fixed evaluation suite: `{history.get('evaluation_suite_id') or 'not recorded'}`",
            ]
        )
        if method_thresholds:
            rendered_thresholds = ", ".join(
                f"{key}={_number(value)}" for key, value in sorted(method_thresholds.items())
            )
            lines.append(f"Method-readiness thresholds: `{rendered_thresholds}`.")
        lines.extend(
            [
                "",
                "| # | Role | Changed variable | Primary | Improvement | Decision |",
                "|---:|---|---|---:|---:|---|",
            ]
        )
        for index, item in enumerate(experiments, start=1):
            primary = item.get("performance", {}).get("primary", {})
            proposal = item.get("proposal", {})
            lines.append(
                "| "
                f"{index} | {_table_value(item.get('role'))} | "
                f"{_table_value(proposal.get('changed_variable'))} | "
                f"{_number(primary.get('candidate_value'))} | "
                f"{_signed_number(primary.get('improvement'))} | "
                f"{_table_value(item.get('decision', {}).get('decision'))} |"
            )
        for index, item in enumerate(experiments, start=1):
            proposal = item.get("proposal", {})
            performance = item.get("performance", {})
            primary = performance.get("primary", {})
            decision = item.get("decision", {})
            learning = item.get("learning", {})
            assessment = item.get("outcome_assessment") or {}
            lines.extend(
                [
                    "",
                    f"### {index}. {item.get('proposal_id', 'unknown')}",
                    "",
                    f"- Hypothesis: {proposal.get('hypothesis') or 'Not recorded.'}",
                    f"- Prediction: {proposal.get('expected_outcome') or 'Not recorded.'}",
                    f"- Falsifier: {proposal.get('falsification_criterion') or 'Not recorded.'}",
                    (
                        f"- Primary performance: `{primary.get('metric_name') or 'unknown'}` "
                        f"{_number(primary.get('candidate_value'))} vs "
                        f"{_number(primary.get('reference_value'))}; improvement "
                        f"{_signed_number(primary.get('improvement'))}; "
                        f"{_pass_label(primary.get('passed'))}."
                    ),
                    (
                        f"- Decision: `{decision.get('decision') or 'unknown'}` "
                        f"(`{decision.get('reason_code') or 'not_recorded'}`)."
                    ),
                    f"- Finding: {learning.get('summary') or 'No quality finding was recorded.'}",
                ]
            )
            if assessment:
                failure_label = (
                    f"failed experiment ({assessment.get('failure_kind') or 'unspecified'})"
                    if assessment.get("is_failure") is True
                    else "not a failed experiment"
                )
                lines.append(
                    f"- Outcome assessment: `{assessment.get('classification') or 'inconclusive'}`; "
                    f"{failure_label}."
                )
                tradeoffs = assessment.get("observed_tradeoffs") or []
                if tradeoffs:
                    lines.append(
                        "- Observed tradeoffs: "
                        + ", ".join(f"`{item}`" for item in tradeoffs)
                        + "."
                    )
                improvements = assessment.get("observed_improvements") or []
                if improvements:
                    lines.append(
                        "- Observed improvements: "
                        + ", ".join(f"`{item}`" for item in improvements)
                        + "."
                    )
                lines.append(
                    f"- Evidence strength: `{assessment.get('evidence_strength') or 'unknown'}`."
                )
            protected = performance.get("protected_metrics") or []
            if protected:
                lines.append("- Protected metrics:")
                for metric in protected:
                    lines.append(
                        "  - "
                        f"`{metric.get('metric_name', 'unknown')}`: "
                        f"{_number(metric.get('candidate_value'))} vs "
                        f"{_number(metric.get('reference_value'))}; regression "
                        f"{_number(metric.get('observed_regression'))}; limit "
                        f"{_number(metric.get('maximum_regression'))}; "
                        f"{_pass_label(metric.get('passed'))}"
                    )
            data = item.get("data") or {}
            quality = data.get("quality") or {}
            if quality:
                lines.append(
                    "- Data quality: "
                    f"{_number(quality.get('accepted_rows'))}/"
                    f"{_number(quality.get('generated_rows'))} accepted; verification pass rate "
                    f"{_number(quality.get('verification_pass_rate'))}."
                )
            failure_comparison = (item.get("failure_analysis") or {}).get("comparison") or []
            if failure_comparison:
                lines.append("- Behavioral error categories:")
                for comparison in failure_comparison:
                    lines.append(
                        "  - "
                        f"`{comparison.get('category', 'unknown')}`: "
                        f"{_number(comparison.get('reference_count'))} to "
                        f"{_number(comparison.get('candidate_count'))} "
                        f"({_signed_number(comparison.get('delta'))}; "
                        f"{comparison.get('status', 'unknown')})"
                    )
            references = item.get("evidence_references") or []
            if references:
                lines.append(
                    "- Evidence: " + ", ".join(f"`{reference}`" for reference in references)
                )
        powered_experiments = [item for item in experiments if item.get("experiment_power")]
        if powered_experiments:
            lines.extend(["", "## Experiment power", ""])
            for item in powered_experiments:
                power = item["experiment_power"]
                evaluation = power.get("evaluation") or {}
                seed = power.get("seed_uncertainty") or {}
                sequential = power.get("sequential_stopping") or {}
                lines.extend(
                    [
                        f"### {item.get('proposal_id', 'unknown')}",
                        "",
                        f"- Evaluation sample count: `{_number(evaluation.get('sample_count'))}`",
                        (
                            "- Sample-size sufficiency: "
                            f"`{(evaluation.get('sufficiency') or {}).get('status', 'not_assessed')}`"
                        ),
                        f"- Seed evidence: `{seed.get('status') or 'not_grouped'}`",
                        (
                            "- Sequential stopping: "
                            f"`{sequential.get('status') or 'not_predeclared'}`"
                        ),
                    ]
                )
                uncertainty = evaluation.get("uncertainty") or {}
                if uncertainty.get("interval_lower") is not None:
                    lines.append(
                        "- Evaluator-authored interval: "
                        f"{_number(uncertainty.get('interval_lower'))} to "
                        f"{_number(uncertainty.get('interval_upper'))} "
                        f"(`{uncertainty.get('method') or 'unknown'}`, "
                        f"confidence={_number(uncertainty.get('confidence_level'))})."
                    )
                for limitation in power.get("limitations") or []:
                    lines.append(f"- Limitation: {limitation}")
    hypothesis_families = history.get("hypothesis_families") or []
    if hypothesis_families:
        lines.extend(["", "## Hypothesis families", ""])
        for family in hypothesis_families:
            lifecycle = family.get("lifecycle") or {}
            conclusion = lifecycle.get("conclusion") or {}
            follow_up = lifecycle.get("follow_up") or {}
            lines.append(
                f"- `{family.get('hypothesis_family_id', 'unknown')}`: evidence "
                f"`{family.get('status', 'unknown')}`; lifecycle "
                f"`{lifecycle.get('status', 'open')}`."
            )
            if conclusion.get("summary"):
                lines.append(f"  - Conclusion: {conclusion['summary']}")
            if follow_up.get("hypothesis_family_id"):
                lines.append(
                    f"  - Follow-up `{follow_up['hypothesis_family_id']}`: "
                    f"{follow_up.get('hypothesis') or 'Not recorded.'}"
                )
    diagnostic_results = history.get("diagnostic_results") or []
    if diagnostic_results:
        lines.extend(["", "## Research diagnostics", ""])
        for item in diagnostic_results:
            lines.extend(
                [
                    f"### {item.get('probe_family', 'diagnostic')}",
                    "",
                    f"- Question: {item.get('question') or 'Not recorded.'}",
                    f"- Hypothesis: {item.get('hypothesis') or 'Not recorded.'}",
                    f"- Status: `{item.get('status') or 'unknown'}`",
                ]
            )
            measurements = item.get("measurements") or []
            if measurements:
                lines.append("- Measurements:")
                for measurement in measurements:
                    unit = measurement.get("unit") or "unitless"
                    lines.append(
                        "  - "
                        f"`{measurement.get('name', 'unknown')}` = "
                        f"{_number(measurement.get('value'))} {unit} "
                        f"(n={_number(measurement.get('sample_count'))})"
                    )
            comparison_contract = item.get("comparison_contract") or {}
            if item.get("probe_family") == "reward_integrity_probe" and isinstance(
                comparison_contract, dict
            ):
                reward_spec_digest = comparison_contract.get("reward_spec_digest")
                canary_suite_id = comparison_contract.get("canary_suite_id")
                if reward_spec_digest:
                    lines.append(f"- Reward spec: `{reward_spec_digest}`")
                if canary_suite_id:
                    lines.append(f"- Canary suite: `{canary_suite_id}`")
            if item.get("probe_family") == "preference_integrity_probe" and isinstance(
                comparison_contract, dict
            ):
                preference_dataset_digest = comparison_contract.get("preference_dataset_digest")
                labeling_contract_digest = comparison_contract.get("labeling_contract_digest")
                if preference_dataset_digest:
                    lines.append(f"- Preference dataset: `{preference_dataset_digest}`")
                if labeling_contract_digest:
                    lines.append(f"- Labeling contract: `{labeling_contract_digest}`")
            observations = item.get("observations") or []
            if observations:
                lines.append("- Observations:")
                for observation in observations:
                    lines.append(
                        "  - "
                        f"{observation.get('summary', 'Not recorded.')} "
                        f"(count={_number(observation.get('count'))})"
                    )
            if item.get("unsupported_reason"):
                lines.append(f"- Unsupported reason: `{item.get('unsupported_reason')}`")
    lines.extend(["", "## Engineering evidence", ""])
    lines.append(f"- Smoke attempts: {len(smoke)} (runtime/semantics/memory evidence only)")
    lines.append(f"- Full-training attempts: {len(full)}")
    lines.append(f"- Persisted comparison records: {len(comparisons)}")
    lines.extend(
        [
            "",
            "## Attempts",
            "",
            "| Attempt | Stage | Status | Candidate digest |",
            "|---|---|---|---|",
        ]
    )
    for item in attempts:
        digest = str(item.get("candidate_digest", ""))
        lines.append(
            f"| `{item.get('attempt_id', '')}` | {item.get('stage', '')} | {item.get('status', '')} | `{digest[:12]}` |"
        )
    lines.extend(
        ["", "## Sealed evidence", "", "| Schema | SHA-256 | Bytes | Valid |", "|---|---|---:|---|"]
    )
    for item in snapshot["artifacts"]:
        lines.append(
            f"| {item.get('schema_name', '')} | `{item.get('sha256', '')}` | {item.get('size_bytes', 0)} | {item.get('valid', False)} |"
        )
    lines.extend(["", "## Flags", ""])
    if snapshot["flags"]:
        lines.extend(f"- {flag}" for flag in snapshot["flags"])
    else:
        lines.append("- No implementation flags were recorded for this export.")
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            "Every table and chart in this package is derived from `campaign_evidence.json`; `export_manifest.json` records the SHA-256 of each generated file.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:g}"


def _signed_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:+g}"


def _pass_label(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "not applicable"


def _table_value(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def export_campaign_evidence(
    value: CampaignExportSnapshot,
    output_directory: Path,
) -> dict[str, Any]:
    """Write deterministic evidence, chart, Word, and PDF projections."""

    snapshot = value.safe_payload()
    source_digest = canonical_hash(snapshot)
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise CampaignExportError("campaign_export_directory_not_empty")

    evidence_path = output_directory / "campaign_evidence.json"
    evidence_path.write_bytes(_json_bytes(snapshot) + b"\n")
    _write_csv(
        output_directory / "attempts.csv",
        (
            "attempt_id",
            "study_id",
            "stage",
            "status",
            "candidate_digest",
            "created_at",
            "updated_at",
        ),
        snapshot["attempts"],
    )
    _write_csv(
        output_directory / "artifacts.csv",
        (
            "artifact_id",
            "producer_action_id",
            "schema_name",
            "sha256",
            "size_bytes",
            "sealed",
            "valid",
            "created_at",
        ),
        snapshot["artifacts"],
    )
    _write_csv(
        output_directory / "comparisons.csv",
        (
            "comparison_digest",
            "champion_digest",
            "candidate_digest",
            "sample_count",
            "verdict",
            "blocking_reasons",
            "warnings",
            "created_at",
        ),
        snapshot["comparisons"],
    )
    (output_directory / "training_loss.svg").write_text(_loss_svg(snapshot), encoding="utf-8")
    loss_png = output_directory / "training_loss.png"
    write_loss_png(snapshot, loss_png)
    (output_directory / "campaign_report.md").write_text(
        _markdown(snapshot, source_digest), encoding="utf-8", newline="\n"
    )
    write_campaign_docx(
        snapshot,
        source_digest,
        loss_png,
        output_directory / "campaign_report.docx",
    )
    write_campaign_pdf(
        snapshot,
        source_digest,
        loss_png,
        output_directory / "campaign_report.pdf",
    )

    files = []
    for path in sorted(output_directory.iterdir(), key=lambda item: item.name):
        files.append(
            {"name": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    manifest = {
        "schema_version": "campaign_export_manifest.v1",
        "campaign_id": snapshot["campaign"].get("campaign_id"),
        "source_digest": source_digest,
        "quality_findings_available": bool(
            (
                any(
                    item.get("stage") == "full_training" and item.get("status") == "completed"
                    for item in snapshot["attempts"]
                )
                and snapshot["comparisons"]
            )
            or any(
                item.get("result", {}).get("outcome") == "completed"
                and item.get("result", {}).get("provenance") == "real"
                for item in (snapshot.get("autoresearch_history") or {}).get("experiments", [])
            )
        ),
        "files": files,
    }
    (output_directory / "export_manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
    return manifest


__all__ = [
    "CampaignExportError",
    "CampaignExportSnapshot",
    "export_campaign_evidence",
]
