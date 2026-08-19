#!/usr/bin/env python3
"""Validate and render the public AutoResearch publication contract."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 2_000_000
SCHEMA_VERSION = "open_frontiers.research_post.v1"
ROOT_FIELDS = {
    "schema_version",
    "publication",
    "experiment",
    "results",
    "narrative",
    "training_rungs",
    "visuals",
    "claims",
    "sources",
    "provenance",
}
PRIVATE_PATTERNS = (
    ("private path", re.compile(r"(?i)(?:[a-z]:\\|/(?:users|home|var|tmp|etc)/)")),
    ("private URI", re.compile(r"(?i)\b(?:file|ssh)://")),
    ("IP address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    (
        "credential material",
        re.compile(r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]"),
    ),
)


class ValidationError(ValueError):
    """Raised when a research-post package violates the public contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be an array")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    if "REPLACE_" in value:
        raise ValidationError(f"{name} still contains a template marker")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number


def _required(mapping: dict[str, Any], fields: set[str], name: str) -> None:
    missing = sorted(fields - mapping.keys())
    if missing:
        raise ValidationError(f"{name} missing required fields: {', '.join(missing)}")


def _reject_unexpected(mapping: dict[str, Any], fields: set[str], name: str) -> None:
    unexpected = sorted(mapping.keys() - fields)
    if unexpected:
        raise ValidationError(f"{name} has unexpected fields: {', '.join(unexpected)}")


def _validate_public_boundary(value: Any, name: str = "document") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_public_boundary(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_boundary(child, f"{name}[{index}]")
    elif isinstance(value, str):
        for label, pattern in PRIVATE_PATTERNS:
            if pattern.search(value):
                raise ValidationError(f"{name} contains {label}")


def validate_post(document: Any) -> dict[str, Any]:
    post = _mapping(document, "document")
    _required(post, ROOT_FIELDS, "document")
    unexpected = sorted(post.keys() - ROOT_FIELDS)
    if unexpected:
        raise ValidationError(f"document has unexpected fields: {', '.join(unexpected)}")
    if post["schema_version"] != SCHEMA_VERSION:
        raise ValidationError(f"schema_version must be {SCHEMA_VERSION}")

    publication = _mapping(post["publication"], "publication")
    _required(publication, {"slug", "title", "summary", "approval"}, "publication")
    for field in ("slug", "title", "summary"):
        _text(publication.get(field), f"publication.{field}")
    approval = _mapping(publication["approval"], "publication.approval")
    _required(
        approval,
        {"status", "approved_by", "approved_at", "feedback"},
        "publication.approval",
    )
    status = approval.get("status")
    if status not in {"draft", "approved"}:
        raise ValidationError("publication.approval.status must be draft or approved")
    if status == "approved":
        try:
            _text(approval.get("approved_by"), "publication.approval.approved_by")
            _text(approval.get("approved_at"), "publication.approval.approved_at")
        except ValidationError as error:
            raise ValidationError(
                "approved publication requires explicit human approval"
            ) from error
    elif approval.get("approved_by") is not None or approval.get("approved_at") is not None:
        raise ValidationError("draft publication cannot carry human approval metadata")
    feedback = _list(approval.get("feedback"), "publication.approval.feedback")
    unresolved_feedback = False
    for index, value in enumerate(feedback):
        item = _mapping(value, f"publication.approval.feedback[{index}]")
        _required(item, {"note", "status"}, f"publication.approval.feedback[{index}]")
        _text(item.get("note"), f"publication.approval.feedback[{index}].note")
        if item.get("status") not in {"open", "addressed", "declined"}:
            raise ValidationError(f"publication.approval.feedback[{index}].status is invalid")
        unresolved_feedback |= item.get("status") == "open"
    if status == "approved" and unresolved_feedback:
        raise ValidationError("approved publication has unresolved feedback")

    experiment = _mapping(post["experiment"], "experiment")
    experiment_fields = {"question", "hypothesis", "model", "method", "intervention", "evaluation"}
    _required(experiment, experiment_fields, "experiment")
    _reject_unexpected(experiment, experiment_fields | {"method_selection"}, "experiment")
    for field in experiment_fields:
        _text(experiment.get(field), f"experiment.{field}")
    method_selection = experiment.get("method_selection")
    if method_selection is not None:
        method_selection = _mapping(method_selection, "experiment.method_selection")
        method_fields = {
            "selected_method",
            "selection_authority",
            "rationale",
            "alternatives",
        }
        _required(method_selection, method_fields, "experiment.method_selection")
        _reject_unexpected(method_selection, method_fields, "experiment.method_selection")
        _text(
            method_selection.get("selected_method"), "experiment.method_selection.selected_method"
        )
        if method_selection.get("selection_authority") not in {"host_agent", "human"}:
            raise ValidationError("experiment.method_selection.selection_authority is invalid")
        _text(method_selection.get("rationale"), "experiment.method_selection.rationale")
        alternatives = _list(
            method_selection.get("alternatives"), "experiment.method_selection.alternatives"
        )
        if len(alternatives) > 6:
            raise ValidationError("experiment.method_selection.alternatives exceeds 6 items")
        alternative_fields = {"method", "status", "reason"}
        for index, value in enumerate(alternatives):
            alternative = _mapping(value, f"experiment.method_selection.alternatives[{index}]")
            _required(
                alternative,
                alternative_fields,
                f"experiment.method_selection.alternatives[{index}]",
            )
            _reject_unexpected(
                alternative,
                alternative_fields,
                f"experiment.method_selection.alternatives[{index}]",
            )
            _text(
                alternative.get("method"),
                f"experiment.method_selection.alternatives[{index}].method",
            )
            if alternative.get("status") not in {
                "eligible",
                "not_selected",
                "blocked",
                "diagnostic_needed",
                "unsupported_by_runner",
            }:
                raise ValidationError(
                    f"experiment.method_selection.alternatives[{index}].status is invalid"
                )
            _text(
                alternative.get("reason"),
                f"experiment.method_selection.alternatives[{index}].reason",
            )

    results = _mapping(post["results"], "results")
    _required(results, {"primary", "secondary", "decision"}, "results")
    _reject_unexpected(results, {"primary", "secondary", "decision", "failure_analysis"}, "results")
    primary = _mapping(results["primary"], "results.primary")
    primary_fields = {"name", "unit", "baseline", "candidate", "delta", "direction"}
    _required(primary, primary_fields, "results.primary")
    for field in ("name", "unit"):
        _text(primary.get(field), f"results.primary.{field}")
    if primary.get("direction") not in {"higher_is_better", "lower_is_better"}:
        raise ValidationError("results.primary.direction is invalid")
    baseline = _finite(primary.get("baseline"), "results.primary.baseline")
    candidate = _finite(primary.get("candidate"), "results.primary.candidate")
    delta = _finite(primary.get("delta"), "results.primary.delta")
    if not math.isclose(candidate - baseline, delta, rel_tol=1e-9, abs_tol=1e-12):
        raise ValidationError("primary delta does not equal candidate minus baseline")
    _list(results["secondary"], "results.secondary")
    if results.get("decision") not in {"baseline", "keep", "discard"}:
        raise ValidationError("results.decision must be baseline, keep, or discard")
    failures = _list(results.get("failure_analysis", []), "results.failure_analysis")
    if len(failures) > 12:
        raise ValidationError("results.failure_analysis exceeds 12 items")
    failure_fields = {
        "category",
        "summary",
        "baseline_count",
        "candidate_count",
        "delta",
        "status",
    }
    for index, value in enumerate(failures):
        failure = _mapping(value, f"results.failure_analysis[{index}]")
        _required(failure, failure_fields, f"results.failure_analysis[{index}]")
        _reject_unexpected(failure, failure_fields, f"results.failure_analysis[{index}]")
        _text(failure.get("category"), f"results.failure_analysis[{index}].category")
        _text(failure.get("summary"), f"results.failure_analysis[{index}].summary")
        counts = []
        for field in ("baseline_count", "candidate_count", "delta"):
            count = failure.get(field)
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValidationError(f"results.failure_analysis[{index}].{field} must be integer")
            counts.append(count)
        baseline_count, candidate_count, failure_delta = counts
        if baseline_count < 0 or candidate_count < 0:
            raise ValidationError("failure counts cannot be negative")
        if candidate_count - baseline_count != failure_delta:
            raise ValidationError("failure delta does not equal candidate minus baseline")
        expected_status = (
            "improved" if failure_delta < 0 else "regressed" if failure_delta > 0 else "unchanged"
        )
        if failure.get("status") != expected_status:
            raise ValidationError(f"results.failure_analysis[{index}].status is inconsistent")

    narrative = _mapping(post["narrative"], "narrative")
    narrative_fields = {"simple", "technical", "judgement", "limitations", "next_experiment"}
    _required(narrative, narrative_fields, "narrative")
    for field in ("simple", "technical", "judgement", "next_experiment"):
        _text(narrative.get(field), f"narrative.{field}")
    limitations = _list(narrative["limitations"], "narrative.limitations")
    if not limitations:
        raise ValidationError("narrative.limitations must include at least one limitation")
    for index, limitation in enumerate(limitations):
        _text(limitation, f"narrative.limitations[{index}]")

    rungs = _list(post["training_rungs"], "training_rungs")
    if len(rungs) < 2:
        raise ValidationError("training_rungs must include at least two rungs")
    orders: list[int] = []
    for index, value in enumerate(rungs):
        rung = _mapping(value, f"training_rungs[{index}]")
        _required(
            rung, {"order", "label", "method", "status", "summary"}, f"training_rungs[{index}]"
        )
        order = rung.get("order")
        if isinstance(order, bool) or not isinstance(order, int):
            raise ValidationError(f"training_rungs[{index}].order must be an integer")
        orders.append(order)
        for field in ("label", "method", "summary"):
            _text(rung.get(field), f"training_rungs[{index}].{field}")
        if rung.get("status") not in {"planned", "completed", "kept", "discarded", "failed"}:
            raise ValidationError(f"training_rungs[{index}].status is invalid")
    if orders != list(range(1, len(rungs) + 1)):
        raise ValidationError("training_rungs orders must be consecutive and begin at 1")

    visuals = _list(post["visuals"], "visuals")
    visual_ids: set[str] = set()
    visual_types: set[str] = set()
    for index, value in enumerate(visuals):
        visual = _mapping(value, f"visuals[{index}]")
        _required(visual, {"id", "type", "title", "data"}, f"visuals[{index}]")
        visual_id = _text(visual.get("id"), f"visuals[{index}].id")
        if visual_id in visual_ids:
            raise ValidationError(f"duplicate visual id: {visual_id}")
        visual_ids.add(visual_id)
        visual_types.add(_text(visual.get("type"), f"visuals[{index}].type"))
        _text(visual.get("title"), f"visuals[{index}].title")
        _mapping(visual.get("data"), f"visuals[{index}].data")
    missing_visuals = {"metric_comparison", "training_rungs"} - visual_types
    if missing_visuals:
        raise ValidationError(
            f"visuals missing required types: {', '.join(sorted(missing_visuals))}"
        )

    evidence = _list(
        _mapping(post["provenance"], "provenance").get("evidence"), "provenance.evidence"
    )
    if not evidence:
        raise ValidationError("provenance.evidence must not be empty")
    evidence_ids: set[str] = set()
    for index, value in enumerate(evidence):
        item = _mapping(value, f"provenance.evidence[{index}]")
        _required(item, {"id", "kind", "digest"}, f"provenance.evidence[{index}]")
        evidence_id = _text(item.get("id"), f"provenance.evidence[{index}].id")
        evidence_ids.add(evidence_id)
        _text(item.get("kind"), f"provenance.evidence[{index}].kind")
        digest = _text(item.get("digest"), f"provenance.evidence[{index}].digest")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError(f"provenance.evidence[{index}].digest must be lowercase SHA-256")
    _text(post["provenance"].get("generated_at"), "provenance.generated_at")

    sources = _list(post["sources"], "sources")
    source_ids: set[str] = set()
    for index, value in enumerate(sources):
        source = _mapping(value, f"sources[{index}]")
        _required(source, {"id", "title", "url", "year", "role"}, f"sources[{index}]")
        source_id = _text(source.get("id"), f"sources[{index}].id")
        source_ids.add(source_id)
        for field in ("title", "url", "role"):
            _text(source.get(field), f"sources[{index}].{field}")
        year = source.get("year")
        if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
            raise ValidationError(f"sources[{index}].year is invalid")

    claims = _list(post["claims"], "claims")
    if not claims:
        raise ValidationError("claims must not be empty")
    for index, value in enumerate(claims):
        claim = _mapping(value, f"claims[{index}]")
        _required(claim, {"id", "text", "evidence_refs", "source_refs"}, f"claims[{index}]")
        _text(claim.get("id"), f"claims[{index}].id")
        _text(claim.get("text"), f"claims[{index}].text")
        claim_evidence = _list(claim.get("evidence_refs"), f"claims[{index}].evidence_refs")
        if not claim_evidence:
            raise ValidationError(f"claims[{index}] needs at least one evidence reference")
        unknown_evidence = sorted(set(claim_evidence) - evidence_ids)
        if unknown_evidence:
            raise ValidationError(
                f"claims[{index}] references unknown evidence: {', '.join(unknown_evidence)}"
            )
        claim_sources = _list(claim.get("source_refs"), f"claims[{index}].source_refs")
        unknown_sources = sorted(set(claim_sources) - source_ids)
        if unknown_sources:
            raise ValidationError(
                f"claims[{index}] references unknown sources: {', '.join(unknown_sources)}"
            )

    _validate_public_boundary(post)
    return post


def _load(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValidationError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    try:
        return validate_post(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON: {error.msg}") from error


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(post: dict[str, Any]) -> str:
    publication = post["publication"]
    experiment = post["experiment"]
    results = post["results"]
    primary = results["primary"]
    narrative = post["narrative"]
    approval = publication["approval"]
    approval_text = (
        f"Approved by {approval['approved_by']} at {approval['approved_at']}"
        if approval["status"] == "approved"
        else "Draft — human approval required"
    )
    lines = [
        f"# {publication['title']}",
        "",
        approval_text,
        "",
        publication["summary"],
        "",
        "## Experiment",
        "",
        f"**Question:** {experiment['question']}",
        "",
        f"**Hypothesis:** {experiment['hypothesis']}",
        "",
        f"**Model:** {experiment['model']}",
        "",
        f"**Method:** {experiment['method']}",
        "",
        f"**Controlled intervention:** {experiment['intervention']}",
        "",
        f"**Fixed evaluation:** {experiment['evaluation']}",
        "",
        "## Result",
        "",
        "| Metric | Baseline | Candidate | Delta | Decision |",
        "|---|---:|---:|---:|---|",
        f"| {_escape(primary['name'])} | {primary['baseline']} | {primary['candidate']} | {primary['delta']} | {results['decision']} |",
        "",
        "## In plain language",
        "",
        narrative["simple"],
        "",
        "## Technical interpretation",
        "",
        narrative["technical"],
    ]
    method_selection = experiment.get("method_selection")
    if method_selection is not None:
        lines.extend(
            [
                "",
                "## Method selection",
                "",
                f"**Selected:** {method_selection['selected_method']}",
                "",
                method_selection["rationale"],
            ]
        )
        if method_selection["alternatives"]:
            lines.extend(["", "| Alternative | Status | Reason |", "|---|---|---|"])
            for alternative in method_selection["alternatives"]:
                lines.append(
                    f"| {_escape(alternative['method'])} | {alternative['status']} | {_escape(alternative['reason'])} |"
                )
    failures = results.get("failure_analysis", [])
    if failures:
        lines.extend(
            [
                "",
                "## Failure analysis",
                "",
                "| Category | Baseline | Candidate | Delta | Status | Summary |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for failure in failures:
            lines.append(
                f"| {_escape(failure['category'])} | {failure['baseline_count']} | {failure['candidate_count']} | {failure['delta']} | {failure['status']} | {_escape(failure['summary'])} |"
            )
    lines.extend(
        [
            "",
            "## Training rungs",
            "",
            "| # | Rung | Method | Status | Summary |",
            "|---:|---|---|---|---|",
        ]
    )
    for rung in post["training_rungs"]:
        lines.append(
            f"| {rung['order']} | {_escape(rung['label'])} | {_escape(rung['method'])} | {rung['status']} | {_escape(rung['summary'])} |"
        )
    lines.extend(["", "## Judgment", "", narrative["judgement"], "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in narrative["limitations"])
    lines.extend(["", "## Next experiment", "", narrative["next_experiment"]])
    if approval["feedback"]:
        lines.extend(["", "## Review feedback", ""])
        lines.extend(f"- [{item['status']}] {item['note']}" for item in approval["feedback"])
    if post["sources"]:
        lines.extend(["", "## Sources", ""])
        lines.extend(f"- [{source['title']}]({source['url']})" for source in post["sources"])
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            *[f"- `{item['kind']}`: `{item['digest']}`" for item in post["provenance"]["evidence"]],
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("input", type=Path)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--output", required=True, type=Path)
    render_parser.add_argument("input", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        post = _load(args.input)
        if args.command == "render":
            args.output.write_text(render_markdown(post), encoding="utf-8")
        else:
            print(json.dumps({"ok": True, "schema_version": SCHEMA_VERSION}))
        return 0
    except (OSError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
