"""Pure agent-facing goal and Markdown projections of experiment state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime


def build_agent_brief(
    snapshot: object,
    *,
    control_room_url: str | None = None,
    stop_conditions: Sequence[str] = (),
) -> dict[str, object]:
    """Project one experiment snapshot without storing or changing campaign state."""

    data = _snapshot_mapping(snapshot)
    workspace_id = _required_text(data, "workspace_id")
    campaign_id = _required_text(data, "campaign_id")
    campaign = _mapping(data.get("campaign"))
    objective = _required_text(campaign, "objective")
    journey = _mapping_items(data.get("journey"))
    active_work = _optional_mapping(data.get("active_work"))
    decision_surface = _mapping(data.get("decision_surface"))

    completed_work = [
        {
            "phase": _text(phase.get("phase_id")),
            "evidence_count": _integer(phase.get("evidence_count")),
        }
        for phase in journey
        if phase.get("state") == "complete" and _text(phase.get("phase_id"))
    ]
    current_work = _current_work(journey, active_work)
    next_action = _next_action(decision_surface, journey)
    stop_reason = _text(campaign.get("stop_reason"))
    configured_stop_conditions = [
        condition for value in stop_conditions if (condition := _text(value)) is not None
    ]
    if stop_reason is not None and stop_reason not in configured_stop_conditions:
        configured_stop_conditions.append(stop_reason)
    goal = {
        "objective": objective,
        "success_criteria": _success_criteria(data.get("metrics")),
        "stop_conditions": configured_stop_conditions,
        "completed_work": completed_work,
        "current_work": current_work,
        "next_action": next_action,
        "resume": {
            "reference": f"{workspace_id}/{campaign_id}",
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "aggregate_version": _required_integer(data, "aggregate_version"),
            "manifest_revision": _required_integer(data, "manifest_revision"),
            "authorization_revision": _required_integer(data, "authorization_revision"),
            "latest_event_cursor": _required_integer(data, "latest_event_cursor"),
            "snapshot_at": _timestamp(data.get("snapshot_at")),
        },
        "control_room": {
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "url": _text(control_room_url),
        },
    }
    return {
        "schema_version": "bashgym.agent_brief.v1",
        "goal": goal,
        "markdown": _render_markdown(data, goal),
    }


def _snapshot_mapping(snapshot: object) -> Mapping[str, object]:
    if isinstance(snapshot, Mapping):
        return snapshot
    model_dump = getattr(snapshot, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        if isinstance(value, Mapping):
            return value
    raise TypeError("snapshot must be a mapping or expose model_dump(mode='json')")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _required_text(source: Mapping[str, object], key: str) -> str:
    value = _text(source.get(key))
    if value is None:
        raise ValueError(f"snapshot requires non-empty {key}")
    return value


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_integer(source: Mapping[str, object], key: str) -> int:
    value = _integer(source.get(key))
    if value is None:
        raise ValueError(f"snapshot requires integer {key}")
    return value


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _timestamp(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _success_criteria(value: object) -> list[str]:
    criteria: list[str] = []
    for metric in _mapping_items(value):
        target = _number(metric.get("target"))
        name = _text(metric.get("display_name")) or _text(metric.get("metric_id"))
        if target is None or name is None:
            continue
        criterion = f"{name} target {_format_number(target)}"
        unit = _text(metric.get("unit"))
        if unit:
            criterion += f" {unit}"
        direction = _text(metric.get("direction"))
        if direction:
            criterion += f" ({direction})"
        criteria.append(criterion)
    return criteria


def _current_phase(
    journey: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    for state in ("active", "blocked", "ready", "failed"):
        for phase in journey:
            if phase.get("state") == state:
                return phase
    return None


def _current_work(
    journey: tuple[Mapping[str, object], ...],
    active_work: Mapping[str, object] | None,
) -> dict[str, object] | None:
    phase = _current_phase(journey)
    if phase is None and active_work is None:
        return None
    blocker = _mapping(phase.get("primary_blocker")) if phase else {}
    summary = None
    if active_work is not None:
        summary = _text(active_work.get("hypothesis_summary")) or _text(
            active_work.get("primary_variable_summary")
        )
    summary = summary or _text(blocker.get("summary"))
    return {
        "phase": _text(phase.get("phase_id")) if phase else None,
        "state": _text(phase.get("state")) if phase else None,
        "stage": _text(active_work.get("stage")) if active_work else None,
        "summary": summary,
        "progress_fraction": (
            _number(active_work.get("progress_fraction")) if active_work else None
        ),
        "eta_seconds": _number(active_work.get("eta_seconds")) if active_work else None,
    }


def _next_action(
    decision_surface: Mapping[str, object],
    journey: tuple[Mapping[str, object], ...],
) -> dict[str, object] | None:
    actions = _mapping_items(decision_surface.get("next_actions"))
    if actions:
        action = actions[0]
        action_id = _text(action.get("action"))
        if action_id:
            return {
                "action": action_id,
                "capability": _text(action.get("capability")),
                "freshness_class": _text(action.get("freshness_class")),
                "requires_human_work": (
                    action.get("requires_human_work")
                    if isinstance(action.get("requires_human_work"), bool)
                    else None
                ),
                "source": "decision_surface",
            }
    recovery_actions = decision_surface.get("recovery_actions")
    if isinstance(recovery_actions, Sequence) and not isinstance(recovery_actions, (str, bytes)):
        for value in recovery_actions:
            action_id = _text(value)
            if action_id:
                return {
                    "action": action_id,
                    "capability": None,
                    "freshness_class": None,
                    "requires_human_work": None,
                    "source": "recovery",
                }
    phase = _current_phase(journey)
    phase_actions = phase.get("next_action_ids") if phase else None
    if isinstance(phase_actions, Sequence) and not isinstance(phase_actions, (str, bytes)):
        for value in phase_actions:
            action_id = _text(value)
            if action_id:
                return {
                    "action": action_id,
                    "capability": None,
                    "freshness_class": None,
                    "requires_human_work": None,
                    "source": "journey",
                }
    return None


def _render_markdown(data: Mapping[str, object], goal: Mapping[str, object]) -> str:
    campaign = _mapping(data.get("campaign"))
    current = _mapping(goal.get("current_work"))
    next_action = _mapping(goal.get("next_action"))
    title = _text(campaign.get("title")) or _required_text(data, "campaign_id")
    lines = [
        f"### AutoResearch · {title}",
        "",
        f"**Objective:** {_required_text(campaign, 'objective')}",
        f"**Timeline:** {_timeline(data.get('journey'))}",
        f"**Current stage:** {_stage_summary(campaign, current)}",
        f"**Baseline/champion vs candidate:** {_candidate_comparison(data)}",
        f"**Budget:** {_budget_summary(data.get('budget'))}",
        f"**Latest finding:** {_latest_finding(data)}",
        f"**Next action:** {_next_action_summary(next_action)}",
    ]
    success_criteria = goal.get("success_criteria")
    if isinstance(success_criteria, Sequence) and success_criteria:
        lines.append(
            "**Success criteria:** "
            + "; ".join(item for item in success_criteria if isinstance(item, str))
        )
    resume = _mapping(goal.get("resume"))
    lines.append(
        "**Resume:** "
        f"{_text(resume.get('reference')) or 'Not reported'} · "
        f"aggregate v{resume.get('aggregate_version')} · "
        f"manifest v{resume.get('manifest_revision')} · "
        f"cursor {resume.get('latest_event_cursor')}"
    )
    control_room = _mapping(goal.get("control_room"))
    url = _text(control_room.get("url"))
    if url:
        lines.append(f"[Open experiment view]({url})")
    return "\n".join(lines)


def _timeline(value: object) -> str:
    journey = _mapping_items(value)
    if not journey:
        return "Not reported in this snapshot."
    state_icons = {
        "complete": "✓",
        "active": "●",
        "blocked": "⚠",
        "ready": "◐",
        "failed": "✕",
        "skipped": "–",
        "not_started": "○",
    }
    items = []
    for phase in journey:
        phase_id = _text(phase.get("phase_id"))
        state = _text(phase.get("state"))
        if phase_id:
            items.append(f"{_humanize(phase_id)} {state_icons.get(state or '', '?')}")
    return " → ".join(items) or "Not reported in this snapshot."


def _stage_summary(campaign: Mapping[str, object], current: Mapping[str, object]) -> str:
    stage = _text(current.get("stage"))
    if stage:
        parts = [_humanize(stage)]
        progress = _number(current.get("progress_fraction"))
        if progress is not None:
            parts.append(f"{_format_number(progress * 100)}%")
        eta = _number(current.get("eta_seconds"))
        if eta is not None:
            parts.append(f"ETA {_format_duration(eta)}")
        return " · ".join(parts)
    phase = _text(current.get("phase"))
    state = _text(current.get("state"))
    if phase:
        return f"{_humanize(phase)} ({_humanize(state) if state else 'state unreported'})"
    return _humanize(_text(campaign.get("status"))) or "Not reported in this snapshot."


def _candidate_comparison(data: Mapping[str, object]) -> str:
    champion = _optional_mapping(data.get("champion"))
    candidate = _optional_mapping(data.get("candidate"))
    if champion is None and candidate is None:
        return "Not reported in this snapshot."
    return f"{_candidate_label(champion)} → {_candidate_label(candidate)}"


def _candidate_label(candidate: Mapping[str, object] | None) -> str:
    if candidate is None:
        return "not reported"
    reference = _text(candidate.get("candidate_ref")) or "unnamed candidate"
    details = []
    verdict = _text(candidate.get("comparison_verdict"))
    gate_state = _text(candidate.get("gate_state"))
    evaluation = _text(candidate.get("latest_comparable_evaluation_id"))
    if verdict:
        details.append(_humanize(verdict))
    elif gate_state:
        details.append(_humanize(gate_state))
    if evaluation:
        details.append(f"evaluation {evaluation}")
    return f"{reference} ({'; '.join(details)})" if details else reference


def _budget_summary(value: object) -> str:
    budget = _mapping(value)
    resources = _mapping_items(budget.get("resources"))
    if not resources:
        return "No resource usage reported in this snapshot."
    summaries = []
    for resource in resources:
        unit = _text(resource.get("unit")) or "resource"
        settled = _number(resource.get("settled"))
        reserved = _number(resource.get("reserved"))
        limit = _number(resource.get("limit"))
        remaining = _number(resource.get("remaining"))
        if None in (settled, reserved, limit, remaining):
            summaries.append(f"{unit}: usage values incomplete")
            continue
        summary = (
            f"{unit}: {_format_number(settled)} settled + "
            f"{_format_number(reserved)} reserved / {_format_number(limit)}; "
            f"{_format_number(remaining)} remaining"
        )
        if resource.get("blocked") is True:
            blocker = _text(resource.get("blocker_code"))
            summary += f"; blocked{f' ({blocker})' if blocker else ''}"
        summaries.append(summary)
    return "; ".join(summaries)


def _latest_finding(data: Mapping[str, object]) -> str:
    campaign = _mapping(data.get("campaign"))
    stop_reason = _text(campaign.get("stop_reason"))
    if stop_reason:
        return f"Campaign stop reason: {stop_reason}."
    candidate = _optional_mapping(data.get("candidate"))
    if candidate is not None:
        verdict = _text(candidate.get("comparison_verdict"))
        if verdict:
            return f"Candidate comparison verdict: {_humanize(verdict)}."
    decision_blocker = _mapping(_mapping(data.get("decision_surface")).get("blocker"))
    blocker_summary = _text(decision_blocker.get("summary"))
    if blocker_summary:
        return blocker_summary
    phase = _current_phase(_mapping_items(data.get("journey")))
    phase_blocker = _mapping(phase.get("primary_blocker")) if phase else {}
    return _text(phase_blocker.get("summary")) or "Not reported in this snapshot."


def _next_action_summary(action: Mapping[str, object]) -> str:
    action_id = _text(action.get("action"))
    if not action_id:
        return "None reported."
    source = _text(action.get("source"))
    return f"{action_id} ({source})" if source in {"recovery", "journey"} else action_id


def _format_number(value: int | float) -> str:
    return f"{value:.15g}" if isinstance(value, float) else str(value)


def _format_duration(seconds: int | float) -> str:
    if seconds < 60:
        return f"{_format_number(seconds)}s"
    if seconds < 3600:
        return f"{_format_number(seconds / 60)}m"
    return f"{_format_number(seconds / 3600)}h"


def _humanize(value: str | None) -> str:
    return value.replace("_", " ") if value else ""


__all__ = ["build_agent_brief"]
