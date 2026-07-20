"""Workspace-scoped Skill Lab contracts and paired evaluation runs."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from bashgym.api import agent_routes

router = APIRouter(prefix="/api/skill-lab", tags=["skill-lab"])

_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "min_uplift": 0.1,
    "min_forced_pass_rate": 0.8,
    "min_routing_precision": 0.9,
    "min_routing_recall": 0.85,
    "max_false_activation_rate": 0.05,
}
_THRESHOLD_ALIASES = {
    "min_precision": "min_routing_precision",
    "min_recall": "min_routing_recall",
    "max_false_activation": "max_false_activation_rate",
}
_MARKER_RE = re.compile(r"(?im)^\s*SKILL_LAB_SHOULD_INVOKE\s*:\s*(true|false)\s*$")
_PROTECTED_SKILL_NAMES = frozenset({"bashgym", "bashgym-operator", "training"})


class SkillLabContractRequest(BaseModel):
    endpoint_id: str | None = None
    cases: list[dict[str, Any]] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class SkillLabRunRequest(SkillLabContractRequest):
    workspace_id: str
    skill_id: str
    endpoint_id: str
    suite_id: str | None = None


class SkillLabPlanRequest(BaseModel):
    workspace_id: str
    skill_id: str
    endpoint_id: str
    goal: str = ""
    depth: Literal["quick", "thorough"] = "quick"


class SkillLabRun(BaseModel):
    run_id: str
    workspace_id: str
    skill_id: str
    skill_name: str
    skill_revision: str
    endpoint_id: str
    model: str = ""
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: dict[str, int]
    kpis: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    suite_id: str | None = None


class SkillLabContract(BaseModel):
    skill_id: str
    workspace_id: str
    endpoint_id: str | None = None
    cases: list[dict[str, Any]]
    thresholds: dict[str, Any]
    updated_at: str


class SkillLabPlan(SkillLabContract):
    summary: str
    evaluation_calls: int
    generation_model: str = ""


class SkillLabSkillList(BaseModel):
    skills: list[dict[str, Any]]
    generated_at: str


class SkillLabSkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    content: str
    confirmed: bool = False


class SkillLabSkillUpdateRequest(BaseModel):
    content: str
    expected_revision: str | None = None
    confirmed: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _storage_root() -> Path:
    from bashgym.config import get_bashgym_dir

    root = get_bashgym_dir() / "skill_lab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _workspace_key(workspace_id: str) -> str:
    return hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]


def _identifier_key(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:32]


def _record_path(kind: str, identifier: str, workspace_id: str | None = None) -> Path:
    root = _storage_root() / kind
    if workspace_id is not None:
        root = root / _workspace_key(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{identifier}.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _skill_inventory() -> list[agent_routes.ToolkitSkill]:
    _roots, skills, _warnings = agent_routes._scan_skill_roots()
    return skills


def _skill_or_404(skill_id: str) -> agent_routes.ToolkitSkill:
    for skill in _skill_inventory():
        if skill.skill_id == skill_id:
            return skill
    raise HTTPException(status_code=404, detail="Skill not found")


def _skill_content(skill: agent_routes.ToolkitSkill) -> str:
    if not skill.path:
        tools = ", ".join(skill.allowed_tools) or "none declared"
        return f"# {skill.name}\n\n{skill.description}\n\n" f"Allowed tools: {tools}\n"
    try:
        return Path(skill.path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Skill content is unavailable") from exc


def _skill_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().casefold()).strip("-")
    if not slug or len(slug) > 80:
        raise HTTPException(status_code=422, detail="Skill name must produce a short file-safe id")
    return slug


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _complete_skill_document(name: str, description: str, content: str) -> str:
    normalized = content.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="SKILL.md content is required")
    if normalized.startswith("---"):
        return normalized + "\n"
    safe_name = name.replace("\n", " ").strip()
    safe_description = description.replace("\n", " ").strip()
    return (
        "---\n"
        f"name: {json.dumps(safe_name)}\n"
        f"description: {json.dumps(safe_description)}\n"
        "---\n\n"
        f"{normalized}\n"
    )


def _editable_skill_path(skill: agent_routes.ToolkitSkill) -> Path:
    if not skill.path:
        raise HTTPException(
            status_code=409, detail="Registry-backed skills cannot be edited as files"
        )
    path = Path(skill.path).resolve()
    allowed_roots = [
        root.resolve() for _label, root in agent_routes._toolkit_skill_root_candidates()
    ]
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Skill path is outside configured skill roots")
    if path.name != "SKILL.md":
        raise HTTPException(status_code=409, detail="Only SKILL.md files can be updated")
    return path


def _refreshed_skill(path: Path) -> dict[str, Any]:
    agent_routes._TOOLKIT_CACHE.clear()
    canonical = str(path.resolve()).casefold()
    for skill in _skill_inventory():
        if skill.path and str(Path(skill.path).resolve()).casefold() == canonical:
            return skill.model_dump()
    raise HTTPException(status_code=500, detail="Saved skill could not be reloaded")


def _normalize_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            raise HTTPException(status_code=422, detail=f"Case {index + 1} must be an object")
        prompt = raw_case.get("prompt", raw_case.get("input", raw_case.get("message")))
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=422, detail=f"Case {index + 1} needs a prompt")

        case_id = str(raw_case.get("case_id") or raw_case.get("id") or f"case-{index + 1}")
        expected = raw_case.get("expected_patterns", raw_case.get("expected", []))
        forbidden = raw_case.get("forbidden_patterns", raw_case.get("forbidden", []))
        if isinstance(expected, str):
            expected = [expected]
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        if not isinstance(expected, list) or not isinstance(forbidden, list):
            raise HTTPException(status_code=422, detail=f"Case {case_id} patterns must be lists")
        if not expected and not forbidden:
            raise HTTPException(
                status_code=422,
                detail=f"Case {case_id} needs an expected or forbidden pattern",
            )
        normalized.append(
            {
                "case_id": case_id,
                "name": str(raw_case.get("name") or case_id),
                "prompt": prompt.strip(),
                "expected_patterns": [str(pattern) for pattern in expected],
                "forbidden_patterns": [str(pattern) for pattern in forbidden],
                "should_invoke": bool(raw_case.get("should_invoke", False)),
            }
        )
    if not normalized:
        raise HTTPException(status_code=422, detail="At least one Skill Lab case is required")
    return normalized


def _normalize_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    normalized_input = dict(thresholds)
    for old_key, new_key in _THRESHOLD_ALIASES.items():
        if old_key in normalized_input and new_key not in normalized_input:
            normalized_input[new_key] = normalized_input[old_key]
    merged = {**_DEFAULT_THRESHOLDS, **normalized_input}
    for key in _DEFAULT_THRESHOLDS:
        try:
            merged[key] = float(merged[key])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Threshold {key} must be numeric")
    return merged


def _json_object_from_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise HTTPException(status_code=502, detail="Agent could not build an evaluation plan")
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail="Agent returned an invalid evaluation plan",
            ) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=502, detail="Agent returned an invalid evaluation plan")
    return value


def _validate_generated_plan(value: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list):
        raise HTTPException(status_code=502, detail="Agent plan did not include evaluation cases")
    cases = _normalize_cases(raw_cases[:expected_count])
    if len(cases) != expected_count:
        raise HTTPException(status_code=502, detail="Agent plan returned too few evaluation cases")
    if not any(case["should_invoke"] for case in cases):
        raise HTTPException(status_code=502, detail="Agent plan needs target examples")
    if not any(not case["should_invoke"] for case in cases):
        raise HTTPException(status_code=502, detail="Agent plan needs non-target examples")
    return cases


def _contract_payload(
    skill_id: str,
    workspace_id: str,
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
    endpoint_id: str | None = None,
) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "workspace_id": workspace_id,
        "endpoint_id": endpoint_id,
        "cases": _normalize_cases(cases),
        "thresholds": _normalize_thresholds(thresholds),
        "updated_at": _now(),
    }


def _load_contract(skill_id: str, workspace_id: str) -> dict[str, Any] | None:
    return _read_json(_record_path("contracts", _identifier_key(skill_id), workspace_id))


def _save_suite(
    suite_id: str,
    workspace_id: str,
    skill_id: str,
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> None:
    _write_json(
        _record_path("suites", suite_id, workspace_id),
        {
            "suite_id": suite_id,
            "workspace_id": workspace_id,
            "skill_id": skill_id,
            "cases": cases,
            "thresholds": thresholds,
            "created_at": _now(),
        },
    )


def _run_dump(run: SkillLabRun) -> dict[str, Any]:
    return run.model_dump()


def _load_run(run_id: str) -> SkillLabRun | None:
    runs_root = _storage_root() / "runs"
    paths = list(runs_root.rglob(f"{run_id}.json")) if runs_root.exists() else []
    data = _read_json(paths[0]) if paths else None
    if data is None:
        return None
    try:
        return SkillLabRun.model_validate(data)
    except ValueError:
        return None


def _save_run(run: SkillLabRun) -> None:
    _write_json(_record_path("runs", run.run_id, run.workspace_id), _run_dump(run))


def _pattern_match(pattern: str, response: str) -> bool:
    try:
        return re.search(pattern, response, flags=re.IGNORECASE | re.MULTILINE) is not None
    except re.error:
        return False


def _marker(response: str) -> tuple[bool | None, bool]:
    if not isinstance(response, str):
        return None, False
    matches = _MARKER_RE.findall(response)
    if len(matches) != 1:
        return None, False
    return matches[0].lower() == "true", True


def _verify_attempt(
    case: dict[str, Any],
    arm: str,
    response: str,
    error: str | None = None,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    expected_matches = {
        pattern: _pattern_match(pattern, response) for pattern in case["expected_patterns"]
    }
    forbidden_matches = {
        pattern: _pattern_match(pattern, response) for pattern in case["forbidden_patterns"]
    }
    should_invoke, marker_valid = _marker(response)
    expected_marker = {
        "baseline": False,
        "available": bool(case["should_invoke"]),
        "forced": True,
    }[arm]
    passed = (
        error is None
        and marker_valid
        and should_invoke is not None
        and should_invoke == expected_marker
        and all(expected_matches.values())
        and not any(forbidden_matches.values())
    )
    return {
        "case_id": case["case_id"],
        "case_name": case["name"],
        "arm": arm,
        "response": response,
        "output": response,
        "expected_matches": expected_matches,
        "forbidden_matches": forbidden_matches,
        "should_invoke": should_invoke,
        "invoked": bool(should_invoke),
        "marker_valid": marker_valid,
        "passed": passed,
        "error": error,
        "duration_ms": round(duration_ms, 2),
        "criterion_results": {
            "routing_marker": marker_valid and should_invoke == expected_marker,
            "expected_patterns": all(expected_matches.values()),
            "forbidden_patterns": not any(forbidden_matches.values()),
        },
    }


def _wilson_interval(passed: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    proportion = passed / total
    denominator = 1 + (z * z / total)
    centre = proportion + (z * z / (2 * total))
    margin = z * math.sqrt((proportion * (1 - proportion) / total) + (z * z / (4 * total * total)))
    return {
        "low": max(0.0, (centre - margin) / denominator),
        "high": min(1.0, (centre + margin) / denominator),
    }


def _calculate_kpis(
    cases: list[dict[str, Any]], attempts: list[dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    by_arm = {
        arm: [attempt for attempt in attempts if attempt["arm"] == arm]
        for arm in ("baseline", "available", "forced")
    }
    positive_case_ids = {case["case_id"] for case in cases if case["should_invoke"]}
    score_case_ids = positive_case_ids or {case["case_id"] for case in cases}
    scored_by_arm = {
        arm: [attempt for attempt in arm_attempts if attempt["case_id"] in score_case_ids]
        for arm, arm_attempts in by_arm.items()
    }
    rates = {
        arm: (
            sum(1 for attempt in arm_attempts if attempt["passed"]) / len(arm_attempts)
            if arm_attempts
            else 0.0
        )
        for arm, arm_attempts in scored_by_arm.items()
    }
    passed_scored = {
        arm: sum(1 for attempt in attempts_for_arm if attempt["passed"])
        for arm, attempts_for_arm in scored_by_arm.items()
    }
    confidence = {
        arm: _wilson_interval(passed_scored[arm], len(scored_by_arm[arm])) for arm in scored_by_arm
    }
    positives = sum(1 for case in cases if case["should_invoke"])
    negatives = len(cases) - positives
    available_by_case = {attempt["case_id"]: attempt for attempt in by_arm["available"]}
    true_positive = false_positive = false_negative = true_negative = 0
    invalid_markers = 0
    for case in cases:
        attempt = available_by_case.get(case["case_id"])
        actual = attempt.get("should_invoke") if attempt else None
        if actual is None or not attempt.get("marker_valid", False):
            invalid_markers += 1
            actual = False
        if case["should_invoke"] and actual:
            true_positive += 1
        elif case["should_invoke"] and not actual:
            false_negative += 1
        elif not case["should_invoke"] and actual:
            false_positive += 1
        else:
            true_negative += 1

    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_activation = false_positive / negatives if negatives else 0.0
    uplift = rates["available"] - rates["baseline"]
    sample_counts = {
        "total": len(cases),
        "positive": positives,
        "negative": negatives,
        "attempts": len(attempts),
        "by_arm": {arm: len(arm_attempts) for arm, arm_attempts in by_arm.items()},
        "passed_by_arm": {
            arm: sum(1 for attempt in arm_attempts if attempt["passed"])
            for arm, arm_attempts in by_arm.items()
        },
        "invalid_markers": invalid_markers,
    }
    checks = {
        "uplift": uplift >= float(thresholds["min_uplift"]),
        "forced_pass_rate": rates["forced"] >= float(thresholds["min_forced_pass_rate"]),
        "routing_precision": precision >= float(thresholds["min_routing_precision"]),
        "routing_recall": recall >= float(thresholds["min_routing_recall"]),
        "false_activation": false_activation <= float(thresholds["max_false_activation_rate"]),
        "markers": invalid_markers == 0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    durations = [float(attempt.get("duration_ms") or 0.0) for attempt in attempts]
    if all(checks.values()):
        verdict = "effective"
    elif checks["markers"] and checks["forced_pass_rate"] and uplift >= 0:
        verdict = "watch"
    else:
        verdict = "fail"
    return {
        "baseline_pass_rate": rates["baseline"],
        "available_pass_rate": rates["available"],
        "forced_pass_rate": rates["forced"],
        "success_uplift": uplift,
        "baseline_success_rate": rates["baseline"],
        "available_success_rate": rates["available"],
        "forced_success_rate": rates["forced"],
        "uplift": uplift,
        "routing_precision": precision,
        "routing_recall": recall,
        "routing_f1": f1,
        "false_activation": false_activation,
        "false_activation_rate": false_activation,
        "false_activation_count": false_positive,
        "sample_counts": sample_counts,
        "evaluated_cases": len(cases),
        "target_cases": positives,
        "negative_cases": negatives,
        "confidence": {
            **confidence,
            "success_uplift": {
                "low": confidence["available"]["low"] - confidence["baseline"]["high"],
                "high": confidence["available"]["high"] - confidence["baseline"]["low"],
            },
        },
        "average_duration_ms": sum(durations) / len(durations) if durations else 0.0,
        "total_duration_ms": sum(durations),
        "token_usage_available": False,
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "checks": checks,
        "reasons": failed_checks,
        "verdict": verdict,
    }


async def _execute_run(
    run_id: str,
    skill: agent_routes.ToolkitSkill,
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> None:
    run = _load_run(run_id)
    if run is None:
        return
    run.status = "running"
    run.started_at = _now()
    _save_run(run)
    skill_content = _skill_content(skill)
    errors: list[str] = []
    arms = ("baseline", "available", "forced")
    for case in cases:
        for arm in arms:
            context = (
                "Skill Lab evaluation context. Do not invoke the skill.\n"
                if arm == "baseline"
                else f"Skill Lab evaluation context. The skill is available:\n{skill_content}\n"
            )
            if arm == "forced":
                context += "The skill is forced for this arm; invoke it for this response.\n"
            message = (
                f"[Skill Lab arm: {arm}]\n{case['prompt']}\n\n"
                "Finish with exactly one line: SKILL_LAB_SHOULD_INVOKE: true or "
                "SKILL_LAB_SHOULD_INVOKE: false"
            )
            response_text = ""
            error: str | None = None
            started = time.monotonic()
            try:
                result = agent_routes.chat_with_agent_endpoint(
                    run.endpoint_id,
                    agent_routes.AgentEndpointChatRequest(
                        message=message,
                        context=context,
                        enable_skill_lab_tools=False,
                    ),
                )
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, BaseModel):
                    result_data = result.model_dump()
                elif isinstance(result, dict):
                    result_data = result
                else:
                    result_data = {"response": str(result)}
                response_text = str(result_data.get("response") or "")
                if not run.model:
                    run.model = str(result_data.get("model") or "")
            except Exception as exc:  # Network/provider failures fail closed.
                error = str(exc).strip() or exc.__class__.__name__
                errors.append(f"{case['case_id']}/{arm}: {error}")
            run.attempts.append(
                _verify_attempt(
                    case,
                    arm,
                    response_text,
                    error,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            )
            run.progress = {"completed": len(run.attempts), "total": len(cases) * len(arms)}
            _save_run(run)

    run.kpis = _calculate_kpis(cases, run.attempts, thresholds)
    run.completed_at = _now()
    run.status = "failed" if errors else "completed"
    run.error = "; ".join(errors) if errors else None
    if errors:
        run.kpis["verdict"] = "fail"
        run.kpis["reasons"] = ["agent_or_provider_error", *run.kpis.get("reasons", [])]
    _save_run(run)


@router.get("/skills", response_model=SkillLabSkillList)
async def list_skill_lab_skills():
    skills = [
        skill.model_dump() for skill in _skill_inventory() if skill.catalog_status == "active"
    ]
    return SkillLabSkillList(skills=skills, generated_at=_now())


@router.post("/skills", status_code=201)
async def create_skill_lab_skill(request: SkillLabSkillCreateRequest):
    if not request.confirmed:
        raise HTTPException(status_code=409, detail="Explicit confirmation is required")
    slug = _skill_slug(request.name)
    root = agent_routes._repo_root() / "assistant" / "workspace" / "skills"
    skill_path = root / slug / "SKILL.md"
    if skill_path.exists():
        raise HTTPException(
            status_code=409, detail="A workspace skill with this name already exists"
        )
    document = _complete_skill_document(request.name, request.description, request.content)
    _atomic_write_text(skill_path, document)
    return _refreshed_skill(skill_path)


@router.get("/skills/{skill_id}")
async def get_skill_lab_skill(skill_id: str):
    skill = _skill_or_404(skill_id)
    data = skill.model_dump()
    data["content"] = _skill_content(skill)
    data["skill_name"] = skill.name
    data["skill_revision"] = skill.revision
    return data


@router.post("/plans", response_model=SkillLabPlan)
async def draft_skill_lab_plan(request: SkillLabPlanRequest):
    skill = _skill_or_404(request.skill_id)
    if not request.endpoint_id.strip():
        raise HTTPException(status_code=422, detail="Connect an agent before evaluating")
    case_count = 4 if request.depth == "quick" else 8
    positive_count = case_count // 2
    goal = request.goal.strip() or skill.description or "the skill's declared purpose"
    skill_content = _skill_content(skill)[:20_000]
    prompt = (
        "Build a held-out evaluation plan for the capability described below. "
        "Return JSON only, with one top-level key named cases. "
        f"Create exactly {case_count} cases: {positive_count} requests that should invoke "
        f"the capability and {case_count - positive_count} realistic nearby requests that "
        "should not invoke it. Each case needs name, prompt, should_invoke, "
        "expected_patterns, and forbidden_patterns. Patterns are short, robust, "
        "case-insensitive regular expressions checked against the answer. Target cases "
        "must test useful output, not merely mention the capability name. Non-target cases "
        'may use [".+"] as expected_patterns. Avoid brittle full-sentence matching.\n\n'
        f"Evaluation focus: {goal}\n\n"
        f"Capability name: {skill.name}\n"
        f"Capability description: {skill.description}\n\n"
        f"Instructions:\n{skill_content}"
    )
    result = agent_routes.chat_with_agent_endpoint(
        request.endpoint_id,
        agent_routes.AgentEndpointChatRequest(
            message=prompt,
            workspace_id=request.workspace_id,
            enable_skill_lab_tools=False,
        ),
    )
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, BaseModel):
        result_data = result.model_dump()
    elif isinstance(result, dict):
        result_data = result
    else:
        result_data = {"response": str(result)}
    response_text = str(result_data.get("response") or "")
    if not response_text:
        raise HTTPException(status_code=502, detail="Agent returned no evaluation plan")
    cases = _validate_generated_plan(_json_object_from_text(response_text), case_count)
    return SkillLabPlan(
        skill_id=skill.skill_id,
        workspace_id=request.workspace_id,
        endpoint_id=request.endpoint_id,
        cases=cases,
        thresholds=dict(_DEFAULT_THRESHOLDS),
        updated_at=_now(),
        summary=goal,
        evaluation_calls=len(cases) * 3,
        generation_model=str(result_data.get("model") or ""),
    )


@router.put("/skills/{skill_id}")
async def update_skill_lab_skill(skill_id: str, request: SkillLabSkillUpdateRequest):
    if not request.confirmed:
        raise HTTPException(status_code=409, detail="Explicit confirmation is required")
    skill = _skill_or_404(skill_id)
    if skill.name.casefold() in _PROTECTED_SKILL_NAMES:
        raise HTTPException(
            status_code=403,
            detail=(
                "This is a source-managed critical BashGym skill. Create a reviewed "
                "repository change and redeploy the operator bundle; self-improvement "
                "and Skill Lab edits are not allowed."
            ),
        )
    if request.expected_revision and request.expected_revision != skill.revision:
        raise HTTPException(
            status_code=409,
            detail="Skill revision changed; inspect it again before updating",
        )
    skill_path = _editable_skill_path(skill)
    document = request.content.strip()
    if not document:
        raise HTTPException(status_code=422, detail="SKILL.md content is required")
    _atomic_write_text(skill_path, document + "\n")
    return _refreshed_skill(skill_path)


@router.get("/contracts/{skill_id}", response_model=SkillLabContract)
async def get_skill_lab_contract(skill_id: str, workspace_id: str = Query(..., min_length=1)):
    _skill_or_404(skill_id)
    data = _load_contract(skill_id, workspace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Skill Lab contract not found")
    return SkillLabContract.model_validate(data)


@router.put("/contracts/{skill_id}", response_model=SkillLabContract)
async def put_skill_lab_contract(
    skill_id: str,
    request: SkillLabContractRequest,
    workspace_id: str = Query(..., min_length=1),
):
    _skill_or_404(skill_id)
    data = _contract_payload(
        skill_id,
        workspace_id,
        request.cases,
        request.thresholds,
        request.endpoint_id,
    )
    _write_json(_record_path("contracts", _identifier_key(skill_id), workspace_id), data)
    return SkillLabContract.model_validate(data)


@router.post("/runs", response_model=SkillLabRun, status_code=202)
async def launch_skill_lab_run(request: SkillLabRunRequest, background_tasks: BackgroundTasks):
    skill = _skill_or_404(request.skill_id)
    contract = _load_contract(request.skill_id, request.workspace_id)
    cases = request.cases or (contract or {}).get("cases", [])
    thresholds = request.thresholds or (contract or {}).get("thresholds", {})
    normalized_cases = _normalize_cases(cases)
    normalized_thresholds = _normalize_thresholds(thresholds)
    if not request.endpoint_id.strip():
        raise HTTPException(status_code=422, detail="Agent endpoint is required")
    suite_id = request.suite_id or f"suite-{uuid.uuid4().hex}"
    _save_suite(
        suite_id,
        request.workspace_id,
        request.skill_id,
        normalized_cases,
        normalized_thresholds,
    )
    run = SkillLabRun(
        run_id=f"run-{uuid.uuid4().hex}",
        workspace_id=request.workspace_id,
        skill_id=skill.skill_id,
        skill_name=skill.name,
        skill_revision=skill.revision,
        endpoint_id=request.endpoint_id,
        status="queued",
        created_at=_now(),
        progress={"completed": 0, "total": len(normalized_cases) * 3},
        suite_id=suite_id,
    )
    _save_run(run)
    background_tasks.add_task(
        _execute_run,
        run.run_id,
        skill,
        normalized_cases,
        normalized_thresholds,
    )
    return run


@router.get("/runs", response_model=list[SkillLabRun])
async def list_skill_lab_runs(
    workspace_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
):
    runs_dir = _storage_root() / "runs"
    runs: list[SkillLabRun] = []
    if runs_dir.exists():
        for path in runs_dir.rglob("*.json"):
            run = _load_run(path.stem)
            if run is not None and run.workspace_id == workspace_id:
                runs.append(run)
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return runs[:limit]


@router.get("/runs/{run_id}", response_model=SkillLabRun)
async def get_skill_lab_run(run_id: str):
    run = _load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Skill Lab run not found")
    return run
