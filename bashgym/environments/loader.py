"""Load and normalize executable environment specs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentAxis,
    EnvironmentSpec,
    FixtureSpec,
    RolloutSpec,
    VerifierSpec,
    stable_environment_id,
)

INSTRUCTION_FIELDS = (
    "instruction",
    "task_instruction",
    "task",
    "prompt",
    "question",
    "problem_statement",
    "task_description",
    "description",
)

AXIS_FIELDS = (
    "domain",
    "skill",
    "skill_type",
    "persona",
    "fixture",
    "fixture_kind",
    "language",
    "task_complexity",
    "command_complexity",
    "verifier_kind",
)


def _first_str(record: dict[str, Any], fields: Iterable[str], default: str = "") -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _extract_files(record: dict[str, Any]) -> dict[str, str]:
    files = record.get("files") or record.get("source_files") or record.get("workspace_files") or {}
    if isinstance(files, dict):
        return {str(k): str(v) for k, v in files.items()}
    out: dict[str, str] = {}
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                path = item.get("path") or item.get("name")
                content = item.get("content") or item.get("text") or ""
                if path:
                    out[str(path)] = str(content)
    return out


def _extract_verifier(record: dict[str, Any]) -> VerifierSpec:
    raw = record.get("verifier") or record.get("checker") or {}
    if isinstance(raw, str):
        raw = {"command": raw}
    if not isinstance(raw, dict):
        raw = {}
    command = (
        raw.get("command")
        or raw.get("cmd")
        or record.get("verify_command")
        or record.get("test_cmd")
        or record.get("checker_command")
        or "./verify.sh"
    )
    kind = (
        raw.get("kind")
        or record.get("verifier_kind")
        or record.get("reward_kind")
        or "exact_success"
    )
    return VerifierSpec.from_dict(
        {
            **raw,
            "command": command,
            "kind": kind,
            "path": raw.get("path", record.get("verify_path", "verify.sh")),
        }
    )


def _extract_build(record: dict[str, Any]) -> BuildSpec:
    raw = record.get("build") or {}
    if not isinstance(raw, dict):
        raw = {}
    dockerfile = (
        raw.get("dockerfile")
        or record.get("dockerfile")
        or record.get("Dockerfile")
        or "Dockerfile"
    )
    base_image = raw.get("base_image") or record.get("base_image")
    compose_file = raw.get("compose_file") or record.get("compose_file")
    setup_commands = raw.get("setup_commands") or record.get("setup_commands") or []
    return BuildSpec.from_dict(
        {
            **raw,
            "dockerfile": dockerfile,
            "base_image": base_image,
            "compose_file": compose_file,
            "setup_commands": setup_commands,
        }
    )


def _extract_rollout(record: dict[str, Any]) -> RolloutSpec:
    raw = record.get("rollout") or record.get("harness") or {}
    if isinstance(raw, str):
        raw = {"harness": raw}
    if not isinstance(raw, dict):
        raw = {}
    return RolloutSpec.from_dict(raw)


def _extract_axes(record: dict[str, Any]) -> list[EnvironmentAxis]:
    axes: list[EnvironmentAxis] = []
    raw_axes = record.get("axes")
    if isinstance(raw_axes, list):
        for raw in raw_axes:
            if isinstance(raw, dict):
                axes.append(EnvironmentAxis.from_dict(raw))
    for field in AXIS_FIELDS:
        value = record.get(field)
        if value is not None:
            for item in _as_list(value):
                axes.append(EnvironmentAxis(name=field, value=item))
    seen: set[tuple[str, str]] = set()
    unique: list[EnvironmentAxis] = []
    for axis in axes:
        key = (axis.name, axis.value)
        if key not in seen:
            seen.add(key)
            unique.append(axis)
    return unique


def _extract_fixtures(record: dict[str, Any]) -> list[FixtureSpec]:
    raw = record.get("fixtures") or record.get("artifacts") or []
    if isinstance(raw, dict):
        raw = [raw]
    fixtures: list[FixtureSpec] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                fixtures.append(FixtureSpec.from_dict(item))
            elif isinstance(item, str):
                fixtures.append(FixtureSpec(path=item))
    return fixtures


def environment_from_record(
    record: dict[str, Any],
    *,
    source: str = "external",
    source_uri: str | None = None,
    preserve_raw: bool = True,
) -> EnvironmentSpec:
    """Normalize a TMax/Harbor/DataDesigner-like row into ``EnvironmentSpec``."""
    metadata = dict(record.get("metadata") or {})
    if preserve_raw:
        metadata.setdefault("raw_record", record)
    instruction = _first_str(record, INSTRUCTION_FIELDS)
    domain = _first_str(record, ("domain", "category", "task_domain"), "unknown")
    skills = _as_list(record.get("skills") or record.get("skill") or record.get("skill_type"))
    env_id = str(
        record.get("id")
        or record.get("task_id")
        or record.get("name")
        or stable_environment_id(source, instruction, record)
    )
    spec = EnvironmentSpec(
        id=env_id,
        instruction=instruction,
        source=source,
        domain=domain,
        skills=skills,
        axes=_extract_axes(record),
        fixtures=_extract_fixtures(record),
        verifier=_extract_verifier(record),
        build=_extract_build(record),
        rollout=_extract_rollout(record),
        files=_extract_files(record),
        source_uri=source_uri or record.get("source_uri"),
        license=record.get("license"),
        metadata=metadata,
    )
    return spec


def save_environment(spec: EnvironmentSpec, path: str | Path) -> Path:
    """Write one environment JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_environment(path: str | Path) -> EnvironmentSpec:
    """Load an environment JSON file or a directory containing ``env.json``."""
    p = Path(path)
    if p.is_dir():
        for name in ("env.json", "environment.json", "task.json"):
            candidate = p / name
            if candidate.exists():
                p = candidate
                break
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return EnvironmentSpec.from_dict(data)


def load_environments(path: str | Path) -> list[EnvironmentSpec]:
    """Load all environment specs from a file, JSONL, or directory."""
    p = Path(path)
    if p.is_dir():
        specs: list[EnvironmentSpec] = []
        for candidate in sorted(p.rglob("*.json")):
            if candidate.name in {"env.json", "environment.json", "task.json"}:
                specs.append(load_environment(candidate))
        return specs
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        specs = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    specs.append(EnvironmentSpec.from_dict(json.loads(line)))
        return specs
    return [load_environment(p)]
