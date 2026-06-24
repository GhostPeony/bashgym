"""Serializable contracts for executable terminal-agent environments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values while preserving falsy meaningful values."""
    return {k: v for k, v in data.items() if v is not None}


def stable_environment_id(
    source: str, instruction: str, metadata: dict[str, Any] | None = None
) -> str:
    """Create a deterministic id from stable environment content."""
    payload = {
        "source": source or "",
        "instruction": instruction or "",
        "metadata_id": (metadata or {}).get("id")
        or (metadata or {}).get("task_id")
        or (metadata or {}).get("name")
        or "",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"env_{digest[:16]}"


@dataclass
class EnvironmentAxis:
    """One sampled or imported dimension of an environment recipe."""

    name: str
    value: str
    source: str = "imported"
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "name": self.name,
                "value": self.value,
                "source": self.source,
                "weight": self.weight,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentAxis:
        return cls(
            name=str(data.get("name", "")),
            value=str(data.get("value", "")),
            source=str(data.get("source", "imported")),
            weight=data.get("weight"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class FixtureSpec:
    """A concrete artifact shipped with a task: file, image, archive, service, etc."""

    path: str
    kind: str = "file"
    description: str = ""
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "path": self.path,
                "kind": self.kind,
                "description": self.description,
                "checksum": self.checksum,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixtureSpec:
        return cls(
            path=str(data.get("path", "")),
            kind=str(data.get("kind", "file")),
            description=str(data.get("description", "")),
            checksum=data.get("checksum"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class VerifierSpec:
    """Programmatic reward/checker for an environment."""

    kind: str = "exact_success"
    command: str = "./verify.sh"
    path: str | None = "verify.sh"
    reward_type: str = "binary"
    success_threshold: float = 1.0
    timeout_sec: int = 120
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_graded(self) -> bool:
        return self.reward_type not in {"binary", "pass_fail"} or self.success_threshold < 1.0

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "kind": self.kind,
                "command": self.command,
                "path": self.path,
                "reward_type": self.reward_type,
                "success_threshold": self.success_threshold,
                "timeout_sec": self.timeout_sec,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VerifierSpec:
        data = data or {}
        return cls(
            kind=str(data.get("kind", "exact_success")),
            command=str(data.get("command") or data.get("cmd") or "./verify.sh"),
            path=data.get("path", "verify.sh"),
            reward_type=str(data.get("reward_type", "binary")),
            success_threshold=float(data.get("success_threshold", 1.0)),
            timeout_sec=int(data.get("timeout_sec", data.get("timeout", 120))),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class BuildSpec:
    """How to construct the environment workspace/container."""

    context_dir: str = "."
    dockerfile: str | None = "Dockerfile"
    base_image: str | None = None
    compose_file: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    network_disabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "context_dir": self.context_dir,
                "dockerfile": self.dockerfile,
                "base_image": self.base_image,
                "compose_file": self.compose_file,
                "setup_commands": self.setup_commands,
                "network_disabled": self.network_disabled,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BuildSpec:
        data = data or {}
        setup = data.get("setup_commands") or data.get("setup") or []
        if isinstance(setup, str):
            setup = [setup]
        return cls(
            context_dir=str(data.get("context_dir", ".")),
            dockerfile=data.get("dockerfile", "Dockerfile"),
            base_image=data.get("base_image"),
            compose_file=data.get("compose_file"),
            setup_commands=[str(c) for c in setup],
            network_disabled=bool(data.get("network_disabled", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class RolloutSpec:
    """Harness/runtime limits for model attempts."""

    harness: str = "bashgym-persistent-shell"
    max_steps: int = 64
    max_tool_calls: int = 64
    timeout_sec: int = 7200
    bash_timeout_sec: int = 120
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 65536
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "harness": self.harness,
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "timeout_sec": self.timeout_sec,
                "bash_timeout_sec": self.bash_timeout_sec,
                "max_prompt_tokens": self.max_prompt_tokens,
                "max_response_tokens": self.max_response_tokens,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RolloutSpec:
        data = data or {}
        max_steps = data.get("max_steps", data.get("max_tool_env_steps", 64))
        return cls(
            harness=str(data.get("harness", "bashgym-persistent-shell")),
            max_steps=int(max_steps),
            max_tool_calls=int(data.get("max_tool_calls", max_steps)),
            timeout_sec=int(data.get("timeout_sec", data.get("timeout", 7200))),
            bash_timeout_sec=int(data.get("bash_timeout_sec", 120)),
            max_prompt_tokens=int(data.get("max_prompt_tokens", 2048)),
            max_response_tokens=int(data.get("max_response_tokens", 65536)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class EnvironmentSpec:
    """A first-class executable terminal task for SFT/RL/eval."""

    id: str
    instruction: str
    source: str = "bashgym"
    domain: str = "unknown"
    skills: list[str] = field(default_factory=list)
    axes: list[EnvironmentAxis] = field(default_factory=list)
    fixtures: list[FixtureSpec] = field(default_factory=list)
    verifier: VerifierSpec = field(default_factory=VerifierSpec)
    build: BuildSpec = field(default_factory=BuildSpec)
    rollout: RolloutSpec = field(default_factory=RolloutSpec)
    files: dict[str, str] = field(default_factory=dict)
    source_uri: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verifiable(self) -> bool:
        return bool(self.verifier.command or self.verifier.path)

    def axis_value(self, name: str) -> str | None:
        """Return a normalized axis value from explicit axes or common fields."""
        key = name.strip().lower()
        if key == "domain":
            return self.domain
        if key in {"skill", "skills", "skill_type"}:
            return ",".join(self.skills) if self.skills else None
        for axis in self.axes:
            if axis.name.strip().lower() == key:
                return axis.value
        meta_value = self.metadata.get(key)
        if meta_value is not None:
            return str(meta_value)
        return None

    def validation_errors(self) -> list[str]:
        """Return structural issues that prevent environment execution."""
        errors: list[str] = []
        if not self.id:
            errors.append("missing id")
        if not self.instruction:
            errors.append("missing instruction")
        if not self.is_verifiable:
            errors.append("missing verifier command/path")
        if self.rollout.max_tool_calls <= 0:
            errors.append("rollout.max_tool_calls must be positive")
        if self.verifier.timeout_sec <= 0:
            errors.append("verifier.timeout_sec must be positive")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "id": self.id,
                "instruction": self.instruction,
                "source": self.source,
                "domain": self.domain,
                "skills": self.skills,
                "axes": [a.to_dict() for a in self.axes],
                "fixtures": [f.to_dict() for f in self.fixtures],
                "verifier": self.verifier.to_dict(),
                "build": self.build.to_dict(),
                "rollout": self.rollout.to_dict(),
                "files": self.files,
                "source_uri": self.source_uri,
                "license": self.license,
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentSpec:
        metadata = dict(data.get("metadata") or {})
        instruction = str(
            data.get("instruction")
            or data.get("task_instruction")
            or data.get("prompt")
            or data.get("task")
            or ""
        )
        source = str(data.get("source", "bashgym"))
        env_id = str(data.get("id") or stable_environment_id(source, instruction, metadata))
        axes = [
            a if isinstance(a, EnvironmentAxis) else EnvironmentAxis.from_dict(a)
            for a in data.get("axes", [])
        ]
        fixtures = [
            f if isinstance(f, FixtureSpec) else FixtureSpec.from_dict(f)
            for f in data.get("fixtures", [])
        ]
        return cls(
            id=env_id,
            instruction=instruction,
            source=source,
            domain=str(data.get("domain", "unknown")),
            skills=[str(s) for s in data.get("skills", [])],
            axes=axes,
            fixtures=fixtures,
            verifier=(
                data["verifier"]
                if isinstance(data.get("verifier"), VerifierSpec)
                else VerifierSpec.from_dict(data.get("verifier"))
            ),
            build=(
                data["build"]
                if isinstance(data.get("build"), BuildSpec)
                else BuildSpec.from_dict(data.get("build"))
            ),
            rollout=(
                data["rollout"]
                if isinstance(data.get("rollout"), RolloutSpec)
                else RolloutSpec.from_dict(data.get("rollout"))
            ),
            files={str(k): str(v) for k, v in (data.get("files") or {}).items()},
            source_uri=data.get("source_uri"),
            license=data.get("license"),
            metadata=metadata,
        )
