"""Install the reviewed BashGym skill bundle into supported agent hosts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOSTS = ("codex", "claude", "hermes")
RECEIPT_NAME = ".bashgym-skill-bundle-receipt.json"
LOCK_SCHEMA = "bashgym.operator-bundle-lock.v2"
RECEIPT_SCHEMA = "bashgym.operator-skills-receipt.v1"
INSTALL_SCHEMA = "bashgym.operator-skills-install.v1"
CHECK_SCHEMA = "bashgym.operator-skills-check.v1"


@dataclass(frozen=True)
class SkillBundle:
    source_root: Path
    files: dict[str, str]
    bundle_id: str

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(sorted({Path(relative).parts[0] for relative in self.files}))


def _canonical_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(_canonical_bytes(path)).hexdigest()


def _packaged_skill_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assistant" / "workspace" / "skills"


def _load_bundle(source_root: Path | None = None) -> SkillBundle:
    skills_root = (source_root or _packaged_skill_root()).resolve()
    operator_root = skills_root / "bashgym-operator"
    lock_path = operator_root / "bundle.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("packaged BashGym skill lock is unavailable or invalid") from exc
    if lock.get("schema_version") != LOCK_SCHEMA or not isinstance(lock.get("files"), dict):
        raise ValueError("packaged BashGym skill lock schema is unsupported")

    files: dict[str, str] = {}
    for raw_relative, raw_expected in lock["files"].items():
        relative = str(raw_relative)
        expected = str(raw_expected)
        candidate = (operator_root / relative).resolve()
        try:
            installed_relative = candidate.relative_to(skills_root).as_posix()
        except ValueError as exc:
            raise ValueError("packaged BashGym skill lock escapes its bundle root") from exc
        if not candidate.is_file() or _digest(candidate) != expected:
            raise ValueError("packaged BashGym skill bundle failed integrity verification")
        files[installed_relative] = expected

    lock_relative = lock_path.relative_to(skills_root).as_posix()
    files[lock_relative] = _digest(lock_path)
    bundle_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    bundle_id = hashlib.sha256(bundle_payload).hexdigest()
    return SkillBundle(source_root=skills_root, files=files, bundle_id=bundle_id)


def _host_home(host: str) -> Path:
    home = Path.home()
    if host == "codex":
        return Path(os.environ.get("CODEX_HOME", home / ".codex")).expanduser()
    if host == "claude":
        configured = os.environ.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_HOME")
        return Path(configured or home / ".claude").expanduser()
    if host == "hermes":
        return Path(os.environ.get("HERMES_HOME", home / ".hermes")).expanduser()
    raise ValueError(f"unsupported agent host: {host}")


def resolve_host(host: str | None) -> tuple[str, Path]:
    if host is not None:
        selected = host.casefold()
        if selected not in HOSTS:
            raise ValueError("agent host must be codex, claude, or hermes")
        return selected, _host_home(selected) / "skills"

    explicit = {
        "codex": bool(os.environ.get("CODEX_HOME")),
        "claude": bool(os.environ.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_HOME")),
        "hermes": bool(os.environ.get("HERMES_HOME")),
    }
    candidates = [name for name, configured in explicit.items() if configured]
    if not candidates:
        candidates = [name for name in HOSTS if _host_home(name).is_dir()]
    if len(candidates) != 1:
        raise ValueError("agent host detection is ambiguous; pass --host codex, claude, or hermes")
    selected = candidates[0]
    return selected, _host_home(selected) / "skills"


def _receipt(bundle: SkillBundle, host: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "host": host,
        "bundle_id": bundle.bundle_id,
        "skill_names": list(bundle.skill_names),
        "files": bundle.files,
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _write_staged_bundle(bundle: SkillBundle, staging_root: Path, host: str) -> None:
    for relative in bundle.files:
        source = bundle.source_root / relative
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    receipt_path = staging_root / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(_receipt(bundle, host), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _swap_staged_bundle(bundle: SkillBundle, staging_root: Path, skills_root: Path) -> None:
    transaction = uuid.uuid4().hex
    replaced: list[tuple[Path, Path | None]] = []
    entries = [*bundle.skill_names, RECEIPT_NAME]
    try:
        for name in entries:
            destination = skills_root / name
            backup = skills_root / f".bashgym-skill-backup-{transaction}-{name}"
            previous: Path | None = None
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup)
                previous = backup
            try:
                os.replace(staging_root / name, destination)
            except Exception:
                if previous is not None:
                    os.replace(previous, destination)
                raise
            replaced.append((destination, previous))
    except Exception:
        for destination, previous in reversed(replaced):
            _remove_path(destination)
            if previous is not None:
                os.replace(previous, destination)
        raise
    else:
        for _destination, previous in replaced:
            if previous is not None:
                _remove_path(previous)


def check_skills(*, host: str | None = None) -> dict[str, Any]:
    selected, skills_root = resolve_host(host)
    bundle = _load_bundle()
    mismatches: list[str] = []
    receipt_path = skills_root / RECEIPT_NAME
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = None
        mismatches.append(RECEIPT_NAME)
    if receipt != _receipt(bundle, selected) and RECEIPT_NAME not in mismatches:
        mismatches.append(RECEIPT_NAME)

    expected_paths = set(bundle.files)
    observed_paths: set[str] = set()
    for skill_name in bundle.skill_names:
        skill_root = skills_root / skill_name
        if skill_root.is_dir() and not skill_root.is_symlink():
            observed_paths.update(
                path.relative_to(skills_root).as_posix()
                for path in skill_root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
        elif skill_root.exists() or skill_root.is_symlink():
            mismatches.append(skill_name)
    mismatches.extend(sorted(expected_paths.symmetric_difference(observed_paths)))
    for relative, expected in bundle.files.items():
        candidate = skills_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            if relative not in mismatches:
                mismatches.append(relative)
            continue
        try:
            actual = _digest(candidate)
        except (OSError, UnicodeError):
            actual = ""
        if actual != expected and relative not in mismatches:
            mismatches.append(relative)
    unique_mismatches = sorted(set(mismatches))
    return {
        "schema_version": CHECK_SCHEMA,
        "host": selected,
        "skills_root": str(skills_root.resolve()),
        "bundle_id": bundle.bundle_id,
        "file_count": len(bundle.files),
        "verified": not unique_mismatches,
        "mismatches": unique_mismatches,
    }


def install_skills(*, host: str | None = None) -> dict[str, Any]:
    selected, skills_root = resolve_host(host)
    bundle = _load_bundle()
    skills_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".bashgym-skills-stage-", dir=str(skills_root)))
    try:
        _write_staged_bundle(bundle, staging_root, selected)
        _swap_staged_bundle(bundle, staging_root, skills_root)
    finally:
        _remove_path(staging_root)
    result = check_skills(host=selected)
    return {**result, "schema_version": INSTALL_SCHEMA}
