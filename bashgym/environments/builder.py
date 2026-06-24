"""Materialize executable environment specs to local task directories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bashgym.environments.contracts import EnvironmentSpec

PROTECTED_MANIFEST_NAME = ".bashgym_manifest.json"
PROTECTED_MANIFEST_VERSION = "bashgym.environment_manifest.v1"


@dataclass
class EnvironmentBuild:
    env_id: str
    path: Path
    files_written: list[Path]
    manifest_path: Path | None = None
    protected_files: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "path": str(self.path),
            "files_written": [str(p) for p in self.files_written],
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "protected_files": self.protected_files or [],
        }


def _safe_child(root: Path, relative_path: str) -> Path:
    """Resolve a POSIX-ish task path and ensure it stays inside ``root``."""
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"environment file path escapes task root: {relative_path!r}")
    target = (root / Path(*pure.parts)).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"environment file path escapes task root: {relative_path!r}") from exc
    return target


def _safe_environment_root(output_dir: str | Path, env_id: str) -> Path:
    """Resolve the environment directory and ensure ``env_id`` cannot escape it."""
    base = Path(output_dir).resolve()
    root = (base / env_id).resolve()
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"environment id escapes output directory: {env_id!r}") from exc
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str) -> str:
    pure = PurePosixPath(path.replace("\\", "/"))
    return str(PurePosixPath(*pure.parts))


def _metadata_protected_paths(spec: EnvironmentSpec) -> set[str]:
    raw_paths = []
    if isinstance(spec.metadata.get("protected_paths"), list):
        raw_paths.extend(spec.metadata["protected_paths"])
    if isinstance(spec.rollout.metadata.get("protected_paths"), list):
        raw_paths.extend(spec.rollout.metadata["protected_paths"])
    return {_normalize_relative_path(str(path)) for path in raw_paths if str(path).strip()}


def protected_environment_paths(spec: EnvironmentSpec) -> list[str]:
    """Return relative files that should not be edited by rollout commands."""

    if spec.metadata.get("allow_protected_file_edits") or spec.rollout.metadata.get(
        "allow_protected_file_edits"
    ):
        return []

    protected = {"env.json"}
    if spec.verifier.path:
        protected.add(_normalize_relative_path(spec.verifier.path))

    for relative_path in spec.files:
        normalized = _normalize_relative_path(relative_path)
        parts = PurePosixPath(normalized).parts
        if parts and parts[0] in {"test", "tests"}:
            protected.add(normalized)

    for fixture in spec.fixtures:
        normalized = _normalize_relative_path(fixture.path)
        if fixture.metadata.get("private") or fixture.metadata.get("protected"):
            protected.add(normalized)
        if fixture.kind in {"private", "test", "tests", "verifier"}:
            protected.add(normalized)

    protected.update(_metadata_protected_paths(spec))
    return sorted(protected)


def write_environment_manifest(spec: EnvironmentSpec, root: Path) -> Path:
    """Persist protected-file checksums for rollout tamper detection."""

    protected_files: dict[str, str] = {}
    for relative_path in protected_environment_paths(spec):
        target = _safe_child(root, relative_path)
        if target.exists() and target.is_file():
            protected_files[relative_path] = _sha256_file(target)

    manifest = {
        "schema_version": PROTECTED_MANIFEST_VERSION,
        "environment_id": spec.id,
        "protected_files": protected_files,
    }
    manifest_path = root / PROTECTED_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def audit_environment_manifest(root: str | Path) -> dict:
    """Compare current protected files with the materialization manifest."""

    root_path = Path(root)
    manifest_path = root_path / PROTECTED_MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "ok": True,
            "manifest_found": False,
            "checked_paths": 0,
            "tampered_paths": [],
            "missing_paths": [],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected_files = manifest.get("protected_files") or {}
    tampered_paths: list[str] = []
    missing_paths: list[str] = []
    for relative_path, expected_hash in protected_files.items():
        target = _safe_child(root_path, str(relative_path))
        if not target.exists() or not target.is_file():
            missing_paths.append(str(relative_path))
            continue
        actual_hash = _sha256_file(target)
        if actual_hash != expected_hash:
            tampered_paths.append(str(relative_path))

    return {
        "ok": not tampered_paths and not missing_paths,
        "manifest_found": True,
        "schema_version": manifest.get("schema_version"),
        "environment_id": manifest.get("environment_id"),
        "checked_paths": len(protected_files),
        "tampered_paths": tampered_paths,
        "missing_paths": missing_paths,
    }


def materialize_environment(
    spec: EnvironmentSpec,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> EnvironmentBuild:
    """Write ``spec`` as an executable environment bundle.

    The bundle contains ``env.json`` plus any files embedded in ``spec.files``.
    This deliberately does not run Docker or shell commands; build/smoke happens
    in later harness layers.
    """
    root = _safe_environment_root(output_dir, spec.id)
    if root.exists() and not overwrite:
        raise FileExistsError(f"environment already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    files_written: list[Path] = []
    env_json = root / "env.json"
    env_json.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files_written.append(env_json)

    for relative_path, content in sorted(spec.files.items()):
        target = _safe_child(root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files_written.append(target)

    if spec.build.dockerfile and spec.build.dockerfile not in spec.files:
        dockerfile = _safe_child(root, spec.build.dockerfile)
        if not dockerfile.exists():
            base = spec.build.base_image or "python:3.11-slim"
            dockerfile.write_text(
                f"FROM {base}\nWORKDIR /workspace\nCOPY . /workspace\n", encoding="utf-8"
            )
            files_written.append(dockerfile)

    if spec.verifier.path and spec.verifier.path not in spec.files:
        verifier = _safe_child(root, spec.verifier.path)
        if not verifier.exists():
            verifier.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nexit 1\n", encoding="utf-8"
            )
            files_written.append(verifier)

    manifest_path = write_environment_manifest(spec, root)
    files_written.append(manifest_path)

    return EnvironmentBuild(
        env_id=spec.id,
        path=root,
        files_written=files_written,
        manifest_path=manifest_path,
        protected_files=protected_environment_paths(spec),
    )
