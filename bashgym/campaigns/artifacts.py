"""Two-phase, HMAC-authenticated artifact sealing for campaign actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import ArtifactOutput, SealedActionResult

SEAL_FILENAME = "sealed_action_result.v1.json"


class ArtifactSealError(RuntimeError):
    code = "campaign_artifact_seal_invalid"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _safe_output_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path(SEAL_FILENAME):
        raise ArtifactSealError(f"{ArtifactSealError.code}: unsafe output path")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactSealError(f"{ArtifactSealError.code}: output escapes seal root") from exc
    return resolved


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class ArtifactSealer:
    """Seal action outputs on one filesystem and verify them after a restart."""

    def __init__(self, key: bytes, *, key_version: str):
        if len(key) < 32:
            raise ValueError("artifact seal key must contain at least 32 bytes")
        if not key_version:
            raise ValueError("artifact seal key version is required")
        self._key = key
        self.key_version = key_version

    def describe_outputs(
        self, temporary_directory: Path, schemas: dict[str, str]
    ) -> tuple[ArtifactOutput, ...]:
        """Hash and describe an explicit, sorted set of closed output files."""

        outputs = []
        for relative_path in sorted(schemas):
            path = _safe_output_path(temporary_directory, relative_path)
            if not path.is_file():
                raise ArtifactSealError(f"{ArtifactSealError.code}: missing output {relative_path}")
            digest, size = _hash_file(path)
            outputs.append(
                ArtifactOutput(
                    path=relative_path,
                    sha256=digest,
                    size_bytes=size,
                    schema_name=schemas[relative_path],
                )
            )
        return tuple(outputs)

    def _signature(self, manifest: SealedActionResult) -> str:
        payload = manifest.model_dump(mode="json")
        return hmac.new(self._key, _canonical_bytes(payload), hashlib.sha256).hexdigest()

    def seal(
        self,
        temporary_directory: Path,
        sealed_directory: Path,
        manifest: SealedActionResult,
    ) -> Path:
        """Validate, sign, flush, and atomically rename one action-owned directory."""

        temporary_directory = temporary_directory.resolve()
        sealed_directory = sealed_directory.resolve()
        if not temporary_directory.is_dir():
            raise ArtifactSealError(f"{ArtifactSealError.code}: temporary directory missing")
        if sealed_directory.exists():
            raise ArtifactSealError(f"{ArtifactSealError.code}: sealed destination exists")
        actual_files = {
            path.relative_to(temporary_directory).as_posix()
            for path in temporary_directory.rglob("*")
            if path.is_file()
        }
        expected_files = {output.path for output in manifest.outputs}
        if actual_files != expected_files:
            raise ArtifactSealError(
                f"{ArtifactSealError.code}: manifest does not cover the exact output set"
            )
        for output in manifest.outputs:
            path = _safe_output_path(temporary_directory, output.path)
            digest, size = _hash_file(path)
            if digest != output.sha256 or size != output.size_bytes:
                raise ArtifactSealError(
                    f"{ArtifactSealError.code}: output changed before seal: {output.path}"
                )
            try:
                with path.open("r+b") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                # Some Windows filesystems do not expose a flushable descriptor
                # for every file type. Hash verification still protects content,
                # and the seal file plus parent rename are flushed separately.
                pass

        envelope = {
            "schema_version": "sealed_action_result_envelope.v1",
            "key_version": self.key_version,
            "manifest": manifest.model_dump(mode="json"),
            "signature": self._signature(manifest),
        }
        seal_path = temporary_directory / SEAL_FILENAME
        with seal_path.open("wb") as handle:
            handle.write(_canonical_bytes(envelope))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary_directory)
        sealed_directory.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(sealed_directory.parent)
        os.replace(temporary_directory, sealed_directory)
        _fsync_directory(sealed_directory.parent)
        return sealed_directory

    def verify(
        self,
        sealed_directory: Path,
        *,
        expected_action_id: str | None = None,
        expected_workspace_id: str | None = None,
        expected_campaign_id: str | None = None,
        expected_study_id: str | None = None,
        expected_attempt_id: str | None = None,
        expected_manifest_revision: int | None = None,
        expected_candidate_digest: str | None = None,
        expected_input_digest: str | None = None,
        expected_claim_generation: int | None = None,
    ) -> SealedActionResult:
        """Verify signature, fencing identity, output hashes, and exact file coverage."""

        sealed_directory = sealed_directory.resolve()
        try:
            envelope = json.loads((sealed_directory / SEAL_FILENAME).read_text(encoding="utf-8"))
            if envelope.get("schema_version") != "sealed_action_result_envelope.v1":
                raise ValueError("wrong envelope schema")
            if envelope.get("key_version") != self.key_version:
                raise ValueError("wrong seal key version")
            manifest = SealedActionResult.model_validate(envelope["manifest"])
            if not hmac.compare_digest(envelope["signature"], self._signature(manifest)):
                raise ValueError("signature mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ArtifactSealError(f"{ArtifactSealError.code}: invalid seal envelope") from exc

        if expected_action_id is not None and manifest.action_id != expected_action_id:
            raise ArtifactSealError(f"{ArtifactSealError.code}: action identity mismatch")
        expected_identity = {
            "workspace": (expected_workspace_id, manifest.workspace_id),
            "campaign": (expected_campaign_id, manifest.campaign_id),
            "study": (expected_study_id, manifest.study_id),
            "attempt": (expected_attempt_id, manifest.attempt_id),
            "manifest revision": (expected_manifest_revision, manifest.manifest_revision),
            "candidate digest": (expected_candidate_digest, manifest.candidate_digest),
        }
        for label, (expected, actual) in expected_identity.items():
            if expected is not None and expected != actual:
                raise ArtifactSealError(f"{ArtifactSealError.code}: {label} mismatch")
        if expected_input_digest is not None and manifest.input_digest != expected_input_digest:
            raise ArtifactSealError(f"{ArtifactSealError.code}: input digest mismatch")
        if (
            expected_claim_generation is not None
            and manifest.claim_generation != expected_claim_generation
        ):
            raise ArtifactSealError(f"{ArtifactSealError.code}: claim generation mismatch")

        actual_files = {
            path.relative_to(sealed_directory).as_posix()
            for path in sealed_directory.rglob("*")
            if path.is_file() and path.name != SEAL_FILENAME
        }
        expected_files = {output.path for output in manifest.outputs}
        if actual_files != expected_files:
            raise ArtifactSealError(f"{ArtifactSealError.code}: sealed output set changed")
        for output in manifest.outputs:
            digest, size = _hash_file(_safe_output_path(sealed_directory, output.path))
            if digest != output.sha256 or size != output.size_bytes:
                raise ArtifactSealError(
                    f"{ArtifactSealError.code}: sealed output hash mismatch: {output.path}"
                )
        return manifest


__all__ = ["ArtifactSealError", "ArtifactSealer", "SEAL_FILENAME"]
