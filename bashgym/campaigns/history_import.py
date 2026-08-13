"""Read-only historical evidence attestations, distinct from live action seals."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class HistoricalImportError(ValueError):
    code = "campaign_historical_import_invalid"


@dataclass(frozen=True)
class HistoricalSource:
    logical_name: str
    path: Path
    schema_name: str
    provenance: dict[str, Any]


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class HistoricalImportAttestor:
    """Hash approved evidence without exposing paths or imitating a live worker result."""

    def __init__(
        self,
        key: bytes,
        *,
        key_version: str,
        allowed_roots: tuple[Path, ...],
        protected_roots: tuple[Path, ...] = (),
    ) -> None:
        if len(key) < 32:
            raise ValueError("historical import key must contain at least 32 bytes")
        if not key_version:
            raise ValueError("historical import key version is required")
        if not allowed_roots:
            raise ValueError("historical import requires at least one allowed root")
        self._key = key
        self.key_version = key_version
        self.allowed_roots = tuple(root.resolve() for root in allowed_roots)
        self.protected_roots = tuple(root.resolve() for root in protected_roots)

    def _signature(self, manifest: dict[str, Any]) -> str:
        return hmac.new(self._key, _canonical_bytes(manifest), hashlib.sha256).hexdigest()

    def attest(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        sources: tuple[HistoricalSource, ...],
        imported_at: datetime,
        import_reason: str,
    ) -> dict[str, Any]:
        if not sources:
            raise HistoricalImportError(f"{HistoricalImportError.code}: no sources")
        if not import_reason.strip():
            raise HistoricalImportError(f"{HistoricalImportError.code}: reason required")
        seen: set[str] = set()
        items = []
        for source in sorted(sources, key=lambda item: item.logical_name):
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", source.logical_name)
                or source.logical_name in seen
            ):
                raise HistoricalImportError(
                    f"{HistoricalImportError.code}: invalid logical source name"
                )
            seen.add(source.logical_name)
            # Reject the user-supplied leaf before resolving it.  Checking the
            # resolved path alone cannot detect a symlink because ``resolve``
            # deliberately dereferences it.
            if source.path.is_symlink():
                raise HistoricalImportError(
                    f"{HistoricalImportError.code}: source must be a regular file"
                )
            resolved = source.path.resolve()
            if _inside(resolved, self.protected_roots):
                raise HistoricalImportError(
                    f"{HistoricalImportError.code}: protected source excluded"
                )
            if not _inside(resolved, self.allowed_roots):
                raise HistoricalImportError(
                    f"{HistoricalImportError.code}: source outside approved roots"
                )
            if not resolved.is_file():
                raise HistoricalImportError(
                    f"{HistoricalImportError.code}: source must be a regular file"
                )
            digest, size = _hash_file(resolved)
            items.append(
                {
                    "logical_name": source.logical_name,
                    "schema_name": source.schema_name,
                    "sha256": digest,
                    "size_bytes": size,
                    "provenance": source.provenance,
                }
            )
        manifest = {
            "schema_version": "import_attestation.v1",
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "study_id": study_id,
            "read_only": True,
            "live_action_result": False,
            "import_reason": import_reason,
            "sources": items,
            "imported_at": imported_at.isoformat(),
        }
        return {
            "schema_version": "import_attestation_envelope.v1",
            "key_version": self.key_version,
            "manifest": manifest,
            "signature": self._signature(manifest),
        }

    def verify(self, envelope: dict[str, Any]) -> dict[str, Any]:
        try:
            if envelope["schema_version"] != "import_attestation_envelope.v1":
                raise ValueError("wrong envelope")
            if envelope["key_version"] != self.key_version:
                raise ValueError("wrong key version")
            manifest = envelope["manifest"]
            if manifest["schema_version"] != "import_attestation.v1":
                raise ValueError("wrong manifest")
            if (
                manifest.get("read_only") is not True
                or manifest.get("live_action_result") is not False
            ):
                raise ValueError("wrong import semantics")
            if not hmac.compare_digest(envelope["signature"], self._signature(manifest)):
                raise ValueError("signature mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoricalImportError(
                f"{HistoricalImportError.code}: invalid attestation"
            ) from exc
        return manifest

    def write(self, envelope: dict[str, Any], destination: Path) -> Path:
        self.verify(envelope)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise HistoricalImportError(f"{HistoricalImportError.code}: destination already exists")
        destination.write_bytes(_canonical_bytes(envelope) + b"\n")
        return destination


__all__ = [
    "HistoricalImportAttestor",
    "HistoricalImportError",
    "HistoricalSource",
]
