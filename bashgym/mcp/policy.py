"""Security policy helpers for MCP client connections.

This module is deliberately transport-agnostic.  It validates remote targets,
resolves opaque secret references, and turns a stdio argv into an approved,
fingerprinted launch description.  It never launches a process or opens a
network connection itself.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class McpPolicyError(ValueError):
    """Raised when an MCP connection violates local safety policy."""


class SecretResolutionError(McpPolicyError):
    """Raised when an opaque secret reference cannot be resolved safely."""


class ExecutableFingerprintMismatchError(McpPolicyError):
    """Raised when a stdio executable changed after it was approved."""


SecretResolver = Callable[[str], str | None]
AddressResolver = Callable[..., Sequence[tuple]]

_SECRET_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_SENSITIVE_NAME_RE = re.compile(
    r"(?:authorization|cookie|api[-_]?key|token|secret|password)", re.IGNORECASE
)
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_METADATA_ADDRESSES = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("100.100.100.200"),
}


@dataclass(frozen=True)
class ValidatedRemoteUrl:
    """A normalized remote MCP URL and every address observed during policy checks."""

    url: str
    host: str
    port: int
    addresses: tuple[str, ...]
    is_loopback: bool


@dataclass(frozen=True)
class ExecutableFingerprint:
    """Stable approval material for one concrete executable file."""

    path: str
    size: int
    modified_ns: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class StdioLaunch:
    """Validated argv-only stdio launch material."""

    command: str
    args: tuple[str, ...]
    fingerprint: ExecutableFingerprint


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise McpPolicyError("session_id must be a simple non-empty identifier")
    return session_id


def validate_secret_ref_name(name: str) -> str:
    """Validate an opaque secret reference without resolving or logging its value."""

    if not isinstance(name, str) or not _SECRET_REF_RE.fullmatch(name):
        raise SecretResolutionError(
            "secret reference names must use letters, digits, and underscores"
        )
    return name


def resolve_secret_reference(name: str, resolver: SecretResolver) -> str:
    """Resolve one opaque secret reference through an injected credential provider."""

    validate_secret_ref_name(name)
    if not callable(resolver):
        raise SecretResolutionError("a secret resolver is required")
    value = resolver(name)
    if not isinstance(value, str) or not value:
        raise SecretResolutionError(f"secret reference is unavailable: {name}")
    if "\x00" in value or "\r" in value or "\n" in value:
        raise SecretResolutionError(f"secret reference contains unsafe control characters: {name}")
    return value


def resolve_secret_mapping(
    references: Mapping[str, str] | None,
    resolver: SecretResolver | None,
    *,
    target: str,
) -> dict[str, str]:
    """Resolve environment or header values from opaque secret reference names."""

    if not references:
        return {}
    if resolver is None:
        raise SecretResolutionError("secret references require an injected resolver")

    output: dict[str, str] = {}
    name_re = _ENV_NAME_RE if target == "environment" else _HEADER_NAME_RE
    for output_name, reference_name in references.items():
        if not isinstance(output_name, str) or not name_re.fullmatch(output_name):
            raise SecretResolutionError(f"invalid {target} name")
        output[output_name] = resolve_secret_reference(reference_name, resolver)
    return output


def prepare_http_headers(
    headers: Mapping[str, str] | None = None,
    secret_header_refs: Mapping[str, str] | None = None,
    resolver: SecretResolver | None = None,
) -> dict[str, str]:
    """Build HTTP headers while forcing credential-shaped values through secret refs."""

    output: dict[str, str] = {}
    for name, value in (headers or {}).items():
        if not isinstance(name, str) or not _HEADER_NAME_RE.fullmatch(name):
            raise McpPolicyError("invalid HTTP header name")
        if _SENSITIVE_NAME_RE.search(name):
            raise McpPolicyError(f"credential-shaped header must use a secret reference: {name}")
        if not isinstance(value, str) or any(char in value for char in ("\x00", "\r", "\n")):
            raise McpPolicyError(f"unsafe HTTP header value: {name}")
        output[name] = value

    resolved = resolve_secret_mapping(secret_header_refs, resolver, target="header")
    existing_names = {name.lower() for name in output}
    for name, value in resolved.items():
        if name.lower() in existing_names:
            raise McpPolicyError(f"duplicate HTTP header: {name}")
        output[name] = value
    return output


def prepare_stdio_environment(
    environment: Mapping[str, str] | None = None,
    secret_env_refs: Mapping[str, str] | None = None,
    resolver: SecretResolver | None = None,
) -> dict[str, str]:
    """Build an explicit child environment without accepting inline credentials."""

    if os.name == "nt":
        allowed_parent_names = (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
        )
    else:
        allowed_parent_names = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
    # Never hand the MCP SDK ``None`` here.  Its stdio transport has its own
    # small platform allowlist, and this explicit map ensures application/API
    # secrets from the parent environment are not intentionally propagated.
    output: dict[str, str] = {
        name: os.environ[name] for name in allowed_parent_names if os.environ.get(name)
    }
    for name, value in (environment or {}).items():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise McpPolicyError("invalid environment variable name")
        if _SENSITIVE_NAME_RE.search(name):
            raise McpPolicyError(
                f"credential-shaped environment variable must use a secret reference: {name}"
            )
        if not isinstance(value, str) or "\x00" in value:
            raise McpPolicyError(f"unsafe environment value: {name}")
        output[name] = value

    resolved = resolve_secret_mapping(secret_env_refs, resolver, target="environment")
    duplicates = set(output).intersection(resolved)
    if duplicates:
        raise McpPolicyError(f"duplicate environment variable: {sorted(duplicates)[0]}")
    output.update(resolved)
    return output


def _resolved_ip_addresses(
    host: str,
    port: int,
    resolver: AddressResolver,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        results = resolver(host, port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise McpPolicyError(f"could not resolve MCP host: {host}") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for result in results:
        try:
            raw_address = result[4][0]
            addresses.add(ipaddress.ip_address(raw_address.split("%", 1)[0]))
        except (IndexError, TypeError, ValueError) as exc:
            raise McpPolicyError("resolver returned an invalid address") from exc
    if not addresses:
        raise McpPolicyError(f"MCP host resolved to no addresses: {host}")
    return tuple(sorted(addresses, key=lambda address: (address.version, int(address))))


def _is_restricted_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return bool(
        address in _METADATA_ADDRESSES
        or address.is_private
        or address.is_link_local
        or address.is_loopback
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )


def validate_remote_url(
    url: str,
    *,
    allow_private_network: bool = False,
    resolver: AddressResolver = socket.getaddrinfo,
) -> ValidatedRemoteUrl:
    """Validate a Streamable HTTP target before the MCP SDK opens it.

    Cleartext HTTP is accepted only when every resolved address is loopback.
    By default every private, link-local, metadata, reserved, multicast, and
    loopback address is rejected except for that explicit loopback development
    case.  All DNS answers must satisfy the same rule.
    """

    if not isinstance(url, str) or not url.strip():
        raise McpPolicyError("MCP URL is required")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise McpPolicyError("invalid MCP URL") from exc

    if parsed.scheme not in {"https", "http"}:
        raise McpPolicyError("remote MCP transport requires HTTPS")
    if not parsed.hostname:
        raise McpPolicyError("MCP URL requires a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise McpPolicyError("credentials are not allowed in MCP URLs")
    if parsed.fragment:
        raise McpPolicyError("fragments are not allowed in MCP URLs")

    port = port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolved_ip_addresses(parsed.hostname, port, resolver)
    all_loopback = all(address.is_loopback for address in addresses)

    if parsed.scheme == "http" and not all_loopback:
        raise McpPolicyError("cleartext HTTP is allowed only for explicit loopback MCPs")
    if not allow_private_network and not all_loopback:
        restricted = [address for address in addresses if _is_restricted_address(address)]
        if restricted:
            raise McpPolicyError(f"MCP host resolves to a restricted address: {restricted[0]}")
    if parsed.scheme == "https" and not allow_private_network and all_loopback:
        # Loopback is an explicit local-development exception for either scheme.
        pass

    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
    return ValidatedRemoteUrl(
        url=normalized,
        host=parsed.hostname,
        port=port,
        addresses=tuple(str(address) for address in addresses),
        is_loopback=all_loopback,
    )


def fingerprint_executable(path: str | os.PathLike[str]) -> ExecutableFingerprint:
    """Hash a concrete executable so later launches can detect replacement."""

    executable = Path(path).resolve(strict=True)
    if not executable.is_file():
        raise McpPolicyError("stdio command must resolve to an executable file")
    stat_result = executable.stat()
    digest = hashlib.sha256()
    with executable.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ExecutableFingerprint(
        path=str(executable),
        size=stat_result.st_size,
        modified_ns=stat_result.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def prepare_stdio_launch(
    command: str,
    args: Sequence[str] | None = None,
    *,
    expected_fingerprint: ExecutableFingerprint | Mapping[str, object] | None = None,
) -> StdioLaunch:
    """Resolve and fingerprint an argv-only stdio launch; no shell is involved."""

    if not isinstance(command, str) or not command or "\x00" in command:
        raise McpPolicyError("stdio command is required")
    if isinstance(args, (str, bytes)):
        raise McpPolicyError("stdio args must be an argv sequence, not a shell string")
    argv = tuple(args or ())
    if any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
        raise McpPolicyError("stdio args must be safe strings")

    resolved = shutil.which(command)
    if resolved is None:
        candidate = Path(command).expanduser()
        if candidate.is_file():
            resolved = str(candidate)
    if resolved is None:
        raise McpPolicyError(f"stdio executable was not found: {command}")

    fingerprint = fingerprint_executable(resolved)
    if expected_fingerprint is not None:
        expected = (
            expected_fingerprint.to_dict()
            if isinstance(expected_fingerprint, ExecutableFingerprint)
            else dict(expected_fingerprint)
        )
        if fingerprint.to_dict() != expected:
            raise ExecutableFingerprintMismatchError(
                "stdio executable fingerprint changed after approval"
            )
    return StdioLaunch(command=fingerprint.path, args=argv, fingerprint=fingerprint)
