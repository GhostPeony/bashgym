"""
SSH Config Discovery Parser

Parses ~/.ssh/config to find candidate SSH devices for remote training.
Supports Host blocks, HostName, User, Port, IdentityFile directives,
and one level of Include expansion.
"""

import glob
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Hosts to exclude from device discovery — Git forges, loopback, known CI/CD targets
FILTERED_HOSTS: frozenset = frozenset({
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "ssh.dev.azure.com",
    "vs-ssh.visualstudio.com",
    "localhost",
    "127.0.0.1",
    "::1",
})


@dataclass
class SSHHostEntry:
    """A single Host block parsed from an SSH config file."""

    alias: str
    hostname: Optional[str] = None   # HostName directive; falls back to alias if absent
    user: Optional[str] = None
    port: int = 22
    identity_file: Optional[str] = None


@dataclass
class SSHCandidate:
    """An SSH device candidate ready for display in the UI."""

    ssh_alias: str
    host: str                          # Resolved hostname (or alias if no HostName)
    username: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    already_added: bool = False
    existing_device_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_ssh_config_path() -> Path:
    """Return the platform-appropriate path to ~/.ssh/config."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", Path.home()))
    else:
        base = Path.home()
    return base / ".ssh" / "config"


def _strip_quotes(value: str) -> str:
    """Remove surrounding double or single quotes from a value."""
    value = value.strip()
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        return value[1:-1]
    return value


def _parse_ssh_config_file(path: Path) -> List[SSHHostEntry]:
    """
    Parse a single SSH config file and return a list of SSHHostEntry objects.

    Rules:
    - Lines starting with '#' are comments and are skipped.
    - 'Host' starts a new block; the previous block is saved.
    - Wildcard host patterns ('*' or '?') are skipped.
    - Keys and values are split on the first whitespace or '='.
    - IdentityFile is scoped to its Host block (not inherited from globals).
    """
    if not path.exists():
        return []

    entries: List[SSHHostEntry] = []

    # State for the current block
    current_alias: Optional[str] = None
    current_hostname: Optional[str] = None
    current_user: Optional[str] = None
    current_port: int = 22
    current_identity_file: Optional[str] = None

    def _save_block() -> None:
        """Persist the current block if it has a valid alias."""
        if current_alias is None:
            return
        # Skip wildcards
        if "*" in current_alias or "?" in current_alias:
            return
        entries.append(SSHHostEntry(
            alias=current_alias,
            hostname=current_hostname,
            user=current_user,
            port=current_port,
            identity_file=current_identity_file,
        ))

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Split on first '=' or first whitespace
        if "=" in line:
            key, _, value = line.partition("=")
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                key, value = parts[0], ""
            else:
                key, value = parts

        key = key.strip().lower()
        value = _strip_quotes(value.strip())

        if key == "host":
            # Save the previous block before starting a new one
            _save_block()
            # Reset state for the new block
            current_alias = value
            current_hostname = None
            current_user = None
            current_port = 22
            current_identity_file = None

        elif current_alias is None:
            # We're in the global section — only track things that don't
            # bleed into per-host blocks (IdentityFile is intentionally skipped here).
            pass

        elif key == "hostname":
            current_hostname = value

        elif key == "user":
            current_user = value

        elif key == "port":
            try:
                current_port = int(value)
            except ValueError:
                pass

        elif key == "identityfile":
            current_identity_file = value

    # Don't forget the last block
    _save_block()

    return entries


def _resolve_includes(config_path: Path) -> List[SSHHostEntry]:
    """
    Parse the main SSH config file and any files referenced by Include directives.

    Only one level of Include is processed (nested includes are ignored).
    Included files are parsed first; the main file is appended last so that
    explicit host blocks in the main file take precedence on lookup.
    """
    if not config_path.exists():
        return []

    all_entries: List[SSHHostEntry] = []
    include_paths: List[Path] = []

    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, _, value = line.partition("=")
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, value = parts

        if key.strip().lower() == "include":
            pattern = _strip_quotes(value.strip())
            # Expand ~ and environment variables in the pattern
            pattern = os.path.expandvars(os.path.expanduser(pattern))
            # Resolve relative paths relative to ~/.ssh/
            if not os.path.isabs(pattern):
                pattern = str(config_path.parent / pattern)
            for matched in sorted(glob.glob(pattern)):
                include_paths.append(Path(matched))

    # Parse included files first (standard SSH behaviour)
    for inc_path in include_paths:
        all_entries.extend(_parse_ssh_config_file(inc_path))

    # Then parse the main config
    all_entries.extend(_parse_ssh_config_file(config_path))

    return all_entries


def _is_filtered(entry: SSHHostEntry) -> bool:
    """Return True if this entry should be excluded from discovery."""
    effective_host = (entry.hostname or entry.alias or "").lower().strip()
    return effective_host in FILTERED_HOSTS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_ssh_devices(existing_devices: Optional[List[Dict]] = None) -> Dict:
    """
    Parse ~/.ssh/config and return candidate SSH devices.

    Parameters
    ----------
    existing_devices:
        Optional list of already-configured device dicts. Each dict should
        contain at least an 'id' key and one of 'host'/'hostname'/'ssh_alias'
        for cross-referencing.

    Returns
    -------
    dict with:
        'candidates'      — list of candidate dicts (SSHCandidate fields)
        'ssh_config_path' — string path to the SSH config that was parsed
    """
    config_path = get_ssh_config_path()
    entries = _resolve_includes(config_path)

    # Build a lookup of existing devices keyed by host/alias for O(1) lookups
    existing_lookup: Dict[str, str] = {}  # host_or_alias -> device_id
    if existing_devices:
        for device in existing_devices:
            device_id = str(device.get("id", ""))
            for field_name in ("host", "hostname", "ssh_alias", "alias"):
                val = device.get(field_name, "")
                if val:
                    existing_lookup[val.lower()] = device_id

    candidates: List[Dict] = []

    for entry in entries:
        if _is_filtered(entry):
            continue

        effective_host = entry.hostname or entry.alias

        # Check if this host is already in the device list
        lookup_keys = {entry.alias.lower(), effective_host.lower()} if effective_host else {entry.alias.lower()}
        existing_id: Optional[str] = None
        for key in lookup_keys:
            if key in existing_lookup:
                existing_id = existing_lookup[key]
                break

        # Expand ~ in key path
        key_path = entry.identity_file
        if key_path:
            key_path = os.path.expandvars(os.path.expanduser(key_path))

        candidate = SSHCandidate(
            ssh_alias=entry.alias,
            host=effective_host,
            username=entry.user,
            port=entry.port,
            key_path=key_path,
            already_added=existing_id is not None,
            existing_device_id=existing_id,
        )

        candidates.append({
            "ssh_alias": candidate.ssh_alias,
            "host": candidate.host,
            "username": candidate.username,
            "port": candidate.port,
            "key_path": candidate.key_path,
            "already_added": candidate.already_added,
            "existing_device_id": candidate.existing_device_id,
        })

    return {
        "candidates": candidates,
        "ssh_config_path": str(config_path),
    }
