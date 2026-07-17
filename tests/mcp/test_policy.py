from __future__ import annotations

import os
import socket
import sys

import pytest

from bashgym.mcp import policy
from bashgym.mcp.policy import (
    ExecutableFingerprint,
    ExecutableFingerprintMismatchError,
    McpPolicyError,
    SecretResolutionError,
    prepare_http_headers,
    prepare_stdio_environment,
    prepare_stdio_launch,
    resolve_secret_reference,
    validate_remote_url,
    validate_secret_ref_name,
)


def resolver_for(*addresses: str):
    def resolve(host, port, *, type=socket.SOCK_STREAM):
        del host
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                type,
                socket.IPPROTO_TCP,
                "",
                (address, port, 0, 0) if ":" in address else (address, port),
            )
            for address in addresses
        ]

    return resolve


def test_remote_url_requires_https_except_loopback():
    public = resolver_for("93.184.216.34")
    assert validate_remote_url("https://mcp.example.test/mcp", resolver=public).port == 443
    with pytest.raises(McpPolicyError, match="cleartext"):
        validate_remote_url("http://mcp.example.test/mcp", resolver=public)

    local = validate_remote_url(
        "http://localhost:8765/mcp", resolver=resolver_for("127.0.0.1", "::1")
    )
    assert local.is_loopback is True
    assert set(local.addresses) == {"127.0.0.1", "::1"}


@pytest.mark.parametrize(
    "url, message",
    [
        ("https://user:password@example.test/mcp", "credentials"),
        ("https://example.test/mcp#fragment", "fragments"),
        ("ftp://example.test/mcp", "requires HTTPS"),
    ],
)
def test_remote_url_rejects_unsafe_url_components(url, message):
    with pytest.raises(McpPolicyError, match=message):
        validate_remote_url(url, resolver=resolver_for("93.184.216.34"))


@pytest.mark.parametrize(
    "address",
    ["10.0.0.4", "169.254.169.254", "169.254.1.20", "224.0.0.1", "0.0.0.0"],
)
def test_remote_url_rejects_every_restricted_dns_answer(address):
    with pytest.raises(McpPolicyError, match="restricted address"):
        validate_remote_url(
            "https://mcp.example.test/mcp",
            resolver=resolver_for("93.184.216.34", address),
        )


def test_private_network_requires_explicit_opt_in():
    result = validate_remote_url(
        "https://mcp.internal.test/mcp",
        allow_private_network=True,
        resolver=resolver_for("10.0.0.4"),
    )
    assert result.addresses == ("10.0.0.4",)


def test_secret_references_are_validated_and_resolved_by_callback():
    calls = []

    def resolver(name):
        calls.append(name)
        return "Bearer opaque-value"

    assert validate_secret_ref_name("MCP_API_TOKEN") == "MCP_API_TOKEN"
    assert resolve_secret_reference("MCP_API_TOKEN", resolver) == "Bearer opaque-value"
    assert calls == ["MCP_API_TOKEN"]
    with pytest.raises(SecretResolutionError):
        validate_secret_ref_name("../../token")
    with pytest.raises(SecretResolutionError, match="unavailable"):
        resolve_secret_reference("MISSING", lambda _: None)


def test_inline_credentials_are_rejected_but_opaque_refs_work(monkeypatch):
    with pytest.raises(McpPolicyError, match="secret reference"):
        prepare_http_headers({"Authorization": "Bearer inline"})
    with pytest.raises(McpPolicyError, match="secret reference"):
        prepare_stdio_environment({"MCP_API_TOKEN": "inline"})

    assert prepare_http_headers(
        {"X-Client": "bashgym"},
        {"Authorization": "MCP_TOKEN"},
        lambda _: "Bearer resolved",
    ) == {"X-Client": "bashgym", "Authorization": "Bearer resolved"}
    monkeypatch.setenv("BASHGYM_SECRET_CANARY", "must-not-reach-child")
    child_environment = prepare_stdio_environment(
        {"MODE": "test"}, {"MCP_API_TOKEN": "MCP_TOKEN"}, lambda _: "resolved"
    )
    assert child_environment["MODE"] == "test"
    assert child_environment["MCP_API_TOKEN"] == "resolved"
    assert "BASHGYM_SECRET_CANARY" not in child_environment
    assert child_environment.get("PATH") == os.environ.get("PATH")


def test_stdio_launch_is_argv_only_and_fingerprint_is_enforced():
    launch = prepare_stdio_launch(sys.executable, ["-V"])
    assert launch.command == launch.fingerprint.path
    assert launch.args == ("-V",)
    assert len(launch.fingerprint.sha256) == 64
    assert (
        prepare_stdio_launch(
            sys.executable,
            ["-V"],
            expected_fingerprint=launch.fingerprint,
        ).fingerprint
        == launch.fingerprint
    )

    changed = launch.fingerprint.to_dict()
    changed["sha256"] = "0" * 64
    with pytest.raises(ExecutableFingerprintMismatchError):
        prepare_stdio_launch(sys.executable, ["-V"], expected_fingerprint=changed)
    with pytest.raises(McpPolicyError, match="argv sequence"):
        prepare_stdio_launch(sys.executable, "-V")


def test_stdio_launch_preserves_virtualenv_invocation_path_when_fingerprint_resolves_target(
    monkeypatch,
    tmp_path,
):
    invocation = tmp_path / "venv" / "bin" / "python"
    target = tmp_path / "base" / "bin" / "python"
    invocation.parent.mkdir(parents=True)
    invocation.write_bytes(b"venv launcher")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"base interpreter")
    fingerprint = ExecutableFingerprint(
        path=str(target.resolve()),
        size=target.stat().st_size,
        modified_ns=target.stat().st_mtime_ns,
        sha256="a" * 64,
    )
    monkeypatch.setattr(policy.shutil, "which", lambda _command: str(invocation))
    monkeypatch.setattr(policy, "fingerprint_executable", lambda _path: fingerprint)

    launch = prepare_stdio_launch("python", ["-m", "example"])

    assert launch.command == str(invocation)
    assert launch.fingerprint == fingerprint
