"""Hosted MCP OAuth storage and loopback callback tests."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

import bashgym.mcp.oauth as oauth_module
from bashgym.mcp.oauth import CredentialOAuthStorage, LoopbackOAuthCallback


@pytest.mark.asyncio
async def test_oauth_storage_keeps_tokens_in_credential_backend(monkeypatch):
    values: dict[str, str] = {}
    monkeypatch.setattr(oauth_module, "get_secret", values.get)
    monkeypatch.setattr(oauth_module, "has_secret", lambda key: key in values)
    monkeypatch.setattr(
        oauth_module, "set_secret", lambda key, value: values.__setitem__(key, value)
    )
    monkeypatch.setattr(
        oauth_module, "delete_secret", lambda key: values.pop(key, None) is not None
    )

    storage = CredentialOAuthStorage("MCP_OAUTH_TEST")
    tokens = OAuthToken(
        access_token="access-do-not-log",
        refresh_token="refresh-do-not-log",
        expires_in=3600,
    )
    client = OAuthClientInformationFull(
        redirect_uris=["http://127.0.0.1:8765/callback"],
        client_id="client-id",
    )
    await storage.set_tokens(tokens)
    await storage.set_client_info(client)

    assert (await storage.get_tokens()).access_token == "access-do-not-log"
    assert (await storage.get_client_info()).client_id == "client-id"
    assert storage.has_tokens() is True
    assert storage.clear() is True
    assert values == {}


@pytest.mark.asyncio
async def test_loopback_callback_accepts_one_authorization_code():
    callback = LoopbackOAuthCallback(timeout_seconds=2)
    await callback.start()
    parsed = urlsplit(callback.redirect_uri)
    waiting = asyncio.create_task(callback.wait_for_code())
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    writer.write(
        b"GET /callback?code=code-123&state=state-456 HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()

    assert b"200 OK" in response
    assert await waiting == ("code-123", "state-456")
    await callback.close()
