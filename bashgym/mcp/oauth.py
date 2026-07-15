"""Interactive OAuth support for hosted Streamable HTTP MCP servers."""

from __future__ import annotations

import asyncio
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qs, urlsplit

from mcp.client.auth import OAuthClientProvider, OAuthFlowError, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from bashgym.secrets import delete_secret, get_secret, has_secret, set_secret


def oauth_storage_namespace(workspace_id: str, profile_id: str, server_url: str) -> str:
    """Return a stable opaque credential namespace without embedding the URL."""

    digest = sha256(f"{workspace_id}\0{profile_id}\0{server_url}".encode()).hexdigest()[:32]
    return f"MCP_OAUTH_{digest.upper()}"


class CredentialOAuthStorage(TokenStorage):
    """Store OAuth tokens and dynamic client information outside profile storage."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.tokens_key = f"{namespace}_TOKENS"
        self.client_key = f"{namespace}_CLIENT"

    async def get_tokens(self) -> OAuthToken | None:
        raw = await asyncio.to_thread(get_secret, self.tokens_key)
        return OAuthToken.model_validate_json(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(set_secret, self.tokens_key, tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = await asyncio.to_thread(get_secret, self.client_key)
        return OAuthClientInformationFull.model_validate_json(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(set_secret, self.client_key, client_info.model_dump_json())

    def has_tokens(self) -> bool:
        return has_secret(self.tokens_key)

    def clear(self) -> bool:
        return delete_secret(self.tokens_key) | delete_secret(self.client_key)


class LoopbackOAuthCallback:
    """One-shot loopback callback with no query logging or durable state."""

    def __init__(self, *, port: int | None = None, timeout_seconds: float = 300):
        self.requested_port = port or 0
        self.timeout_seconds = timeout_seconds
        self.server: asyncio.Server | None = None
        self.result: asyncio.Future[tuple[str, str | None]] | None = None
        self.redirect_uri = ""
        self.authorization_url: str | None = None

    async def start(self) -> None:
        if self.server is not None:
            return
        self.result = asyncio.get_running_loop().create_future()
        self.server = await asyncio.start_server(
            self._handle_request,
            host="127.0.0.1",
            port=self.requested_port,
            limit=16 * 1024,
        )
        socket = self.server.sockets[0]
        port = int(socket.getsockname()[1])
        self.redirect_uri = f"http://127.0.0.1:{port}/callback"

    async def open_browser(self, authorization_url: str) -> None:
        self.authorization_url = authorization_url
        opened = await asyncio.to_thread(webbrowser.open_new_tab, authorization_url)
        if not opened:
            raise OAuthFlowError("The authorization page could not be opened in a browser.")

    async def wait_for_code(self) -> tuple[str, str | None]:
        if self.result is None:
            raise OAuthFlowError("OAuth callback listener is not running.")
        try:
            return await asyncio.wait_for(asyncio.shield(self.result), self.timeout_seconds)
        except TimeoutError as exc:
            raise OAuthFlowError("OAuth sign-in timed out before the callback arrived.") from exc

    async def _handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        status = "200 OK"
        title = "MCP sign-in complete"
        detail = "You can close this tab and return to BashGym."
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            first_line = request.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            method, target, _version = first_line.split(" ", 2)
            parsed = urlsplit(target)
            if method != "GET" or parsed.path != "/callback":
                status = "404 Not Found"
                title = "Invalid callback"
                detail = "This callback path is not valid."
            else:
                query = parse_qs(parsed.query)
                error = query.get("error", [None])[0]
                code = query.get("code", [None])[0]
                state = query.get("state", [None])[0]
                if error:
                    status = "400 Bad Request"
                    title = "MCP sign-in declined"
                    detail = "Authorization was not completed. Return to BashGym to try again."
                    if self.result is not None and not self.result.done():
                        self.result.set_exception(
                            OAuthFlowError("Authorization was declined or failed.")
                        )
                elif not code:
                    status = "400 Bad Request"
                    title = "Missing authorization code"
                    detail = "The provider did not return an authorization code."
                elif self.result is not None and not self.result.done():
                    self.result.set_result((code, state))
        except (TimeoutError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            status = "400 Bad Request"
            title = "Invalid OAuth callback"
            detail = "The callback request could not be read."
        body = (
            "<!doctype html><meta charset='utf-8'><title>"
            + title
            + "</title><style>body{font:16px system-ui;max-width:42rem;margin:12vh auto;padding:2rem;"
            "background:#17131f;color:#f4efff}main{border:2px solid #a78bfa;padding:2rem;"
            "border-radius:12px}h1{color:#c4b5fd}</style><main><h1>"
            + title
            + "</h1><p>"
            + detail
            + "</p></main>"
        ).encode()
        response = (
            f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode("ascii") + body
        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if self.result is not None and not self.result.done():
            self.result.cancel()


@dataclass(frozen=True)
class OAuthRuntimeConfig:
    storage: CredentialOAuthStorage
    scopes: tuple[str, ...] = ()
    callback_port: int | None = None
    client_id: str | None = None
    client_secret: str | None = None
    timeout_seconds: float = 300


@asynccontextmanager
async def oauth_provider(
    server_url: str, config: OAuthRuntimeConfig
) -> AsyncIterator[OAuthClientProvider]:
    """Create an official SDK provider and keep its callback listener alive."""

    callback = LoopbackOAuthCallback(
        port=config.callback_port, timeout_seconds=config.timeout_seconds
    )
    await callback.start()
    metadata = OAuthClientMetadata(
        redirect_uris=[callback.redirect_uri],
        client_name="BashGym MCP Server",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=" ".join(config.scopes) or None,
    )
    if config.client_id:
        existing = await config.storage.get_client_info()
        if existing is None or existing.client_id != config.client_id:
            client_payload = metadata.model_dump()
            client_payload["token_endpoint_auth_method"] = (
                "client_secret_basic" if config.client_secret else "none"
            )
            await config.storage.set_client_info(
                OAuthClientInformationFull(
                    **client_payload,
                    client_id=config.client_id,
                    client_secret=config.client_secret,
                )
            )
    provider = OAuthClientProvider(
        server_url,
        metadata,
        config.storage,
        redirect_handler=callback.open_browser,
        callback_handler=callback.wait_for_code,
        timeout=config.timeout_seconds,
    )
    try:
        yield provider
    finally:
        await callback.close()


def oauth_status(storage: CredentialOAuthStorage) -> dict[str, Any]:
    return {"has_tokens": storage.has_tokens(), "storage": "credential_store"}


__all__ = [
    "CredentialOAuthStorage",
    "LoopbackOAuthCallback",
    "OAuthRuntimeConfig",
    "oauth_provider",
    "oauth_status",
    "oauth_storage_namespace",
]
