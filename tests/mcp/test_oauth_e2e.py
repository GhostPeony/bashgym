"""End-to-end hosted MCP OAuth discovery, browser callback, and token use."""

from __future__ import annotations

import asyncio
import json
import socket
from urllib.parse import parse_qs, urlencode
from urllib.request import urlopen

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

import bashgym.mcp.oauth as oauth_module
from bashgym.mcp.client_runtime import McpClientRuntime
from bashgym.mcp.oauth import OAuthRuntimeConfig


class _MemoryStorage:
    def __init__(self):
        self.tokens = None
        self.client_info = None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client_info

    async def set_client_info(self, client_info):
        self.client_info = client_info


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _oauth_mcp_app(base_url: str) -> FastAPI:
    app = FastAPI()

    @app.get("/.well-known/oauth-protected-resource")
    @app.get("/.well-known/oauth-protected-resource/mcp")
    async def protected_resource():
        return {
            "resource": f"{base_url}/mcp",
            "authorization_servers": [base_url],
            "scopes_supported": ["mcp.read"],
        }

    @app.get("/.well-known/oauth-authorization-server")
    async def authorization_metadata():
        return {
            "issuer": base_url,
            "authorization_endpoint": f"{base_url}/authorize",
            "token_endpoint": f"{base_url}/token",
            "registration_endpoint": f"{base_url}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["mcp.read"],
        }

    @app.post("/register")
    async def register(request: Request):
        payload = await request.json()
        return {**payload, "client_id": "bashgym-oauth-fixture"}

    @app.get("/authorize")
    async def authorize(request: Request):
        query = request.query_params
        redirect_uri = query["redirect_uri"]
        callback_query = urlencode({"code": "fixture-code", "state": query["state"]})
        return RedirectResponse(f"{redirect_uri}?{callback_query}")

    @app.post("/token")
    async def token(request: Request):
        form = parse_qs((await request.body()).decode())
        assert form["code"] == ["fixture-code"]
        return {
            "access_token": "fixture-access-token",
            "refresh_token": "fixture-refresh-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "mcp.read",
        }

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp(request: Request):
        if request.headers.get("authorization") != "Bearer fixture-access-token":
            return Response(
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer resource_metadata="{base_url}/.well-known/'
                        'oauth-protected-resource"'
                    )
                },
            )
        if request.method != "POST":
            return Response(status_code=405)
        payload = json.loads(await request.body())
        request_id = payload.get("id")
        method = payload.get("method")
        if method == "notifications/initialized":
            return Response(status_code=202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "oauth-fixture", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "oauth_ping",
                        "description": "Prove that OAuth completed.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        else:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "not found"},
                }
            )
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

    return app


@pytest.mark.asyncio
async def test_hosted_oauth_opens_login_callback_and_connects(monkeypatch):
    port = _unused_port()
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            _oauth_mcp_app(base_url),
            host="127.0.0.1",
            port=port,
            log_level="critical",
        )
    )
    server_task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 10
    while not server.started and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert server.started

    def complete_login(url: str) -> bool:
        with urlopen(url, timeout=10) as response:
            response.read()
        return True

    monkeypatch.setattr(oauth_module.webbrowser, "open_new_tab", complete_login)
    storage = _MemoryStorage()
    try:
        async with McpClientRuntime() as runtime:
            connected = await runtime.connect_http(
                "oauth-reference",
                f"{base_url}/mcp",
                oauth_config=OAuthRuntimeConfig(
                    storage=storage,
                    scopes=("mcp.read",),
                    timeout_seconds=10,
                ),
            )
            assert connected["initialization"]["serverInfo"]["name"] == "oauth-fixture"
            assert {tool["name"] for tool in connected["inventory"]["tools"]} == {"oauth_ping"}
            assert storage.tokens.access_token == "fixture-access-token"
            assert storage.client_info.client_id == "bashgym-oauth-fixture"
    finally:
        server.should_exit = True
        await server_task
