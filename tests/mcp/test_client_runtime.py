from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from bashgym.mcp.client_runtime import (
    InventoryLimitError,
    McpClientRuntime,
    SessionClosedError,
    SessionNotFoundError,
    ToolCallTimeoutError,
    ToolResultTooLargeError,
    _inventory,
    _paginate,
)
from bashgym.mcp.oauth import OAuthRuntimeConfig

ROOT = Path(__file__).resolve().parents[2]


class _MemoryOAuthStorage:
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


async def _connect_reference_stdio(runtime: McpClientRuntime, session_id: str = "reference"):
    return await runtime.connect_stdio(
        session_id,
        sys.executable,
        ["-m", "bashgym.mcp.reference_server"],
        cwd=str(ROOT),
        environment={"PYTHONPATH": str(ROOT)},
    )


@pytest.mark.asyncio
async def test_stdio_runtime_initializes_inventories_calls_refreshes_and_closes():
    runtime = McpClientRuntime()
    connected = await _connect_reference_stdio(runtime)
    try:
        assert connected["transport"] == "stdio"
        assert connected["initialization"]["protocolVersion"] in {
            "2025-06-18",
            "2025-11-25",
        }
        inventory = connected["inventory"]
        assert "read_fixture" in {tool["name"] for tool in inventory["tools"]}
        assert "reference://guide" in {resource["uri"] for resource in inventory["resources"]}
        assert inventory["resourceTemplates"]
        assert "inspect_fixture" in {prompt["name"] for prompt in inventory["prompts"]}
        initialization = await runtime.initialization("reference")
        initialization["protocolVersion"] = "mutated-by-caller"
        assert (await runtime.initialization("reference"))["protocolVersion"] != "mutated-by-caller"

        read_result = await runtime.call_tool("reference", "read_fixture", {"name": "alpha"})
        assert read_result["isError"] is False
        assert "first deterministic" in read_result["content"][0]["text"]

        structured = await runtime.call_tool("reference", "structured_sum", {"values": [3, 1, 4]})
        assert structured["structuredContent"] == {
            "total": 8,
            "count": 3,
            "minimum": 1,
            "maximum": 4,
        }

        invalid = await runtime.call_tool("reference", "require_positive", {"value": -1})
        assert invalid["isError"] is True
        refreshed = await runtime.refresh("reference")
        assert len(refreshed["tools"]) == len(inventory["tools"])
    finally:
        await runtime.aclose()

    with pytest.raises(SessionNotFoundError):
        await runtime.refresh("reference")


@pytest.mark.asyncio
async def test_call_timeout_cancels_request_without_killing_session():
    async with McpClientRuntime() as runtime:
        await _connect_reference_stdio(runtime, "timeout")
        with pytest.raises(ToolCallTimeoutError):
            await runtime.call_tool(
                "timeout",
                "slow_operation",
                {"delay_ms": 2000},
                timeout_seconds=0.05,
            )
        result = await runtime.call_tool("timeout", "read_fixture", {"name": "beta"})
        assert result["isError"] is False


@pytest.mark.asyncio
async def test_abort_interrupts_an_inflight_call_and_removes_session():
    runtime = McpClientRuntime()
    await _connect_reference_stdio(runtime, "abort")
    call = asyncio.create_task(
        runtime.call_tool(
            "abort",
            "slow_operation",
            {"delay_ms": 5000},
            timeout_seconds=10,
        )
    )
    await asyncio.sleep(0.1)
    await runtime.abort("abort")
    with pytest.raises(SessionClosedError):
        await call
    with pytest.raises(SessionNotFoundError):
        await runtime.refresh("abort")


@pytest.mark.asyncio
async def test_call_result_cap_is_enforced():
    async with McpClientRuntime() as runtime:
        await _connect_reference_stdio(runtime, "cap")
        with pytest.raises(ToolResultTooLargeError):
            await runtime.call_tool(
                "cap",
                "repeat_text",
                {"text": "abcdefghij", "count": 100},
                max_result_bytes=200,
            )


@pytest.mark.asyncio
async def test_pagination_has_page_item_and_cursor_cycle_limits():
    pages = {
        None: SimpleNamespace(tools=[{"name": "one"}], nextCursor="page-2"),
        "page-2": SimpleNamespace(tools=[{"name": "two"}], nextCursor=None),
    }

    async def list_tools(cursor=None):
        return pages[cursor]

    assert await _paginate(list_tools, "tools", max_pages=2, max_items=2) == [
        {"name": "one"},
        {"name": "two"},
    ]
    with pytest.raises(InventoryLimitError, match="page limit"):
        await _paginate(list_tools, "tools", max_pages=1, max_items=2)
    with pytest.raises(InventoryLimitError, match="item limit"):
        await _paginate(list_tools, "tools", max_pages=2, max_items=1)

    async def cyclic(cursor=None):
        del cursor
        return SimpleNamespace(tools=[], nextCursor="same")

    with pytest.raises(InventoryLimitError, match="repeated a cursor"):
        await _paginate(cyclic, "tools", max_pages=3, max_items=2)


@pytest.mark.asyncio
async def test_optional_inventory_method_not_found_becomes_a_warning():
    class PartialResourceSession:
        async def list_resources(self, cursor=None):
            del cursor
            return SimpleNamespace(resources=[{"uri": "fixture://one"}], nextCursor=None)

        async def list_resource_templates(self, cursor=None):
            del cursor
            raise McpError(ErrorData(code=-32601, message="Method not found"))

    inventory = await _inventory(
        PartialResourceSession(),
        {"capabilities": {"resources": {}}},
        max_pages=2,
        max_items_per_kind=10,
    )

    assert inventory["resources"] == [{"uri": "fixture://one"}]
    assert inventory["resourceTemplates"] == []
    assert inventory["warnings"] == ["optional_method_not_supported:resources/templates/list"]


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def _wait_for_server(port: int, process: asyncio.subprocess.Process) -> None:
    deadline = asyncio.get_running_loop().time() + 30
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"reference HTTP server exited with {process.returncode}")
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.05)
        else:
            writer.close()
            await writer.wait_closed()
            del reader
            return
    raise TimeoutError("reference HTTP server did not start")


@pytest.mark.asyncio
async def test_streamable_http_runtime_uses_loopback_transport():
    port = _unused_loopback_port()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "bashgym.mcp.reference_server",
        "--transport",
        "streamable-http",
        "--port",
        str(port),
        cwd=ROOT,
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await _wait_for_server(port, process)
        async with McpClientRuntime() as runtime:
            connected = await runtime.connect_http(
                "http-reference",
                f"http://127.0.0.1:{port}/mcp",
                oauth_config=OAuthRuntimeConfig(storage=_MemoryOAuthStorage()),
            )
            assert connected["transport"] == "streamable_http"
            assert connected["target"]["isLoopback"] is True
            result = await runtime.call_tool("http-reference", "read_fixture", {"name": "alpha"})
            assert result["isError"] is False
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
