"""Task-owned MCP client sessions for the Workbench inspection runtime.

The official MCP SDK's transport and ``ClientSession`` context managers are
opened, used, and closed by one long-lived actor task per connection.  Public
methods communicate with that task through a queue, avoiding cancel-scope and
task-affinity bugs when desktop API requests arrive on different tasks.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.auth import OAuthFlowError
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError

from bashgym.mcp.oauth import OAuthRuntimeConfig, oauth_provider
from bashgym.mcp.policy import (
    AddressResolver,
    ExecutableFingerprint,
    SecretResolver,
    prepare_http_headers,
    prepare_stdio_environment,
    prepare_stdio_launch,
    validate_remote_url,
    validate_session_id,
)


class McpRuntimeError(RuntimeError):
    """Base error for the managed MCP client runtime."""


class SessionAlreadyExistsError(McpRuntimeError):
    pass


class SessionNotFoundError(McpRuntimeError):
    pass


class SessionClosedError(McpRuntimeError):
    pass


class InventoryLimitError(McpRuntimeError):
    pass


class ToolResultTooLargeError(McpRuntimeError):
    pass


class ToolCallTimeoutError(McpRuntimeError):
    pass


class McpOAuthError(McpRuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.safe_message = message
        super().__init__(message)


@dataclass
class _Command:
    operation: Literal["refresh", "call_tool", "close"]
    parameters: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass
class _Actor:
    queue: asyncio.Queue[_Command]
    ready: asyncio.Future[dict[str, Any]]
    task: asyncio.Task[None]
    transport: Literal["stdio", "streamable_http"]
    fingerprint: ExecutableFingerprint | None = None


ConnectorFactory = Callable[[], AbstractAsyncContextManager[tuple[Any, ...]]]


def _plain_json(value: Any) -> Any:
    """Convert Pydantic SDK models and JSON-compatible values to plain data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _next_cursor(result: Any) -> str | None:
    cursor = getattr(result, "nextCursor", None)
    if cursor is None:
        cursor = getattr(result, "next_cursor", None)
    return cursor


async def _paginate(
    method: Callable[..., Awaitable[Any]],
    item_attribute: str,
    *,
    max_pages: int,
    max_items: int,
) -> list[dict[str, Any]]:
    """Exhaust one SDK list method with hard page/item/cursor-cycle limits."""

    if max_pages < 1 or max_items < 1:
        raise InventoryLimitError("inventory limits must be positive")
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for page_number in range(1, max_pages + 1):
        result = await method(cursor=cursor)
        page_items = list(getattr(result, item_attribute, []) or [])
        if len(items) + len(page_items) > max_items:
            raise InventoryLimitError(
                f"{item_attribute} inventory exceeds the {max_items}-item limit"
            )
        items.extend(_plain_json(item) for item in page_items)
        cursor = _next_cursor(result)
        if not cursor:
            return items
        if cursor in seen_cursors:
            raise InventoryLimitError(f"{item_attribute} pagination repeated a cursor")
        seen_cursors.add(cursor)
        if page_number == max_pages:
            raise InventoryLimitError(
                f"{item_attribute} inventory exceeds the {max_pages}-page limit"
            )
    return items


def _safe_httpx_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """MCP SDK HTTP factory with redirects and ambient proxy state disabled."""

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(30.0, read=300.0),
        auth=auth,
        follow_redirects=False,
        trust_env=False,
    )


@asynccontextmanager
async def _streamable_http_connector(
    url: str,
    *,
    headers: dict[str, str] | None,
    connect_timeout_seconds: float,
    stream_timeout_seconds: float,
    oauth_config: OAuthRuntimeConfig | None,
):
    timeout = httpx.Timeout(
        connect_timeout_seconds,
        read=stream_timeout_seconds,
    )
    async with AsyncExitStack() as stack:
        auth = (
            await stack.enter_async_context(oauth_provider(url, oauth_config))
            if oauth_config is not None
            else None
        )
        http_client = await stack.enter_async_context(
            _safe_httpx_client(headers=headers, timeout=timeout, auth=auth)
        )
        streams = await stack.enter_async_context(
            streamable_http_client(
                url,
                http_client=http_client,
                terminate_on_close=True,
            )
        )
        yield streams


async def _inventory(
    session: ClientSession,
    initialization: dict[str, Any],
    *,
    max_pages: int,
    max_items_per_kind: int,
) -> dict[str, Any]:
    capabilities = initialization.get("capabilities", {})
    output: dict[str, list[dict[str, Any]]] = {
        "tools": [],
        "resources": [],
        "resourceTemplates": [],
        "prompts": [],
        "warnings": [],
    }

    async def optional_list(
        method: Callable[..., Awaitable[Any]], item_attribute: str, method_name: str
    ) -> list[dict[str, Any]]:
        try:
            return await _paginate(
                method,
                item_attribute,
                max_pages=max_pages,
                max_items=max_items_per_kind,
            )
        except McpError as exc:
            if exc.error.code != -32601:
                raise
            output["warnings"].append(f"optional_method_not_supported:{method_name}")
            return []

    if capabilities.get("tools") is not None:
        output["tools"] = await optional_list(session.list_tools, "tools", "tools/list")
    if capabilities.get("resources") is not None:
        output["resources"] = await optional_list(
            session.list_resources, "resources", "resources/list"
        )
        output["resourceTemplates"] = await optional_list(
            session.list_resource_templates,
            "resourceTemplates",
            "resources/templates/list",
        )
    if capabilities.get("prompts") is not None:
        output["prompts"] = await optional_list(session.list_prompts, "prompts", "prompts/list")
    return output


async def _execute_command(
    command: _Command,
    session: ClientSession,
    initialization: dict[str, Any],
) -> tuple[bool, Any]:
    if command.operation == "close":
        return True, None
    if command.operation == "refresh":
        return False, await _inventory(session, initialization, **command.parameters)
    if command.operation == "call_tool":
        timeout_seconds = float(command.parameters["timeout_seconds"])
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        progress: list[dict[str, Any]] = []

        async def capture_progress(
            current: float,
            total: float | None,
            message: str | None,
        ) -> None:
            progress.append({"progress": current, "total": total, "message": message})

        try:
            async with asyncio.timeout(timeout_seconds):
                result = await session.call_tool(
                    command.parameters["tool_name"],
                    command.parameters.get("arguments"),
                    read_timeout_seconds=timedelta(seconds=timeout_seconds),
                    progress_callback=capture_progress,
                )
        except TimeoutError as exc:
            raise ToolCallTimeoutError(f"tool call exceeded {timeout_seconds:g} seconds") from exc

        normalized = _plain_json(result)
        normalized["progress"] = progress
        encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        max_result_bytes = int(command.parameters["max_result_bytes"])
        if max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive")
        if len(encoded) > max_result_bytes:
            raise ToolResultTooLargeError(
                f"tool result is {len(encoded)} bytes; limit is {max_result_bytes}"
            )
        return False, normalized
    raise McpRuntimeError(f"unknown actor operation: {command.operation}")


async def _session_actor(
    connector_factory: ConnectorFactory,
    queue: asyncio.Queue[_Command],
    ready: asyncio.Future[dict[str, Any]],
) -> None:
    """Own the entire SDK transport/session lifetime in this one task."""

    active_command: _Command | None = None
    try:
        async with connector_factory() as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                initialization = _plain_json(await session.initialize())
                if not ready.done():
                    ready.set_result(initialization)

                while True:
                    active_command = await queue.get()
                    try:
                        should_close, result = await _execute_command(
                            active_command, session, initialization
                        )
                    except asyncio.CancelledError:
                        if not active_command.future.done():
                            active_command.future.set_exception(
                                SessionClosedError("MCP session was aborted")
                            )
                        raise
                    except BaseException as exc:
                        if not active_command.future.done():
                            active_command.future.set_exception(exc)
                    else:
                        if not active_command.future.done():
                            active_command.future.set_result(result)
                        if should_close:
                            return
                    finally:
                        active_command = None
    except asyncio.CancelledError:
        if active_command is not None and not active_command.future.done():
            active_command.future.set_exception(SessionClosedError("MCP session was cancelled"))
        raise
    except BaseException as exc:
        if not ready.done():
            ready.set_exception(exc)
        elif active_command is not None and not active_command.future.done():
            active_command.future.set_exception(exc)
    finally:
        if not ready.done():
            ready.set_exception(SessionClosedError("MCP session closed before initialization"))
        while not queue.empty():
            pending = queue.get_nowait()
            if not pending.future.done():
                pending.future.set_exception(SessionClosedError("MCP session is closed"))


class McpClientRuntime:
    """Manage task-owned MCP SDK sessions for inspection and safe manual calls."""

    def __init__(
        self,
        *,
        max_pages: int = 20,
        max_items_per_kind: int = 1000,
        default_call_timeout_seconds: float = 30.0,
        default_max_result_bytes: int = 1024 * 1024,
    ) -> None:
        if max_pages < 1 or max_items_per_kind < 1:
            raise ValueError("inventory limits must be positive")
        self.max_pages = max_pages
        self.max_items_per_kind = max_items_per_kind
        self.default_call_timeout_seconds = default_call_timeout_seconds
        self.default_max_result_bytes = default_max_result_bytes
        self._actors: dict[str, _Actor] = {}

    async def connect_stdio(
        self,
        session_id: str,
        command: str,
        args: Sequence[str] | None = None,
        *,
        cwd: str | None = None,
        environment: Mapping[str, str] | None = None,
        secret_env_refs: Mapping[str, str] | None = None,
        resolve_secret: SecretResolver | None = None,
        expected_fingerprint: ExecutableFingerprint | Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        session_id = validate_session_id(session_id)
        launch = prepare_stdio_launch(command, args, expected_fingerprint=expected_fingerprint)
        child_environment = prepare_stdio_environment(environment, secret_env_refs, resolve_secret)
        parameters = StdioServerParameters(
            command=launch.command,
            args=list(launch.args),
            env=child_environment,
            cwd=cwd,
        )
        actor = await self._connect(
            session_id,
            "stdio",
            lambda: stdio_client(parameters),
            fingerprint=launch.fingerprint,
        )
        try:
            return await self._connect_result(session_id, actor)
        except BaseException:
            try:
                await self.abort(session_id)
            except (SessionClosedError, SessionNotFoundError):
                pass
            raise

    async def connect_http(
        self,
        session_id: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        secret_header_refs: Mapping[str, str] | None = None,
        resolve_secret: SecretResolver | None = None,
        allow_private_network: bool = False,
        address_resolver: AddressResolver | None = None,
        connect_timeout_seconds: float = 30.0,
        stream_timeout_seconds: float = 300.0,
        oauth_config: OAuthRuntimeConfig | None = None,
    ) -> dict[str, Any]:
        session_id = validate_session_id(session_id)
        validation_options: dict[str, Any] = {"allow_private_network": allow_private_network}
        if address_resolver is not None:
            validation_options["resolver"] = address_resolver
        target = validate_remote_url(url, **validation_options)
        resolved_headers = prepare_http_headers(headers, secret_header_refs, resolve_secret)
        actor = await self._connect(
            session_id,
            "streamable_http",
            lambda: _streamable_http_connector(
                target.url,
                headers=resolved_headers or None,
                connect_timeout_seconds=connect_timeout_seconds,
                stream_timeout_seconds=stream_timeout_seconds,
                oauth_config=oauth_config,
            ),
        )
        try:
            result = await self._connect_result(session_id, actor)
        except BaseException as exc:
            try:
                await self.abort(session_id)
            except (SessionClosedError, SessionNotFoundError):
                pass
            if isinstance(exc, OAuthFlowError):
                lowered = str(exc).lower()
                if "timed out" in lowered:
                    raise McpOAuthError(
                        "oauth_timeout", "MCP sign-in timed out before it completed."
                    ) from exc
                if "declined" in lowered:
                    raise McpOAuthError(
                        "oauth_declined", "MCP sign-in was declined or cancelled."
                    ) from exc
                if "opened" in lowered:
                    raise McpOAuthError(
                        "oauth_browser_unavailable",
                        "The MCP authorization page could not be opened.",
                    ) from exc
                raise McpOAuthError(
                    "oauth_failed", "The hosted MCP OAuth flow could not be completed."
                ) from exc
            raise
        result["target"] = {
            "url": target.url,
            "addresses": list(target.addresses),
            "isLoopback": target.is_loopback,
        }
        return result

    async def _connect(
        self,
        session_id: str,
        transport: Literal["stdio", "streamable_http"],
        connector_factory: ConnectorFactory,
        *,
        fingerprint: ExecutableFingerprint | None = None,
    ) -> _Actor:
        if session_id in self._actors:
            raise SessionAlreadyExistsError(f"MCP session already exists: {session_id}")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_Command] = asyncio.Queue()
        ready: asyncio.Future[dict[str, Any]] = loop.create_future()
        task = loop.create_task(
            _session_actor(connector_factory, queue, ready),
            name=f"bashgym-mcp-{session_id}",
        )
        actor = _Actor(
            queue=queue,
            ready=ready,
            task=task,
            transport=transport,
            fingerprint=fingerprint,
        )
        self._actors[session_id] = actor
        try:
            await ready
        except BaseException:
            self._actors.pop(session_id, None)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        return actor

    async def _connect_result(self, session_id: str, actor: _Actor) -> dict[str, Any]:
        initialization = await actor.ready
        inventory = await self.refresh(session_id)
        result: dict[str, Any] = {
            "sessionId": session_id,
            "transport": actor.transport,
            "initialization": initialization,
            "inventory": inventory,
        }
        if actor.fingerprint is not None:
            result["executableFingerprint"] = actor.fingerprint.to_dict()
        return result

    def _actor(self, session_id: str) -> _Actor:
        actor = self._actors.get(validate_session_id(session_id))
        if actor is None:
            raise SessionNotFoundError(f"MCP session does not exist: {session_id}")
        if actor.task.done():
            self._actors.pop(session_id, None)
            exception = actor.task.exception() if not actor.task.cancelled() else None
            raise SessionClosedError(f"MCP session is closed: {session_id}") from exception
        return actor

    async def _request(
        self,
        session_id: str,
        operation: Literal["refresh", "call_tool", "close"],
        **parameters: Any,
    ) -> Any:
        actor = self._actor(session_id)
        future = asyncio.get_running_loop().create_future()
        await actor.queue.put(_Command(operation, parameters, future))
        return await future

    async def refresh(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        return await self._request(
            session_id,
            "refresh",
            max_pages=self.max_pages,
            max_items_per_kind=self.max_items_per_kind,
        )

    async def initialization(self, session_id: str) -> dict[str, Any]:
        """Return defensive plain-JSON initialization metadata for a session."""

        actor = self._actor(session_id)
        initialization = await actor.ready
        return json.loads(json.dumps(initialization, ensure_ascii=False))

    async def call_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
        max_result_bytes: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name is required")
        return await self._request(
            session_id,
            "call_tool",
            tool_name=tool_name,
            arguments=dict(arguments or {}),
            timeout_seconds=(
                self.default_call_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
            max_result_bytes=(
                self.default_max_result_bytes if max_result_bytes is None else max_result_bytes
            ),
        )

    async def close(self, session_id: str, *, timeout_seconds: float = 5.0) -> None:
        actor = self._actor(session_id)
        try:
            await asyncio.wait_for(self._request(session_id, "close"), timeout=timeout_seconds)
            await asyncio.wait_for(actor.task, timeout=timeout_seconds)
        except TimeoutError:
            actor.task.cancel()
            await asyncio.gather(actor.task, return_exceptions=True)
            raise SessionClosedError("MCP session did not close before the deadline")
        finally:
            self._actors.pop(session_id, None)

    async def abort(self, session_id: str) -> None:
        """Immediately cancel an actor and tear down its owned transport/process.

        Unlike queued ``close``, abort does not wait behind an in-flight tool
        call.  Cancelling the actor enters the SDK context managers' cleanup in
        the same task that opened them, which closes HTTP streams or terminates
        the stdio process tree.
        """

        actor = self._actor(session_id)
        self._actors.pop(session_id, None)
        actor.task.cancel()
        await asyncio.gather(actor.task, return_exceptions=True)

    async def aclose(self) -> None:
        for session_id in list(self._actors):
            try:
                await self.close(session_id)
            except (SessionClosedError, SessionNotFoundError):
                self._actors.pop(session_id, None)

    async def __aenter__(self) -> McpClientRuntime:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()
