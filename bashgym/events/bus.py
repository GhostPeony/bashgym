"""
Lightweight typed EventBus for Bash Gym.

Provides synchronous and asynchronous pub/sub with type-based dispatch.
All typed events flow through here; the WebSocket layer registers as a
global handler to bridge events onto the wire.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple typed pub/sub event bus.

    Handlers are registered per event type. Global handlers receive all events
    (useful for the WebSocket bridge that serializes everything to the wire).
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = {}
        self._global_handlers: list[Callable] = []

    # --------------------------------------------------------------------- #
    # Registration
    # --------------------------------------------------------------------- #

    def on(self, event_type: type, handler: Callable) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def off(self, event_type: type, handler: Callable) -> None:
        """Remove a handler for a specific event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def on_all(self, handler: Callable) -> None:
        """Register a global handler that receives every event."""
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    def off_all(self, handler: Callable) -> None:
        """Remove a global handler."""
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            pass

    # --------------------------------------------------------------------- #
    # Dispatch
    # --------------------------------------------------------------------- #

    def emit(self, event: Any) -> None:
        """
        Synchronously dispatch an event to matched handlers.

        Calls type-specific handlers first, then global handlers.
        If a handler is a coroutine function, it is scheduled on the
        running event loop (fire-and-forget). If no loop is running,
        the async handler is silently skipped.
        """
        event_type = type(event)
        handlers = list(self._handlers.get(event_type, []))
        handlers.extend(self._global_handlers)

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    # Schedule async handler on the running loop
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        # No running loop -- skip async handler in sync context
                        logger.debug(
                            "Skipping async handler %s (no running event loop)",
                            handler.__name__,
                        )
                else:
                    handler(event)
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    handler.__name__,
                    event_type.__name__,
                )

    async def emit_async(self, event: Any) -> None:
        """
        Asynchronously dispatch an event to matched handlers.

        Awaits coroutine handlers; calls sync handlers directly.
        """
        event_type = type(event)
        handlers = list(self._handlers.get(event_type, []))
        handlers.extend(self._global_handlers)

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    handler.__name__,
                    event_type.__name__,
                )


# Module-level singleton
event_bus = EventBus()
