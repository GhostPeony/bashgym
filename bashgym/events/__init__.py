"""
Bash Gym Event System.

Provides a typed EventBus and domain event dataclasses. All internal
components emit events through the singleton `event_bus`; the WebSocket
layer bridges them onto the wire for frontend consumption.

Usage:
    from bashgym.events import EventBus, event_bus
    from bashgym.events.types import TrainingStarted, TraceClassified

    # Subscribe to specific events
    event_bus.on(TrainingStarted, my_handler)

    # Emit an event
    event_bus.emit(TrainingStarted(run_id="abc", strategy="sft"))
"""

from bashgym.events.bus import EventBus, event_bus

__all__ = [
    "EventBus",
    "event_bus",
]
