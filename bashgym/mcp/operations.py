"""Pure lifecycle rules for durable MCP Workbench operations."""

from __future__ import annotations

from typing import Any

from bashgym.mcp.contracts import McpOperation, OperationState, utc_now


class InvalidOperationTransitionError(ValueError):
    """Raised when an operation attempts a forbidden state transition."""


TERMINAL_OPERATION_STATES = frozenset(
    {
        OperationState.COMPLETED,
        OperationState.FAILED,
        OperationState.CANCELLED,
        OperationState.CANCELLED_UPSTREAM_UNKNOWN,
        OperationState.INTERRUPTED,
    }
)
UNSET_OPERATION_RESULT = object()

_ALLOWED_TRANSITIONS: dict[OperationState, frozenset[OperationState]] = {
    OperationState.QUEUED: frozenset(
        {
            OperationState.RUNNING,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.INTERRUPTED,
        }
    ),
    OperationState.RUNNING: frozenset(
        {
            OperationState.WAITING_FOR_APPROVAL,
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.CANCELLED_UPSTREAM_UNKNOWN,
            OperationState.INTERRUPTED,
        }
    ),
    OperationState.WAITING_FOR_APPROVAL: frozenset(
        {
            OperationState.RUNNING,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.CANCELLED_UPSTREAM_UNKNOWN,
            OperationState.INTERRUPTED,
        }
    ),
    **{state: frozenset() for state in TERMINAL_OPERATION_STATES},
}


def is_terminal_operation_state(state: OperationState) -> bool:
    """Return whether a state is terminal."""

    return state in TERMINAL_OPERATION_STATES


def ensure_operation_transition(current: OperationState, target: OperationState) -> None:
    """Validate one operation state transition.

    Repeating a state is allowed so cancellation and terminal persistence can
    be idempotent.
    """

    if current == target:
        return
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidOperationTransitionError(
            f"Cannot transition operation from {current} to {target}"
        )


def transitioned_operation(
    operation: McpOperation,
    target: OperationState,
    *,
    error_code: str | None = None,
    safe_message: str | None = None,
    result: dict[str, Any] | None | object = UNSET_OPERATION_RESULT,
) -> McpOperation:
    """Return a validated next revision of an operation."""

    ensure_operation_transition(operation.state, target)
    now = utc_now()
    started_at = operation.started_at
    if target == OperationState.RUNNING and started_at is None:
        started_at = now
    completed_at = operation.completed_at
    if is_terminal_operation_state(target) and completed_at is None:
        completed_at = now
    next_result = operation.result if result is UNSET_OPERATION_RESULT else result
    return McpOperation.model_validate(
        operation.model_copy(
            update={
                "state": target,
                "revision": operation.revision + (target != operation.state),
                "updated_at": now,
                "started_at": started_at,
                "completed_at": completed_at,
                "error_code": error_code,
                "safe_message": safe_message,
                "result": next_result,
            }
        ).model_dump()
    )
