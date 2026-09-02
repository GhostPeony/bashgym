"""Map proven exit codes to a failure class; unknown causes count as execution."""

from __future__ import annotations

from bashgym.campaigns.contracts import FailureClass

_CONFIGURATION_EXIT_CODES = frozenset({126, 127})
_INFRASTRUCTURE_EXIT_CODES = frozenset({137, 143})
_PERMISSION_EXIT_CODES = frozenset({77})
NON_SCIENTIFIC_FAILURE_CLASSES = frozenset(
    {FailureClass.INFRASTRUCTURE, FailureClass.PERMISSION, FailureClass.CONFIGURATION}
)


def classify_exit_code(exit_code: int | None) -> FailureClass:
    """126/127 command problems, 137/143 kills, 77 permission; everything else executed."""

    if exit_code in _CONFIGURATION_EXIT_CODES:
        return FailureClass.CONFIGURATION
    if exit_code in _INFRASTRUCTURE_EXIT_CODES:
        return FailureClass.INFRASTRUCTURE
    if exit_code in _PERMISSION_EXIT_CODES:
        return FailureClass.PERMISSION
    return FailureClass.EXECUTION


__all__ = ["NON_SCIENTIFIC_FAILURE_CLASSES", "classify_exit_code"]
