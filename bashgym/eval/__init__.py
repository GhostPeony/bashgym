"""Held-out trace evaluation: structured tool-call metrics + session-clustered
paired-bootstrap statistics for answering "is this fine-tune actually better?".
"""

from .gate import GateThresholds, GateVerdict, evaluate_gate
from .metrics import score_tool_call, tool_arg_f1, tool_name_match
from .split import HoldoutSplit, contamination, example_hash, make_holdout_split
from .stats import BootstrapResult, paired_bootstrap

__all__ = [
    "tool_name_match",
    "tool_arg_f1",
    "score_tool_call",
    "paired_bootstrap",
    "BootstrapResult",
    "make_holdout_split",
    "HoldoutSplit",
    "contamination",
    "example_hash",
    "evaluate_gate",
    "GateThresholds",
    "GateVerdict",
]
