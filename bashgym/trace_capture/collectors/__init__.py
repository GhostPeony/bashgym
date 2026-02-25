"""
Collectors Package

Data collectors for extracting structured records from Claude Code's
local storage (subagents, edits, plans, todos, prompts, debug info, environment).

Each collector inherits from BaseCollector and produces typed records.
"""

from .base import (
    BaseCollector,
    CollectorRecord,
    SubagentRecord,
    EditRecord,
    PlanRecord,
    TodoRecord,
    PromptRecord,
    DebugRecord,
    EnvironmentRecord,
    CollectorScanResult,
    CollectorBatchResult,
    get_claude_dir,
    get_collected_dir,
)

__all__ = [
    "BaseCollector",
    "CollectorRecord",
    "SubagentRecord",
    "EditRecord",
    "PlanRecord",
    "TodoRecord",
    "PromptRecord",
    "DebugRecord",
    "EnvironmentRecord",
    "CollectorScanResult",
    "CollectorBatchResult",
    "get_claude_dir",
    "get_collected_dir",
]
