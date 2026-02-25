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
from .subagent import SubagentCollector
from .edit import EditCollector
from .plan import PlanCollector
from .todo import TodoCollector
from .prompt import PromptCollector
from .environment import EnvironmentCollector
from .debug import DebugCollector
from .scanner import ClaudeDataScanner, ALL_SOURCES
from .index import build_cross_reference_index

__all__ = [
    "BaseCollector",
    "CollectorRecord",
    "SubagentCollector",
    "EditCollector",
    "PlanCollector",
    "TodoCollector",
    "PromptCollector",
    "EnvironmentCollector",
    "DebugCollector",
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
    "ClaudeDataScanner",
    "ALL_SOURCES",
    "build_cross_reference_index",
]
