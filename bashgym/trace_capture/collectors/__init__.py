"""
Collectors Package

Data collectors for extracting structured records from Claude Code's
local storage (subagents, edits, plans, todos, prompts, debug info, environment).

Each collector inherits from BaseCollector and produces typed records.
"""

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorRecord,
    CollectorScanResult,
    DebugRecord,
    EditRecord,
    EnvironmentRecord,
    PlanRecord,
    PromptRecord,
    SubagentRecord,
    TodoRecord,
    get_claude_dir,
    get_collected_dir,
)
from .debug import DebugCollector
from .edit import EditCollector
from .environment import EnvironmentCollector
from .index import build_cross_reference_index
from .plan import PlanCollector
from .prompt import PromptCollector
from .scanner import ALL_SOURCES, ClaudeDataScanner
from .subagent import SubagentCollector
from .todo import TodoCollector

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
