"""
Orchestrator Module

Multi-agent orchestration system that decomposes development specs
into parallel Claude Code worker sessions.

Supports multiple LLM providers for spec decomposition:
- Anthropic Claude (default, Opus recommended)
- OpenAI (GPT-4o, o1)
- Google Gemini (2.5 Pro)
- Ollama (any local model)

Workers always execute via Claude Code CLI regardless of planning provider.
"""

from bashgym.orchestrator.agent import OrchestrationAgent
from bashgym.orchestrator.dispatcher import WorkerPool
from bashgym.orchestrator.models import (
    LLMConfig,
    LLMProvider,
    MergeResult,
    OrchestratorSpec,
    TaskNode,
    TaskPriority,
    TaskStatus,
    WorkerConfig,
    WorkerResult,
)
from bashgym.orchestrator.synthesizer import ResultSynthesizer, SynthesisReport
from bashgym.orchestrator.task_dag import CyclicDependencyError, TaskDAG
from bashgym.orchestrator.worktree import WorktreeManager

__all__ = [
    # Provider config
    "LLMProvider",
    "LLMConfig",
    # Data models
    "TaskStatus",
    "TaskPriority",
    "OrchestratorSpec",
    "TaskNode",
    "WorkerConfig",
    "WorkerResult",
    "MergeResult",
    # Core classes
    "TaskDAG",
    "CyclicDependencyError",
    "WorkerPool",
    "WorktreeManager",
    "OrchestrationAgent",
    "ResultSynthesizer",
    "SynthesisReport",
]
