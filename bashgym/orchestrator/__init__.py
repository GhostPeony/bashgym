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

from bashgym.orchestrator.models import (
    LLMProvider,
    LLMConfig,
    TaskStatus,
    TaskPriority,
    OrchestratorSpec,
    TaskNode,
    WorkerConfig,
    WorkerResult,
    MergeResult,
)
from bashgym.orchestrator.task_dag import TaskDAG, CyclicDependencyError
from bashgym.orchestrator.dispatcher import WorkerPool
from bashgym.orchestrator.worktree import WorktreeManager
from bashgym.orchestrator.agent import OrchestrationAgent
from bashgym.orchestrator.synthesizer import ResultSynthesizer, SynthesisReport

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
