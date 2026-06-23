"""
Bash Gym - A Self-Improving Agentic Development Gym

Train smaller language models from agent execution traces.
"""

import importlib

__version__ = "0.1.0"

# Core imports for convenience
# Layer imports
from bashgym.arena import AgentRunner, SandboxManager
from bashgym.config import Settings, get_settings
from bashgym.factory import DataFactory, TraceProcessor
from bashgym.gym import BashGymEnv, ModelRouter, Trainer
from bashgym.judge import Verifier


def __getattr__(name: str):
    """Lazily expose heavier integration subpackages for dotted-path patching."""

    if name == "integrations":
        module = importlib.import_module("bashgym.integrations")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Arena
    "SandboxManager",
    "AgentRunner",
    # Judge
    "Verifier",
    # Factory
    "DataFactory",
    "TraceProcessor",
    # Gym
    "Trainer",
    "BashGymEnv",
    "ModelRouter",
    "integrations",
]
