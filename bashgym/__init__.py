"""
Bash Gym - A Self-Improving Agentic Development Gym

Train smaller language models from agent execution traces.
"""

__version__ = "0.1.0"

# Core imports for convenience
from bashgym.config import Settings, get_settings

# Layer imports
from bashgym.arena import SandboxManager, AgentRunner
from bashgym.judge import Verifier
from bashgym.factory import DataFactory, TraceProcessor
from bashgym.gym import Trainer, BashGymEnv, ModelRouter

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
]
