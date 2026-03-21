"""
Bash Gym - A Self-Improving Agentic Development Gym

Train smaller language models from agent execution traces.
"""

__version__ = "0.1.0"

# Core imports for convenience
# Layer imports
from bashgym.arena import AgentRunner, SandboxManager
from bashgym.config import Settings, get_settings
from bashgym.factory import DataFactory, TraceProcessor
from bashgym.gym import BashGymEnv, ModelRouter, Trainer
from bashgym.judge import Verifier

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
