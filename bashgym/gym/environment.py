"""
Gym Environment for The Gym Layer

Custom RL environment for training agentic models using
NVIDIA NeMo Gym. Implements the environment interface for
GRPO and RLVR training strategies.

Instrumentation:
  - Guardrails check commands before execution
  - Profiler tracks episode traces and step spans
  - Blocked commands result in negative rewards

Module 4: Training (The "Gym")
"""

import os
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
import random

# Import instrumentation (optional)
try:
    from bashgym.core import get_instrumentation, Instrumentation
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    Instrumentation = None

if TYPE_CHECKING:
    from bashgym.core import Instrumentation


class ActionType(Enum):
    """Types of actions the agent can take."""
    BASH = "bash"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    SUBMIT = "submit"  # Submit solution for verification


@dataclass
class Action:
    """An action taken by the agent."""

    action_type: ActionType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class Observation:
    """An observation from the environment."""

    content: str
    success: bool
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "success": self.success,
            "done": self.done,
            "info": self.info
        }


@dataclass
class GymEnvConfig:
    """Configuration for the Gym environment."""

    # Task settings
    max_steps: int = 50
    timeout_per_step: int = 60

    # Reward settings
    step_penalty: float = -0.01
    success_reward: float = 1.0
    failure_penalty: float = -0.5
    partial_reward_enabled: bool = True
    blocked_action_penalty: float = -0.2  # Penalty for guardrail-blocked actions

    # Verification settings
    verify_on_submit: bool = True
    auto_submit_on_max_steps: bool = True

    # Sandbox settings
    use_sandbox: bool = True
    sandbox_image: str = "python:3.10-slim"

    # Logging
    log_trajectory: bool = True
    trajectory_dir: str = "data/trajectories"

    # Instrumentation
    enable_guardrails: bool = True
    enable_profiling: bool = True


class BashGymEnv:
    """
    Custom RL Environment for training agentic models.

    Implements a Gymnasium-compatible interface for:
    - Task execution in sandboxed environments
    - Reward computation based on verification
    - Trajectory logging for training

    Compatible with NVIDIA NeMo Gym and standard RL libraries.
    """

    def __init__(
        self,
        config: Optional[GymEnvConfig] = None,
        sandbox_manager=None,
        verifier=None,
        instrumentation: Optional["Instrumentation"] = None
    ):
        """Initialize the gym environment."""
        self.config = config or GymEnvConfig()
        self.sandbox_manager = sandbox_manager
        self.verifier = verifier

        # Instrumentation (guardrails + profiling)
        self._instrumentation = instrumentation
        if self._instrumentation is None and HAS_INSTRUMENTATION:
            if self.config.enable_guardrails or self.config.enable_profiling:
                self._instrumentation = get_instrumentation()

        # Episode state
        self.current_task: Optional[str] = None
        self.current_sandbox_id: Optional[str] = None
        self.step_count: int = 0
        self.trajectory: List[Dict[str, Any]] = []
        self.episode_reward: float = 0.0
        self.done: bool = False
        self._current_trace_id: Optional[str] = None

        # Ensure trajectory directory exists
        Path(self.config.trajectory_dir).mkdir(parents=True, exist_ok=True)

    @property
    def instrumentation(self) -> Optional["Instrumentation"]:
        """Get the instrumentation instance."""
        return self._instrumentation

    def get_guardrail_events(self) -> List[Dict[str, Any]]:
        """Get guardrail events from this environment's session."""
        if self._instrumentation:
            return [e.to_dict() for e in self._instrumentation.get_guardrail_events()]
        return []

    def reset(
        self,
        task: str,
        task_id: Optional[str] = None,
        initial_files: Optional[Dict[str, str]] = None
    ) -> Observation:
        """
        Reset the environment with a new task.

        Args:
            task: The task description/prompt
            task_id: Optional unique task identifier
            initial_files: Optional initial files for the workspace

        Returns:
            Initial observation
        """
        # End previous trace if exists
        if self._current_trace_id and self._instrumentation:
            self._instrumentation.end_trace(self._current_trace_id)

        # Clean up previous episode
        if self.current_sandbox_id and self.sandbox_manager:
            self.sandbox_manager.cleanup_sandbox(self.current_sandbox_id)

        # Reset state
        self.current_task = task
        self.step_count = 0
        self.trajectory = []
        self.episode_reward = 0.0
        self.done = False

        # Start new profiler trace for this episode
        if self._instrumentation and self.config.enable_profiling:
            self._current_trace_id = self._instrumentation.start_trace(
                f"episode:{task_id or 'unnamed'}",
                metadata={"task": task[:200], "max_steps": self.config.max_steps}
            )

        # Create new sandbox if enabled
        if self.config.use_sandbox and self.sandbox_manager:
            sandbox = self.sandbox_manager.create_sandbox(
                task_id=task_id,
                initial_files=initial_files
            )
            self.current_sandbox_id = sandbox.sandbox_id
            self.sandbox_manager.start_sandbox(sandbox.sandbox_id)

        # Create initial observation
        initial_obs = Observation(
            content=f"Task: {task}\n\nYou are in a fresh workspace. Begin by exploring and planning your approach.",
            success=True,
            done=False,
            info={"step": 0, "task": task}
        )

        # Log initial state
        self._log_step(None, initial_obs, 0.0)

        return initial_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action and return the result.

        Args:
            action: The action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1

        # Check if already done
        if self.done:
            return self._terminal_observation(), 0.0, True, {"error": "Episode already done"}

        # Execute action
        obs, base_reward = self._execute_action(action)

        # Apply step penalty
        reward = base_reward + self.config.step_penalty

        # Check termination conditions
        if action.action_type == ActionType.SUBMIT:
            self.done = True
            reward += self._compute_verification_reward()
        elif self.step_count >= self.config.max_steps:
            self.done = True
            if self.config.auto_submit_on_max_steps:
                reward += self._compute_verification_reward()
            else:
                reward += self.config.failure_penalty

        # Update episode reward
        self.episode_reward += reward

        # Log step
        self._log_step(action, obs, reward)

        # Prepare info dict
        info = {
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "action_type": action.action_type.value
        }

        return obs, reward, self.done, info

    def _execute_action(self, action: Action) -> Tuple[Observation, float]:
        """Execute an action and return observation and immediate reward."""

        if action.action_type == ActionType.BASH:
            return self._execute_bash(action.content)
        elif action.action_type == ActionType.READ:
            return self._execute_read(action.content)
        elif action.action_type == ActionType.WRITE:
            return self._execute_write(action.content, action.metadata)
        elif action.action_type == ActionType.EDIT:
            return self._execute_edit(action.content, action.metadata)
        elif action.action_type == ActionType.SUBMIT:
            return self._execute_submit()
        else:
            return Observation(
                content=f"Unknown action type: {action.action_type}",
                success=False,
                done=False
            ), -0.1

    def _execute_bash(self, command: str) -> Tuple[Observation, float]:
        """Execute a bash command."""
        if not self.sandbox_manager or not self.current_sandbox_id:
            # Simulate execution for testing
            return Observation(
                content=f"[Simulated] Executed: {command}",
                success=True,
                done=False
            ), 0.0

        result = self.sandbox_manager.execute_command(
            self.current_sandbox_id,
            command,
            timeout=self.config.timeout_per_step
        )

        if result.get("blocked"):
            return Observation(
                content="Command blocked: potentially dangerous",
                success=False,
                done=False,
                info={"blocked": True}
            ), -0.2

        success = result["exit_code"] == 0
        content = result["stdout"] if success else result["stderr"]

        return Observation(
            content=content or "(no output)",
            success=success,
            done=False,
            info={"exit_code": result["exit_code"]}
        ), 0.0 if success else -0.05

    def _execute_read(self, file_path: str) -> Tuple[Observation, float]:
        """Read a file."""
        command = f"cat {file_path}"
        return self._execute_bash(command)

    def _execute_write(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Tuple[Observation, float]:
        """Write content to a file."""
        file_path = metadata.get("file_path", "output.txt")
        # Escape content for shell
        escaped_content = content.replace("'", "'\"'\"'")
        command = f"cat > {file_path} << 'OUROBOROS_EOF'\n{content}\nOUROBOROS_EOF"
        return self._execute_bash(command)

    def _execute_edit(
        self,
        edit_spec: str,
        metadata: Dict[str, Any]
    ) -> Tuple[Observation, float]:
        """Edit a file using sed or similar."""
        file_path = metadata.get("file_path", "")
        if not file_path:
            return Observation(
                content="Error: file_path required for edit",
                success=False,
                done=False
            ), -0.1

        # Use sed for simple edits
        command = f"sed -i '{edit_spec}' {file_path}"
        return self._execute_bash(command)

    def _execute_submit(self) -> Tuple[Observation, float]:
        """Submit the solution for verification."""
        return Observation(
            content="Solution submitted for verification.",
            success=True,
            done=True,
            info={"submitted": True}
        ), 0.0

    # =========================================================================
    # Async Methods with Instrumentation
    # =========================================================================

    async def step_async(
        self,
        action: Action,
        model_source: Optional[str] = None
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action with instrumentation (async).

        Args:
            action: The action to execute
            model_source: "student" or "teacher" for guardrail tracking

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1

        # Check if already done
        if self.done:
            return self._terminal_observation(), 0.0, True, {"error": "Episode already done"}

        # Execute action with instrumentation
        obs, base_reward, was_blocked = await self._execute_action_async(action, model_source)

        # Apply step penalty
        reward = base_reward + self.config.step_penalty

        # Check termination conditions
        if action.action_type == ActionType.SUBMIT:
            self.done = True
            reward += self._compute_verification_reward()
        elif self.step_count >= self.config.max_steps:
            self.done = True
            if self.config.auto_submit_on_max_steps:
                reward += self._compute_verification_reward()
            else:
                reward += self.config.failure_penalty

        # Update episode reward
        self.episode_reward += reward

        # Log step
        self._log_step(action, obs, reward)

        # Prepare info dict
        info = {
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "action_type": action.action_type.value,
            "blocked": was_blocked,
            "model_source": model_source
        }

        # End trace if episode is done
        if self.done and self._current_trace_id and self._instrumentation:
            self._instrumentation.end_trace(self._current_trace_id)
            self._current_trace_id = None

        return obs, reward, self.done, info

    async def _execute_action_async(
        self,
        action: Action,
        model_source: Optional[str] = None
    ) -> Tuple[Observation, float, bool]:
        """Execute an action with guardrails and profiling (async)."""
        was_blocked = False

        if action.action_type == ActionType.BASH:
            obs, reward, was_blocked = await self._execute_bash_async(
                action.content, model_source
            )
            return obs, reward, was_blocked

        elif action.action_type == ActionType.READ:
            obs, reward, was_blocked = await self._execute_bash_async(
                f"cat {action.content}", model_source
            )
            return obs, reward, was_blocked

        elif action.action_type == ActionType.WRITE:
            # Check content with guardrails before writing
            content = action.content
            file_path = action.metadata.get("file_path", "output.txt")

            if self._instrumentation and self.config.enable_guardrails:
                async with self._instrumentation.instrument_output(
                    content,
                    location="gym.execute_write",
                    model_source=model_source
                ) as ctx:
                    if not ctx.allowed:
                        return Observation(
                            content="Write blocked: content failed safety check",
                            success=False,
                            done=False,
                            info={"blocked": True}
                        ), self.config.blocked_action_penalty, True
                    content = ctx.content  # May have PII filtered

            command = f"cat > {file_path} << 'OUROBOROS_EOF'\n{content}\nOUROBOROS_EOF"
            obs, reward, was_blocked = await self._execute_bash_async(command, model_source)
            return obs, reward, was_blocked

        elif action.action_type == ActionType.EDIT:
            file_path = action.metadata.get("file_path", "")
            if not file_path:
                return Observation(
                    content="Error: file_path required for edit",
                    success=False,
                    done=False
                ), -0.1, False

            command = f"sed -i '{action.content}' {file_path}"
            obs, reward, was_blocked = await self._execute_bash_async(command, model_source)
            return obs, reward, was_blocked

        elif action.action_type == ActionType.SUBMIT:
            return self._execute_submit()[0], self._execute_submit()[1], False

        else:
            return Observation(
                content=f"Unknown action type: {action.action_type}",
                success=False,
                done=False
            ), -0.1, False

    async def _execute_bash_async(
        self,
        command: str,
        model_source: Optional[str] = None
    ) -> Tuple[Observation, float, bool]:
        """Execute a bash command with guardrails and profiling (async)."""
        was_blocked = False

        # Use instrumentation if available
        if self._instrumentation and self.config.enable_guardrails:
            async with self._instrumentation.instrument_command(
                command,
                location="gym.execute_bash",
                model_source=model_source
            ) as ctx:
                if not ctx.allowed:
                    # Command was blocked by guardrails
                    return Observation(
                        content=f"Command blocked: {ctx.check_result.blocked_reason if ctx.check_result else 'safety violation'}",
                        success=False,
                        done=False,
                        info={"blocked": True, "reason": ctx.check_result.blocked_reason if ctx.check_result else None}
                    ), self.config.blocked_action_penalty, True

                # Command allowed - execute it
                if not self.sandbox_manager or not self.current_sandbox_id:
                    ctx.set_result(success=True, output=f"[Simulated] {command}")
                    return Observation(
                        content=f"[Simulated] Executed: {command}",
                        success=True,
                        done=False
                    ), 0.0, False

                result = self.sandbox_manager.execute_command(
                    self.current_sandbox_id,
                    command,
                    timeout=self.config.timeout_per_step
                )

                if result.get("blocked"):
                    ctx.set_result(success=False, output="Sandbox blocked")
                    return Observation(
                        content="Command blocked by sandbox",
                        success=False,
                        done=False,
                        info={"blocked": True}
                    ), self.config.blocked_action_penalty, True

                success = result["exit_code"] == 0
                content = result["stdout"] if success else result["stderr"]

                ctx.set_result(
                    success=success,
                    output=content[:100] if content else "",
                    exit_code=result["exit_code"]
                )

                return Observation(
                    content=content or "(no output)",
                    success=success,
                    done=False,
                    info={"exit_code": result["exit_code"]}
                ), 0.0 if success else -0.05, False

        else:
            # No instrumentation - fall back to sync version
            obs, reward = self._execute_bash(command)
            return obs, reward, False

    def _compute_verification_reward(self) -> float:
        """Compute reward based on verification results."""
        if not self.verifier or not self.current_sandbox_id:
            # No verifier - return neutral reward
            return 0.0

        # Get workspace path
        if self.sandbox_manager and self.current_sandbox_id:
            sandbox = self.sandbox_manager.active_sandboxes.get(self.current_sandbox_id)
            if sandbox:
                result = self.verifier.verify(
                    workspace_path=sandbox.workspace_path,
                    task_id=self.current_sandbox_id
                )

                if result.success:
                    return self.config.success_reward
                elif self.config.partial_reward_enabled and result.total_tests > 0:
                    # Partial reward based on passed tests
                    pass_rate = result.passed_tests / result.total_tests
                    return pass_rate * self.config.success_reward * 0.5
                else:
                    return self.config.failure_penalty

        return self.config.failure_penalty

    def _terminal_observation(self) -> Observation:
        """Return a terminal observation."""
        return Observation(
            content="Episode has ended.",
            success=True,
            done=True,
            info={"terminal": True}
        )

    def _log_step(
        self,
        action: Optional[Action],
        observation: Observation,
        reward: float
    ) -> None:
        """Log a step to the trajectory."""
        if not self.config.log_trajectory:
            return

        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action.to_dict() if action else None,
            "observation": observation.to_dict(),
            "reward": reward,
            "cumulative_reward": self.episode_reward
        }

        self.trajectory.append(step_data)

    def save_trajectory(self, path: Optional[Path] = None) -> Path:
        """Save the current trajectory to a file."""
        if path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = Path(self.config.trajectory_dir) / f"trajectory_{timestamp}.json"

        trajectory_data = {
            "task": self.current_task,
            "total_steps": self.step_count,
            "total_reward": self.episode_reward,
            "success": self.episode_reward > 0,
            "trajectory": self.trajectory
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        return path

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the current trajectory."""
        return self.trajectory.copy()

    def close(self) -> None:
        """Clean up the environment."""
        if self.current_sandbox_id and self.sandbox_manager:
            self.sandbox_manager.cleanup_sandbox(self.current_sandbox_id)
        self.current_sandbox_id = None


class BatchGymEnv:
    """
    Manages multiple gym environments for parallel training.

    Useful for GRPO where we need to generate multiple
    trajectories for the same task.
    """

    def __init__(
        self,
        num_envs: int = 4,
        config: Optional[GymEnvConfig] = None,
        sandbox_manager=None,
        verifier=None
    ):
        """Initialize batch environment."""
        self.num_envs = num_envs
        self.envs = [
            BashGymEnv(config, sandbox_manager, verifier)
            for _ in range(num_envs)
        ]

    def reset_all(
        self,
        task: str,
        task_id_prefix: Optional[str] = None
    ) -> List[Observation]:
        """Reset all environments with the same task."""
        observations = []
        for i, env in enumerate(self.envs):
            task_id = f"{task_id_prefix}_{i}" if task_id_prefix else None
            obs = env.reset(task, task_id=task_id)
            observations.append(obs)
        return observations

    def step_all(
        self,
        actions: List[Action]
    ) -> List[Tuple[Observation, float, bool, Dict[str, Any]]]:
        """Execute actions in all environments."""
        results = []
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            results.append(result)
        return results

    def get_all_trajectories(self) -> List[List[Dict[str, Any]]]:
        """Get trajectories from all environments."""
        return [env.get_trajectory() for env in self.envs]

    def get_rewards(self) -> List[float]:
        """Get episode rewards from all environments."""
        return [env.episode_reward for env in self.envs]

    def close_all(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


def main():
    """Example usage of the Gym Environment."""
    # Create environment
    config = GymEnvConfig(
        max_steps=10,
        success_reward=1.0,
        use_sandbox=False  # Disable sandbox for demo
    )

    env = BashGymEnv(config)

    # Reset with a task
    obs = env.reset(
        task="Create a Python script that prints 'Hello, World!'",
        task_id="demo_task"
    )
    print(f"Initial observation: {obs.content}")

    # Simulate some actions
    actions = [
        Action(ActionType.BASH, "echo 'print(\"Hello, World!\")' > hello.py"),
        Action(ActionType.BASH, "python hello.py"),
        Action(ActionType.SUBMIT, "")
    ]

    for action in actions:
        obs, reward, done, info = env.step(action)
        print(f"Action: {action.action_type.value}")
        print(f"Observation: {obs.content[:100]}...")
        print(f"Reward: {reward}, Done: {done}")
        print("---")

        if done:
            break

    # Save trajectory
    path = env.save_trajectory()
    print(f"Trajectory saved to: {path}")

    env.close()


if __name__ == "__main__":
    main()
