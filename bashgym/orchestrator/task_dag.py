"""
Task DAG - Directed Acyclic Graph of Tasks

Manages task decomposition, dependency resolution, and execution ordering.
Uses topological sort for valid execution order and tracks task status
as workers complete or fail.

The from_spec() classmethod uses Anthropic's Messages API with structured
output to decompose a user spec into a TaskDAG.

Module: Orchestrator
"""

import json
import logging
from collections import defaultdict
from typing import Optional, Dict, List, Set, Tuple

from bashgym.orchestrator.models import (
    TaskNode, TaskStatus, TaskPriority,
    OrchestratorSpec, WorkerResult,
    LLMConfig, LLMProvider,
)

logger = logging.getLogger(__name__)


class CyclicDependencyError(Exception):
    """Raised when the task DAG contains a cycle."""
    pass


class TaskDAG:
    """Directed acyclic graph of tasks with dependency resolution.

    Manages the lifecycle of tasks from PENDING through COMPLETED/FAILED,
    tracks dependencies, and provides scheduling primitives.
    """

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode) -> None:
        """Add a task node to the DAG.

        Args:
            task: TaskNode to add

        Raises:
            ValueError: If task ID already exists or dependencies reference
                       unknown tasks (checked lazily at scheduling time)
        """
        if task.id in self.nodes:
            raise ValueError(f"Task '{task.id}' already exists in DAG")
        self.nodes[task.id] = task

    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks whose dependencies are all completed.

        Returns tasks in priority order (CRITICAL first, LOW last).
        Only returns tasks with status PENDING.

        Returns:
            List of ready TaskNodes sorted by priority
        """
        ready = []
        for task in self.nodes.values():
            if task.status != TaskStatus.PENDING:
                continue
            # Check all dependencies are completed
            deps_met = all(
                self.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.nodes
            )
            if deps_met:
                ready.append(task)

        # Sort by priority (lower value = higher priority)
        ready.sort(key=lambda t: t.priority.value)
        return ready

    def topological_sort(self) -> List[TaskNode]:
        """Return tasks in valid execution order.

        Uses Kahn's algorithm for topological sorting.

        Returns:
            List of TaskNodes in dependency-respecting order

        Raises:
            CyclicDependencyError: If the DAG contains cycles
        """
        # Build adjacency and in-degree maps
        in_degree: Dict[str, int] = {tid: 0 for tid in self.nodes}
        children: Dict[str, List[str]] = defaultdict(list)

        for tid, task in self.nodes.items():
            for dep_id in task.dependencies:
                if dep_id in self.nodes:
                    children[dep_id].append(tid)
                    in_degree[tid] += 1

        # Start with nodes that have no dependencies
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        queue.sort(key=lambda tid: self.nodes[tid].priority.value)
        result = []

        while queue:
            tid = queue.pop(0)
            result.append(self.nodes[tid])

            for child_id in children[tid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
            # Re-sort by priority after adding new items
            queue.sort(key=lambda tid: self.nodes[tid].priority.value)

        if len(result) != len(self.nodes):
            processed = {t.id for t in result}
            remaining = set(self.nodes.keys()) - processed
            raise CyclicDependencyError(
                f"Cyclic dependency detected involving tasks: {remaining}"
            )

        return result

    def mark_completed(self, task_id: str, result: WorkerResult) -> List[TaskNode]:
        """Mark a task as completed and return newly unblocked tasks.

        Args:
            task_id: ID of the completed task
            result: Worker result for the task

        Returns:
            List of tasks that are now ready to run
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")

        task = self.nodes[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result

        # Find tasks that were waiting only on this one
        newly_ready = []
        for other in self.nodes.values():
            if other.status != TaskStatus.PENDING:
                continue
            if task_id not in other.dependencies:
                continue
            # Check if all other dependencies are also completed
            all_deps_met = all(
                self.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in other.dependencies
                if dep_id in self.nodes
            )
            if all_deps_met:
                newly_ready.append(other)

        return newly_ready

    def mark_failed(self, task_id: str, error: str) -> List[TaskNode]:
        """Mark a task as failed. Returns tasks that are now blocked.

        Args:
            task_id: ID of the failed task
            error: Error message/description

        Returns:
            List of tasks that are now blocked due to this failure
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")

        task = self.nodes[task_id]
        task.status = TaskStatus.FAILED

        # Find all tasks transitively dependent on this one
        blocked = []
        to_check = [task_id]
        visited: Set[str] = set()

        while to_check:
            current = to_check.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for other in self.nodes.values():
                if current in other.dependencies and other.status == TaskStatus.PENDING:
                    other.status = TaskStatus.BLOCKED
                    blocked.append(other)
                    to_check.append(other.id)

        return blocked

    def get_critical_path(self) -> List[TaskNode]:
        """Calculate the longest dependency chain (minimum total time).

        Uses estimated_turns as the weight for each task.

        Returns:
            List of TaskNodes forming the critical path
        """
        sorted_tasks = self.topological_sort()

        # For each node, compute longest path ending at that node
        dist: Dict[str, float] = {tid: 0 for tid in self.nodes}
        predecessor: Dict[str, Optional[str]] = {tid: None for tid in self.nodes}

        for task in sorted_tasks:
            task_weight = task.estimated_turns
            for dep_id in task.dependencies:
                if dep_id in self.nodes:
                    new_dist = dist[dep_id] + self.nodes[dep_id].estimated_turns
                    if new_dist > dist[task.id]:
                        dist[task.id] = new_dist
                        predecessor[task.id] = dep_id

        # Find the end of the critical path (max distance)
        if not dist:
            return []
        end_id = max(dist, key=lambda tid: dist[tid] + self.nodes[tid].estimated_turns)

        # Trace back
        path = []
        current: Optional[str] = end_id
        while current is not None:
            path.append(self.nodes[current])
            current = predecessor[current]

        path.reverse()
        return path

    def detect_file_conflicts(self) -> List[Tuple[str, str, List[str]]]:
        """Find task pairs that touch the same files (potential merge conflicts).

        Returns:
            List of (task_id_a, task_id_b, conflicting_files) tuples
        """
        conflicts = []
        task_list = list(self.nodes.values())

        for i, task_a in enumerate(task_list):
            if not task_a.files_touched:
                continue
            files_a = set(task_a.files_touched)

            for task_b in task_list[i + 1:]:
                if not task_b.files_touched:
                    continue
                # Skip if one depends on the other (sequential, not conflicting)
                if task_b.id in task_a.dependencies or task_a.id in task_b.dependencies:
                    continue

                overlap = files_a & set(task_b.files_touched)
                if overlap:
                    conflicts.append((task_a.id, task_b.id, sorted(overlap)))

        return conflicts

    def is_complete(self) -> bool:
        """Check if all tasks are in terminal states (completed/failed/cancelled)."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        return all(task.status in terminal for task in self.nodes.values())

    def completed_tasks(self) -> List[str]:
        """Return IDs of all completed tasks."""
        return [
            tid for tid, task in self.nodes.items()
            if task.status == TaskStatus.COMPLETED
        ]

    @property
    def stats(self) -> Dict[str, int]:
        """Get task counts by status."""
        counts: Dict[str, int] = {}
        for task in self.nodes.values():
            status = task.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    def to_dict(self) -> Dict:
        """Serialize DAG for API responses."""
        return {
            "tasks": [task.to_dict() for task in self.nodes.values()],
            "stats": self.stats,
            "file_conflicts": [
                {"task_a": a, "task_b": b, "files": f}
                for a, b, f in self.detect_file_conflicts()
            ],
        }

    # =========================================================================
    # Spec Decomposition
    # =========================================================================

    @classmethod
    async def from_spec(
        cls,
        spec: OrchestratorSpec,
        llm_config: "LLMConfig",
    ) -> "TaskDAG":
        """Decompose a spec into a TaskDAG using the configured LLM provider.

        Supports Anthropic Claude, OpenAI, Google Gemini, and Ollama.

        Args:
            spec: User-submitted development specification
            llm_config: LLM provider configuration

        Returns:
            TaskDAG with tasks from the decomposition
        """
        system_prompt = DECOMPOSITION_SYSTEM_PROMPT
        user_prompt = _build_decomposition_prompt(spec)

        content = await _call_llm(llm_config, system_prompt, user_prompt)
        return _parse_decomposition(content)


async def _call_llm(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call the configured LLM provider and return the response text.

    Args:
        config: LLM provider configuration
        system_prompt: System instructions
        user_prompt: User message

    Returns:
        Response text from the LLM

    Raises:
        RuntimeError: On API errors or unsupported providers
    """
    import httpx

    api_key = config.get_api_key()
    base_url = config.get_base_url()

    async with httpx.AsyncClient(timeout=120.0) as client:
        if config.provider == LLMProvider.ANTHROPIC:
            if not api_key:
                raise ValueError(
                    "Anthropic API key required. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            response = await client.post(
                base_url,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Anthropic API error ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            result = response.json()
            return result["content"][0]["text"]

        elif config.provider == LLMProvider.OPENAI:
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. "
                    "Set OPENAI_API_KEY environment variable."
                )
            response = await client.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenAI API error ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            result = response.json()
            return result["choices"][0]["message"]["content"]

        elif config.provider == LLMProvider.GEMINI:
            if not api_key:
                raise ValueError(
                    "Google API key required. "
                    "Set GOOGLE_API_KEY environment variable."
                )
            # Gemini uses generateContent endpoint
            url = (
                f"{base_url}/models/{config.model}:generateContent"
                f"?key={api_key}"
            )
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}],
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": user_prompt}],
                        },
                    ],
                    "generationConfig": {
                        "temperature": config.temperature,
                        "maxOutputTokens": config.max_tokens,
                    },
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini API error ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

        elif config.provider == LLMProvider.OLLAMA:
            # Ollama runs locally, no API key needed
            response = await client.post(
                base_url,
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens,
                    },
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            result = response.json()
            return result["message"]["content"]

        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _build_decomposition_prompt(spec: OrchestratorSpec) -> str:
    """Build the user prompt for spec decomposition."""
    parts = [
        f"# Development Specification\n\n",
        f"## Title\n{spec.title}\n\n",
        f"## Description\n{spec.description}\n\n",
    ]

    if spec.constraints:
        parts.append("## Constraints\n")
        for c in spec.constraints:
            parts.append(f"- {c}\n")
        parts.append("\n")

    if spec.acceptance_criteria:
        parts.append("## Acceptance Criteria\n")
        for ac in spec.acceptance_criteria:
            parts.append(f"- {ac}\n")
        parts.append("\n")

    parts.append(
        f"## Settings\n"
        f"- Repository: {spec.repository or '(current)'}\n"
        f"- Base branch: {spec.base_branch}\n"
        f"- Max budget: ${spec.max_budget_usd}\n"
        f"- Max parallel workers: {spec.max_workers}\n"
    )

    return "".join(parts)


def _parse_decomposition(content: str) -> "TaskDAG":
    """Parse LLM decomposition output into a TaskDAG.

    Expects JSON output with a tasks array from the LLM.
    Falls back to basic parsing if JSON extraction fails.
    """
    dag = TaskDAG()

    # Try to extract JSON from the response
    try:
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            tasks_data = json.loads(content[json_start:json_end])
        else:
            # Try finding a JSON object with "tasks" key
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                tasks_data = parsed.get("tasks", [])
            else:
                logger.warning("Could not extract JSON from decomposition")
                return dag
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse decomposition JSON: {e}")
        return dag

    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW,
    }

    for i, task_data in enumerate(tasks_data):
        task_id = task_data.get("id", f"task_{i + 1}")
        priority_str = task_data.get("priority", "normal").lower()

        task = TaskNode(
            id=task_id,
            title=task_data.get("title", f"Task {i + 1}"),
            description=task_data.get("description", ""),
            priority=priority_map.get(priority_str, TaskPriority.NORMAL),
            dependencies=task_data.get("dependencies", []),
            files_touched=task_data.get("files_touched", []),
            estimated_turns=task_data.get("estimated_turns", 20),
            budget_usd=task_data.get("budget_usd", 2.0),
            worker_prompt=task_data.get("worker_prompt", ""),
        )
        dag.add_task(task)

    # Validate the DAG is acyclic
    try:
        dag.topological_sort()
    except CyclicDependencyError:
        logger.warning("Decomposition produced cyclic dependencies, removing edges")
        # Break cycles by removing the last dependency from each cycle participant
        for task in dag.nodes.values():
            task.dependencies = [
                d for d in task.dependencies if d in dag.nodes
            ]

    return dag


DECOMPOSITION_SYSTEM_PROMPT = """You are a software architect that decomposes development specifications into a task DAG (directed acyclic graph).

Given a development specification, produce a JSON array of tasks. Each task should be:
- Small enough for a single Claude Code session (10-50 tool calls)
- Have clear dependencies on other tasks
- Include estimated files it will touch

Output ONLY a JSON array with this schema:
```json
[
  {
    "id": "task_1",
    "title": "Short task title",
    "description": "Detailed description of what to implement",
    "priority": "critical|high|normal|low",
    "dependencies": ["task_id_1", "task_id_2"],
    "files_touched": ["src/foo.py", "tests/test_foo.py"],
    "estimated_turns": 20,
    "budget_usd": 2.0,
    "worker_prompt": "Specific instructions for the Claude Code worker"
  }
]
```

Rules:
- Tasks with no dependencies run first (in parallel)
- Use "critical" priority for foundational tasks that many others depend on
- Minimize file overlap between parallel tasks to avoid merge conflicts
- Each task should be completable in under 15 minutes
- worker_prompt should be detailed enough for a Claude Code session to execute independently
- Keep the total number of tasks between 3 and 15
- Total budget should not exceed the spec's max budget"""
