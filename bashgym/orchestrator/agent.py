"""
Orchestration Agent

The supervisor that decomposes specs, coordinates workers,
and manages the full lifecycle of an orchestrated development session.

Supports multiple LLM providers for spec decomposition:
- Anthropic Claude (Opus recommended for planning)
- OpenAI (GPT-4o, o1, etc.)
- Google Gemini (2.5 Pro, etc.)
- Ollama (any local model)

Workers always use Claude Code CLI regardless of the planning provider.

Phases:
1. PLAN - Decompose spec into TaskDAG via configured LLM provider
2. DISPATCH - Spawn workers for ready tasks in git worktrees
3. MONITOR - Track progress, handle failures, retry with modified prompts
4. SYNTHESIZE - Merge worktrees, verify results, feed traces to training

Module: Orchestrator
"""

import logging
from pathlib import Path
from typing import Optional, List, Callable, Awaitable

from bashgym.orchestrator.models import (
    OrchestratorSpec, TaskNode, TaskStatus,
    WorkerConfig, WorkerResult,
    LLMConfig, LLMProvider,
)
from bashgym.orchestrator.task_dag import TaskDAG
from bashgym.orchestrator.dispatcher import WorkerPool
from bashgym.orchestrator.worktree import WorktreeManager
from bashgym.orchestrator.prompts import WORKER_SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """The supervisor that decomposes specs and coordinates workers.

    Usage:
        # Anthropic Claude (default)
        agent = OrchestrationAgent()

        # OpenAI
        agent = OrchestrationAgent(
            llm_config=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        )

        # Local Ollama
        agent = OrchestrationAgent(
            llm_config=LLMConfig(provider=LLMProvider.OLLAMA, model="qwen2.5-coder:32b")
        )

        dag = await agent.submit_spec(spec)
        results = await agent.execute(dag)  # Workers always use Claude Code CLI
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        max_workers: int = 5,
        repo_path: Optional[Path] = None,
        use_worktrees: bool = True,
        on_task_started: Optional[Callable[[TaskNode], Awaitable[None]]] = None,
        on_task_completed: Optional[Callable[[TaskNode, WorkerResult], Awaitable[None]]] = None,
        on_task_failed: Optional[Callable[[TaskNode, WorkerResult], Awaitable[None]]] = None,
    ):
        """Initialize the orchestration agent.

        The LLM config controls which provider decomposes specs into task DAGs.
        Workers always use Claude Code CLI regardless of this setting.

        Supported providers:
        - anthropic: Claude models (Opus recommended for planning)
        - openai: GPT-4o, o1, etc.
        - gemini: Gemini 2.5 Pro, etc.
        - ollama: Any local model (qwen2.5-coder, llama3, etc.)

        Args:
            llm_config: LLM provider config for spec decomposition.
                        Defaults to Anthropic Claude Opus.
            max_workers: Maximum parallel workers
            repo_path: Repository path for worktree management
            use_worktrees: Whether to use git worktrees for isolation
            on_task_started: Callback when a task starts
            on_task_completed: Callback when a task completes
            on_task_failed: Callback when a task fails
        """
        self.llm_config = llm_config or LLMConfig()
        self.pool = WorkerPool(max_workers=max_workers)
        self.use_worktrees = use_worktrees

        if repo_path and use_worktrees:
            self.worktrees = WorktreeManager(repo_path)
        else:
            self.worktrees = None

        self.dag: Optional[TaskDAG] = None

        # Callbacks
        self._on_task_started = on_task_started
        self._on_task_completed = on_task_completed
        self._on_task_failed = on_task_failed

    # =========================================================================
    # Phase 1: PLAN
    # =========================================================================

    async def submit_spec(self, spec: OrchestratorSpec) -> TaskDAG:
        """Phase 1: Decompose spec into TaskDAG using configured LLM.

        Returns the DAG for user approval before execution.
        The LLM provider is determined by self.llm_config (Claude, OpenAI,
        Gemini, or Ollama).

        Args:
            spec: User-submitted development specification

        Returns:
            TaskDAG with decomposed tasks for approval
        """
        provider_name = self.llm_config.provider.value
        model_name = self.llm_config.model
        logger.info(
            f"Decomposing spec: {spec.title} "
            f"(provider={provider_name}, model={model_name})"
        )
        self.dag = await TaskDAG.from_spec(spec, self.llm_config)

        task_count = len(self.dag.nodes)
        conflicts = self.dag.detect_file_conflicts()

        logger.info(
            f"Decomposed into {task_count} tasks, "
            f"{len(conflicts)} potential file conflicts"
        )

        return self.dag

    # =========================================================================
    # Phase 2-4: DISPATCH, MONITOR, SYNTHESIZE
    # =========================================================================

    async def execute(
        self,
        dag: Optional[TaskDAG] = None,
        base_branch: str = "main",
    ) -> List[WorkerResult]:
        """Execute an approved TaskDAG.

        1. Get ready tasks (no unmet dependencies)
        2. Create worktrees for each
        3. Spawn workers up to max_workers
        4. As workers complete, mark done and spawn newly unblocked tasks
        5. On failure, retry with modified prompt (up to max_retries)
        6. Collect all results
        7. Merge worktrees
        8. Feed traces to training pipeline

        Args:
            dag: TaskDAG to execute (uses self.dag if None)
            base_branch: Git branch to base worktrees on

        Returns:
            List of all WorkerResults
        """
        dag = dag or self.dag
        if not dag:
            raise ValueError("No DAG to execute. Call submit_spec() first.")

        results: List[WorkerResult] = []
        total_tasks = len(dag.nodes)

        logger.info(f"Starting execution of {total_tasks} tasks")

        while not dag.is_complete():
            # Get tasks ready to run
            ready = dag.get_ready_tasks()

            if not ready and self.pool.active_count == 0:
                # No ready tasks and no active workers = deadlock
                blocked = [
                    t for t in dag.nodes.values()
                    if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
                ]
                if blocked:
                    logger.error(
                        f"Deadlock detected: {len(blocked)} tasks stuck. "
                        f"Marking as failed."
                    )
                    for task in blocked:
                        task.status = TaskStatus.FAILED
                break

            # Spawn workers for ready tasks (up to available slots)
            for task in ready[:self.pool.available_slots]:
                await self._spawn_task_worker(task, dag, base_branch)

            if self.pool.active_count == 0:
                continue

            # Wait for any worker to finish
            try:
                result = await self.pool.wait_for_any(timeout=600)
            except Exception as e:
                logger.error(f"Error waiting for workers: {e}")
                break

            results.append(result)

            if result.success:
                newly_ready = dag.mark_completed(result.task_id, result)
                logger.info(
                    f"Task {result.task_id} completed. "
                    f"{len(newly_ready)} tasks unblocked."
                )
                if self._on_task_completed:
                    task_node = dag.nodes[result.task_id]
                    await self._on_task_completed(task_node, result)
            else:
                await self._handle_failure(dag, result)

        # Merge all worktrees
        merge_results = []
        if self.worktrees:
            for task_id in dag.completed_tasks():
                merge_result = await self.worktrees.merge(task_id)
                merge_results.append(merge_result)
                if not merge_result.success:
                    logger.warning(
                        f"Merge failed for task {task_id}: "
                        f"{merge_result.conflicts}"
                    )

            await self.worktrees.cleanup_all()

        # Summary
        completed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_cost = sum(r.cost_usd for r in results)
        total_time = sum(r.duration_seconds for r in results)

        logger.info(
            f"Execution complete: {completed}/{total_tasks} tasks succeeded, "
            f"{failed} failed. Cost: ${total_cost:.2f}, Time: {total_time:.0f}s"
        )

        return results

    async def _spawn_task_worker(
        self,
        task: TaskNode,
        dag: TaskDAG,
        base_branch: str,
    ) -> None:
        """Create worktree and spawn worker for a task."""
        # Create worktree if enabled
        if self.worktrees:
            try:
                task.worktree_path = await self.worktrees.create(
                    task.id,
                    f"task/{task.id}",
                    base_branch,
                )
            except Exception as e:
                logger.error(f"Failed to create worktree for {task.id}: {e}")
                task.status = TaskStatus.FAILED
                return

        # Build worker config
        config = WorkerConfig(
            max_turns=task.estimated_turns,
            max_budget_usd=task.budget_usd,
            system_prompt_append=WORKER_SYSTEM_PROMPT,
            worktree_path=task.worktree_path,
        )

        # Spawn the worker
        try:
            await self.pool.spawn_worker(task, config)
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

            if self._on_task_started:
                await self._on_task_started(task)
        except Exception as e:
            logger.error(f"Failed to spawn worker for {task.id}: {e}")
            task.status = TaskStatus.FAILED

    async def _handle_failure(
        self,
        dag: TaskDAG,
        result: WorkerResult,
    ) -> None:
        """Handle a failed worker: retry or mark as failed."""
        task = dag.nodes[result.task_id]

        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.worker_prompt = RETRY_PROMPT_TEMPLATE.format(
                error=result.error or "Unknown error",
                previous_output=result.output[:1000],
                original_prompt=task.worker_prompt or task.description,
            )
            task.status = TaskStatus.PENDING
            logger.info(
                f"Retrying task {task.id} "
                f"(attempt {task.retry_count}/{task.max_retries})"
            )
        else:
            blocked = dag.mark_failed(result.task_id, result.error or "")
            logger.warning(
                f"Task {task.id} failed after {task.retry_count} retries. "
                f"{len(blocked)} tasks blocked."
            )
            if self._on_task_failed:
                await self._on_task_failed(task, result)

    # =========================================================================
    # Trace Ingestion (for training pipeline)
    # =========================================================================

    async def ingest_traces(self, results: List[WorkerResult]) -> int:
        """Feed orchestration traces into the Factory pipeline.

        Multi-agent traces are high-value training signal:
        - Task decomposition patterns
        - Tool-use sequences
        - Error recovery strategies

        Args:
            results: Worker results to ingest

        Returns:
            Number of traces ingested
        """
        count = 0
        for result in results:
            if result.session_id and result.success:
                # The session trace lives in ~/.claude/projects/
                # The trace import pipeline picks these up automatically
                count += 1
                logger.debug(
                    f"Trace available for task {result.task_id}: "
                    f"session {result.session_id}"
                )

        logger.info(f"Marked {count} traces for training pipeline ingestion")
        return count
