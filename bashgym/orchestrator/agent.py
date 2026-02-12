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
from bashgym.orchestrator.synthesizer import ResultSynthesizer, SynthesisReport
from bashgym.orchestrator.prompts import (
    WORKER_SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE,
    RETRY_ANALYSIS_SYSTEM, RETRY_ANALYSIS_TEMPLATE,
)

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
        job_id: Optional[str] = None,
        model_router=None,
        on_task_started: Optional[Callable[[TaskNode], Awaitable[None]]] = None,
        on_task_completed: Optional[Callable[[TaskNode, WorkerResult], Awaitable[None]]] = None,
        on_task_failed: Optional[Callable[[TaskNode, WorkerResult], Awaitable[None]]] = None,
    ):
        """Initialize the orchestration agent.

        The LLM config controls which provider decomposes specs into task DAGs.
        Workers use Claude Code CLI by default, but low-priority/simple tasks
        can be routed to a fine-tuned student model via the model_router.

        Supported providers for planning:
        - anthropic: Claude models (Opus recommended)
        - openai: GPT-4o, o1, etc.
        - gemini: Gemini 2.5 Pro, etc.
        - ollama: Any local model (qwen2.5-coder, llama3, etc.)

        Args:
            llm_config: LLM provider config for spec decomposition.
                        Defaults to Anthropic Claude Opus.
            max_workers: Maximum parallel workers
            repo_path: Repository path for worktree management
            use_worktrees: Whether to use git worktrees for isolation
            job_id: Optional job ID for WebSocket broadcasts
            model_router: Optional ModelRouter for student model routing.
                         When set, LOW priority tasks are routed through
                         the router's confidence-based strategy.
            on_task_started: Callback when a task starts
            on_task_completed: Callback when a task completes
            on_task_failed: Callback when a task fails
        """
        self.llm_config = llm_config or LLMConfig()
        self.pool = WorkerPool(max_workers=max_workers)
        self.use_worktrees = use_worktrees
        self.job_id = job_id or ""
        self.model_router = model_router

        if repo_path and use_worktrees:
            self.worktrees = WorktreeManager(repo_path)
        else:
            self.worktrees = None

        self.dag: Optional[TaskDAG] = None

        # Budget tracking
        self._total_spent_usd: float = 0.0
        self._budget_limit_usd: float = 0.0  # Set from spec during execute()
        self._budget_exceeded: bool = False

        # Routing stats
        self._tasks_routed_student: int = 0
        self._tasks_routed_teacher: int = 0

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
        budget_usd: Optional[float] = None,
    ) -> List[WorkerResult]:
        """Execute an approved TaskDAG.

        1. Get ready tasks (no unmet dependencies)
        2. Create worktrees for each
        3. Spawn workers up to max_workers
        4. As workers complete, mark done and spawn newly unblocked tasks
        5. On failure, retry with LLM-rewritten prompt (up to max_retries)
        6. Track budget — auto-cancel remaining if exceeded
        7. Collect all results
        8. Synthesize: merge worktrees with LLM conflict resolution
        9. Feed traces to training pipeline

        Args:
            dag: TaskDAG to execute (uses self.dag if None)
            base_branch: Git branch to base worktrees on
            budget_usd: Total budget cap (overrides spec's max_budget_usd)

        Returns:
            List of all WorkerResults
        """
        dag = dag or self.dag
        if not dag:
            raise ValueError("No DAG to execute. Call submit_spec() first.")

        results: List[WorkerResult] = []
        total_tasks = len(dag.nodes)

        # Initialize budget tracking
        self._total_spent_usd = 0.0
        self._budget_exceeded = False
        self._budget_limit_usd = budget_usd or 0.0

        logger.info(
            f"Starting execution of {total_tasks} tasks"
            + (f" (budget: ${self._budget_limit_usd:.2f})"
               if self._budget_limit_usd else "")
        )

        while not dag.is_complete():
            # Budget check before spawning more
            if self._budget_exceeded:
                logger.warning("Budget exceeded — cancelling remaining tasks")
                await self._cancel_remaining(dag)
                break

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

            # Update budget tracking
            self._total_spent_usd += result.cost_usd
            await self._broadcast_budget_update(dag, results)

            if self._budget_limit_usd and self._total_spent_usd >= self._budget_limit_usd:
                self._budget_exceeded = True
                logger.warning(
                    f"Budget limit reached: ${self._total_spent_usd:.2f} "
                    f">= ${self._budget_limit_usd:.2f}"
                )

            if result.success:
                newly_ready = dag.mark_completed(result.task_id, result)
                logger.info(
                    f"Task {result.task_id} completed. "
                    f"{len(newly_ready)} tasks unblocked."
                )
                await self._broadcast_task_completed(result, len(newly_ready))
                if self._on_task_completed:
                    task_node = dag.nodes[result.task_id]
                    await self._on_task_completed(task_node, result)
            else:
                await self._handle_failure(dag, result)

        # Synthesize: merge worktrees with conflict resolution
        synthesizer = ResultSynthesizer(
            worktrees=self.worktrees,
            llm_config=self.llm_config,
            auto_resolve_conflicts=True,
        )
        report = await synthesizer.synthesize(dag, results, base_branch)

        # Summary
        completed = report.completed_tasks
        failed = report.failed_tasks

        logger.info(
            f"Execution complete: {completed}/{total_tasks} tasks succeeded, "
            f"{failed} failed. Cost: ${self._total_spent_usd:.2f}, "
            f"Merges: {report.merge_successes} ok / {report.merge_failures} failed"
        )

        await self._broadcast_complete(report)

        # Store report for API access
        self._last_report = report

        return results

    @property
    def budget_status(self) -> dict:
        """Current budget tracking status."""
        return {
            "spent_usd": round(self._total_spent_usd, 4),
            "limit_usd": round(self._budget_limit_usd, 2),
            "remaining_usd": round(
                max(0, self._budget_limit_usd - self._total_spent_usd), 4
            ) if self._budget_limit_usd else None,
            "exceeded": self._budget_exceeded,
        }

    async def _cancel_remaining(self, dag: TaskDAG) -> None:
        """Cancel all active workers and mark pending tasks as cancelled."""
        await self.pool.cancel_all()
        for task in dag.nodes.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                task.status = TaskStatus.CANCELLED

    async def _spawn_task_worker(
        self,
        task: TaskNode,
        dag: TaskDAG,
        base_branch: str,
    ) -> None:
        """Create worktree and spawn worker for a task."""
        # Budget pre-check: don't spawn if budget is nearly exhausted
        if self._budget_limit_usd:
            remaining = self._budget_limit_usd - self._total_spent_usd
            if remaining < task.budget_usd * 0.1:
                logger.warning(
                    f"Skipping task {task.id} — insufficient budget "
                    f"(${remaining:.2f} remaining, task needs ${task.budget_usd:.2f})"
                )
                task.status = TaskStatus.CANCELLED
                return

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

        # Build worker config — cap per-worker budget to remaining job budget
        worker_budget = task.budget_usd
        if self._budget_limit_usd:
            remaining = self._budget_limit_usd - self._total_spent_usd
            worker_budget = min(task.budget_usd, remaining)

        # Student model routing: low-priority tasks can use fine-tuned models
        worker_model = "sonnet"
        use_student = self._should_route_to_student(task)
        if use_student:
            worker_model = "student"
            self._tasks_routed_student += 1
            logger.info(f"Routing task {task.id} to student model")
        else:
            self._tasks_routed_teacher += 1

        config = WorkerConfig(
            model=worker_model,
            max_turns=task.estimated_turns,
            max_budget_usd=worker_budget,
            system_prompt_append=WORKER_SYSTEM_PROMPT,
            worktree_path=task.worktree_path,
        )

        # Spawn the worker
        try:
            await self.pool.spawn_worker(task, config)
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

            await self._broadcast_task_started(task)
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
        """Handle a failed worker: retry with LLM-rewritten prompt or mark as failed."""
        task = dag.nodes[result.task_id]

        if task.retry_count < task.max_retries:
            task.retry_count += 1

            # Try LLM-assisted prompt rewriting for smarter retries
            new_prompt = await self._rewrite_prompt_for_retry(task, result)
            task.worker_prompt = new_prompt

            task.status = TaskStatus.PENDING

            await self._broadcast_task_failed(
                result, will_retry=True
            )
            logger.info(
                f"Retrying task {task.id} "
                f"(attempt {task.retry_count}/{task.max_retries})"
            )
        else:
            blocked = dag.mark_failed(result.task_id, result.error or "")

            await self._broadcast_task_failed(
                result, will_retry=False
            )
            logger.warning(
                f"Task {task.id} failed after {task.retry_count} retries. "
                f"{len(blocked)} tasks blocked."
            )
            if self._on_task_failed:
                await self._on_task_failed(task, result)

    async def _rewrite_prompt_for_retry(
        self,
        task: TaskNode,
        result: WorkerResult,
    ) -> str:
        """Use the LLM to analyze the failure and generate an improved prompt.

        Falls back to the static RETRY_PROMPT_TEMPLATE if the LLM call fails.
        """
        original_prompt = task.worker_prompt or task.description

        # Try LLM-assisted rewriting
        try:
            from bashgym.orchestrator.task_dag import _call_llm

            improved = await _call_llm(
                self.llm_config,
                RETRY_ANALYSIS_SYSTEM,
                RETRY_ANALYSIS_TEMPLATE.format(
                    task_title=task.title,
                    original_prompt=original_prompt[:2000],
                    error=result.error or "Unknown error",
                    previous_output=result.output[-1500:],
                    attempt=task.retry_count,
                    max_attempts=task.max_retries,
                ),
            )

            if improved and len(improved.strip()) > 20:
                logger.info(
                    f"LLM rewrote retry prompt for task {task.id} "
                    f"({len(improved)} chars)"
                )
                return improved.strip()

        except Exception as e:
            logger.debug(f"LLM retry rewrite failed, using template: {e}")

        # Fallback to static template
        return RETRY_PROMPT_TEMPLATE.format(
            error=result.error or "Unknown error",
            previous_output=result.output[:1000],
            original_prompt=original_prompt,
        )

    # =========================================================================
    # Student Model Routing
    # =========================================================================

    def _should_route_to_student(self, task: TaskNode) -> bool:
        """Decide whether a task should use the student model.

        Routes to student when:
        1. A model_router is configured
        2. The router has a registered student model
        3. The task is LOW priority (simple tasks)
        4. The router's confidence threshold is met

        CRITICAL and HIGH priority tasks always use Claude (teacher).
        """
        if not self.model_router:
            return False

        from bashgym.orchestrator.models import TaskPriority

        # Only route LOW priority tasks to student
        if task.priority != TaskPriority.LOW:
            return False

        try:
            # Check if student model is registered and has sufficient confidence
            student = self.model_router.get_student_model()
            if not student:
                return False

            # Use the router's confidence-based decision
            decision = self.model_router.route(
                prompt=task.worker_prompt or task.description,
                task_complexity=self._estimate_complexity(task),
            )

            # ModelType.STUDENT means the router chose the student
            from bashgym.gym.router import ModelType
            return decision.model_type == ModelType.STUDENT

        except Exception as e:
            logger.debug(f"Student routing check failed: {e}")
            return False

    @staticmethod
    def _estimate_complexity(task: TaskNode) -> float:
        """Estimate task complexity as 0.0-1.0 for the router.

        Simple heuristic based on task properties.
        """
        score = 0.0

        # More files = more complex
        file_count = len(task.files_touched)
        if file_count <= 1:
            score += 0.1
        elif file_count <= 3:
            score += 0.3
        else:
            score += 0.6

        # More estimated turns = more complex
        if task.estimated_turns <= 10:
            score += 0.1
        elif task.estimated_turns <= 25:
            score += 0.3
        else:
            score += 0.4

        return min(1.0, score)

    # =========================================================================
    # WebSocket Broadcasting
    # =========================================================================

    async def _broadcast_task_started(self, task: TaskNode) -> None:
        """Broadcast task started event via WebSocket."""
        try:
            from bashgym.api.websocket import broadcast_orchestration_task_started
            await broadcast_orchestration_task_started(
                job_id=self.job_id,
                task_id=task.id,
                task_title=task.title,
                worker_count=self.pool.active_count,
            )
        except Exception:
            pass  # WebSocket not available (e.g., CLI mode)

    async def _broadcast_task_completed(
        self, result: WorkerResult, newly_unblocked: int
    ) -> None:
        """Broadcast task completed event via WebSocket."""
        try:
            from bashgym.api.websocket import broadcast_orchestration_task_completed
            await broadcast_orchestration_task_completed(
                job_id=self.job_id,
                task_id=result.task_id,
                cost_usd=result.cost_usd,
                duration_seconds=result.duration_seconds,
                newly_unblocked=newly_unblocked,
            )
        except Exception:
            pass

    async def _broadcast_task_failed(
        self, result: WorkerResult, will_retry: bool
    ) -> None:
        """Broadcast task failed event via WebSocket."""
        try:
            from bashgym.api.websocket import broadcast_orchestration_task_failed
            await broadcast_orchestration_task_failed(
                job_id=self.job_id,
                task_id=result.task_id,
                error=result.error or "Unknown error",
                will_retry=will_retry,
            )
        except Exception:
            pass

    async def _broadcast_budget_update(
        self, dag: TaskDAG, results: List[WorkerResult]
    ) -> None:
        """Broadcast budget status via WebSocket."""
        if not self._budget_limit_usd:
            return
        try:
            from bashgym.api.websocket import broadcast_orchestration_budget_update
            await broadcast_orchestration_budget_update(
                job_id=self.job_id,
                spent_usd=self._total_spent_usd,
                budget_usd=self._budget_limit_usd,
                task_count=sum(1 for r in results if r.success),
            )
        except Exception:
            pass

    async def _broadcast_complete(self, report: SynthesisReport) -> None:
        """Broadcast orchestration complete event via WebSocket."""
        try:
            from bashgym.api.websocket import broadcast_orchestration_complete
            await broadcast_orchestration_complete(
                job_id=self.job_id,
                completed=report.completed_tasks,
                failed=report.failed_tasks,
                total_cost=self._total_spent_usd,
                total_time=report.total_duration_seconds,
                merge_successes=report.merge_successes,
                merge_failures=report.merge_failures,
            )
        except Exception:
            pass

    # =========================================================================
    # Trace Ingestion (for training pipeline)
    # =========================================================================

    async def ingest_traces(self, results: List[WorkerResult]) -> int:
        """Feed orchestration traces into the Factory training pipeline.

        Imports Claude Code session traces from completed workers using
        the existing ClaudeSessionImporter infrastructure.

        Multi-agent traces are high-value training signal:
        - Task decomposition patterns
        - Tool-use sequences
        - Error recovery strategies

        Args:
            results: Worker results to ingest

        Returns:
            Number of traces successfully ingested
        """
        try:
            from bashgym.trace_capture.importers import ClaudeSessionImporter
        except ImportError:
            logger.debug("trace_capture not available, skipping ingestion")
            return 0

        importer = ClaudeSessionImporter()
        count = 0

        for result in results:
            if not result.session_id or not result.success:
                continue

            # Find the session file in Claude's projects directory
            session_file = self._find_session_file(importer, result.session_id)
            if not session_file:
                logger.debug(
                    f"Session file not found for {result.task_id} "
                    f"(session {result.session_id})"
                )
                continue

            try:
                import_result = await importer.import_session_async(
                    session_file, force=False
                )
                if import_result.error:
                    logger.warning(
                        f"Trace import failed for {result.task_id}: "
                        f"{import_result.error}"
                    )
                elif import_result.skipped:
                    logger.debug(
                        f"Trace already imported for {result.task_id}: "
                        f"{import_result.skip_reason}"
                    )
                else:
                    count += 1
                    logger.debug(
                        f"Imported {import_result.steps_imported} steps "
                        f"from task {result.task_id}"
                    )
            except Exception as e:
                logger.warning(f"Trace ingestion error for {result.task_id}: {e}")

        logger.info(f"Ingested {count} traces into training pipeline")
        return count

    @staticmethod
    def _find_session_file(importer, session_id: str):
        """Find a Claude Code session file by session ID."""
        from pathlib import Path

        projects_dir = importer.find_projects_dir()
        if not projects_dir:
            return None

        # Session files are at ~/.claude/projects/<slug>/<session_id>.jsonl
        for session_file, _ in importer.find_session_files():
            if session_id in session_file.stem:
                return session_file

        return None
