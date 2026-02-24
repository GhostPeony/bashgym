"""E2E tests for the full orchestration pipeline.

Tests the flow: spec → DAG decomposition → execution → synthesis
using real git repos and mocked LLM/CLI backends.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bashgym.orchestrator.agent import OrchestrationAgent
from bashgym.orchestrator.models import (
    LLMConfig,
    LLMProvider,
    OrchestratorSpec,
    TaskNode,
    TaskPriority,
    TaskStatus,
    WorkerResult,
)
from bashgym.orchestrator.synthesizer import ResultSynthesizer, SynthesisReport
from bashgym.orchestrator.task_dag import TaskDAG

from tests.orchestrator.conftest import make_result, make_task


# =============================================================================
# Spec Decomposition
# =============================================================================


class TestSpecDecomposition:
    """Tests that submit_spec correctly decomposes into a TaskDAG."""

    @pytest.mark.asyncio
    async def test_submit_spec_returns_dag(self, sample_spec, mock_llm_decomposition):
        agent = OrchestrationAgent()
        dag = await agent.submit_spec(sample_spec)

        assert isinstance(dag, TaskDAG)
        assert len(dag.nodes) == 3
        mock_llm_decomposition.assert_called_once()

    @pytest.mark.asyncio
    async def test_dag_has_correct_dependencies(self, sample_spec, mock_llm_decomposition):
        agent = OrchestrationAgent()
        dag = await agent.submit_spec(sample_spec)

        # task_1 has no deps
        assert dag.nodes["task_1"].dependencies == []
        # task_2 and task_3 depend on task_1
        assert "task_1" in dag.nodes["task_2"].dependencies
        assert "task_1" in dag.nodes["task_3"].dependencies

    @pytest.mark.asyncio
    async def test_dag_tasks_start_pending(self, sample_spec, mock_llm_decomposition):
        agent = OrchestrationAgent()
        dag = await agent.submit_spec(sample_spec)

        for task in dag.nodes.values():
            assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_creates_context_builder(self, sample_spec, mock_llm_decomposition):
        agent = OrchestrationAgent()
        dag = await agent.submit_spec(sample_spec)

        assert agent.context_builder is not None
        assert agent._spec_title == sample_spec.title

    @pytest.mark.asyncio
    async def test_dag_stats_after_decomposition(self, sample_spec, mock_llm_decomposition):
        agent = OrchestrationAgent()
        dag = await agent.submit_spec(sample_spec)

        stats = dag.stats
        assert stats.get("pending", 0) == 3


# =============================================================================
# Execution with mocked workers
# =============================================================================


class TestExecution:
    """Tests for the execute() phase with mocked workers."""

    @pytest.mark.asyncio
    async def test_execute_single_task_success(self):
        """DAG with 1 task → mock worker succeeds → 1 result."""
        dag = TaskDAG()
        dag.add_task(make_task("only", "Only task"))

        result = make_result("only", success=True, cost=0.50)

        agent = OrchestrationAgent(use_worktrees=False)
        agent.dag = dag

        # Mock the pool to return our result
        with patch.object(agent.pool, "spawn_worker", new_callable=AsyncMock), \
             patch.object(agent.pool, "wait_for_any", new_callable=AsyncMock, return_value=result), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock), \
             patch.object(type(agent.pool), "active_count", new_callable=lambda: property(lambda self: 1)), \
             patch.object(type(agent.pool), "available_slots", new_callable=lambda: property(lambda self: 4)):

            # After one result, DAG should be complete
            def side_effect_complete(*args, **kwargs):
                dag.nodes["only"].status = TaskStatus.RUNNING
                return AsyncMock()()

            agent.pool.spawn_worker.side_effect = side_effect_complete

            results = await agent.execute(dag)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_linear_chain(self, linear_dag):
        """A→B→C — each completes in order."""
        dag = linear_dag
        call_count = 0
        task_order = ["A", "B", "C"]

        agent = OrchestrationAgent(use_worktrees=False)

        results_sequence = [
            make_result("A", success=True),
            make_result("B", success=True),
            make_result("C", success=True),
        ]

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_sequence[call_count]
            call_count += 1
            return r

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            # Simulate active_count based on call state
            type(agent.pool).active_count = property(
                lambda self: 1 if call_count < 3 else 0
            )
            type(agent.pool).available_slots = property(
                lambda self: 4
            )

            results = await agent.execute(dag)

        assert len(results) == 3
        assert all(r.success for r in results)
        # All tasks should be COMPLETED
        for tid in ["A", "B", "C"]:
            assert dag.nodes[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_failure_blocks_dependents(self, three_task_dag):
        """task_1 fails → task_2 and task_3 should be BLOCKED."""
        dag = three_task_dag
        fail_result = make_result("task_1", success=False, error="Build failed")

        agent = OrchestrationAgent(use_worktrees=False)

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", new_callable=AsyncMock, return_value=fail_result), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock), \
             patch.object(
                 agent, "_rewrite_prompt_for_retry", new_callable=AsyncMock,
                 return_value="retry prompt",
             ):
            type(agent.pool).active_count = property(lambda self: 1)
            type(agent.pool).available_slots = property(lambda self: 4)

            # task_1 has max_retries=2, so it will retry before final failure
            # Set max_retries to 0 so it fails immediately
            dag.nodes["task_1"].max_retries = 0

            results = await agent.execute(dag)

        assert dag.nodes["task_1"].status == TaskStatus.FAILED
        assert dag.nodes["task_2"].status == TaskStatus.BLOCKED
        assert dag.nodes["task_3"].status == TaskStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_parallel_tasks_dispatched(self, parallel_dag):
        """A and B have no deps — both should be spawnable simultaneously."""
        dag = parallel_dag
        spawned_tasks = []

        agent = OrchestrationAgent(use_worktrees=False)

        async def mock_spawn(task, config=None):
            spawned_tasks.append(task.id)
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        call_count = 0
        results_seq = [
            make_result("A", success=True),
            make_result("B", success=True),
        ]

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_seq[call_count]
            call_count += 1
            return r

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(
                lambda self: max(0, 2 - call_count)
            )
            type(agent.pool).available_slots = property(
                lambda self: 5 - max(0, 2 - call_count)
            )

            results = await agent.execute(dag)

        # Both A and B should have been spawned
        assert "A" in spawned_tasks
        assert "B" in spawned_tasks
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_diamond_dag_d_waits_for_b_and_c(self, diamond_dag):
        """A→B, A→C, B→D, C→D — D should only start after both B and C."""
        dag = diamond_dag
        spawn_order = []
        call_count = 0
        results_seq = [
            make_result("A"),
            make_result("B"),
            make_result("C"),
            make_result("D"),
        ]

        agent = OrchestrationAgent(use_worktrees=False)

        async def mock_spawn(task, config=None):
            spawn_order.append(task.id)
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_seq[call_count]
            call_count += 1
            return r

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(lambda self: 1)
            type(agent.pool).available_slots = property(lambda self: 4)

            results = await agent.execute(dag)

        assert len(results) == 4
        # D must come after both B and C
        d_idx = spawn_order.index("D")
        assert "B" in spawn_order[:d_idx]
        assert "C" in spawn_order[:d_idx]


# =============================================================================
# Budget Tracking
# =============================================================================


class TestBudgetTracking:
    """Tests for budget tracking during execution."""

    @pytest.mark.asyncio
    async def test_budget_tracks_spending(self):
        """Execute 3 tasks at $0.50 each → budget shows $1.50 spent."""
        dag = TaskDAG()
        dag.add_task(make_task("a", budget=0.50))
        dag.add_task(make_task("b", budget=0.50))
        dag.add_task(make_task("c", budget=0.50))

        call_count = 0
        results_seq = [
            make_result("a", cost=0.50),
            make_result("b", cost=0.50),
            make_result("c", cost=0.50),
        ]

        agent = OrchestrationAgent(use_worktrees=False)

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_seq[call_count]
            call_count += 1
            return r

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(
                lambda self: 1 if call_count < 3 else 0
            )
            type(agent.pool).available_slots = property(lambda self: 4)

            results = await agent.execute(dag, budget_usd=10.0)

        assert agent._total_spent_usd == pytest.approx(1.50)
        status = agent.budget_status
        assert status["spent_usd"] == pytest.approx(1.50, abs=0.01)
        assert status["limit_usd"] == 10.0
        assert status["exceeded"] is False

    @pytest.mark.asyncio
    async def test_budget_exhaustion_cancels_remaining(self):
        """Budget=$1.00, 3 tasks at $0.50 → 3rd should not start."""
        dag = TaskDAG()
        dag.add_task(make_task("a", budget=0.50))
        dag.add_task(make_task("b", budget=0.50))
        dag.add_task(make_task("c", budget=0.50))

        call_count = 0
        results_seq = [
            make_result("a", cost=0.50),
            make_result("b", cost=0.50),
        ]
        spawned = []

        agent = OrchestrationAgent(use_worktrees=False)

        async def mock_spawn(task, config=None):
            spawned.append(task.id)
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_seq[call_count]
            call_count += 1
            return r

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(
                lambda self: 1 if call_count < 2 else 0
            )
            type(agent.pool).available_slots = property(lambda self: 4)

            results = await agent.execute(dag, budget_usd=1.0)

        assert agent._budget_exceeded is True
        assert agent.budget_status["exceeded"] is True
        # Task c should be CANCELLED
        assert dag.nodes["c"].status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_budget_status_property(self):
        """Verify budget_status returns correct fields."""
        agent = OrchestrationAgent(use_worktrees=False)
        agent._total_spent_usd = 3.50
        agent._budget_limit_usd = 10.0
        agent._budget_exceeded = False

        status = agent.budget_status
        assert status["spent_usd"] == pytest.approx(3.50, abs=0.01)
        assert status["limit_usd"] == 10.0
        assert status["remaining_usd"] == pytest.approx(6.50, abs=0.01)
        assert status["exceeded"] is False


# =============================================================================
# Synthesis
# =============================================================================


class TestSynthesis:
    """Tests for the synthesis/merge phase."""

    @pytest.mark.asyncio
    async def test_synthesis_report_from_results(self):
        """Verify report aggregates costs, durations, and files."""
        dag = TaskDAG()
        dag.add_task(make_task("a", files=["src/a.py"]))
        dag.add_task(make_task("b", files=["src/b.py"]))

        # Mark both completed
        result_a = make_result("a", cost=1.0, duration=30.0, files=["src/a.py"])
        result_b = make_result("b", cost=2.0, duration=45.0, files=["src/b.py"])
        dag.mark_completed("a", result_a)
        dag.mark_completed("b", result_b)

        synthesizer = ResultSynthesizer(worktrees=None)
        report = await synthesizer.synthesize(dag, [result_a, result_b])

        assert report.total_tasks == 2
        assert report.completed_tasks == 2
        assert report.failed_tasks == 0
        assert report.total_cost_usd == pytest.approx(3.0)
        assert report.total_duration_seconds == pytest.approx(75.0)
        assert "src/a.py" in report.files_modified
        assert "src/b.py" in report.files_modified

    @pytest.mark.asyncio
    async def test_synthesis_skips_failed_tasks(self):
        """Only completed tasks appear in merge results."""
        dag = TaskDAG()
        dag.add_task(make_task("a"))
        dag.add_task(make_task("b"))

        result_a = make_result("a", success=True, cost=1.0, duration=30.0)
        result_b = make_result("b", success=False, cost=0.5, duration=10.0, error="fail")
        dag.mark_completed("a", result_a)
        dag.mark_failed("b", "fail")

        synthesizer = ResultSynthesizer(worktrees=None)
        report = await synthesizer.synthesize(dag, [result_a, result_b])

        assert report.completed_tasks == 1
        assert report.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_synthesis_with_real_worktrees(self, async_git_repo):
        """Merge 2 worktree branches via synthesizer with real git."""
        repo = async_git_repo
        git_helper = repo._git

        mgr = WorktreeManager(repo)

        # Create worktrees and make changes
        wt_a = await mgr.create("a", "task/a", "main")
        wt_b = await mgr.create("b", "task/b", "main")

        # Add different files in each
        (wt_a / "file_a.txt").write_text("content A\n")
        await git_helper("add", "file_a.txt", cwd=wt_a)
        await git_helper("commit", "-m", "Add file A", cwd=wt_a)

        (wt_b / "file_b.txt").write_text("content B\n")
        await git_helper("add", "file_b.txt", cwd=wt_b)
        await git_helper("commit", "-m", "Add file B", cwd=wt_b)

        # Build DAG with both completed
        dag = TaskDAG()
        dag.add_task(make_task("a"))
        dag.add_task(make_task("b"))

        result_a = make_result("a", files=["file_a.txt"])
        result_b = make_result("b", files=["file_b.txt"])
        dag.mark_completed("a", result_a)
        dag.mark_completed("b", result_b)

        # Switch to main for merge
        await git_helper("checkout", "main", cwd=repo)

        synthesizer = ResultSynthesizer(worktrees=mgr)
        report = await synthesizer.synthesize(dag, [result_a, result_b], "main")

        assert report.merge_successes == 2
        assert report.merge_failures == 0
        # Both files should be on main
        assert (repo / "file_a.txt").exists()
        assert (repo / "file_b.txt").exists()


# =============================================================================
# Callbacks
# =============================================================================


class TestCallbacks:
    """Tests for task lifecycle callbacks."""

    @pytest.mark.asyncio
    async def test_on_task_started_callback(self):
        """Verify on_task_started fires with correct task."""
        started_tasks = []

        async def on_started(task):
            started_tasks.append(task.id)

        dag = TaskDAG()
        dag.add_task(make_task("t1"))

        result = make_result("t1")

        agent = OrchestrationAgent(
            use_worktrees=False,
            on_task_started=on_started,
        )

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", new_callable=AsyncMock, return_value=result), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(lambda self: 1)
            type(agent.pool).available_slots = property(lambda self: 4)

            await agent.execute(dag)

        assert "t1" in started_tasks

    @pytest.mark.asyncio
    async def test_on_task_completed_callback(self):
        """Verify on_task_completed fires with task and result."""
        completed = []

        async def on_completed(task, worker_result):
            completed.append((task.id, worker_result.success))

        dag = TaskDAG()
        dag.add_task(make_task("t1"))

        result = make_result("t1", success=True)

        agent = OrchestrationAgent(
            use_worktrees=False,
            on_task_completed=on_completed,
        )

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", new_callable=AsyncMock, return_value=result), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(lambda self: 1)
            type(agent.pool).available_slots = property(lambda self: 4)

            await agent.execute(dag)

        assert len(completed) == 1
        assert completed[0] == ("t1", True)

    @pytest.mark.asyncio
    async def test_on_task_failed_callback(self):
        """Verify on_task_failed fires with error info."""
        failed = []

        async def on_failed(task, worker_result):
            failed.append((task.id, worker_result.error))

        dag = TaskDAG()
        dag.add_task(make_task("t1"))
        dag.nodes["t1"].max_retries = 0  # No retries, fail immediately

        result = make_result("t1", success=False, error="Compilation error")

        agent = OrchestrationAgent(
            use_worktrees=False,
            on_task_failed=on_failed,
        )

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", new_callable=AsyncMock, return_value=result), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(lambda self: 1)
            type(agent.pool).available_slots = property(lambda self: 4)

            await agent.execute(dag)

        assert len(failed) == 1
        assert failed[0][0] == "t1"
        assert "Compilation error" in failed[0][1]


# =============================================================================
# Full Pipeline: spec → decompose → execute → synthesize
# =============================================================================


class TestFullPipeline:
    """Tests the complete spec-to-results pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_llm_and_workers(
        self, sample_spec, mock_llm_decomposition,
    ):
        """Full flow: submit_spec → execute → results."""
        agent = OrchestrationAgent(use_worktrees=False)

        # Phase 1: Decompose
        dag = await agent.submit_spec(sample_spec)
        assert len(dag.nodes) == 3

        # Phase 2-4: Execute with mocked workers
        call_count = 0
        results_seq = [
            make_result("task_1", cost=0.30),
            make_result("task_2", cost=0.50),
            make_result("task_3", cost=0.20),
        ]

        async def mock_spawn(task, config=None):
            task.status = TaskStatus.RUNNING
            dag.nodes[task.id].status = TaskStatus.RUNNING

        async def mock_wait_any(timeout=600):
            nonlocal call_count
            r = results_seq[call_count]
            call_count += 1
            return r

        with patch.object(agent.pool, "spawn_worker", side_effect=mock_spawn), \
             patch.object(agent.pool, "wait_for_any", side_effect=mock_wait_any), \
             patch.object(agent.pool, "cancel_all", new_callable=AsyncMock):
            type(agent.pool).active_count = property(
                lambda self: 1 if call_count < 3 else 0
            )
            type(agent.pool).available_slots = property(lambda self: 4)

            results = await agent.execute(dag, budget_usd=5.0)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert agent._total_spent_usd == pytest.approx(1.0)

        # Synthesis report should exist
        assert hasattr(agent, "_last_report")
        report = agent._last_report
        assert report.total_tasks == 3
        assert report.completed_tasks == 3
