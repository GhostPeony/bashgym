# tests/orchestrator/test_orchestrator_workers.py
"""Tests for WorkerPool, WorktreeManager, and ResultSynthesizer."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from bashgym.orchestrator.models import (
    TaskNode, TaskStatus, TaskPriority,
    WorkerConfig, WorkerResult, MergeResult,
    LLMConfig, LLMProvider,
)
from bashgym.orchestrator.dispatcher import WorkerPool
from bashgym.orchestrator.worktree import WorktreeManager
from bashgym.orchestrator.synthesizer import (
    ResultSynthesizer, SynthesisReport, CONFLICT_RESOLUTION_PROMPT,
)
from bashgym.orchestrator.task_dag import TaskDAG


# =============================================================================
# WorkerPool Tests
# =============================================================================


class TestWorkerPoolInit:
    """Test WorkerPool initialization."""

    def test_default_init(self):
        pool = WorkerPool()
        assert pool.max_workers == 5
        assert pool.default_config.model == "sonnet"
        assert pool.active_count == 0
        assert pool.available_slots == 5
        assert pool.total_cost == 0.0

    def test_custom_init(self):
        config = WorkerConfig(model="opus", max_turns=50, max_budget_usd=5.0)
        pool = WorkerPool(max_workers=3, default_config=config)
        assert pool.max_workers == 3
        assert pool.default_config.model == "opus"
        assert pool.available_slots == 3


class TestWorkerPoolSpawn:
    """Test worker spawning."""

    @pytest.mark.asyncio
    async def test_spawn_respects_max_workers(self):
        pool = WorkerPool(max_workers=1)

        # Simulate a running worker by adding to _processes directly
        pool._processes["t1"] = MagicMock()

        task2 = TaskNode(id="t2", title="Task 2", description="Do 2")
        with pytest.raises(RuntimeError, match="Max workers"):
            await pool.spawn_worker(task2)

    @pytest.mark.asyncio
    async def test_spawn_builds_correct_command(self):
        pool = WorkerPool(max_workers=5)
        task = TaskNode(
            id="t1",
            title="Test Task",
            description="A test",
            worker_prompt="Do the thing",
        )

        captured_cmd = None

        async def mock_exec(*args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = args
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'{}', b''))
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await pool.spawn_worker(task)
            # Let the async task run
            await asyncio.sleep(0.1)

        assert captured_cmd is not None
        assert captured_cmd[0] == "claude"
        assert captured_cmd[1] == "-p"
        assert captured_cmd[2] == "Do the thing"

    @pytest.mark.asyncio
    async def test_spawn_uses_task_prompt_or_description(self):
        pool = WorkerPool(max_workers=5)

        # When worker_prompt is set, it's used
        task = TaskNode(
            id="t1", title="Test", description="Desc",
            worker_prompt="Custom prompt",
        )

        captured_cmds = []

        async def mock_exec(*args, **kwargs):
            captured_cmds.append(args)
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'{}', b''))
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await pool.spawn_worker(task)
            await asyncio.sleep(0.1)

        assert "Custom prompt" in captured_cmds[0]

    @pytest.mark.asyncio
    async def test_spawn_sets_cwd_to_worktree(self):
        pool = WorkerPool(max_workers=5)
        task = TaskNode(
            id="t1", title="Test", description="Desc",
            worktree_path=Path("/tmp/worktree/t1"),
        )

        captured_kwargs = None

        async def mock_exec(*args, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'{}', b''))
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await pool.spawn_worker(task)
            await asyncio.sleep(0.1)

        assert captured_kwargs["cwd"] == str(Path("/tmp/worktree/t1"))


class TestWorkerPoolResults:
    """Test result collection and waiting."""

    def _prepare_pool_for_run(self, pool, worker_id="t1"):
        """Pre-populate tracking structures that spawn_worker normally creates."""
        pool._output_buffers[worker_id] = []
        pool._completion_events[worker_id] = asyncio.Event()

    @pytest.mark.asyncio
    async def test_run_worker_parses_json_output(self):
        pool = WorkerPool(max_workers=5)
        self._prepare_pool_for_run(pool)

        output = json.dumps({
            "session_id": "sess-123",
            "result": "All done!",
            "usage": {"total_tokens": 5000},
            "cost_usd": 0.15,
        })

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(output.encode(), b'')
        )
        mock_proc.returncode = 0
        mock_proc.kill = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await pool._run_worker(
                "t1", ["claude", "-p", "test"], None, 60.0
            )

        assert result.success is True
        assert result.task_id == "t1"
        assert result.session_id == "sess-123"
        assert result.cost_usd == 0.15
        assert result.tokens_used == 5000

    @pytest.mark.asyncio
    async def test_run_worker_handles_timeout(self):
        pool = WorkerPool(max_workers=5)
        self._prepare_pool_for_run(pool)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_proc.kill = AsyncMock()
        mock_proc.returncode = -1

        # After kill, communicate should work
        communicate_calls = [0]

        async def communicate_side_effect():
            communicate_calls[0] += 1
            if communicate_calls[0] == 1:
                raise asyncio.TimeoutError()
            return (b'', b'')

        mock_proc.communicate = AsyncMock(side_effect=communicate_side_effect)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await pool._run_worker(
                "t1", ["claude", "-p", "test"], None, 0.001
            )

        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_worker_handles_cli_not_found(self):
        pool = WorkerPool(max_workers=5)
        self._prepare_pool_for_run(pool)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            result = await pool._run_worker(
                "t1", ["claude", "-p", "test"], None, 60.0
            )

        assert result.success is False
        assert "claude CLI not found" in result.error

    @pytest.mark.asyncio
    async def test_run_worker_handles_nonzero_exit(self):
        pool = WorkerPool(max_workers=5)
        self._prepare_pool_for_run(pool)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b'partial output', b'Error: something broke')
        )
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await pool._run_worker(
                "t1", ["claude", "-p", "test"], None, 60.0
            )

        assert result.success is False
        assert result.exit_code == 1
        assert "something broke" in result.error

    @pytest.mark.asyncio
    async def test_wait_for_worker_returns_cached_result(self):
        pool = WorkerPool()
        # Pre-populate results
        result = WorkerResult(
            task_id="t1", session_id="s1", success=True,
            output="done", exit_code=0, duration_seconds=1.0,
        )
        pool._results["t1"] = result
        pool._completion_events["t1"] = asyncio.Event()
        pool._completion_events["t1"].set()

        got = await pool.wait_for_worker("t1")
        assert got.task_id == "t1"
        assert got.success is True

    @pytest.mark.asyncio
    async def test_wait_for_worker_unknown_raises(self):
        pool = WorkerPool()
        with pytest.raises(KeyError, match="not found"):
            await pool.wait_for_worker("nonexistent")

    @pytest.mark.asyncio
    async def test_wait_for_any_no_workers_raises(self):
        pool = WorkerPool()
        with pytest.raises(RuntimeError, match="No active workers"):
            await pool.wait_for_any()


class TestWorkerPoolCancel:
    """Test worker cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_worker(self):
        pool = WorkerPool()

        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))

        pool._processes["t1"] = mock_proc
        pool._tasks["t1"] = AsyncMock()
        pool._tasks["t1"].cancel = MagicMock()

        await pool.cancel_worker("t1")

        mock_proc.terminate.assert_called_once()
        assert "t1" not in pool._processes

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        pool = WorkerPool()

        for i in range(3):
            mock_proc = AsyncMock()
            mock_proc.terminate = MagicMock()
            mock_proc.kill = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b'', b''))
            pool._processes[f"t{i}"] = mock_proc
            pool._tasks[f"t{i}"] = AsyncMock()
            pool._tasks[f"t{i}"].cancel = MagicMock()

        await pool.cancel_all()
        assert pool.active_count == 0


class TestWorkerPoolProperties:
    """Test computed properties."""

    def test_total_cost(self):
        pool = WorkerPool()
        pool._results["t1"] = WorkerResult(
            task_id="t1", session_id="", success=True,
            output="", exit_code=0, duration_seconds=10,
            cost_usd=0.50,
        )
        pool._results["t2"] = WorkerResult(
            task_id="t2", session_id="", success=False,
            output="", exit_code=1, duration_seconds=5,
            cost_usd=0.25,
        )
        assert pool.total_cost == pytest.approx(0.75)

    def test_get_worker_output(self):
        pool = WorkerPool()
        pool._output_buffers["t1"] = ["line1", "line2"]
        assert pool.get_worker_output("t1") == ["line1", "line2"]
        assert pool.get_worker_output("nonexistent") == []


# =============================================================================
# WorktreeManager Tests
# =============================================================================


class TestWorktreeManager:
    """Test git worktree management."""

    def test_init_defaults(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        assert mgr.repo_path == tmp_path
        assert mgr.worktree_base == tmp_path / ".worktrees"
        assert mgr.active_count == 0

    def test_init_custom_base(self, tmp_path):
        custom_base = tmp_path / "my_worktrees"
        mgr = WorktreeManager(tmp_path, worktree_base=custom_base)
        assert mgr.worktree_base == custom_base

    @pytest.mark.asyncio
    async def test_create_worktree_success(self, tmp_path):
        mgr = WorktreeManager(tmp_path)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            path = await mgr.create("t1", "task/t1", "main")

        assert path == tmp_path / ".worktrees" / "t1"
        assert "t1" in mgr.active_worktrees
        assert mgr._branches["t1"] == "task/t1"

    @pytest.mark.asyncio
    async def test_create_worktree_branch_exists_fallback(self, tmp_path):
        mgr = WorktreeManager(tmp_path)

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            proc = AsyncMock()
            if call_count[0] == 1:
                # First call fails with "already exists"
                proc.communicate = AsyncMock(
                    return_value=(b'', b'branch already exists')
                )
                proc.returncode = 1
            else:
                # Second call succeeds (without -b flag)
                proc.communicate = AsyncMock(return_value=(b'', b''))
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            path = await mgr.create("t1", "task/t1", "main")

        assert call_count[0] == 2
        assert "t1" in mgr.active_worktrees

    @pytest.mark.asyncio
    async def test_create_worktree_failure_raises(self, tmp_path):
        mgr = WorktreeManager(tmp_path)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b'', b'fatal: some error')
        )
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Failed to create worktree"):
                await mgr.create("t1", "task/t1", "main")

    @pytest.mark.asyncio
    async def test_merge_success(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        mgr._branches["t1"] = "task/t1"

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            proc = AsyncMock()
            if call_count[0] == 1:
                # git diff --name-only
                proc.communicate = AsyncMock(
                    return_value=(b'file1.py\nfile2.py', b'')
                )
                proc.returncode = 0
            else:
                # git merge
                proc.communicate = AsyncMock(return_value=(b'', b''))
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await mgr.merge("t1", "main")

        assert result.success is True
        assert result.branch == "task/t1"
        assert result.files_merged == ["file1.py", "file2.py"]

    @pytest.mark.asyncio
    async def test_merge_conflict(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        mgr._branches["t1"] = "task/t1"

        call_count = [0]

        async def mock_exec(*args, **kwargs):
            call_count[0] += 1
            proc = AsyncMock()
            if call_count[0] == 1:
                # git diff --name-only
                proc.communicate = AsyncMock(
                    return_value=(b'conflicted.py', b'')
                )
                proc.returncode = 0
            elif call_count[0] == 2:
                # git merge -> CONFLICT
                proc.communicate = AsyncMock(
                    return_value=(b'CONFLICT in conflicted.py', b'CONFLICT')
                )
                proc.returncode = 1
            elif call_count[0] == 3:
                # git diff --name-only --diff-filter=U
                proc.communicate = AsyncMock(
                    return_value=(b'conflicted.py', b'')
                )
                proc.returncode = 0
            else:
                # git merge --abort
                proc.communicate = AsyncMock(return_value=(b'', b''))
                proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await mgr.merge("t1", "main")

        assert result.success is False
        assert result.conflicts == ["conflicted.py"]
        assert "conflicts detected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_merge_unknown_task(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        result = await mgr.merge("nonexistent", "main")
        assert result.success is False
        assert "No branch found" in result.error

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree_and_branch(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        mgr.active_worktrees["t1"] = tmp_path / ".worktrees" / "t1"
        mgr._branches["t1"] = "task/t1"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await mgr.cleanup("t1")

        assert "t1" not in mgr.active_worktrees
        assert "t1" not in mgr._branches

    @pytest.mark.asyncio
    async def test_cleanup_all(self, tmp_path):
        mgr = WorktreeManager(tmp_path)
        for i in range(3):
            mgr.active_worktrees[f"t{i}"] = tmp_path / ".worktrees" / f"t{i}"
            mgr._branches[f"t{i}"] = f"task/t{i}"

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await mgr.cleanup_all()

        assert mgr.active_count == 0


# =============================================================================
# SynthesisReport Tests
# =============================================================================


class TestSynthesisReport:
    """Test SynthesisReport dataclass."""

    def test_defaults(self):
        report = SynthesisReport()
        assert report.total_tasks == 0
        assert report.completed_tasks == 0
        assert report.failed_tasks == 0
        assert report.merge_successes == 0
        assert report.merge_failures == 0
        assert report.conflicts_resolved == 0
        assert report.conflicts_unresolved == 0
        assert report.total_cost_usd == 0.0
        assert report.merge_results == []
        assert report.files_modified == []

    def test_to_dict(self):
        report = SynthesisReport(
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            merge_successes=3,
            merge_failures=1,
            conflicts_resolved=2,
            total_cost_usd=1.2345,
            total_duration_seconds=120.6789,
            files_modified=["a.py", "b.py"],
        )
        d = report.to_dict()
        assert d["total_tasks"] == 5
        assert d["completed_tasks"] == 4
        assert d["total_cost_usd"] == 1.2345  # rounded to 4 decimals
        assert d["total_duration_seconds"] == 120.7  # rounded to 1 decimal
        assert d["files_modified"] == ["a.py", "b.py"]


# =============================================================================
# ResultSynthesizer Tests
# =============================================================================


class TestResultSynthesizer:
    """Test ResultSynthesizer merge and conflict resolution."""

    def _make_dag(self):
        """Create a simple 2-task DAG for testing."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="Task 1", description="Do 1",
            files_touched=["a.py"],
        ))
        dag.add_task(TaskNode(
            id="t2", title="Task 2", description="Do 2",
            dependencies=["t1"], files_touched=["b.py"],
        ))
        dag.mark_completed("t1", WorkerResult(
            task_id="t1", session_id="s1", success=True,
            output="done", exit_code=0, duration_seconds=10.0,
            cost_usd=0.50,
        ))
        dag.mark_completed("t2", WorkerResult(
            task_id="t2", session_id="s2", success=True,
            output="done", exit_code=0, duration_seconds=15.0,
            cost_usd=0.75,
        ))
        return dag

    def _make_results(self):
        return [
            WorkerResult(
                task_id="t1", session_id="s1", success=True,
                output="done1", exit_code=0, duration_seconds=10.0,
                cost_usd=0.50, files_modified=["a.py"],
            ),
            WorkerResult(
                task_id="t2", session_id="s2", success=True,
                output="done2", exit_code=0, duration_seconds=15.0,
                cost_usd=0.75, files_modified=["b.py"],
            ),
        ]

    @pytest.mark.asyncio
    async def test_synthesize_without_worktrees(self):
        """Synthesize without worktrees skips merge."""
        dag = self._make_dag()
        results = self._make_results()

        synth = ResultSynthesizer(worktrees=None)
        report = await synth.synthesize(dag, results, "main")

        assert report.total_tasks == 2
        assert report.completed_tasks == 2
        assert report.failed_tasks == 0
        assert report.total_cost_usd == pytest.approx(1.25)
        assert report.merge_successes == 0  # No merges attempted
        assert sorted(report.files_modified) == ["a.py", "b.py"]

    @pytest.mark.asyncio
    async def test_synthesize_with_successful_merges(self):
        """Synthesize with worktrees that merge cleanly."""
        dag = self._make_dag()
        results = self._make_results()

        mock_worktrees = AsyncMock(spec=WorktreeManager)
        mock_worktrees.merge = AsyncMock(side_effect=[
            MergeResult(task_id="t1", branch="task/t1", success=True,
                        files_merged=["a.py"]),
            MergeResult(task_id="t2", branch="task/t2", success=True,
                        files_merged=["b.py"]),
        ])
        mock_worktrees.cleanup_all = AsyncMock()

        synth = ResultSynthesizer(worktrees=mock_worktrees)
        report = await synth.synthesize(dag, results, "main")

        assert report.merge_successes == 2
        assert report.merge_failures == 0
        mock_worktrees.cleanup_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_merge_failure_no_conflicts(self):
        """Merge fails without conflicts (e.g., unrelated error)."""
        dag = self._make_dag()
        results = self._make_results()

        mock_worktrees = AsyncMock(spec=WorktreeManager)
        mock_worktrees.merge = AsyncMock(side_effect=[
            MergeResult(task_id="t1", branch="task/t1", success=True,
                        files_merged=["a.py"]),
            MergeResult(task_id="t2", branch="task/t2", success=False,
                        error="Detached HEAD"),
        ])
        mock_worktrees.cleanup_all = AsyncMock()

        synth = ResultSynthesizer(worktrees=mock_worktrees)
        report = await synth.synthesize(dag, results, "main")

        assert report.merge_successes == 1
        assert report.merge_failures == 1

    @pytest.mark.asyncio
    async def test_synthesize_with_conflict_resolution_disabled(self):
        """Conflicts without auto-resolve just fails."""
        dag = self._make_dag()
        results = self._make_results()

        mock_worktrees = AsyncMock(spec=WorktreeManager)
        mock_worktrees.merge = AsyncMock(side_effect=[
            MergeResult(task_id="t1", branch="task/t1", success=True,
                        files_merged=["a.py"]),
            MergeResult(task_id="t2", branch="task/t2", success=False,
                        conflicts=["shared.py"],
                        error="Merge conflicts detected"),
        ])
        mock_worktrees.cleanup_all = AsyncMock()

        synth = ResultSynthesizer(
            worktrees=mock_worktrees,
            auto_resolve_conflicts=False,
        )
        report = await synth.synthesize(dag, results, "main")

        assert report.merge_failures == 1
        assert report.conflicts_unresolved == 1
        assert report.conflicts_resolved == 0

    @pytest.mark.asyncio
    async def test_synthesize_merges_in_topological_order(self):
        """Tasks are merged in dependency order."""
        dag = self._make_dag()
        results = self._make_results()

        merge_order = []

        async def track_merge(task_id, target_branch):
            merge_order.append(task_id)
            return MergeResult(
                task_id=task_id, branch=f"task/{task_id}",
                success=True, files_merged=[],
            )

        mock_worktrees = AsyncMock(spec=WorktreeManager)
        mock_worktrees.merge = AsyncMock(side_effect=track_merge)
        mock_worktrees.cleanup_all = AsyncMock()

        synth = ResultSynthesizer(worktrees=mock_worktrees)
        await synth.synthesize(dag, results, "main")

        # t1 should be merged before t2 (dependency order)
        assert merge_order == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_synthesize_aggregates_failed_results(self):
        """Report includes failed result counts."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T1", description="D1"))
        dag.mark_completed("t1", WorkerResult(
            task_id="t1", session_id="", success=True,
            output="done", exit_code=0, duration_seconds=10.0,
        ))

        results = [
            WorkerResult(
                task_id="t1", session_id="", success=True,
                output="done", exit_code=0, duration_seconds=10.0,
                cost_usd=0.50,
            ),
            WorkerResult(
                task_id="t2", session_id="", success=False,
                output="", exit_code=1, duration_seconds=5.0,
                cost_usd=0.20, error="Broke",
            ),
        ]

        synth = ResultSynthesizer()
        report = await synth.synthesize(dag, results, "main")

        assert report.completed_tasks == 1
        assert report.failed_tasks == 1
        assert report.total_cost_usd == pytest.approx(0.70)


class TestResultSynthesizerConflictResolution:
    """Test LLM-assisted conflict resolution."""

    @pytest.mark.asyncio
    async def test_resolve_conflicts_calls_llm(self, tmp_path):
        """Conflict resolution reads conflict markers and calls LLM."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="Task One", description="First task",
        ))

        failed_merge = MergeResult(
            task_id="t1", branch="task/t1", success=False,
            conflicts=["conflict.py"],
            files_merged=["conflict.py"],
            error="Merge conflicts detected",
        )

        llm_config = LLMConfig(provider=LLMProvider.ANTHROPIC)

        # Create a real conflict file on disk
        conflict_content = "<<<<<<< HEAD\noriginal code\n=======\nnew code\n>>>>>>> task/t1\n"
        conflict_file = tmp_path / "conflict.py"
        conflict_file.write_text(conflict_content, encoding="utf-8")

        # Mock worktree manager with real repo_path
        mock_worktrees = MagicMock(spec=WorktreeManager)
        mock_worktrees.repo_path = tmp_path
        mock_worktrees._branches = {"t1": "task/t1"}

        synth = ResultSynthesizer(
            worktrees=mock_worktrees,
            llm_config=llm_config,
            auto_resolve_conflicts=True,
        )

        merge_call_count = [0]

        async def mock_exec(*args, **kwargs):
            merge_call_count[0] += 1
            proc = AsyncMock()
            if merge_call_count[0] == 1:
                # Re-attempt merge fails (expected)
                proc.communicate = AsyncMock(return_value=(b'', b''))
                proc.returncode = 1
            else:
                # git add / git commit succeed
                proc.communicate = AsyncMock(return_value=(b'', b''))
                proc.returncode = 0
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("bashgym.orchestrator.synthesizer._call_llm",
                  new_callable=AsyncMock,
                  return_value="resolved code"),
        ):
            result = await synth._resolve_conflicts(
                dag, "t1", failed_merge, "main"
            )

        assert result.success is True
        # Verify the file was written with resolved content
        assert conflict_file.read_text(encoding="utf-8") == "resolved code"

    @pytest.mark.asyncio
    async def test_resolve_conflicts_aborts_on_llm_failure(self):
        """If LLM resolution fails, merge is aborted."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="Task One", description="First task",
        ))

        failed_merge = MergeResult(
            task_id="t1", branch="task/t1", success=False,
            conflicts=["conflict.py"],
            files_merged=["conflict.py"],
        )

        llm_config = LLMConfig(provider=LLMProvider.ANTHROPIC)

        mock_worktrees = MagicMock(spec=WorktreeManager)
        mock_worktrees.repo_path = Path("/tmp/repo")
        mock_worktrees._branches = {"t1": "task/t1"}

        synth = ResultSynthesizer(
            worktrees=mock_worktrees,
            llm_config=llm_config,
        )

        async def mock_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b'', b''))
            proc.returncode = 1  # merge re-attempt fails, as expected
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("bashgym.orchestrator.synthesizer._call_llm",
                  new_callable=AsyncMock,
                  side_effect=Exception("API error")),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text",
                         return_value="<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>>"),
        ):
            result = await synth._resolve_conflicts(
                dag, "t1", failed_merge, "main"
            )

        assert result.success is False
        assert "incomplete" in result.error.lower()

    @pytest.mark.asyncio
    async def test_resolve_conflicts_no_worktrees_returns_failed(self):
        """Without worktrees, conflict resolution returns the original failure."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T1", description="D1"))

        failed_merge = MergeResult(
            task_id="t1", branch="task/t1", success=False,
            conflicts=["file.py"],
        )

        synth = ResultSynthesizer(worktrees=None, llm_config=LLMConfig())
        result = await synth._resolve_conflicts(
            dag, "t1", failed_merge, "main"
        )
        assert result.success is False


class TestConflictResolutionPrompt:
    """Test the conflict resolution prompt template."""

    def test_prompt_formats_correctly(self):
        formatted = CONFLICT_RESOLUTION_PROMPT.format(
            task_a_title="Auth Feature",
            task_a_description="Add authentication",
            task_b_title="Changes on main",
            task_b_description="Existing code",
            file_path="auth.py",
            conflict_content="<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>>",
        )
        assert "Auth Feature" in formatted
        assert "auth.py" in formatted
        assert "<<<<<<< HEAD" in formatted
        assert "Task A" in formatted
