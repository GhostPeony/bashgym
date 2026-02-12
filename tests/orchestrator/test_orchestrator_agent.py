# tests/orchestrator/test_orchestrator_agent.py
"""Tests for OrchestrationAgent and API routes."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from bashgym.orchestrator.models import (
    OrchestratorSpec, TaskNode, TaskStatus, TaskPriority,
    WorkerConfig, WorkerResult, MergeResult,
    LLMConfig, LLMProvider,
)
from bashgym.orchestrator.task_dag import TaskDAG
from bashgym.orchestrator.agent import OrchestrationAgent
from bashgym.orchestrator.synthesizer import SynthesisReport


# =============================================================================
# OrchestrationAgent Initialization
# =============================================================================


class TestOrchestrationAgentInit:
    """Test OrchestrationAgent initialization."""

    def test_default_init(self):
        agent = OrchestrationAgent()
        assert agent.llm_config.provider == LLMProvider.ANTHROPIC
        assert agent.pool.max_workers == 5
        assert agent.worktrees is None
        assert agent.dag is None
        assert agent._total_spent_usd == 0.0
        assert agent._budget_exceeded is False

    def test_init_with_custom_llm(self):
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        agent = OrchestrationAgent(llm_config=config)
        assert agent.llm_config.provider == LLMProvider.OPENAI
        assert agent.llm_config.model == "gpt-4o"

    def test_init_with_repo_creates_worktrees(self, tmp_path):
        agent = OrchestrationAgent(repo_path=tmp_path, use_worktrees=True)
        assert agent.worktrees is not None
        assert agent.worktrees.repo_path == tmp_path

    def test_init_without_worktrees(self, tmp_path):
        agent = OrchestrationAgent(repo_path=tmp_path, use_worktrees=False)
        assert agent.worktrees is None

    def test_init_with_job_id(self):
        agent = OrchestrationAgent(job_id="test-123")
        assert agent.job_id == "test-123"

    def test_init_with_model_router(self):
        mock_router = MagicMock()
        agent = OrchestrationAgent(model_router=mock_router)
        assert agent.model_router is mock_router

    def test_init_with_callbacks(self):
        cb_start = AsyncMock()
        cb_complete = AsyncMock()
        cb_fail = AsyncMock()
        agent = OrchestrationAgent(
            on_task_started=cb_start,
            on_task_completed=cb_complete,
            on_task_failed=cb_fail,
        )
        assert agent._on_task_started is cb_start
        assert agent._on_task_completed is cb_complete
        assert agent._on_task_failed is cb_fail


# =============================================================================
# Budget Status
# =============================================================================


class TestBudgetStatus:
    """Test budget tracking property."""

    def test_budget_status_no_limit(self):
        agent = OrchestrationAgent()
        status = agent.budget_status
        assert status["spent_usd"] == 0.0
        assert status["limit_usd"] == 0.0
        assert status["remaining_usd"] is None
        assert status["exceeded"] is False

    def test_budget_status_with_spending(self):
        agent = OrchestrationAgent()
        agent._total_spent_usd = 3.5
        agent._budget_limit_usd = 10.0
        status = agent.budget_status
        assert status["spent_usd"] == 3.5
        assert status["limit_usd"] == 10.0
        assert status["remaining_usd"] == pytest.approx(6.5)
        assert status["exceeded"] is False

    def test_budget_status_exceeded(self):
        agent = OrchestrationAgent()
        agent._total_spent_usd = 12.0
        agent._budget_limit_usd = 10.0
        agent._budget_exceeded = True
        status = agent.budget_status
        assert status["exceeded"] is True
        assert status["remaining_usd"] == 0.0


# =============================================================================
# Submit Spec (Phase 1)
# =============================================================================


class TestSubmitSpec:
    """Test spec decomposition."""

    @pytest.mark.asyncio
    async def test_submit_spec_calls_from_spec(self):
        agent = OrchestrationAgent()
        spec = OrchestratorSpec(
            title="Test Spec",
            description="Build a thing",
        )

        mock_dag = TaskDAG()
        mock_dag.add_task(TaskNode(id="t1", title="T1", description="D1"))

        with patch.object(
            TaskDAG, "from_spec",
            new_callable=AsyncMock,
            return_value=mock_dag,
        ):
            dag = await agent.submit_spec(spec)

        assert dag is mock_dag
        assert agent.dag is mock_dag
        assert len(dag.nodes) == 1

    @pytest.mark.asyncio
    async def test_submit_spec_uses_agent_llm_config(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        agent = OrchestrationAgent(llm_config=config)
        spec = OrchestratorSpec(title="Test", description="Test")

        captured_config = None

        async def mock_from_spec(spec_arg, llm_config):
            nonlocal captured_config
            captured_config = llm_config
            dag = TaskDAG()
            dag.add_task(TaskNode(id="t1", title="T1", description="D1"))
            return dag

        with patch.object(TaskDAG, "from_spec", side_effect=mock_from_spec):
            await agent.submit_spec(spec)

        assert captured_config.provider == LLMProvider.OLLAMA
        assert captured_config.model == "llama3"


# =============================================================================
# Execute (Phase 2-4)
# =============================================================================


class TestExecute:
    """Test DAG execution."""

    @pytest.mark.asyncio
    async def test_execute_requires_dag(self):
        agent = OrchestrationAgent()
        with pytest.raises(ValueError, match="No DAG"):
            await agent.execute()

    @pytest.mark.asyncio
    async def test_execute_single_task_success(self):
        """Execute a simple 1-task DAG where the worker succeeds."""
        agent = OrchestrationAgent()

        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T1", description="Do it"))

        result = WorkerResult(
            task_id="t1", session_id="s1", success=True,
            output="done", exit_code=0, duration_seconds=5.0,
            cost_usd=0.25,
        )

        # Mock pool to return our result immediately
        agent.pool = MagicMock()
        agent.pool.available_slots = 5
        agent.pool.active_count = 1  # Looks like we have a worker
        agent.pool.spawn_worker = AsyncMock()
        agent.pool.wait_for_any = AsyncMock(return_value=result)
        agent.pool.cancel_all = AsyncMock()

        # After wait_for_any returns, active_count goes to 0
        type(agent.pool).active_count = PropertyMock(
            side_effect=[1, 0, 0, 0]
        )

        with patch.object(
            agent, "_broadcast_task_started", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_task_completed", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_budget_update", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_complete", new_callable=AsyncMock
        ):
            results = await agent.execute(dag=dag)

        assert len(results) == 1
        assert results[0].success is True
        assert agent._total_spent_usd == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_execute_with_budget_limit(self):
        """Budget limit is tracked during execution."""
        agent = OrchestrationAgent()
        agent._budget_limit_usd = 0.0  # Will be set by execute

        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T1", description="D1"))

        result = WorkerResult(
            task_id="t1", session_id="", success=True,
            output="done", exit_code=0, duration_seconds=5.0,
            cost_usd=5.0,
        )

        agent.pool = MagicMock()
        agent.pool.available_slots = 5
        agent.pool.spawn_worker = AsyncMock()
        agent.pool.wait_for_any = AsyncMock(return_value=result)
        agent.pool.cancel_all = AsyncMock()
        type(agent.pool).active_count = PropertyMock(
            side_effect=[1, 0, 0, 0]
        )

        with patch.object(
            agent, "_broadcast_task_started", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_task_completed", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_budget_update", new_callable=AsyncMock
        ), patch.object(
            agent, "_broadcast_complete", new_callable=AsyncMock
        ):
            results = await agent.execute(dag=dag, budget_usd=1.0)

        # Budget exceeded: 5.0 > 1.0
        assert agent._budget_exceeded is True
        assert agent._total_spent_usd == pytest.approx(5.0)


# =============================================================================
# Failure Handling
# =============================================================================


class TestHandleFailure:
    """Test task failure and retry handling."""

    @pytest.mark.asyncio
    async def test_handle_failure_retries(self):
        agent = OrchestrationAgent()
        dag = TaskDAG()
        task = TaskNode(
            id="t1", title="T1", description="D1",
            max_retries=2, retry_count=0,
        )
        dag.add_task(task)

        result = WorkerResult(
            task_id="t1", session_id="", success=False,
            output="error output", exit_code=1, duration_seconds=5.0,
            error="Something broke",
        )

        with patch.object(
            agent, "_rewrite_prompt_for_retry",
            new_callable=AsyncMock,
            return_value="Try again with fix",
        ), patch.object(
            agent, "_broadcast_task_failed", new_callable=AsyncMock
        ):
            await agent._handle_failure(dag, result)

        assert task.retry_count == 1
        assert task.status == TaskStatus.PENDING
        assert task.worker_prompt == "Try again with fix"

    @pytest.mark.asyncio
    async def test_handle_failure_exhausts_retries(self):
        agent = OrchestrationAgent()
        dag = TaskDAG()
        task = TaskNode(
            id="t1", title="T1", description="D1",
            max_retries=1, retry_count=1,  # Already at max
            dependencies=[],
        )
        dag.add_task(task)

        # Add a dependent task
        task2 = TaskNode(
            id="t2", title="T2", description="D2",
            dependencies=["t1"],
        )
        dag.add_task(task2)

        result = WorkerResult(
            task_id="t1", session_id="", success=False,
            output="", exit_code=1, duration_seconds=5.0,
            error="Still broken",
        )

        with patch.object(
            agent, "_broadcast_task_failed", new_callable=AsyncMock
        ):
            await agent._handle_failure(dag, result)

        assert task.status == TaskStatus.FAILED
        assert task2.status == TaskStatus.BLOCKED


# =============================================================================
# Prompt Rewriting
# =============================================================================


class TestRewritePrompt:
    """Test LLM-assisted retry prompt rewriting."""

    @pytest.mark.asyncio
    async def test_rewrite_uses_llm_when_available(self):
        agent = OrchestrationAgent()
        task = TaskNode(
            id="t1", title="Test Task", description="Original prompt",
            retry_count=1, max_retries=3,
        )
        result = WorkerResult(
            task_id="t1", session_id="", success=False,
            output="error trace here", exit_code=1, duration_seconds=5.0,
            error="FileNotFoundError: config.json",
        )

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            return_value="Improved prompt: Create config.json first, then proceed.",
        ):
            new_prompt = await agent._rewrite_prompt_for_retry(task, result)

        assert "Improved prompt" in new_prompt
        assert "config.json" in new_prompt

    @pytest.mark.asyncio
    async def test_rewrite_falls_back_to_template(self):
        agent = OrchestrationAgent()
        task = TaskNode(
            id="t1", title="Test Task", description="Original prompt",
            retry_count=1, max_retries=3,
        )
        result = WorkerResult(
            task_id="t1", session_id="", success=False,
            output="some output", exit_code=1, duration_seconds=5.0,
            error="Some error",
        )

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            side_effect=Exception("API unavailable"),
        ):
            new_prompt = await agent._rewrite_prompt_for_retry(task, result)

        # Falls back to RETRY_PROMPT_TEMPLATE
        assert "Some error" in new_prompt
        assert "Original prompt" in new_prompt

    @pytest.mark.asyncio
    async def test_rewrite_rejects_too_short_llm_response(self):
        """If LLM returns garbage (< 20 chars), use template fallback."""
        agent = OrchestrationAgent()
        task = TaskNode(
            id="t1", title="Test Task", description="Do the thing",
            retry_count=1, max_retries=3,
        )
        result = WorkerResult(
            task_id="t1", session_id="", success=False,
            output="output", exit_code=1, duration_seconds=5.0,
            error="Error",
        )

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            return_value="OK",  # Too short
        ):
            new_prompt = await agent._rewrite_prompt_for_retry(task, result)

        # Should fall back to template
        assert "Do the thing" in new_prompt


# =============================================================================
# Student Model Routing
# =============================================================================


class TestStudentRouting:
    """Test confidence-based student model routing."""

    def test_no_router_returns_false(self):
        agent = OrchestrationAgent(model_router=None)
        task = TaskNode(
            id="t1", title="T1", description="D1",
            priority=TaskPriority.LOW,
        )
        assert agent._should_route_to_student(task) is False

    def test_high_priority_never_routes_to_student(self):
        mock_router = MagicMock()
        agent = OrchestrationAgent(model_router=mock_router)

        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            task = TaskNode(
                id="t1", title="T1", description="D1",
                priority=priority,
            )
            assert agent._should_route_to_student(task) is False

    def test_low_priority_queries_router(self):
        mock_router = MagicMock()
        mock_router.get_student_model.return_value = "student-model-v1"

        # Create a mock ModelType enum value
        mock_student_type = MagicMock()

        mock_decision = MagicMock()
        mock_decision.model_type = mock_student_type
        mock_router.route.return_value = mock_decision

        agent = OrchestrationAgent(model_router=mock_router)
        task = TaskNode(
            id="t1", title="T1", description="D1",
            priority=TaskPriority.LOW,
        )

        # Patch ModelType.STUDENT at the import location inside the method
        with patch("bashgym.gym.router.ModelType") as MockModelType:
            MockModelType.STUDENT = mock_student_type
            result = agent._should_route_to_student(task)

        assert result is True
        mock_router.route.assert_called_once()

    def test_router_exception_returns_false(self):
        mock_router = MagicMock()
        mock_router.get_student_model.side_effect = Exception("Router error")

        agent = OrchestrationAgent(model_router=mock_router)
        task = TaskNode(
            id="t1", title="T1", description="D1",
            priority=TaskPriority.LOW,
        )

        assert agent._should_route_to_student(task) is False


# =============================================================================
# Complexity Estimation
# =============================================================================


class TestEstimateComplexity:
    """Test task complexity estimation."""

    def test_simple_task(self):
        task = TaskNode(
            id="t1", title="T1", description="D1",
            files_touched=["a.py"],
            estimated_turns=5,
        )
        score = OrchestrationAgent._estimate_complexity(task)
        assert score == pytest.approx(0.2)  # 0.1 (1 file) + 0.1 (<=10 turns)

    def test_moderate_task(self):
        task = TaskNode(
            id="t1", title="T1", description="D1",
            files_touched=["a.py", "b.py", "c.py"],
            estimated_turns=20,
        )
        score = OrchestrationAgent._estimate_complexity(task)
        assert score == pytest.approx(0.6)  # 0.3 (3 files) + 0.3 (20 turns)

    def test_complex_task(self):
        task = TaskNode(
            id="t1", title="T1", description="D1",
            files_touched=["a.py", "b.py", "c.py", "d.py", "e.py"],
            estimated_turns=50,
        )
        score = OrchestrationAgent._estimate_complexity(task)
        assert score == pytest.approx(1.0)  # 0.6 (5 files) + 0.4 (50 turns), capped at 1.0

    def test_no_files_few_turns(self):
        task = TaskNode(
            id="t1", title="T1", description="D1",
            files_touched=[],
            estimated_turns=3,
        )
        score = OrchestrationAgent._estimate_complexity(task)
        assert score == pytest.approx(0.2)  # 0.1 (<=1 file) + 0.1 (<=10 turns)


# =============================================================================
# Cancel Remaining
# =============================================================================


class TestCancelRemaining:
    """Test cancellation of remaining tasks."""

    @pytest.mark.asyncio
    async def test_cancel_remaining_cancels_pool(self):
        agent = OrchestrationAgent()
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="T1", description="D1",
            status=TaskStatus.PENDING,
        ))
        dag.add_task(TaskNode(
            id="t2", title="T2", description="D2",
            status=TaskStatus.BLOCKED,
        ))

        # Need to set status after add_task since add_task resets it
        dag.nodes["t1"].status = TaskStatus.PENDING
        dag.nodes["t2"].status = TaskStatus.BLOCKED

        agent.pool = MagicMock()
        agent.pool.cancel_all = AsyncMock()

        await agent._cancel_remaining(dag)

        agent.pool.cancel_all.assert_called_once()
        assert dag.nodes["t1"].status == TaskStatus.CANCELLED
        assert dag.nodes["t2"].status == TaskStatus.CANCELLED


# =============================================================================
# WebSocket Broadcasting
# =============================================================================


class TestBroadcasting:
    """Test WebSocket broadcasting (silent failure in CLI mode)."""

    @pytest.mark.asyncio
    async def test_broadcast_task_started_silent_on_import_error(self):
        """Broadcasting should not raise even if websocket module unavailable."""
        agent = OrchestrationAgent()
        task = TaskNode(id="t1", title="T1", description="D1")

        with patch(
            "bashgym.api.websocket.broadcast_orchestration_task_started",
            side_effect=ImportError("no websocket"),
        ):
            # Should not raise
            await agent._broadcast_task_started(task)

    @pytest.mark.asyncio
    async def test_broadcast_task_completed_silent(self):
        agent = OrchestrationAgent()
        result = WorkerResult(
            task_id="t1", session_id="", success=True,
            output="", exit_code=0, duration_seconds=5.0,
            cost_usd=0.5,
        )
        # Should not raise even if broadcast fails
        await agent._broadcast_task_completed(result, newly_unblocked=2)

    @pytest.mark.asyncio
    async def test_broadcast_budget_skipped_without_limit(self):
        agent = OrchestrationAgent()
        agent._budget_limit_usd = 0.0  # No limit
        dag = TaskDAG()

        # Should be a no-op when no budget limit
        await agent._broadcast_budget_update(dag, [])


# =============================================================================
# Trace Ingestion
# =============================================================================


class TestTraceIngestion:
    """Test trace ingestion for training pipeline."""

    @pytest.mark.asyncio
    async def test_ingest_traces_no_importer_returns_zero(self):
        agent = OrchestrationAgent()
        results = [
            WorkerResult(
                task_id="t1", session_id="s1", success=True,
                output="done", exit_code=0, duration_seconds=5.0,
            ),
        ]

        with patch.dict(
            "sys.modules",
            {"bashgym.trace_capture.importers": None},
        ):
            # The code does `from bashgym.trace_capture.importers import ...`
            # which will raise ImportError when module is None
            count = await agent.ingest_traces(results)

        assert count == 0

    @pytest.mark.asyncio
    async def test_ingest_skips_failed_results(self):
        agent = OrchestrationAgent()

        results = [
            WorkerResult(
                task_id="t1", session_id="s1", success=False,
                output="error", exit_code=1, duration_seconds=5.0,
            ),
            WorkerResult(
                task_id="t2", session_id="", success=True,  # No session_id
                output="done", exit_code=0, duration_seconds=5.0,
            ),
        ]

        mock_importer_cls = MagicMock()
        mock_importer = MagicMock()
        mock_importer_cls.return_value = mock_importer

        mock_module = MagicMock()
        mock_module.ClaudeSessionImporter = mock_importer_cls

        with patch.dict(
            "sys.modules",
            {"bashgym.trace_capture.importers": mock_module},
        ):
            count = await agent.ingest_traces(results)

        assert count == 0  # Both skipped: one failed, one no session_id


# =============================================================================
# API Routes Tests
# =============================================================================


class TestOrchestratorRoutes:
    """Test API route request/response models."""

    def test_llm_config_request_defaults(self):
        from bashgym.api.orchestrator_routes import LLMConfigRequest
        req = LLMConfigRequest()
        assert req.provider == "anthropic"
        assert req.model is None
        assert req.temperature == 0.3

    def test_spec_request_defaults(self):
        from bashgym.api.orchestrator_routes import SpecRequest
        req = SpecRequest(title="Test", description="Do something")
        assert req.title == "Test"
        assert req.base_branch == "main"
        assert req.max_budget_usd == 10.0
        assert req.max_workers == 5
        assert req.constraints == []

    def test_approve_request_defaults(self):
        from bashgym.api.orchestrator_routes import ApproveRequest
        req = ApproveRequest()
        assert req.base_branch == "main"

    def test_retry_request_defaults(self):
        from bashgym.api.orchestrator_routes import RetryRequest
        req = RetryRequest()
        assert req.modified_prompt is None


class TestBuildLLMConfig:
    """Test the _build_llm_config helper."""

    def test_build_default_config(self):
        from bashgym.api.orchestrator_routes import _build_llm_config
        config = _build_llm_config(None)
        assert config.provider == LLMProvider.ANTHROPIC

    def test_build_openai_config(self):
        from bashgym.api.orchestrator_routes import (
            _build_llm_config, LLMConfigRequest,
        )
        req = LLMConfigRequest(provider="openai", model="gpt-4o")
        config = _build_llm_config(req)
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4o"

    def test_build_ollama_config(self):
        from bashgym.api.orchestrator_routes import (
            _build_llm_config, LLMConfigRequest,
        )
        req = LLMConfigRequest(
            provider="ollama",
            model="qwen2.5-coder:32b",
            base_url="http://localhost:11434/api/chat",
        )
        config = _build_llm_config(req)
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "qwen2.5-coder:32b"
        assert config.base_url == "http://localhost:11434/api/chat"

    def test_build_unknown_provider_raises(self):
        from bashgym.api.orchestrator_routes import (
            _build_llm_config, LLMConfigRequest,
        )
        req = LLMConfigRequest(provider="unknown-ai")
        with pytest.raises(ValueError, match="Unknown provider"):
            _build_llm_config(req)

    def test_build_gemini_config(self):
        from bashgym.api.orchestrator_routes import (
            _build_llm_config, LLMConfigRequest,
        )
        req = LLMConfigRequest(provider="gemini", api_key="test-key")
        config = _build_llm_config(req)
        assert config.provider == LLMProvider.GEMINI
        assert config.api_key == "test-key"

    def test_build_config_with_empty_model_auto_resolves(self):
        from bashgym.api.orchestrator_routes import (
            _build_llm_config, LLMConfigRequest,
        )
        req = LLMConfigRequest(provider="openai")
        config = _build_llm_config(req)
        # model=None in request -> model="" in LLMConfig -> auto-resolve to gpt-4o
        assert config.model == "gpt-4o"
