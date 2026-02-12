"""Tests for orchestrator data models, TaskDAG, and prompts."""

import pytest
from unittest.mock import patch, AsyncMock

from bashgym.orchestrator.models import (
    LLMProvider, LLMConfig, _PROVIDER_DEFAULTS,
    TaskStatus, TaskPriority,
    OrchestratorSpec, TaskNode,
    WorkerConfig, WorkerResult, MergeResult,
)
from bashgym.orchestrator.task_dag import TaskDAG, CyclicDependencyError
from bashgym.orchestrator.prompts import (
    WORKER_SYSTEM_PROMPT, RETRY_PROMPT_TEMPLATE,
    RETRY_ANALYSIS_TEMPLATE,
)


# =============================================================================
# LLMProvider and LLMConfig
# =============================================================================

class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_all_providers_exist(self):
        """Should have all four supported providers."""
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_provider_from_string(self):
        """Should construct from string value."""
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC
        assert LLMProvider("ollama") == LLMProvider.OLLAMA


class TestLLMConfig:
    """Tests for LLMConfig provider configuration."""

    def test_default_is_anthropic(self):
        """Default config should use Anthropic Claude Opus."""
        config = LLMConfig()
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-opus-4-6"

    def test_auto_resolve_model_per_provider(self):
        """Each provider should auto-resolve its default model."""
        for provider, defaults in _PROVIDER_DEFAULTS.items():
            config = LLMConfig(provider=LLMProvider(provider))
            assert config.model == defaults["model"], (
                f"{provider} should default to {defaults['model']}"
            )

    def test_explicit_model_overrides_default(self):
        """Explicitly setting model should override provider default."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="o1")
        assert config.model == "o1"

    def test_get_base_url_from_defaults(self):
        """Should resolve base URL from provider defaults."""
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        assert config.get_base_url() == "http://localhost:11434/api/chat"

    def test_explicit_base_url_overrides(self):
        """Explicit base_url should override provider default."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            base_url="http://myserver:11434/api/chat",
        )
        assert config.get_base_url() == "http://myserver:11434/api/chat"

    def test_get_api_key_from_env(self):
        """Should resolve API key from environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}):
            config = LLMConfig(provider=LLMProvider.ANTHROPIC)
            assert config.get_api_key() == "test-key-123"

    def test_explicit_api_key_overrides_env(self):
        """Explicit api_key should override env variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            config = LLMConfig(api_key="explicit-key")
            assert config.get_api_key() == "explicit-key"

    def test_ollama_no_api_key_needed(self):
        """Ollama should return empty string for API key."""
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        assert config.get_api_key() == ""

    def test_to_dict_serialization(self):
        """Should serialize to dict correctly."""
        config = LLMConfig(
            provider=LLMProvider.GEMINI,
            temperature=0.5,
            max_tokens=2048,
        )
        d = config.to_dict()
        assert d["provider"] == "gemini"
        assert d["model"] == "gemini-2.5-pro"
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 2048


# =============================================================================
# Task Status and Priority
# =============================================================================

class TestTaskEnums:
    """Tests for TaskStatus and TaskPriority enums."""

    def test_task_status_values(self):
        """Should have all expected status values."""
        statuses = {s.value for s in TaskStatus}
        assert statuses == {
            "pending", "assigned", "running", "completed",
            "failed", "blocked", "cancelled",
        }

    def test_task_priority_ordering(self):
        """Critical should have lowest value (highest priority)."""
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value


# =============================================================================
# OrchestratorSpec
# =============================================================================

class TestOrchestratorSpec:
    """Tests for OrchestratorSpec dataclass."""

    def test_spec_defaults(self):
        """Should have sensible defaults."""
        spec = OrchestratorSpec(
            title="Test", description="Test desc"
        )
        assert spec.base_branch == "main"
        assert spec.max_budget_usd == 10.0
        assert spec.max_workers == 5
        assert spec.constraints == []

    def test_spec_to_dict(self):
        """Should serialize all fields to dict."""
        spec = OrchestratorSpec(
            title="Add auth",
            description="Implement OAuth",
            constraints=["No breaking changes"],
            max_budget_usd=5.0,
        )
        d = spec.to_dict()
        assert d["title"] == "Add auth"
        assert d["constraints"] == ["No breaking changes"]
        assert d["max_budget_usd"] == 5.0


# =============================================================================
# TaskNode
# =============================================================================

class TestTaskNode:
    """Tests for TaskNode dataclass."""

    def test_task_defaults(self):
        """Should initialize with sensible defaults."""
        task = TaskNode(id="t1", title="Test", description="desc")
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.retry_count == 0
        assert task.max_retries == 2
        assert task.dependencies == []

    def test_task_to_dict(self):
        """Should serialize to dict for API responses."""
        task = TaskNode(
            id="task_1",
            title="Implement feature",
            description="Add new feature",
            priority=TaskPriority.HIGH,
            files_touched=["src/foo.py"],
        )
        d = task.to_dict()
        assert d["id"] == "task_1"
        assert d["priority"] == 2  # HIGH = 2
        assert d["status"] == "pending"
        assert d["files_touched"] == ["src/foo.py"]


# =============================================================================
# WorkerConfig
# =============================================================================

class TestWorkerConfig:
    """Tests for WorkerConfig and CLI argument generation."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = WorkerConfig()
        assert config.max_turns == 30
        assert config.max_budget_usd == 5.0
        assert "Read" in config.allowed_tools
        assert "Bash" in config.allowed_tools

    def test_to_cli_args(self):
        """Should generate correct claude CLI arguments."""
        config = WorkerConfig(
            max_turns=15,
            max_budget_usd=2.0,
            allowed_tools=["Read", "Write"],
        )
        args = config.to_cli_args()
        assert "--output-format" in args
        assert "json" in args
        assert "--max-turns" in args
        assert "15" in args
        assert "--max-budget-usd" in args
        assert "2.0" in args
        assert "--allowedTools" in args
        assert "Read,Write" in args

    def test_to_cli_args_with_system_prompt(self):
        """Should include append-system-prompt when set."""
        config = WorkerConfig(system_prompt_append="Be focused")
        args = config.to_cli_args()
        assert "--append-system-prompt" in args
        assert "Be focused" in args

    def test_to_cli_args_with_disallowed_tools(self):
        """Should include disallowed tools flag."""
        config = WorkerConfig(disallowed_tools=["Bash"])
        args = config.to_cli_args()
        assert "--disallowedTools" in args
        assert "Bash" in args


# =============================================================================
# WorkerResult and MergeResult
# =============================================================================

class TestWorkerResult:
    """Tests for WorkerResult dataclass."""

    def test_result_to_dict_truncates_output(self):
        """Should truncate output to 2000 chars in API responses."""
        result = WorkerResult(
            task_id="t1",
            session_id="sess_1",
            success=True,
            output="x" * 5000,
            exit_code=0,
            duration_seconds=30.0,
        )
        d = result.to_dict()
        assert len(d["output"]) == 2000
        assert d["success"] is True


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_merge_result_defaults(self):
        """Should default to empty lists for conflicts/files."""
        result = MergeResult(
            task_id="t1", branch="task/t1", success=True
        )
        assert result.conflicts == []
        assert result.files_merged == []


# =============================================================================
# TaskDAG
# =============================================================================

class TestTaskDAG:
    """Tests for TaskDAG dependency resolution and scheduling."""

    @pytest.fixture
    def simple_dag(self):
        """DAG: A -> B -> C (linear chain)."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="A", title="A", description="Task A"))
        dag.add_task(TaskNode(
            id="B", title="B", description="Task B",
            dependencies=["A"],
        ))
        dag.add_task(TaskNode(
            id="C", title="C", description="Task C",
            dependencies=["B"],
        ))
        return dag

    @pytest.fixture
    def diamond_dag(self):
        """DAG: A -> B, A -> C, B -> D, C -> D (diamond)."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="A", title="A", description="Root"))
        dag.add_task(TaskNode(
            id="B", title="B", description="Left",
            dependencies=["A"],
        ))
        dag.add_task(TaskNode(
            id="C", title="C", description="Right",
            dependencies=["A"],
        ))
        dag.add_task(TaskNode(
            id="D", title="D", description="Merge",
            dependencies=["B", "C"],
        ))
        return dag

    def test_add_task(self):
        """Should add tasks to the DAG."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T", description="D"))
        assert "t1" in dag.nodes
        assert len(dag.nodes) == 1

    def test_add_duplicate_task_raises(self):
        """Should reject duplicate task IDs."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="t1", title="T", description="D"))
        with pytest.raises(ValueError, match="already exists"):
            dag.add_task(TaskNode(id="t1", title="T2", description="D2"))

    def test_get_ready_tasks_initial(self, simple_dag):
        """Only root tasks (no dependencies) should be ready initially."""
        ready = simple_dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "A"

    def test_get_ready_tasks_after_completion(self, simple_dag):
        """Completing A should make B ready."""
        result = WorkerResult(
            task_id="A", session_id="s", success=True,
            output="done", exit_code=0, duration_seconds=1.0,
        )
        simple_dag.mark_completed("A", result)
        ready = simple_dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "B"

    def test_get_ready_tasks_priority_sort(self):
        """Ready tasks should be sorted by priority (critical first)."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="low", title="L", description="D",
            priority=TaskPriority.LOW,
        ))
        dag.add_task(TaskNode(
            id="crit", title="C", description="D",
            priority=TaskPriority.CRITICAL,
        ))
        dag.add_task(TaskNode(
            id="norm", title="N", description="D",
            priority=TaskPriority.NORMAL,
        ))
        ready = dag.get_ready_tasks()
        assert [t.id for t in ready] == ["crit", "norm", "low"]

    def test_topological_sort_linear(self, simple_dag):
        """Should return A, B, C in order."""
        order = simple_dag.topological_sort()
        ids = [t.id for t in order]
        assert ids.index("A") < ids.index("B") < ids.index("C")

    def test_topological_sort_diamond(self, diamond_dag):
        """A should come first, D last in diamond DAG."""
        order = diamond_dag.topological_sort()
        ids = [t.id for t in order]
        assert ids[0] == "A"
        assert ids[-1] == "D"

    def test_cyclic_dependency_detected(self):
        """Should raise CyclicDependencyError for cycles."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="X", title="X", description="D", dependencies=["Y"]
        ))
        dag.add_task(TaskNode(
            id="Y", title="Y", description="D", dependencies=["X"]
        ))
        with pytest.raises(CyclicDependencyError):
            dag.topological_sort()

    def test_mark_completed_returns_newly_ready(self, diamond_dag):
        """Completing both B and C should make D ready."""
        result = WorkerResult(
            task_id="A", session_id="s", success=True,
            output="", exit_code=0, duration_seconds=1.0,
        )
        diamond_dag.mark_completed("A", result)

        # Complete B — D not yet ready (C still pending)
        result_b = WorkerResult(
            task_id="B", session_id="s", success=True,
            output="", exit_code=0, duration_seconds=1.0,
        )
        newly = diamond_dag.mark_completed("B", result_b)
        assert len(newly) == 0

        # Complete C — D should now be ready
        result_c = WorkerResult(
            task_id="C", session_id="s", success=True,
            output="", exit_code=0, duration_seconds=1.0,
        )
        newly = diamond_dag.mark_completed("C", result_c)
        assert len(newly) == 1
        assert newly[0].id == "D"

    def test_mark_failed_blocks_dependents(self, simple_dag):
        """Failing A should block B and C."""
        blocked = simple_dag.mark_failed("A", "crash")
        assert len(blocked) == 2
        assert simple_dag.nodes["B"].status == TaskStatus.BLOCKED
        assert simple_dag.nodes["C"].status == TaskStatus.BLOCKED

    def test_detect_file_conflicts(self):
        """Should detect overlapping file touches between parallel tasks."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="T1", description="D",
            files_touched=["src/app.py", "src/utils.py"],
        ))
        dag.add_task(TaskNode(
            id="t2", title="T2", description="D",
            files_touched=["src/app.py", "src/db.py"],
        ))
        conflicts = dag.detect_file_conflicts()
        assert len(conflicts) == 1
        assert "src/app.py" in conflicts[0][2]

    def test_no_conflicts_between_dependent_tasks(self):
        """Tasks with dependency relationship should not be flagged."""
        dag = TaskDAG()
        dag.add_task(TaskNode(
            id="t1", title="T1", description="D",
            files_touched=["src/app.py"],
        ))
        dag.add_task(TaskNode(
            id="t2", title="T2", description="D",
            files_touched=["src/app.py"],
            dependencies=["t1"],
        ))
        conflicts = dag.detect_file_conflicts()
        assert len(conflicts) == 0

    def test_is_complete(self, simple_dag):
        """Should return True only when all tasks are terminal."""
        assert not simple_dag.is_complete()

        for task in simple_dag.nodes.values():
            task.status = TaskStatus.COMPLETED
        assert simple_dag.is_complete()

    def test_is_complete_with_mixed_terminal(self):
        """Failed and cancelled are also terminal states."""
        dag = TaskDAG()
        dag.add_task(TaskNode(id="a", title="A", description="D"))
        dag.add_task(TaskNode(id="b", title="B", description="D"))

        dag.nodes["a"].status = TaskStatus.COMPLETED
        dag.nodes["b"].status = TaskStatus.FAILED
        assert dag.is_complete()

    def test_completed_tasks(self, simple_dag):
        """Should return only IDs of completed tasks."""
        simple_dag.nodes["A"].status = TaskStatus.COMPLETED
        simple_dag.nodes["B"].status = TaskStatus.FAILED
        assert simple_dag.completed_tasks() == ["A"]

    def test_stats(self, diamond_dag):
        """Should return counts by status."""
        diamond_dag.nodes["A"].status = TaskStatus.COMPLETED
        diamond_dag.nodes["B"].status = TaskStatus.RUNNING
        stats = diamond_dag.stats
        assert stats["completed"] == 1
        assert stats["running"] == 1
        assert stats["pending"] == 2

    def test_to_dict(self, simple_dag):
        """Should serialize DAG with tasks, stats, and conflicts."""
        d = simple_dag.to_dict()
        assert "tasks" in d
        assert "stats" in d
        assert "file_conflicts" in d
        assert len(d["tasks"]) == 3

    def test_critical_path_linear(self, simple_dag):
        """Critical path of a linear DAG should include all tasks."""
        path = simple_dag.get_critical_path()
        ids = [t.id for t in path]
        assert "A" in ids
        assert "C" in ids

    def test_mark_completed_unknown_task_raises(self):
        """Should raise ValueError for unknown task ID."""
        dag = TaskDAG()
        result = WorkerResult(
            task_id="x", session_id="s", success=True,
            output="", exit_code=0, duration_seconds=1.0,
        )
        with pytest.raises(ValueError, match="not found"):
            dag.mark_completed("x", result)


# =============================================================================
# TaskDAG.from_spec (mocked LLM)
# =============================================================================

class TestTaskDAGFromSpec:
    """Tests for spec decomposition via LLM."""

    @pytest.mark.asyncio
    async def test_from_spec_parses_json_response(self):
        """Should parse LLM JSON response into TaskDAG."""
        spec = OrchestratorSpec(
            title="Test", description="Test spec"
        )
        llm_config = LLMConfig()

        mock_response = """[
            {
                "id": "task_1",
                "title": "Setup",
                "description": "Initial setup",
                "priority": "critical",
                "dependencies": [],
                "files_touched": ["setup.py"],
                "estimated_turns": 10,
                "budget_usd": 1.0
            },
            {
                "id": "task_2",
                "title": "Implement",
                "description": "Main work",
                "priority": "high",
                "dependencies": ["task_1"],
                "files_touched": ["src/main.py"],
                "estimated_turns": 25,
                "budget_usd": 3.0
            }
        ]"""

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            dag = await TaskDAG.from_spec(spec, llm_config)

        assert len(dag.nodes) == 2
        assert dag.nodes["task_1"].priority == TaskPriority.CRITICAL
        assert dag.nodes["task_2"].dependencies == ["task_1"]
        assert dag.nodes["task_2"].estimated_turns == 25

    @pytest.mark.asyncio
    async def test_from_spec_handles_json_in_object(self):
        """Should extract tasks from a JSON object with 'tasks' key."""
        spec = OrchestratorSpec(title="T", description="D")
        llm_config = LLMConfig()

        mock_response = """Here's the decomposition:
        {
            "tasks": [
                {"id": "t1", "title": "Do it", "description": "D"}
            ]
        }
        """

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            dag = await TaskDAG.from_spec(spec, llm_config)

        assert len(dag.nodes) == 1
        assert "t1" in dag.nodes

    @pytest.mark.asyncio
    async def test_from_spec_handles_bad_json(self):
        """Should return empty DAG on unparseable response."""
        spec = OrchestratorSpec(title="T", description="D")
        llm_config = LLMConfig()

        with patch(
            "bashgym.orchestrator.task_dag._call_llm",
            new_callable=AsyncMock,
            return_value="Sorry, I can't do that.",
        ):
            dag = await TaskDAG.from_spec(spec, llm_config)

        assert len(dag.nodes) == 0


# =============================================================================
# Prompts
# =============================================================================

class TestPrompts:
    """Tests for prompt templates."""

    def test_worker_system_prompt_not_empty(self):
        """Worker system prompt should be substantive."""
        assert len(WORKER_SYSTEM_PROMPT) > 100
        assert "task" in WORKER_SYSTEM_PROMPT.lower()

    def test_retry_template_formatting(self):
        """Retry template should accept all required format keys."""
        result = RETRY_PROMPT_TEMPLATE.format(
            error="ImportError: no module named foo",
            previous_output="Traceback...",
            original_prompt="Fix the import",
        )
        assert "ImportError" in result
        assert "Fix the import" in result

    def test_retry_analysis_template_formatting(self):
        """Retry analysis template should accept all required format keys."""
        result = RETRY_ANALYSIS_TEMPLATE.format(
            task_title="Fix auth",
            original_prompt="Implement OAuth",
            error="timeout",
            previous_output="partial output",
            attempt=1,
            max_attempts=2,
        )
        assert "Fix auth" in result
        assert "timeout" in result
        assert "1 of 2" in result
