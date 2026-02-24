"""Shared fixtures for orchestrator e2e tests.

Provides real git repos (via tmp_path), mock LLM responses,
mock worker processes, and pre-built DAG structures.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bashgym.orchestrator.models import (
    LLMConfig,
    LLMProvider,
    MergeResult,
    OrchestratorSpec,
    TaskNode,
    TaskPriority,
    TaskStatus,
    WorkerConfig,
    WorkerResult,
)
from bashgym.orchestrator.task_dag import TaskDAG
from bashgym.orchestrator.worktree import WorktreeManager


# =============================================================================
# Real git repo fixtures
# =============================================================================


@pytest.fixture
def real_git_repo(tmp_path):
    """Create a real git repository in tmp_path with an initial commit.

    Contains:
    - README.md
    - src/main.py

    Configures git user.name/email for test commits.
    Returns Path to the repo root.
    """
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Helper to run git synchronously
    def git(*args, cwd=None):
        proc = asyncio.get_event_loop().run_until_complete(
            asyncio.create_subprocess_exec(
                "git", *args,
                cwd=str(cwd or repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        out, err = asyncio.get_event_loop().run_until_complete(proc.communicate())
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {err.decode()}"
            )
        return out.decode()

    git("init", "-b", "main")
    git("config", "user.name", "Test User")
    git("config", "user.email", "test@example.com")

    # Create initial files
    readme = repo / "README.md"
    readme.write_text("# Test Repo\n\nA test repository for e2e tests.\n")

    src = repo / "src"
    src.mkdir()
    main_py = src / "main.py"
    main_py.write_text('def main():\n    print("hello")\n')

    git("add", "-A")
    git("commit", "-m", "Initial commit")

    return repo


@pytest.fixture
async def async_git_repo(tmp_path):
    """Async version of real_git_repo for async test functions.

    Same structure as real_git_repo but uses await for git commands.
    """
    repo = tmp_path / "test_repo"
    repo.mkdir()

    async def git(*args, cwd=None):
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(cwd or repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {err.decode()}"
            )
        return out.decode()

    await git("init", "-b", "main")
    await git("config", "user.name", "Test User")
    await git("config", "user.email", "test@example.com")

    readme = repo / "README.md"
    readme.write_text("# Test Repo\n\nA test repository for e2e tests.\n")

    src = repo / "src"
    src.mkdir()
    main_py = src / "main.py"
    main_py.write_text('def main():\n    print("hello")\n')

    await git("add", "-A")
    await git("commit", "-m", "Initial commit")

    repo._git = git  # Attach helper for use in tests
    return repo


# =============================================================================
# Spec fixtures
# =============================================================================


@pytest.fixture
def sample_spec():
    """A representative OrchestratorSpec for testing."""
    return OrchestratorSpec(
        title="Add user authentication",
        description="Implement JWT-based auth with login, register, and token refresh endpoints.",
        constraints=[
            "Use bcrypt for password hashing",
            "Tokens expire after 1 hour",
        ],
        acceptance_criteria=[
            "POST /auth/login returns JWT on valid credentials",
            "Protected routes return 401 without valid token",
        ],
        max_budget_usd=5.0,
        max_workers=3,
    )


# =============================================================================
# Mock LLM fixtures
# =============================================================================

THREE_TASK_DECOMPOSITION = {
    "tasks": [
        {
            "id": "task_1",
            "title": "Create data models",
            "description": "Create User and Token SQLAlchemy models in src/models.py",
            "priority": "normal",
            "dependencies": [],
            "files_touched": ["src/models.py"],
            "estimated_turns": 10,
            "budget_usd": 1.0,
        },
        {
            "id": "task_2",
            "title": "Add API endpoints",
            "description": "Add login, register, and refresh endpoints in src/api.py",
            "priority": "normal",
            "dependencies": ["task_1"],
            "files_touched": ["src/api.py"],
            "estimated_turns": 15,
            "budget_usd": 2.0,
        },
        {
            "id": "task_3",
            "title": "Write tests",
            "description": "Write pytest tests for auth endpoints in tests/test_models.py",
            "priority": "low",
            "dependencies": ["task_1"],
            "files_touched": ["tests/test_models.py"],
            "estimated_turns": 10,
            "budget_usd": 1.0,
        },
    ]
}


@pytest.fixture
def mock_llm_decomposition():
    """Patch _call_llm to return a 3-task decomposition JSON."""
    response_text = json.dumps(THREE_TASK_DECOMPOSITION)

    with patch(
        "bashgym.orchestrator.task_dag._call_llm",
        new_callable=AsyncMock,
        return_value=response_text,
    ) as mock_llm:
        yield mock_llm


@pytest.fixture
def mock_llm_all():
    """Patch _call_llm everywhere it's imported (task_dag + synthesizer + agent)."""
    response_text = json.dumps(THREE_TASK_DECOMPOSITION)

    with patch(
        "bashgym.orchestrator.task_dag._call_llm",
        new_callable=AsyncMock,
        return_value=response_text,
    ) as mock_dag_llm, patch(
        "bashgym.orchestrator.synthesizer._call_llm",
        new_callable=AsyncMock,
        return_value="resolved content",
    ) as mock_synth_llm:
        yield mock_dag_llm, mock_synth_llm


# =============================================================================
# Mock worker fixtures
# =============================================================================


def _make_mock_process(returncode=0, stdout_data=None, stderr_data=b""):
    """Create a mock asyncio.subprocess.Process."""
    if stdout_data is None:
        stdout_data = json.dumps({
            "result": "Task completed successfully",
            "session_id": "test-session-001",
            "cost_usd": 0.05,
            "usage": {"total_tokens": 1500},
        }).encode()

    mock_proc = AsyncMock()
    mock_proc.returncode = returncode
    mock_proc.communicate = AsyncMock(return_value=(stdout_data, stderr_data))
    mock_proc.kill = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.pid = 12345
    return mock_proc


@pytest.fixture
def mock_worker_success():
    """Patch create_subprocess_exec to simulate a successful worker."""
    mock_proc = _make_mock_process(returncode=0)

    with patch(
        "asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_exec:
        yield mock_exec


@pytest.fixture
def mock_worker_failure():
    """Patch create_subprocess_exec to simulate a failed worker."""
    mock_proc = _make_mock_process(
        returncode=1,
        stdout_data=b"",
        stderr_data=b"Error: compilation failed at line 42",
    )

    with patch(
        "asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_exec:
        yield mock_exec


# =============================================================================
# Pre-built DAG fixtures
# =============================================================================


def make_task(
    task_id: str,
    title: str = "",
    deps: list = None,
    files: list = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    budget: float = 2.0,
    status: TaskStatus = TaskStatus.PENDING,
) -> TaskNode:
    """Helper to create a TaskNode with sensible defaults."""
    return TaskNode(
        id=task_id,
        title=title or f"Task {task_id}",
        description=f"Description for {task_id}",
        priority=priority,
        status=status,
        dependencies=deps or [],
        files_touched=files or [],
        estimated_turns=10,
        budget_usd=budget,
    )


@pytest.fixture
def three_task_dag():
    """Pre-built 3-task DAG:

    task_1 (no deps) → task_2 (depends on task_1)
                      → task_3 (depends on task_1)
    """
    dag = TaskDAG()
    dag.add_task(make_task("task_1", "Create data models", files=["src/models.py"]))
    dag.add_task(make_task("task_2", "Add API endpoints", deps=["task_1"], files=["src/api.py"]))
    dag.add_task(make_task(
        "task_3", "Write tests", deps=["task_1"],
        files=["tests/test_models.py"], priority=TaskPriority.LOW,
    ))
    return dag


@pytest.fixture
def diamond_dag():
    """Diamond-shaped DAG:

    A → B → D
    A → C → D
    """
    dag = TaskDAG()
    dag.add_task(make_task("A", "Root task"))
    dag.add_task(make_task("B", "Left branch", deps=["A"]))
    dag.add_task(make_task("C", "Right branch", deps=["A"]))
    dag.add_task(make_task("D", "Merge point", deps=["B", "C"]))
    return dag


@pytest.fixture
def linear_dag():
    """Linear chain: A → B → C"""
    dag = TaskDAG()
    dag.add_task(make_task("A", "First task"))
    dag.add_task(make_task("B", "Second task", deps=["A"]))
    dag.add_task(make_task("C", "Third task", deps=["B"]))
    return dag


@pytest.fixture
def parallel_dag():
    """Two independent tasks: A and B (no dependencies)."""
    dag = TaskDAG()
    dag.add_task(make_task("A", "Independent task A"))
    dag.add_task(make_task("B", "Independent task B"))
    return dag


# =============================================================================
# WorkerResult factory
# =============================================================================


def make_result(
    task_id: str,
    success: bool = True,
    cost: float = 0.50,
    duration: float = 10.0,
    files: list = None,
    error: str = None,
) -> WorkerResult:
    """Helper to create a WorkerResult."""
    return WorkerResult(
        task_id=task_id,
        session_id=f"session-{task_id}",
        success=success,
        output=f"Output for {task_id}" if success else "",
        exit_code=0 if success else 1,
        duration_seconds=duration,
        tokens_used=1500,
        cost_usd=cost,
        files_modified=files or [],
        error=error,
    )
