"""Tests for WorkerContextBuilder — inter-agent context awareness."""

import pytest
from pathlib import Path
from unittest.mock import patch

from bashgym.orchestrator.models import (
    TaskNode, TaskStatus, TaskPriority, WorkerResult,
)
from bashgym.orchestrator.context_builder import (
    WorkerContextBuilder, ORCHESTRATION_SEPARATOR,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_result(task_id, files_modified=None):
    """Helper to create a WorkerResult."""
    return WorkerResult(
        task_id=task_id,
        session_id=f"sess_{task_id}",
        success=True,
        output="done",
        exit_code=0,
        duration_seconds=30.0,
        files_modified=files_modified or [],
    )


@pytest.fixture
def three_task_dag():
    """DAG with three tasks: models (root), auth (depends on models), api (depends on models).

    models ---> auth
       \\-----> api
    auth and api are parallel siblings.
    """
    nodes = {
        "task_1": TaskNode(
            id="task_1",
            title="Database models",
            description="Create User and Session models",
            priority=TaskPriority.CRITICAL,
            files_touched=["src/models.py", "tests/test_models.py"],
            provides=[
                {"file": "src/models.py", "exports": ["User", "Session", "get_db"], "description": "Core models"},
            ],
        ),
        "task_2": TaskNode(
            id="task_2",
            title="Implement authentication",
            description="Add auth middleware",
            priority=TaskPriority.HIGH,
            dependencies=["task_1"],
            files_touched=["src/auth.py", "tests/test_auth.py"],
            provides=[
                {"file": "src/auth.py", "exports": ["authenticate", "AuthMiddleware"], "description": "Auth functions"},
            ],
            consumes=[
                {"from_task": "task_1", "file": "src/models.py", "imports": ["User", "get_db"], "description": "User model"},
            ],
        ),
        "task_3": TaskNode(
            id="task_3",
            title="API endpoints",
            description="REST API for users",
            priority=TaskPriority.NORMAL,
            dependencies=["task_1"],
            files_touched=["src/api.py", "tests/test_api.py"],
            consumes=[
                {"from_task": "task_1", "file": "src/models.py", "imports": ["User"], "description": "User model"},
                {"from_task": "task_2", "file": "src/auth.py", "imports": ["authenticate"], "description": "Auth check"},
            ],
        ),
    }
    return nodes


@pytest.fixture
def builder(three_task_dag):
    """WorkerContextBuilder with the three-task DAG."""
    conflicts = [("task_2", "task_3", ["src/shared.py"])]
    return WorkerContextBuilder(
        dag_nodes=three_task_dag,
        spec_title="User Management System",
        file_conflicts=conflicts,
    )


@pytest.fixture
def parallel_dag():
    """DAG with two fully parallel tasks (no dependencies)."""
    nodes = {
        "task_a": TaskNode(
            id="task_a",
            title="Frontend components",
            description="Build React components",
            files_touched=["src/App.tsx", "src/Button.tsx"],
        ),
        "task_b": TaskNode(
            id="task_b",
            title="Backend API",
            description="Build Express routes",
            files_touched=["src/routes.ts", "src/App.tsx"],
        ),
    }
    return nodes


# =============================================================================
# System Prompt Generation
# =============================================================================

class TestBuildSystemPrompt:
    """Tests for build_system_prompt()."""

    def test_contains_task_identity(self, builder, three_task_dag):
        """System prompt should include task ID and title."""
        task = three_task_dag["task_2"]
        prompt = builder.build_system_prompt(task)
        assert "[task_2]" in prompt
        assert "Implement authentication" in prompt
        assert "User Management System" in prompt

    def test_contains_owned_files(self, builder, three_task_dag):
        """Should list files this task owns."""
        task = three_task_dag["task_2"]
        prompt = builder.build_system_prompt(task)
        assert "src/auth.py" in prompt
        assert "tests/test_auth.py" in prompt

    def test_contains_forbidden_files(self, builder, three_task_dag):
        """Should list files owned by parallel tasks as forbidden."""
        task = three_task_dag["task_2"]
        prompt = builder.build_system_prompt(task)
        # task_3 is parallel to task_2 (both depend on task_1)
        assert "src/api.py" in prompt
        assert "task_3" in prompt

    def test_dependency_files_not_forbidden(self, builder, three_task_dag):
        """Files from dependency tasks should not be forbidden."""
        task = three_task_dag["task_2"]
        prompt = builder.build_system_prompt(task)
        # task_1 is a dependency, so its files should NOT appear as forbidden
        # (task_2 depends on task_1, so they're sequential)
        lines = prompt.split("\n")
        forbidden_section = False
        forbidden_text = ""
        for line in lines:
            if "Do NOT modify" in line:
                forbidden_section = True
            elif line.startswith("##"):
                forbidden_section = False
            elif forbidden_section:
                forbidden_text += line
        assert "src/models.py" not in forbidden_text

    def test_contains_behavioral_rules(self, builder, three_task_dag):
        """Should include rules about file modification and testing."""
        task = three_task_dag["task_1"]
        prompt = builder.build_system_prompt(task)
        assert "Only modify files you own" in prompt
        assert "Run tests" in prompt

    def test_root_task_no_forbidden_from_dependents(self, builder, three_task_dag):
        """Root task should not list dependent tasks' files as forbidden.

        task_2 and task_3 depend on task_1, so they run after task_1.
        Sequential tasks can share files safely.
        """
        task = three_task_dag["task_1"]
        prompt = builder.build_system_prompt(task)
        # task_2 depends on task_1 — sequential, not forbidden
        # task_3 depends on task_1 — sequential, not forbidden
        lines = prompt.split("\n")
        forbidden_section = False
        forbidden_text = ""
        for line in lines:
            if "Do NOT modify" in line:
                forbidden_section = True
            elif line.startswith("##"):
                forbidden_section = False
            elif forbidden_section:
                forbidden_text += line
        assert "src/auth.py" not in forbidden_text
        assert "src/api.py" not in forbidden_text


# =============================================================================
# CLAUDE.md Generation
# =============================================================================

class TestBuildClaudeMd:
    """Tests for build_claude_md()."""

    def test_contains_header(self, builder, three_task_dag):
        """Should have an orchestration header with task ID and title."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "# Orchestration Context for [task_2]" in md
        assert "Implement authentication" in md

    def test_contains_sibling_table(self, builder, three_task_dag):
        """Should include a table of all sibling tasks."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "## Sibling Tasks" in md
        assert "task_1" in md
        assert "task_3" in md
        assert "Database models" in md
        assert "API endpoints" in md

    def test_sibling_relationship_labels(self, builder, three_task_dag):
        """Should describe relationships: dependency, provides, parallel."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        # task_1 is a dependency that provides User, Session, get_db
        assert "provides" in md
        # task_3 depends on task_2 (consumes authenticate)
        assert "parallel" in md or "depends on you" in md or "consumes" in md

    def test_contains_provides_section(self, builder, three_task_dag):
        """Should list what this task exports."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "## You PROVIDE" in md
        assert "authenticate" in md
        assert "AuthMiddleware" in md
        assert "src/auth.py" in md

    def test_contains_consumes_section(self, builder, three_task_dag):
        """Should list what this task imports from dependencies."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "## You CONSUME" in md
        assert "task_1" in md
        assert "User" in md
        assert "get_db" in md

    def test_contains_conflict_warnings(self, builder, three_task_dag):
        """Should warn about file conflicts with parallel tasks."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "File Conflict Warnings" in md
        assert "src/shared.py" in md

    def test_contains_completion_placeholder(self, builder, three_task_dag):
        """Should have a placeholder section for completion updates."""
        task = three_task_dag["task_2"]
        md = builder.build_claude_md(task)
        assert "## Completed Dependencies" in md
        assert "updates appended here" in md

    def test_no_provides_section_when_empty(self, builder, three_task_dag):
        """Should omit PROVIDE section if task has no provides."""
        task = three_task_dag["task_3"]
        md = builder.build_claude_md(task)
        assert "## You PROVIDE" not in md

    def test_no_conflict_warnings_when_none(self, three_task_dag):
        """Should omit conflict section if no conflicts involve this task."""
        builder = WorkerContextBuilder(
            dag_nodes=three_task_dag,
            spec_title="Test",
            file_conflicts=[],
        )
        task = three_task_dag["task_1"]
        md = builder.build_claude_md(task)
        assert "File Conflict Warnings" not in md

    def test_consumers_listed(self, builder, three_task_dag):
        """Providers should see who consumes their exports."""
        task = three_task_dag["task_1"]
        md = builder.build_claude_md(task)
        assert "## You PROVIDE" in md
        assert "Consumed by:" in md
        # task_2 and task_3 both consume from task_1
        assert "task_2" in md
        assert "task_3" in md


# =============================================================================
# Dynamic Updates
# =============================================================================

class TestBuildUpdate:
    """Tests for build_update()."""

    def test_update_contains_task_info(self, builder, three_task_dag):
        """Update should include completed task ID, title, and timestamp."""
        result = _make_result("task_1", files_modified=["src/models.py"])
        update = builder.build_update("task_1", result)
        assert "[task_1]" in update
        assert "Database models" in update
        assert "COMPLETED" in update

    def test_update_contains_files_modified(self, builder, three_task_dag):
        """Update should list files that were modified."""
        result = _make_result(
            "task_1",
            files_modified=["src/models.py", "tests/test_models.py"],
        )
        update = builder.build_update("task_1", result)
        assert "src/models.py" in update
        assert "tests/test_models.py" in update

    def test_update_contains_exports(self, builder, three_task_dag):
        """Update should include what's now available from provides."""
        result = _make_result("task_1")
        update = builder.build_update("task_1", result)
        assert "User" in update
        assert "Session" in update
        assert "get_db" in update

    def test_update_empty_for_unknown_task(self, builder):
        """Should return empty string for unknown task ID."""
        result = _make_result("unknown")
        update = builder.build_update("unknown", result)
        assert update == ""

    def test_update_no_exports_for_task_without_provides(self, builder, three_task_dag):
        """Task without provides should not have 'Now available' lines."""
        result = _make_result("task_3")
        update = builder.build_update("task_3", result)
        assert "Now available" not in update


# =============================================================================
# File Ownership Resolution
# =============================================================================

class TestFileOwnership:
    """Tests for _resolve_file_ownership()."""

    def test_owned_files_match_files_touched(self, builder, three_task_dag):
        """Owned files should be exactly the task's files_touched."""
        task = three_task_dag["task_2"]
        owned, _ = builder._resolve_file_ownership(task)
        assert owned == ["src/auth.py", "tests/test_auth.py"]

    def test_parallel_task_files_are_forbidden(self, parallel_dag):
        """Parallel tasks' files should appear in forbidden list."""
        builder = WorkerContextBuilder(
            dag_nodes=parallel_dag,
            spec_title="Test",
        )
        task = parallel_dag["task_a"]
        _, forbidden = builder._resolve_file_ownership(task)
        forbidden_files = [f for f, _ in forbidden]
        # task_b owns src/routes.ts — should be forbidden for task_a
        assert "src/routes.ts" in forbidden_files

    def test_shared_file_between_parallel_tasks(self, parallel_dag):
        """File owned by both parallel tasks: owned for self, forbidden via other."""
        builder = WorkerContextBuilder(
            dag_nodes=parallel_dag,
            spec_title="Test",
        )
        task = parallel_dag["task_a"]
        owned, forbidden = builder._resolve_file_ownership(task)
        # src/App.tsx is in both tasks' files_touched
        assert "src/App.tsx" in owned
        # It shouldn't appear in forbidden since it's already owned
        forbidden_files = [f for f, _ in forbidden]
        assert "src/App.tsx" not in forbidden_files

    def test_dependency_files_not_forbidden(self, builder, three_task_dag):
        """Files from a dependency should not be forbidden."""
        task = three_task_dag["task_2"]
        _, forbidden = builder._resolve_file_ownership(task)
        forbidden_files = [f for f, _ in forbidden]
        # task_1 is a dependency — sequential, so not forbidden
        assert "src/models.py" not in forbidden_files
        assert "tests/test_models.py" not in forbidden_files

    def test_no_forbidden_when_single_task(self):
        """Single task should have no forbidden files."""
        nodes = {
            "only": TaskNode(
                id="only",
                title="Solo",
                description="Lone task",
                files_touched=["src/main.py"],
            ),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Test")
        _, forbidden = builder._resolve_file_ownership(nodes["only"])
        assert forbidden == []


# =============================================================================
# CLAUDE.md File Operations
# =============================================================================

class TestClaudeMdFileOps:
    """Tests for write_claude_md() and append_update()."""

    def test_write_claude_md_creates_file(self, builder, three_task_dag, tmp_path):
        """Should create CLAUDE.md in the worktree directory."""
        task = three_task_dag["task_2"]
        path = builder.write_claude_md(task, tmp_path)
        assert path.exists()
        assert path.name == "CLAUDE.md"
        content = path.read_text(encoding="utf-8")
        assert "Orchestration Context" in content

    def test_preserves_existing_claude_md(self, builder, three_task_dag, tmp_path):
        """Should prepend orchestration context and preserve original."""
        existing = "# My Project\n\nExisting project instructions here."
        (tmp_path / "CLAUDE.md").write_text(existing, encoding="utf-8")

        task = three_task_dag["task_2"]
        path = builder.write_claude_md(task, tmp_path)
        content = path.read_text(encoding="utf-8")

        # Orchestration context comes first
        assert content.startswith("# Orchestration Context")
        # Separator present
        assert ORCHESTRATION_SEPARATOR.strip() in content
        # Original content preserved
        assert "Existing project instructions here" in content

    def test_append_update_adds_to_file(self, builder, three_task_dag, tmp_path):
        """Should append update text to existing CLAUDE.md."""
        task = three_task_dag["task_2"]
        builder.write_claude_md(task, tmp_path)

        result = _make_result("task_1", files_modified=["src/models.py"])
        update = builder.build_update("task_1", result)
        builder.append_update(tmp_path, update)

        content = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
        assert "[task_1]" in content
        assert "COMPLETED" in content

    def test_append_update_noop_without_file(self, builder, tmp_path):
        """Should silently skip if CLAUDE.md doesn't exist."""
        result = _make_result("task_1")
        update = builder.build_update("task_1", result)
        # Should not raise
        builder.append_update(tmp_path, update)
        assert not (tmp_path / "CLAUDE.md").exists()

    def test_multiple_updates_append_sequentially(self, builder, three_task_dag, tmp_path):
        """Multiple updates should all be visible in the file."""
        task = three_task_dag["task_3"]
        builder.write_claude_md(task, tmp_path)

        # Simulate task_1 completing
        result1 = _make_result("task_1", files_modified=["src/models.py"])
        builder.append_update(tmp_path, builder.build_update("task_1", result1))

        # Simulate task_2 completing
        result2 = _make_result("task_2", files_modified=["src/auth.py"])
        builder.append_update(tmp_path, builder.build_update("task_2", result2))

        content = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
        assert "[task_1]" in content
        assert "[task_2]" in content
        assert "src/models.py" in content
        assert "src/auth.py" in content


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for WorkerContextBuilder."""

    def test_no_siblings(self):
        """Single-task DAG should produce valid context without sibling table."""
        nodes = {
            "solo": TaskNode(
                id="solo",
                title="Only task",
                description="Do everything",
                files_touched=["src/main.py"],
            ),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Solo Project")

        prompt = builder.build_system_prompt(nodes["solo"])
        assert "[solo]" in prompt

        md = builder.build_claude_md(nodes["solo"])
        assert "## Sibling Tasks" not in md

    def test_no_files_touched(self):
        """Task with no files_touched should produce valid context."""
        nodes = {
            "t1": TaskNode(id="t1", title="Research", description="Explore codebase"),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Test")
        prompt = builder.build_system_prompt(nodes["t1"])
        assert "[t1]" in prompt
        assert "File Ownership" not in prompt

    def test_no_provides_or_consumes(self):
        """Tasks with no interface contracts should omit those sections."""
        nodes = {
            "t1": TaskNode(
                id="t1", title="Task A", description="Do A",
                files_touched=["a.py"],
            ),
            "t2": TaskNode(
                id="t2", title="Task B", description="Do B",
                files_touched=["b.py"],
            ),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Test")
        md = builder.build_claude_md(nodes["t1"])
        assert "## You PROVIDE" not in md
        assert "## You CONSUME" not in md

    def test_empty_file_conflicts(self):
        """No conflicts should produce no warnings section."""
        nodes = {
            "t1": TaskNode(id="t1", title="A", description="D"),
        }
        builder = WorkerContextBuilder(
            dag_nodes=nodes, spec_title="Test", file_conflicts=[]
        )
        md = builder.build_claude_md(nodes["t1"])
        assert "File Conflict Warnings" not in md

    def test_many_files_truncated_in_sibling_table(self):
        """Sibling with many files should show first 3 + count."""
        nodes = {
            "t1": TaskNode(id="t1", title="A", description="D"),
            "t2": TaskNode(
                id="t2", title="B", description="D",
                files_touched=["a.py", "b.py", "c.py", "d.py", "e.py"],
            ),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Test")
        md = builder.build_claude_md(nodes["t1"])
        assert "(+2)" in md

    def test_dependency_chain_transitive(self):
        """Transitive dependencies should not be marked as forbidden."""
        nodes = {
            "t1": TaskNode(
                id="t1", title="Base", description="D",
                files_touched=["base.py"],
            ),
            "t2": TaskNode(
                id="t2", title="Middle", description="D",
                dependencies=["t1"],
                files_touched=["mid.py"],
            ),
            "t3": TaskNode(
                id="t3", title="Top", description="D",
                dependencies=["t2"],
                files_touched=["top.py"],
            ),
        }
        builder = WorkerContextBuilder(dag_nodes=nodes, spec_title="Test")
        # For t3, t1 is a transitive dependency — its files should not be forbidden
        _, forbidden = builder._resolve_file_ownership(nodes["t3"])
        forbidden_files = [f for f, _ in forbidden]
        assert "base.py" not in forbidden_files
        assert "mid.py" not in forbidden_files
