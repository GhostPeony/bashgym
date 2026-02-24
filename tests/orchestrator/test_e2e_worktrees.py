"""E2E tests for git worktree operations.

Uses real git repositories (via tmp_path) to test WorktreeManager's
create, merge, conflict detection, cleanup, and isolation guarantees.

No mocking of git — these are true integration tests against real repos.
"""

import asyncio
from pathlib import Path

import pytest

from bashgym.orchestrator.worktree import WorktreeManager


# =============================================================================
# Helpers
# =============================================================================


async def git(*args, cwd):
    """Run a git command, raise on failure."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {err.decode()}")
    return out.decode().strip()


async def commit_file(repo, filepath, content, message):
    """Write a file and commit it in the given repo/worktree."""
    full_path = Path(repo) / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    await git("add", filepath, cwd=repo)
    await git("commit", "-m", message, cwd=repo)


async def get_file_content(repo, filepath):
    """Read a file from the repo."""
    full_path = Path(repo) / filepath
    return full_path.read_text()


async def get_current_branch(repo):
    """Get the current branch name."""
    return await git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)


# =============================================================================
# Worktree Lifecycle
# =============================================================================


class TestWorktreeLifecycle:
    """Tests for creating and cleaning up worktrees."""

    @pytest.mark.asyncio
    async def test_create_worktree_creates_directory(self, async_git_repo):
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        path = await mgr.create("task_1", "task/task_1", "main")

        assert path.exists()
        assert path.is_dir()
        # Verify branch was created
        branches = await git("branch", "--list", cwd=repo)
        assert "task/task_1" in branches

    @pytest.mark.asyncio
    async def test_create_worktree_has_repo_files(self, async_git_repo):
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        path = await mgr.create("task_1", "task/task_1", "main")

        # Worktree should contain files from the base branch
        assert (path / "README.md").exists()
        assert (path / "src" / "main.py").exists()
        content = (path / "README.md").read_text()
        assert "Test Repo" in content

    @pytest.mark.asyncio
    async def test_create_multiple_worktrees(self, async_git_repo):
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        await mgr.create("t1", "task/t1", "main")
        await mgr.create("t2", "task/t2", "main")
        await mgr.create("t3", "task/t3", "main")

        assert mgr.active_count == 3
        assert "t1" in mgr.active_worktrees
        assert "t2" in mgr.active_worktrees
        assert "t3" in mgr.active_worktrees

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree(self, async_git_repo):
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        path = await mgr.create("task_1", "task/task_1", "main")
        assert path.exists()

        await mgr.cleanup("task_1")

        assert not path.exists()
        assert mgr.active_count == 0
        # Branch should be deleted
        branches = await git("branch", "--list", cwd=repo)
        assert "task/task_1" not in branches

    @pytest.mark.asyncio
    async def test_cleanup_all_removes_everything(self, async_git_repo):
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        paths = []
        for i in range(3):
            p = await mgr.create(f"t{i}", f"task/t{i}", "main")
            paths.append(p)

        assert mgr.active_count == 3

        await mgr.cleanup_all()

        assert mgr.active_count == 0
        for p in paths:
            assert not p.exists()


# =============================================================================
# Worktree Merge
# =============================================================================


class TestWorktreeMerge:
    """Tests for merging worktree branches back into main."""

    @pytest.mark.asyncio
    async def test_merge_clean_new_file(self, async_git_repo):
        """Add a new file in worktree → merge succeeds."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt_path = await mgr.create("task_1", "task/task_1", "main")

        # Add a new file in the worktree
        await commit_file(wt_path, "src/models.py", "class User: pass\n", "Add models")

        # Switch back to main for merge
        await git("checkout", "main", cwd=repo)

        result = await mgr.merge("task_1", "main")

        assert result.success is True
        assert result.task_id == "task_1"
        # New file should be on main
        assert (repo / "src" / "models.py").exists()

    @pytest.mark.asyncio
    async def test_merge_modified_file(self, async_git_repo):
        """Modify an existing file in worktree → merge succeeds."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt_path = await mgr.create("task_1", "task/task_1", "main")

        # Modify existing file
        new_content = 'def main():\n    print("hello world")\n'
        await commit_file(wt_path, "src/main.py", new_content, "Update main.py")

        await git("checkout", "main", cwd=repo)

        result = await mgr.merge("task_1", "main")

        assert result.success is True
        content = await get_file_content(repo, "src/main.py")
        assert "hello world" in content

    @pytest.mark.asyncio
    async def test_merge_conflict_detected(self, async_git_repo):
        """Modify same file on main AND worktree → conflict detected."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt_path = await mgr.create("task_1", "task/task_1", "main")

        # Modify in worktree
        await commit_file(
            wt_path, "src/main.py",
            'def main():\n    print("from worktree")\n',
            "Worktree change",
        )

        # Modify same file on main (different content)
        await git("checkout", "main", cwd=repo)
        await commit_file(
            repo, "src/main.py",
            'def main():\n    print("from main")\n',
            "Main change",
        )

        result = await mgr.merge("task_1", "main")

        assert result.success is False
        assert len(result.conflicts) > 0
        assert "src/main.py" in result.conflicts

    @pytest.mark.asyncio
    async def test_merge_multiple_files(self, async_git_repo):
        """Worktree modifies 3 files → all present on main after merge."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt_path = await mgr.create("task_1", "task/task_1", "main")

        # Create/modify multiple files
        await commit_file(wt_path, "src/models.py", "class User: pass\n", "Add models")
        await commit_file(wt_path, "src/api.py", "def login(): pass\n", "Add API")
        await commit_file(wt_path, "tests/test_api.py", "def test_login(): pass\n", "Add tests")

        await git("checkout", "main", cwd=repo)

        result = await mgr.merge("task_1", "main")

        assert result.success is True
        assert (repo / "src" / "models.py").exists()
        assert (repo / "src" / "api.py").exists()
        assert (repo / "tests" / "test_api.py").exists()

    @pytest.mark.asyncio
    async def test_merge_no_branch_returns_error(self, async_git_repo):
        """Merging a non-existent task returns an error MergeResult."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        result = await mgr.merge("nonexistent", "main")

        assert result.success is False
        assert "No branch found" in result.error


# =============================================================================
# Worktree Isolation
# =============================================================================


class TestWorktreeIsolation:
    """Tests that worktrees don't bleed changes between each other."""

    @pytest.mark.asyncio
    async def test_worktrees_are_isolated(self, async_git_repo):
        """Changes in one worktree don't appear in another."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt1 = await mgr.create("t1", "task/t1", "main")
        wt2 = await mgr.create("t2", "task/t2", "main")

        # Write different files in each
        await commit_file(wt1, "file_a.txt", "content A\n", "Add file_a")
        await commit_file(wt2, "file_b.txt", "content B\n", "Add file_b")

        # Each worktree should only have its own file
        assert (wt1 / "file_a.txt").exists()
        assert not (wt1 / "file_b.txt").exists()
        assert (wt2 / "file_b.txt").exists()
        assert not (wt2 / "file_a.txt").exists()

    @pytest.mark.asyncio
    async def test_worktree_doesnt_affect_main(self, async_git_repo):
        """Main branch is unchanged until merge."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt = await mgr.create("t1", "task/t1", "main")

        await commit_file(wt, "new_file.txt", "hello\n", "Add file")

        # Main should NOT have the new file yet
        await git("checkout", "main", cwd=repo)
        assert not (repo / "new_file.txt").exists()

    @pytest.mark.asyncio
    async def test_sequential_merges_accumulate(self, async_git_repo):
        """Merging A, then B → main has both sets of changes."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        wt1 = await mgr.create("t1", "task/t1", "main")
        wt2 = await mgr.create("t2", "task/t2", "main")

        await commit_file(wt1, "file_a.txt", "A\n", "Add A")
        await commit_file(wt2, "file_b.txt", "B\n", "Add B")

        await git("checkout", "main", cwd=repo)

        result1 = await mgr.merge("t1", "main")
        assert result1.success is True

        result2 = await mgr.merge("t2", "main")
        assert result2.success is True

        # Main should have both files
        assert (repo / "file_a.txt").exists()
        assert (repo / "file_b.txt").exists()

    @pytest.mark.asyncio
    async def test_worktree_based_on_correct_branch(self, async_git_repo):
        """Worktree starts from the base branch's HEAD."""
        repo = async_git_repo
        mgr = WorktreeManager(repo)

        # Add a file to main first
        await commit_file(repo, "base_file.txt", "base content\n", "Add base file")

        wt = await mgr.create("t1", "task/t1", "main")

        # Worktree should have the base_file
        assert (wt / "base_file.txt").exists()
        assert (wt / "base_file.txt").read_text() == "base content\n"
