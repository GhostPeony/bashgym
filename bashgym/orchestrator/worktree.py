"""
Git Worktree Manager

Creates and manages git worktrees for task isolation.
Each worker operates in its own worktree on a separate branch,
preventing merge conflicts during parallel execution.

Module: Orchestrator
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from bashgym.orchestrator.models import MergeResult

logger = logging.getLogger(__name__)


class WorktreeManager:
    """Manages git worktrees for task isolation.

    Each task gets its own worktree with a dedicated branch.
    After completion, branches are merged back into the target.
    """

    def __init__(
        self,
        repo_path: Path,
        worktree_base: Optional[Path] = None,
    ):
        self.repo_path = Path(repo_path)
        self.worktree_base = worktree_base or (self.repo_path / ".worktrees")
        self.active_worktrees: Dict[str, Path] = {}
        self._branches: Dict[str, str] = {}  # task_id -> branch_name

    async def create(
        self,
        task_id: str,
        branch_name: str,
        base_branch: str = "main",
    ) -> Path:
        """Create an isolated git worktree for a task.

        Runs: git worktree add <path> -b <branch_name> <base_branch>

        Args:
            task_id: Unique task identifier
            branch_name: Name for the new branch
            base_branch: Branch to base the worktree on

        Returns:
            Path to the worktree directory
        """
        worktree_path = self.worktree_base / task_id
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the worktree with a new branch
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add",
            str(worktree_path), "-b", branch_name, base_branch,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode().strip()
            # Branch might already exist, try without -b
            if "already exists" in error:
                proc = await asyncio.create_subprocess_exec(
                    "git", "worktree", "add",
                    str(worktree_path), branch_name,
                    cwd=str(self.repo_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create worktree: {stderr.decode().strip()}"
                    )
            else:
                raise RuntimeError(f"Failed to create worktree: {error}")

        self.active_worktrees[task_id] = worktree_path
        self._branches[task_id] = branch_name

        logger.info(f"Created worktree for task {task_id} at {worktree_path}")
        return worktree_path

    async def merge(
        self,
        task_id: str,
        target_branch: str = "main",
    ) -> MergeResult:
        """Merge a task's worktree branch back into the target.

        Switches to target branch and merges the task branch.

        Args:
            task_id: Task whose branch to merge
            target_branch: Branch to merge into

        Returns:
            MergeResult with success status and any conflicts
        """
        if task_id not in self._branches:
            return MergeResult(
                task_id=task_id,
                branch="unknown",
                success=False,
                error=f"No branch found for task {task_id}",
            )

        branch = self._branches[task_id]

        # Get list of files changed in the branch
        diff_proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", f"{target_branch}...{branch}",
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        diff_out, _ = await diff_proc.communicate()
        files_merged = diff_out.decode().strip().split("\n") if diff_out.decode().strip() else []

        # Perform the merge
        proc = await asyncio.create_subprocess_exec(
            "git", "merge", branch, "--no-ff",
            "-m", f"Merge task/{task_id}: {branch}",
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode().strip()

            # Check for merge conflicts
            if "CONFLICT" in error or "conflict" in stdout.decode():
                # Get list of conflicting files
                conflict_proc = await asyncio.create_subprocess_exec(
                    "git", "diff", "--name-only", "--diff-filter=U",
                    cwd=str(self.repo_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                conflict_out, _ = await conflict_proc.communicate()
                conflicts = conflict_out.decode().strip().split("\n")

                # Abort the merge
                await asyncio.create_subprocess_exec(
                    "git", "merge", "--abort",
                    cwd=str(self.repo_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                logger.warning(
                    f"Merge conflicts for task {task_id}: {conflicts}"
                )
                return MergeResult(
                    task_id=task_id,
                    branch=branch,
                    success=False,
                    conflicts=conflicts,
                    files_merged=files_merged,
                    error="Merge conflicts detected",
                )

            return MergeResult(
                task_id=task_id,
                branch=branch,
                success=False,
                error=error,
            )

        logger.info(f"Successfully merged task {task_id} ({branch})")
        return MergeResult(
            task_id=task_id,
            branch=branch,
            success=True,
            files_merged=files_merged,
        )

    async def cleanup(self, task_id: str) -> None:
        """Remove a worktree and optionally its branch.

        Runs:
            git worktree remove <path>
            git branch -d <branch>

        Args:
            task_id: Task whose worktree to remove
        """
        if task_id in self.active_worktrees:
            worktree_path = self.active_worktrees[task_id]

            # Remove the worktree
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "remove", str(worktree_path), "--force",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            del self.active_worktrees[task_id]

        # Delete the branch
        if task_id in self._branches:
            branch = self._branches[task_id]
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "-d", branch,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            del self._branches[task_id]

        logger.info(f"Cleaned up worktree for task {task_id}")

    async def cleanup_all(self) -> None:
        """Remove all active worktrees."""
        task_ids = list(self.active_worktrees.keys())
        for task_id in task_ids:
            await self.cleanup(task_id)

        # Prune any stale worktree references
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "prune",
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    @property
    def active_count(self) -> int:
        return len(self.active_worktrees)
