"""
Result Synthesizer

Aggregates results from parallel workers, handles merge conflicts
(with optional LLM-assisted resolution), and produces execution summaries.

Module: Orchestrator
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from bashgym.orchestrator.models import (
    MergeResult, WorkerResult, LLMConfig, TaskNode,
)
from bashgym.orchestrator.worktree import WorktreeManager
from bashgym.orchestrator.task_dag import TaskDAG, _call_llm

logger = logging.getLogger(__name__)


CONFLICT_RESOLUTION_PROMPT = """You are resolving a git merge conflict between two parallel development tasks.

## Task A: {task_a_title}
{task_a_description}

## Task B: {task_b_title}
{task_b_description}

## Conflicting file: {file_path}

## Conflict markers:
```
{conflict_content}
```

Produce the resolved file content that correctly integrates both changes.
Output ONLY the resolved code — no explanations, no markdown fences.
If the changes are truly incompatible, prefer Task A's version (higher priority)
and add a TODO comment noting what was lost from Task B."""


@dataclass
class SynthesisReport:
    """Summary of the synthesis (merge) phase."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    merge_successes: int = 0
    merge_failures: int = 0
    conflicts_resolved: int = 0
    conflicts_unresolved: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    merge_results: List[MergeResult] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "merge_successes": self.merge_successes,
            "merge_failures": self.merge_failures,
            "conflicts_resolved": self.conflicts_resolved,
            "conflicts_unresolved": self.conflicts_unresolved,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "files_modified": self.files_modified,
        }


class ResultSynthesizer:
    """Aggregates worker results and merges worktree branches.

    Handles the SYNTHESIZE phase of orchestration:
    1. Merge completed task branches in dependency order
    2. Attempt LLM-assisted conflict resolution when git merge fails
    3. Produce a summary report of the entire execution
    """

    def __init__(
        self,
        worktrees: Optional[WorktreeManager] = None,
        llm_config: Optional[LLMConfig] = None,
        auto_resolve_conflicts: bool = True,
    ):
        """Initialize the synthesizer.

        Args:
            worktrees: WorktreeManager for merge operations
            llm_config: LLM config for conflict resolution (uses same
                        provider as the orchestrator)
            auto_resolve_conflicts: Whether to attempt LLM conflict resolution
        """
        self.worktrees = worktrees
        self.llm_config = llm_config
        self.auto_resolve = auto_resolve_conflicts

    async def synthesize(
        self,
        dag: TaskDAG,
        results: List[WorkerResult],
        target_branch: str = "main",
    ) -> SynthesisReport:
        """Merge all completed task branches and produce a report.

        Merges in topological order to respect dependencies — tasks that
        others depend on get merged first.

        Args:
            dag: The executed TaskDAG
            results: Worker results from execution
            target_branch: Branch to merge into

        Returns:
            SynthesisReport with merge outcomes and summary stats
        """
        report = SynthesisReport(
            total_tasks=len(dag.nodes),
            completed_tasks=sum(1 for r in results if r.success),
            failed_tasks=sum(1 for r in results if not r.success),
            total_cost_usd=sum(r.cost_usd for r in results),
            total_duration_seconds=sum(r.duration_seconds for r in results),
        )

        # Collect all modified files across results
        all_files = set()
        for r in results:
            all_files.update(r.files_modified)
        report.files_modified = sorted(all_files)

        if not self.worktrees:
            logger.info("No worktree manager — skipping merge phase")
            return report

        # Merge in topological order for correct layering
        completed_ids = dag.completed_tasks()
        try:
            topo_order = dag.topological_sort()
            merge_order = [t.id for t in topo_order if t.id in completed_ids]
        except Exception:
            merge_order = completed_ids

        for task_id in merge_order:
            merge_result = await self.worktrees.merge(task_id, target_branch)

            if merge_result.success:
                report.merge_successes += 1
                report.merge_results.append(merge_result)
                logger.info(f"Merged task {task_id} successfully")
            elif merge_result.conflicts and self.auto_resolve and self.llm_config:
                # Attempt LLM-assisted conflict resolution
                resolved = await self._resolve_conflicts(
                    dag, task_id, merge_result, target_branch
                )
                report.merge_results.append(resolved)
                if resolved.success:
                    report.conflicts_resolved += len(merge_result.conflicts)
                    report.merge_successes += 1
                    logger.info(
                        f"Resolved {len(merge_result.conflicts)} conflicts "
                        f"for task {task_id} via LLM"
                    )
                else:
                    report.conflicts_unresolved += len(merge_result.conflicts)
                    report.merge_failures += 1
                    logger.warning(
                        f"Could not resolve conflicts for task {task_id}"
                    )
            else:
                report.merge_failures += 1
                report.merge_results.append(merge_result)
                if merge_result.conflicts:
                    report.conflicts_unresolved += len(merge_result.conflicts)
                logger.warning(
                    f"Merge failed for task {task_id}: "
                    f"{merge_result.error or merge_result.conflicts}"
                )

        # Cleanup all worktrees
        await self.worktrees.cleanup_all()

        logger.info(
            f"Synthesis complete: {report.merge_successes} merged, "
            f"{report.merge_failures} failed, "
            f"{report.conflicts_resolved} conflicts resolved"
        )

        return report

    async def _resolve_conflicts(
        self,
        dag: TaskDAG,
        task_id: str,
        failed_merge: MergeResult,
        target_branch: str,
    ) -> MergeResult:
        """Attempt LLM-assisted merge conflict resolution.

        For each conflicting file:
        1. Read the conflict markers from git
        2. Ask the LLM to produce resolved content
        3. Write the resolution and stage it
        4. Complete the merge

        Args:
            dag: TaskDAG for task context
            task_id: ID of the task whose merge failed
            failed_merge: The MergeResult with conflict details
            target_branch: Target branch for merge

        Returns:
            Updated MergeResult (success or failure)
        """
        if not self.worktrees or not self.llm_config:
            return failed_merge

        task = dag.nodes.get(task_id)
        if not task:
            return failed_merge

        repo_path = self.worktrees.repo_path
        branch = self.worktrees._branches.get(task_id, "unknown")

        # Re-attempt the merge to get conflict markers
        proc = await asyncio.create_subprocess_exec(
            "git", "merge", branch, "--no-ff",
            "-m", f"Merge task/{task_id}: {branch}",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode == 0:
            # Merge succeeded this time (race condition, previous state changed)
            return MergeResult(
                task_id=task_id,
                branch=branch,
                success=True,
                files_merged=failed_merge.files_merged,
            )

        # Try to resolve each conflicting file
        all_resolved = True
        for conflict_file in failed_merge.conflicts:
            if not conflict_file.strip():
                continue

            conflict_path = repo_path / conflict_file
            if not conflict_path.exists():
                all_resolved = False
                continue

            try:
                conflict_content = conflict_path.read_text(encoding="utf-8")
            except Exception:
                all_resolved = False
                continue

            # Skip if no actual conflict markers
            if "<<<<<<<" not in conflict_content:
                continue

            # Ask LLM to resolve
            try:
                resolved_content = await _call_llm(
                    self.llm_config,
                    "You are a senior developer resolving merge conflicts.",
                    CONFLICT_RESOLUTION_PROMPT.format(
                        task_a_title=task.title,
                        task_a_description=task.description[:500],
                        task_b_title=f"Changes on {target_branch}",
                        task_b_description="Existing code on the target branch",
                        file_path=conflict_file,
                        conflict_content=conflict_content[:4000],
                    ),
                )

                # Write resolution
                conflict_path.write_text(resolved_content, encoding="utf-8")

                # Stage the resolved file
                await asyncio.create_subprocess_exec(
                    "git", "add", conflict_file,
                    cwd=str(repo_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            except Exception as e:
                logger.warning(
                    f"LLM conflict resolution failed for {conflict_file}: {e}"
                )
                all_resolved = False

        if all_resolved:
            # Complete the merge commit
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "--no-edit",
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                return MergeResult(
                    task_id=task_id,
                    branch=branch,
                    success=True,
                    files_merged=failed_merge.files_merged,
                    conflicts=failed_merge.conflicts,  # Record that conflicts existed
                )

        # Abort if we couldn't resolve everything
        await asyncio.create_subprocess_exec(
            "git", "merge", "--abort",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        return MergeResult(
            task_id=task_id,
            branch=branch,
            success=False,
            conflicts=failed_merge.conflicts,
            files_merged=failed_merge.files_merged,
            error="LLM conflict resolution incomplete",
        )
