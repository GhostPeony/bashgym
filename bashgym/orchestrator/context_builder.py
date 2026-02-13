"""
Worker Context Builder

Generates per-task context from the DAG so that parallel Claude Code
workers have awareness of siblings, file ownership, and interface contracts.

Two output channels:
- System prompt (--append-system-prompt): fixed at spawn, hard constraints
- CLAUDE.md (written to worktree): dynamic, updated as tasks complete

Module: Orchestrator
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bashgym.orchestrator.models import TaskNode, TaskStatus, WorkerResult

logger = logging.getLogger(__name__)

# Separator used when prepending orchestration context to an existing CLAUDE.md
ORCHESTRATION_SEPARATOR = (
    "\n\n---\n\n"
    "<!-- END ORCHESTRATION CONTEXT — original project CLAUDE.md below -->\n\n"
)

COMPLETION_HEADER = "\n\n## Completed Dependencies\n"


class WorkerContextBuilder:
    """Builds per-worker context from a TaskDAG.

    Injected into each worker via two channels:
    - build_system_prompt() → --append-system-prompt (fixed at spawn)
    - build_claude_md() → CLAUDE.md in the worktree (updatable on disk)
    - build_update() → appended to CLAUDE.md as sibling tasks complete
    """

    def __init__(
        self,
        dag_nodes: Dict[str, TaskNode],
        spec_title: str,
        file_conflicts: Optional[List[Tuple[str, str, List[str]]]] = None,
    ):
        """
        Args:
            dag_nodes: All task nodes in the DAG (TaskDAG.nodes)
            spec_title: The project/spec title for context
            file_conflicts: Pre-computed file conflict tuples from
                           TaskDAG.detect_file_conflicts()
        """
        self._nodes = dag_nodes
        self._spec_title = spec_title
        self._file_conflicts = file_conflicts or []

    # =========================================================================
    # System Prompt (fixed at spawn)
    # =========================================================================

    def build_system_prompt(self, task: TaskNode) -> str:
        """Generate the system prompt appended via --append-system-prompt.

        Contains task identity, file ownership rules, and behavioral
        constraints. Kept compact (~400 tokens) since it's injected into
        every API call for the worker's session.
        """
        owned, forbidden = self._resolve_file_ownership(task)

        parts = [
            f"You are Worker [{task.id}] in a multi-agent development session.",
            f"Project: {self._spec_title}",
            f"Task: {task.title}",
            "",
        ]

        if owned:
            parts.append("## File Ownership")
            parts.append(f"You OWN: {', '.join(owned)}")

        if forbidden:
            forbidden_lines = []
            for filepath, owner_id in forbidden:
                forbidden_lines.append(f"  {filepath} ({owner_id})")
            parts.append("Do NOT modify:")
            parts.extend(forbidden_lines)

        parts.append("")
        parts.append("## Rules")
        parts.append(
            "- Only modify files you own. Read any file freely."
        )
        parts.append(
            "- If you need a function/class from another task, "
            "import it — do not redefine it."
        )
        parts.append(
            "- Keep changes minimal and focused on your task."
        )
        parts.append(
            "- Run tests after making changes to verify correctness."
        )

        return "\n".join(parts)

    # =========================================================================
    # CLAUDE.md (dynamic, updatable)
    # =========================================================================

    def build_claude_md(self, task: TaskNode) -> str:
        """Generate the orchestration context section for CLAUDE.md.

        Contains the sibling task table, interface contracts, and a
        placeholder for completion updates. Written to the worktree
        before the worker starts; updated as siblings complete.
        """
        parts = [
            f"# Orchestration Context for [{task.id}]: {task.title}",
            "",
        ]

        # Sibling task table
        siblings = self._get_siblings(task)
        if siblings:
            parts.append("## Sibling Tasks")
            parts.append("")
            parts.append(
                "| Task | Title | Files | Status | Relationship |"
            )
            parts.append(
                "|------|-------|-------|--------|--------------|"
            )
            for sib in siblings:
                relationship = self._describe_relationship(task, sib)
                files_str = ", ".join(sib.files_touched[:3])
                if len(sib.files_touched) > 3:
                    files_str += f" (+{len(sib.files_touched) - 3})"
                parts.append(
                    f"| {sib.id} | {sib.title} | {files_str} "
                    f"| {sib.status.value} | {relationship} |"
                )
            parts.append("")

        # Conflict warnings
        task_conflicts = [
            (a, b, files) for a, b, files in self._file_conflicts
            if task.id in (a, b)
        ]
        if task_conflicts:
            parts.append("## File Conflict Warnings")
            parts.append("")
            for a, b, files in task_conflicts:
                other = b if a == task.id else a
                parts.append(
                    f"- **{other}** also touches: {', '.join(files)}"
                )
            parts.append("")

        # Interface contracts: what this task provides
        if task.provides:
            parts.append("## You PROVIDE")
            parts.append("")
            for contract in task.provides:
                file = contract.get("file", "?")
                exports = contract.get("exports", [])
                desc = contract.get("description", "")
                exports_str = ", ".join(exports) if exports else ""
                line = f"- `{file}`"
                if exports_str:
                    line += f": {exports_str}"
                if desc:
                    line += f" — {desc}"
                parts.append(line)

            # Find who consumes from this task
            consumers = self._find_consumers(task.id)
            if consumers:
                parts.append("")
                parts.append("Consumed by: " + ", ".join(consumers))
            parts.append("")

        # Interface contracts: what this task consumes
        if task.consumes:
            parts.append("## You CONSUME")
            parts.append("")
            for contract in task.consumes:
                from_task = contract.get("from_task", "?")
                file = contract.get("file", "?")
                imports = contract.get("imports", [])
                desc = contract.get("description", "")
                imports_str = ", ".join(imports) if imports else ""
                line = f"- From **{from_task}** `{file}`"
                if imports_str:
                    line += f": {imports_str}"
                if desc:
                    line += f" — {desc}"
                parts.append(line)
            parts.append("")

        # Completion updates placeholder
        parts.append("## Completed Dependencies")
        parts.append("")
        parts.append("(updates appended here as dependencies finish)")
        parts.append("")

        return "\n".join(parts)

    # =========================================================================
    # Dynamic Updates
    # =========================================================================

    def build_update(
        self,
        completed_task_id: str,
        result: WorkerResult,
    ) -> str:
        """Generate a completion notification to append to a worker's CLAUDE.md.

        Called when a sibling task completes. The text is appended to the
        running worker's CLAUDE.md file so it picks up the new context
        on its next CLAUDE.md re-read.
        """
        completed = self._nodes.get(completed_task_id)
        if not completed:
            return ""

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        parts = [
            f"\n### [{completed_task_id}] {completed.title} — COMPLETED ({timestamp})",
        ]

        if result.files_modified:
            parts.append(f"Files changed: {', '.join(result.files_modified)}")

        # Include what was provided
        if completed.provides:
            for contract in completed.provides:
                file = contract.get("file", "")
                exports = contract.get("exports", [])
                if exports:
                    parts.append(
                        f"Now available: `{file}` exports {', '.join(exports)}"
                    )

        return "\n".join(parts)

    # =========================================================================
    # CLAUDE.md File Operations
    # =========================================================================

    def write_claude_md(
        self,
        task: TaskNode,
        worktree_path: Path,
    ) -> Path:
        """Write orchestration context to CLAUDE.md in the worktree.

        If the worktree already contains a CLAUDE.md (from the repo),
        the orchestration context is prepended and the original content
        is preserved below a separator.

        Returns:
            Path to the written CLAUDE.md
        """
        claude_md_path = worktree_path / "CLAUDE.md"
        orchestration_context = self.build_claude_md(task)

        existing_content = ""
        if claude_md_path.exists():
            try:
                existing_content = claude_md_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(f"Could not read existing CLAUDE.md: {e}")

        if existing_content:
            content = (
                orchestration_context
                + ORCHESTRATION_SEPARATOR
                + existing_content
            )
        else:
            content = orchestration_context

        claude_md_path.write_text(content, encoding="utf-8")
        logger.debug(f"Wrote CLAUDE.md for {task.id} at {claude_md_path}")
        return claude_md_path

    def append_update(
        self,
        worktree_path: Path,
        update_text: str,
    ) -> None:
        """Append a completion update to a worker's CLAUDE.md.

        The worker will pick this up on its next CLAUDE.md re-read cycle.
        """
        claude_md_path = worktree_path / "CLAUDE.md"
        if not claude_md_path.exists():
            logger.debug(
                f"No CLAUDE.md at {claude_md_path}, skipping update"
            )
            return

        try:
            with open(claude_md_path, "a", encoding="utf-8") as f:
                f.write(update_text + "\n")
            logger.debug(f"Appended update to {claude_md_path}")
        except OSError as e:
            logger.warning(f"Failed to append update to CLAUDE.md: {e}")

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _resolve_file_ownership(
        self, task: TaskNode
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Derive owned and forbidden files for a task.

        Owned: files listed in this task's files_touched.
        Forbidden: files owned by parallel (non-dependent) tasks.

        Returns:
            (owned_files, [(forbidden_file, owner_task_id), ...])
        """
        owned = list(task.files_touched)

        # Build set of tasks in this task's dependency chain (up and down)
        related = self._get_dependency_chain(task.id)

        forbidden: List[Tuple[str, str]] = []
        for other_id, other in self._nodes.items():
            if other_id == task.id:
                continue
            if other_id in related:
                continue  # Sequential tasks can share files
            for f in other.files_touched:
                if f not in owned:
                    forbidden.append((f, other_id))

        # Deduplicate by file (keep first owner)
        seen_files: Set[str] = set()
        unique_forbidden = []
        for filepath, owner in forbidden:
            if filepath not in seen_files:
                seen_files.add(filepath)
                unique_forbidden.append((filepath, owner))

        return owned, unique_forbidden

    def _get_dependency_chain(self, task_id: str) -> Set[str]:
        """Get all tasks transitively related via dependencies.

        Includes both ancestors (tasks this depends on) and descendants
        (tasks that depend on this). These tasks run sequentially so
        file sharing is safe.
        """
        related: Set[str] = set()

        # Ancestors (upstream)
        to_visit = list(self._nodes.get(task_id, TaskNode(
            id="", title="", description=""
        )).dependencies)
        while to_visit:
            dep_id = to_visit.pop()
            if dep_id in related or dep_id not in self._nodes:
                continue
            related.add(dep_id)
            to_visit.extend(self._nodes[dep_id].dependencies)

        # Descendants (downstream)
        for other_id, other in self._nodes.items():
            if task_id in other.dependencies:
                related.add(other_id)
                # Recurse down
                stack = [other_id]
                while stack:
                    current = stack.pop()
                    for nid, node in self._nodes.items():
                        if current in node.dependencies and nid not in related:
                            related.add(nid)
                            stack.append(nid)

        return related

    def _get_siblings(self, task: TaskNode) -> List[TaskNode]:
        """Get all tasks except this one, sorted by ID."""
        return sorted(
            [n for n in self._nodes.values() if n.id != task.id],
            key=lambda n: n.id,
        )

    def _describe_relationship(
        self, task: TaskNode, other: TaskNode
    ) -> str:
        """Describe the relationship between two tasks."""
        if other.id in task.dependencies:
            # What does the dependency provide?
            provides_desc = []
            for contract in other.provides:
                exports = contract.get("exports", [])
                if exports:
                    provides_desc.extend(exports)
            if provides_desc:
                return f"provides {', '.join(provides_desc[:3])}"
            return "dependency"

        if task.id in other.dependencies:
            # What does this task provide to the other?
            consumes_desc = []
            for contract in other.consumes:
                if contract.get("from_task") == task.id:
                    imports = contract.get("imports", [])
                    consumes_desc.extend(imports)
            if consumes_desc:
                return f"consumes your {', '.join(consumes_desc[:3])}"
            return "depends on you"

        return "parallel"

    def _find_consumers(self, task_id: str) -> List[str]:
        """Find task IDs that consume from the given task."""
        consumers = []
        for other_id, other in self._nodes.items():
            if other_id == task_id:
                continue
            for contract in other.consumes:
                if contract.get("from_task") == task_id:
                    consumers.append(other_id)
                    break
        return sorted(consumers)
