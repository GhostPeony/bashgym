"""
Worker Context Builder — Three-Layer Prompt Composition

Generates per-task context using a three-layer "onion model":
  1. Identity  (static)  — role, coding standards, tool guidelines
  2. Narrative  (dynamic) — sibling status, file ownership, interface contracts
  3. Focus     (task)    — requirements, files, budget, success criteria

Plus transition markers appended as siblings complete.

Two output channels:
- System prompt (--append-system-prompt): fixed at spawn, hard constraints
- CLAUDE.md (written to worktree): dynamic, updated as tasks complete

Module: Orchestrator
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from bashgym.orchestrator.models import TaskNode, TaskStatus, WorkerResult

if TYPE_CHECKING:
    from bashgym.orchestrator.shared_state import SharedState

logger = logging.getLogger(__name__)

# Separator used when prepending orchestration context to an existing CLAUDE.md
ORCHESTRATION_SEPARATOR = (
    "\n\n---\n\n"
    "<!-- END ORCHESTRATION CONTEXT — original project CLAUDE.md below -->\n\n"
)

COMPLETION_HEADER = "\n\n## Completed Dependencies\n"


class WorkerContextBuilder:
    """Builds per-worker context from a TaskDAG using three-layer composition.

    Layers:
    - Identity:   static across all workers in a run
    - Narrative:   dynamic sibling/ownership/contract awareness (no LLM call)
    - Focus:      task-specific requirements and constraints

    Injected into each worker via two channels:
    - build_system_prompt() -> --append-system-prompt (fixed at spawn)
    - build_claude_md()     -> CLAUDE.md in the worktree (updatable on disk)
    - build_update()        -> appended to CLAUDE.md as sibling tasks complete
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
    # Layer 1: Identity (static)
    # =========================================================================

    def build_identity_layer(self) -> str:
        """Layer 1: Static. Same across all workers in a run.

        Contents:
        - BashGym worker role description
        - Coding standards and quality expectations
        - Tool usage guidelines
        - General rules (don't commit unless asked, etc.)
        """
        parts = [
            "## Identity",
            "",
            f"You are a BashGym worker in a multi-agent development session.",
            f"Project: {self._spec_title}",
            "",
            "### Coding Standards",
            "- Write clean, well-typed Python 3.10+ with type hints",
            "- Use dataclasses for configuration and result objects",
            "- Keep functions focused and under 50 lines where practical",
            "- Add docstrings to public classes and methods",
            "",
            "### Rules",
            "- Only modify files you own. Read any file freely.",
            "- If you need a function/class from another task, "
            "import it — do not redefine it.",
            "- Keep changes minimal and focused on your task.",
            "- Run tests after making changes to verify correctness.",
            "- Do not commit unless explicitly asked.",
            "",
        ]
        return "\n".join(parts)

    # =========================================================================
    # Layer 2: Narrative (dynamic, deterministic — no LLM call)
    # =========================================================================

    def build_narrative_layer(
        self,
        task: TaskNode,
        dag: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> str:
        """Layer 2: Dynamic context, deterministic (no LLM call).

        Contents:
        - Sibling task table (what's running, completed, blocked)
        - File ownership map (which worker owns which files)
        - Interface contracts (provides/consumes from DAG)
        - Shared discoveries (from shared_state if available)
        - Completion updates from siblings
        """
        parts = [
            "## Context",
            "",
        ]

        # ---- Sibling task table ----
        siblings = self._get_siblings(task)
        if siblings:
            parts.append("### Sibling Tasks")
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

        # ---- File ownership map ----
        owned, forbidden = self._resolve_file_ownership(task)
        if owned or forbidden:
            parts.append("### File Ownership")
            parts.append("")
            if owned:
                parts.append(f"You OWN: {', '.join(owned)}")
            if forbidden:
                parts.append("")
                parts.append("Do NOT modify (owned by other workers):")
                for filepath, owner_id in forbidden:
                    parts.append(f"  - `{filepath}` ({owner_id})")
            parts.append("")

        # ---- Conflict warnings ----
        task_conflicts = [
            (a, b, files) for a, b, files in self._file_conflicts
            if task.id in (a, b)
        ]
        if task_conflicts:
            parts.append("### File Conflict Warnings")
            parts.append("")
            for a, b, files in task_conflicts:
                other = b if a == task.id else a
                parts.append(
                    f"- **{other}** also touches: {', '.join(files)}"
                )
            parts.append("")

        # ---- Interface contracts: what this task provides ----
        if task.provides:
            parts.append("### You PROVIDE")
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

            consumers = self._find_consumers(task.id)
            if consumers:
                parts.append("")
                parts.append("Consumed by: " + ", ".join(consumers))
            parts.append("")

        # ---- Interface contracts: what this task consumes ----
        if task.consumes:
            parts.append("### You CONSUME")
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

        # ---- Shared discoveries (from shared_state if available) ----
        if shared_state is not None:
            discoveries = self._format_shared_discoveries(shared_state)
            if discoveries:
                parts.append("### Shared Discoveries")
                parts.append("")
                parts.append(discoveries)
                parts.append("")

        # ---- Completion updates placeholder ----
        parts.append("### Completed Dependencies")
        parts.append("")
        parts.append("(updates appended here as dependencies finish)")
        parts.append("")

        return "\n".join(parts)

    # =========================================================================
    # Layer 3: Focus (task-specific)
    # =========================================================================

    def build_focus_layer(self, task: TaskNode) -> str:
        """Layer 3: Task-specific requirements.

        Contents:
        - Task title and description
        - Files to touch
        - Dependencies (what must complete first)
        - Budget constraints
        - Specific success criteria
        """
        parts = [
            "## Task",
            "",
            f"### [{task.id}]: {task.title}",
            "",
        ]

        if task.description:
            parts.append(task.description)
            parts.append("")

        if task.worker_prompt:
            parts.append("### Instructions")
            parts.append("")
            parts.append(task.worker_prompt)
            parts.append("")

        if task.files_touched:
            parts.append("### Files to Touch")
            parts.append("")
            for f in task.files_touched:
                parts.append(f"- `{f}`")
            parts.append("")

        if task.dependencies:
            dep_names = []
            for dep_id in task.dependencies:
                dep = self._nodes.get(dep_id)
                if dep:
                    dep_names.append(f"{dep_id} ({dep.title})")
                else:
                    dep_names.append(dep_id)
            parts.append("### Dependencies")
            parts.append("")
            parts.append(
                "These tasks must complete before yours: "
                + ", ".join(dep_names)
            )
            parts.append("")

        # Budget constraints
        parts.append("### Constraints")
        parts.append("")
        parts.append(f"- Estimated turns: {task.estimated_turns}")
        parts.append(f"- Budget: ${task.budget_usd:.2f}")
        parts.append("")

        return "\n".join(parts)

    # =========================================================================
    # Transition Markers (sibling updates)
    # =========================================================================

    def build_transition_marker(
        self,
        completed_task: TaskNode,
        result: WorkerResult,
    ) -> str:
        """Transition marker for sibling updates.

        Contents:
        - What was completed (summary)
        - Files that were modified
        - What's now available (provides from completed task)
        - Reflection prompt: 'Consider if this affects your approach.'
        """
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        parts = [
            "",
            f"---",
            f"**Transition: [{completed_task.id}] {completed_task.title} "
            f"completed at {timestamp}**",
            "",
        ]

        if result.files_modified:
            parts.append(
                f"Files changed: {', '.join(result.files_modified)}"
            )

        # What was provided
        if completed_task.provides:
            for contract in completed_task.provides:
                file = contract.get("file", "")
                exports = contract.get("exports", [])
                if exports:
                    parts.append(
                        f"Now available: `{file}` exports "
                        f"{', '.join(exports)}"
                    )

        if result.error:
            parts.append(f"Note: completed with warnings — {result.error}")

        parts.append("")
        parts.append(
            "*Consider if this affects your current approach or "
            "unblocks any part of your task.*"
        )

        return "\n".join(parts)

    # =========================================================================
    # Public Interface (preserved signatures)
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

    def build_claude_md(
        self,
        task: TaskNode,
        dag: Optional[Any] = None,
        shared_state: Optional[Any] = None,
    ) -> str:
        """Assembles all three layers with clear section markers.

        When dag or shared_state is provided, they are passed to the
        narrative layer for richer context. Preserves existing behavior
        when called with just task.
        """
        parts = [
            f"# Orchestration Context for [{task.id}]: {task.title}",
            "",
        ]

        # Layer 1: Identity
        parts.append(self.build_identity_layer())

        # Layer 2: Narrative
        parts.append(self.build_narrative_layer(
            task, dag=dag, shared_state=shared_state
        ))

        # Layer 3: Focus
        parts.append(self.build_focus_layer(task))

        return "\n".join(parts)

    def build_update(
        self,
        completed_task_id: str,
        result: WorkerResult,
    ) -> str:
        """Generate a completion notification to append to a worker's CLAUDE.md.

        Uses build_transition_marker internally for richer context.

        Called when a sibling task completes. The text is appended to the
        running worker's CLAUDE.md file so it picks up the new context
        on its next CLAUDE.md re-read.
        """
        completed = self._nodes.get(completed_task_id)
        if not completed:
            return ""

        return self.build_transition_marker(completed, result)

    # =========================================================================
    # CLAUDE.md File Operations
    # =========================================================================

    def write_claude_md(
        self,
        task: TaskNode,
        worktree_path: Path,
        shared_state: Optional["SharedState"] = None,
    ) -> Path:
        """Write orchestration context to CLAUDE.md in the worktree.

        If the worktree already contains a CLAUDE.md (from the repo),
        the orchestration context is prepended and the original content
        is preserved below a separator.

        Args:
            task: The task node for this worker
            worktree_path: Path to the worker's git worktree
            shared_state: Optional SharedState for including discoveries

        Returns:
            Path to the written CLAUDE.md
        """
        claude_md_path = worktree_path / "CLAUDE.md"
        orchestration_context = self.build_claude_md(
            task, shared_state=shared_state
        )

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

    def _format_shared_discoveries(self, shared_state: Any) -> str:
        """Format shared_state into human-readable discoveries for CLAUDE.md.

        Handles both SharedState objects (with snapshot/get_history) and
        plain dicts. Falls back gracefully if the format is unexpected.
        """
        if not shared_state:
            return ""

        lines: List[str] = []
        try:
            # SharedState object with snapshot() method
            if hasattr(shared_state, "snapshot"):
                data = shared_state.snapshot()
                if not data:
                    return ""
                lines.append(
                    "Other workers have shared the following discoveries:"
                )
                lines.append("")
                for key, value in sorted(data.items()):
                    # Truncate long values for readability
                    value_str = str(value)
                    if len(value_str) > 300:
                        value_str = value_str[:297] + "..."
                    lines.append(f"- **{key}**: {value_str}")

                # Show recent writers for attribution
                if hasattr(shared_state, "get_history"):
                    recent = shared_state.get_history(limit=5)
                    if recent:
                        writers = sorted(
                            set(c.writer_id for c in recent)
                        )
                        lines.append("")
                        lines.append(
                            f"Recent contributors: {', '.join(writers)}"
                        )

            # Plain dict
            elif hasattr(shared_state, "items"):
                if not shared_state:
                    return ""
                for key, value in shared_state.items():
                    lines.append(f"- **{key}**: {value}")

            else:
                lines.append(f"- {shared_state}")

        except Exception:
            logger.debug("Could not format shared_state, skipping")
            return ""

        return "\n".join(lines)
