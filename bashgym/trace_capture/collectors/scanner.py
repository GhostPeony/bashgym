"""
ClaudeDataScanner Orchestrator

Provides a single interface to orchestrate all seven concrete collectors
(SubagentCollector, EditCollector, PlanCollector, PromptCollector,
TodoCollector, EnvironmentCollector, DebugCollector) for comprehensive
.claude data capture.

Usage::

    scanner = ClaudeDataScanner()
    status  = scanner.status()                   # quick overview
    results = scanner.scan_all()                  # dry-run scan
    results = scanner.collect_all()               # collect everything
    result  = scanner.collect_source("plans")     # collect one source
    results = scanner.collect_all(sources=["plans", "prompts"])  # subset
"""

from pathlib import Path
from typing import Any

from .base import (
    CollectorBatchResult,
    CollectorScanResult,
    get_claude_dir,
    get_collected_dir,
)
from .debug import DebugCollector
from .edit import EditCollector
from .environment import EnvironmentCollector
from .plan import PlanCollector
from .prompt import PromptCollector
from .subagent import SubagentCollector
from .todo import TodoCollector

ALL_SOURCES: list[str] = [
    "subagents",
    "edits",
    "plans",
    "prompts",
    "todos",
    "environments",
    "debug",
]


class ClaudeDataScanner:
    """Orchestrates all collectors for comprehensive .claude data capture.

    Each method accepts optional ``sources``, ``since``, and
    ``project_filter`` parameters so callers can narrow the scope of a
    scan or collection run without touching individual collectors.

    Attributes
    ----------
    claude_dir : Path
        Root of the Claude Code configuration directory (``~/.claude/``).
    collected_dir : Path
        Root of the output directory (``~/.bashgym/collected/``).
    """

    def __init__(self) -> None:
        self.claude_dir: Path = get_claude_dir()
        self.collected_dir: Path = get_collected_dir()
        self._collectors = {
            "subagents": SubagentCollector,
            "edits": EditCollector,
            "plans": PlanCollector,
            "prompts": PromptCollector,
            "todos": TodoCollector,
            "environments": EnvironmentCollector,
            "debug": DebugCollector,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collector(self, source_type: str):
        """Create a collector instance with this scanner's directories.

        Parameters
        ----------
        source_type : str
            One of the keys in ``_collectors``.

        Returns
        -------
        BaseCollector
            A freshly constructed collector.

        Raises
        ------
        KeyError
            If *source_type* is not a recognised collector name.
        """
        cls = self._collectors[source_type]
        return cls(
            claude_dir=self.claude_dir,
            collected_dir=self.collected_dir,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan_all(
        self,
        sources: list[str] | None = None,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> dict[str, CollectorScanResult]:
        """Dry-run scan of all (or selected) sources.

        Parameters
        ----------
        sources : list of str, optional
            Restrict to these source types.  Defaults to all.
        since : str, optional
            ISO-8601 timestamp — only data newer than this is considered.
        project_filter : str, optional
            Substring match on project name.

        Returns
        -------
        dict mapping source name to CollectorScanResult
        """
        sources = sources or ALL_SOURCES
        results: dict[str, CollectorScanResult] = {}
        for source in sources:
            if source in self._collectors:
                results[source] = self._get_collector(source).scan(
                    since=since,
                    project_filter=project_filter,
                )
        return results

    def collect_all(
        self,
        sources: list[str] | None = None,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> dict[str, CollectorBatchResult]:
        """Collect from all (or selected) sources.

        Parameters
        ----------
        sources : list of str, optional
            Restrict to these source types.  Defaults to all.
        since : str, optional
            ISO-8601 timestamp — only data newer than this is collected.
        project_filter : str, optional
            Substring match on project name.

        Returns
        -------
        dict mapping source name to CollectorBatchResult
        """
        sources = sources or ALL_SOURCES
        results: dict[str, CollectorBatchResult] = {}
        for source in sources:
            if source in self._collectors:
                results[source] = self._get_collector(source).collect_all(
                    since=since,
                    project_filter=project_filter,
                )
        return results

    def collect_source(
        self,
        source: str,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorBatchResult:
        """Collect a single source type.

        Parameters
        ----------
        source : str
            The source type to collect (e.g. ``"plans"``).
        since : str, optional
            ISO-8601 timestamp.
        project_filter : str, optional
            Substring match on project name.

        Returns
        -------
        CollectorBatchResult

        Raises
        ------
        KeyError
            If *source* is not a recognised collector name.
        """
        return self._get_collector(source).collect_all(
            since=since,
            project_filter=project_filter,
        )

    def build_index(self) -> dict[str, Any]:
        """Build or rebuild the cross-reference index.

        Walks ``collected_dir`` and groups all records by ``session_id``.

        Returns
        -------
        dict
            The cross-reference index.
        """
        from .index import build_cross_reference_index

        return build_cross_reference_index(self.collected_dir)

    def status(self) -> dict[str, dict]:
        """Get collection status for all sources.

        Returns
        -------
        dict
            Mapping of source name to ``{"total": int, "collected": int,
            "available": int}``.
        """
        status: dict[str, dict] = {}
        for source in ALL_SOURCES:
            scan = self._get_collector(source).scan()
            status[source] = {
                "total": scan.total_found,
                "collected": scan.already_collected,
                "available": scan.new_available,
            }
        return status
