"""Shared pytest configuration.

Tests marked ``@pytest.mark.network`` make real outbound calls (NVIDIA/HF APIs,
SSH, Ollama subprocesses). They are SKIPPED by default so the suite stays
hermetic and never hangs on a blocked socket — run them explicitly with
``pytest --run-network`` against live services.

The February 2026 multi-agent Orchestrator dashboard is a retained legacy
feature, not the AutoResearch campaign controller. Its process/worktree-heavy
suite is excluded from normal collection; pass ``--run-legacy-orchestrator``
when maintaining that feature explicitly.
"""

from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parent
_LEGACY_ORCHESTRATOR_TESTS = _REPOSITORY_ROOT / "tests" / "orchestrator"


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (real external services).",
    )
    parser.addoption(
        "--run-legacy-orchestrator",
        action="store_true",
        default=False,
        help="Collect the retained legacy multi-agent Orchestrator test suite.",
    )


def pytest_ignore_collect(collection_path, config):
    """Keep the unused legacy Orchestrator out of the default developer loop."""

    if config.getoption("--run-legacy-orchestrator"):
        return False
    candidate = Path(str(collection_path)).resolve()
    return candidate == _LEGACY_ORCHESTRATOR_TESTS or _LEGACY_ORCHESTRATOR_TESTS in (
        candidate.parents
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="network test; pass --run-network to include")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
