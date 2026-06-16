"""Shared pytest configuration.

Tests marked ``@pytest.mark.network`` make real outbound calls (NVIDIA/HF APIs,
SSH, Ollama subprocesses). They are SKIPPED by default so the suite stays
hermetic and never hangs on a blocked socket — run them explicitly with
``pytest --run-network`` against live services.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (real external services).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="network test; pass --run-network to include")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
