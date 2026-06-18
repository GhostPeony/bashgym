"""BashGym Data Designer plugins.

Exposes BashGym capabilities as first-class Data Designer extensions (usable from
the ``data-designer`` CLI and any DD config). Currently a SEED_READER that reads
BashGym's *processed* gold-trace schema (``data/gold_traces/*.json``) — the
complement to DD's native ``AgentRolloutSeedSource``, which reads *raw* rollouts.

Registered via the ``data_designer.plugins`` entry point in pyproject.toml; load
with ``pip install -e .`` so the ``data-designer`` CLI discovers it.
"""
