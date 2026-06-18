"""Config class for the BashGym gold-trace seed reader plugin."""

from __future__ import annotations

from typing import Literal

from data_designer.config.seed_source import FileSystemSeedSource


class GoldTraceSeedSource(FileSystemSeedSource):
    """Seed dataset source for BashGym's processed gold traces.

    Reads ``data/gold_traces/*.json`` (BashGym's classified trace schema) and
    emits one seed row per trace with the same fields the Factory's ``from_traces``
    derives (``seed_task``/``seed_tools``/``seed_complexity``/``seed_language``/
    ``seed_step_count``), so any Data Designer pipeline can seed from gold traces.

    Inherited fields: ``path`` (gold-traces directory), ``recursive``.
    """

    seed_type: Literal["bashgym_gold_trace"] = "bashgym_gold_trace"
    # Gold traces are *.json; default the pattern so callers only pass `path`.
    file_pattern: str = "*.json"
