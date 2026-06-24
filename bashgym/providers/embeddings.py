"""Embedding response parsing and a sync adapter for rwml.

Two concerns live here, both deliberately free of network/IO:

1. Pure parsers that turn a provider's JSON embeddings payload into a plain
   ``list[list[float]]`` (one vector per input, input order preserved). These
   are unit-testable without httpx.
2. ``make_embed_fn``: wraps a provider's async ``embed`` into the synchronous
   ``Callable[[str], Sequence[float]]`` that ``bashgym.gym.rwml`` expects as
   its ``embed_fn`` (single string -> single vector).

No bashgym imports here beyond typing, to avoid circular dependencies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Protocol


def parse_embeddings_response(payload: dict[str, Any]) -> list[list[float]]:
    """Parse an OpenAI ``/v1/embeddings`` payload into ``list[list[float]]``.

    Expects ``{"data": [{"embedding": [...], "index": i}, ...]}``. Entries are
    sorted by ``index`` so the output order matches the request input order.
    Missing/empty ``data`` yields an empty list.
    """
    data = payload.get("data") or []
    ordered = sorted(data, key=lambda item: item.get("index", 0))
    return [[float(x) for x in item.get("embedding", [])] for item in ordered]


def parse_ollama_embeddings_response(payload: dict[str, Any]) -> list[list[float]]:
    """Parse an Ollama embeddings payload into ``list[list[float]]``.

    Handles both the current ``/api/embed`` shape
    (``{"embeddings": [[...], ...]}``) and the legacy ``/api/embeddings`` shape
    (``{"embedding": [...]}``, a single vector). Empty payload yields ``[]``.
    """
    if "embeddings" in payload:
        return [[float(x) for x in vec] for vec in (payload.get("embeddings") or [])]
    if "embedding" in payload:
        return [[float(x) for x in (payload.get("embedding") or [])]]
    return []


class _EmbeddingProvider(Protocol):
    """Minimal structural type: anything with an async ``embed``."""

    async def embed(self, texts: list[str], *, model: str | None = ...) -> list[list[float]]: ...


def make_embed_fn(
    provider: _EmbeddingProvider, model: str | None = None
) -> Callable[[str], Sequence[float]]:
    """Adapt a provider's async ``embed`` to rwml's sync single-string contract.

    Returns ``Callable[[str], Sequence[float]]`` (rwml's ``EmbedFn``): it embeds
    one string and returns that string's vector, driving the provider's
    coroutine to completion synchronously. ``model`` (if given) is forwarded on
    every call; the caller picks it from the live catalog (nothing is hardcoded
    here).
    """

    def embed_fn(text: str) -> Sequence[float]:
        vectors = _run_sync(provider.embed([text], model=model))
        if not vectors:
            raise RuntimeError("embed returned no vectors")
        return vectors[0]

    return embed_fn


def _run_sync(coro: Any) -> Any:
    """Run a coroutine to completion from sync code.

    Uses ``asyncio.run`` when no loop is running. If called from within a
    running loop, runs the coroutine on a dedicated loop in a worker thread so
    we never call ``asyncio.run`` re-entrantly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
