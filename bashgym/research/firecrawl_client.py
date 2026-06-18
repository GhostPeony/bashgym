"""Firecrawl Research client — query the academic-paper + GitHub research index.

Wraps Firecrawl's research endpoints (``/v2/search/research/papers`` and
``/v2/search/research/github``) so BashGym can ground its AutoResearch loop and a
platform news feed in the literature + GitHub history of the training ecosystem.

This is an *advisory* surface: the HTTP transport is injected (so it is unit-
testable with ``httpx.MockTransport``) and every request degrades to an empty
result on a missing key or any error — it never sits on a training-critical path.
The key comes from ``FIRECRAWL_API_KEY``; no new dependency (``httpx`` is already
in the stack) and no settings-schema change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://api.firecrawl.dev/v2"

# Repos whose issues/PRs/discussions drive BashGym's training stack — the default
# scope for the news feed so it surfaces signal (kernels, loss variants, recipes).
TRACKED_REPOS: tuple[str, ...] = (
    "unslothai/unsloth",
    "huggingface/trl",
    "huggingface/transformers",
    "NVIDIA-NeMo/NeMo",
    "vllm-project/vllm",
    "linkedin/Liger-Kernel",
)


@dataclass
class ResearchPaper:
    paper_id: str
    title: str
    abstract: str = ""
    score: float = 0.0
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "score": self.score,
            "url": self.url,
        }


@dataclass
class GithubFinding:
    repository: str
    url: str
    snippet: str = ""
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "repository": self.repository,
            "url": self.url,
            "snippet": self.snippet,
            "title": self.title,
        }


def _items(data: Any) -> list[dict]:
    """Pull the result list out of Firecrawl's response (dict-wrapped or bare list)."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "results", "papers", "items", "hits"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _paper_url(item: dict) -> str:
    url = item.get("url") or item.get("pdfUrl") or ""
    if url:
        return str(url)
    pid = str(item.get("primaryId") or item.get("paperId") or "")
    if pid.startswith("arxiv:"):
        return f"https://arxiv.org/abs/{pid.split(':', 1)[1]}"
    return ""


def _parse_papers(data: Any) -> list[ResearchPaper]:
    out: list[ResearchPaper] = []
    for item in _items(data):
        pid = item.get("paperId") or item.get("primaryId") or item.get("id")
        if not pid:
            continue
        out.append(
            ResearchPaper(
                paper_id=str(pid),
                title=str(item.get("title") or "").strip(),
                abstract=str(item.get("abstract") or item.get("summary") or "").strip(),
                score=float(item.get("score") or 0.0),
                url=_paper_url(item),
            )
        )
    return out


def _parse_github(data: Any) -> list[GithubFinding]:
    out: list[GithubFinding] = []
    for item in _items(data):
        url = item.get("url") or item.get("html_url")
        if not url:
            continue
        out.append(
            GithubFinding(
                repository=str(item.get("repository") or item.get("repo") or "").strip(),
                url=str(url),
                snippet=str(item.get("snippet") or item.get("content") or "").strip(),
                title=str(item.get("title") or "").strip(),
            )
        )
    return out


@dataclass
class FirecrawlResearchClient:
    """Async client for Firecrawl's research index. Inject ``client`` in tests."""

    api_key: str | None = None
    base_url: str = BASE_URL
    timeout: float = 30.0
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("FIRECRAWL_API_KEY")
        self.base_url = self.base_url.rstrip("/")
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

    @property
    def configured(self) -> bool:
        """True when an API key is available (the feed/advisor degrade without one)."""
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        try:
            resp = await self.client.get(
                f"{self.base_url}{path}", params=clean, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:  # noqa: BLE001 - advisory: any failure yields no results
            return None

    async def search_papers(
        self,
        query: str,
        *,
        k: int = 10,
        categories: str | None = None,
        since: str | None = None,
        until: str | None = None,
        authors: str | None = None,
    ) -> list[ResearchPaper]:
        """Semantic paper search. ``since``/``until`` are ``YYYY-MM-DD`` bounds."""
        data = await self._get(
            "/search/research/papers",
            {
                "query": query,
                "k": k,
                "categories": categories,
                "from": since,
                "to": until,
                "authors": authors,
            },
        )
        return _parse_papers(data)

    async def search_github(self, query: str, *, k: int = 10) -> list[GithubFinding]:
        """Search GitHub issues/PRs/discussions/docs across the research index."""
        data = await self._get("/search/research/github", {"query": query, "k": k})
        return _parse_github(data)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
