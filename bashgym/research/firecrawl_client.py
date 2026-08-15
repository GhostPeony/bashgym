"""Bounded Firecrawl search for research context.

The current Firecrawl surface is one ``/v2/search`` endpoint.  BashGym prefers
the reviewed local CLI when it is installed and falls back to the HTTP API when
an API key is available.  Retrieval is advisory: failures are explicit typed
results and never become evaluation or promotion evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

BASE_URL = "https://api.firecrawl.dev/v2"
ALLOWED_CATEGORIES = frozenset({"research", "github", "pdf"})

TRACKED_REPOS: tuple[str, ...] = (
    "unslothai/unsloth",
    "huggingface/trl",
    "huggingface/transformers",
    "NVIDIA-NeMo/NeMo",
    "vllm-project/vllm",
    "linkedin/Liger-Kernel",
)


@dataclass(frozen=True)
class ResearchSource:
    """Small normalized search result; page bodies are intentionally excluded."""

    title: str
    url: str
    summary: str = ""
    category: str = ""
    published_at: str | None = None
    source_id: str = ""
    repository: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class ResearchSearchResult:
    query: str
    categories: tuple[str, ...]
    sources: tuple[ResearchSource, ...]
    provider: Literal["firecrawl_cli", "firecrawl_api", "none"]
    status: Literal["available", "unavailable"]
    code: str | None = None
    search_id: str | None = None
    credits_used: float | int | None = None


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


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _cli_argv_prefix(executable: str) -> tuple[str, ...]:
    """Resolve npm's Windows batch shim to its fixed Node entrypoint."""

    path = Path(executable)
    if path.suffix.casefold() not in {".cmd", ".bat"}:
        return (executable,)
    entrypoint = path.parent / "node_modules" / "firecrawl-cli" / "dist" / "index.js"
    local_node = path.parent / "node.exe"
    node = str(local_node) if local_node.is_file() else shutil.which("node")
    if not entrypoint.is_file() or node is None:
        return ()
    return (node, str(entrypoint))


def _default_command_runner(
    argv: Sequence[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executable and argv are typed, no shell
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        shell=False,
    )


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _result_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        web = data.get("web")
        if isinstance(web, list):
            return [item for item in web if isinstance(item, dict)]
        for key in ("results", "papers", "items", "hits"):
            items = data.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _valid_response_envelope(payload: Any) -> bool:
    if isinstance(payload, list):
        return True
    if not isinstance(payload, dict) or payload.get("success") is False:
        return False
    return isinstance(payload.get("data"), (list, dict))


def _paper_url(item: dict[str, Any]) -> str:
    url = _bounded_text(item.get("url") or item.get("pdfUrl"), 2_000)
    if url:
        return url
    paper_id = _bounded_text(item.get("primaryId") or item.get("paperId") or item.get("id"), 500)
    if paper_id.startswith("arxiv:"):
        return f"https://arxiv.org/abs/{paper_id.split(':', 1)[1]}"
    return ""


def _normalize_sources(payload: Any, *, category: str, limit: int) -> tuple[ResearchSource, ...]:
    sources: list[ResearchSource] = []
    for item in _result_items(payload):
        url = _paper_url(item)
        source_id = _bounded_text(
            item.get("paperId") or item.get("primaryId") or item.get("id"), 500
        )
        if not url and category == "research" and not source_id:
            continue
        if not url:
            url = _bounded_text(item.get("html_url"), 2_000)
        if not url and category != "research":
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        sources.append(
            ResearchSource(
                title=_bounded_text(item.get("title"), 500),
                url=url,
                summary=_bounded_text(
                    item.get("description")
                    or item.get("abstract")
                    or item.get("summary")
                    or item.get("snippet")
                    or item.get("content"),
                    1_000,
                ),
                category=_bounded_text(item.get("category") or category, 50),
                published_at=_bounded_text(item.get("publishedAt") or item.get("published_at"), 100)
                or None,
                source_id=source_id,
                repository=_bounded_text(item.get("repository") or item.get("repo"), 500),
                score=score,
            )
        )
        if len(sources) >= limit:
            break
    return tuple(sources)


def _failure_code(message: str) -> str:
    normalized = message.lower()
    if "402" in normalized or "credit" in normalized or "quota" in normalized:
        return "firecrawl_quota_exhausted"
    if any(token in normalized for token in ("401", "403", "auth", "unauthorized")):
        return "firecrawl_auth_required"
    return "firecrawl_provider_error"


@dataclass
class FirecrawlResearchClient:
    """CLI-first bounded research search with an HTTP fallback."""

    api_key: str | None = None
    base_url: str = BASE_URL
    timeout: float = 30.0
    client: httpx.AsyncClient | None = field(default=None, repr=False)
    prefer_cli: bool = True
    cli_executable: str | None = None
    command_runner: CommandRunner | None = field(default=None, repr=False)
    last_result: ResearchSearchResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("FIRECRAWL_API_KEY")
        self.base_url = self.base_url.rstrip("/")
        if self.prefer_cli and self.cli_executable is None:
            self.cli_executable = shutil.which("firecrawl")
        if not self.prefer_cli:
            self.cli_executable = ""
        if self.command_runner is None:
            self.command_runner = _default_command_runner
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

    @property
    def configured(self) -> bool:
        return bool(self.cli_executable or self.api_key)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _validate_search(
        query: str, categories: Sequence[str], k: int
    ) -> tuple[str, tuple[str, ...]]:
        clean_query = query.strip()
        clean_categories = tuple(dict.fromkeys(str(item).strip() for item in categories))
        if (
            not clean_query
            or len(clean_query) > 1_000
            or not clean_categories
            or not 1 <= k <= 10
            or any(item not in ALLOWED_CATEGORIES for item in clean_categories)
        ):
            raise ValueError("firecrawl_search_contract_invalid")
        return clean_query, clean_categories

    def _unavailable(
        self,
        query: str,
        categories: tuple[str, ...],
        provider: Literal["firecrawl_cli", "firecrawl_api", "none"],
        code: str,
    ) -> ResearchSearchResult:
        result = ResearchSearchResult(
            query=query,
            categories=categories,
            sources=(),
            provider=provider,
            status="unavailable",
            code=code,
        )
        self.last_result = result
        return result

    def _available(
        self,
        query: str,
        categories: tuple[str, ...],
        provider: Literal["firecrawl_cli", "firecrawl_api"],
        payload: Any,
        k: int,
    ) -> ResearchSearchResult:
        sources: list[ResearchSource] = []
        for category in categories:
            for source in _normalize_sources(payload, category=category, limit=k):
                if source.url not in {existing.url for existing in sources}:
                    sources.append(source)
                if len(sources) >= k:
                    break
            if len(sources) >= k:
                break
        metadata = payload if isinstance(payload, dict) else {}
        result = ResearchSearchResult(
            query=query,
            categories=categories,
            sources=tuple(sources),
            provider=provider,
            status="available",
            search_id=_bounded_text(metadata.get("id"), 500) or None,
            credits_used=metadata.get("creditsUsed"),
        )
        self.last_result = result
        return result

    async def search(
        self, query: str, *, categories: Sequence[str], k: int = 10
    ) -> ResearchSearchResult:
        query, normalized_categories = self._validate_search(query, categories, k)

        cli_failure_code: str | None = None
        cli_prefix = _cli_argv_prefix(self.cli_executable) if self.cli_executable else ()
        if self.cli_executable and not cli_prefix:
            self.cli_executable = ""
        if cli_prefix:
            argv = [
                *cli_prefix,
                "search",
                query,
                "--categories",
                ",".join(normalized_categories),
                "--limit",
                str(k),
                "--json",
            ]
            try:
                completed = await asyncio.to_thread(self.command_runner, argv, self.timeout)
            except (FileNotFoundError, OSError):
                self.cli_executable = ""
            except subprocess.TimeoutExpired:
                cli_failure_code = "firecrawl_timeout"
            else:
                if completed.returncode != 0:
                    cli_failure_code = _failure_code(completed.stderr)
                else:
                    try:
                        payload = json.loads(completed.stdout)
                    except (TypeError, json.JSONDecodeError):
                        cli_failure_code = "firecrawl_invalid_response"
                    else:
                        if _valid_response_envelope(payload):
                            return self._available(
                                query, normalized_categories, "firecrawl_cli", payload, k
                            )
                        cli_failure_code = "firecrawl_invalid_response"

        if cli_failure_code and not self.api_key:
            return self._unavailable(
                query, normalized_categories, "firecrawl_cli", cli_failure_code
            )

        if not self.api_key:
            return self._unavailable(
                query, normalized_categories, "none", "firecrawl_cli_unavailable"
            )

        body = {
            "query": query,
            "categories": [{"type": category} for category in normalized_categories],
            "limit": k,
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/search", json=body, headers=self._headers()
            )
        except httpx.TimeoutException:
            return self._unavailable(
                query, normalized_categories, "firecrawl_api", "firecrawl_timeout"
            )
        except httpx.HTTPError:
            return self._unavailable(
                query, normalized_categories, "firecrawl_api", "firecrawl_provider_error"
            )
        if response.status_code >= 400:
            return self._unavailable(
                query,
                normalized_categories,
                "firecrawl_api",
                _failure_code(f"{response.status_code} {response.text[:200]}"),
            )
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return self._unavailable(
                query,
                normalized_categories,
                "firecrawl_api",
                "firecrawl_invalid_response",
            )
        if not _valid_response_envelope(payload):
            return self._unavailable(
                query,
                normalized_categories,
                "firecrawl_api",
                "firecrawl_invalid_response",
            )
        return self._available(query, normalized_categories, "firecrawl_api", payload, k)

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
        del categories, since, until, authors  # current search API has no equivalent filters
        result = await self.search(query, categories=("research",), k=k)
        if result.status != "available":
            return []
        papers: list[ResearchPaper] = []
        for source in result.sources:
            paper_id = source.source_id
            if not paper_id:
                match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", source.url)
                paper_id = f"arxiv:{match.group(1)}" if match else ""
            if not paper_id:
                continue
            papers.append(
                ResearchPaper(
                    paper_id=paper_id,
                    title=source.title,
                    abstract=source.summary,
                    score=source.score,
                    url=source.url,
                )
            )
        return papers

    async def search_github(self, query: str, *, k: int = 10) -> list[GithubFinding]:
        result = await self.search(query, categories=("github",), k=k)
        if result.status != "available":
            return []
        return [
            GithubFinding(
                repository=source.repository,
                url=source.url,
                snippet=source.summary,
                title=source.title,
            )
            for source in result.sources
        ]

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
