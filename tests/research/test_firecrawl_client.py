"""Hermetic tests for the Firecrawl research client (httpx.MockTransport)."""

from __future__ import annotations

import json
import subprocess

import httpx

from bashgym.research import firecrawl_client
from bashgym.research.firecrawl_client import FirecrawlResearchClient


def _client(handler, api_key="fc-test"):
    return FirecrawlResearchClient(
        api_key=api_key,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        prefer_cli=False,
    )


def test_default_cli_runner_suppresses_the_windows_console(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(firecrawl_client.os, "name", "nt")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    firecrawl_client._default_command_runner(["firecrawl", "--status"], 12.0)

    assert captured["argv"] == ["firecrawl", "--status"]
    assert captured["kwargs"]["creationflags"] == 0x08000000


async def test_search_papers_parses_auth_and_params():
    captured = {}

    def handler(req):
        captured["path"] = req.url.path
        captured["auth"] = req.headers.get("authorization")
        captured["json"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "arxiv:1706.03762",
                        "title": "Attention Is All You Need",
                        "abstract": "The Transformer...",
                        "score": 0.91,
                    }
                ]
            },
        )

    papers = await _client(handler).search_papers("transformers", k=5, categories="cs.LG")
    assert len(papers) == 1
    p = papers[0]
    assert p.paper_id == "arxiv:1706.03762"
    assert p.url == "https://arxiv.org/abs/1706.03762"  # derived from arxiv id
    assert p.score == 0.91
    assert captured["path"].endswith("/v2/search")
    assert captured["auth"] == "Bearer fc-test"
    assert captured["json"]["categories"] == [{"type": "research"}]
    assert captured["json"]["limit"] == 5


async def test_search_github_parses():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "repository": "huggingface/trl",
                        "url": "https://github.com/huggingface/trl/pull/9",
                        "snippet": "Add AsyncGRPO",
                        "title": "AsyncGRPO",
                    }
                ]
            },
        )

    res = await _client(handler).search_github("grpo async", k=3)
    assert res[0].repository == "huggingface/trl"
    assert res[0].url.endswith("/pull/9")


async def test_cli_search_uses_typed_argv_and_parses_research_results():
    captured = {}

    def runner(argv, timeout):
        captured["argv"] = argv
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"success":true,"data":{"web":[{"url":"https://arxiv.org/abs/2608.09696","title":"Model Discovery Agent","description":"Bayesian experiment design","category":"research"}]},"id":"search-1","creditsUsed":1}',
            stderr="",
        )

    client = FirecrawlResearchClient(
        api_key="",
        cli_executable="C:/tools/firecrawl.exe",
        command_runner=runner,
    )
    result = await client.search(
        "information gain for experiments",
        categories=("research", "pdf"),
        k=3,
    )

    assert captured["argv"] == [
        "C:/tools/firecrawl.exe",
        "search",
        "information gain for experiments",
        "--categories",
        "research,pdf",
        "--limit",
        "3",
        "--json",
    ]
    assert captured["timeout"] == 30.0
    assert result.status == "available"
    assert result.provider == "firecrawl_cli"
    assert result.search_id == "search-1"
    assert result.credits_used == 1
    assert result.sources[0].title == "Model Discovery Agent"
    assert result.sources[0].category == "research"


async def test_windows_npm_shim_is_resolved_to_node_without_a_shell(tmp_path):
    shim = tmp_path / "firecrawl.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")
    node = tmp_path / "node.exe"
    node.write_bytes(b"")
    entrypoint = tmp_path / "node_modules" / "firecrawl-cli" / "dist" / "index.js"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    observed = []

    def runner(argv, _timeout):
        observed.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout='{"data": []}', stderr="")

    client = FirecrawlResearchClient(
        api_key="",
        cli_executable=str(shim),
        command_runner=runner,
    )
    try:
        await client.search("a & literal query", categories=("research",), k=1)
    finally:
        await client.close()

    assert observed[0][:2] == (str(node), str(entrypoint))
    assert observed[0][3] == "a & literal query"


async def test_cli_quota_failure_is_explicit_and_sanitized():
    def runner(argv, timeout):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="",
            stderr="Error: Request failed with status code 402. secret-local-detail",
        )

    client = FirecrawlResearchClient(
        api_key="",
        cli_executable="firecrawl",
        command_runner=runner,
    )
    result = await client.search("x", categories=("research",), k=1)

    assert result.status == "unavailable"
    assert result.code == "firecrawl_quota_exhausted"
    assert "secret-local-detail" not in str(result)


async def test_cli_timeout_and_invalid_json_are_typed():
    def timeout_runner(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    timeout_client = FirecrawlResearchClient(
        api_key="", cli_executable="firecrawl", command_runner=timeout_runner
    )
    timeout = await timeout_client.search("x", categories=("research",), k=1)
    assert timeout.code == "firecrawl_timeout"

    def invalid_runner(argv, timeout):
        return subprocess.CompletedProcess(argv, 0, stdout="not-json", stderr="")

    invalid_client = FirecrawlResearchClient(
        api_key="", cli_executable="firecrawl", command_runner=invalid_runner
    )
    invalid = await invalid_client.search("x", categories=("research",), k=1)
    assert invalid.code == "firecrawl_invalid_response"


async def test_cli_malformed_success_envelope_is_unavailable():
    def runner(argv, timeout):
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"success":false,"error":"provider failed"}', stderr=""
        )

    client = FirecrawlResearchClient(api_key="", cli_executable="firecrawl", command_runner=runner)
    result = await client.search("x", categories=("research",), k=1)

    assert result.status == "unavailable"
    assert result.code == "firecrawl_invalid_response"


async def test_cli_failure_falls_back_to_api_when_configured():
    def runner(argv, timeout):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="provider failed")

    def handler(req):
        return httpx.Response(200, json={"data": {"web": []}})

    client = FirecrawlResearchClient(
        api_key="fc-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        cli_executable="firecrawl",
        command_runner=runner,
    )
    result = await client.search("x", categories=("research",), k=1)

    assert result.status == "available"
    assert result.provider == "firecrawl_api"


async def test_api_auth_failure_is_typed_without_response_details():
    def handler(req):
        return httpx.Response(401, text="credential-specific-provider-detail")

    client = _client(handler)
    result = await client.search("x", categories=("research",), k=1)
    assert result.code == "firecrawl_auth_required"
    assert "credential-specific-provider-detail" not in str(result)


async def test_missing_cli_falls_back_to_current_api_search():
    captured = {}

    def handler(req):
        captured["path"] = req.url.path
        return httpx.Response(200, json={"data": {"web": []}})

    client = FirecrawlResearchClient(
        api_key="fc-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        cli_executable="",
    )
    result = await client.search("x", categories=("github",), k=2)

    assert captured["path"] == "/v2/search"
    assert result.status == "available"
    assert result.provider == "firecrawl_api"


async def test_search_rejects_unbounded_or_unknown_categories():
    client = FirecrawlResearchClient(api_key="", cli_executable="")

    for categories, k in [(("web",), 1), (("research",), 11), ((), 1)]:
        try:
            await client.search("x", categories=categories, k=k)
        except ValueError as exc:
            assert str(exc) == "firecrawl_search_contract_invalid"
        else:  # pragma: no cover - makes the failure message explicit
            raise AssertionError("invalid search contract was accepted")


async def test_bare_list_response_supported():
    def handler(req):
        return httpx.Response(200, json=[{"paperId": "p1", "title": "T", "abstract": "a"}])

    papers = await _client(handler).search_papers("x")
    assert papers[0].paper_id == "p1"


async def test_error_yields_empty_not_crash():
    def handler(req):
        return httpx.Response(500, text="boom")

    c = _client(handler)
    assert await c.search_papers("x") == []
    assert await c.search_github("x") == []


async def test_missing_fields_skipped():
    def handler(req):
        # no id / no url → skipped, not crash
        return httpx.Response(200, json={"data": [{"title": "no id"}]})

    assert await _client(handler).search_papers("x") == []


def test_configured_flag():
    assert FirecrawlResearchClient(api_key="fc-1", prefer_cli=False).configured is True
    assert FirecrawlResearchClient(api_key="", cli_executable="").configured is False


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env")
    assert FirecrawlResearchClient().configured is True
    assert FirecrawlResearchClient().api_key == "fc-env"
