"""Hermetic tests for the Firecrawl research client (httpx.MockTransport)."""

from __future__ import annotations

import httpx

from bashgym.research.firecrawl_client import FirecrawlResearchClient


def _client(handler, api_key="fc-test"):
    return FirecrawlResearchClient(
        api_key=api_key, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def test_search_papers_parses_auth_and_params():
    captured = {}

    def handler(req):
        captured["path"] = req.url.path
        captured["auth"] = req.headers.get("authorization")
        captured["params"] = dict(req.url.params)
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
    assert captured["path"].endswith("/search/research/papers")
    assert captured["auth"] == "Bearer fc-test"
    assert captured["params"]["categories"] == "cs.LG"
    assert captured["params"]["k"] == "5"


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
    assert FirecrawlResearchClient(api_key="fc-1").configured is True
    assert FirecrawlResearchClient(api_key="").configured is False


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-env")
    assert FirecrawlResearchClient().configured is True
    assert FirecrawlResearchClient().api_key == "fc-env"
