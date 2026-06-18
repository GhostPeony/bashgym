"""Tests for the /api/research news + advise routes (advisor patched, no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app
from bashgym.research.advisor import AdviceReport, NewsItem, ResearchAdvisor

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    yield


def test_news_returns_feed(monkeypatch):
    async def fake_feed(self, **kw):
        return [NewsItem(kind="github", title="AsyncGRPO", url="u1", source="huggingface/trl")]

    monkeypatch.setattr(ResearchAdvisor, "news_feed", fake_feed)
    r = client.get("/api/research/news?k=5")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    assert data["items"][0]["title"] == "AsyncGRPO"
    assert data["configured"] is False  # no FIRECRAWL_API_KEY set


def test_news_configured_when_key_present(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-xyz")

    async def fake_feed(self, **kw):
        return []

    monkeypatch.setattr(ResearchAdvisor, "news_feed", fake_feed)
    r = client.get("/api/research/news")
    assert r.json()["configured"] is True


def test_advise_returns_report_and_resolves_family(monkeypatch):
    captured = {}

    async def fake_advise(self, context):
        captured["family"] = context.family
        captured["strategy"] = context.strategy
        return AdviceReport(
            context={"base_model": context.base_model},
            techniques=[{"title": "GSPO", "url": "u", "abstract": "", "score": 0.8}],
            issues=[],
            prior={"suggested": {"use_liger": True}, "notes": ["gemma → liger"]},
        )

    monkeypatch.setattr(ResearchAdvisor, "advise", fake_advise)
    r = client.post(
        "/api/research/advise",
        json={"base_model": "google/gemma-4-31B-it", "strategy": "grpo"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["prior"]["suggested"]["use_liger"] is True
    assert data["techniques"][0]["title"] == "GSPO"
    # family resolved from the base model in the route
    assert "gemma" in captured["family"].lower()
    assert captured["strategy"] == "grpo"
