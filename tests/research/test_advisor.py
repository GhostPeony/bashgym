"""Tests for ResearchAdvisor — advice (grounded prior) + news feed, with a fake client."""

from __future__ import annotations

from bashgym.research.advisor import ResearchAdvisor, TrainingContext
from bashgym.research.firecrawl_client import GithubFinding, ResearchPaper


class _FakeClient:
    def __init__(self, papers=None, github=None):
        self._papers = papers or []
        self._github = github or []
        self.configured = True
        self.queries: list[str] = []

    async def search_papers(self, query, **kw):
        self.queries.append(query)
        return self._papers

    async def search_github(self, query, **kw):
        self.queries.append(query)
        return self._github

    async def close(self):
        pass


async def test_advise_builds_report_and_grounded_prior():
    papers = [
        ResearchPaper(
            paper_id="p1",
            title="GSPO: Group Sequence Policy Optimization",
            abstract="sequence-level GSPO is more stable for MoE...",
            score=0.8,
            url="https://arxiv.org/abs/x",
        )
    ]
    github = [GithubFinding(repository="huggingface/trl", url="u1", snippet="gspo loss_type")]
    advisor = ResearchAdvisor(_FakeClient(papers, github))

    rep = await advisor.advise(
        TrainingContext(base_model="google/gemma-4-31B-it", strategy="grpo", family="gemma4")
    )
    assert rep.techniques[0]["title"].startswith("GSPO")
    assert rep.issues[0]["repository"] == "huggingface/trl"
    # prior is grounded: "gspo"/"sequence-level" in abstracts -> gspo; gemma family -> liger
    assert rep.prior["suggested"]["grpo_loss_type"] == "gspo"
    assert rep.prior["suggested"]["use_liger"] is True
    assert rep.prior["notes"]


async def test_advise_baseline_prior_without_signal():
    advisor = ResearchAdvisor(_FakeClient(papers=[], github=[]))
    rep = await advisor.advise(
        TrainingContext(base_model="Qwen/Qwen3-8B", strategy="sft", family="qwen3")
    )
    assert rep.prior["suggested"]["learning_rate"] == [1e-5, 3e-5]
    assert rep.prior["suggested"]["lora_r"] == [16, 32]
    assert "grpo_loss_type" not in rep.prior["suggested"]  # no RL signal
    assert "use_liger" not in rep.prior["suggested"]  # not gemma, no liger term


async def test_news_feed_dedupes_and_scopes_to_tracked_repos():
    github = [
        GithubFinding(repository="huggingface/trl", url="g1", snippet="tracked"),
        GithubFinding(repository="random/other", url="g2", snippet="not tracked"),
        GithubFinding(repository="", url="g3", snippet="unlabeled kept"),
    ]
    papers = [ResearchPaper(paper_id="p1", title="X", abstract="a", score=0.5, url="g1")]  # dup g1
    advisor = ResearchAdvisor(_FakeClient(papers, github))

    feed = await advisor.news_feed(k=10)
    urls = [i.url for i in feed]
    assert "g2" not in urls  # scoped out (not a tracked repo)
    assert "g3" in urls  # unlabeled hit kept
    assert urls.count("g1") == 1  # deduped across github + paper


async def test_news_feed_caps_to_k():
    github = [GithubFinding(repository="", url=f"u{i}", snippet="s") for i in range(30)]
    advisor = ResearchAdvisor(_FakeClient(papers=[], github=github))
    feed = await advisor.news_feed(k=5)
    assert len(feed) == 5
