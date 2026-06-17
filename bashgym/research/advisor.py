"""ResearchAdvisor — ground AutoResearch + a platform news feed in the research index.

Two surfaces over :class:`FirecrawlResearchClient`:

- :meth:`advise` — from a training context (model/family/strategy/weak domain),
  compose targeted paper + GitHub queries and return recommended techniques,
  relevant open issues, and a coarse *prior* over hyperparameters that a future
  ``GroundedHyperparamSearchSpace`` can bias its mutations toward.
- :meth:`news_feed` — a ranked, deduped feed of training-method changes from the
  tracked ecosystem repos (GitHub) plus recent training-method papers.

The advisor never invents facts: the prior keys off terms actually present in the
retrieved abstracts plus known family facts (e.g. Gemma's large vocab), so its
suggestions are grounded in what the search surfaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .firecrawl_client import TRACKED_REPOS, FirecrawlResearchClient

# Topics that define "relevant to how BashGym trains" — the news feed's lens.
DEFAULT_TOPICS: tuple[str, ...] = (
    "LoRA fine-tuning",
    "GRPO reinforcement learning LLM",
    "fused linear cross entropy kernel",
    "preference optimization DPO",
    "QLoRA quantization",
)


@dataclass
class TrainingContext:
    base_model: str = ""
    strategy: str = "sft"  # sft | dpo | grpo | rlvr | distillation
    family: str = ""  # resolved family (gemma4, qwen3, llama3, ...)
    domain: str = ""  # optional weak capability to target


@dataclass
class NewsItem:
    kind: str  # "github" | "paper"
    title: str
    url: str
    source: str  # repo (github) or "arxiv" (paper)
    summary: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "summary": self.summary,
            "score": self.score,
        }


@dataclass
class AdviceReport:
    context: dict
    techniques: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    prior: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "techniques": self.techniques,
            "issues": self.issues,
            "prior": self.prior,
        }


def _dedupe_rank(items: list[NewsItem], k: int) -> list[NewsItem]:
    """Drop duplicate URLs, sort by score (desc), cap to ``k``."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in items:
        if it.url and it.url not in seen:
            seen.add(it.url)
            unique.append(it)
    unique.sort(key=lambda i: i.score, reverse=True)
    return unique[:k]


def _prior_for(context: TrainingContext, abstracts: str) -> dict:
    """A coarse, *grounded* hyperparameter prior: baseline ranges + nudges keyed
    off terms in the retrieved abstracts and known family facts."""
    text = abstracts.lower()
    fam = context.family.lower()
    strat = context.strategy.lower()
    suggested: dict[str, Any] = {
        "learning_rate": [1e-5, 3e-5] if strat in ("sft", "dpo") else [1e-6, 1e-5],
        "lora_r": [16, 32],
    }
    notes: list[str] = []
    if strat in ("grpo", "rlvr") and (
        "gspo" in text or "sequence-level" in text or "moe" in text or "moe" in fam
    ):
        suggested["grpo_loss_type"] = "gspo"
        notes.append("Retrieved work favors GSPO (sequence-level) for long-seq/MoE stability.")
    if "liger" in text or "fused cross" in text or "fused linear" in text or "gemma" in fam:
        suggested["use_liger"] = True
        notes.append(
            "Large-vocab family (e.g. Gemma) → enable Liger fused-CE on the plain backend."
        )
    return {"suggested": suggested, "notes": notes}


class ResearchAdvisor:
    """Compose research queries into AutoResearch advice + a news feed."""

    def __init__(self, client: FirecrawlResearchClient):
        self.client = client

    async def advise(self, context: TrainingContext) -> AdviceReport:
        target = (context.family or context.base_model).strip()
        paper_query = f"{context.strategy} fine-tuning {target} {context.domain}".strip()
        github_query = f"{target} {context.strategy} training".strip()

        papers = await self.client.search_papers(paper_query, k=8, categories="cs.LG")
        issues = await self.client.search_github(github_query, k=8)

        techniques = [
            {"title": p.title, "url": p.url, "abstract": p.abstract[:280], "score": p.score}
            for p in papers
            if p.title
        ]
        abstracts = " ".join(f"{p.title} {p.abstract}" for p in papers)
        return AdviceReport(
            context=asdict(context),
            techniques=techniques,
            issues=[f.to_dict() for f in issues],
            prior=_prior_for(context, abstracts),
        )

    async def news_feed(
        self,
        *,
        repos: tuple[str, ...] = TRACKED_REPOS,
        topics: tuple[str, ...] = DEFAULT_TOPICS,
        k: int = 20,
        since: str | None = None,
    ) -> list[NewsItem]:
        query = " OR ".join(topics[:4])
        items: list[NewsItem] = []

        findings = await self.client.search_github(query, k=k)
        repo_set = set(repos)
        for f in findings:
            # Scope to tracked repos when the index labels the repository; keep
            # unlabeled hits rather than silently dropping them.
            if f.repository and repo_set and f.repository not in repo_set:
                continue
            items.append(
                NewsItem(
                    kind="github",
                    title=f.title or f.repository or "GitHub result",
                    url=f.url,
                    source=f.repository or "github",
                    summary=f.snippet[:280],
                )
            )

        papers = await self.client.search_papers(query, k=k, categories="cs.LG", since=since)
        for p in papers:
            items.append(
                NewsItem(
                    kind="paper",
                    title=p.title,
                    url=p.url,
                    source="arxiv",
                    summary=p.abstract[:280],
                    score=p.score,
                )
            )

        return _dedupe_rank(items, k)
