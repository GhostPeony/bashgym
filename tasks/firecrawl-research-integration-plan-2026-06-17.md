# Firecrawl Research Integration + Platform News Feed — Plan

*Compiled 2026-06-17. Grounds BashGym's AutoResearch in the literature + GitHub history of the training ecosystem, and surfaces a platform news feed of training-method changes. API contract from the live Firecrawl Research docs (`docs.firecrawl.dev/features/research`).*

---

## 0. Goal

Two outcomes, one integration:

1. **Leverage our own AutoResearch better** — turn the existing blind evolutionary search (`bashgym/gym/autoresearch.py`, `SearchSpace.mutate`) into a *literature-grounded* search: before mutating hyperparameters/schemas, consult papers + GitHub for what actually works for this model/strategy and bias the search toward published recipes and known-good ranges.
2. **A general news feed** — a platform surface that auto-surfaces the latest training-method changes (GitHub issues/PRs/discussions on `unslothai/unsloth`, `huggingface/trl`, `NVIDIA-NeMo`, etc., + recent relevant papers), so the flywheel stays current without manual repo-watching. This is exactly how the existing memory shortlist (GSPO, Liger #1186, unsloth#4867, AsyncGRPO) was assembled — automate it.

---

## 1. The Firecrawl Research API (contract we build against)

Base `https://api.firecrawl.dev/v2`, auth `Authorization: Bearer fc-…` (optional but recommended), all GET:

| Endpoint | Use | Key response fields |
|---|---|---|
| `GET /search/research/papers?query=&k=&categories=&from=&to=&authors=` | semantic paper search | `paperId`, `primaryId`, `title`, `abstract`, `score` |
| `GET /search/research/papers/{id}?query=&k=` | metadata + passage extraction | metadata, matched passages |
| `GET /search/research/papers/{id}/similar?intent=&mode=similar\|citers\|references&k=` | citation-graph expansion | ranked related papers |
| `GET /search/research/github?query=&k=` | issues/PRs/discussions/docs | `repository`, `url`, `snippet` |

It's an **academic-paper + GitHub index** (semantic search + citation graph), not generic web scraping — a *complement* to our HF dataset scanner, not a replacement.

---

## 2. Architecture (mirrors existing BashGym patterns)

```
bashgym/research/
  firecrawl_client.py   FirecrawlResearchClient  (httpx, injectable transport, FIRECRAWL_API_KEY)
  advisor.py            ResearchAdvisor          (advise() + news_feed(), client injected)
bashgym/api/research_routes.py   + GET /api/research/news, POST /api/research/advise
bashgym/gym/autoresearch.py      (Phase 2) GroundedHyperparamSearchSpace consumes advisor priors
frontend/.../research/            (Phase 3) News feed + per-config advice panel
```

Design rules carried over from the providers/eval work:
- **Injected HTTP transport** → unit-testable with `httpx.MockTransport`, no live calls in CI.
- **Graceful degradation** → no API key or a failed request yields an empty result, never a crash (the news feed/advisor are advisory, never on a critical path).
- **No new dependency** → `httpx` is already a dep; the key comes from `FIRECRAWL_API_KEY` env (no settings-schema churn).
- **Normalized dataclasses** → `ResearchPaper`, `GithubFinding`, `NewsItem`, `AdviceReport` decouple callers from Firecrawl's JSON.

---

## 3. Phases

### Phase 0 — Foundation + both deliverables at the API level (THIS turn)
- `firecrawl_client.py`: `search_papers`, `search_github` → normalized dataclasses; key from env; injectable client; graceful.
- `advisor.py`:
  - `advise(context)` — from a training context (`base_model`/family, `strategy`, optional weak `domain`), compose targeted queries (papers: "{strategy} fine-tuning {family} hyperparameters"; github over the tracked repos) → `AdviceReport` (recommended techniques from paper hits + relevant open issues + a coarse hyperparameter **prior**).
  - `news_feed(repos, topics, k)` — ranked, deduped `NewsItem` feed from tracked training repos (GitHub) + recent papers (cs.LG/cs.CL training methods).
- `research_routes.py`: `GET /api/research/news`, `POST /api/research/advise`.
- Hermetic tests for client (MockTransport), advisor (fake client), routes (TestClient + monkeypatch).

### Phase 1 — Caching + freshness (small follow-up)
- Disk cache under `~/.bashgym/research/firecrawl/` with TTL (the feed shouldn't re-hit the API every load); reuse the existing research-dir convention.

### Phase 2 — Wire priors into AutoResearch (the "leverage" payoff)
- `GroundedHyperparamSearchSpace(HyperparamSearchSpace)` whose `mutate()` biases proposals toward the advisor's prior ranges (e.g. LoRA r, lr, loss variant) instead of a uniform random walk; falls back to the base behavior when no advice is available. Logged with citations for auditability.

### Phase 3 — UI
- A **Research / News** surface: the news feed (training-method changes) + a "what would help this run" advice panel keyed to the current Training Config. Botanical Brutalism.

### Phase 4 — Compose with Data Designer (optional, later)
- A Firecrawl-backed GitHub seed source feeding Data Designer's `from_unstructured`/`from_external` (real issues → SWE-style tasks). Slots into the v0.6.1 DD upgrade plan.

---

## 4. "Does a news feed make sense?" — yes, scoped

It earns its place because the platform's reason to exist is riding ecosystem advances. Scope guards so it's signal not noise:
- **Tracked repos only** by default (`unslothai/unsloth`, `huggingface/trl`, `NVIDIA-NeMo/*`, `vllm-project/vllm`) + a curated topic set (training methods, kernels, quantization, RL-for-LLMs).
- **Relevance-ranked + deduped**, capped (`k`), newest-first.
- **Advisory only** — degrades to empty without a key; never blocks training.

---

## 5. Verification posture (honest)

- Phase 0 client/advisor/routes are verified **hermetically** (MockTransport + injected fakes), mirroring the providers/eval/Fireworks work.
- The **live** behavior (real result quality, exact field coverage) needs a `FIRECRAWL_API_KEY` — that's the one live-verification step, surfaced (the feed shows an actionable "set FIRECRAWL_API_KEY" state when absent).
