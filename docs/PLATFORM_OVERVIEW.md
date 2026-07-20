# BashGym — Platform Overview

BashGym is a self-improving agentic development gym. It captures your AI coding
sessions, turns the successful ones into training data, fine-tunes specialist
models on that data, and routes live inference between a frontier teacher model
and your trained student — then feeds the new sessions back in. The result is a
closed loop that gets cheaper and more capable the more you code.

This document explains what the platform is, how the pieces fit, and the design
decisions behind them. For step-by-step setup see [GETTING_STARTED.md](GETTING_STARTED.md);
for the training stack in depth see [TRAINING_SETUP.md](TRAINING_SETUP.md).

---

## The core idea

Every coding session an AI agent runs is latent training data. The agent reads
files, runs commands, edits code, and either succeeds (tests pass, the task is
done) or fails. That trajectory — the reasoning, the tool calls, the outcome — is
exactly the supervision signal you need to train a model to do the same work.

Most of that signal is thrown away. BashGym keeps it. It ingests the trace
backlog already sitting in your agent histories (`~/.claude/projects/` for Claude
Code, plus Codex, Gemini, and Copilot stores), classifies each session by
quality, synthesizes clean training examples from the good ones, fine-tunes a
small open-weight model, and deploys it as a "student" that handles the work it
has learned — falling back to the teacher when it is unsure.

You do not start from zero. The first action is `import`, not "install hooks":
the platform reads your existing history so the flywheel starts with data on day
one. Hooks are for capturing *future* sessions, not a prerequisite.

---

## The Ouroboros flywheel

```
        ┌──────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
        │   ACT    │────▶│  VERIFY  │────▶│ SYNTHESIZE │────▶│  TRAIN   │
        │ (Arena)  │     │ (Judge)  │     │ (Factory)  │     │  (Gym)   │
        └──────────┘     └──────────┘     └────────────┘     └──────────┘
             ▲                                                     │
             │                                                     ▼
             │                                              ┌────────────┐
             └──────────────────── DEPLOY ◀─────────────────│  (Router)  │
                                                            └────────────┘
```

| Stage | Layer | What happens |
|-------|-------|--------------|
| **ACT** | Arena | An agent runs a task — live in your editor (captured via hooks) or in an isolated Docker sandbox driven by the Claude CLI. |
| **VERIFY** | Judge | Tests run (pytest / bats / `verify.sh`), exit codes and pass rates are recorded, and an LLM-as-judge scores quality across multiple dimensions. |
| **SYNTHESIZE** | Factory | Verified sessions become structured training examples (tool-call messages with `<thinking>`/`<plan>`/`<reflection>` tags), optionally augmented with synthetic data. |
| **TRAIN** | Gym | SFT / DPO / GRPO / distillation / Cascade RL fine-tune an open-weight base model, locally or on a remote training host. |
| **DEPLOY** | Router | The trained student is served (GGUF → Ollama) and the router sends inference to it, falling back to the teacher (Claude) on low confidence. |

Each turn of the loop improves the student, which lets the router send it more
traffic, which lowers cost — while the new sessions it produces become the next
round of training data.

---

## Architecture

The codebase is organized as a Python package (`bashgym/`) behind a FastAPI
backend, with a React/Electron front end. The layers map directly to the flywheel.

| Layer | Package | Responsibility | Key files |
|-------|---------|----------------|-----------|
| **Arena** | `bashgym/arena/` | Docker sandbox lifecycle + Claude CLI agent runner with guardrails and PII filtering. | `sandbox.py`, `runner.py` |
| **Judge** | `bashgym/judge/` | Test execution, LLM-as-judge scoring, benchmark harnesses, guardrails. | `verifier.py`, `evaluator.py`, `semantic_judge.py` |
| **Factory** | `bashgym/factory/` | Trace → training-example synthesis, quality scoring, synthetic data generation, DPO-pair extraction. | `trace_processor.py`, `example_generator.py`, `data_factory.py`, `data_designer.py` |
| **Gym** | `bashgym/gym/` | SFT/DPO/GRPO/distillation trainers, Cascade RL scheduler, compatibility search engines, teacher/student router. | `trainer.py`, `cascade_scheduler.py`, `autoresearch.py`, `router.py` |
| **Campaigns** | `bashgym/campaigns/` | Durable AutoResearch control plane: registered bindings, bounded hypotheses, execution leases, evidence, human oversight, recovery, and decisions. | `control_room.py`, `worker.py`, `guided_setup.py`, `campaign_recovery.py` |
| **Providers** | `bashgym/providers/` | Pluggable inference behind one interface — Anthropic (teacher), NVIDIA NIM and Ollama (student), with live model discovery. | `base.py`, `registry.py`, `anthropic.py`, `nim.py`, `ollama.py` |
| **Trace capture** | `bashgym/trace_capture/` | Importers for Claude Code, Codex, Gemini, and Copilot histories; hook-based live capture. | `importers/claude_history.py` |
| **Orchestrator (legacy)** | `bashgym/orchestrator/` | Retained compatibility code that is not wired into the current desktop product or AutoResearch authority path; candidate for removal after dependency audit. | `task_dag.py`, `dispatcher.py` |
| **Pipeline** | `bashgym/pipeline/` | Watches agent history, imports/classifies new traces, and auto-triggers downstream stages on thresholds. | `orchestrator.py`, `threshold_monitor.py` |

The backend exposes ~130 REST endpoints plus a WebSocket for live training-log
and event streaming. The front end ships in two modes from one codebase: an
Electron desktop app (native terminals, canvas workspace) and a browser web app.

---

## What makes the design hold up

These are the parts a technical reader should look at closely.

### Provider abstraction with live model discovery
Inference is hidden behind one `InferenceProvider` interface (`generate`,
`health_check`, `list_models`, `warm_up`). A `ProviderRegistry` maps each model
to its provider and monitors health. Catalogs are **discovered live** from each
provider's `/v1/models` endpoint at runtime, with a current static fallback, so
the available-model list never silently rots to a retired model. Adding a new
backend is implementing one class.

### The AutoCurriculum Compiler
Synthetic training data is generated with NVIDIA NeMo Data Designer — a column
DAG of samplers, LLM text/structured columns, LLM-as-judge columns, and Jinja2
expression columns. The novel part is the **SchemaResearcher**: an evolutionary
search that treats a Data Designer pipeline config as a genome, mutates it
(temperatures, judge thresholds, column topology), generates real training data
from each candidate, and scores it — first by a fast judge-score filter, then by
a 50-step micro-train that measures actual downstream loss. The system searches
for data-generation recipes that produce measurably better models, instead of
hand-tuning prompts.

### Cascade RL with multi-objective distillation
Rather than one monolithic fine-tune, the Cascade scheduler trains
domain-specialist stages sequentially (file operations → bash → search →
multi-step reasoning), chaining each stage's checkpoint into the next, then
optionally distills the experts back into one unified student via multi-objective
policy distillation. Domains can be tool-defined or auto-discovered per repository.

### AutoResearch
The official path is a durable, baseline-first campaign over explicitly
registered model, data, evaluator, compute, and source bindings. It preserves
bounded hypotheses, budgets, leases, sealed evidence, human decisions, and
restart recovery in one authoritative Control Room projection. The earlier
population-based hyperparameter, trace, and schema-search engines are hidden,
unsupported compatibility code. Their `/api/autoresearch/*` routes are absent
by default and register only when
`BASHGYM_ENABLE_LEGACY_AUTORESEARCH=true`; they are not a durable campaign or
Control Room capability. See
[Durable AutoResearch Campaigns](training/autoresearch-campaign.md).

### Two-tier hardware, one workflow
The trainer runs LoRA fine-tunes on a consumer GPU and full or larger fine-tunes
on a remote unified-memory training host over SSH — streaming logs back to the
same dashboard, with pause/resume/cancel via process signals. The user picks the
backend; the workflow is identical.

### Trace quality as a first-class signal
Sessions are scored across multiple dimensions (success rate, verification,
complexity, tool diversity, efficiency, length) and classified into gold / silver
/ bronze / failed / pending tiers. Only high-quality trajectories become training
data, and the same judge scores drive both the dataset filter and the
SchemaResearcher's fitness function.

---

## Design decisions and trade-offs

**Teacher/student split, not replacement.** The student never has to be as good
as the teacher everywhere — only good enough on the work it has seen, with a
confidence-gated fallback. This makes a small open-weight model useful long
before it is "done," and it makes the cost curve bend down gradually rather than
requiring a big-bang cutover.

**Drive the backend, don't fork it.** Every workflow goes through the FastAPI
backend, which owns the hard parts (training-subprocess lifecycle, registry init,
WebSocket streaming, orphaned-process recovery). The UI and any future CLI are
clients of that one contract, which keeps behavior consistent and avoids drift.

**Verify before you train.** A trajectory is only training data if it passed
verification. Exit-code-zero (or LLM-judged quality above threshold) is the gate;
failed trajectories are kept separately for DPO negatives, not discarded.

**Start from the backlog.** Requiring hooks-first onboarding would mean every
user starts with an empty gym. Ingesting existing agent history means the flywheel
has thousands of sessions to work with immediately.

**Trade-off accepted: classification cost.** LLM-as-judge quality scoring is an
API call per trace, which makes bulk classification slow. The platform mitigates
with fast heuristic pre-filters and tiered thresholds, but high-fidelity scoring
is deliberately not free — quality of the training set is worth the spend.

---

## Where the platform stands

| Signal | Value |
|--------|-------|
| Gold traces available | ~3,140 |
| Training strategies | 6 (SFT, DPO, GRPO, RLVR, Distillation, Cascade RL) |
| Synthetic-data pipeline types | 5 (SFT, DPO, tool-use, external, unstructured) |
| Inference providers | 3 (Anthropic, NVIDIA NIM, Ollama) behind one interface |
| Cascade RL domains | 4 (file ops, bash, search, multi-step) |
| Front-end views | 14 dashboards (one React codebase, desktop + web) |
| Test coverage | Hundreds of tests across factory, gym, providers, and API layers |

---

## Read next

- [GETTING_STARTED.md](GETTING_STARTED.md) — install to first trained model, step by step.
- [TRAINING_SETUP.md](TRAINING_SETUP.md) — the training stack in technical depth (hardware tiers, strategies, the data factory, remote training).
- [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) — trace format, quality tiers, and the example-generation pipeline.
- [API.md](API.md) — REST API reference.
