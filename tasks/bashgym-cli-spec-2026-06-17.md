# BashGym CLI — Design Spec

> Status: DRAFT for review · 2026-06-17 · Author: Cade (+ Claude)
> Goal source: `/goal` — "understand the features/tools BashGym provides and spec an ideal CLI built on open source + best practices, focused on compound engineering to make the platform super usable."

## 1. Why a CLI (motivation)

Today the **only front door to BashGym is the React/Electron UI**. The FastAPI backend exposes ~130 endpoints + a WebSocket, but every workflow — import traces, score them, generate examples, train, deploy, route — is driven by clicking. That has three costs this session made concrete:

1. **No automation surface.** You can't cron the retrain loop, gate CI on an eval, or script the flywheel. The DGX training, the threshold pipeline, the cascade — all require a human in the UI.
2. **The agent that generates the traces can't drive the gym.** BashGym *learns from Claude Code traces*. An agent-drivable CLI lets Claude Code (or a scheduled agent) close its own loop: capture → classify → train → deploy, unattended. This is the literal expression of the project's thesis.
3. **The UI is a fragile single point of failure.** This session alone: the browser build crashed to a black screen (no error boundary), the model catalog was offering retired models, and a dormant web-auth gate blocked local use. A CLI is the **resilient, scriptable front door** that keeps the platform usable when the UI isn't.

**Start from the backlog, not from zero.** The UI onboarding leads with "install hooks," but hooks only capture *future* sessions. Most users already have a large **backlog of agent traces** in `~/.claude/projects/` (Claude Code), the Codex history store, and Gemini/Copilot histories. The CLI's first move should therefore be **`bashgym traces import`** — ingest that existing history so the flywheel starts with data immediately (BashGym already ships the importers in `trace_capture/importers/`, and Data Designer 0.6.1's `AgentRolloutSeedSource` reads `~/.claude/projects` + Codex natively). Hooks (`bashgym hooks install`) are for ongoing forward capture, **not a prerequisite to begin**. From there the flywheel (ACT → VERIFY → SYNTHESIZE → TRAIN → DEPLOY) runs as a linear command sequence the CLI makes first-class.

## 2. Goals / non-goals

**Goals**
- Expose the full flywheel (ACT → VERIFY → SYNTHESIZE → TRAIN → DEPLOY) plus Cascade RL, AutoResearch, Orchestrator at **workflow altitude**, not endpoint-by-endpoint.
- Be **agent-first and human-friendly**: structured/JSON-first output the agent parses, Rich rendering on a TTY, clean non-interactive machine mode for cron/CI.
- Make the tool **compound**: every run gets recorded, learned from, and suggests the next step, so the system gets easier to operate the more it's used.
- Reuse the existing backend (single source of truth) — do not re-implement orchestration.

**Non-goals**
- Replacing the UI (the canvas/terminal grid stays Electron's job).
- 1:1 coverage of all 130 endpoints (that's a worse Swagger UI).
- Shipping a public/distributable tool *first* (internal power-use + agent-driving is the wedge; OSS polish comes later).

## 3. Design decisions locked this session

| Decision | Choice | Rationale |
|---|---|---|
| **Primary driver** | Agent-first, human-friendly | BashGym compounds from agent traces; an agent-drivable CLI closes the loop. Structured-first output + Rich on TTY + machine mode. |
| **Compound spine** | All three, as one self-improving operator layer | Run ledger → learned defaults → next-action hints → autopilot. "Boil the lake": these reinforce each other. |

## 4. Premises (the design rests on these)

1. The FastAPI backend is the source of truth; the CLI drives it, never forks its orchestration (training-subprocess lifecycle, registry init, WS streaming, orphan recovery all live server-side).
2. Value is workflow-altitude commands + the compound layer, not endpoint parity.
3. "Compound" is **substrate**: a ledger write + next-action hint is cross-cutting middleware every command inherits for free, not per-command code.
4. Doing nothing has a real cost — there is no automation/agent surface today.
5. Distribution counts even internally: the `bashgym` console script already exists in `pyproject.toml` (currently the server runner); new CLI deps go behind an optional extra so the lean core (`anthropic`, `httpx`, `docker`, `python-dotenv`) stays lean.

## 5. Architecture — three approaches

The core fork is how tightly the CLI couples to the running backend.

### Approach A — Thin HTTP client (minimal viable)
A Typer app that calls the FastAPI backend over `httpx` (already a dependency), mirroring workflow endpoints. Requires the server running.
- **Effort:** S/M · **Risk:** Low
- ✅ Zero logic duplication; the API is the contract. WebSocket streaming already exists for live logs. Every new endpoint is instantly reachable.
- ✅ Works against a remote backend (e.g. the DGX) unchanged.
- ❌ Requires a running server for everything, even trivial local reads; another process to babysit.

### Approach B — Library-direct (no server)
The CLI imports `bashgym` modules directly (Trainer, ExampleGenerator, ProviderRegistry…).
- **Effort:** L · **Risk:** Med/High
- ✅ No daemon; fast for one-shot CI ops.
- ❌ Re-implements the app wiring that `create_app()` does (registry init, WS broadcasts, training-subprocess lifecycle, orphan recovery) → drift and divergence. Heavy imports (torch) slow cold start. Can't reuse the live server's streaming/state.

### Approach C — Hybrid, daemon-aware (ideal architecture) — RECOMMENDED
A thin client (Approach A) that also **manages the backend's lifecycle**: it detects a running server, auto-starts one if absent (`bashgym serve` under the hood), and routes a small set of pure-local commands (config, hooks install, ledger inspection, trace file peeking) without needing the server at all.
- **Effort:** M/L · **Risk:** Low/Med
- ✅ Best UX: `bashgym train` "just works" whether or not the server is up; one-shot local ops stay instant; long-running ops use the live server + WS streaming.
- ✅ The CLI becomes the front door *and* the daemon manager — natural home for the compound layer.
- ❌ More to build (health detection, lifecycle, two code paths) — mitigated by phasing (ship A first, layer C).

**Recommendation: C, delivered in phases — ship the thin client (A) first, then make it daemon-aware (C).** Rejected B outright: re-implementing the server's wiring violates premise #1 and guarantees drift.

## 6. The compound layer (what makes this more than a wrapper)

Implemented as Typer **middleware / a result wrapper** so every command inherits it:

1. **Run ledger** — every invocation appends a record to `~/.bashgym/cli/runs.jsonl`: command, resolved args, result summary, outcome, duration, the backend run-id it spawned. Runs become replayable: `bashgym replay <run-id>` re-issues the exact command. This is the literal "every command makes the next easier."
2. **Learned defaults** — AutoResearch already searches hyperparameters; the CLI persists the best-found config per repo/strategy to `~/.bashgym/cli/defaults.json`, so `bashgym train` with no args uses the best-known config. `bashgym train --explain` shows where each default came from (a prior run / an autoresearch result / the global default).
3. **Next-action hints** — every command's output ends with a `next:` block. On a cold start the very first hint is the backlog ("847 Claude Code + Codex sessions in your history not yet imported → `bashgym traces import`"), then it chains forward ("3,164 gold traces ready → `bashgym train`", "training run abc done → `bashgym deploy abc`"). The system teaches its operator. In `--json` mode this is a `next` array of `{reason, command}` the agent can chain.
4. **Capability manifest** — `bashgym manifest --json` emits a machine-readable map of commands, args, and current state (counts, active models, thresholds) so an agent discovers what to do without docs. Generated from the Typer app + a live state probe.
5. **Autopilot** — `bashgym loop` runs the threshold-driven flywheel autonomously (the existing pipeline watcher, surfaced as a foreground command): watch → import → classify → when gold-trace threshold hit, generate → train → eval → (gated) deploy → repeat. Each cycle improves the student → routes more to it → cheaper. Built *on* the ledger + hints, with `--dry-run` and per-stage gates.

## 7. Command tree (workflow altitude)

```
bashgym init                      # one-shot setup: install hooks, write config, health-check
bashgym serve [--port] [--bg]     # start/stop/status the backend daemon
bashgym status                    # one screen: traces by tier, models, active teacher/student, pipeline thresholds, next action
bashgym manifest [--json]         # machine-readable capability + state map (agent entrypoint)

# ACT / capture  (START HERE: ingest existing agent history — hooks are optional, forward-only)
bashgym traces import [--source claude|codex|gemini|copilot|all] [--since DATE]   # ingest ~/.claude/projects, Codex/Gemini/Copilot backlog
bashgym hooks install|status|uninstall                                            # ongoing forward capture — NOT required to start
bashgym traces ls [--tier gold|silver|bronze|failed|pending] [--repo X]
bashgym traces show <id> | promote <id> | demote <id> | classify [--all]

# SYNTHESIZE / factory
bashgym examples gen <trace-id|--tier gold> [--repo X]
bashgym examples export [--repo X] [--out DIR]        # NeMo JSONL train/val
bashgym factory synth [--strategy trace_seeded] [--provider anthropic|nim] [--preset ...]

# TRAIN / gym
bashgym train [--strategy sft|dpo|grpo|distill] [--base MODEL] [--remote dgx] [--export gguf]
bashgym train status <run> | logs <run> [-f] | pause|resume|cancel <run>
bashgym cascade start [--domains ...] [--mode simulate|real] | status | distill
bashgym research start [--params ...] [--max N] | status | pause|resume|stop   # AutoResearch

# DEPLOY / serving
bashgym models ls | show <id> | export <run> --format gguf|hf | deploy <run> --target ollama
bashgym providers ls | health | warmup <model>
bashgym models refresh                         # live-discover catalogs (the fix from this session, as a command)
bashgym router config | set-student <provider> <model> | strategy <name> | stats
bashgym eval run <model> [--bench humaneval,mbpp,...] | status <job>

# AUTONOMY
bashgym orchestrate <spec.md> | status <job> | cancel <job>
bashgym loop [--dry-run] [--until cycles=N]     # autopilot flywheel
bashgym devices discover|add|ls|use <id>        # SSH/DGX training targets
bashgym pipeline status | trigger <stage>

# substrate
bashgym config get|set|show [--json]
bashgym runs ls | show <id> | replay <id>       # the run ledger
```

## 8. Output contract

- **TTY (human):** Rich tables, spinners, live log streaming, color. Sensible interactive prompts **only** when stdin is a TTY.
- **`--json` / non-TTY (agent + machine):** single JSON object to stdout — `{ok, data, next, run_id}` — no prompts, no color, meaningful **exit codes** (0 ok, 1 error, 2 usage, 3 backend-unreachable, 4 auth). Detect TTY with `sys.stdout.isatty()`; honor `--json`, `NO_COLOR`, `BASHGYM_JSON=1`.
- Long ops stream progress over the existing WebSocket; `--wait`/`--no-wait` choose blocking vs fire-and-return-run-id.
- Every result (both modes) carries the `next` hints and the ledger `run_id`.

## 9. Open-source stack & best practices

- **Typer** (Click-based, type-hint-driven) for the command tree + **Rich** for rendering — the modern, widely-adopted standard (gold references: `gh`, Simon Willison's `llm`). Behind a `bashgym[cli]` optional extra (`typer`, `rich`); `httpx` is already core.
- **Don't hand-maintain the API client** where avoidable: generate a typed client from the backend's OpenAPI schema (FastAPI already serves `/openapi.json`) for the thin endpoint layer; hand-write only the workflow + compound layer on top. Keeps the CLI in lockstep with the API (premise #2's "no drift").
- **XDG / platform dirs** for state (`~/.bashgym/cli/`), **layered config** (flags > env > `.env` > defaults), `--json` everywhere, idempotent commands, `--dry-run` on anything destructive/expensive.
- Repurpose the existing `[project.scripts] bashgym = "main:main"` → a Typer entrypoint; keep `bashgym serve` as the path to today's server runner.

## 10. Distribution

- Console script `bashgym` (already declared) → Typer app. `pip install -e .` for dev; `bashgym[cli]` extra pulls Typer/Rich.
- Phase-2 OSS: `pipx install bashgym`, shell completion (`bashgym --install-completion`), `--version`, a one-line `bashgym init` that scaffolds `.env` + installs hooks.

## 11. Phased plan

- **MVP (compound from day one):** thin client (A) for `status`, `traces`, `examples`, `train`, `models`, `providers`, `models refresh`, `config`, `hooks` + the **ledger + next-action hints** middleware. Ships the spine that compounds even before autopilot.
- **Phase 2:** daemon-awareness (C) — auto-detect/auto-start backend, pure-local commands; learned defaults; `manifest --json`; `runs replay`.
- **Phase 3:** `bashgym loop` autopilot + `cascade`/`research`/`orchestrate` + remote DGX (`--remote dgx`) + OSS packaging/completion.

## 12. Open questions / risks

- **Backend lifecycle on Windows** (the known zombie-port-8003 issue) — daemon-awareness must health-check and reap cleanly.
- **Auth in deployed mode** — the CLI's machine path should use `X-API-Key` (the backend already supports `BASHGYM_API_KEY`), so cron/CI work against a web-mode backend without cookies.
- **Learned-defaults provenance** — must be explainable (`--explain`) and overridable; never silently pick a config the user can't trace.
- **`loop` blast radius** — autopilot that trains/deploys unattended needs hard gates (eval-gated deploy, `--dry-run`, budget caps).

## 13. The assignment (concrete next step)

Build the **MVP skeleton**: a Typer app at `bashgym/cli/` wired to the console script, with `traces import` (ingest your existing `~/.claude/projects` + Codex backlog) → `status` → `traces ls` → `train` over the thin HTTP client, and the ledger+hints middleware — then dogfood it by importing your real trace backlog and driving one retrain from the terminal. That slice proves the spine (agent-legible output, the ledger, the next-action chain) end-to-end, starting from real existing data rather than an empty gym.

## What I noticed about how you think
- You stopped the abstract design twice to ground it in reality — "is there a playwright ability where you can act as the user" and "our models seem very outdated." You design from the running system, not the diagram.
- "I don't ever remember implementing that… we were working on our localized version that runs in the electron app" — you keep a tight mental model of scope and flag drift fast. The CLI spec leans into that: Electron-local first, web/OSS deferred.
- "do you know what to do?" — you want decisiveness over deliberation. The phased plan front-loads a single proving slice instead of a big-bang build.
