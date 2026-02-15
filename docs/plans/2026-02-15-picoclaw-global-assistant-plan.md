# PicoClaw Global Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up picoclaw as the user-facing chat shell for BashGym, reachable via Discord and Telegram with full control over all BashGym subsystems.

**Architecture:** picoclaw (Go binary) runs as a Docker gateway service alongside BashGym's FastAPI backend. picoclaw handles chat channels, identity, memory, and skill routing. Skills wrap BashGym API calls via curl. The two services communicate over Docker's internal network.

**Tech Stack:** Go (picoclaw), Docker Compose, Bash (skill scripts), Markdown (identity/skill definitions)

**Design Doc:** `docs/plans/2026-02-15-picoclaw-global-assistant-design.md`

---

### Task 1: Import picoclaw via git subtree

**Files:**
- Create: `assistant/picoclaw/` (entire subtree from upstream)

**Step 1: Add picoclaw as a git subtree**

```bash
git subtree add --prefix=assistant/picoclaw \
  https://github.com/sipeed/picoclaw.git main --squash
```

**Step 2: Verify the import**

Run: `ls assistant/picoclaw/cmd/picoclaw/main.go`
Expected: File exists

Run: `ls assistant/picoclaw/pkg/agent/loop.go`
Expected: File exists

Run: `ls assistant/picoclaw/go.mod`
Expected: File exists

**Step 3: Verify it builds**

Run: `cd assistant/picoclaw && go build ./cmd/picoclaw/`
Expected: Binary compiles (requires Go 1.21+). If Go is not installed locally, skip — Docker will handle the build.

**Step 4: Commit**

The subtree add already creates a commit. Verify with:
```bash
git log --oneline -1
```
Expected: Squashed commit message referencing picoclaw

---

### Task 2: Create workspace identity files

**Files:**
- Create: `assistant/workspace/IDENTITY.md`
- Create: `assistant/workspace/SOUL.md`
- Create: `assistant/workspace/AGENT.md`
- Create: `assistant/workspace/USER.md`
- Create: `assistant/workspace/memory/MEMORY.md`

**Step 1: Create the workspace directory structure**

```bash
mkdir -p assistant/workspace/memory
mkdir -p assistant/workspace/skills
```

**Step 2: Write IDENTITY.md**

```markdown
# GhostWork Assistant

A system-aware development assistant for the GhostWork ecosystem.
Manages orchestration, training, traces, models, and infrastructure
across all Ghost Peony projects.

Built on picoclaw, powered by Claude.
```

**Step 3: Write SOUL.md**

```markdown
# Soul

## Personality
- Direct and concise — no filler, no hedging
- Technically precise — uses correct terminology
- Proactive — surfaces problems before asked
- Calm under pressure — errors are data, not crises

## Values
- Accuracy over speed
- Show don't tell — include data, not vague summaries
- Respect the user's time — front-load the important information
- Transparency — always explain what actions are being taken
```

**Step 4: Write AGENT.md**

```markdown
# Agent Instructions

You are a development operations assistant with full access to the
GhostWork/BashGym system via API.

## Core Behavior
- Always confirm before destructive operations (stopping training, demoting traces)
- Report system status with concrete numbers (GPU %, trace counts, budget remaining)
- When submitting orchestrator specs, summarize the plan before asking for approval
- Use skills to interact with BashGym — don't improvise API calls
- Remember user preferences and past decisions in memory files
- If an API call fails, report the error and suggest next steps

## API Access
All BashGym interaction goes through HTTP calls to $BASHGYM_API_URL.
Use the api.sh helper script in each skill's scripts/ directory.
The base URL is set via environment variable — never hardcode it.

## Destructive Operations (require confirmation)
- POST /api/training/{run_id}/stop
- POST /api/traces/{trace_id}/demote
- DELETE /api/orchestrator/{job_id}
- DELETE /api/models/{model_id}
```

**Step 5: Write USER.md**

```markdown
# User

## Preferences
- Communication style: direct, technical
- Timezone: (learned from interaction)
- Primary projects: ghostwork/bashgym

## Context
- Uses Claude Code as primary development tool
- Botanical Brutalist design ideology across projects
- Training pipeline: traces -> examples -> SFT/DPO on Qwen2.5-Coder
```

**Step 6: Write memory/MEMORY.md**

```markdown
# Memory

Persistent memory for the GhostWork Assistant. Updated automatically
as the assistant learns user preferences and system patterns.
```

**Step 7: Verify files exist**

Run: `ls -la assistant/workspace/`
Expected: IDENTITY.md, SOUL.md, AGENT.md, USER.md, memory/, skills/

**Step 8: Commit**

```bash
git add assistant/workspace/
git commit -m "feat(assistant): add workspace identity files

IDENTITY.md, SOUL.md, AGENT.md, USER.md, and memory directory
for picoclaw workspace configuration."
```

---

### Task 3: Create shared API helper script

**Files:**
- Create: `assistant/workspace/scripts/api.sh`

**Step 1: Write the api.sh helper**

```bash
#!/bin/bash
# BashGym API helper for picoclaw skills
# Usage: api.sh METHOD ENDPOINT [JSON_BODY]
# Examples:
#   api.sh GET /health
#   api.sh GET /traces?status=gold
#   api.sh POST /training/start '{"strategy":"sft","base_model":"Qwen/Qwen2.5-Coder-1.5B-Instruct"}'
#   api.sh DELETE /orchestrator/job-123

set -euo pipefail

METHOD="${1:?Usage: api.sh METHOD ENDPOINT [BODY]}"
ENDPOINT="${2:?Usage: api.sh METHOD ENDPOINT [BODY]}"
BODY="${3:-}"
API_URL="${BASHGYM_API_URL:?BASHGYM_API_URL not set}"

CURL_ARGS=(
  -s
  -w "\n%{http_code}"
  -X "$METHOD"
  "${API_URL}${ENDPOINT}"
  -H "Content-Type: application/json"
)

if [ -n "$BODY" ]; then
  CURL_ARGS+=(-d "$BODY")
fi

RESPONSE=$(curl "${CURL_ARGS[@]}")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY_OUT=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ge 400 ]; then
  echo "ERROR ($HTTP_CODE): $BODY_OUT" >&2
  exit 1
fi

echo "$BODY_OUT"
```

**Step 2: Make it executable**

```bash
chmod +x assistant/workspace/scripts/api.sh
```

**Step 3: Verify syntax**

Run: `bash -n assistant/workspace/scripts/api.sh`
Expected: No output (no syntax errors)

**Step 4: Commit**

```bash
git add assistant/workspace/scripts/
git commit -m "feat(assistant): add shared API helper script

Wraps curl with error handling, HTTP status checking, and
BASHGYM_API_URL environment variable support."
```

---

### Task 4: Create system skill

Start with the simplest skill to establish the pattern.

**Files:**
- Create: `assistant/workspace/skills/system/SKILL.md`

**Step 1: Write the system skill**

```markdown
---
name: system
description: "Check BashGym system status, GPU utilization, hardware info, and aggregate statistics. Use when asked about system health, what's running, GPU usage, memory, disk space, or general system stats."
---

# System Status

Query BashGym system health and hardware information.

## Health Check

Quick service health check:
```bash
scripts/api.sh GET /health
```

Returns: `{"status": "ok", "version": "..."}`

## System Information

Full hardware report (CPU, GPU, memory, disk):
```bash
scripts/api.sh GET /system/info
```

Returns GPU utilization, CUDA availability, memory usage, disk space.

## GPU Details

Dedicated GPU endpoint:
```bash
scripts/api.sh GET /system/gpus
```

## Aggregate Statistics

Cross-subsystem stats (trace counts, model counts, training runs):
```bash
scripts/api.sh GET /stats
```

## Model Recommendations

System-recommended models based on available hardware:
```bash
scripts/api.sh GET /system/recommendations
```

## Example Interactions

User: "How's the system doing?"
→ Call /health, then /system/info for GPU/memory. Summarize key numbers.

User: "Is my GPU being used?"
→ Call /system/gpus. Report utilization %, memory used/total.

User: "Give me a status report"
→ Call /stats for counts, /system/info for hardware. Present as a concise dashboard.
```

**Step 2: Verify frontmatter**

Run: `head -4 assistant/workspace/skills/system/SKILL.md`
Expected: YAML frontmatter with `name: system` and `description:`

**Step 3: Commit**

```bash
git add assistant/workspace/skills/system/
git commit -m "feat(assistant): add system status skill

Wraps /health, /system/info, /system/gpus, /stats endpoints."
```

---

### Task 5: Create orchestrator skill

**Files:**
- Create: `assistant/workspace/skills/orchestrator/SKILL.md`

**Step 1: Write the orchestrator skill**

Note: The actual API prefix is `/api/orchestrator/` (not `/api/orchestrate/`). The submit endpoint returns the plan directly. There is no separate `/plan` endpoint.

```markdown
---
name: orchestrator
description: "Submit development specs, approve execution plans, monitor workers, retry failed tasks, check job status, and manage orchestration jobs. Use when asked to start a development job, check worker progress, approve a plan, retry something, or list orchestrator jobs."
---

# Orchestrator

Submit development specs for multi-agent decomposition and parallel execution.

## Submit a Spec

Decompose a development task into a parallel execution plan:
```bash
scripts/api.sh POST /orchestrator/submit '{
  "title": "Add user authentication",
  "description": "Implement JWT-based auth with login/register endpoints",
  "constraints": ["Use existing User model", "Add pytest tests"],
  "acceptance_criteria": ["All tests pass", "Endpoints return proper status codes"],
  "max_workers": 3,
  "budget_usd": 5.0,
  "provider": "anthropic"
}'
```

Returns: job_id and decomposed task plan. **Always summarize the plan for the user before approving.**

## Approve a Plan

After reviewing the plan, approve to begin execution:
```bash
scripts/api.sh POST /orchestrator/{job_id}/approve '{}'
```

## Check Job Status

Monitor worker progress and task states:
```bash
scripts/api.sh GET /orchestrator/{job_id}/status
```

Returns: task states (pending/running/completed/failed), worker assignments, budget usage.

## Retry a Failed Task

Retry a specific failed task within a job:
```bash
scripts/api.sh POST /orchestrator/{job_id}/task/{task_id}/retry '{}'
```

## List All Jobs

```bash
scripts/api.sh GET /orchestrator/jobs
```

## Delete a Job

**Destructive — confirm with user first.**
```bash
scripts/api.sh DELETE /orchestrator/{job_id}
```

## List Available Providers

Check which LLM providers are configured for planning:
```bash
scripts/api.sh GET /orchestrator/providers
```

## Example Interactions

User: "Build me an auth system with JWT"
→ Ask for any constraints. Compose a spec. Call POST /orchestrator/submit.
   Summarize the decomposed plan. Ask user to approve before calling /approve.

User: "What's the status of my last job?"
→ Call GET /orchestrator/jobs to find the latest. Call GET /orchestrator/{job_id}/status.
   Report task states and budget usage.

User: "Task 3 failed, retry it"
→ Call POST /orchestrator/{job_id}/task/{task_id}/retry. Report result.
```

**Step 2: Commit**

```bash
git add assistant/workspace/skills/orchestrator/
git commit -m "feat(assistant): add orchestrator skill

Wraps /orchestrator/* endpoints for spec submission, plan approval,
worker monitoring, and task retry."
```

---

### Task 6: Create training skill

**Files:**
- Create: `assistant/workspace/skills/training/SKILL.md`

**Step 1: Write the training skill**

```markdown
---
name: training
description: "Start, stop, pause, resume, and monitor model training runs. Check training progress, view metrics, and get GPU utilization during training. Use when asked to train a model, check training status, view loss curves, stop a run, or see training logs."
---

# Training

Manage SFT/DPO/GRPO training runs on local hardware.

## Start Training

```bash
scripts/api.sh POST /training/start '{
  "strategy": "sft",
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "selected_repos": [],
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 2e-5
}'
```

Strategy options: `sft`, `dpo`, `grpo`
Empty `selected_repos` means train on all gold traces.

## Check Training Status

```bash
scripts/api.sh GET /training/{run_id}
```

Returns: epoch, loss, learning rate, elapsed time, estimated completion.

## List Training Runs

```bash
scripts/api.sh GET /training
```

## Pause Training

```bash
scripts/api.sh POST /training/{run_id}/pause '{}'
```

## Resume Training

```bash
scripts/api.sh POST /training/{run_id}/resume '{}'
```

## Stop Training

**Destructive — confirm with user first.**
```bash
scripts/api.sh POST /training/{run_id}/stop '{}'
```

## GPU Utilization During Training

```bash
scripts/api.sh GET /system/info
```

Report GPU utilization %, memory used/total, temperature if available.

## Example Interactions

User: "Start training on my gold traces"
→ Call POST /training/start with defaults. Report run_id and initial status.

User: "How's training going?"
→ Call GET /training to find active run. Call GET /training/{run_id} for metrics.
   Also call GET /system/info for GPU stats. Report epoch, loss, GPU %.

User: "Stop the current training"
→ Confirm with user. Find active run_id. Call POST /training/{run_id}/stop.
```

**Step 2: Commit**

```bash
git add assistant/workspace/skills/training/
git commit -m "feat(assistant): add training management skill

Wraps /training/* endpoints for start, stop, pause, resume,
and status monitoring of SFT/DPO/GRPO runs."
```

---

### Task 7: Create traces skill

**Files:**
- Create: `assistant/workspace/skills/traces/SKILL.md`

**Step 1: Write the traces skill**

```markdown
---
name: traces
description: "Browse, promote, demote, classify, and generate training examples from Claude Code execution traces. Use when asked about traces, gold traces, trace counts, promoting or demoting traces, generating examples, exporting training data, or syncing traces."
---

# Traces

Manage Claude Code execution traces and convert them to training examples.

## List Traces

Filter by status (gold, pending, failed):
```bash
scripts/api.sh GET "/traces?status=gold"
scripts/api.sh GET "/traces?status=pending"
scripts/api.sh GET /traces
```

## Trace Statistics

Quick counts by status:
```bash
scripts/api.sh GET /traces/stats
```

## List Repositories

See which repos have traces:
```bash
scripts/api.sh GET /traces/repos
```

## Get Gold Traces

```bash
scripts/api.sh GET /traces/gold
```

## Promote a Trace

Move a pending trace to gold status:
```bash
scripts/api.sh POST /traces/{trace_id}/promote '{}'
```

## Demote a Trace

**Destructive — confirm with user first.**
Move a gold trace back to failed:
```bash
scripts/api.sh POST /traces/{trace_id}/demote '{}'
```

## Generate Training Examples

Convert a trace into training examples:
```bash
scripts/api.sh POST /traces/{trace_id}/generate-examples '{}'
```

## List Training Examples

```bash
scripts/api.sh GET /training/examples
```

## Export to NeMo Format

Export generated examples for training:
```bash
scripts/api.sh POST /training/export '{}'
```

## Sync Traces

Import new traces from Claude Code history:
```bash
scripts/api.sh POST /traces/sync '{}'
```

## Auto-Classify Traces

Automatically classify pending traces:
```bash
scripts/api.sh POST /traces/auto-classify '{}'
```

## Example Interactions

User: "How many gold traces do I have?"
→ Call GET /traces/stats. Report counts per status.

User: "Promote trace abc123"
→ Call POST /traces/abc123/promote. Confirm success.

User: "Generate training data from my gold traces"
→ Call GET /traces/gold to list them. For each, call POST /traces/{id}/generate-examples.
   Then call POST /training/export to create NeMo JSONL.

User: "Sync new traces"
→ Call POST /traces/sync. Report how many new traces were imported.
```

**Step 2: Commit**

```bash
git add assistant/workspace/skills/traces/
git commit -m "feat(assistant): add traces management skill

Wraps /traces/* endpoints for browsing, promoting, demoting,
example generation, export, and sync."
```

---

### Task 8: Create models skill

**Files:**
- Create: `assistant/workspace/skills/models/SKILL.md`

**Step 1: Write the models skill**

```markdown
---
name: models
description: "Browse, compare, evaluate, and manage trained models. View leaderboard, trends, lineage, and artifacts. Deploy models to Ollama. Use when asked about models, model comparison, which model is best, model evaluation, deploying a model, or downloading a model."
---

# Models

Browse and manage the model registry.

## List Models

```bash
scripts/api.sh GET /models
```

## Leaderboard

Ranked models by evaluation score:
```bash
scripts/api.sh GET /models/leaderboard
```

## Trends

Model performance over time:
```bash
scripts/api.sh GET /models/trends
```

## Compare Models

```bash
scripts/api.sh POST /models/compare '{"model_ids": ["model-a", "model-b"]}'
```

## Get Model Details

```bash
scripts/api.sh GET /models/{model_id}
```

## Evaluate a Model

Run evaluation benchmarks:
```bash
scripts/api.sh POST /models/{model_id}/evaluate '{}'
```

## Deploy to Ollama

Make a model available locally via Ollama:
```bash
scripts/api.sh POST /models/{model_id}/deploy-ollama '{}'
```

## Delete a Model

**Destructive — confirm with user first.**
```bash
scripts/api.sh DELETE /models/{model_id}
```

## Model Artifacts

View training artifacts (configs, checkpoints):
```bash
scripts/api.sh GET /models/{model_id}/artifacts
```

## Example Interactions

User: "Which model is best?"
→ Call GET /models/leaderboard. Report top 3 with scores.

User: "Compare model-a and model-b"
→ Call POST /models/compare. Present side-by-side metrics.

User: "Deploy my latest model to Ollama"
→ Call GET /models to find latest. Call POST /models/{id}/deploy-ollama.
```

**Step 2: Commit**

```bash
git add assistant/workspace/skills/models/
git commit -m "feat(assistant): add models management skill

Wraps /models/* endpoints for browsing, comparing, evaluating,
and deploying trained models."
```

---

### Task 9: Create factory skill

**Files:**
- Create: `assistant/workspace/skills/factory/SKILL.md`

**Step 1: Write the factory skill**

```markdown
---
name: factory
description: "Generate synthetic training data, manage seeds, preview synthesis jobs, and work with the data factory. Use when asked to generate synthetic data, create training examples, augment a dataset, manage seeds, or check factory job status."
---

# Factory

Synthetic data generation and training data management.

## Seeds

List existing seeds:
```bash
scripts/api.sh GET /factory/seeds
```

Create a new seed:
```bash
scripts/api.sh POST /factory/seeds '{"content": "...", "tags": ["python", "refactor"]}'
```

Create seeds from gold traces:
```bash
scripts/api.sh POST /factory/seeds/from-traces '{}'
```

Delete a seed:
```bash
scripts/api.sh DELETE /factory/seeds/{seed_id}
```

## Synthesis

Preview what synthesis would produce:
```bash
scripts/api.sh POST /factory/preview '{"seed_ids": ["seed-1"], "count": 5}'
```

Start synthesis job:
```bash
scripts/api.sh POST /factory/synthesize '{"seed_ids": ["seed-1"], "count": 10}'
```

## Synthetic Data Generation

Generate synthetic training examples:
```bash
scripts/api.sh POST /factory/synthetic/generate '{
  "preset": "code_completion",
  "count": 50
}'
```

List synthetic generation jobs:
```bash
scripts/api.sh GET /factory/synthetic/jobs
```

Check job status:
```bash
scripts/api.sh GET /factory/synthetic/jobs/{job_id}
```

Available presets:
```bash
scripts/api.sh GET /factory/synthetic/presets
```

## Factory Configuration

```bash
scripts/api.sh GET /factory/config
scripts/api.sh PUT /factory/config '{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}'
```

## Example Interactions

User: "Generate 50 synthetic training examples"
→ Call GET /factory/synthetic/presets to show options. Ask which preset.
   Call POST /factory/synthetic/generate. Report job_id.

User: "Create seeds from my gold traces"
→ Call POST /factory/seeds/from-traces. Report how many seeds created.

User: "What seeds do I have?"
→ Call GET /factory/seeds. List with tags and content previews.
```

**Step 2: Commit**

```bash
git add assistant/workspace/skills/factory/
git commit -m "feat(assistant): add factory skill

Wraps /factory/* endpoints for seed management, synthesis,
synthetic data generation, and factory configuration."
```

---

### Task 10: Create picoclaw configuration

**Files:**
- Create: `assistant/config/config.json`
- Create: `assistant/config/config.example.json`
- Create: `assistant/.env.example`

**Step 1: Write config.example.json**

This is the template users copy and fill in:

```json
{
  "agents": {
    "defaults": {
      "workspace": "/root/.picoclaw/workspace",
      "restrict_to_workspace": true,
      "model": "claude-sonnet-4-5-20250929",
      "max_tokens": 8192,
      "temperature": 0.3,
      "max_tool_iterations": 25
    }
  },
  "channels": {
    "telegram": {
      "enabled": false,
      "token": "YOUR_TELEGRAM_BOT_TOKEN",
      "allow_from": ["YOUR_TELEGRAM_USER_ID"]
    },
    "discord": {
      "enabled": false,
      "token": "YOUR_DISCORD_BOT_TOKEN",
      "allow_from": ["YOUR_DISCORD_USER_ID"]
    }
  },
  "providers": {
    "anthropic": {
      "api_key": "",
      "api_base": ""
    }
  },
  "tools": {
    "web": {
      "search": {
        "api_key": "",
        "max_results": 5
      }
    }
  },
  "heartbeat": {
    "enabled": true,
    "interval": 60
  }
}
```

**Step 2: Write .env.example**

```
# picoclaw Gateway Environment
# Copy to .env and fill in your values

# Chat Channel Tokens
TELEGRAM_BOT_TOKEN=
DISCORD_BOT_TOKEN=

# LLM Provider
ANTHROPIC_API_KEY=

# Web Search (optional)
BRAVE_API_KEY=

# BashGym API (set automatically in Docker, override for local dev)
BASHGYM_API_URL=http://bashgym-api:8003/api
```

**Step 3: Copy example to config.json (gitignored)**

```bash
cp assistant/config/config.example.json assistant/config/config.json
```

**Step 4: Add config.json and .env to .gitignore**

Append to the project's `.gitignore`:
```
# Assistant secrets
assistant/config/config.json
assistant/.env
```

**Step 5: Commit**

```bash
git add assistant/config/config.example.json assistant/.env.example
git add .gitignore
git commit -m "feat(assistant): add picoclaw config templates

config.example.json and .env.example with placeholders.
Actual config.json and .env are gitignored."
```

---

### Task 11: Create Dockerfile for picoclaw

**Files:**
- Create: `assistant/Dockerfile`

**Step 1: Write the Dockerfile**

```dockerfile
# ============================================================
# Stage 1: Build the picoclaw binary
# ============================================================
FROM golang:1.23-alpine AS builder

RUN apk add --no-cache git make

WORKDIR /src

# Cache Go dependencies
COPY picoclaw/go.mod picoclaw/go.sum ./
RUN go mod download

# Copy picoclaw source and build
COPY picoclaw/ .
RUN make build || go build -o build/picoclaw ./cmd/picoclaw/

# ============================================================
# Stage 2: Minimal runtime image
# ============================================================
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata curl bash

# Copy picoclaw binary
COPY --from=builder /src/build/picoclaw /usr/local/bin/picoclaw

# Initialize picoclaw home
RUN /usr/local/bin/picoclaw onboard 2>/dev/null || true

# Copy our workspace (will be overridden by volume mount in dev)
COPY workspace/ /root/.picoclaw/workspace/

# Copy shared scripts into workspace
COPY workspace/scripts/ /root/.picoclaw/workspace/scripts/

# Copy config (will be overridden by volume mount)
COPY config/config.example.json /root/.picoclaw/config.json

ENTRYPOINT ["picoclaw"]
CMD ["gateway"]
```

Notes:
- Uses Go 1.23 (not 1.26 from picoclaw's Dockerfile — adjust to match what's in picoclaw's go.mod)
- Includes `curl` and `bash` for the api.sh skill scripts
- Workspace and config are baked in but overridden by Docker Compose volume mounts in dev

**Step 2: Verify Dockerfile syntax**

Run: `docker build --check assistant/` (if Docker supports it) or just visually verify.

**Step 3: Commit**

```bash
git add assistant/Dockerfile
git commit -m "feat(assistant): add Dockerfile for picoclaw gateway

Multi-stage build: Go builder + Alpine runtime with curl/bash
for skill scripts."
```

---

### Task 12: Create Docker Compose configuration

**Files:**
- Create: `docker-compose.yml` (project root — new file, separate from docker/docker-compose.yml)

**Step 1: Check existing docker compose files**

Run: `ls docker/docker-compose.yml`
Expected: Exists (arena/sandbox compose). We create a NEW top-level compose file for the assistant + API stack.

**Step 2: Write docker-compose.yml at project root**

```yaml
# GhostWork Development Stack
# Usage:
#   docker compose up                    # Start all services
#   docker compose up bashgym-api        # API only
#   docker compose up picoclaw-gateway   # Assistant only (needs API)
#   docker compose run --rm picoclaw-agent -m "system status"  # One-shot query

services:
  # ─────────────────────────────────────
  # BashGym API (FastAPI backend)
  # ─────────────────────────────────────
  bashgym-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.arena
    container_name: bashgym-api
    ports:
      - "8003:8003"
    volumes:
      - ./data:/app/data
      - ./bashgym:/app/bashgym
    env_file: .env
    command: ["python", "-m", "uvicorn", "bashgym.api.routes:app", "--host", "0.0.0.0", "--port", "8003"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ─────────────────────────────────────
  # PicoClaw Gateway (long-running bot)
  # ─────────────────────────────────────
  picoclaw-gateway:
    build:
      context: ./assistant
      dockerfile: Dockerfile
    container_name: picoclaw-gateway
    depends_on:
      bashgym-api:
        condition: service_healthy
    volumes:
      - ./assistant/workspace:/root/.picoclaw/workspace
      - ./assistant/config/config.json:/root/.picoclaw/config.json:ro
    env_file: ./assistant/.env
    environment:
      - BASHGYM_API_URL=http://bashgym-api:8003/api
    restart: unless-stopped
    command: ["gateway"]

  # ─────────────────────────────────────
  # PicoClaw Agent (one-shot query)
  #   docker compose run --rm picoclaw-agent -m "check system status"
  # ─────────────────────────────────────
  picoclaw-agent:
    build:
      context: ./assistant
      dockerfile: Dockerfile
    container_name: picoclaw-agent
    profiles:
      - agent
    depends_on:
      bashgym-api:
        condition: service_healthy
    volumes:
      - ./assistant/workspace:/root/.picoclaw/workspace
      - ./assistant/config/config.json:/root/.picoclaw/config.json:ro
    env_file: ./assistant/.env
    environment:
      - BASHGYM_API_URL=http://bashgym-api:8003/api
    entrypoint: ["picoclaw", "agent"]
    stdin_open: true
    tty: true
```

**Step 3: Verify syntax**

Run: `docker compose config --quiet` (from project root)
Expected: No errors

**Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add top-level docker-compose for API + assistant

Three services: bashgym-api (FastAPI), picoclaw-gateway (persistent bot),
picoclaw-agent (one-shot queries). Gateway waits for API health check."
```

---

### Task 13: Verify Docker build

**Files:** None (verification only)

**Step 1: Build the API image**

Run: `docker compose build bashgym-api`
Expected: Builds successfully (may need Dockerfile.arena adjustments — see notes)

Note: If `docker/Dockerfile.arena` doesn't have a CMD for uvicorn, the compose command override handles it. If the Dockerfile.arena is sandbox-specific and not suitable, create a new `Dockerfile.api` at project root. Adapt as needed.

**Step 2: Build the picoclaw image**

Run: `docker compose build picoclaw-gateway`
Expected: Go binary compiles, Alpine image built with curl/bash

**Step 3: Test the API starts**

Run: `docker compose up bashgym-api -d`
Run: `curl http://localhost:8003/api/health`
Expected: `{"status": "ok", ...}`

**Step 4: Test picoclaw one-shot**

Run: `docker compose run --rm picoclaw-agent -m "hello"`
Expected: picoclaw responds via configured LLM provider (requires valid ANTHROPIC_API_KEY in assistant/.env)

**Step 5: Test picoclaw skill access**

Run: `docker compose run --rm picoclaw-agent -m "check system status"`
Expected: The agent uses the system skill to call /api/health and /api/system/info

If any step fails, fix the issue before proceeding.

---

### Task 14: Integration smoke test

**Files:** None (verification only)

**Step 1: Start full stack**

Run: `docker compose up -d`
Expected: bashgym-api and picoclaw-gateway both running

**Step 2: Test each skill via one-shot agent**

Test system skill:
```bash
docker compose run --rm picoclaw-agent -m "give me a system health check"
```

Test traces skill:
```bash
docker compose run --rm picoclaw-agent -m "how many gold traces do I have?"
```

Test models skill:
```bash
docker compose run --rm picoclaw-agent -m "list my models"
```

Test training skill:
```bash
docker compose run --rm picoclaw-agent -m "are any training runs active?"
```

Test orchestrator skill:
```bash
docker compose run --rm picoclaw-agent -m "list orchestrator jobs"
```

Test factory skill:
```bash
docker compose run --rm picoclaw-agent -m "what seeds do I have?"
```

Expected: Each query triggers the appropriate skill, calls the correct API endpoint, and returns meaningful data (even if empty — "No gold traces found" is valid).

**Step 3: Verify memory persistence**

Run: `docker compose run --rm picoclaw-agent -m "remember that I prefer SFT over DPO"`
Check: `cat assistant/workspace/memory/MEMORY.md`
Expected: Memory file updated with the preference

**Step 4: Tear down**

Run: `docker compose down`

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(assistant): complete picoclaw global assistant integration

6 BashGym skills (system, orchestrator, training, traces, models, factory),
Docker Compose stack, workspace identity, and configuration templates.
Verified via one-shot agent smoke tests."
```

---

## Task Summary

| Task | Description | Est. Complexity |
|------|-------------|-----------------|
| 1 | Import picoclaw via git subtree | Low |
| 2 | Create workspace identity files | Low |
| 3 | Create shared API helper script | Low |
| 4 | Create system skill | Low |
| 5 | Create orchestrator skill | Medium |
| 6 | Create training skill | Low |
| 7 | Create traces skill | Medium |
| 8 | Create models skill | Low |
| 9 | Create factory skill | Low |
| 10 | Create picoclaw configuration | Low |
| 11 | Create Dockerfile | Medium |
| 12 | Create Docker Compose | Medium |
| 13 | Verify Docker build | Medium (debugging) |
| 14 | Integration smoke test | Medium (debugging) |

**Total: 14 tasks.** Tasks 1-10 are pure file creation. Tasks 11-14 involve Docker and may require debugging.

## Key Corrections from Design Doc

The design doc had a few inaccuracies fixed in this plan:
- API prefix is `/api/orchestrator/` not `/api/orchestrate/`
- Retry path is `/orchestrator/{job_id}/task/{task_id}/retry` (singular `task`)
- No separate `/plan` endpoint — plan comes from the submit response
- Factory has more endpoints than originally listed (synthetic/, designer/, seeds/)
- Models has 14 endpoints including leaderboard, trends, deploy-ollama, artifacts
- Training has pause/resume in addition to start/stop
- Traces has stats, repos, sync, auto-classify endpoints
