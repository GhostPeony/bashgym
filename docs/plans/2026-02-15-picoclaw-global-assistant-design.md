# PicoClaw Global Assistant Design

**Date:** 2026-02-15
**Status:** Approved
**Approach:** Vanilla picoclaw fork + BashGym skills (Approach A)

---

## Overview

A conversational AI assistant built on [picoclaw](https://github.com/sipeed/picoclaw) that serves as the user-facing shell for the entire GhostWork/BashGym ecosystem. Reachable via Discord and Telegram, with full control over orchestration, training, traces, models, factory, and system infrastructure.

picoclaw handles chat channels, identity, memory, and skill routing. BashGym's existing FastAPI backend handles all domain logic. The two run as separate Docker services on a shared network.

## Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Discord/Telegram │         │   Electron UI    │
│    (channels)     │         │   (existing)     │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         ▼                            │
┌──────────────────┐                  │
│ picoclaw-gateway │                  │
│   (Go binary)    │                  │
│                  │                  │
│  IDENTITY.md     │                  │
│  SOUL.md         │                  │
│  AGENT.md        │                  │
│  skills/         │                  │
│  memory/         │                  │
└────────┬─────────┘                  │
         │ HTTP (Docker internal)     │
         ▼                            ▼
┌──────────────────────────────────────┐
│         bashgym-api (FastAPI)        │
│         port 8003                    │
│                                      │
│  /api/orchestrate/*                  │
│  /api/training/*                     │
│  /api/traces/*                       │
│  /api/models/*                       │
│  /api/system/*                       │
│  /api/factory/*                      │
└──────────────────────────────────────┘
```

### Docker Compose

```yaml
services:
  bashgym-api:
    build: .
    ports: ["8003:8003"]
    volumes:
      - ./data:/app/data
    env_file: .env

  picoclaw-gateway:
    build: ./assistant
    depends_on: [bashgym-api]
    volumes:
      - ./assistant/workspace:/root/.picoclaw/workspace
      - ./assistant/config/config.json:/root/.picoclaw/config.json:ro
    env_file: ./assistant/.env
    environment:
      - BASHGYM_API_URL=http://bashgym-api:8003/api
    restart: unless-stopped
```

## Directory Structure

```
ghostwork/
├── assistant/                        # NEW - picoclaw integration
│   ├── picoclaw/                     # Mirrored picoclaw source (git subtree)
│   │   ├── cmd/picoclaw/main.go
│   │   ├── pkg/
│   │   │   ├── agent/               # Agent loop, memory, context
│   │   │   ├── channels/            # Telegram, Discord, Slack, etc.
│   │   │   ├── providers/           # Anthropic, OpenAI, Gemini, etc.
│   │   │   ├── tools/               # Shell, filesystem, web, spawn, subagent
│   │   │   ├── skills/              # Skill loader
│   │   │   ├── session/             # Session management
│   │   │   ├── config/              # Config loading
│   │   │   └── ...
│   │   ├── go.mod
│   │   ├── go.sum
│   │   └── Makefile
│   ├── workspace/                    # Our custom workspace
│   │   ├── IDENTITY.md
│   │   ├── SOUL.md
│   │   ├── AGENT.md
│   │   ├── USER.md
│   │   ├── memory/
│   │   │   └── MEMORY.md
│   │   └── skills/
│   │       ├── orchestrator/
│   │       │   ├── SKILL.md
│   │       │   └── scripts/api.sh
│   │       ├── training/
│   │       │   └── SKILL.md
│   │       ├── traces/
│   │       │   └── SKILL.md
│   │       ├── models/
│   │       │   └── SKILL.md
│   │       ├── system/
│   │       │   └── SKILL.md
│   │       └── factory/
│   │           └── SKILL.md
│   ├── config/
│   │   ├── config.json
│   │   └── config.example.json
│   ├── Dockerfile
│   └── .env
├── docker-compose.yml                # Updated with picoclaw-gateway service
├── bashgym/                          # Existing - unchanged
└── frontend/                         # Existing - unchanged
```

## Workspace Identity

### IDENTITY.md

```markdown
# GhostWork Assistant

A system-aware development assistant for the GhostWork ecosystem.
Manages orchestration, training, traces, models, and infrastructure
across all Ghost Peony projects.

Built on picoclaw, powered by Claude.
```

### SOUL.md

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

### AGENT.md

```markdown
# Agent Instructions

You are a development operations assistant with full access to the
GhostWork/BashGym system via API.

## Guidelines
- Always confirm before destructive operations (stopping training, demoting traces)
- Report system status with concrete numbers (GPU %, trace counts, budget remaining)
- When submitting orchestrator specs, summarize the plan before asking for approval
- Use skills to interact with BashGym — don't improvise API calls
- Remember user preferences and past decisions in memory files
- If an API call fails, report the error and suggest next steps
```

### USER.md

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

## Skills

### Shared API Helper

All skills share a common `scripts/api.sh` helper located at the workspace level or duplicated per skill:

```bash
#!/bin/bash
# Usage: api.sh GET /orchestrate/jobs
# Usage: api.sh POST /training/start '{"strategy":"sft"}'
METHOD=$1; ENDPOINT=$2; BODY=$3
curl -s -X "$METHOD" "${BASHGYM_API_URL}${ENDPOINT}" \
  -H "Content-Type: application/json" \
  ${BODY:+-d "$BODY"}
```

### Skill 1: orchestrator

**Triggers:** submit spec, start job, check workers, approve plan, retry task, job status

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/orchestrate/submit` | Submit a development spec |
| GET | `/orchestrate/{job_id}/plan` | View decomposed task plan |
| POST | `/orchestrate/{job_id}/approve` | Approve plan and begin execution |
| GET | `/orchestrate/{job_id}/status` | Worker progress, task states, budget |
| POST | `/orchestrate/{job_id}/tasks/{task_id}/retry` | Retry a failed task |
| GET | `/orchestrate/jobs` | List all orchestration jobs |

### Skill 2: training

**Triggers:** start training, check training, view loss, stop training, training logs

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/training/start` | Start SFT/DPO/GRPO run |
| GET | `/training/{run_id}` | Run status and metrics |
| GET | `/training/logs` | Real-time log tail |
| POST | `/training/{run_id}/stop` | Stop a running training job |
| GET | `/system/info` | GPU utilization during training |

### Skill 3: traces

**Triggers:** show traces, promote, demote, generate examples, gold trace count

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/traces` | List traces with filters (gold/pending/failed) |
| POST | `/traces/{id}/promote` | Promote trace to gold |
| POST | `/traces/{id}/demote` | Demote trace to failed |
| POST | `/traces/{id}/generate-examples` | Generate training examples from trace |
| GET | `/training/examples` | List generated training examples |
| POST | `/training/export` | Export examples to NeMo JSONL format |

### Skill 4: models

**Triggers:** list models, compare models, model lineage, best model

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/models` | List all registered models |
| GET | `/models/{id}` | Model details and metadata |
| POST | `/models/compare` | Compare two models head-to-head |
| GET | `/models/{id}/lineage` | Training lineage tree |

### Skill 5: system

**Triggers:** system status, GPU usage, health check, what's running

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service health check |
| GET | `/system/info` | Hardware, GPU, memory, disk |
| GET | `/stats` | Aggregate statistics across subsystems |

### Skill 6: factory

**Triggers:** generate synthetic data, create examples, augment dataset, seeds

**Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/factory/synthetic/generate` | Generate synthetic training examples |
| GET | `/factory/examples` | List generated examples |
| GET | `/factory/seeds` | View seed data |

## Configuration

### config.json

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
      "enabled": true,
      "token": "${TELEGRAM_BOT_TOKEN}",
      "allow_from": ["YOUR_TELEGRAM_USER_ID"]
    },
    "discord": {
      "enabled": true,
      "token": "${DISCORD_BOT_TOKEN}",
      "allow_from": ["YOUR_DISCORD_USER_ID"]
    }
  },
  "providers": {
    "anthropic": {
      "api_key": "${ANTHROPIC_API_KEY}"
    }
  },
  "tools": {
    "web": {
      "search": {
        "api_key": "${BRAVE_API_KEY}",
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

### Environment Variables (assistant/.env)

```
TELEGRAM_BOT_TOKEN=...
DISCORD_BOT_TOKEN=...
ANTHROPIC_API_KEY=...
BRAVE_API_KEY=...
BASHGYM_API_URL=http://bashgym-api:8003/api
```

## Security Model

| Layer | Mechanism |
|-------|-----------|
| Channel auth | `allow_from` whitelist — only specified user IDs can interact |
| Workspace isolation | `restrict_to_workspace: true` — picoclaw can only access its own workspace |
| API boundary | All BashGym interaction via FastAPI HTTP — no direct DB/file access |
| Network isolation | Docker internal network — picoclaw-gateway only reaches bashgym-api |
| Destructive ops | AGENT.md instructs LLM to confirm before destructive operations (soft guardrail) |
| Shell safety | picoclaw blocks dangerous commands (rm -rf /, fork bombs, etc.) by default |

## Local Mirroring Strategy

picoclaw source is mirrored via git subtree:

```bash
# Initial import
git subtree add --prefix=assistant/picoclaw \
  https://github.com/sipeed/picoclaw.git main --squash

# Pull upstream updates
git subtree pull --prefix=assistant/picoclaw \
  https://github.com/sipeed/picoclaw.git main --squash
```

**Rationale:** Subtree over submodule because:
- Code lives directly in our repo, fully self-contained
- Can modify picoclaw files directly without detached HEAD issues
- Cloning ghostwork gets everything — no submodule init needed
- Selective upstream syncing when we choose

**Kept from picoclaw:** `cmd/`, `pkg/`, `go.mod`, `go.sum`, `Makefile`
**Replaced with ours:** `workspace/`, `config/`, `Dockerfile`

## Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| Approach A (skills + curl) over B (Go tool) | Fastest to ship, skills are easy to iterate, can upgrade later |
| Claude Sonnet 4.5 as default model | Best balance of capability and cost for operational tasks |
| temperature 0.3 | Operational assistant should be precise, not creative |
| git subtree over submodule | Self-contained, easier to modify, no dangling references |
| Neutral identity over branded | Functional tool, not a product persona |
| No BashGym API auth | Channel whitelist + Docker isolation is sufficient for single-user setup |

## Future Considerations

- **Upgrade to Approach B** if curl-based skills prove unreliable (structured Go tool)
- **Add MCP support** if other agents need to consume BashGym's API
- **WebSocket streaming** for real-time training log tailing via chat
- **Heartbeat jobs** for automated monitoring (e.g., alert if training diverges, GPU temp high)
- **Additional channels** (Slack, WhatsApp) as needed
