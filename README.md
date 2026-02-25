# Bash Gym

**Train your own coding model from your Claude Code sessions.**

<img width="1563" height="908" alt="Bash Gym home screen" src="https://github.com/user-attachments/assets/472d7ada-d57b-4e79-8604-c27e4fc99348" />

---

## The Problem

Training an open source model is hard. Getting quality training data is harder. But if you've been using Claude Code, you're already sitting on exactly the data you need — months of real coding sessions, tool calls, file edits, and bash commands across your actual projects. That data is just sitting in `~/.claude/projects/`.

Bash Gym turns those sessions into training data and fine-tunes small, free-to-run open source models (1.5B–7B parameters) that learn from how Claude works on *your* code.

The result: a local model trained on your projects, your patterns, your conventions. Runs on your hardware. Costs nothing per query.

---

## How It Works

```
Use Claude Code normally
        ↓
Hooks capture every session as structured traces
        ↓
Score, filter, and segment traces into training examples
        ↓
Fine-tune a small model (Qwen 1.5B–7B) with LoRA
        ↓
Run locally via Ollama — free, fast, private
```

| Stage | What Happens |
|-------|--------------|
| **Capture** | Hooks record every tool call, file edit, and command from your Claude Code sessions. |
| **Curate** | Traces are scored on 6 quality metrics. Good sessions become gold training data, bad ones become negative examples. |
| **Synthesize** | Gold traces are segmented into task-response pairs. PII is scrubbed. Gaps are filled with synthetic augmentation. |
| **Train** | SFT, DPO, or GRPO fine-tunes a small model using Unsloth (2–5x faster, 50–80% less VRAM). |
| **Route** | Confidence-based routing shifts simple tasks from Claude to your local model over time. |

Each cycle improves the model. More usage → more traces → better data → better model.

---

## Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | Backend API and training scripts |
| **Node.js** | 18+ LTS | Frontend dashboard |
| **Anthropic API key** | — | Get one at [console.anthropic.com](https://console.anthropic.com/) |
| **CUDA GPU** | 8GB+ VRAM | Only needed for local training. Not required for trace capture or the dashboard. |

On Windows, `npm install` requires Visual Studio Build Tools with "Desktop development with C++" (for `node-pty` native compilation).

No GPU? Use [HuggingFace Cloud Training](#cloud-training) instead ($0.60–$4.50/hr).

### 1. Install

```bash
git clone https://github.com/GhostPeony/bashgym.git
cd bashgym

# Python dependencies
pip install -r requirements.txt

# Training dependencies (optional — skip if no local GPU)
pip install -r requirements-training.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and add your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

That's the only required key. Everything else has working defaults. See [Configuration](#configuration) for optional keys.

### 3. Install Trace Capture Hooks

These hooks run silently alongside Claude Code and capture every session as structured JSON:

```bash
cp bashgym/hooks/*.py ~/.claude/hooks/
```

Verify: launch the app and check the sidebar — "Hooks" should show "Installed."

### 4. Start

```bash
# Windows (PowerShell)
.\dev.ps1                  # Backend + frontend
.\dev.ps1 -Electron        # Backend + desktop app

# macOS / Linux
./dev.sh                   # Backend + frontend
./dev.sh --electron        # Backend + desktop app

# Docker (any platform)
docker compose up
```

Backend starts on `localhost:8003`, frontend on `localhost:5173`.

### 5. Use Claude Code Normally

That's it. Work on your projects with Claude Code like you always do. Every session is captured automatically. Check the **Traces** tab in the dashboard to watch them accumulate.

Once you have 20–30 gold traces, you're ready to train. See **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** for the full walkthrough from first trace to first trained model.

---

## Dashboard

Electron + React app with real-time WebSocket updates.

| Section | Purpose |
|---------|---------|
| **Workspace** | Multi-terminal canvas with node graph — terminals, browsers, integration nodes (Neon, Vercel, Context) linked with edges |
| **Traces** | Browse captured sessions, view quality scores, promote/demote |
| **Training** | Configure and monitor training runs, view loss curves |
| **Agent** | Peony — built-in assistant that can import traces, trigger training, search HuggingFace, run commands |
| **Models** | Browse trained models, view lineage, compare evaluations |
| **Router** | Monitor teacher/student routing decisions |

---

## Training

### Strategies

| Strategy | What It Does | When To Use |
|----------|--------------|-------------|
| **SFT** | Trains the model to reproduce successful traces | Start here. Works with 20–30 gold traces. |
| **DPO** | Learns from pairs of good and bad responses | When you have both gold and failed traces. |
| **GRPO** | RL-based — model generates solutions, learns from rewards | Advanced. Needs more data. |

### Base Models

| Model | Parameters | VRAM |
|-------|------------|------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | 8GB (default) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | 16GB |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 12GB |

All training uses QLoRA (4-bit quantization) by default. Output: LoRA adapter (~50MB), merged weights, or GGUF for Ollama.

### Cloud Training

No local GPU? Use HuggingFace Cloud Training (Unsloth Jobs):

| Tier | GPU | VRAM | Cost/hr |
|------|-----|------|---------|
| `t4-small` | T4 | 16GB | $0.60 |
| `a10g-small` | A10G | 24GB | $1.05 |
| `a100-large` | A100 | 80GB | $4.50 |

Requires HuggingFace Pro subscription.

---

## Canvas & Integration Nodes

The workspace is a node graph where terminals, browsers, and integration nodes connect via edges to share context.

| Node Type | What It Does |
|-----------|--------------|
| **Terminal** | Claude Code session with live status, metrics, tool history |
| **Browser** | Live web preview — screenshots route to linked terminals |
| **Context** | Persistent notes, file references, URLs, snippets — sends content to linked terminals |
| **Neon** | Database schema introspection, query execution — sends schema/results to terminals |
| **Vercel** | Deploy status, build logs, v0 AI generation — sends code/logs to terminals |

**Shift+drag** to box-select multiple nodes and auto-connect them. Connected nodes share context through edge routing.

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `BASE_MODEL` | No | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Fine-tuning base model |
| `HF_TOKEN` | No | — | HuggingFace token (for cloud training) |
| `NVIDIA_API_KEY` | No | — | NVIDIA NIM API key (for augmentation) |
| `ROUTING_STRATEGY` | No | `confidence_based` | Model routing strategy |

---

## Project Structure

```
bashgym/
├── bashgym/                  # Python package
│   ├── arena/                # Execution (runner, sandbox)
│   ├── judge/                # Verification (evaluator, guardrails, benchmarks)
│   ├── factory/              # Data synthesis (trace processor, example generator)
│   ├── gym/                  # Training (trainer, environment, router)
│   ├── hooks/                # Claude Code trace capture hooks
│   ├── models/               # Model registry and lifecycle
│   ├── api/                  # REST API + WebSocket
│   └── integrations/         # HuggingFace, NeMo, Ollama
│
├── frontend/                 # Electron + React dashboard
│   ├── src/components/       # 72+ React components
│   ├── src/stores/           # Zustand state management
│   └── electron/             # Main process + secure storage
│
├── assistant/                # Peony chat assistant (Go + Docker)
├── tests/                    # Test suite
└── docker-compose.yml        # Full stack deployment
```

---

## FAQ

**Do I need a GPU?**
No. Trace capture, curation, and the dashboard work without one. Training requires a CUDA GPU (8GB+ VRAM) or you can use HuggingFace Cloud Training.

**How many traces before I can train?**
20–30 gold traces for a basic SFT run. 100+ traces produce noticeably better results. More repos and task diversity = more generalizable model.

**Does this send my code anywhere?**
Only to the LLM providers you already use (Anthropic, etc.). Traces, training data, and models stay local. HuggingFace uploads only happen when you explicitly push.

**What's a trace vs a training example?**
A trace is a complete Claude Code session (many tool calls, potentially 30+ minutes). A training example is a single task-response pair extracted from that trace. One trace typically produces 1–5 examples.

**Can I use other base models?**
Yes. Set `BASE_MODEL` in `.env` to any HuggingFace model ID. Any causal LM compatible with Transformers + LoRA should work.

---

## License

MIT — see [LICENSE](LICENSE).
