# Bash Gym

**A Self-Improving Agentic Development Gym**

<img width="1563" height="908" alt="Bash Gym home screen" src="https://github.com/user-attachments/assets/472d7ada-d57b-4e79-8604-c27e4fc99348" />

Bash Gym trains smaller language models (SLMs) from agent execution traces. It implements a continuous improvement flywheel where your daily Claude Code usage automatically generates training data for your own personalized coding assistant.

```
ACT (Arena) → VERIFY (Judge) → SYNTHESIZE (Factory) → TRAIN (Gym) → DEPLOY
     ↑                                                                  |
     +------------------------------------------------------------------+
```

---

## Why Bash Gym

AI-assisted coding is expensive, latency-bound, and generic. Bash Gym solves all three:

| Problem | How Bash Gym Solves It |
|---------|----------------------|
| **API costs at scale** ($50-200/mo per developer) | Route simple tasks to a local 1.5-7B model. 40-70% cost reduction at steady state. |
| **Latency on simple tasks** (2-8s round-trip) | Local inference for routed tasks (~50ms). 10-50x faster on routine completions. |
| **Generic behavior** (doesn't know your codebase) | Fine-tuned on *your* coding patterns, conventions, and repositories. |
| **No learning from corrections** | Every correction and successful interaction becomes training data automatically. |
| **Vendor lock-in** | You own all traces, models, and training data. Export to HuggingFace, Ollama, or GGUF. |

The flywheel is self-reinforcing: more usage generates more traces, which produce better training data, which improve the student model, which handles more tasks, which generates even more high-quality traces.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous Trace Capture** | Hooks into Claude Code to automatically capture execution traces |
| **6-Metric Quality Framework** | Comprehensive trace scoring for training data quality |
| **Privacy by Design** | PII detection across 20+ types with differential privacy support |
| **Multiple Training Strategies** | SFT, DPO, GRPO, RLVR, and distillation |
| **Model Registry** | Full lifecycle tracking with lineage, metrics, and artifacts |
| **Progressive Routing** | Confidence-based handoff from teacher to student models |
| **Real-Time Dashboard** | Electron + React frontend with WebSocket updates |
| **Multi-Cloud Deployment** | Export to HuggingFace, NVIDIA NIM, Ollama |
| **Comprehensive Benchmarks** | HumanEval, MBPP, BigCodeBench, SWE-bench, and more |
| **Safety Guardrails** | Injection detection, content moderation, dangerous command blocking |
| **Multi-Agent Orchestrator** | Decompose specs into parallel tasks with DAG-based execution |
| **Peony Chat Assistant** | Discord/Telegram bot with full system control via natural language |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/GhostPeony/bashgym.git
cd bashgym
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Configure
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY (the only required key)

# Install trace capture hooks
cp bashgym/hooks/*.py ~/.claude/hooks/

# Start (pick your platform)

# Windows (PowerShell)
.\dev.ps1

# macOS / Linux
./dev.sh

# Any platform (Docker)
docker compose up
```

The backend starts on port 8003, the frontend on port 5173. Open `http://localhost:5173` or use `.\dev.ps1 -Electron` / `./dev.sh --electron` for the desktop app.

For training capabilities, also install: `pip install -r requirements-training.txt`

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                       THE OUROBOROS FLYWHEEL                          │
│                                                                        │
│    ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌─────────┐        │
│    │   ACT   │───▶│ VERIFY  │───▶│ SYNTHESIZE│───▶│  TRAIN  │        │
│    │ (Arena) │    │ (Judge) │    │ (Factory) │    │  (Gym)  │        │
│    └─────────┘    └─────────┘    └───────────┘    └─────────┘        │
│         ▲                                              │              │
│         │                                              │              │
│         └──────────────── DEPLOY ◀─────────────────────┘              │
│                           (Router)                                     │
└──────────────────────────────────────────────────────────────────────┘
```

| Stage | Module | What Happens |
|-------|--------|--------------|
| **ACT** | Arena | You use Claude Code normally. Hooks capture every tool call, file edit, and bash command as structured traces. |
| **VERIFY** | Judge | Traces are scored on 6 quality metrics. Tests validate solutions. Guardrails check safety. |
| **SYNTHESIZE** | Factory | Gold traces are segmented into training examples. PII is scrubbed. Synthetic augmentation fills gaps. |
| **TRAIN** | Gym | SFT, DPO, GRPO, RLVR, or distillation fine-tunes a small model (1.5-7B) using Unsloth acceleration. |
| **DEPLOY** | Router | Confidence-based routing progressively shifts traffic from teacher (Claude) to student (your model). |

---

## Glossary

Key terms used throughout Bash Gym:

| Term | Definition |
|------|-----------|
| **Trace** | A complete record of a Claude Code coding session — every tool call, file edit, bash command, and response. Stored as JSON. |
| **Gold Trace** | A high-quality trace (>=90% success rate, >=0.75 quality score) used as positive training data. |
| **Failed Trace** | A low-quality trace (<60% success) used as negative examples in DPO training — the model learns what *not* to do. |
| **Training Example** | A single task-response pair extracted from a trace. One trace may contain multiple examples. Format: NeMo JSONL. |
| **Session** | A complete coding interaction (one trace). Contains many tool calls. A session is segmented into training examples. |
| **SFT** | Supervised Fine-Tuning. Trains the model to reproduce successful traces. The simplest and most common strategy. |
| **DPO** | Direct Preference Optimization. Trains using pairs of good and bad responses, teaching the model to prefer correct behavior. |
| **GRPO** | Group Relative Policy Optimization. RL-based training where the model generates solutions and learns from reward signals. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards. Like GRPO but the reward comes from running tests — solutions must actually pass. |
| **Distillation** | Knowledge transfer from a large teacher model (Claude) to a smaller student model. |
| **LoRA** | Low-Rank Adaptation. Fine-tunes only ~1% of model parameters, producing a small adapter (~50MB) instead of rewriting the full model. |
| **QLoRA** | Quantized LoRA. Same as LoRA but loads the base model in 4-bit precision, dramatically reducing VRAM usage. |
| **GGUF** | A model file format optimized for local inference. Used by Ollama and llama.cpp. Supports various quantization levels (q4, q8, etc.). |
| **Teacher Model** | The large, expensive cloud model (Claude) that produces high-quality outputs. |
| **Student Model** | The small, local model (1.5-7B parameters) that you fine-tune to replicate the teacher's behavior. |
| **Router** | The system that decides whether to send a task to the teacher or student model, based on confidence, complexity, or other strategies. |
| **Flywheel** | The self-reinforcing loop: Act -> Verify -> Synthesize -> Train -> Deploy -> Act. Each cycle improves the student. |
| **Arena** | The execution layer. Where Claude Code runs tasks (either natively on your PC or in Docker sandboxes). |
| **Judge** | The verification layer. Scores traces, runs tests, checks safety, evaluates quality. |
| **Factory** | The data synthesis layer. Transforms raw traces into training-ready datasets with privacy guarantees. |
| **Gym** | The training layer. Fine-tunes models using various strategies (SFT, DPO, GRPO, etc.). |
| **Task DAG** | Directed Acyclic Graph. The orchestrator decomposes a spec into tasks with dependencies — tasks execute in parallel where possible. |
| **Worktree** | A git worktree — an isolated copy of the repository. Each orchestrator worker gets its own worktree to avoid conflicts. |
| **Peony** | The conversational AI assistant that provides remote control over Bash Gym via Discord and Telegram. |
| **PII** | Personally Identifiable Information. Emails, phone numbers, API keys, etc. Automatically detected and scrubbed from training data. |
| **Unsloth** | A training acceleration library. Makes fine-tuning 2-5x faster and uses 50-80% less memory. |

---

## Getting Started Tutorial

For the complete 10-step walkthrough from install to first trained model, see **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

The short version:

1. Install and configure (add `ANTHROPIC_API_KEY` to `.env`)
2. Install trace capture hooks (`cp bashgym/hooks/*.py ~/.claude/hooks/`)
3. Use Claude Code normally for a week — traces accumulate automatically
4. Review and curate traces in the Traces dashboard
5. Generate training examples from gold traces
6. Train your first model (SFT on Qwen 1.5B — takes 30-90 min with a GPU)
7. Deploy to Ollama and enable progressive routing

No local GPU? Use [HuggingFace Cloud Training](#huggingface-integration) instead.

---

## Architecture

### Layer Architecture

| Layer | Module | Purpose | Key Components |
|-------|--------|---------|----------------|
| **Arena** | `bashgym.arena` | Execution & instrumentation | Sandbox, Runner, Hooks |
| **Judge** | `bashgym.judge` | Verification & evaluation | Verifier, Evaluator, Guardrails, Benchmarks |
| **Factory** | `bashgym.factory` | Data synthesis | DataFactory, TraceProcessor, SafeSynthesizer, SchemaBuilder |
| **Gym** | `bashgym.gym` | Training & routing | Trainer, Environment, Router |
| **Models** | `bashgym.models` | Registry & lifecycle | Profile, Registry, Evaluator |
| **Observability** | `bashgym.observability` | Profiling & tracing | AgentProfiler |
| **Orchestrator** | `bashgym.orchestrator` | Multi-agent coordination | Agent, TaskDAG, Dispatcher, Synthesizer |
| **Integrations** | `bashgym.integrations` | External services | NeMo, HuggingFace, Bashbros |
| **API** | `bashgym.api` | REST & WebSocket | Routes, Schemas, WebSocket |
| **Assistant** | `assistant/` | Chat interface | Peony (Go), Skills, Identity |

### Project Structure

```
bashgym/
├── bashgym/                           # Main Python package
│   ├── arena/                         # Execution layer (runner, sandbox)
│   ├── judge/                         # Verification layer (verifier, evaluator, guardrails, benchmarks)
│   ├── factory/                       # Data synthesis (data_factory, trace_processor, safe_synthesizer, schema_builder)
│   ├── gym/                           # Training layer (trainer, environment, router)
│   ├── models/                        # Model registry (profile, registry, evaluator)
│   ├── hooks/                         # Claude Code instrumentation (post_tool_use, pre_tool_use, stop)
│   ├── observability/                 # Performance tracking (profiler)
│   ├── orchestrator/                  # Multi-agent orchestration (agent, task_dag, dispatcher, worktree)
│   ├── integrations/                  # External integrations (NeMo, HuggingFace, Bashbros)
│   ├── trace_capture/                 # Trace import system (adapters, importers)
│   ├── providers/                     # Model providers (Ollama)
│   └── api/                           # REST API + WebSocket
│
├── assistant/                         # Peony chat assistant (Go + Docker)
│   ├── picoclaw/                      # Go runtime (builds as "peony")
│   └── workspace/                     # Identity, skills, memory
│
├── frontend/                          # Electron + React UI (72+ components)
│   ├── src/components/                # React components
│   ├── src/stores/                    # Zustand state management
│   └── electron/                      # Electron main process
│
├── tests/                             # Test suite
├── docs/                              # Documentation
├── docker/                            # Docker configuration (sandbox)
├── docker-compose.yml                 # Full stack: API + Peony assistant
└── .env.example                       # Environment template
```

---

## Execution Modes

Bash Gym supports two execution modes. **Native mode is the default** for daily use.

| Mode | Where Claude Runs | Use Case |
|------|-------------------|----------|
| **Native** | On your PC directly | Real projects, daily tasks, desktop control |
| **Sandbox** | Docker containers | Autonomous batch runs for training data |

### Native Mode (Default)

Claude Code runs directly on your system. The hooks silently capture successful executions to build your personalized training dataset. **No Docker required.**

### Sandbox Mode (Optional)

For autonomous batch runs: agents run in isolated Docker containers, network-isolated and resource-limited.

---

## Peony Assistant

Peony is a conversational AI assistant that provides full remote control over the entire BashGym ecosystem via Discord and Telegram. Built on a Go runtime forked from [picoclaw](https://github.com/sipeed/picoclaw), it runs as a Docker service alongside the BashGym API.

```
┌──────────────────┐         ┌──────────────────┐
│  Discord/Telegram │         │   Electron UI    │
│    (channels)     │         │   (existing)     │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         ▼                            │
┌──────────────────┐                  │
│  peony-gateway   │                  │
│   (Go binary)    │                  │
│   Skills → HTTP  │                  │
└────────┬─────────┘                  │
         │ Docker internal network    │
         ▼                            ▼
┌──────────────────────────────────────┐
│         bashgym-api (FastAPI)        │
│         port 8003                    │
└──────────────────────────────────────┘
```

### Skills

| Skill | Triggers | Operations |
|-------|----------|------------|
| **system** | "system status", "GPU usage", "health check" | Health, hardware info, GPU, stats |
| **orchestrator** | "submit spec", "approve plan", "retry task" | Spec submission, plan approval, worker monitoring |
| **training** | "start training", "check progress", "stop run" | SFT/DPO/GRPO start/stop/pause/resume |
| **traces** | "gold traces", "promote", "generate examples" | Browse, promote/demote, example generation, export |
| **models** | "list models", "compare", "deploy to Ollama" | Leaderboard, comparison, evaluation, deployment |
| **factory** | "generate data", "create seeds" | Synthetic generation, seed management |

### Setup

```bash
# 1. Copy config templates
cp assistant/.env.example assistant/.env
cp assistant/config/config.example.json assistant/config/config.json

# 2. Fill in your tokens (TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, ANTHROPIC_API_KEY)
# 3. Start the full stack
docker compose up

# 4. Or run a one-shot query
docker compose run --rm peony-agent -m "system status"
```

---

## Orchestrator

The orchestrator decomposes development specs into parallel tasks and dispatches them to isolated Claude Code workers.

### How It Works

```
1. User submits a spec (title, description, constraints, acceptance criteria)
2. LLM decomposes spec into a Task DAG (directed acyclic graph)
3. User reviews and approves the plan
4. Tasks execute in dependency order with parallel workers
5. Each worker runs in an isolated git worktree
6. Results merge back, traces feed the training pipeline
```

### Multi-Provider Planning

| Provider | Models | Notes |
|----------|--------|-------|
| **Anthropic** | Claude Opus 4.6, Sonnet 4.5 | Default, recommended |
| **OpenAI** | GPT-4o, o1 | Alternative |
| **Google** | Gemini 2.5 Pro | Alternative |
| **Ollama** | qwen2.5-coder:32b, etc. | Local, free |

Workers always use Claude Code CLI regardless of planning provider.

### Example

```bash
curl -X POST http://localhost:8003/api/orchestrate/submit \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Add JWT authentication",
    "description": "Implement login/register with JWT tokens",
    "constraints": ["Use existing User model", "Add pytest tests"],
    "acceptance_criteria": ["All tests pass", "Proper error handling"],
    "max_workers": 3,
    "budget_usd": 5.0
  }'

# Or via Peony (Discord/Telegram):
# "Build me an auth system with JWT, max 3 workers, $5 budget"
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API key |
| `NVIDIA_API_KEY` | No | - | NVIDIA NIM API key |
| `HF_TOKEN` | No | - | HuggingFace token |
| `BASE_MODEL` | No | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Base model for fine-tuning |
| `AUGMENTATION_PROVIDER` | No | `anthropic` | Data augmentation provider |
| `ROUTING_STRATEGY` | No | `confidence_based` | Model routing strategy |
| `USE_NEMO_GYM` | No | `false` | Enable cloud training |

### Available Models

#### Claude 4.5 Models (Teacher)
| Model | API ID | Use Case |
|-------|--------|----------|
| **Claude Opus 4.5** | `claude-opus-4-5-20251101` | Best quality |
| **Claude Sonnet 4.5** | `claude-sonnet-4-5-20250929` | Recommended |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | Fast |

#### Fine-Tuning Base Models (Student)
| Model | Parameters | Use Case |
|-------|------------|----------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | Default, fast training |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Better quality |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Alternative |

---

## Frontend Dashboard

The Bash Gym frontend is an Electron + React application with real-time WebSocket updates.

| Section | Purpose |
|---------|---------|
| **Home** | Central dashboard with flywheel visualization and system overview |
| **Workspace** | Multi-terminal grid with canvas view, file browser, and live status detection |
| **Training** | Monitor training progress, view loss curves, track system resources |
| **Models** | Browse trained models, view lineage, compare performance |
| **Factory** | Generate synthetic data, manage datasets |
| **Traces** | Explore gold/silver/bronze/failed traces, view quality metrics |
| **Orchestrator** | Submit specs, review plans, monitor parallel workers |
| **Agent** | Conversational interface to Peony assistant |
| **Router** | Monitor teacher/student routing decisions and performance |
| **Evaluator** | Run benchmarks, view evaluation results |
| **Guardrails** | Monitor safety checks, PII redaction events |
| **HuggingFace** | Cloud training, Spaces deployment, dataset management |

---

## HuggingFace Integration

Full HuggingFace ecosystem integration for cloud training via Unsloth Jobs, datasets, and deployment.

| Feature | Description |
|---------|-------------|
| **Cloud Training (Unsloth Jobs)** | Submit training jobs via `hf jobs uv run` with PEP 723 inline deps |
| **SFT / DPO / Distillation** | All three training strategies supported in cloud mode |
| **Dataset Management** | Upload and manage training datasets on HF Hub |
| **Inference API** | Use HuggingFace Inference Providers for generation and embeddings |
| **Spaces Deployment** | Deploy interactive Gradio demos |

### Hardware Tiers

| Tier | GPU | VRAM | Cost/hr | Best For |
|------|-----|------|---------|----------|
| `t4-small` | T4 | 16GB | $0.60 | Small models (<3B) |
| `a10g-small` | A10G | 24GB | $1.05 | Medium models (3-7B) |
| `a10g-large` | A10G | 24GB | $1.80 | Longer training runs |
| `a100-large` | A100 | 80GB | $4.50 | Large models (7B+) |
| `h100` | H100 | 80GB | $10.00 | Maximum performance |

All GPU tiers require a HuggingFace Pro subscription.

---

## API Reference

The API server provides 90+ REST endpoints and WebSocket real-time updates.

```bash
# Development (hot reload)
python run_backend.py

# Interactive API docs
open http://localhost:8003/docs
```

For the full endpoint reference, see **[docs/API.md](docs/API.md)**.

---

## Docker

### Full Stack (API + Peony Assistant)

```bash
# Configure secrets
cp assistant/.env.example assistant/.env
cp assistant/config/config.example.json assistant/config/config.json
# Edit both files with your tokens and API keys

# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# One-shot query via Peony
docker compose run --rm peony-agent -m "system status"
```

| Service | Image | Purpose |
|---------|-------|---------|
| `bashgym-api` | `Dockerfile.api` | FastAPI server on port 8003 with healthcheck |
| `peony-gateway` | `assistant/Dockerfile` | Long-running Discord/Telegram bot |
| `peony-agent` | `assistant/Dockerfile` | One-shot CLI queries |

### Sandbox Mode (Arena)

```bash
cd docker && docker compose build && docker compose up -d
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=bashgym --cov-report=html

# Run specific module tests
pytest tests/test_benchmarks.py -v
pytest tests/test_data_factory.py -v
```

---

## Security

| Feature | Description |
|---------|-------------|
| **Sandboxing** | All agent execution in network-isolated Docker containers |
| **Resource Limits** | Memory and CPU constraints on sandboxes |
| **Command Blocking** | Blocks rm -rf, fork bombs, disk operations |
| **API Key Protection** | Sensitive values hidden in logs and responses |
| **PII Detection** | Automatic detection and redaction across 20+ types |
| **Injection Detection** | Prompt injection attack prevention |
| **Content Moderation** | Filtering of harmful content |

---

## FAQ & Troubleshooting

### General

**Q: Do I need a GPU to use Bash Gym?**
A: No. Trace capture, curation, and the dashboard work without a GPU. Training requires a CUDA-capable GPU (8GB+ VRAM for 1.5B models, 16GB+ for 7B). Alternatively, use **HuggingFace Cloud Training** (Unsloth Jobs) to train on remote GPUs — supports T4, A10G, A100, and H100 hardware tiers starting at $0.60/hr with a HuggingFace Pro subscription.

**Q: How many traces do I need before training?**
A: You can start with as few as 20-30 gold traces for a basic SFT run. 100+ traces produce noticeably better results. The more diverse your traces (different repos, task types), the more generalizable the student model.

**Q: Does Bash Gym send my code to any external service?**
A: Only to the LLM providers you configure (Anthropic, OpenAI, etc.) — the same calls Claude Code already makes. Traces, training data, and models are stored locally. HuggingFace uploads only happen when you explicitly push.

**Q: What's the difference between a trace and a training example?**
A: A **trace** is a complete coding session (potentially many tool calls over 30+ minutes). A **training example** is a single task-response pair extracted from that trace. One trace typically produces 1-5 training examples. See the [Glossary](#glossary) for more terms.

**Q: Can I use base models other than Qwen?**
A: Yes. Set `BASE_MODEL` in your `.env` to any HuggingFace model ID. Qwen2.5-Coder and Llama 3.2 are tested. Any causal LM compatible with HuggingFace Transformers should work with LoRA/QLoRA training.

### Installation & Startup

**Q: Port 8003 is already in use.**
A: Kill existing processes:
```bash
# Windows
.\kill_api.ps1

# macOS/Linux
lsof -i :8003 | grep LISTEN
kill -9 <PID>
```

**Q: The frontend can't connect to the API.**
A: Check that the backend is running (`python run_backend.py`). The frontend expects the API at `http://localhost:8003`. If using a different port, update `frontend/.env.local`:
```
VITE_API_URL=http://localhost:YOUR_PORT/api
VITE_WS_URL=ws://localhost:YOUR_PORT/ws
```

**Q: Hooks show "Not installed" in the sidebar.**
A: Copy hooks to the correct directory:
```bash
cp bashgym/hooks/*.py ~/.claude/hooks/
```
Or use **Settings > Trace Capture** to install hooks from the UI.

**Q: `npm install` fails in the frontend directory.**
A: The frontend uses `node-pty` which requires native compilation. Ensure you have:
- Node.js 18+ (LTS recommended)
- Python 3.10+ (for node-gyp)
- On Windows: Visual Studio Build Tools with "Desktop development with C++"

### Training

**Q: Training fails with "CUDA out of memory".**
A: Reduce memory usage:
1. Use QLoRA (4-bit quantization) — enabled by default
2. Reduce batch size to 1-2
3. Reduce sequence length (default 2048, try 1024)
4. Use the smaller 1.5B model instead of 7B
5. Close other GPU-consuming applications

**Q: Training seems stuck or very slow.**
A: Check GPU utilization with `nvidia-smi -l 1`. If GPU utilization is 0%, the training process may have crashed silently. Check training logs in the Training dashboard or at `~/.bashgym/models/{run_id}/`.

**Q: Where are my trained models saved?**
A: Models are saved to `~/.bashgym/models/{run_id}/`:
- `checkpoint-*/` — intermediate checkpoints
- `final/` — final LoRA adapter
- `merged/` — full merged weights (if merge was requested)
- `model-q4_k_m.gguf` — quantized GGUF (if auto-export enabled)

### Frontend & Electron

**Q: The Electron app refreshes and kills my terminals.**
A: Ctrl+R is blocked by default. Use **Settings > Reload App** instead.

**Q: Terminal shows no activity / Claude commands aren't detected.**
A: Restart the Electron app (Settings > Reload App). Terminal status detection parses live output — if the app was updated while running, the detection patterns may be stale.

### Peony Assistant

**Q: How do I set up the Discord bot?**
A:
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application and add a Bot
3. Enable **MESSAGE_CONTENT** intent under Bot settings
4. Copy the bot token to `assistant/.env` (`DISCORD_BOT_TOKEN=...`)
5. Add your Discord user ID to the whitelist in `assistant/config/config.json`
6. Run `docker compose up`

**Q: How do I set up the Telegram bot?**
A:
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot and copy the token to `assistant/.env` (`TELEGRAM_BOT_TOKEN=...`)
3. Add your Telegram user ID to the whitelist in `assistant/config/config.json`
4. Run `docker compose up`

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, code style, and PR process.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
