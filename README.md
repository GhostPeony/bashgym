# Bash Gym

**Turn your AI coding sessions into fine-tuned models.**

<img width="1563" height="908" alt="Bash Gym home screen" src="https://github.com/user-attachments/assets/472d7ada-d57b-4e79-8604-c27e4fc99348" />

---

Your AI coding history is training data. Every session from Claude Code, Gemini CLI, OpenCode, Codex, or Copilot CLI — tool calls, file edits, bash commands, multi-step reasoning — is a structured trace of expert coding behavior. Bash Gym captures those traces, curates them into training examples, and fine-tunes models that learn from how you actually work. Train locally or via the cloud.

---

## How It Works

```
Use any AI coding tool (Claude Code, Gemini CLI, OpenCode, Codex, Copilot CLI)
        ↓
Adapters capture every session as structured traces
        ↓
Pipeline scores, classifies, and segments traces into training examples
        ↓
Fine-tune a model with LoRA (SFT, DPO, or GRPO)
        ↓
Export to LoRA adapter, merged weights, or GGUF → run via Ollama
```

| Stage | What Happens |
|-------|--------------|
| **Capture** | Adapters record every tool call, file edit, and command from your AI coding sessions (Claude Code, Gemini CLI, OpenCode, Codex, Copilot CLI) as structured JSON. Historical sessions can be bulk-imported. |
| **Curate** | Traces are scored on 6 quality metrics. Good sessions become gold training data, bad ones become negative examples for DPO. PII is scrubbed. |
| **Synthesize** | Gold traces are segmented into task-response pairs. Gaps are filled with synthetic augmentation via NVIDIA NeMo Data Designer or LLM-based generation. |
| **Train** | SFT, DPO, or GRPO fine-tunes a model using Unsloth (2–5x faster, 50–80% less VRAM). Train locally or via the cloud. |
| **Evaluate** | 11 benchmarks (HumanEval, MBPP, BigCodeBench, SWE-bench, GSM8K, and more) score the result. |
| **Route** | Confidence-based routing shifts simple tasks from Claude to your trained model over time. |

The pipeline can run automatically — file watchers monitor session directories across all configured tools and trigger import → classify → train when thresholds are met.

---

## Supported AI Coding Tools

| Tool | Live Capture | Historical Import | Hook Location |
|------|-------------|-------------------|---------------|
| **Claude Code** | Yes | Yes | `~/.claude/hooks/` |
| **Gemini CLI** | Yes | Yes | `~/.gemini/settings.json` |
| **OpenCode** | Yes | Yes | `~/.config/opencode/plugins/` |
| **Codex** | Import only | Yes | `~/.codex/` |
| **Copilot CLI** | Yes | Yes | `~/.copilot/hooks/` |

Live capture hooks run silently alongside your coding tool and record sessions in real time. Historical import harvests existing session data already on disk — no hooks needed.

The dashboard **Settings > Agents** tab provides one-click installation for all detected tools.

---

## Terminal Canvas

The workspace is an infinite canvas where terminals, browsers, and integration nodes live side by side and connect with edges to share context.

<img width="1563" height="908" alt="Terminal canvas with connected nodes" src="https://github.com/user-attachments/assets/472d7ada-d57b-4e79-8604-c27e4fc99348" />

**Terminals** are real PTY sessions (xterm.js + WebGL rendering, 10k line scrollback). Agent status is detected live from output — idle, running, tool calling, waiting for input — with CWD extraction from shell prompts. Drag files from the built-in file browser or your OS directly into a terminal to insert the path.

**Browsers** render live pages in a Chromium webview. Take full-page or element-level screenshots (crosshair picker to select a specific element) and route them to connected terminals — feed page screenshots directly to Claude as context.

**Integration nodes** plug external services into the canvas:

| Node | What It Does |
|------|--------------|
| **Context** | Freeform notes, file contents, URLs, or snippets. Reload files and fetch URLs on demand. |
| **Neon** | Connect to a Postgres database. Introspect schema, run queries, send results to terminals as markdown. |
| **Vercel** | Monitor deployments, pull build logs, generate code with v0, push previews to browser nodes. |

**Edges are context channels.** Connect any two nodes and content flows between them — schema from Neon, build logs from Vercel, screenshots from browsers, notes from context nodes — all prefilled into linked terminal inputs for review before sending.

**Shift+drag** to box-select nodes and auto-connect them in a full mesh. Three view modes: grid (all panels visible), single (one at a time), or canvas (click to open floating popups). Viewport, node positions, and all settings persist across sessions.

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

No GPU? Use [HuggingFace Cloud Training](#cloud-training) instead.

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

Auto-detect installed tools and configure adapters for all of them:

```bash
python -m bashgym.trace_capture.setup
```

Or use the dashboard **Settings > Agents** tab for one-click installation.

To bulk-import historical sessions from all detected tools:

```bash
python -m bashgym.trace_capture.setup import-all --days 60
```

Verify: launch the app and check **Settings > Agents** — each configured tool should show "Installed."

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

### 5. Use Your AI Coding Tools

That's it. Work on your projects with Claude Code, Gemini CLI, OpenCode, Copilot CLI, or any other configured tool like you always do. Every session is captured automatically. Check the **Traces** tab in the dashboard to watch them accumulate.

Once you have 20–30 gold traces, you're ready to train. See **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** for the full walkthrough from first trace to first trained model.

---

## Platform Overview

### Workspace

See [Terminal Canvas](#terminal-canvas) above for the full breakdown — real PTY terminals, live browsers with element-level screenshots, Neon/Vercel/Context integration nodes, edge-based context routing.

### Data Factory

Trace import, quality scoring, segmentation, and synthetic data generation.

- **Trace import**: Reads from all configured tool session directories — manual, via file watcher, or bulk historical import
- **Quality scoring**: 6 metrics classify traces as gold, pending, or failed
- **Segmentation**: Splits multi-task sessions into individual training examples (time gaps, git commits, directory changes)
- **Synthetic generation**: NVIDIA NeMo Data Designer v0.5.0 integration with 5 pipeline types (`coding_agent_sft`, `coding_agent_dpo`, `tool_use_sft`, `from_external`, `from_unstructured`) plus LLM-based augmentation via Anthropic or NVIDIA NIM
- **PII scrubbing**: Automatic redaction before training

### Training

See [Training](#training) for details.

- **Strategies**: SFT, DPO, GRPO, RLVR, Distillation
- **Acceleration**: Unsloth with QLoRA (4-bit quantization) by default
- **Compute**: Local GPU or HuggingFace cloud
- **Output**: LoRA adapter, merged weights (16-bit), GGUF (for Ollama/llama.cpp/LM Studio)

### Orchestrator

Multi-agent task decomposition across LLM providers with git worktree isolation.

| Provider | Default Model |
|----------|---------------|
| **Anthropic** | `claude-opus-4-6` |
| **OpenAI** | `gpt-4o` |
| **Gemini** | `gemini-2.5-pro` |
| **Ollama** | `qwen2.5-coder:32b` |

Four phases: PLAN → DISPATCH → MONITOR → SYNTHESIZE. Workers always use Claude Code CLI regardless of the planning provider.

### Evaluation

11 benchmarks for scoring trained models:

| Benchmark | Focus |
|-----------|-------|
| HumanEval | Function-level code generation (164 problems) |
| MBPP | Mostly basic programming problems (974 problems) |
| BigCodeBench | Complex, multi-library coding tasks |
| SWE-bench | Real GitHub issue resolution |
| DS-1000 | Data science problems |
| BFCL | Function calling accuracy |
| GSM8K | Math reasoning |
| ARC | Abstract reasoning |
| HellaSwag | Commonsense inference |
| ToxiGen | Toxicity detection |
| BBQ | Bias measurement |

Plus LLM-as-Judge scoring (correctness, helpfulness, safety) and RAG metrics (faithfulness, answer relevancy, context precision) via NeMo Evaluator integration.

### Pipeline

Automated loop: watch → import → classify → train.

- **Watcher**: `watchdog`-based file observer on configured tool session directories with debouncing
- **Quality gate**: Configurable thresholds for auto-classification
- **Threshold monitor**: Triggers training when enough gold traces accumulate
- Can run fully automated or step-by-step from the dashboard

### Agent (Peony)

Built-in assistant with tool use and system context. Available in-app and as a standalone bot.

- **In-app**: Chat panel in the dashboard — can import traces, trigger training, search HuggingFace, run commands
- **Multi-platform**: Go binary (`picoclaw`) with adapters for Discord, Telegram, Slack, WhatsApp, DingTalk, LINE, Feishu, QQ, and OneBot

### Safety

Guardrails pipeline with 5 check types applied to inputs and outputs:

| Check | Description |
|-------|-------------|
| **Injection detection** | Regex patterns for prompt injection attempts |
| **Content moderation** | Blocks harmful or inappropriate content |
| **Topic control** | Constrains responses to allowed domains |
| **Code safety** | Blocks dangerous commands (`rm -rf /`, fork bombs, `curl \| sh`, privilege escalation) |
| **PII filter** | Opt-in detection and redaction of personal information |

### Observability

Profiler with span-level tracing for agent workflows. Tracks token usage, latency, and guardrail performance. Backend support for Phoenix, Langfuse, and OpenTelemetry.

### Achievements

Gamified progress tracking across trace collection, quality, training, and mastery categories. 5 rarity tiers (common through legendary) with point scoring.

---

## Training

### Strategies

| Strategy | What It Does | When To Use |
|----------|--------------|-------------|
| **SFT** | Trains the model to reproduce successful traces | Start here. Works with 20–30 gold traces. |
| **DPO** | Learns from pairs of good and bad responses | When you have both gold and failed traces. |
| **GRPO** | RL-based — model generates solutions, learns from verifiable rewards | Advanced. Needs more data. |
| **RLVR** | Reinforcement learning with verifiable rewards | Alternative RL approach. |
| **Distillation** | Transfers knowledge from a larger model to a smaller one | When you want a compact model that mimics a larger teacher. |

### Base Models

Tested and supported base models:

| Model | Parameters | VRAM |
|-------|------------|------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | 8GB (default) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | 16GB |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 12GB |

All training uses QLoRA (4-bit quantization) by default.

### Output Formats

| Format | Location | Use Case |
|--------|----------|----------|
| **LoRA adapter** | `lora_adapters/` | ~50MB, composable, hot-swappable |
| **Merged weights** | `merged/` | Full 16-bit model, ready for inference |
| **GGUF** | `exported_gguf/` | Quantized (default `q4_k_m`), for Ollama / llama.cpp / LM Studio / GPT4All |

### Cloud Training

No local GPU? Use HuggingFace Cloud Training (Unsloth Jobs). Multiple GPU tiers available from T4 to H100. Requires HuggingFace Pro subscription.

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `OPENAI_API_KEY` | No | — | OpenAI API key (for Codex trace capture and orchestrator routing) |
| `GOOGLE_API_KEY` | No | — | Google/Gemini API key (for Gemini CLI trace capture and orchestrator routing) |
| `BASE_MODEL` | No | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Fine-tuning base model |
| `HF_TOKEN` | No | — | HuggingFace token (for cloud training and model push) |
| `NVIDIA_API_KEY` | No | — | NVIDIA NIM API key (for synthetic data augmentation) |
| `ROUTING_STRATEGY` | No | `confidence_based` | Model routing strategy |
| `AUGMENTATION_PROVIDER` | No | `anthropic` | Synthetic data provider: `anthropic` or `nim` |

API keys can also be managed through the dashboard at **Settings > API Keys**, which provides secure storage via the Electron keychain.

---

## Project Structure

```
bashgym/
├── bashgym/                  # Python package
│   ├── arena/                # Execution (runner, sandbox)
│   ├── judge/                # Verification (evaluator, guardrails, benchmarks)
│   ├── factory/              # Data synthesis (trace processor, example generator, NeMo Data Designer)
│   ├── gym/                  # Training (trainer, environment, router)
│   ├── hooks/                # Legacy Claude Code hooks (kept for compatibility)
│   ├── trace_capture/        # Multi-agent trace capture
│   │   ├── adapters/         # Tool-specific hooks (Claude Code, Gemini, OpenCode, Codex, Copilot)
│   │   ├── importers/        # Historical session importers per tool
│   │   ├── detector.py       # Auto-detection of installed AI coding tools
│   │   └── setup.py          # CLI for hook installation and bulk import
│   ├── models/               # Model registry and lifecycle
│   ├── orchestrator/         # Multi-agent task decomposition (agent, DAG, dispatcher, worktree)
│   ├── pipeline/             # Automated watcher, quality gate, orchestrator
│   ├── achievements/         # Progress tracking and gamification
│   ├── observability/        # Profiler, span tracing, backend integrations
│   ├── api/                  # REST API + WebSocket
│   └── integrations/         # HuggingFace, NeMo, Ollama
│
├── frontend/                 # Electron + React dashboard
│   ├── src/components/       # 72+ React components
│   ├── src/stores/           # Zustand state management
│   └── electron/             # Main process + secure storage
│
├── assistant/                # Peony chat assistant (Go + Docker)
│   └── picoclaw/             # Multi-platform bot (Discord, Telegram, Slack, etc.)
│
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
A trace is a complete coding session from any supported tool (many tool calls, potentially 30+ minutes). A training example is a single task-response pair extracted from that trace. One trace typically produces 1–5 examples.

**Can I use other base models?**
Yes. Set `BASE_MODEL` in `.env` to a different model ID. Larger models need more VRAM (or use cloud training).

**What about synthetic data?**
The data factory supports NVIDIA NeMo Data Designer for structured synthetic generation, plus LLM-based augmentation using Anthropic or NVIDIA NIM models. Useful for filling gaps in your trace coverage.

---

## License

MIT — see [LICENSE](LICENSE).
