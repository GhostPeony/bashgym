# Bash Gym

**Turn your AI coding sessions into fine-tuned models.**


<img width="1745" height="906" alt="autobashgym" src="https://github.com/user-attachments/assets/3ba9430e-e910-4d74-8860-d9e4c88161d3" />


---

Every AI coding session is a chain-of-thought reasoning trace — step-by-step problem solving with verifiable outcomes. Every session from Claude Code, Gemini CLI, OpenCode, Codex, or Copilot CLI — tool calls, file edits, bash commands, multi-step reasoning — is a structured trace of expert coding behavior. Bash Gym captures these traces and uses them to train a reasoning language model with the same techniques behind frontier RLMs: GRPO for reinforcement learning, RLVR for verifiable reward signals from test results, and distillation to transfer reasoning from a large teacher into a small local model. The result is a personal RLM trained on how you actually think through code — your conventions, your repos, your patterns.

---

## How It Works

```
Use any AI coding tool (Claude Code, Gemini CLI, OpenCode, Codex, Copilot CLI)
        ↓
Adapters capture every session as structured traces
        ↓
Pipeline scores, classifies, and segments traces into training examples
        ↓
Fine-tune with SFT, DPO, GRPO, RLVR, or Distillation (Unsloth + QLoRA)
        ↓
Export to LoRA adapter, merged weights, or GGUF → run via Ollama
```

| Stage | What Happens |
|-------|--------------|
| **Capture** | Adapters record every tool call, file edit, and command from your AI coding sessions (Claude Code, Gemini CLI, OpenCode, Codex, Copilot CLI) as structured JSON. Historical sessions can be bulk-imported. |
| **Curate** | Traces are scored on 6 quality metrics. Good sessions become gold training data, bad ones become negative examples for DPO. PII is scrubbed. |
| **Synthesize** | Gold traces are segmented into task-response pairs. Gaps are filled with synthetic augmentation via NVIDIA NeMo Data Designer or LLM-based generation. |
| **Train** | SFT, DPO, GRPO, RLVR, or Distillation fine-tunes a model using Unsloth (2–5x faster, 50–80% less VRAM). Train locally, on a remote GPU over SSH, or via HuggingFace cloud. |
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

<img width="1744" height="915" alt="bgcanvas" src="https://github.com/user-attachments/assets/d30355b8-b64e-4122-902e-6f496b7be33a" />

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
- **Decision extraction**: Structured decision logging captures WHY an agent chose each action — intent, alternatives considered, reasoning, and outcome. Enables step-level DPO pair generation rather than whole-trace pairing.
- **Segmentation**: Splits multi-task sessions into individual training examples (time gaps, git commits, directory changes)
- **Synthetic generation**: NVIDIA NeMo Data Designer integration with 5 pipeline types (`coding_agent_sft`, `coding_agent_dpo`, `tool_use_sft`, `from_external`, `from_unstructured`) plus LLM-based augmentation via Anthropic or NVIDIA NIM. Multi-provider support assigns different models per column (e.g., NIM for code generation, Anthropic for judge scoring).
- **Schema evolution**: AutoResearch evolves Data Designer pipeline configs — temperatures, judge rubrics, column toggles, provider assignments — finding schemas that produce better training data than hand-tuned presets
- **Embedding dedup**: Semantic deduplication via NIM API embeddings removes near-duplicate examples with configurable similarity threshold and diversity scoring
- **PII scrubbing**: Automatic redaction before training

### Semantic Judge

LLM-based quality evaluation that goes beyond test pass/fail. Evaluates traces on four axes:

- **Solution approach** — Is it idiomatic? Minimal? Over-engineered?
- **Decision quality** — Were good trade-offs made? Did the agent recover well from errors?
- **Code quality** — Is the written code clean and maintainable?
- **Task alignment** — Does the solution match the original intent?

Uses Claude Haiku for cost efficiency (~$2-5 per 1,000 traces). Integrated into the quality gate: traces that pass tests but score low on quality are demoted from gold to pending, keeping training data clean. Fully optional — disable it and the pipeline works as before.

### Training

See [Training](#training) for details.

- **Strategies**: SFT, DPO, GRPO, RLVR, Distillation, Cascade RL
- **Cascade RL**: Sequential domain-by-domain training (file ops → bash → search → multi-step reasoning) with per-domain reward functions, checkpoint chaining, and MOPD distillation to merge domain experts
- **Acceleration**: Unsloth with QLoRA (4-bit quantization) by default
- **Providers**: Pluggable inference via Anthropic, NVIDIA NIM, and Ollama. Ollama models are auto-discovered at startup — any model you've pulled is immediately available as a Student model
- **Compute**: Local GPU, remote SSH (e.g. DGX Spark), or HuggingFace cloud
- **Output**: LoRA adapter, merged weights (16-bit), GGUF (for Ollama/llama.cpp/LM Studio)
- **Training goals**: Define weighted success criteria and hard/soft constraints instead of optimizing a single loss scalar. The outcome aggregator tracks progress and recommends when to stop, adjust, or continue.
- **AutoResearch**: Three evolutionary search modes (hyperparameters, trace curation, schema evolution) — see [AutoResearch](#autoresearch) for the full breakdown

### Dual-Loop Evolution

Two feedback loops that improve agent behavior at different speeds:

| Loop | Mechanism | Latency | Effect |
|------|-----------|---------|--------|
| **Fast loop** | Prompt evolution — analyze failure patterns, mutate worker prompts, evaluate, deploy best variant | Minutes | Immediate behavioral change |
| **Slow loop** | Weight training — accumulate traces, generate examples, fine-tune model | Hours | Deep, permanent learning |

The fast loop extracts recurring failure patterns from decision logs (wrong tool choices, missing context, anti-patterns) and uses an LLM to generate targeted prompt patches. Variants are scored against held-out traces and kept only if they improve quality. The best variant feeds into the orchestrator's worker prompts automatically.

The slow loop continues as before: gold traces become training examples, the model gets fine-tuned. But now gold traces are generated by workers using evolved prompts, so training data quality improves from both loops.

### Orchestrator

Multi-agent task decomposition across LLM providers with git worktree isolation.

| Provider | Default Model |
|----------|---------------|
| **Anthropic** | `claude-opus-4-6` |
| **OpenAI** | `gpt-4o` |
| **Gemini** | `gemini-2.5-pro` |
| **Ollama** | `qwen2.5-coder:32b` |

Four phases: PLAN → DISPATCH → MONITOR → SYNTHESIZE. Workers always use Claude Code CLI regardless of the planning provider.

**Three-layer prompt composition** structures worker context into Identity (static role/standards), Narrative (dynamic sibling progress, shared discoveries, file ownership), and Focus (task-specific requirements). Transition markers update running workers when siblings complete.

**Shared memory** provides per-key locked state that parallel workers can read and write. Workers share discoveries (e.g., "the API uses Bearer auth") via a file-based IPC protocol. The orchestrator polls worktree state files and merges entries into a central store with conflict detection.

### Event System

Typed event bus with 37+ event types covering pipeline, training, orchestration, judge, and system domains. All components emit structured events that are bridged to the WebSocket layer for real-time frontend updates. Supports both sync and async handlers with type-based dispatch.

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
| **SFT** | Supervised fine-tuning — trains the model to reproduce successful traces | Start here. Works with 20–30 gold traces. |
| **DPO** | Direct Preference Optimization — learns from pairs of good and bad responses | When you have both gold and failed traces. |
| **GRPO** | Group Relative Policy Optimization via `trl.GRPOTrainer`. Three tiered reward functions: syntax (`ast.parse`), execution (`subprocess`), and verification (`pytest`). Model generates multiple completions per prompt and learns from the reward signal. | When you want RL-based training. Needs test cases for verification mode. |
| **RLVR** | RL with Verifiable Rewards — GRPO with verification-locked rewards. Test results from `pytest` are the reward signal: pass rate becomes the score. | When your traces include test code. The strongest signal for code correctness. |
| **Distillation** | Knowledge distillation from a large teacher (e.g. Claude) into a small student. Combines soft labels (KL divergence) and hard labels (cross-entropy) weighted by alpha. Supports offline and on-policy modes. | When you want a compact model that reasons like a larger one. |
| **Cascade RL** | Sequential domain-by-domain GRPO training inspired by Nemotron Cascade 2. Trains each coding domain independently with tailored reward functions, then merges domain experts via MOPD distillation. | When you want domain-specialized training. Best results with diverse traces across file editing, bash, search, and multi-step tasks. |

### Base Models

Any HuggingFace model compatible with Unsloth works. Set `BASE_MODEL` in the dashboard or `.env` to any model ID. Ollama models are auto-discovered for inference — train on HuggingFace weights, deploy via Ollama.

**Recommended starting points:**

| Model | Parameters | VRAM | Notes |
|-------|------------|------|-------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | 8GB | Default. Fast training, good for iteration. |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | 16GB | Better quality, needs more VRAM. |
| `Qwen/Qwen3.5-4B` | 4B | 10GB | Newest Qwen (Feb 2026). Strong reasoning. |
| `Qwen/Qwen3.5-9B` | 9B | 18GB | Best Qwen dense model for coding. |
| `nvidia/Nemotron-Cascade-2-30B-A3B` | 30B (3B active) | 20GB | MoE. Ideal for Cascade RL on DGX Spark. |
| `nvidia/Nemotron-3-Nano-4B-Instruct` | 4B | 10GB | NVIDIA's compact coding model. |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 12GB | Alternative architecture. |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 2.4B | 10GB | Strong code performance for size. |

These are suggestions, not restrictions. Any `AutoModelForCausalLM`-compatible model from HuggingFace will work — including Mistral, StarCoder, CodeGemma, Yi, Phi, etc. All training uses QLoRA (4-bit quantization) by default, so VRAM requirements are roughly `model_params / 2` GB.

### Output Formats

| Format | Location | Use Case |
|--------|----------|----------|
| **LoRA adapter** | `lora_adapters/` | ~50MB, composable, hot-swappable |
| **Merged weights** | `merged/` | Full 16-bit model, ready for inference |
| **GGUF** | `exported_gguf/` | Quantized (default `q4_k_m`), for Ollama / llama.cpp / LM Studio / GPT4All |

### Remote SSH Training

Train on a remote machine (e.g. NVIDIA DGX Spark) over SSH. The dashboard uploads the training script, streams logs in real time, and supports pause/resume/cancel. All five training strategies (SFT, DPO, GRPO, RLVR, Distillation) generate strategy-specific scripts that run on the remote host. The dashboard shows connection status and a pre-flight check with GPU detection before each run.

### Device Management

Plug-and-play SSH device registry for remote training targets. No manual `.env` editing required.

- **Auto-discovery**: Parses `~/.ssh/config` to find candidate devices (filters out GitHub, GitLab, and other code forges automatically)
- **Dashboard UI**: DeviceManager panel for adding, editing, removing, and connection-testing devices
- **GPU detection**: Pre-flight checks report GPU model, VRAM, CUDA version, OS, and available disk space
- **Persistent registry**: Devices are stored in `~/.bashgym/devices.json` and survive restarts
- **Environment import**: On first startup, existing `SSH_REMOTE_*` environment variables are auto-imported as the default device

### AutoResearch

Three evolutionary search loops that continuously improve your training pipeline. Each runs independently, keeps improvements, and converges on optimal configurations — inspired by Karpathy's autoresearch.

| Mode | What It Evolves | What It Searches | How It Evaluates |
|------|----------------|-----------------|-----------------|
| **Hyperparameter search** | Training configuration | Learning rate, LoRA rank/alpha/dropout, batch size, sequence length, quantization, warmup ratio | Short training runs (50-100 steps) measuring validation loss |
| **Trace research** | Data curation strategy | Quality thresholds, segmentation boundaries, cognitive tag inclusion, silver trace ratio, dedup threshold, per-repo caps | End-to-end: filter traces → generate examples → micro-train → measure loss |
| **Schema evolution** | Data Designer pipeline configs | Temperatures, column toggles, judge rubrics, provider assignments, code validation, embedding dedup | Two-stage: judge scores filter bad candidates fast (25 examples), then micro-train validates top 5 |

**Schema evolution** is the newest mode — the AutoCurriculum Compiler. Instead of tuning hyperparameters or data curation rules, it evolves the entire data generation pipeline. A `SchemaSearchSpace` mutates Data Designer configs (which models generate code vs judge quality, what temperature, how many judge dimensions, whether to include code validation), evaluates each mutant by generating real training data and measuring downstream training loss, and keeps winners. The template library maps failure patterns from your traces to starting templates — if your model keeps picking the wrong tool, the schema evolves toward tool-use-focused training data.

All three modes share the same evolutionary engine (`SearchSpace` ABC → `AutoResearcher` loop) and UI (start/stop/pause/resume, real-time experiment streaming via WebSocket, loss curves, generation cards with mutation diffs).

**Embedding-based dedup** runs across all modes via NIM API: computes semantic similarity between training examples and removes near-duplicates (configurable threshold, default 0.95). Diversity scores are tracked in the quality dashboard.

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
| `BASE_MODEL` | No | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Any HuggingFace model ID for fine-tuning |
| `HF_TOKEN` | No | — | HuggingFace token (for cloud training and model push) |
| `NVIDIA_API_KEY` | No | — | NVIDIA NIM API key (for synthetic data augmentation) |
| `ROUTING_STRATEGY` | No | `confidence_based` | Model routing strategy |
| `AUGMENTATION_PROVIDER` | No | `anthropic` | Synthetic data provider: `anthropic` or `nim` |
| `OLLAMA_ENABLED` | No | `true` | Enable Ollama local inference provider |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `SSH_REMOTE_HOST` | No | — | Remote training host (e.g. DGX Spark IP) |
| `SSH_REMOTE_USER` | No | — | SSH username for remote training |
| `SSH_REMOTE_KEY_PATH` | No | `~/.ssh/id_rsa` | Path to SSH private key |

API keys can also be managed through the dashboard at **Settings > API Keys**, which provides secure storage via the Electron keychain.

---

## Project Structure

```
bashgym/
├── bashgym/                  # Python package
│   ├── api/                  # REST API + WebSocket
│   ├── arena/                # Execution (runner, sandbox)
│   ├── events/               # Typed EventBus (bus, event types, WebSocket bridge)
│   ├── factory/              # Data synthesis (trace processor, example generator, decision extractor)
│   ├── gym/                  # Training (trainer, autoresearch, cascade, training goals)
│   │   ├── trainer.py        # SFT, DPO, GRPO, RLVR, Distillation
│   │   ├── autoresearch.py   # SearchSpace ABC + evolutionary hyperparameter search
│   │   ├── schema_search_space.py # Schema evolution for Data Designer configs
│   │   ├── cascade_scheduler.py # Cascade RL + MOPD distillation
│   │   ├── trace_researcher.py # Data curation optimization
│   │   └── remote_trainer.py # SSH-based remote training
│   ├── judge/                # Verification (evaluator, semantic judge, guardrails, benchmarks)
│   ├── models/               # Model registry and lifecycle
│   ├── orchestrator/         # Multi-agent decomposition (agent, DAG, shared state, context builder)
│   ├── pipeline/             # Automated watcher, quality gate, semantic evaluation
│   ├── providers/            # Inference providers (Anthropic, NIM, Ollama)
│   ├── device_registry.py    # JSON-backed SSH device storage
│   ├── device_discovery.py   # ~/.ssh/config parser for device auto-discovery
│   ├── trace_capture/        # Multi-agent trace capture
│   │   ├── adapters/         # Tool-specific hooks (Claude Code, Gemini, OpenCode, Codex, Copilot)
│   │   ├── importers/        # Historical session importers per tool
│   │   ├── detector.py       # Auto-detection of installed AI coding tools
│   │   └── setup.py          # CLI for hook installation and bulk import
│   ├── achievements/         # Progress tracking and gamification
│   ├── observability/        # Profiler, span tracing, backend integrations
│   ├── api/                  # REST API + WebSocket
│   │   ├── device_routes.py  # Device management endpoints
│   │   ├── autoresearch_routes.py # AutoResearch + schema research endpoints
│   │   └── cascade_routes.py # Cascade RL + MOPD endpoints
│   └── integrations/         # HuggingFace, NeMo, Ollama
│
├── frontend/                 # Electron + React dashboard
│   ├── src/components/       # 72+ React components
│   │   └── training/         # DeviceManager, AutoResearchPanel, TrainingConfig, etc.
│   ├── src/stores/           # Zustand state management
│   └── electron/             # Main process + secure storage
│
├── assistant/                # Peony chat assistant (Go + Docker)
│   └── picoclaw/             # Multi-platform bot (Discord, Telegram, Slack, etc.)
│
├── tests/                    # Test suite
├── run_backend.py            # Backend entry point (uvicorn)
├── dev.ps1 / dev.sh          # Dev environment launchers
└── docker-compose.yml        # Full stack deployment
```

---

## Roadmap

See [TODOS.md](TODOS.md) for the full roadmap with details.

**Recently shipped:**
- **AutoCurriculum Compiler** — Data Designer schemas evolve via evolutionary search (SchemaResearcher). Two-stage evaluation: judge scores filter bad candidates fast, micro-training validates winners. Template library auto-selects pipelines from failure analysis.
- **Cascade RL** — Sequential domain-by-domain GRPO training inspired by Nemotron Cascade 2. Four coding domains (file ops, bash, search, multi-step reasoning) each get tailored reward functions. MOPD distillation merges domain experts into one model.

**Up next:**
- **Black-Box On-Policy Distillation** — Real-time teacher inference (Claude/NIM) during training, targeting the student's actual weaknesses instead of using stale pre-generated outputs
- **Schema Sharing** — Export/import evolved schemas with provenance so others can skip the search

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
Yes — any HuggingFace model that works with Unsloth/transformers. Qwen, Llama, Mistral, DeepSeek, Phi, StarCoder, CodeGemma, Yi, and more. Set `BASE_MODEL` in the dashboard or `.env`. Larger models need more VRAM (or use cloud training). After training, export to GGUF and run via Ollama, llama.cpp, LM Studio, or any GGUF-compatible runtime.

**What about synthetic data?**
The data factory supports NVIDIA NeMo Data Designer for structured synthetic generation, plus LLM-based augmentation using Anthropic or NVIDIA NIM models. Useful for filling gaps in your trace coverage.

---

## License

MIT — see [LICENSE](LICENSE).
