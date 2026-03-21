# Bash Gym - Agent Documentation

> A Self-Improving Agentic Development Gym

This document provides comprehensive guidance for AI agents working with this codebase.

## Quick Reference

```bash
# Start dev environment (backend + frontend)
.\dev.ps1                      # Both services
.\dev.ps1 -BackendOnly         # API only
.\dev.ps1 -FrontendOnly        # Frontend only
.\dev.ps1 -Electron            # Backend + Electron app

# Or start manually in separate terminals:
python run_backend.py           # API on port 8003 (hot reload)
cd frontend && npm run dev      # Vite on port 5173

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Run tests
pytest tests/ -v
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE OUROBOROS FLYWHEEL                       │
│    ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌─────────┐   │
│    │   ACT   │───▶│ VERIFY  │───▶│ SYNTHESIZE│───▶│  TRAIN  │   │
│    │ (Arena) │    │ (Judge) │    │ (Factory) │    │  (Gym)  │   │
│    └─────────┘    └─────────┘    └───────────┘    └─────────┘   │
│         ▲                                              │         │
│         └──────────────── DEPLOY ◀─────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Layers

| Layer | Files | Purpose |
|-------|-------|---------|
| **Arena** | `sandbox.py`, `agent_runner.py` | Docker sandbox execution, Claude CLI wrapper |
| **Judge** | `verifier.py`, `verify.sh` | Test execution, solution validation |
| **Factory** | `data_factory.py`, `trace_processor.py`, `example_generator.py` | Training data synthesis from traces |
| **Gym** | `trainer.py`, `gym_env.py`, `model_router.py` | SFT/DPO/GRPO training, RL environment |
| **Config** | `settings.py` | Centralized configuration |
| **Hooks** | `post_tool_use.py`, `session_end.py` | Claude Code instrumentation |

---

## File Reference

### Core Modules

#### `sandbox.py` - Docker Sandbox Manager
Manages isolated Docker containers for safe code execution.

Key classes:
- `SandboxConfig` - Container configuration (memory, CPU, network)
- `SandboxInstance` - Represents a running container
- `SandboxManager` - Creates/manages sandboxes

Important methods:
- `create_sandbox()` - Create new isolated environment
- `execute_command()` - Run command in sandbox
- `_is_dangerous_command()` - Safety check for harmful commands

#### `agent_runner.py` - Claude CLI Wrapper
Runs Claude Code CLI within sandboxed environments.

Key classes:
- `AgentConfig` - Claude CLI configuration
- `TaskResult` - Execution result dataclass
- `AgentRunner` - Main runner class

#### `verifier.py` - Test Execution Engine
Runs verification tests to determine solution correctness.

Key classes:
- `VerificationConfig` - Test configuration
- `VerificationResult` - Test results with pass/fail counts
- `Verifier` - Test runner

Supports: pytest, bats, npm test, Makefile, custom verify.sh

#### `trainer.py` - Model Fine-Tuning
Handles SFT, DPO, and GRPO training strategies.

Key classes:
- `TrainerConfig` - Training hyperparameters
- `TrainingRun` - Training run state
- `Trainer` - Main SFT/DPO trainer
- `GRPOTrainer` - GRPO-specific trainer

#### `gym_env.py` - RL Environment
Gymnasium-compatible environment for RL training.

Key classes:
- `ActionType` - BASH, READ, WRITE, EDIT, SUBMIT
- `Action` - Agent action dataclass
- `Observation` - Environment observation
- `BashGymEnv` - Main RL environment
- `BatchGymEnv` - Parallel environment wrapper

#### `model_router.py` - Teacher/Student Routing
Routes requests between Teacher (Claude) and Student (fine-tuned) models.

Key classes:
- `RoutingStrategy` - TEACHER_ONLY, STUDENT_ONLY, CONFIDENCE_BASED, etc.
- `ModelRouter` - Main routing logic (delegates to ProviderRegistry when available)

#### `settings.py` - Configuration Management
Centralized settings loaded from environment variables.

```python
from settings import get_settings
settings = get_settings()
print(settings.api.anthropic_api_key)
print(settings.training.base_model)
print(settings.ollama.base_url)
```

### Provider Abstraction Layer

The model router delegates inference to pluggable providers via a `ProviderRegistry`.

| Provider | Class | Local? | API Key | Use Case |
|----------|-------|--------|---------|----------|
| Anthropic | `AnthropicProvider` | No | Yes | Teacher (Claude) |
| NVIDIA NIM | `NIMProvider` | No | Yes | Cloud student inference |
| Ollama | `OllamaProvider` | Yes | No | Local student inference (DGX Spark) |

**Key files:**
- `bashgym/providers/base.py` — `InferenceProvider` ABC, `ProviderResponse`, `HealthStatus`
- `bashgym/providers/registry.py` — `ProviderRegistry` (model↔provider mapping, health monitoring)
- `bashgym/providers/anthropic.py` — Claude API provider
- `bashgym/providers/nim.py` — NVIDIA NIM provider
- `bashgym/providers/ollama.py` — Ollama local provider (with warm-up, VRAM tracking, network security)

**API endpoints:**
- `GET /api/providers/health` — Health status of all providers
- `POST /api/router/student-provider?provider_type=ollama&model_name=qwen2.5-coder:7b` — Set student model
- `GET /api/router/config` — Full router config with active Teacher/Student models
- `POST /api/providers/ollama/warmup?model_name=...` — Pre-load model into VRAM

**Local inference loop (DGX Spark):**
```
Train → GGUF export → Deploy to Ollama → Set as Student → Router sends inference locally → Collect traces → Train again
```

### Remote Training (DGX Spark)

Remote SSH training enables executing training runs on a DGX Spark via SSH, streaming logs back to the dashboard.

**Key files:**
- `bashgym/gym/remote_trainer.py` — SSH-based remote training execution
- `bashgym/config.py` — `SSHSettings` reads `SSH_REMOTE_*` env vars

**Flow:** Generate script locally → SFTP upload → SSH exec with nohup → stream logs → SFTP download artifacts

**API:**
- `GET /api/ssh/preflight` — verify remote machine is ready
- `POST /api/training/start` with `use_remote_ssh: true` — start remote training

**UI:** "DGX Spark" backend option in Training Config, SSH status in SystemInfoPanel

**Process control:** pause/resume/cancel via SSH signals (SIGSTOP/SIGCONT/SIGTERM)

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_REMOTE_ENABLED` | `false` | Enable remote training |
| `SSH_REMOTE_HOST` | | DGX Spark IP/hostname |
| `SSH_REMOTE_USER` | | SSH username |
| `SSH_REMOTE_PORT` | `22` | SSH port |
| `SSH_REMOTE_KEY_PATH` | `~/.ssh/id_rsa` | Path to SSH private key |
| `SSH_REMOTE_WORK_DIR` | `~/bashgym-training` | Remote working directory |

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `NVIDIA_API_KEY` | No | NVIDIA NIM API key |
| `BASE_MODEL` | No | Fine-tuning base model (default: Qwen2.5-Coder-1.5B) |
| `USE_NEMO_GYM` | No | Enable cloud training (default: false) |
| `OLLAMA_ENABLED` | No | Enable Ollama local inference (default: true) |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | No | Default Ollama model (empty = auto-detect) |
| `AUGMENTATION_PROVIDER` | No | Data augmentation provider: `anthropic` or `nim` (default: anthropic) |

See `.env.example` for complete list.

### Available Models

#### Anthropic Claude 4.5 Models (Current)

| Model | API ID | Use Case | Pricing |
|-------|--------|----------|---------|
| **Claude Opus 4.5** | `claude-opus-4-5-20251101` | Best quality, premium tasks | $5/$25 per MTok |
| **Claude Sonnet 4.5** | `claude-sonnet-4-5-20250929` | Recommended for most tasks | $3/$15 per MTok |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | Fast, cost-effective | $1/$5 per MTok |

#### Anthropic Claude 4 Models (Legacy)

| Model | API ID |
|-------|--------|
| Claude Sonnet 4 | `claude-sonnet-4-20250514` |
| Claude Opus 4 | `claude-opus-4-20250514` |
| Claude Opus 4.1 | `claude-opus-4-1-20250805` |

#### Data Augmentation (Synthetic Data Generation)

| Provider | Model | Use Case |
|----------|-------|----------|
| **Anthropic** | `claude-sonnet-4-5-20250929` | High-quality augmentation (recommended) |
| **Anthropic** | `claude-haiku-4-5-20251001` | Faster, lower cost |
| **NVIDIA NIM** | `qwen/qwen2.5-coder-32b-instruct` | Cost-effective code augmentation |
| **NVIDIA NIM** | `qwen/qwen2.5-coder-7b-instruct` | Fastest, lowest cost |

#### Fine-Tuning Base Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | Default, fast training |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Better quality, needs more VRAM |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Alternative base model |

#### NVIDIA NIM Available Models (183 total)

Key coding models available via `NIM_ENDPOINT`:
- `qwen/qwen2.5-coder-32b-instruct` - Best Qwen coder
- `qwen/qwen2.5-coder-7b-instruct` - Fast Qwen coder
- `meta/codellama-70b` - Meta's CodeLlama
- `mistralai/codestral-22b-instruct-v0.1` - Mistral coder
- `deepseek-ai/deepseek-coder-6.7b-instruct` - DeepSeek coder
- `bigcode/starcoder2-15b` - StarCoder 2

---

## Testing

Tests are organized under the `tests/` directory.

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/gym/ -v
pytest tests/api/ -v
pytest tests/factory/ -v

# With coverage
pytest tests/ --cov=bashgym --cov-report=html
```

---

## Common Tasks

### Adding a New Training Strategy

1. Add strategy to `TrainingStrategy` enum in `trainer.py`
2. Implement `train_{strategy}()` method in `Trainer` class
3. Add script generation in `_generate_{strategy}_script()`
4. Add tests in `tests/gym/`

### Modifying Sandbox Security

Edit `SandboxManager._is_dangerous_command()` in `sandbox.py`.

Current blocked patterns:
- `rm -rf /`
- `:(){:|:&};:` (fork bomb)
- `/dev/sda`
- Privilege escalation (`sudo`, `chmod 777`)

### Adding New Verification Methods

1. Add detection in `Verifier._find_*` methods
2. Add execution in `Verifier._run_*` methods
3. Update `VERIFY_SCRIPT_TEMPLATE` if needed

---

## Docker Setup

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f bashgym-api
```

---

## Troubleshooting

### Import Errors

The codebase supports both standalone and package usage:
```python
try:
    from sandbox import SandboxManager
except ImportError:
    from .sandbox import SandboxManager
```

### Windows Compatibility

`post_tool_use.py` uses platform-specific file locking:
- Windows: `msvcrt`
- Unix: `fcntl`

### Docker Not Available

Set `use_sandbox=False` in `GymEnvConfig` to run in simulation mode.

### API Key Issues

Ensure your `.env` file exists and contains valid keys:
```bash
python -c "from settings import get_settings; get_settings().api.validate()"
```

---

## Code Style

- Python 3.10+ with type hints
- Dataclasses for configuration/results
- Async where beneficial (httpx, training)
- 100 char line length (black/ruff)

---

## Directory Structure

```
bashgym/                 # Core Python package
├── api/                 # FastAPI routes, schemas, WebSocket
├── arena/               # Docker sandbox, Claude CLI wrapper
├── judge/               # Verification and evaluation
├── factory/             # Training data synthesis
├── gym/                 # Training loop, autoresearch, model router
├── providers/           # Inference providers (Anthropic, NIM, Ollama)
├── trace_capture/       # Import traces from Claude, Gemini, Copilot
├── orchestrator/        # Task decomposition and multi-agent dispatch
├── pipeline/            # Pipeline config and threshold monitoring
├── integrations/        # HuggingFace, NeMo
├── agent/               # Memory, tools, skills
└── config.py            # Settings management
frontend/                # React + Vite + Electron UI
tests/                   # Test suite
assistant/               # Peony assistant gateway
run_backend.py           # Backend entry point (uvicorn)
dev.ps1 / dev.sh         # Dev environment launchers
requirements.txt         # Core dependencies
requirements-training.txt # ML dependencies
Dockerfile.api           # API container
Dockerfile.web           # Full-stack container
docker-compose.yml       # Production stack
```

---

## Runtime Configuration

### Trace Storage Locations

Traces are imported from Claude Code's history directory:
- **Source**: `%USERPROFILE%\.claude\projects\` (Windows) or `~/.claude/projects/` (Unix)
- **Importer**: `bashgym/trace_capture/importers/claude_history.py`
- **Processed traces**: Stored in `data/traces/`, `data/gold_traces/`, `data/failed_traces/`

Each trace file is a **session** containing multiple tool calls (e.g., "16 read, 11 bash, 2 glob").

### Sessions vs Training Examples

**Sessions** and **Training Examples** are distinct concepts:

| Concept | Description | Format |
|---------|-------------|--------|
| **Session** | A complete coding interaction with many tool calls | `data/traces/*.json` |
| **Training Example** | A single task-response pair for fine-tuning | NeMo JSONL format |

A single session may contain multiple logical tasks. The `ExampleGenerator` segments sessions into training examples.

### Training Data Format (NVIDIA NeMo)

Training examples use NeMo-compatible JSONL format:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Key files:
- `bashgym/factory/example_generator.py` - Session-to-example conversion
- `bashgym/factory/data_factory.py` - TrainingExample and DPOExample classes

### Example Generation Pipeline

```
Session (traces/*.json)
    │
    ▼ ExampleGenerator.segment_session()
Segments (logical task boundaries, carry repo_name/repo_path)
    │
    ▼ ExampleGenerator.segment_to_example()
TrainingExample objects (structured tool-call messages with cognitive tags)
    │
    ▼ ExampleGenerator.export_for_nemo(repo_filter=["ghostwork"])
train.jsonl + val.jsonl (optionally filtered by repo)
```

Segmentation heuristics:
- Time gaps > 5 minutes
- Git commits (task completion)
- Directory changes (different project)
- Cognitive span boundaries (new reasoning chain)

**Cognitive tags in training examples**: When `include_cognitive=True` (default), assistant messages include `<thinking>`, `<plan>`, and `<reflection>` XML tags wrapping the agent's reasoning before each tool call. Duplicate text across tags is suppressed. Import limits: 10KB for thinking blocks, 5KB for text blocks.

**Repo context**: `TaskSegment` carries `repo_name`/`repo_path` from session metadata's `primary_repo`. These flow into `TrainingExample.metadata` for repo-aware filtering at export time.

### Trace Classification Criteria

Traces are classified into three statuses:

| Status | Criteria | Location |
|--------|----------|----------|
| **Gold** | `verification_passed=true` OR manual promotion OR (quality >= 0.7 AND success_rate >= 0.8) | `data/gold_traces/` |
| **Failed** | `verification_passed=false` OR success_rate < 0.3 OR manual demotion | `data/failed_traces/` |
| **Pending** | Raw sessions awaiting classification | `data/traces/` |

### API Endpoints for Training

```bash
# Generate examples from a session
POST /api/traces/{trace_id}/generate-examples

# List generated examples
GET /api/training/examples

# Export to NeMo format (supports repo_filter for repo-specific training)
POST /api/training/export

# Start training
POST /api/training/start
```

### Programmatic Export with Repo Filter

```python
from bashgym.factory.example_generator import ExampleGenerator
gen = ExampleGenerator()
examples, stats = gen.process_directory(Path("data/gold_traces"))
# Export only examples from a specific repo
gen.export_for_nemo(examples, Path("output"), repo_filter=["ghostwork"])
```

### Monitoring Training Progress

**Check if training is running:**
```bash
# Find training processes
wmic process where "commandline like '%train_sft%'" get processid,commandline

# Or on Unix
ps aux | grep train_sft
```

**Check GPU utilization:**
```bash
# Quick status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Continuous monitoring (updates every 1 second)
nvidia-smi -l 1
```

**Training output locations:**
- **Training scripts**: `~/.bashgym/models/{run_id}/train_sft.py`
- **Checkpoints**: `~/.bashgym/models/{run_id}/checkpoint-*`
- **Final model**: `~/.bashgym/models/{run_id}/final/`
- **Merged model**: `~/.bashgym/models/{run_id}/merged/`

**Training data:**
- **Source traces**: `data/gold_traces/*.json`
- **Generated examples**: `~/.bashgym/training_batches/train.jsonl`

**Python environment for training:**
- Training requires Python 3.12 with CUDA support (Python 3.14 lacks PyTorch CUDA wheels)
- Unsloth + PyTorch CUDA 13.0: `pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
- The trainer auto-detects Python 3.12 at `C:\Users\{user}\AppData\Local\Programs\Python\Python312\python.exe`

**Real-time training logs:**
- Logs are streamed via WebSocket (`training:log` message type)
- Frontend displays logs in `TrainingLogs` component on the Training Dashboard
- Backend broadcasts each log line via `broadcast_training_log()` in `websocket.py`
- The trainer accepts a `log_callback` parameter that's called for each stdout line

**Repo-based training:**
- Users can select which repos to include in training (Generalist/Selected/Single)
- Traces are filtered by `primary_repo.name` matching selected repos
- Empty `selected_repos` array means train on all gold traces
- Training Dashboard shows which repos are being used

### Frontend Configuration

Frontend environment variables in `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8002/api
VITE_WS_URL=ws://localhost:8002/ws
```

### Starting the Application

**Unified dev script** (recommended):
```powershell
.\dev.ps1                      # Backend (port 8003) + Frontend (port 5173)
.\dev.ps1 -BackendOnly         # API only
.\dev.ps1 -FrontendOnly        # Frontend only
.\dev.ps1 -Electron            # Backend + Electron app
```

**Manual startup** (separate terminals):
```bash
# Terminal 1 - Backend API (port 8003, hot reload)
python run_backend.py

# Terminal 2 - Frontend (port 5173)
cd frontend && npm run dev
```

**Kill bashgym processes** (Windows):
```powershell
Get-Process -Name "python" | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process
```

### Known Issues

- Port 8000 can get stuck with zombie processes on Windows
- Use port 8002 as workaround if 8000 won't release
- Frontend is Electron-based (`frontend/node_modules/.bin/../electron/cli.js`)

### Common Bugs to Avoid

**JSON Parsing in Electron IPC**: When fetching API responses in `electron/main.ts`, NEVER use `response.json()` directly. Always use `response.text()` first, then `JSON.parse()` with try/catch. Otherwise, non-JSON error responses (like "Internal Server Error") will crash the app with `SyntaxError: Unexpected token`.

```typescript
// WRONG - crashes on non-JSON responses
const data = await response.json()

// CORRECT - handles all responses
const text = await response.text()
try {
  const data = JSON.parse(text)
  return { ok: response.ok, data }
} catch {
  return { ok: false, error: text }
}
```
