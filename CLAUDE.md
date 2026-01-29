# Bash Gym - Agent Documentation

> A Self-Improving Agentic Development Gym

This document provides comprehensive guidance for AI agents working with this codebase.

## Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest test_bashgym.py -v

# Run with a task
python main.py --task "Write a hello world script"

# Get help
python main.py --help
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

#### `main.py` - Pipeline Orchestrator
Entry point for the system. Handles CLI arguments and orchestrates the full pipeline.

```bash
# Single task
python main.py --task "Refactor utils.py"

# Batch processing
python main.py --batch tasks.jsonl --output results/

# Training
python main.py --train --dataset data/sft_batch.jsonl --strategy sft
```

Key classes:
- `BashGymConfig` - Main configuration dataclass
- `BashGym` - Main orchestrator class
- `parse_args()` - CLI argument parser

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
- `ModelRouter` - Main routing logic

#### `settings.py` - Configuration Management
Centralized settings loaded from environment variables.

```python
from settings import get_settings
settings = get_settings()
print(settings.api.anthropic_api_key)
print(settings.training.base_model)
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `NVIDIA_API_KEY` | No | NVIDIA NIM API key |
| `BASE_MODEL` | No | Fine-tuning base model (default: Qwen2.5-Coder-1.5B) |
| `USE_NEMO_GYM` | No | Enable cloud training (default: false) |
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

The test suite in `test_bashgym.py` provides 72 test functions covering all modules.

```bash
# Run all tests
pytest test_bashgym.py -v

# Run specific module tests
pytest test_bashgym.py::TestTrainer -v

# With coverage
pytest test_bashgym.py --cov=. --cov-report=html
```

### Test Classes

- `TestSettings` - Configuration loading
- `TestSandboxManager` - Dangerous command detection
- `TestVerifier` - Test file discovery
- `TestTraceProcessor` - Quality scoring, redaction
- `TestDataFactory` - Training example creation
- `TestTrainer` - Script generation
- `TestBashGymEnv` - Environment step/reset
- `TestModelRouter` - Routing strategies

---

## Common Tasks

### Adding a New Training Strategy

1. Add strategy to `TrainingStrategy` enum in `trainer.py`
2. Implement `train_{strategy}()` method in `Trainer` class
3. Add script generation in `_generate_{strategy}_script()`
4. Add tests in `test_bashgym.py`

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
# Build containers
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f arena
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
bashgym/
├── main.py              # CLI entry point
├── settings.py          # Configuration
├── sandbox.py           # Docker sandbox (Arena)
├── agent_runner.py      # Claude CLI wrapper (Arena)
├── verifier.py          # Test execution (Judge)
├── verify.sh            # Verification script (Judge)
├── data_factory.py      # Training data synthesis (Factory)
├── trace_processor.py   # Trace processing (Factory)
├── trainer.py           # Model training (Gym)
├── gym_env.py           # RL environment (Gym)
├── model_router.py      # Model routing (Gym)
├── post_tool_use.py     # Claude hook
├── session_end.py       # Claude hook
├── test_bashgym.py     # Test suite
├── requirements.txt     # Core dependencies
├── requirements-training.txt  # ML dependencies
├── pyproject.toml       # Package config
├── .env.example         # Environment template
└── docker/              # Docker configuration
    ├── Dockerfile.arena
    ├── Dockerfile.sandbox
    └── docker-compose.yml
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
Segments (logical task boundaries)
    │
    ▼ ExampleGenerator.segment_to_example()
TrainingExample objects
    │
    ▼ ExampleGenerator.export_for_nemo()
train.jsonl + val.jsonl
```

Segmentation heuristics:
- Time gaps > 5 minutes
- Git commits (task completion)
- Directory changes (different project)
- File scope changes

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

# Export to NeMo format
POST /api/training/export

# Start training
POST /api/training/start
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

**API Server** (port 8002):
```bash
uvicorn bashgym.api.routes:app --host 0.0.0.0 --port 8002
```

**Frontend** (from `frontend/` directory):
```bash
npm run dev
```

**Kill bashgym processes** (Windows):
```powershell
wmic process where "name='node.exe'" get processid,commandline | findstr "bashgym"
taskkill /F /PID <pid>
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
