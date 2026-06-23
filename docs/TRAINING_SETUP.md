# BashGym — Training Setup

This document describes the training stack in technical detail: the hardware
tiers, the model families, the six training strategies, the data-generation
factory, the provider/serving layer, and the remote-training flow. For the exact
hyperparameter values and quick-start recipes see
[training-config-guide.md](training-config-guide.md); for the beginner/operator
curriculum see [training/overview.md](training/overview.md); for how a session
becomes training data see [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md).

---

## 1. Hardware tiers

Training runs across two tiers from the same workflow. The trainer abstracts the
backend; the operator picks where a run executes.

### Local — consumer GPU
- **NVIDIA GeForce RTX 3080 Ti, 12 GB VRAM.**
- Practical ceiling: small models in the low billions of parameters (for
  example, Gemma 4 E2B trains in ~8 GB with Unsloth). The binding constraint is
  the float32 fused cross-entropy step over a large vocabulary, which dominates
  VRAM at the loss layer regardless of LoRA's parameter savings.
- Used for fast iteration, smoke tests, and 1.5 B specialist fine-tunes.

### Remote — datacenter-class unified memory
- **NVIDIA DGX Spark (GB10 Blackwell), 128 GB unified memory, CUDA 13.0.**
- Unified memory removes the discrete-VRAM ceiling, so larger and full
  (non-LoRA) fine-tunes, MoE models, and longer-context runs are feasible here.
- Also hosts Ollama for local student inference, so the same box trains and
  serves — train, export GGUF, deploy to Ollama on the same machine.
- Reached over SSH (see §6). All serious training runs here.

### Runtime requirements
- Training requires **Python 3.12 with CUDA-enabled PyTorch** (Python 3.14 lacks
  CUDA wheels at time of writing). The trainer auto-detects a Python 3.12
  interpreter for the training subprocess.
- Acceleration via **Unsloth** (2–5× faster, lower memory) on top of PyTorch +
  Hugging Face Transformers + PEFT + TRL.

---

## 2. Base models

BashGym does not mandate a base model. The trainer fine-tunes **any open-weight
HuggingFace causal language model**, accelerated with Unsloth, so you choose the
target that fits your hardware and goal. Unsloth's supported set is wide and grows
continuously; as of mid-2026 it spans, among others:

- **Gemma 4** — E2B (trains in ~8 GB locally), E4B, 12B, 26B-A4B MoE, 31B
- **Qwen3** — incl. Qwen3.6 / 3.7, the Qwen3-Coder MoE variants, and small 4B–14B dense models
- **DeepSeek** — V3.x / V4-class and R1 distillations
- **Llama 4** — Scout / Maverick (very long context); Llama 3.x
- **Mistral** (Small, Devstral) and **Phi-4**

Smaller models (a few billion parameters) fine-tune on the consumer GPU; larger
and mixture-of-experts targets run on the DGX Spark, where unified memory makes
them viable. These are examples, not requirements — set `BASE_MODEL` to whatever
you choose. For the current supported list see the
[Unsloth model catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog).

---

## 3. Training strategies

Six strategies are integrated, selected per run. Each maps to a trainer path in
`bashgym/gym/trainer.py` (`TrainingStrategy` enum) and a generated training
script.

| Strategy | What it does | When to use |
|----------|--------------|-------------|
| **SFT** | Supervised fine-tuning on gold trajectories formatted as tool-call chat messages with cognitive tags. | The baseline. Teach the student the patterns in your verified sessions. |
| **DPO** | Direct preference optimization on chosen/rejected pairs, mined from decision points in traces (a failed approach vs the successful one). | Sharpen behavior where the trace shows a clear better/worse choice. |
| **GRPO** | Group relative policy optimization — on-policy RL against a reward signal (syntax validity, execution success, or test verification). | Optimize for verifiable outcomes, not just imitation. |
| **RLVR** | RL from verifiable rewards — the verification result is the reward. | Tasks with a hard pass/fail gate. |
| **Distillation** | Knowledge distillation from a teacher (Claude) into the student base model. | Transfer teacher capability into a deployable open-weight model. |
| **Cascade RL** | Sequential domain-specialist training with multi-objective policy distillation (see §5). | Build a generalist by composing specialists. |

Fine-tuning uses **LoRA / QLoRA** adapters (4-bit quantization where the hardware
requires it) so runs fit on the consumer GPU; the adapter is merged into the base
weights at export. Exact ranks, learning rates, warmup, gradient accumulation,
and sequence lengths are in [training-config-guide.md](training-config-guide.md).

---

## 4. From traces to training data

Training data is produced by the Factory layer, then optionally amplified by
synthetic generation.

### 4.1 Trace → examples
1. **Import** existing agent history (`~/.claude/projects/` for Claude Code, plus
   Codex / Gemini / Copilot stores) via `bashgym/trace_capture/importers/`.
2. **Classify** each session by a multi-dimension quality score (success rate,
   verification, complexity, tool diversity, efficiency, length) into
   gold / silver / bronze / failed / pending tiers.
3. **Segment** gold sessions into logical tasks (time gaps, git commits,
   directory changes, cognitive-span boundaries) and **convert** each segment to a
   `TrainingExample`: structured tool-call messages with `<thinking>`, `<plan>`,
   and `<reflection>` tags wrapping the agent's reasoning before each tool call.
4. **Export** to NeMo-compatible JSONL (`{"messages": [...]}`) with a train/val
   split, optionally filtered to specific repositories for repo-aware specialists.

### 4.2 Synthetic data — NVIDIA NeMo Data Designer
The factory generates synthetic training data with **NVIDIA NeMo Data Designer**
(≥ 0.6.1), modeled as a column DAG:

| Column type | Role |
|-------------|------|
| `SamplerColumnConfig` | Seed distributions / categorical inputs |
| `LLMTextColumnConfig` | Free-text generation |
| `LLMStructuredColumnConfig` | Pydantic-enforced structured output (e.g. `AgentSolution`, `ToolUseConversation`) |
| `LLMJudgeColumnConfig` | LLM-as-judge scoring on named dimensions (correctness, tool_usage, completeness) |
| `ExpressionColumnConfig` | Jinja2 templating — flatten structured objects into training text, compute DPO chosen/rejected from dual judge scores |
| Processor columns | Filtering, dedup, post-processing |

Five pipeline presets cover SFT, DPO, tool-use, external, and unstructured data.
Per-column model assignment lets a fast code model generate solutions while a
stronger model judges them.

### 4.3 The AutoCurriculum Compiler (SchemaResearcher)
Rather than hand-tuning generation prompts, the **SchemaResearcher** treats a Data
Designer pipeline config as an evolvable genome and searches for recipes that
produce measurably better models:

1. Represent the pipeline config as a YAML-serializable dict (the genome).
2. Mutate it — toggle booleans, perturb temperatures/thresholds within bounds,
   swap column choices.
3. Evaluate in two stages:
   - **Stage 1 (fast):** generate ~25 examples, take the average judge score,
     drop the bottom candidates. ~50% filtered cheaply.
   - **Stage 2 (slow):** generate the full dataset from survivors, micro-train for
     ~50 steps, measure actual validation loss.
4. Keep the best genome, repeat across generations.

This reduces evaluation cost ~5× versus full-training every candidate while
keeping the fitness signal grounded in real downstream loss.

---

## 5. Cascade RL and multi-objective distillation

The Cascade scheduler (`bashgym/gym/cascade_scheduler.py`) trains
domain-specialist stages in sequence and chains their checkpoints:

```
base ──▶ stage 1 (file ops) ──▶ stage 2 (bash) ──▶ stage 3 (search) ──▶ stage 4 (multi-step)
                                                                              │
                                                            MOPD distillation ▼
                                                                     unified student
```

- **Domains** are defined by tool filter and minimum step count
  (file_operations, bash_commands, search_and_navigate, multi_step_reasoning), or
  auto-discovered per repository when enough traces exist for that repo.
- Each stage's reward mode (`syntax` / `execution` / `verification`) sets the data
  contract — verification mode runs per-example tests and scores pass/fail.
- After the cascade, **multi-objective policy distillation (MOPD)** merges the
  domain experts into one unified student.
- Stages chain checkpoints: stage N's best checkpoint becomes stage N+1's base.

---

## 6. Remote training over SSH

Remote runs execute on the DGX Spark and stream back to the dashboard
(`bashgym/gym/remote_trainer.py`):

```
generate training script locally
        │  SFTP upload
        ▼
remote host: ssh exec under nohup  ──▶ training runs
        │                                   │
        │  stream stdout (→ WebSocket)      │
        ▼                                   ▼
dashboard log view              SFTP download artifacts (checkpoints, merged, GGUF)
```

- **Process control** from the dashboard: pause / resume / cancel map to
  `SIGSTOP` / `SIGCONT` / `SIGTERM` on the remote process.
- **Orphan recovery:** if the backend restarts, it reconnects to in-flight remote
  runs from persisted state rather than losing them.
- **Configuration** via `SSH_REMOTE_*` environment variables (host, user, port,
  key path, working directory). A `GET /api/ssh/preflight` check verifies the
  remote machine is ready (Python, disk, network) before a run starts.

Training artifacts land under `~/.bashgym/models/{run_id}/`: intermediate
`checkpoint-*`, `final/`, `merged/` (LoRA merged into base), and the GGUF export.

---

## 7. Serving and the provider layer

Inference is pluggable behind one `InferenceProvider` interface
(`bashgym/providers/`), with a `ProviderRegistry` mapping models to providers and
monitoring health.

| Provider | Local? | Role | Notes |
|----------|--------|------|-------|
| **Anthropic (Claude)** | No | Teacher | Frontier model for distillation and router fallback. |
| **NVIDIA NIM** | No | Cloud student inference | OpenAI-compatible; 100+ served models. |
| **Ollama** | Yes | Local student inference | Serves GGUF on the DGX Spark; warm-up + VRAM tracking. |

**Live model discovery:** both the Anthropic and NIM providers query their
`/v1/models` endpoints at runtime so the catalog reflects what is actually served,
with a current static fallback if the endpoint is unreachable. This is exposed as
a refresh operation, so the available-model list never silently points at a
retired model.

**Teacher/student router** (`bashgym/gym/router.py`): routes each request by
strategy — teacher-only, student-only, confidence-based, task-complexity,
progressive (gradually shift traffic to the student as its success rate rises), or
random-sample — with automatic fallback to the teacher on low confidence or a
guardrail block.

---

## 8. Hyperparameter search (AutoResearch)

`bashgym/gym/autoresearch.py` runs a population-based evolutionary search over
training hyperparameters (learning rate, LoRA rank/alpha, dropout, warmup ratio,
gradient accumulation, batch size, max sequence length, 4-bit on/off):

- **Simulate mode** — a cheap surrogate loss model for fast exploration of the
  search space.
- **Real mode** — trains each candidate and measures validation loss.
- Mutation rate and scale are configurable; experiments stream to the dashboard.
- The same `SearchSpace` abstraction backs both this and the data-pipeline
  SchemaResearcher (§4.3).

---

## 9. Verification and evaluation

A trajectory only becomes training data if it passes verification, and a trained
model is gated on evaluation before deployment.

- **Verification** (`bashgym/judge/verifier.py`): discovers and runs tests
  (pytest, bats, `npm test`, Makefile targets, custom `verify.sh`), records exit
  code and pass/fail counts. Exit-code-zero is the gate that admits a trajectory
  to the gold pool.
- **LLM-as-judge** (`bashgym/judge/`): scores quality across dimensions for both
  the dataset filter and the SchemaResearcher fitness function.
- **Benchmarks**: a benchmark harness supports standard code/agentic suites for
  comparing student against teacher.

---

## 10. The closed local inference loop

The training and serving tiers compose into a continuous loop on the DGX Spark:

```
Train (remote) ──▶ GGUF export ──▶ deploy to Ollama ──▶ set as router Student
       ▲                                                        │
       │                                                        ▼
   retrain  ◀────  collect new traces  ◀────  router sends inference locally
```

Each cycle: the student gets better → the router sends it more traffic → fewer
teacher calls → lower cost — while the sessions the student produces become the
next training set.

---

## Read next

- [training/overview.md](training/overview.md) — how the training gym works from first principles.
- [training/strategy-guide.md](training/strategy-guide.md) — concrete starting settings and strategy selection.
- [training/agent-cli.md](training/agent-cli.md) — machine-readable CLI commands for agents setting up runs and analyzing replay artifacts.
- [training/world-models.md](training/world-models.md) — ECHO/RWML contracts, replay payloads, backend integration, and telemetry boundaries.
- [training/metrics-runbook.md](training/metrics-runbook.md) — diagnose flat pass@k, zero reward variance, timeouts, and verifier failures.
- [training-config-guide.md](training-config-guide.md) — exact hyperparameters, LoRA/QLoRA settings, and quick-start recipes.
- [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) — trace format, quality tiers, and the example-generation pipeline.
- [PLATFORM_OVERVIEW.md](PLATFORM_OVERVIEW.md) — the platform architecture and design rationale.
- [GETTING_STARTED.md](GETTING_STARTED.md) — install to first trained model, step by step.
