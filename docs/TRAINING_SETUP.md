# BashGym — Training Setup

This document describes the training stack in technical detail: the hardware
tiers, the model families, the training strategies, the data-generation
factory, the provider/serving layer, and the local/private compute flow. For the exact
hyperparameter values and quick-start recipes see
[training-config-guide.md](training-config-guide.md); for the beginner/operator
curriculum see [training/overview.md](training/overview.md); for the stable,
backend-dependent, and diagnostic capability map see
[training/capability-map.md](training/capability-map.md); for how a session becomes
training data see [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md).

---

## 1. Hardware tiers

Training runs across local and private compute targets from the same
workflow. The trainer abstracts the backend; the operator picks where a run
executes. Hosted backends are optional and never a fallback for a registered
local/private campaign.

### Local — consumer GPU
- Register the actual GPU inventory and available memory instead of selecting a
  repository-owned device profile.
- Use small LoRA/QLoRA runs, bounded smoke tests, and short contexts when memory
  is constrained. The model doctor and trainer preflight decide fit from the
  selected artifact, runtime, and hardware.

### Private compute target
- Larger GPU memory or unified-memory systems for dense models, MoE variants,
  longer-context runs, and installed-backend smokes.
- The target can train, export GGUF, and optionally serve the student locally if
  the operator has an inference runtime such as Ollama available there.
- Use this tier only after local contracts and smoke artifacts are clean enough
  to justify the run.

### Runtime requirements
- Training requires a Python, CUDA, PyTorch, and trainer combination supported
  by the explicitly selected backend. Keep this runtime isolated from the
  lightweight BashGym control-plane environment.
- Acceleration via **Unsloth** (2–5× faster, lower memory) on top of PyTorch +
  Hugging Face Transformers + PEFT + TRL.

---

## 2. Base models

BashGym does not mandate or suggest a repository-owned base model. Training
starts from an operator-selected trainable artifact whose immutable revision,
task, architecture, tokenizer, quantization, runtime, and hardware fit are
accepted by the installed backend. A Hugging Face cache entry, served model,
GGUF/inference quant, or adapter cannot silently satisfy that contract.

Backend catalogs are useful for discovery but are not BashGym compatibility
guarantees. For example, consult the
[Unsloth model catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
when Unsloth is the selected backend, then run BashGym's artifact inspection and
doctor checks against the exact local snapshot before launching training.

---

## 3. Training strategies

Training strategies are integrated, selected per run. Each maps to a trainer path in
`bashgym/gym/trainer.py` (`TrainingStrategy` enum) and a generated training
script.

| Strategy | What it does | When to use |
|----------|--------------|-------------|
| **SFT** | Supervised fine-tuning on gold trajectories formatted as tool-call chat messages with cognitive tags. | The baseline. Teach the student the patterns in your verified sessions. |
| **DPO** | Direct preference optimization on chosen/rejected pairs, mined from decision points in traces (a failed approach vs the successful one). | Sharpen behavior where the trace shows a clear better/worse choice. |
| **GRPO** | Group relative policy optimization — on-policy RL against a reward signal (syntax validity, execution success, or test verification). | Optimize for verifiable outcomes, not just imitation. |
| **RLVR** | RL from verifiable rewards — the verification result is the reward. | Tasks with a hard pass/fail gate. |
| **Distillation** | Knowledge distillation from a teacher (Claude) into the student base model. | Transfer teacher capability into a deployable open-weight model. |
| **Session Distillation** | Hint-injected self-distillation over failed trace spans with masked KL/CE on the same target tokens. | Repair local mistakes, retry loops, and recovery pivots without replacing the trajectory. |
| **Cascade RL** | Sequential domain-specialist training with multi-objective policy distillation (see §5). | Build a generalist by composing specialists. |

Fine-tuning can use **LoRA / QLoRA** adapters when the selected model/backend
supports them. The retention policy decides whether to keep an adapter, merged
weights, or other deployable artifacts. Exact ranks, learning rates, warmup,
gradient accumulation, and sequence lengths are in
[training-config-guide.md](training-config-guide.md).

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

## 6. Private compute training

Private compute runs execute on the configured target and stream back to the
dashboard (`bashgym/gym/remote_trainer.py`):

```
generate training script locally
        │  SFTP upload
        ▼
compute target: launch under nohup  ──▶ training runs
        │                                   │
        │  stream stdout (→ WebSocket)      │
        ▼                                   ▼
dashboard log view              SFTP download artifacts (checkpoints, merged, GGUF)
```

- **Process control** from the dashboard: pause / resume / cancel map to
  `SIGSTOP` / `SIGCONT` / `SIGTERM` on the remote process.
- **Orphan recovery:** if the backend restarts, it reconnects to in-flight remote
  runs from persisted state rather than losing them.
- **Configuration** via private compute environment variables or saved device
  settings (host, user, port, key path, working directory). A preflight check
  verifies the target is ready (Python, disk, network) before a run starts.

Training artifacts land under `~/.bashgym/models/{run_id}/`: intermediate
`checkpoint-*`, `final/`, `merged/` (LoRA merged into base), and the GGUF export.

- **Remote venv contract:** the target must provide a training venv at
  `{SSH_REMOTE_WORK_DIR}/venv`; preflight and the uploaded script both
  activate it.
- **Remote path contract:** generated scripts run inside the uploaded run
  directory, load the dataset by bare filename, and write artifacts to
  `./final` and `./merged` — the locations the artifact download expects.

Each operator keeps machine-specific environment inventories and launch notes
outside the repository. Public contracts use only logical local/private compute
profile IDs and doctor output.

---

## 7. Serving and the provider layer

Inference is pluggable behind one `InferenceProvider` interface
(`bashgym/providers/`), with a `ProviderRegistry` mapping models to providers and
monitoring health.

| Provider | Local? | Role | Notes |
|----------|--------|------|-------|
| **Anthropic (Claude)** | No | Teacher | Frontier model for distillation and router fallback. |
| **NVIDIA NIM** | No | Cloud student inference | OpenAI-compatible; 100+ served models. |
| **Ollama** | Yes | Local student inference | Serves GGUF on a local or private compute target; warm-up + VRAM tracking. |

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

This earlier in-process search loop is distinct from the authoritative durable
campaign controller. The durable path applies the same baseline-first campaign,
budget, lineage, evidence, and promotion contracts across registered models and
trainers; this module remains an interactive search implementation beneath that
broader product surface.

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

The training and serving tiers compose into a continuous loop on any target that
can both train and serve the exported student:

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
- [training/capability-map.md](training/capability-map.md) — full training/eval spread and stable vs backend-dependent status.
- [training/strategy-guide.md](training/strategy-guide.md) — concrete starting settings and strategy selection.
- [training/agent-cli.md](training/agent-cli.md) — machine-readable CLI commands for agents setting up runs and analyzing replay artifacts.
- [training/tmax-terminal-rl-recipe.md](training/tmax-terminal-rl-recipe.md) — TMax-style terminal RL path from environments to backend smoke and release gates.
- [training/private-compute-eval-checklist.md](training/private-compute-eval-checklist.md) — local/private compute backend-smoke and eval checklist.
- [training/autoresearch-campaign.md](training/autoresearch-campaign.md) — durable plan-first activation, doctor, baseline, and candidate path.
- [training/world-models.md](training/world-models.md) — ECHO/RWML contracts, replay payloads, backend integration, and telemetry boundaries.
- [training/metrics-runbook.md](training/metrics-runbook.md) — diagnose flat pass@k, zero reward variance, timeouts, and verifier failures.
- [training-config-guide.md](training-config-guide.md) — exact hyperparameters, LoRA/QLoRA settings, and quick-start recipes.
- [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) — trace format, quality tiers, and the example-generation pipeline.
- [Project structure](../README.md#project-structure) — the current public package map.
- [GETTING_STARTED.md](GETTING_STARTED.md) — install to first trained model, step by step.
