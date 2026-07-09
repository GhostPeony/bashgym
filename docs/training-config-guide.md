# Training Configuration Guide

A practical reference for every option in Bash Gym's Training Configuration panel.
For the broader operator curriculum, start with
[training/overview.md](training/overview.md) and
[training/capability-map.md](training/capability-map.md), then use
[training/strategy-guide.md](training/strategy-guide.md) and
[training/metrics-runbook.md](training/metrics-runbook.md) while configuring and
diagnosing runs. For terminal RL, use
[training/tmax-terminal-rl-recipe.md](training/tmax-terminal-rl-recipe.md); before
private/cloud backend work, use
[training/private-compute-eval-checklist.md](training/private-compute-eval-checklist.md).

---

## Data Source

Choose where your training examples come from.

| Option | When to Use |
|--------|-------------|
| **Gold Traces** (default) | You've collected traces from AI coding sessions and classified good ones as gold. This auto-generates training data. |
| **Custom JSONL** | You have a pre-formatted dataset in NeMo JSONL format (messages array with system/user/assistant roles). |
| **Security** | Ingest a public security dataset (EMBER malware, PhishTank URLs, etc.) directly into training examples. |

For security datasets, you can choose **Direct** mode (fast, no API calls) or **Enriched** mode (uses an LLM to add reasoning to each example).

---

## Training Data Scope

*Only shown when using Gold Traces as data source.*

| Scope | Effect |
|-------|--------|
| **Generalist** | Train on all gold traces across all repos. Produces a versatile model. |
| **Mixed** | Select specific repos. Good for building expertise in 2-3 related projects. |
| **Specialist** | Single repo only. Maximum specialization for one codebase. |

**Tip**: Start with Generalist for your first training run. Specialist training works best when you have 50+ gold traces from a single repo.

---

## Training Backend

| Backend | Requirements | Best For |
|---------|-------------|----------|
| **Local** | CUDA GPU + Unsloth installed | Small models (1.5B-7B) on your own hardware |
| **Private compute target** | Saved device or private compute config | Larger models or when your local GPU is busy |
| **NeMo Cloud** | NVIDIA NeMo API key | Scalable cloud training, largest models |

---

## Training Strategy

### SFT (Supervised Fine-Tuning)
**Start here.** The model learns to imitate the patterns in your gold traces. Direct and reliable.

- Best for: First training run, clear task-response pairs
- Data: Gold traces with high success rates

### DPO (Direct Preference Optimization)
The model learns what's *better* by comparing gold (chosen) and failed (rejected) trace pairs.

- Best for: Second iteration after SFT, when you have both good and bad traces
- Data: Paired examples (gold vs failed)
- **Beta** (0.01-1.0): Controls how conservative the model is. Lower = stays closer to the SFT baseline. Start at **0.1**.

### GRPO (Group Relative Policy Optimization)
RL-based training. Samples multiple completions per prompt, rewards the best ones.

- Best for: Advanced optimization after SFT+DPO, tasks with verifiable outcomes
- Data: Prompts with reward signals
- **Generations per prompt** (2-16): More = better signal but slower. Start at **4**.
- **Temperature** (0.1-2.0): Sampling diversity. Higher = more exploration. Start at **0.7**.

### KD (Knowledge Distillation)
Transfer knowledge from a large teacher model (Claude, GPT-4, 70B+) to your small student model.

- Best for: When you have API access to a powerful model and want to compress its knowledge
- **Teacher Model**: The large model to distill from
- **Temperature** (0.5-10): Higher makes the teacher's output distribution softer, which gives the student more information per example. Start at **2.0**.
- **Alpha** (0-1): Balance between distillation loss and standard task loss. 0.5 is a balanced starting point.

### Session Distillation
Targeted self-distillation for failed trace spans. BashGym inserts a short local
hint before the mistake, scores the same target action under original and hinted
context, and trains only the target span with masked KL/CE.

- Best for: retries, failed commands, local recovery pivots, and small tool-choice mistakes
- Data: `session_distillation_records.jsonl`
- **Alpha** (0-1): Weight on hinted-context KL versus hard-label CE. Start at **0.7**.
- **Temperature** (>0): Softness for the hinted-context distribution. Start at **1.0**.
- **Minimum confidence** (0-1): Filters weak reader guesses. Start at **0.6**.
- **Mask policy**: Use **target_span_only** so unrelated transcript text is not trained.

### Recommended progression
```
SFT -> DPO -> GRPO/RLVR (optional)
         ^
         KD or Session Distillation when the student needs a bridge
```

---

## Base Model

Choose the model architecture to fine-tune. Key considerations:

| VRAM Available | Example targets (mid-2026) |
|----------------|----------------------------|
| 8-12 GB | Gemma 4 E2B; small Qwen3 / Phi-4 dense models |
| 12-24 GB | Gemma 4 E4B; Qwen3 4B-14B; Mistral Small |
| 24-48 GB | Gemma 4 12B / 26B-A4B; Qwen3 dense ~30B |
| 48+ GB | Qwen3-Coder / Qwen3 MoE; DeepSeek V3.x-V4; Llama 4 |

These are examples, not requirements. New models land in Unsloth continuously —
see the [Unsloth model catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
for the live list. Code-specialized variants (e.g. Qwen3-Coder, DeepSeek-Coder)
suit coding tasks; strong general instruct models also work well.

---

## Hyperparameters

### Epochs
Number of complete passes through the training data.

- **1-3**: Standard range. More epochs = more memorization.
- Start with **3** and reduce if you see overfitting (validation loss starts rising).

### Batch Size
Samples processed per GPU forward pass.

- **1**: Safe for 12GB VRAM with QLoRA
- **2-4**: If you have more VRAM headroom
- Increase **gradient accumulation** instead of batch size to avoid OOM

### Learning Rate
How aggressively the model updates weights.

- **2e-5**: Safe default for LoRA fine-tuning
- **1e-5**: More conservative, good for small datasets
- **5e-5**: More aggressive, good for large datasets

### Warmup Ratio
Fraction of total training steps where learning rate linearly ramps up from 0.

- **0.1** (10%): Default, works well for most runs
- **0.03-0.05**: For very long training runs
- **0.0**: No warmup (not recommended)

### Gradient Accumulation Steps
Simulates a larger batch size by accumulating gradients over multiple forward passes before updating weights.

- **Effective batch size** = batch_size × gradient_accumulation_steps
- Default **8** with batch_size **1** = effective batch of **8**
- Increase if you want a larger effective batch without more VRAM

### Save Steps
How often to save a checkpoint. Checkpoints let you recover from crashes and compare model states.

- **100**: Good for short runs
- **500**: Good for longer runs (saves disk space)

### Max Sequence Length
Maximum token length for training examples. Longer sequences need more VRAM.

- **2048**: Default, handles most coding tasks
- **4096**: For longer context (needs more VRAM)
- **512-1024**: To reduce VRAM usage

---

## LoRA Configuration

LoRA (Low-Rank Adaptation) trains a small set of adapter weights instead of the full model, dramatically reducing VRAM usage.

### LoRA Rank
Controls adapter capacity. Higher = more parameters = more expressive but more VRAM.

- **8**: Lightweight, good for small datasets
- **16**: Default, good balance
- **32-64**: More capacity, use for complex tasks or large datasets
- **128**: Maximum, rarely needed

### LoRA Alpha
Scaling factor for LoRA updates. Usually set to **2× the rank**.

- **32**: Default (with rank 16)
- Rule of thumb: `alpha = 2 * rank`

### LoRA Dropout
Regularization to prevent overfitting in LoRA layers.

- **0.05**: Default, light regularization
- **0.1**: More regularization for small datasets
- **0.0**: No dropout (for very large datasets)

### QLoRA (4-bit Quantization)
Loads the base model in 4-bit precision, reducing VRAM by ~50%.

- **Enabled** (default): Recommended for GPUs with ≤24GB VRAM
- **Disabled**: Only if you have abundant VRAM and want maximum precision

**Important**: QLoRA is not recommended for MoE (Mixture of Experts) models like Qwen3.5-35B-A3B — the quantization can cause accuracy loss across expert routing.

---

## Terminal RL and World Models

For terminal-agent RL, use the practical recipes in
[training/strategy-guide.md](training/strategy-guide.md). The
`terminal_rl_tmax_like` profile is the stable starter for environment rollouts:
group size 32 by default, DAPO loss, zero-std filtering, active sampling,
token-level loss, FP32 LM head, and interleaved thinking.

For ECHO/RWML, see [training/world-models.md](training/world-models.md). Starter
defaults are `echo_aux_lambda=0.05`, `rwml_distance_threshold=0.2`,
`rwml_easy_pass_rate_threshold=0.8`, `rwml_easy_keep_probability=0.1`, and
`rwml_history_window=4`. Treat these metrics as diagnostics until they correlate
with heldout pass@k and safety gates.

---

## Quick Start Recipes

### First training run (12GB GPU)
```
Strategy: SFT
Base Model: (a small instruct model, e.g. Gemma 4 E2B)
Epochs: 3
Batch Size: 1
Gradient Accumulation: 8
Learning Rate: 2e-5
LoRA Rank: 16
QLoRA: Enabled
Max Seq Length: 2048
```

### High-quality DPO refinement
```
Strategy: DPO
Base Model: (your SFT checkpoint)
Epochs: 1-2
Beta: 0.1
Learning Rate: 5e-6 (lower than SFT)
```

### Private compute training (larger model)
```
Backend: Private compute target
Strategy: SFT
Base Model: (a larger model, e.g. Gemma 4 E4B or a Qwen3 dense model)
Epochs: 3
Batch Size: 4
Gradient Accumulation: 4
QLoRA: Enabled
Max Seq Length: 4096
```
