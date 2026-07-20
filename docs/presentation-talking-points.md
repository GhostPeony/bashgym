# BashGym: Self-Improving Agentic Development Gym
## Presentation Talking Points

---

## 1. The Ouroboros Flywheel

**Core idea:** An AI agent that improves itself by learning from its own work.

```
ACT (Arena) -> VERIFY (Judge) -> SYNTHESIZE (Factory) -> TRAIN (Gym) -> DEPLOY -> ACT ...
```

- **Arena:** Claude Code works in sandboxed Docker containers, solving real coding tasks
- **Judge:** Automated verification (pytest, bats, custom scripts) determines if the solution actually works
- **Factory:** Successful sessions become structured training data (NeMo JSONL format)
- **Gym:** Fine-tune a smaller student model (any open-weight model — e.g. Gemma 4, Qwen3, Llama 4) on these traces via SFT, DPO, or GRPO
- **Deploy:** Student model serves inference locally via Ollama on DGX Spark, routing gradually shifts from Teacher (Claude) to Student

**Why this matters:** Every coding session makes the system better. No manual data curation needed for the base loop.

---

## 2. What Makes This Different from Standard Fine-Tuning

| Standard Fine-Tuning | BashGym Flywheel |
|---|---|
| Static dataset, curated once | Continuously growing from real work |
| Separate data team | Self-generating from agent traces |
| Train once, deploy | Iterative: train, deploy, collect more, retrain |
| Generic benchmark eval | Task-specific verification (does the code actually pass tests?) |
| Teacher OR student | Teacher-student routing with confidence-based handoff |

**Key insight:** The verification step is what closes the loop. Without automated verification, you can't distinguish good traces from bad ones, and the flywheel doesn't spin.

---

## 3. Training Strategies (All Implemented)

### SFT (Supervised Fine-Tuning)
- Learn from gold traces: "here's what a successful coding session looks like"
- Fastest to train, good baseline
- Risk: can memorize rather than generalize

### DPO (Direct Preference Optimization)
- Uses paired examples: gold trace (chosen) vs failed trace (rejected)
- Teaches the model *what not to do* as well as what to do
- Better alignment with human preferences than pure SFT

### GRPO (Group Relative Policy Optimization)
- Reinforcement learning from verification rewards
- Multiple completions ranked by verification outcome
- Most sample-efficient but needs more compute per step

### Knowledge Distillation
- Teacher (Claude Sonnet/Opus) generates responses, student learns to match
- Can augment trace data with synthetic examples
- Useful when gold traces are sparse

---

## 4. AutoResearch: Durable, controlled improvement campaigns

The current AutoResearch product is a durable campaign control plane—not an
automated hyperparameter loop. It creates a reviewable record that binds
registered model, data, evaluator, and compute profiles; establishes a real
baseline; permits controlled one-variable candidates; retains budget and
evidence lineage; and uses human gates where required. The restart-safe
Training → AutoResearch **Control Room** projects that campaign state, while a
separate, explicit **Start** approval remains required after a campaign reaches
`READY`.

> **Legacy compatibility only:** the earlier hyperparameter, trace-mining, and
> schema-search researchers are hidden, unsupported compatibility material.
> Their `/api/autoresearch/*` routes register only when
> `BASHGYM_ENABLE_LEGACY_AUTORESEARCH=true`; they are never a durable campaign
> path or a Control Room feature.

---

## 5. The Data Pipeline (Trace -> Training Example)

```
Claude Code session (many tool calls)
    |
    v  Segmentation (time gaps, git commits, directory changes)
Task segments (logical units of work)
    |
    v  Quality scoring + classification
Gold / Pending / Failed traces
    |
    v  Example generation (with cognitive tags)
Training examples (NeMo JSONL)
    |
    v  Optional repo filtering
Specialist or generalist training data
```

**Cognitive tags:** Training examples include `<thinking>`, `<plan>`, and `<reflection>` XML tags from the agent's reasoning. This teaches the student model *how to think*, not just what to output.

**Repo-aware training:** Can train a generalist (all repos), specialist (one repo), or mixed model.

---

## 6. Infrastructure

### Local Setup
- **Training:** NVIDIA RTX 3080 Ti (12GB VRAM) — enough for Qwen 1.5B with QLoRA
- **Inference:** DGX Spark running Ollama — serves the student model after training
- **Routing:** Confidence-based model router gradually shifts traffic from Claude to student

### Training Flow
```
Generate script (Unsloth + LoRA) -> Execute on GPU -> Export GGUF -> Deploy to Ollama -> Set as Student
```

### Remote Training (DGX Spark via SSH)
- SSH-based execution for heavier training runs
- Script upload via SFTP, log streaming back to dashboard
- Pause/resume/cancel via SSH signals

---

## 7. 2026 Research Alignment

### What the field is doing now (Q1 2026):
- **Self-play and self-improvement loops** are mainstream — Google DeepMind, Meta FAIR, and multiple startups are publishing on closed-loop agent training
- **Verification-driven training** is the consensus approach — RL from verification rewards (not just human feedback) is the key enabler
- **Smaller models, better data** — the trend is toward 1-7B parameter models that outperform larger ones through better training data curation
- **Constitutional AI meets code** — using automated testing as the "constitution" for code agents

### Where BashGym fits:
- Implements the full loop that most papers only describe theoretically
- Durable campaigns make data and evaluation lineage explicit while teams address
  the data-curation bottleneck
- Knowledge distillation from Claude -> student is the practical version of large-to-small model transfer
- DPO from failed traces is the code-specific version of preference learning from negative examples

---

## 8. What's Next

- **Durable campaign integrations:** Expand verified trainer/evaluator adapters
  without weakening the registered-binding, baseline, budget, or evidence
  contracts
- **Multi-objective optimization:** Optimize for loss AND inference speed AND memory usage simultaneously
- **Cross-project transfer:** Can a model trained on one codebase help with another? Repo-aware training enables this experiment
- **Continuous deployment:** Automatic retraining trigger when gold trace count exceeds threshold
- **Evaluation harness:** Standardized benchmarks beyond just training loss (HumanEval, SWE-bench style tasks)

---

## Key Takeaways for the Audience

1. **The loop is the product.** Individual components (training, verification, data synthesis) exist elsewhere. The closed loop is what creates compound improvement.

2. **Data quality > model size.** A well-curated 1.5B model trained on your own verified coding traces can outperform a general-purpose 7B model on your specific tasks.

3. **Verification is the bottleneck.** Without automated verification, you can't close the loop. Invest in test infrastructure.

4. **AutoResearch makes improvement accountable.** Durable campaign records make
   a baseline, controlled candidate, budget, and evidence reviewable rather than
   treating tuning as an opaque background loop.

5. **This runs on consumer hardware.** RTX 3080 Ti + DGX Spark. No cloud GPU cluster needed for the core loop.
