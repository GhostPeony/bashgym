# BashGym — Platform Recap: Training Methods, When to Use Them, and Adaptability

*A self-improving agentic-development gym. Capture coding-agent traces → synthesize training data → train an open model → gate it on a held-out eval → deploy → collect new traces → repeat.*

Compiled 2026-06-16 from a code-level audit (three parallel agents combed the training, data, and eval/deploy layers; key claims spot-verified against source).

---

## 1. The flywheel in one picture

```
        ┌──────────── ACT (Arena) ─────────────┐
        │  Docker sandbox + agent runner         │
        ▼                                        │
   VERIFY (Judge) ── pass/fail, pass-rate        │
        │                                        │
        ▼                                        │
   SYNTHESIZE (Factory) ── traces → SFT/DPO/GRPO data
        │                                        │
        ▼                                        │
   TRAIN (Gym) ── SFT · DPO · GRPO/RLVR · Distill · Cascade · MOPD
        │                                        │
        ▼                                        │
   EVAL gate ── held-out ship/no-ship + forgetting + pass@k
        │                                        │
        ▼                                        │
   DEPLOY ── GGUF→Ollama / HF Hub / served endpoint ──┘ (new traces)
```

Everything below hangs off this loop. The design separates **what to learn** (training *strategies*), **where to run it** (execution *backends*), and **what model** (a family *profile*) — so each axis can change independently.

---

## 2. Training methods — what each is, and when to reach for it

### Core strategies (`bashgym/gym/trainer.py`, `TrainingStrategy` enum)

| Method | What it does | **Use it when…** | Data format | Compute |
|---|---|---|---|---|
| **SFT** | Imitation — LoRA fine-tune to reproduce gold assistant turns (Unsloth `SFTTrainer`). Cognitive tags (`<thinking>/<plan>/<reflection>`) teach reason-then-act. | You have good traces and want the student to **mimic** them. The default baseline and every cascade stage's fallback. | `messages` JSONL | Local 12GB LoRA |
| **DPO** | Preference — push toward `chosen`, away from `rejected` (implicit-reference DPO). A `DegenerateAccuracyStop` callback aborts if it can't separate the pair. | You have **contrastive** pairs (a good and a bad answer to the same prompt) and want to sharpen beyond imitation. | `chosen`/`rejected` | Local LoRA |
| **GRPO** | RL from verifiable rewards — sample N completions, score each, update by group-relative advantage. Three reward tiers: `syntax` (parses), `execution` (runs), `verification` (pytest pass-rate). | You have a **programmatic correctness signal** (tests/execution) instead of labels. The workhorse for "make it actually solve tasks." | `prompt` (+`tests`) | Local or DGX |
| **RLVR** | GRPO with reward hard-pinned to `verification` (test pass-rate). | You specifically want **test-pass-rate as the objective** and your data carries `tests`. | `prompt` + `tests` | Local or DGX |
| **Distillation** | Student learns from a teacher via KL(soft labels)·T² + α·CE(hard labels). Offline (pre-gen teacher) or on-policy. | You want to **compress** a strong teacher (Claude, or a domain expert checkpoint) into a small deployable student. | teacher logits + labels | Local LoRA |

### GRPO loss variants (`grpo_loss_type`, validated `{grpo, gspo, dr_grpo, dapo, bnpo}`)

Threaded identically into both GRPO backends and surfaced in the Training Config UI:

- **grpo** — TRL default; the safe starting point.
- **gspo** — Qwen's *sequence-level* policy optimization. **Use for long sequences and MoE models** (e.g. the 26B-A4B Gemma) where token-level GRPO is unstable.
- **dr_grpo / dapo / bnpo** — alternative objectives passed straight to TRL for when you're A/B-ing optimizer behavior.

### Orchestration layers (sit on top of the strategies)

- **Cascade RL** (`cascade_scheduler.py`) — *curriculum*. Trains one **domain at a time** (file-ops → bash → search → multi-step reasoning, or **one stage per repo**), chaining each stage's checkpoint into the next as the new base. Each stage independently picks SFT/DPO/GRPO and a domain-appropriate reward, and a **preflight refuses to start a run that can't produce a learning signal** (e.g. verification reward with no `tests`). `weakest_first` orders stages by validation loss so the model's worst domain trains first. **Use when** you want a specialist progression instead of one undifferentiated run.
- **MOPD** (Multi-domain On-Policy Distillation) — the *unify* step after a cascade. Each domain's best checkpoint is the teacher for its own data; all domain datasets combine and distill into **one generalist student**. **Use when** a cascade has produced several specialists you want folded back together.
- **RL Gym environment** (`environment.py`, `BashGymEnv`/`BatchGymEnv`) — a Gymnasium-style sandboxed env (`BASH/READ/WRITE/EDIT/SUBMIT`, verifier reward, step penalty, guardrails, PII filtering). `BatchGymEnv` runs N parallel rollouts — the trajectory source for online GRPO.
- **AutoResearch / SchemaSearch / TraceResearch** (`autoresearch.py`, `schema_search_space.py`, `trace_researcher.py`) — evolutionary *meta-optimizers* (mutate → eval → keep-better) over training hyperparameters, the data-synthesis pipeline, or the curation policy. `LossTargetedMiner` biases mining toward **high-loss (hard) examples** for the current checkpoint. **Use when** you want the platform to tune itself rather than hand-picking knobs.

### Where any strategy runs (execution backends — orthogonal to the above)

| Backend | Mechanism | **Use when…** |
|---|---|---|
| **Local** | Generates a Python script, runs it as a streamed subprocess on the auto-detected Py3.12+CUDA. Defaults tuned for 12GB (RTX 3080 Ti). | Iterating on ≤1.5B models on your own GPU. |
| **Remote SSH (DGX)** | `RemoteTrainer` over `asyncssh`: preflight → SFTP upload → `nohup` exec → stream logs → download artifacts; pause/resume/cancel via signals. | 7B+, Gemma 4, or Qwen 3.5 (needs float32) that won't fit locally. |
| **Managed fine-tune API** | `ManagedFineTuneBackend` drives a hosted upload→create→poll flow via per-platform `FineTuneDialect`. **Together** and **OpenAI** dialects wired. | **No GPU at all** — offload to a hosted provider. |
| **NeMo Gym / Customizer** | Submits a customization job to NVIDIA NeMo Microservices and polls. | You're in the NVIDIA NeMo cloud (`USE_NEMO_GYM`). |

> A degenerate-run guard is shared across local strategies: each writes an `EARLY_STOPPED` sentinel when the signal collapses (loss plateau for SFT, ≤random preference accuracy for DPO, zero reward-variance for GRPO) — so a doomed run stops instead of burning GPU.

---

## 3. How the data that feeds training stays adaptable

The training methods are only as good as the data. The Factory makes that data **source-, repo-, and dataset-agnostic**:

- **Any coding-agent trace in** — six importers behind one `source_tool`-tagged schema: **Claude Code** (incl. subagents, thinking blocks), **Gemini CLI**, **GitHub Copilot** (captures accept/reject = native DPO signal), **Hermes** (closes the deploy→retrain loop), **OpenCode**, **ChatGPT**, plus MCP tool logs. Adding a tool = one importer.
- **Any repo** — `primary_repo` propagates into example metadata; export supports **Generalist / Selected / Single-repo** training via `repo_filter`.
- **Any public dataset** — scan/rank HF datasets → `normalize_public_messages` adapts ShareGPT/OpenHands/SWE-agent/OASST shapes → **decontaminate** (zero shared 13-grams, <0.7 3-gram Jaccard vs benchmark corpora) → **mix** with self-collected traces at a target self-fraction.
- **Synthetic expansion** — NeMo Data Designer pipelines (SFT/DPO/tool-use/external/unstructured), each **LLM-as-Judge gated**, multi-provider (NVIDIA/Anthropic/OpenAI/local); plus trace-seeded generation and security-domain ingesters.
- **Quality gating** — verification + success-rate thresholds (gold ≥0.8, fail ≤0.3) + semantic-judge demotion; a 7-metric quality score (rewards reflection-after-failure even when the env caused the failure); regex+LLM PII redaction + prompt-injection detection; embedding dedup (0.95); and **session/repo-level held-out splits with a contamination assertion** so the eval set can never leak into training.
- **Three preference sources** — whole-trace DPO, embedding-matched failed↔gold DPO, and the newly-surfaced **decision-level DPO** (mine FAILURE→SUCCESS recovery pairs from *within* a gold trace).

---

## 4. Adaptability — the five axes that make this not-just-for-one-setup

1. **Model-agnostic.** A `ModelFamilyProfile` registry (Gemma4 / Qwen3 / Qwen2.5 / Llama3 / generic) is the single source of every family-specific fact — LoRA targets, tool-call format (`openai_json` / `gemma4_delimited` / `qwen_xml` / `hermes`), attention impl, correctness patches, GGUF template, stop tokens. **Adding a new open model = add one profile; trainer/export/eval code is untouched.**
2. **Provider-agnostic inference.** One `InferenceProvider` ABC behind **Anthropic, NVIDIA NIM, Ollama (local *and* remote DGX)**, and **any OpenAI-compatible endpoint** — presets for **Together, Fireworks, OpenRouter, Groq, DeepInfra, Hyperbolic, self-hosted vLLM**, or any custom `base_url`. Connect one from Settings → it's immediately usable as Student and as an eval target.
3. **Compute-agnostic training.** The same generated scripts run **local GPU**, **remote SSH (DGX Spark)**, or a **managed fine-tune API (Together/OpenAI)** — chosen per run. No DGX required.
4. **Multiple deploy targets.** GGUF → Ollama (with a **verified Modelfile template** — the fix for the #1 Gemma-4 deploy bug, guarded by a train↔serve round-trip check), HuggingFace Hub, or any served OpenAI-compatible endpoint.
5. **Portable, rigorous eval.** The held-out ship/no-ship gate (session-clustered paired bootstrap + pre-registered thresholds), forgetting suite, SERA soft scoring, pass@k, and Terminal-Bench/BFCL/SWE-bench orchestration all run against **any served endpoint** through one injected `openai_complete` seam — so the *exact same gate* judges an Ollama checkpoint, a vLLM server, or a cloud model.

---

## 5. Putting it together — a decision guide

- **"I just want a small model that imitates my workflow."** → SFT, local, 1.5B. Gate on the held-out eval before trusting it.
- **"My traces include mistakes-then-fixes."** → SFT + **decision-level DPO** (Data Creator → Quality tab) to learn the recovery.
- **"I have tests / a verifier."** → GRPO (`verification` reward) or RLVR. Long-context or MoE base → set loss to **GSPO**.
- **"I want a broad specialist across coding skills."** → **Cascade RL** (per-domain or per-repo), then **MOPD** to unify.
- **"I want to shrink Claude (or an expert checkpoint) into something I can serve."** → Distillation.
- **"I don't have a big GPU."** → same strategy, **managed fine-tune backend (Together/OpenAI)** or **remote SSH to a DGX**.
- **"I don't want to hand-tune."** → wrap any of the above in **AutoResearch / TraceResearch** and let it evolve toward lower loss on hard examples.

In every case the loop closes the same way: **train → gate on held-out (ship/no-ship) → deploy to Ollama/HF/endpoint → collect the new traces → feed them back.**

---

### Source references (entry points)
- Strategies: `bashgym/gym/trainer.py` (`TrainingStrategy`, `train_sft/dpo/grpo/rlvr/distillation`, `GRPOTrainer`)
- Orchestration: `bashgym/gym/cascade_scheduler.py` (`CascadeScheduler`, `distill_cascade`), `bashgym/gym/environment.py` (`BashGymEnv`)
- Backends: `bashgym/gym/remote_trainer.py`, `bashgym/gym/training_backends/` (`base.py`, `managed.py`)
- Model recipes: `bashgym/families/profiles.py`, `bashgym/families/backends.py`
- Data: `bashgym/trace_capture/importers/`, `bashgym/factory/`, `bashgym/datasets/`, `bashgym/research/`
- Eval/deploy: `bashgym/eval/`, `bashgym/providers/`, `bashgym/models/`, `bashgym/export/gguf.py`
