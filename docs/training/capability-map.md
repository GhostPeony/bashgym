# Training Capability Map

This map is the broad view of what a BashGym user can do with open-model
training today, what each surface is for, and what evidence is required before a
trained model should be trusted.

Use it with:

```bash
bashgym training capabilities --json
bashgym training docs --topic capabilities --json
```

Use `training capabilities` when an agent needs a structured matrix. Use
`training docs --topic capabilities` when a human wants the narrative map.

For concrete starting knobs, read [strategy-guide.md](strategy-guide.md). For the
end-to-end terminal RL recipe, read
[tmax-terminal-rl-recipe.md](tmax-terminal-rl-recipe.md). For run diagnosis, read
[metrics-runbook.md](metrics-runbook.md). For ECHO/RWML and JEPA-style world
models, read [world-models.md](world-models.md). For targeted self-distillation
from failed trace spans, read [session-distillation.md](session-distillation.md).

Source review date: 2026-06-24. The structured capability command includes
`source_refs` for the primary docs used to ground stack claims.

---

## Status key

| Status | Meaning |
|---|---|
| Ready | The platform has user-facing controls, backend/API support, docs, and tests or smoke evidence for the core workflow. |
| Ready with evidence | The feature is usable, but only when the user supplies the required data, verifier, or release evidence. |
| Backend-dependent | BashGym has contracts, replay, adapters, or launch planning, but an installed external trainer must still prove the path. |
| Diagnostic | The signal can inform investigation, curriculum, or release context, but must not approve shipping by itself. |

---

## Full user journey

| Stage | What users can do | Main surfaces | Evidence to keep |
|---|---|---|---|
| Learn and plan | Read the training curriculum, inspect settings, generate starter plans, and decide whether SFT, DPO, reward modeling, GRPO, distillation, Session Distillation, cascade, DPPO, or world-model objectives fit the data. | Training Guides UI, `docs/training/*`, `bashgym manifest`, `bashgym training plan`. | The plan JSON, selected docs, and explicit reason for the chosen strategy. |
| Build data | Convert gold traces, failed traces, custom JSONL, security datasets, synthetic tool-use data, decision-DPO pairs, Session Distillation records, reward examples, and executable terminal environments into trainable artifacts. | Trace import, Data Designer, Decision DPO, Environment Lab, AutoResearch environment recipe proposals. | Dataset manifests, source split, quality labels, contamination checks, verifier metadata, and target masks. |
| Train | Run SFT, DPO, GRPO/RLVR, distillation, Session Distillation, cascade RL, private compute training, managed fine-tunes, local Unsloth/plain scripts, and backend-planned DPPO/world-model smoke jobs. | Training Config, Training Dashboard, trainer API, generated scripts, private compute target settings. | Config snapshots, generated scripts, logs, metrics JSONL, checkpoints, adapter/backend version. |
| Evaluate | Run heldout trace eval, local/model environment rollouts, pass@k, holdout gates, base-vs-candidate comparison, spurious-reward controls, tamper canaries, and public benchmark ingest. | Evaluator, Environment Lab, `/api/eval/*`, model registry. | Release verdict JSON, pass@k reports, holdout manifests, canary results, benchmark manifests. |
| Analyze | Combine metrics, DPPO replay, release evidence, and world-model coverage into conservative findings. | `bashgym training analyze`, Training Monitor, World-Model Quality panel. | Analysis JSON, findings, unresolved blockers, and next action. |
| Prepare backend smoke | Prove local DPPO/ECHO/RWML handoff readiness before private or cloud GPU work. | `bashgym training smoke-bundle`, private compute checklist. | Readiness JSON, launch env, world-model probe, launch script/logs. |
| Promote | Route a student only where it is proven better or good enough, with fallback to teacher/frontier models for weak domains. | Model registry, router, release gate verdicts. | Combined release gate, external benchmark evidence for broad claims, rollback path. |

---

## Platform surfaces

Use this table to decide whether a task belongs in the CLI, API, or UI.

| Surface | Use it for | Key entrypoints |
|---|---|---|
| Agent CLI | New-session handoff, agent automation, compute-target artifact preparation, and post-run diagnosis. | `bashgym manifest --json`, `bashgym training capabilities --json`, `bashgym training plan --strategy <strategy> --json`, `bashgym replay summarize <path> --json`, `bashgym training smoke-bundle --replay <path> --output-dir <dir> --json`, `bashgym training analyze ... --json`. |
| Training API | Starting, monitoring, pausing/resuming/stopping, exporting, and inspecting runs. | `POST /api/training/start`, `GET /api/training/{run_id}`, `GET /api/training/runs`, `GET /api/training/runs/{run_id}/metrics`, `POST /api/training/export`, `POST /api/training/managed/submit`. |
| Environment API | TMax-style environment import, materialization, rollouts, pass@k, holdouts, spurious controls, and tamper canaries. | `POST /api/environments/import-jsonl`, `POST /api/environments/materialize`, `POST /api/eval/environments/passk`, `POST /api/eval/environments/local-rollout-passk`, `POST /api/eval/environments/model-rollout-passk`, `POST /api/eval/environments/holdout-gate`, `POST /api/eval/environments/holdout-comparison`. |
| Eval API | Heldout trace jobs, public benchmark ingest, release verdicts, and DPPO handoff planning. | `POST /api/eval/heldout`, `GET /api/eval/verdict/{model_id}`, `GET /api/eval/benchmark-commands`, `POST /api/eval/benchmarks/external-ingest`, `POST /api/eval/environments/dppo-replay/enrich`, `POST /api/eval/environments/dppo-replay/smoke-plan`. |
| Device and hardware API | Private compute target readiness and model fit checks before large runs. | `GET /api/devices`, `POST /api/devices/discover`, `POST /api/devices/{device_id}/preflight`, `GET /api/ssh/preflight`, `GET /api/system/info`, `GET /api/system/gpus`, `GET /api/system/recommendations`, `GET /api/models/discover`. |
| UI surfaces | Human operator education, guided configuration, manual evidence attachment, and model promotion review. | Training Monitor, Training Configuration, Training Guides, World-Model Quality panel, Factory -> Environment Lab, Evaluator -> Held-out Gate, Evaluator -> External benchmark ingest, Models -> profile/leaderboard/comparison/trends, Settings/Devices. |

---

## Metrics and recipe map

Use the metric catalog to decide whether a number is a setup contract, training
health signal, behavior signal, release blocker, or diagnostic. The CLI exposes
this as `metric_catalog`.

| Catalog | Metrics | Decision rule |
|---|---|---|
| Setup contracts | Dataset size, truncation, `contract_ready`, `optimizer_ready`, `world_model_records`. | Fix before training or before spending private compute/backend time. |
| Optimization health | Train/eval loss, grad norm, learning rate, KL, entropy. | Tune LR, warmup, epochs, sequence length, and loss weights. |
| Session Distillation health | `session_distillation_loss`, `session_distillation_kl`, `session_distillation_ce`, `session_distillation_masked_tokens`, reader confidence. | Trust only when masked loss aligns with heldout decision behavior. |
| Preference health | Chosen/rejected rewards, reward margin, preference accuracy. | Trust only when heldout behavior does not regress against the SFT base. |
| RL signal quality | Reward, `reward_std`, `frac_reward_zero_std`, verifier error rate, timeout rate. | Scale RL only when reward groups have contrast and verifier errors are low. |
| Behavior evidence | pass@1/pass@k, heldout trace delta, holdout comparison, external benchmark score. | Use these to decide whether the model is actually better. |
| Safety/release | Tamper rate, spurious controls, canary failures, verifier error rate. | Any tamper or reward-hacking signal blocks promotion. |
| World-model diagnostics | ECHO loss, RWML pass rate, embedding distance, exit-code/test-result accuracy. | Use for curriculum and diagnosis until correlated with pass@k and safety. |
| Hardware efficiency | Tokens/sec, peak GPU memory, OOM count, backend import status. | Use to size batch, sequence length, backend choice, and compute-target readiness. |

The recipe stages are:

| Stage | Operator question | Proceed when |
|---|---|---|
| Orient | What are we trying to teach: format, preference, verifier outcome, or dynamics? | A strategy and first evaluation gate are selected before training. |
| Data contract | Is the data valid for the selected strategy? | Examples, pairs, environments, or replay pass schema and split checks. |
| Local smoke | Can a tiny run write metrics and artifacts locally? | Metrics JSONL, logs, and expected artifacts exist without loader/template errors. |
| Behavior baseline | What does the base/SFT model solve before new RL or DPPO work? | Heldout trace or environment pass@k baseline is saved. |
| Training run | Did the run improve the intended signal without breaking operations? | Training health, signal quality, timeout, verifier, and OOM metrics are acceptable. |
| Release evidence | Did behavior improve on heldout tasks and controls? | Heldout, pass@k, comparison, spurious, tamper, and benchmark evidence are attached. |
| Backend smoke | Can the installed backend consume the DPPO/ECHO/RWML handoff? | Smoke bundle is ready and one installed-backend smoke saves logs/artifacts. |
| Route or iterate | Where is the student proven good enough, and where should it fall back? | Routing scope, rollback path, and next-data plan are explicit. |

---

## Training capabilities

| Capability | Status | Use when | Key knobs | Evidence that matters |
|---|---|---|---|---|
| SFT | Ready | The model needs tool-call format, repo conventions, command style, or a first local baseline. | `learning_rate`, `epochs`, `batch_size`, `gradient_accumulation_steps`, `max_seq_length`, LoRA/QLoRA settings, backend. | Eval loss plus heldout trace behavior and executable pass@k. |
| DPO | Ready | You have chosen/rejected answers for the same prompt, usually after SFT. | `dpo_beta`, LR, epochs, max length, pair filtering. | Chosen/rejected margin, preference accuracy, and no heldout regression. |
| Reward model / ORM / PRM | Ready with evidence | You need a learned scorer for reward audits, best-of-N, rejection sampling, trajectory scoring, or later RL. | `reward_artifact`, `reward_type`, `reward_loss`, reward scale, train/eval split, LoRA/QLoRA settings for real backends. | Strict reward examples, fixture smoke artifacts, heldout pair accuracy, calibration, length-bias and task-family checks. |
| GRPO/RLVR | Ready with evidence | You have executable verifiers and sampled attempts sometimes pass and sometimes fail. | `training_profile=terminal_rl_tmax_like`, `grpo_group_size`, DAPO/Dr. GRPO loss, active sampling, zero-std filtering, temperature. | Reward, `reward_std`, `frac_reward_zero_std`, pass@1/pass@k, timeout, tamper, verifier status. |
| Distillation | Ready | A smaller model is too weak for RL or you want to compress teacher behavior. | Teacher model, teacher temperature, distillation alpha, on-policy distillation. | Student pass@k, quality against teacher baseline, no tool-format regression. |
| Session Distillation | Ready with evidence | Failed trace spans show local mistakes that can be corrected with a hint without replacing the trajectory. | `session_distillation_alpha`, temperature, min confidence, reader, `target_span_only` mask. | Valid records, masked KL/CE metrics, heldout decision accuracy, tool/command validity, pass@k where available. |
| Cascade RL | Ready with evidence | You need staged domain learning from easier terminal skills to harder multi-step tasks. | Domain stages, stage steps, base model, min examples, mode, private compute target, MOPD settings. | Per-domain holdouts, stage-to-stage forgetting, final generalist holdout. |
| DPPO replay | Backend-dependent | You have served-model terminal rollouts, behavior/train logprobs, and a backend such as verl, SkyRL, or OpenRLHF. | Backend, Binary-TV/KL threshold, replay path, train-logprob enrichment, `bashgym training smoke-bundle`. | One local readiness bundle, then one installed-backend smoke with mask telemetry, reward, pass@k, and saved artifacts. |
| ECHO | Backend-dependent, diagnostic | You want an auxiliary observation-prediction loss over terminal outputs. | `echo_enabled`, `echo_aux_lambda`, action/observation masks, backend loss hook. | ECHO loss trend, observation coverage, heldout correlation. |
| RWML | Backend-dependent, diagnostic | You want an embedding-space next-state reward or curriculum signal. | `rwml_enabled`, distance threshold, easy-pass threshold, easy keep probability, history window, embedding model. | RWML pass rate, distance mean/p95, prediction outliers, heldout correlation. |

---

## Data sources and artifact contracts

| Source | Produces | Best use | Guardrail |
|---|---|---|---|
| Gold traces | Structured tool-call messages, SFT examples, DPO chosen examples. | First SFT baseline and repo/domain specialists. | Verify, deduplicate, preserve source metadata, and watch truncation. |
| Silver/bronze/failed traces | DPO rejected examples and failure analysis. | Preference learning and curriculum gaps. | Do not mix failed traces into SFT as success examples. |
| Custom JSONL | SFT/DPO-compatible message datasets. | Curated external or hand-authored data. | Validate schema, tool-call JSON strings, and manifests. |
| Security datasets | Security-specialist examples and analysis traces. | Malware, phishing, or policy-specialist behavior. | Preserve source labels and enrichment mode. |
| Synthetic Data Designer outputs | Synthetic SFT rows, DPO pairs, tool-use rows, and terminal environment proposals. | Coverage expansion and schema evolution. | Keep seed/source manifests, validators, and decontamination metadata. |
| Source Library local adapters | Local JSON/JSONL source-card records converted into SFT examples, DPO pairs, reward examples, process-reward examples, eval manifests, or environment specs. | Fixture smokes, curated public-source prep, Data Designer handoff, pass@k prep, and external review packets. | Eval-only sources block training by default; converted DPO/reward artifacts must pass strict validators, and environment specs must validate before use. |
| Terminal environments | Rollouts, verifier rewards, pass@k, holdouts, DPPO replay, and world-model payloads. | GRPO/RLVR, DPPO, release gates, and ECHO/RWML. | Require materialization, verifier-only pass, protected-file manifest, and split metadata. |

Key artifacts to preserve through the loop:

- `training_examples.jsonl`: messages, tools, metadata, source trace, quality score.
- `dpo_pairs.jsonl`: prompt identity, chosen/rejected outputs, pair source, quality labels.
- `reward_examples.jsonl`: reward type, prompt/trajectory, reward values or step rewards, label source, reward scale, split/decontamination metadata.
- `reward_eval.json`: heldout pair accuracy, calibration, reward margin, length bias, task-family breakdown, reward variance, and eval-only leakage checks.
- `session_distillation_records.jsonl`: original context, hinted context, hint, target text, target span, loss mask, reader confidence, verifier outcome, and provenance.
- `EnvironmentSpec`: id, instruction, workspace/build hints, verifier, protected-file manifest.
- `metrics.jsonl`: step, loss/reward, reward variance, pass@k, timeout/tamper, world-model metrics.
- `dppo_replay.jsonl`: environment, trajectory, reward, behavior/train logprobs, optional `world_model`.
- `backend_smoke_readiness.json`: local DPPO/ECHO/RWML handoff status before private compute/backend work.
- `release_evidence.json`: heldout verdict, environment gates, external benchmarks, diagnostic world-model quality, and diagnostic learned-reward evidence.

---

## Model, hardware, and config axes

Model-family profiles are the code-level recipe for training a base family. They
define tool-call format, LoRA targets/excludes, attention defaults, patches, and
backend fallback behavior.

| Family | Status | Tool-call format | Notes |
|---|---|---|---|
| Gemma 4 | Profiled | `gemma4_delimited` | Thinking template, multimodal excludes, Gemma-specific patch. |
| Qwen3 / Qwen3.6 | Profiled | `qwen_xml` | Family profile for current Qwen3-compatible checkpoints. Prefer the newest compatible Qwen3.6, Qwen3-Coder, or hosted Qwen3 model that fits the target backend and hardware. |
| Qwen2.5 | Profiled | `qwen_xml` | Stable coder/instruct fallback when the newest Qwen3/Qwen3.6 checkpoint is unavailable or too large. |
| Llama 3 | Profiled | `openai_json` | General instruct baseline and portable adapter experiments. |
| Generic HF causal LM | Fallback | `openai_json` | Use for quick compatibility; add a profile for production-quality support. |

Hardware choices:

| Hardware | Best for | Watch |
|---|---|---|
| Local 12 GB GPU | Fast iteration, smoke tests, small LoRA/QLoRA specialists. | OOM, truncation, large-vocab loss memory. |
| Local 24 GB GPU | Larger local adapters and longer traces. | Eval loss, adapter overfit, checkpoint size. |
| Private compute target | Larger dense/MoE targets, longer-context runs, DPPO/ECHO/RWML backend smoke. | Backend imports, CUDA/Triton compatibility, artifact sync. |
| Cloud backend | Managed fine-tunes, large batch experiments, external trainer backends. | Data egress, benchmark leakage, reproducibility. |

Config axes to decide before a run:

- `data_scope`: generalist, mixed, or specialist.
- `adapter_mode`: LoRA, QLoRA, or full fine-tune.
- `sequence_length`: 2048, 4096, or 8192+.
- `terminal_rl_sampling`: group size, rollout batch, max tool calls, active sampling, zero-std filtering.
- `world_model_objectives`: ECHO, RWML, both, or off.
- `promotion_thresholds`: heldout pass@k, comparison delta, timeout/tamper limits, external benchmark minimums.

---

## Evaluation and release capabilities

| Capability | Status | What it proves | Blockers to respect |
|---|---|---|---|
| Heldout trace eval | Ready | Candidate behavior against baseline traces without training-set reuse. | Negative trace delta, bootstrap CI including zero when improvement is required, forgetting drops. |
| Environment pass@k | Ready | Whether the agent solves executable terminal tasks across attempts. | Broken verifier, timeout surge, invalid/tampered attempts. |
| Environment holdout gate | Ready | Whether a candidate survives grouped unseen environments. | Contamination, pass@1 below threshold, timeout/tamper above threshold. |
| Holdout comparison | Ready | Whether candidate beats base on the same heldout environments. | Candidate delta too small, CI not clearing zero, operational regressions. |
| Spurious-reward control | Ready | Whether reward improvements are likely shortcut/contamination artifacts. | Candidate improves on random or spurious labels. |
| Reward-hacking canaries | Ready | Whether protected verifiers, tests, fixtures, or manifests can be exploited. | Any successful tamper or verifier shortcut. |
| External benchmark ingest | Ready with evidence | Public harness evidence from Harbor/Terminal-Bench, BFCL, SWE-bench, RewardBench/CUARewardBench, lm-eval-style forgetting, or generic JSON. | Missing manifest, harness failures, score below declared minimum, benchmark/train leakage. |
| World-model quality lane | Diagnostic | Whether ECHO/RWML quality metrics exist and improve. | Must not ship by itself; needs correlation with pass@k and safety. |

---

## Backend and stack map

| Stack | BashGym role | When to expose |
|---|---|---|
| Hugging Face TRL | Reference semantics for SFT, DPO, GRPO, PPO, reward modeling, dataset formats, and training metrics. | Default generated-script shape and docs-backed config language. |
| Unsloth | Fast local LoRA/QLoRA SFT, DPO, and GRPO iteration on constrained GPUs. | Default local backend when supported by model/hardware. |
| Plain Transformers + PEFT | Conservative fallback when Unsloth does not support the hardware/model. | Local fallback and model-family compatibility path. |
| verl | External scale-out RL backend for PPO/GRPO-style dataflows with vLLM/SGLang/FSDP/Megatron patterns. | DPPO/GRPO backend smoke candidate. |
| SkyRL | External full-stack RL backend for multi-turn, tool, and environment-driven agent workloads. | Best first candidate for terminal-agent environment integration. |
| OpenRLHF | External Ray/vLLM RLHF/RLVR backend with PPO, GRPO, RLOO, REINFORCE-style options. | Scale-out backend candidate after replay/logprob contracts are stable. |
| Axolotl | YAML-first reproducibility and broad SFT/DPO/GRPO/GDPO config surface. | Optional recipe import/export, not canonical runtime state. |
| torchtune | Transparent PyTorch baselines for SFT/DPO-style experiments. | Research baselines and minimal reproduction recipes. |
| LLaMA-Factory | Broad no-code/low-code model fine-tuning workflow. | Optional user-facing recipe export, not BashGym's terminal RL core. |

BashGym's canonical contract should stay environment/replay/eval first. External
trainers should consume those contracts rather than redefining the product
around one backend.

---

## Ecosystem methods not first-class yet

These methods are visible in current post-training stacks, but BashGym should
not overstate them as stable product workflows until they pass through the
environment/replay/eval contract.

| Method family | Where it appears | BashGym position |
|---|---|---|
| PPO | TRL, verl, OpenRLHF | Backend candidate only; keep GRPO/DPPO as the primary terminal-RL paths today. |
| RLOO / REINFORCE-family | OpenRLHF | Possible future backend algorithm family, not a current first-class BashGym workflow. |
| ORPO / KTO / IPO / SimPO | Unsloth, Axolotl | Preference ecosystem references; DPO remains the first-class preference path today. |
| GDPO / EBFT | Axolotl | Experimental recipe/import candidates; do not advertise as stable BashGym training paths. |
| Multimodal RL | SkyRL | Relevant to future multimodal agents; current BashGym gym is text/shell centered. |

---

## Recommended default paths

### First useful student

```text
gold traces -> SFT -> heldout trace eval -> environment pass@k -> conservative routing
```

Use this when there is no reliable verifier-rich RL dataset yet.

### Preference refinement

```text
SFT checkpoint + chosen/rejected pairs -> DPO -> heldout regression check
```

Use this when failures are comparable alternatives for the same prompt.

### Reward-model lane

```text
reward examples -> strict validation -> fixture smoke -> heldout reward eval -> selected-vs-random control
```

Use this when you need a learned scorer before rejection sampling, best-of-N
selection, reward audits, or policy-gradient work with a learned reward. Attach
RewardBench/CUARewardBench results through external benchmark ingest as eval-only
evidence; do not mix those benchmark sources into reward training by default.

### Verifier-backed terminal RL

```text
executable environments -> pass@k baseline -> GRPO/RLVR -> holdout/comparison/spurious/tamper gates
```

Use this when reward groups have contrast. If `frac_reward_zero_std` is high,
fix task difficulty, reward granularity, or sampling before scaling.

### DPPO/backend path

```text
served-model rollouts -> DPPO replay JSONL -> train-logprob enrichment -> smoke bundle -> one-step backend smoke -> artifacts
```

Use this only after behavior and train logprobs are observable.

### JEPA-style world-model path

```text
terminal replay with world_model payloads -> ECHO/RWML backend smoke -> diagnostic quality metrics -> heldout correlation
```

Use this for dynamics learning, curriculum mining, and future one-step
reranking. Do not use it as a release gate until correlation is proven.

---

## Minimum evidence before promotion

Before a trained open model is routed to real work:

- Heldout trace eval is not worse than the baseline.
- Environment pass@k improves or meets a declared threshold.
- Grouped holdout gate passes.
- Candidate beats base when a base comparison exists.
- Spurious-reward controls stay clear.
- Reward-hacking canaries fail closed.
- External benchmark evidence is attached for broad claims.
- Backend-smoke readiness and private/cloud compute logs are preserved when DPPO/ECHO/RWML was used.
- World-model quality, if present, is attached as diagnostic context only.
- Dataset, environment, and benchmark manifests are preserved.

---

## Sources

- Hugging Face TRL overview: https://huggingface.co/docs/trl/en/index
- TRL GRPO Trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL SFT Trainer: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO Trainer: https://huggingface.co/docs/trl/en/dpo_trainer
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- verl docs: https://verl.readthedocs.io/
- SkyRL docs: https://docs.skyrl.ai/docs
- OpenRLHF docs: https://openrlhf.readthedocs.io/
- Axolotl docs: https://docs.axolotl.ai/
- Qwen3.6 collection: https://huggingface.co/collections/Qwen/qwen36
- torchtune DPO docs: https://meta-pytorch.org/torchtune/0.6/recipes/dpo.html
- LLaMA-Factory docs: https://llamafactory.readthedocs.io/en/latest/
- LeCun, A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- Meta V-JEPA 2: https://ai.meta.com/research/vjepa/
