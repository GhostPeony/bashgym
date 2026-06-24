# Training Capability Map

This map is the broad view of what a BashGym user can do with open-model
training today, what each surface is for, and what evidence is required before a
trained model should be trusted.

Use it with:

```bash
bashgym training docs --topic capabilities --json
```

For concrete starting knobs, read [strategy-guide.md](strategy-guide.md). For
run diagnosis, read [metrics-runbook.md](metrics-runbook.md). For ECHO/RWML and
JEPA-style world models, read [world-models.md](world-models.md).

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
| Learn and plan | Read the training curriculum, inspect settings, generate starter plans, and decide whether SFT, DPO, GRPO, distillation, cascade, DPPO, or world-model objectives fit the data. | Training Guides UI, `docs/training/*`, `bashgym manifest`, `bashgym training plan`. | The plan JSON, selected docs, and explicit reason for the chosen strategy. |
| Build data | Convert gold traces, failed traces, custom JSONL, security datasets, synthetic tool-use data, decision-DPO pairs, and executable terminal environments into trainable artifacts. | Trace import, Data Designer, Decision DPO, Environment Lab, AutoResearch environment recipe proposals. | Dataset manifests, source split, quality labels, contamination checks, and verifier metadata. |
| Train | Run SFT, DPO, GRPO/RLVR, distillation, cascade RL, remote SSH training, managed fine-tunes, local Unsloth/plain scripts, and backend-planned DPPO/world-model smoke jobs. | Training Config, Training Dashboard, trainer API, generated scripts, remote device settings. | Config snapshots, generated scripts, logs, metrics JSONL, checkpoints, adapter/backend version. |
| Evaluate | Run heldout trace eval, local/model environment rollouts, pass@k, holdout gates, base-vs-candidate comparison, spurious-reward controls, tamper canaries, and public benchmark ingest. | Evaluator, Environment Lab, `/api/eval/*`, model registry. | Release verdict JSON, pass@k reports, holdout manifests, canary results, benchmark manifests. |
| Analyze | Combine metrics, DPPO replay, release evidence, and world-model coverage into conservative findings. | `bashgym training analyze`, Training Monitor, World-Model Quality panel. | Analysis JSON, findings, unresolved blockers, and next action. |
| Promote | Route a student only where it is proven better or good enough, with fallback to teacher/frontier models for weak domains. | Model registry, router, release gate verdicts. | Combined release gate, external benchmark evidence for broad claims, rollback path. |

---

## Training capabilities

| Capability | Status | Use when | Key knobs | Evidence that matters |
|---|---|---|---|---|
| SFT | Ready | The model needs tool-call format, repo conventions, command style, or a first local baseline. | `learning_rate`, `epochs`, `batch_size`, `gradient_accumulation_steps`, `max_seq_length`, LoRA/QLoRA settings, backend. | Eval loss plus heldout trace behavior and executable pass@k. |
| DPO | Ready | You have chosen/rejected answers for the same prompt, usually after SFT. | `dpo_beta`, LR, epochs, max length, pair filtering. | Chosen/rejected margin, preference accuracy, and no heldout regression. |
| GRPO/RLVR | Ready with evidence | You have executable verifiers and sampled attempts sometimes pass and sometimes fail. | `training_profile=terminal_rl_tmax_like`, `grpo_group_size`, DAPO/Dr. GRPO loss, active sampling, zero-std filtering, temperature. | Reward, `reward_std`, `frac_reward_zero_std`, pass@1/pass@k, timeout, tamper, verifier status. |
| Distillation | Ready | A smaller model is too weak for RL or you want to compress teacher behavior. | Teacher model, teacher temperature, distillation alpha, on-policy distillation. | Student pass@k, quality against teacher baseline, no tool-format regression. |
| Cascade RL | Ready with evidence | You need staged domain learning from easier terminal skills to harder multi-step tasks. | Domain stages, stage steps, base model, min examples, mode, remote SSH, MOPD settings. | Per-domain holdouts, stage-to-stage forgetting, final generalist holdout. |
| DPPO replay | Backend-dependent | You have served-model terminal rollouts, behavior/train logprobs, and a backend such as verl, SkyRL, or OpenRLHF. | Backend, Binary-TV/KL threshold, replay path, train-logprob enrichment, max smoke steps. | One installed-backend smoke with mask telemetry, reward, pass@k, and saved artifacts. |
| ECHO | Backend-dependent, diagnostic | You want an auxiliary observation-prediction loss over terminal outputs. | `echo_enabled`, `echo_aux_lambda`, action/observation masks, backend loss hook. | ECHO loss trend, observation coverage, heldout correlation. |
| RWML | Backend-dependent, diagnostic | You want an embedding-space next-state reward or curriculum signal. | `rwml_enabled`, distance threshold, easy-pass threshold, easy keep probability, history window, embedding model. | RWML pass rate, distance mean/p95, prediction outliers, heldout correlation. |

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
| External benchmark ingest | Ready with evidence | Public harness evidence from Harbor/Terminal-Bench, BFCL, SWE-bench, lm-eval-style forgetting, or generic JSON. | Missing manifest, harness failures, score below declared minimum, benchmark/train leakage. |
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

### Verifier-backed terminal RL

```text
executable environments -> pass@k baseline -> GRPO/RLVR -> holdout/comparison/spurious/tamper gates
```

Use this when reward groups have contrast. If `frac_reward_zero_std` is high,
fix task difficulty, reward granularity, or sampling before scaling.

### DPPO/backend path

```text
served-model rollouts -> DPPO replay JSONL -> train-logprob enrichment -> one-step backend smoke -> artifacts
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
- torchtune DPO docs: https://meta-pytorch.org/torchtune/0.6/recipes/dpo.html
- LLaMA-Factory docs: https://llamafactory.readthedocs.io/en/latest/
- LeCun, A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- Meta V-JEPA 2: https://ai.meta.com/research/vjepa/
