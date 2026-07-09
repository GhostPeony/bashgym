# BashGym Training Platform Button-Up Plan

Date: 2026-06-23

## Executive read

BashGym now has the right platform shape for open-model agent training:

- Training setup for SFT, DPO, GRPO/RLVR, distillation, cascade RL, and terminal-RL stability profiles.
- Executable terminal environments, local/model rollouts, pass@k, holdout gates, paired comparisons, spurious-reward controls, reward-hacking canaries, and external benchmark ingest.
- Agent-facing docs and CLI commands for training setup, replay summaries, and training analysis.
- JEPA-style ECHO/RWML world-model contracts wired through config, replay export, dashboard metrics, backend adapters, and diagnostic release evidence.

The important truth boundary:

- SFT, DPO, local GRPO script generation, terminal environment eval, release gates, and docs/CLI are real product surfaces.
- DPPO is real replay/planning/math plumbing, but not yet proven by a live installed backend smoke.
- ECHO/RWML is real config/replay/adapter/dashboard plumbing, but not yet proven by a live installed backend loop that logs quality metrics.
- World-model metrics must stay diagnostic-only until they correlate with heldout pass@k, timeout, tamper, and external benchmark outcomes.

## Full platform spread

### What a user can do on BashGym

1. Learn and plan a run
   - Use `docs/training/overview.md`, `docs/training/strategy-guide.md`, `docs/training/metrics-runbook.md`, `docs/training/world-models.md`, and `docs/training/glossary.md`.
   - Use `bashgym manifest --json`, `bashgym training docs --topic <topic> --json`, and `bashgym training plan --strategy <strategy> --json`.
   - Use the Training Guides UI in the Training dashboard.

2. Build training data
   - Train from successful gold traces.
   - Keep failed or weaker traces for DPO rejected examples and failure analysis.
   - Import custom JSONL datasets.
   - Generate tool-use and terminal-environment datasets through Data Designer.
   - Mine decision-level DPO pairs.
   - Generate or optimize executable environment recipes through AutoResearch.

3. Train models
   - SFT: teach tool-call format, terminal behavior, repo conventions, and successful workflow imitation.
   - DPO: refine from chosen/rejected pairs for the same prompt.
   - GRPO/RLVR: optimize against verifier rewards when sampled attempts have reward contrast.
   - Distillation: use teacher outputs to warm up a weaker local model.
   - Cascade RL: train domain stages, then merge/distill domain expertise.
   - DPPO replay path: export scored terminal rollouts, enrich train logprobs, plan backend smoke runs.
   - ECHO/RWML world-model path: export enriched replay and pass settings to backend adapters.

4. Evaluate and gate
   - Heldout trace eval.
   - Environment pass@1/pass@k.
   - Deterministic grouped holdout gates.
   - Base-vs-candidate paired bootstrap comparison.
   - Spurious-reward negative controls.
   - Reward-hacking canaries and protected-file tamper checks.
   - External benchmark command generation and result ingest for Harbor/Terminal-Bench, BFCL, SWE-bench, lm-eval-style forgetting, and generic harness JSON.
   - Combined release verdicts with model-registry recording.

5. Operate the flywheel
   - Use AutoResearch for hyperparameters, trace curation, schema/data generation, and environment recipe proposals.
   - Use `bashgym training analyze` to combine metrics, replay summaries, release evidence, and world-model coverage into next actions.
   - Route conservatively: only promote a student where heldout and executable evidence prove it is better.

## Capability matrix

| Capability | Current status | User promise | Proof to require next |
|---|---|---|---|
| SFT | Implemented via generated Unsloth/plain Transformers scripts and API/UI/CLI config. | Build the first useful local student from gold traces. | Eval loss plus heldout trace and environment pass@k. |
| DPO | Implemented via generated Unsloth/plain scripts and decision-pair tooling. | Improve preferences and failure recovery after SFT. | Preference accuracy, reward margin, and no heldout regression. |
| GRPO/RLVR | Implemented with terminal RL profile, active sampling, zero-std filtering, DAPO/loss variants, and telemetry. | Optimize verifiable terminal outcomes when rewards have contrast. | Reward variance, pass@k, timeout/tamper rates, holdout gates. |
| Distillation | Implemented as a training strategy and cascade/flywheel ingredient. | Warm up weaker models before RL or compress teacher behavior. | Student pass@k against teacher-created tasks and baseline comparisons. |
| Cascade RL | Implemented for domain-staged training and MOPD distillation path. | Train easier skills before harder multi-step domains. | Per-domain holdouts plus final generalist holdout. |
| DPPO | Replay export, logprob enrichment, Binary-TV/KL math, backend selection, and smoke-plan generation are implemented. | Advanced terminal rollout optimization with trust-region masks. | One real installed-backend smoke with saved mask telemetry and before/after pass@k. |
| JEPA/ECHO/RWML | Config, replay payloads, adapter hooks, dashboard parsing, docs, and diagnostic release evidence are implemented. | Auxiliary terminal-dynamics learning and future one-step reranking. | Backend smoke that logs ECHO loss/RWML quality and shows correlation with heldout behavior. |
| Environment Lab | Implemented for import, materialization, local/model rollouts, pass@k, holdouts, canaries, spurious controls, comparison, and DPPO replay. | Turn traces and generated tasks into executable training/eval units. | Keep smoke artifacts and benchmark manifests pinned by version. |
| External benchmarks | Command generation and result ingest implemented. | Attach public harness evidence without mixing benchmark tasks into training. | Official harness runs with exact versions/manifests. |
| Education/CLI | Implemented and verified with JSON output. | Let agents and users understand settings without reading code first. | Add a single capability map page that labels stable vs experimental surfaces. |

## External research mapping

Hugging Face TRL is the best reference API for local SFT, DPO, GRPO, PPO, reward modeling, and related post-training methods. BashGym should keep its generated scripts close to TRL semantics, especially for dataset formats, assistant-only loss, DPO beta/loss variants, GRPO reward metrics, KL/entropy, and loss variants.

Unsloth is the practical low-VRAM/local training path. It supports fast LoRA/QLoRA training, DPO/ORPO/KTO/PPO-style preference work, and GRPO recipes with DAPO/Dr. GRPO-style options. BashGym should continue to use it as the local iteration backend while exposing plain Transformers fallback for unsupported hardware.

verl, SkyRL, and OpenRLHF are the credible scale-out RL lanes. They matter when rollout generation, vLLM/SGLang serving, async execution, Ray clusters, custom environments, or multi-turn tool workflows dominate. BashGym should not duplicate these stacks; it should export clean replay/environment contracts and adapters.

Axolotl, torchtune, and LLaMA-Factory are useful adjacent recipe surfaces:

- Axolotl: YAML-first reproducibility and broad post-training method support.
- torchtune: transparent PyTorch baselines for SFT/DPO experiments.
- LLaMA-Factory: broad no-code/low-code training workflows and model coverage.

JEPA's applicable idea is compact predictive state learning, not replacing BashGym's language model with a latent-only agent. For terminal agents, the useful first move is:

```text
state_t + action_t -> latent(next observation / diff / test state / verifier state)
```

Use that for diagnostics, confidence, curriculum mining, and one-step candidate reranking before letting it shape RL rewards.

## Recommended settings to expose

### SFT

- `strategy=sft`
- `learning_rate=2e-4` for clean QLoRA adapter SFT; `2e-5` or lower for conservative/full fine-tune.
- `epochs=1-3`
- `batch_size=1`, `gradient_accumulation_steps=8` as a safe local baseline.
- `max_seq_length=2048` minimum; prefer `4096-8192+` for long terminal traces when hardware allows.
- `lora_rank=16 or 32`, `lora_alpha=rank or 2*rank`, `lora_dropout=0.05`.
- Watch: train/eval loss, grad norm, truncation, heldout pass@k.

### DPO

- `strategy=dpo`
- `dpo_beta=0.1`
- `learning_rate=5e-6 to 1e-5`
- `epochs=1-2`
- Watch: chosen/rejected rewards, margin, preference accuracy, heldout regression.

### GRPO/RLVR

- `strategy=grpo`
- `training_profile=terminal_rl_tmax_like`
- `grpo_group_size=8-32`; use `32` for serious terminal RL when compute allows.
- `grpo_loss_type=dapo`
- `filter_zero_std_groups=true`
- `active_sampling=true`
- `token_level_loss=true`
- `lm_head_fp32=true`
- `temperature=0.8-1.0` for exploration, then tune down if unstable.
- Watch: reward, `reward_std`, `frac_reward_zero_std`, pass@1/pass@k, timeout rate, verifier error rate, tamper rate, KL/entropy when available.

### DPPO

- Export replay first.
- Require behavior logprobs and train-policy logprob replay.
- Start with `max_steps=1`.
- Use Binary-TV threshold `0.15` or Binary-KL threshold `0.05` as current BashGym defaults.
- Watch: behavior/train logprob readiness, masked update fraction, policy mismatch, reward, pass@k.

### ECHO/RWML

- `echo_enabled=true`
- `echo_aux_lambda=0.05`
- `rwml_enabled=true`
- `rwml_distance_threshold=0.2`
- `rwml_easy_pass_rate_threshold=0.8`
- `rwml_easy_keep_probability=0.1`
- `rwml_history_window=4`
- Watch: replay coverage, ECHO loss, RWML pass rate, embedding-distance mean/p95, exit-code/test-result prediction accuracy, and correlation with heldout pass@k.

## Button-up action plan

### P0 - Make capability status unambiguous

1. Add a stable/experimental/requires-backend capability map in docs and Training Guides.
2. Add UI status copy beside DPPO and ECHO/RWML controls:
   - "Replay and adapter support is wired."
   - "Requires installed backend smoke before production use."
3. Add a backend health card for the frontend's API target so "failed to fetch" points users to `bashgym serve --host 127.0.0.1 --port 8002` or the configured backend URL.
4. Add CLI command examples to the app guide panel so users know how to inspect plans without guessing settings.

### P1 - Prove DPPO on one installed backend

1. Pick one backend first: SkyRL for agent environments or verl/OpenRLHF for broader RL infra.
2. Create an isolated checkout/worktree or external directory for the backend.
3. Generate a tiny replay from 2-5 executable environments.
4. Enrich train logprobs.
5. Run a one-step backend smoke.
6. Save artifacts:
   - backend command/config
   - replay manifest
   - mask telemetry
   - reward/pass@k before and after
   - logs and failure modes
7. Only then promote DPPO from "planned backend" to "validated backend".

### P1 - Prove ECHO/RWML quality metrics

1. Use the same tiny replay with `include_world_model_replay=true`.
2. Wire `WorldModelTrainerAdapter.apply_echo_loss()` into the chosen backend loss path.
3. Wire RWML reward through the backend reward-function interface.
4. Log:
   - `echo_loss`
   - `rwml_pass_rate`
   - `embedding_distance_mean`
   - `embedding_distance_p95`
   - `exit_code_accuracy`
   - `test_result_accuracy`
5. Attach `world_model_quality` as diagnostic release evidence.
6. Compare against pass@k and safety before using any world-model metric as a gate.

### P1 - Improve presets and onboarding

1. Add presets:
   - local 12 GB
   - local 24 GB
   - remote/DGX
   - quick smoke
   - serious terminal RL
2. For each preset, show expected cost/compute and what metrics must move.
3. Keep beginner controls simple and move backend-specific YAML overrides into an expert section.

### P2 - Make experiments comparable

1. Add a run comparison view:
   - config diff
   - data/replay manifest
   - train metrics
   - environment pass@k
   - holdout verdicts
   - external benchmarks
   - world-model diagnostics
2. Add a "promote candidate" workflow that refuses promotion without release evidence.

### P2 - Make reward authoring first-class

1. Add reward component templates:
   - verifier pass/fail
   - partial test progress
   - command count
   - timeout penalty
   - unsafe/tamper penalty
   - style/format checks
2. Show reward breakdowns per rollout group.
3. Keep verifiers authoritative over LLM judges wherever executable checks exist.

### P2 - Add optional recipe import/export

1. Export backend-neutral training recipes.
2. Provide optional adapters for Axolotl YAML, LLaMA-Factory configs, and torchtune baselines.
3. Keep BashGym's canonical source of truth as the environment/replay/eval contract.

## Verification checklist

Already re-verified during this planning pass:

```text
python -m bashgym.cli manifest --json
python -m bashgym.cli training plan --strategy grpo --hardware dgx --data terminal_envs --json
python -m bashgym.cli training plan --strategy world-model --json
python -m pytest tests/cli/test_cli.py tests/api/test_training_schema.py tests/gym/test_world_model_config.py tests/gym/test_world_model_trainer_adapter.py -q -o addopts=
```

Result: 23 passed, 2 skipped, with existing Pydantic/FastAPI deprecation warnings.

Before calling the larger agenda complete:

- Backend health UX exists.
- Capability map says what is stable vs experimental.
- One DPPO backend smoke is recorded.
- One ECHO/RWML backend smoke is recorded.
- World-model quality remains diagnostic until correlation is proven.
- A candidate model can be evaluated through heldout, environment, safety, and external evidence in one release verdict.

## Suggested handoff prompt for a new session

Continue BashGym training-platform button-up from `tasks/training-platform-button-up-plan-2026-06-23.md`.

Start by reading:

- `tasks/todo.md`
- `docs/training/overview.md`
- `docs/training/strategy-guide.md`
- `docs/training/world-models.md`
- `docs/training/metrics-runbook.md`
- `bashgym/gym/dppo_launcher.py`
- `bashgym/gym/world_model_backend.py`
- `frontend/src/components/training/TrainingConfig.tsx`

Goal:

1. Add stable/experimental/requires-backend status to the training docs/UI.
2. Add a backend-health affordance so failed fetches tell the user whether the API is down.
3. Pick one installed backend path for a real DPPO plus ECHO/RWML smoke.
4. Record artifacts and update `tasks/todo.md` with exactly what is now proven.

Do not claim DPPO or JEPA training is production-ready until an installed backend smoke exists.

## Sources

- Hugging Face TRL docs: https://huggingface.co/docs/trl/en/index
- TRL SFT Trainer: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO Trainer: https://huggingface.co/docs/trl/en/dpo_trainer
- TRL GRPO Trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- Unsloth docs: https://unsloth.ai/docs
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Unsloth GRPO tutorial: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
- Unsloth preference optimization: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto
- verl docs: https://verl.readthedocs.io/
- SkyRL docs: https://docs.skyrl.ai/docs
- OpenRLHF docs: https://openrlhf.readthedocs.io/
- Axolotl docs: https://docs.axolotl.ai/
- torchtune DPO docs: https://meta-pytorch.org/torchtune/0.6/recipes/dpo.html
- LLaMA-Factory docs: https://llamafactory.readthedocs.io/en/latest/
- Yann LeCun, A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- Meta I-JEPA explainer: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
- Meta V-JEPA 2: https://ai.meta.com/research/vjepa/
- V-JEPA 2 repo: https://github.com/facebookresearch/vjepa2
