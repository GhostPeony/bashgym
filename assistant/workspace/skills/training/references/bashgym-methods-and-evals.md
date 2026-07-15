# BashGym Training Methods And Eval Gates

Use this reference when Hermes needs to choose a training method, classify an existing run, or decide which evals are mandatory before promotion.

## Evidence Standard

A run is evidence-bearing only when it has:

- `run_state.json` or an API training response with the exact config.
- `metrics.jsonl` with method-relevant metrics.
- `model_profile.json` or equivalent registry profile.
- Method-specific heldout, environment, reward, replay, release-evidence, smoke-bundle, or RunCard artifacts.

Train loss alone is not sufficient. If `eval_loss` is null and `evaluation_history`, `heldout_evals`, and `environment_holdout_evals` are empty, report the run as trained but unevaluated.

For every direct strategy, store the exact `checkpoint_limit`, `artifact_retention`, `auto_push_hf`, `hf_repo_name`, `hf_private`, and `hf_upload_artifact` values with the run. Routine experiments default to one checkpoint during training, `adapter_only` after success, no automatic Hub push, private visibility, and automatic merged-then-adapter selection if upload is later enabled. Read `bashgym-launch-recipes.md` for exact payloads.

## Method Matrix

### SFT

Use for imitation from gold traces, curated messages, or teacher outputs.

Start through `POST /api/training/start` with `strategy: "sft"`. On a configured private compute target, use `use_remote_ssh: true`, `load_in_4bit: false`, and an explicit `sft_backend` (`unsloth` for known-good smoke, `plain` fallback).

Watch metrics: train loss, eval loss if validation data exists, grad norm, learning rate, samples processed, tokens/sec, VRAM/GPU utilization when emitted.

Required eval: heldout trace behavior. If the model will operate in terminal environments, also run environment pass@k and holdout gate. Attach metrics and release evidence to a RunCard before promotion.

### DPO

Use for chosen/rejected preference pairs. Do not use DPO for raw gold traces without preference pairs.

Start through `POST /api/training/start` with `strategy: "dpo"` and valid preference-pair artifacts. Tune `dpo_beta` conservatively.

Watch metrics: preference loss, reward margin, pair accuracy, train/eval divergence, and heldout pair accuracy.

Required eval: strict preference-pair validation, heldout preference behavior, and no regression on heldout trace behavior. Attach preference-pair evidence to the RunCard for promotion.

### GRPO / RLVR / Terminal RL

Use for verifier-scored terminal behavior and grouped rollouts. Use RLVR when the reward is a deterministic verifier or executable success criterion.

Start through `POST /api/training/start` with `strategy: "grpo"` or `strategy: "rlvr"`, plus a non-default terminal profile such as `training_profile: "terminal_rl_tmax_like"`. Use `grpo_reward_mode: "verification"` for verifier reward, `grpo_group_size` for grouped samples, and enable `filter_zero_std_groups` plus `active_sampling` for useful reward variance.

Watch metrics: reward mean/std, zero-std group fraction, accepted groups, pass@k, tool-call failures, KL or divergence if available, and rollout length.

Required eval: model rollout pass@k, environment holdout gate, reward-hacking canaries, and release evidence. A local generated rollout dataset is not DPPO by itself.

### Session Distillation

Use for failed or recovery-rich sessions where a reader inserts a hint and the same target span is re-scored under the hinted context. This is not classic teacher distillation and not DPO.

Start through `POST /api/training/start` with `strategy: "session_distillation"` and a valid `session_distillation_records.jsonl`. Keep `session_distillation_mask_policy: "target_span_only"` unless a documented implementation supports more.

Watch metrics: `session_distillation_loss`, `session_distillation_kl`, `session_distillation_ce`, `session_distillation_masked_tokens`, and heldout recovery behavior.

Required eval: record validation, masked-token metrics, heldout recovery decisions, and terminal pass@k if used in terminal workflows. Attach session-distillation records and metrics to the RunCard.

### DPPO Replay / Backend Smoke

Use for trajectory replay with action-logprob evidence and DPPO backend readiness. Current API fields include DPPO config knobs, but DPPO is not a direct `/api/training/start` strategy in the same sense as SFT/DPO/GRPO.

Flow:

1. Generate or capture terminal rollouts with `POST /api/eval/environments/model-rollout-passk`.
2. Include `capture_logprobs: true` and `dppo_replay_output_path` when producing replay records.
3. Enrich with `POST /api/eval/environments/dppo-replay/enrich`.
4. Build a smoke plan with `POST /api/eval/environments/dppo-replay/smoke-plan`.
5. Run `bashgym training smoke-bundle ...` or the selected installed backend.

Watch metrics: replay validity, action mask fraction, logprob coverage, binary TV/KL thresholds, pass@k before/after, and backend smoke verdict.

Required eval: smoke-bundle readiness plus heldout pass@k/safety comparison. Do not promote a DPPO claim without replay and backend evidence.

### ECHO / RWML

Use as diagnostic world-model auxiliary objectives for observation/state prediction and replay filtering. These metrics are not standalone release evidence.

Enable through `echo_enabled`, `rwml_enabled`, and related RWML thresholds in a method that supports the world-model hooks.

Watch metrics: `echo_loss`, `echo_observation_chars`, `rwml_transitions`, `rwml_pass_rate`, embedding distance, and KL if used.

Required eval: correlation with heldout pass@k, fewer environment failures, or safety improvement. If this correlation is missing, report ECHO/RWML as diagnostic only.

### Reward Model / ORM / PRM

Use for learned reward scoring, outcome reward models, or process reward models. Do not confuse this with GRPO verifier reward.

Use BashGym reward-model validators and eval commands before any training claim. Required artifacts include strict reward examples and `reward_eval.json`.

Watch metrics: heldout pair accuracy, calibration, bias/leakage checks, reward margin, and agreement with executable success where available.

Required eval: strict reward example validation plus heldout reward eval. Promotion requires reward evidence in the RunCard.

### Cascade / MOPD

Use when routing domain-specialized stages or teachers across a multi-stage pipeline.

Treat every stage as a separate method-bearing run with its own data contract and eval gate. Promotion requires per-stage RunCards, final heldout behavior, and no regression in broader routing or safety.

## Classifying Existing Runs

Use these rules when inspecting `data/models/<run_id>`:

- `strategy: "sft"` and `sft_backend: "unsloth"`: BashGym-managed SFT through Unsloth, not RL.
- `training_profile: "default"`: not a TMax-style terminal-RL profile.
- `grpo_*` defaults in an SFT config do not make the run GRPO.
- `echo_enabled: false` and `rwml_enabled: false`: no world-model objective was active.
- Empty heldout/eval arrays in `model_profile.json`: no post-run eval evidence is attached.
- Missing `metrics.jsonl`: no step-level metrics artifact; use only as failed/coarse status evidence.
- A failed run with only `last_metrics` in `run_state.json` is not promotion evidence.

## Minimum Next Steps After Any Training Run

1. Run `bashgym training analyze --run-id <run_id> --models-dir data/models --json`.
2. Inspect `data/models/<run_id>/run_state.json`, `metrics.jsonl`, and `model_profile.json`.
3. Run the method-specific heldout/environment/reward/replay evals listed above.
4. Create or update `data/models/<run_id>/run_card.json`.
5. Validate the RunCard with `--promotion` before calling a model ready for routing or deployment.
