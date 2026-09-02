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

### AutoResearch Method Selection

The host agent chooses the scientific method. BashGym should first expose a
`method_selection` projection from the existing research decision packet. For
each method, read its runner support, evidence, missing evidence, explicit
thresholds, blocking reasons, and smallest recommended probe. A method marked
`unsupported_by_runner` is not made executable by changing a proposal label.

Do not switch methods from one aggregate score. Diagnose the learning signal:

- Use SFT when the model lacks a behavior or reasoning pattern and validated
  demonstrations cover the failing slices.
- Use DPO when the model generates plausible alternatives but ranks them
  incorrectly, and same-prompt chosen/rejected pairs have reliable labels.
- Use GRPO or RLVR only when grouped rollouts have reward contrast and the
  verifier is reliable. All-failed groups need curriculum or supervised
  warm-start data; all-correct groups need harder tasks rather than more RL.
- Use teacher distillation only when a validated teacher beats the student on
  the same suite. Use session distillation only when measured recovery traces
  show that hints or corrected continuations improve the target behavior.
- If a critical signal such as constant output or a broken evaluator is
  present, diagnose it before any training method.

Campaign thresholds must be visible to the agent. Required categories are
demonstration coverage and contamination for SFT; pair count, agreement, and
ambiguity for DPO; rollout count, success band, zero-variance groups, and
verifier errors for GRPO/RLVR; teacher gap and output acceptance for teacher
distillation; and trace count plus recovery lift for session distillation.
Numerical probe ranges are advisory until the campaign records its own values.
For verifier RL, a 5%–95% success band and at most 50% zero-standard-deviation
groups are useful initial probes, not universal launch rules.

Before changing weights, read `recommended_intervention_families` in the same
packet. It compares four intervention families without choosing for the host:

- `prompt_or_context` for instruction, output-shape, or context sensitivity;
- `retrieval_or_tool` for dynamic knowledge or missing external context;
- `weight_update` when at least one installed training method clears its
  evidence thresholds;
- `serving_optimization` when the trained and served representations may differ.

A `probe_recommended` status means the named fixed control can answer a cheaper
question before training. `eligible` means the evidence permits consideration,
not that BashGym selected or started the method. If more than one family remains
plausible, use the smallest discriminating diagnostic and let the fixed suite
decide rather than treating a category label as proof.

For a long parent-to-child fine-tuning lineage, use a `plasticity_probe` only
when the installed diagnostic capability can measure the initial and final
probe metric, retention delta, cumulative training steps and tokens, and dataset
revision count. The recipe must declare one fixed step budget, metric direction,
seed, sample scope, maximum tolerated retention drop, and minimum acceptable
adaptation-efficiency ratio. Repeat the exact recipe digest on at least two
lineage checkpoints. BashGym then reports retention regression separately from
suspected plasticity loss; it does not infer either from a terminal score, a
loss curve, or two probes with different budgets.

Separate passive evidence from active diagnostics. Fixed-suite failures,
checkpoint trajectories, deterministic data-quality summaries, and emitted
training metrics are read from completed work and should not trigger another
run. When a missing measurement would change the method decision, inspect
`research state.diagnostic_capabilities` and submit a bounded diagnostic action
only if the installed runner can plausibly answer it. The matrix is an honest
capability declaration, not a closed design registry: the agent may formulate a
novel bounded probe, and an unsupported result must remain unsupported rather
than becoming a fallback SFT, DPO, GRPO, or RLVR run.

The optional `bashgym-scientific-diagnostics` runner is deliberately narrower
than the open proposal contract. It projects five pinned aggregate evidence
types: matched fixed-budget plasticity receipts, decomposed reward-integrity
evidence with canaries, preference-integrity counts, fixed-suite teacher/
student comparisons with output-validation counts, and paired no-hint/hinted
session-recovery counts. It never reads raw training or held-out examples,
performs teacher inference, evaluates sessions, or runs a fresh generic
optimizer. Use it when those exact aggregates answer the decision. If they do
not, retain the hypothesis and install a runner that can measure it; do not
reinterpret an `unsupported` result as method ineligibility or scientific
failure.

For `teacher_gap_probe`, bind the exact evaluation suite, metric direction,
teacher and student model digests, and output-validation contract. The runner
signs the gap so positive means the teacher is better for either metric
direction, and derives acceptance from accepted/total outputs. For
`recovery_trace_probe`, bind the recovery dataset and reader contract and supply
the paired outcome table for the same cases. The runner derives the recovery
lift and its 95% lower confidence bound; do not provide a hand-authored lift.
Both probes inform readiness only. The installed training runner must still
declare the method, campaign thresholds must pass, and the trained candidate
must still clear its fixed heldout evaluation.

For generated data, distinguish deterministic verification and splitting from
generation itself. Require the receipt to bind the effective generator config
and implementation; if the provider cannot accept a generation seed, record
that generation as unseeded and treat the recipe seed as the split seed only.

Current research basis:

- [On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models (2025)](https://arxiv.org/abs/2506.04695)
  finds RLVR primarily selects existing reasoning patterns and reports that
  high-quality SFT can improve RL optimization for weaker models.
- [Reassessing the Role of Supervised Fine-Tuning in VLM Reasoning (2025)](https://arxiv.org/abs/2512.12690)
  finds the SFT-versus-RL result depends on model capacity, data scale, and
  distribution, and reports deceptive reward signals in RL experiments.
- [Supervised Reinforcement Learning (2025)](https://arxiv.org/abs/2510.25992)
  studies the regime where small models rarely sample correct RLVR solutions
  while ordinary SFT overfits rigid long demonstrations.
- [DAPO (2025)](https://arxiv.org/abs/2503.14476) uses dynamic sampling to
  remove uninformative all-correct and all-incorrect rollout groups.
- [Spurious Rewards (2025)](https://arxiv.org/abs/2506.10947) shows that RLVR
  gains can be model-specific and can occur under rewards that do not measure
  the intended behavior.
- [LLMs Gaming Verifiers (2026)](https://arxiv.org/abs/2604.15149) demonstrates
  verifier exploitation and motivates invariant or adversarial verifier
  canaries before and after RLVR.

The older defining papers remain useful for algorithm definitions, but they do
not override these newer operational findings.

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
