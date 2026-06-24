# JEPA / World-Model Plan for BashGym

Date: 2026-06-23

## Why This Matters

The Claude session added the right first substrate for JEPA-style terminal-agent learning:

- ECHO: a dense auxiliary loss that teaches the policy to predict terminal observations caused by its own actions.
- RWML: an embedding-space reward for predicting the next terminal state from history plus action.
- DPPO replay enrichment: optional `world_model` payloads with RWML transition triplets and tokenizer-free ECHO text spans.

The research takeaway from JEPA, I-JEPA, V-JEPA, and V-JEPA 2 is not "predict every byte." It is: learn a compact latent state where future-relevant structure is predictable. For BashGym, the terminal analogue is:

`task + repo/history + command -> next observation/diff/test/verifier state`

Verifiers, tests, pass@k, tamper checks, and release gates remain the authority. JEPA/ECHO/RWML should add dense representation learning, curriculum signals, and planning/reranking support around those verified rewards.

## Current Status

Done in this slice:

- `TrainingRequest` and `/api/training/start` now accept ECHO/RWML knobs and preserve enabled world-model settings in run metadata.
- `TrainerConfig.world_model_settings()` reports the full ECHO/RWML contract, including RWML easy-sample threshold.
- DPPO replay export can opt into additive `world_model` payloads and reports `world_model_records`.
- `/api/eval/environments/model-rollout-passk` exposes `include_world_model_replay` and `rwml_history_window`.
- DPPO smoke-plan API threads ECHO/RWML settings into `DPPOSmokeLaunchConfig` and backend launch env vars.
- Training Config UI now includes a setup guide for SFT, DPO, GRPO, distillation, and cascade RL.
- Training Config UI now exposes ECHO/RWML controls with starting defaults and metrics to watch.
- Environment Lab can export DPPO replay JSONL with ECHO/RWML world-model data.
- DPPO replay summaries now report replay-level ECHO/RWML coverage telemetry: world-model records, RWML transitions/history, ECHO segments, and observation/action character coverage.
- `bashgym/gym/world_model_backend.py` converts DPPO replay `world_model` payloads into backend-ready ECHO masks, cached/batched RWML scoring, TRL/verl reward hooks, and an ECHO `compute_loss` adapter.
- `docs/training/` now contains operator-facing training curriculum docs: overview, strategy guide, world-model guide, metrics runbook, glossary, and agent CLI guide.
- `bashgym` now exposes an agent-friendly CLI for manifests, training docs, starter training plans, DPPO replay summaries, and API server launch.
- `bashgym training analyze` now combines metrics JSONL, optional DPPO replay, and optional release evidence into agent-readable findings and next actions.
- Release evidence now accepts diagnostic `world_model_quality` payloads with ECHO loss, RWML pass rate, embedding-distance, and prediction-accuracy metrics.
- The Evaluator combined release verdict renders the world-model diagnostic lane without turning it into a ship/no-ship blocker.
- Training Monitor now parses backend-emitted ECHO/RWML stat dictionaries and renders a World-Model Quality panel when those metrics are present.

Still not done:

- No external verl/SkyRL/open-instruct checkout is installed in this worktree, so no live backend smoke has called the adapter inside a real trainer yet.
- No real RWML pre-RL reward training loop has been run.
- Embedding scoring has a cached/batched adapter path; real RWML still needs provider provisioning and backend-owned prediction extraction.
- The dashboard/release-evidence plumbing is ready, but no installed backend has produced real ECHO/RWML quality artifacts in this worktree yet.

## Recommended Starting Settings

SFT:

- QLoRA adapter LR: `2e-4`
- Full fine-tune LR: `2e-5`
- Epochs: `1-3`
- LoRA rank: `16` or `32`
- LoRA alpha: `rank` or `2 * rank`
- Max length: `2048` minimum for traces; `4096-8192+` for long terminal sessions

DPO:

- `beta=0.1`
- Adapter LR around `1e-5`
- Watch chosen/rejected reward margin and preference accuracy

GRPO / Terminal RL:

- Group size: `8-32`; TMax-like profile uses `32`
- Loss: `dapo` for long/variable terminal rollouts
- Enable zero-std filtering and active sampling
- Keep token-level loss and FP32 LM head available for stability
- Use RL only when verifier rewards have contrast

World model:

- ECHO: `echo_aux_lambda=0.05`
- RWML distance threshold: `0.2`
- RWML easy pass threshold: `0.8`
- RWML easy keep probability: `0.1`
- RWML history window: `4`
- Start with one-step prediction/reranking; do not trust imagined multi-step rollouts early

## Metrics to Add Next

Training metrics:

- SFT: train/eval loss, grad norm, token count, truncation warnings, held-out pass@k
- DPO: chosen/rejected rewards, margin, accuracy, chosen/rejected logprobs
- GRPO: reward, reward_std, frac_reward_zero_std, KL, entropy, timeouts, verifier errors, pass@1/pass@k

World-model metrics:

- ECHO environment-prediction loss
- ECHO observation-token coverage
- RWML binary reward/pass rate
- Embedding-distance distribution
- Easy vs hard transition retention
- Exit-code/test-result prediction accuracy
- Prediction-error outliers for curriculum mining

Safety and quality:

- Verifier tamper rate
- Reward-hacking canary pass/fail
- Unsafe command rate
- Holdout contamination status
- Base-vs-candidate holdout comparison

## Build Plan

1. Finish backend adapter smoke
   - Backend-facing adapter now tokenizes ECHO text spans, builds action/observation masks, and exposes `WorldModelTrainerAdapter.apply_echo_loss()`.
   - RWML reward construction now has cached/batched embedding support plus TRL and verl reward factories.
   - Next: run a tiny DPPO/GRPO smoke with an installed backend and save artifacts.

2. Add world-model quality metrics
   - Replay coverage summaries are now parsed and surfaced in Environment Lab.
   - Dashboard and release-evidence surfaces now accept diagnostic prediction-quality metrics: ECHO loss, RWML pass rate, embedding-distance distribution, exit-code/test-result prediction accuracy, and coverage.
   - Next: produce those metrics from an installed backend, save artifacts, and compare trends against held-out pass@k and safety metrics before using them as gates.

3. Use failed/easy rollouts better
   - Preserve zero-std RL groups for world-model learning even when policy-gradient updates drop them.
   - Use RWML prediction error to rank transitions for teacher distillation, curriculum generation, or active sampling.

4. Add curriculum routing
   - If `pass@k=0`, prefer teacher/SFT/curriculum before RL.
   - If `pass@k>0` and `pass@1` is low, prefer GRPO/DPPO.
   - If reward variance is high, use active sampling.
   - If world-model loss is high, use observation/dynamics pretraining.

5. Add one-step planning experiments
   - Generate N candidate commands.
   - Predict next latent/reward with RWML.
   - Execute only the top K candidates.
   - Compare pass@k, command count, timeout rate, and tamper rate against normal sampling.

6. Promote to release evidence only after correlation
   - Do not gate releases on world-model loss until it correlates with held-out terminal pass@k and safety metrics.
   - Initially report ECHO/RWML as diagnostics beside pass@k, holdout, spurious-reward, and benchmark evidence.

## Source Trail

- LeCun, "A Path Towards Autonomous Machine Intelligence": https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- ECHO: https://arxiv.org/abs/2605.24517
- TMax: https://arxiv.org/pdf/2606.23321
- TRL SFT: https://huggingface.co/docs/trl/en/sft_trainer
- TRL GRPO: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL DPO: https://huggingface.co/docs/trl/en/dpo_trainer
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Unsloth advanced RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
