# JEPA Training Docs Handoff

Date: 2026-06-23
Promoted from: `.claude/scratchpad/handoffs/jepa-training-docs-plan.md`

## Purpose

This handoff preserves the session-specific context behind the JEPA/ECHO/RWML
training education work. The durable implementation plan lives in
`tasks/jepa-bashgym-action-plan-2026-06-23.md`, and the hardware/backend handoff
lives in `tasks/jepa-worldmodel-hardware-handoff-2026-06-23.md`.

## What Was Completed

- Researched JEPA/Yann LeCun context, Hugging Face TRL guidance, Unsloth RL
  guidance, and BashGym-specific world-model applications.
- Added product guidance for SFT, DPO, GRPO, distillation, cascade training,
  effective batch size, LoRA settings, max sequence length, and RL group size.
- Added ECHO/RWML controls to training configuration and threaded them through
  backend API schemas, frontend request types, training store metadata, DPPO
  replay export, and DPPO smoke-plan config.
- Created the beginner-facing training curriculum in `docs/training/`, including
  overview, strategy guide, world-model guide, metrics runbook, glossary, and
  agent CLI guide.
- Added the `bashgym` CLI surface for manifests, training docs, starter plans,
  DPPO replay summaries, run analysis, and server launch.
- Added diagnostic world-model release evidence and dashboard parsing for
  ECHO/RWML quality summaries.

## Product Framing

JEPA should be framed as latent predictive world-model learning, not "predict
every raw terminal byte." For BashGym, the useful transition is:

`task + repo/history + command -> next observation/diff/test/verifier state`

Verifiers, tests, pass@k, tamper checks, and release gates remain the source of
truth. ECHO/RWML are diagnostics and auxiliary learning signals until real
backend runs prove correlation with held-out behavior and safety.

## Starting Settings To Preserve

| Area | Starter Setting |
| --- | --- |
| ECHO | `echo_aux_lambda=0.05` |
| RWML distance | `rwml_distance_threshold=0.2` |
| RWML easy pass | `rwml_easy_pass_threshold=0.8` |
| RWML easy keep | `rwml_easy_keep_probability=0.1` |
| RWML history | `rwml_history_window=4` |
| DPO beta | `0.1` |
| QLoRA LR | `2e-4` |
| Full fine-tune LR | `2e-5` |
| GRPO group size | `8-32`, with TMax-style experiments using `32` |

## Remaining Engineering Plan

1. Install or point at one real DPPO/GRPO backend checkout.
   - SkyRL is the likely first choice for terminal/ECHO work.
   - TRL is simplest for scalar RWML reward experiments.
   - verl is useful when distributed reward workers and scale matter.
2. Wire `WorldModelTrainerAdapter.apply_echo_loss()` into the backend loss path.
3. Use the TRL/verl RWML reward factories where the backend supports scalar
   reward hooks.
4. Capture real quality artifacts: ECHO loss, RWML pass rate, embedding-distance
   distribution, exit-code/test-result prediction accuracy, and prediction-error
   outliers.
5. Run a small backend smoke on the GX10 or another configured GPU target.
6. Compare world-model quality metrics against held-out pass@k, command count,
   timeout rate, and safety metrics before promoting them from diagnostics to
   gates.

## Agent Restart Checklist

1. Read `tasks/jepa-bashgym-action-plan-2026-06-23.md`.
2. Read `tasks/jepa-worldmodel-hardware-handoff-2026-06-23.md`.
3. Run `bashgym manifest --json` to discover the agent-facing surface.
4. Run `bashgym training docs --topic overview --json`.
5. Run `bashgym training plan --strategy world-model --json`.
6. Probe the local environment before assuming verl, SkyRL, OpenRLHF, or TMax
   backends are installed.

## Verification Snapshot From The Source Session

- Focused backend slice: 57 passing tests before world-model replay telemetry.
- World-model replay telemetry: 10 passing focused eval tests.
- Backend-facing world-model adapter: 81 passing tests, 1 skipped.
- Agent CLI/docs: 21 passing focused tests plus clean markdown link checks.
- Run analysis CLI: 28 passing focused tests.
- World-model quality diagnostics: 54 passing focused tests plus clean frontend
  typecheck/lint.
- Trainer-facing hooks: 33 passing focused gym tests, 3 skipped.
- No live external backend smoke was completed because no verl, SkyRL, or
  TMax/open-instruct checkout was installed in that session.

## Source Trail

- LeCun, "A Path Towards Autonomous Machine Intelligence":
  https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- ECHO: https://arxiv.org/abs/2605.24517
- TMax: https://arxiv.org/pdf/2606.23321
- TRL SFT: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO: https://huggingface.co/docs/trl/en/dpo_trainer
- TRL GRPO: https://huggingface.co/docs/trl/en/grpo_trainer
- Unsloth RL guide:
  https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
