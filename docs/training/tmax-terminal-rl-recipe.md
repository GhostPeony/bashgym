# TMax-Style Terminal RL Recipe

This recipe is the BashGym path for training open models on terminal-agent
workloads. It follows the TMax lesson: make the executable environment the unit
of training and evaluation, then keep every expensive RL step behind evidence.

Use this when you want to move from "we have traces" to "we can improve a small
open model on verified terminal tasks."

---

## Recipe Overview

```text
environment pool
  -> materialize and verify
  -> served-model rollouts
  -> pass@k and holdout gates
  -> DPPO replay JSONL
  -> train-logprob enrichment
  -> backend smoke bundle
  -> private/cloud backend one-step smoke
  -> release evidence
```

Do not skip the eval stages. RL reward can look good while the model learns a
shortcut, an easy task slice, or a verifier artifact.

The same ladder is exposed for agents:

```bash
bashgym training plan --strategy grpo --data terminal_envs --hardware private_compute --json
```

Read the `readiness_ladder` field before launching a long job and the
`adjustment_rules` field when a metric moves in the wrong direction.

---

## Operator Checkpoints

Do not treat the recipe as one long command. Treat it as checkpoints with saved
evidence at each stage.

| Checkpoint | Proceed when | Stop when |
|---|---|---|
| Orientation | The goal is explicit: SFT format learning, DPO preference refinement, GRPO/RLVR verifier optimization, DPPO backend training, or ECHO/RWML diagnostics. | The run goal is only "improve the model" with no selected gate. |
| Environment/data contract | Examples, pairs, environments, or replay records validate and preserve split/decontamination metadata. | Failed traces are mixed into SFT as wins, DPO pairs do not share prompts, or environments cannot materialize. |
| Local smoke | A tiny local run writes metrics/logs/artifacts and analyzer output is not blocked. | Loader/template errors, high truncation, verifier errors, or OOM events appear. |
| Behavior baseline | Base/SFT pass@k or heldout trace behavior is recorded before RL/DPPO changes. | There is no baseline to compare against. |
| RL signal check | `reward_std` is non-zero for useful groups and `frac_reward_zero_std` is not dominating. | All attempts fail/pass, verifier errors are high, or reward can be hacked. |
| Backend handoff | `bashgym training smoke-bundle` reports `contract_ready=true`; DPPO has train logprobs when needed. | Replay/logprob/world-model coverage is incomplete. |
| Backend smoke | A one-step installed-backend run saves logs, metrics, and launch env artifacts. | Backend imports fail, CUDA/Triton mismatch appears, or artifacts cannot be synced back. |
| Release evidence | Heldout, pass@k, holdout comparison, spurious controls, tamper canaries, and relevant public benchmarks are attached. | Loss/reward improved but behavior gates did not. |

---

## 1. Build An Environment Pool

Sources:

- Imported TMax/Harbor-style JSONL.
- BashGym Data Designer terminal environment proposals.
- Hand-authored `EnvironmentSpec` tasks.
- Converted traces that include a workspace, command path, and verifier.

Required evidence:

- Environment ids are stable.
- Instructions are standalone.
- Build/materialization succeeds.
- Verifier command can pass on a known-good solution.
- Protected-file manifest covers verifier, tests, private fixtures, and `env.json`.
- Split/decontamination metadata is preserved.

Watch:

- Domain balance.
- Skill balance.
- Fixture/verifier-kind balance.
- Estimated difficulty: too easy, learnable, too hard.

---

## 2. Establish Pass@k Before Training

Run local/scripted or served-model rollouts before RL:

```text
environment -> attempts -> verifier -> pass@1/pass@k report
```

You need at least some reward contrast. If every attempt fails, use SFT,
distillation, or easier curriculum. If every attempt passes, keep those tasks for
regression and train on harder ones.

Required evidence:

- `pass@1` and `pass@k`.
- Timeout rate.
- Verifier error rate.
- Tamper/protected-file status.
- Raw observations preserved for audit and ECHO.

---

## 3. Export DPPO Replay

For terminal RL, replay is the backend handoff contract. It should include:

- Prompt and environment spec.
- Commands and terminal observations.
- Verifier status and reward.
- Behavior-policy logprobs when available.
- Optional `world_model` payloads for ECHO/RWML.

For ECHO/RWML, export with world-model replay enabled. Replay stores role-tagged
text spans and RWML transition triplets; the trainer backend tokenizes with the
real model tokenizer.

Check replay quickly:

```bash
bashgym replay summarize data/dppo_replay/latest.jsonl --json
```

---

## 4. Enrich Train Logprobs

DPPO compares behavior-policy and train-policy logprobs. If the replay says
`train_logprob_replay_required_records > 0`, run train-policy logprob replay
before optimizer updates.

Required evidence:

- `behavior_logprobs_ready_records == records`
- `train_logprobs_ready_records == records`
- DPPO mask telemetry exists after enrichment.

---

## 5. Prepare The Backend Smoke Bundle

Run the local readiness bundle:

```bash
bashgym training smoke-bundle \
  --replay data/dppo_replay/latest.jsonl \
  --output-dir data/backend-smokes/latest \
  --base-model <operator-selected-trainable-model> \
  --backend auto \
  --rwml-embedding-model qwen3-embedding \
  --json
```

Interpretation:

| Field | Meaning |
|---|---|
| `contract_ready` | Replay, behavior logprobs, ECHO spans, and RWML targets are shaped correctly. |
| `optimizer_ready` | Train-policy logprobs are ready for DPPO optimizer updates. |
| `backend_launch_ready` | A backend command/script is runnable from the current environment. |

If `contract_ready=false`, fix replay. If `optimizer_ready=false`, enrich
train-policy logprobs. If only `backend_launch_ready=false`, move to backend or
compute-target setup.

---

## 6. Run One Tiny Backend Smoke

Use private/cloud compute only after the local bundle is clean enough to justify
the run.

The first backend smoke should be tiny:

```text
max_steps=1
train_batch_size=1
val_batch_size=1
one replay file
save every step
```

Required artifacts:

- `backend_smoke_readiness.json`
- `dppo_replay_summary.json`
- `world_model_backend_probe.json`
- `dppo_launch_env.json`
- backend stdout/stderr log
- backend metrics JSONL or trainer state
- checkpoint/output directory
- before/after pass@k if the smoke produces a candidate

Watch:

- Backend starts and reads replay.
- ECHO loss hook can run when enabled.
- RWML reward hook can score predictions when enabled.
- DPPO mask telemetry appears.
- No verifier/tamper shortcuts appear.

---

## 7. Release Gate After Training

A trained model is not promoted from RL metrics alone.

Required release evidence:

- Heldout trace eval is not worse than baseline.
- Environment pass@k improves or meets the declared threshold.
- Grouped holdout gate passes.
- Holdout comparison beats base when a base exists.
- Spurious-reward control stays clear.
- Reward-hacking canaries fail closed.
- External benchmark evidence is attached for broad claims.
- ECHO/RWML metrics remain diagnostic unless correlated with heldout pass@k and safety.

---

## Starter Settings

| Area | Starter |
|---|---|
| Training profile | `terminal_rl_tmax_like` |
| GRPO group size | `8` for cheap smoke, `32` for serious terminal RL |
| Loss | `dapo` or backend equivalent |
| Active sampling | enabled |
| Zero-std filtering | enabled for policy updates |
| FP32 LM head | enabled when supported |
| Max tool calls | `64` for serious runs, lower for smoke |
| ECHO lambda | `0.05` |
| RWML distance threshold | `0.2` |
| RWML history window | `4` |

The defaults are starting points. The release gate decides whether they worked.

---

## When To Spend Private Or Cloud Compute

Use larger compute only when the cheap evidence is already clean:

| Evidence | Required before private/cloud compute |
|---|---|
| Environment pool | Materialization and verifier-only checks pass. |
| Rollout contrast | Served attempts include both passing and failing groups, or the run is explicitly a contract-only smoke. |
| Replay | `bashgym replay summarize <replay.jsonl> --json` shows records and behavior logprob coverage. |
| Train logprobs | `train_logprob_replay_required_records` is zero, or enrichment has been run. |
| World-model payloads | ECHO observation chars and RWML transitions are non-zero when enabled. |
| Smoke bundle | `contract_ready=true`; `optimizer_ready=true` for real DPPO optimizer updates. |

If only `backend_launch_ready=false`, the private/cloud target may still be the
right next step because the backend can be installed there. If
`contract_ready=false`, fix the local artifact first.
