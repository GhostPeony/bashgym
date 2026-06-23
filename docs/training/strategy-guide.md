# Training Strategy Guide

Use this guide when you are choosing a training strategy or setting up a run in
the Training Config panel. The goal is practical: pick the simplest strategy that
has the data and reward signal it needs.

---

## Start here

| Situation | Use | Starting recipe | Watch |
|---|---|---|---|
| First model from gold traces | SFT | QLoRA, LoRA rank 16 or 32, 1-3 epochs, max length 2048+. | Eval loss, truncation, heldout pass@k. |
| You have good/bad answers for the same prompt | DPO after SFT | `beta=0.1`, 1-2 epochs, adapter LR around `5e-6` to `1e-5`. | Chosen/rejected margin and preference accuracy. |
| You have executable verifiers and sampled attempts differ | GRPO/RLVR | Terminal RL profile, group size 8-32, DAPO, active sampling. | Reward, reward_std, `frac_reward_zero_std`, pass@k. |
| Model cannot pass any terminal tasks yet | SFT or distillation before RL | Teacher distillation or easier curriculum, then re-evaluate. | pass@k moves above zero. |
| You need domain specialists | Cascade RL | Short staged runs by domain, then MOPD distillation. | Per-domain pass@k and final holdout. |
| You have DPPO backend + rollout replay | DPPO smoke/replay path | Export scored replay JSONL, pick backend, run tiny smoke first. | Behavior/train logprobs, mask telemetry, pass@k. |

Default to SFT when in doubt. RL needs contrast. If every sampled attempt gets
the same reward, group-relative RL has little policy-gradient signal.

---

## Effective batch size

For SFT and DPO:

```text
effective batch = per-device batch size * gradient accumulation steps * devices
```

BashGym's conservative local default is:

```text
batch_size=1, gradient_accumulation_steps=8 -> effective batch 8 on one GPU
```

For GRPO and terminal RL, there are two different "batch" ideas:

| Knob | Meaning |
|---|---|
| `batch_size * gradient_accumulation_steps * devices` | Optimizer batch size. This controls gradient update cadence and memory. |
| `grpo_group_size` or `grpo_num_generations` | Attempts sampled for the same prompt and compared against each other. |
| `prompts_per_rollout_batch` | Number of prompt groups targeted in one terminal rollout batch. |

Terminal rollout sample volume is roughly:

```text
prompts_per_rollout_batch * grpo_group_size
```

The `terminal_rl_tmax_like` profile resolves to group size `32`, prompts per
rollout batch `8`, max tool calls per episode `64`, DAPO loss, zero-std
filtering, active sampling, token-level loss, FP32 LM head, and interleaved
thinking.

---

## Max sequence length

Use the shortest length that preserves the useful trace.

| Length | Use when |
|---|---|
| 2048 | Minimum starter for compact tool-call traces. |
| 4096 | Normal coding traces with several commands or file snippets. |
| 8192+ | Long terminal sessions, multi-file work, or larger context windows on remote hardware. |
| 512-1024 | Emergency memory reduction only. Expect truncation to hurt agent behavior. |

Truncation is not a cosmetic problem. If the model sees the task but loses the
verifier output, final edit, or recovery step, it learns the wrong shape of work.

---

## LoRA and QLoRA

LoRA trains a small adapter instead of the full model. QLoRA loads the base model
in 4-bit precision so the adapter can fit on smaller GPUs.

| Knob | Starter | Adjustment |
|---|---|---|
| LoRA rank | 16 or 32 | Raise to 64 for complex tasks or large datasets; lower to 8 for tiny/local smoke runs. |
| LoRA alpha | rank or 2 * rank | BashGym's conservative default is rank 16, alpha 32. |
| LoRA dropout | 0.05 | Raise to 0.1 for small datasets that overfit; set 0.0 for large clean datasets. |
| QLoRA | Enabled on <=24 GB VRAM | Disable only when memory is abundant or quantization hurts the target model. |

Learning-rate starters:

| Run type | Starter |
|---|---|
| QLoRA adapter SFT | `2e-4` when you have enough clean data and want the setup-guide starter. |
| Conservative LoRA or small SFT | `2e-5` when data is small, fragile, or overfitting quickly. |
| Full fine-tune | `2e-5` or lower. |
| DPO adapter refinement | `5e-6` to `1e-5`. |
| GRPO/DPPO | Start near `1e-6` to `2e-5`, depending on backend and model stability. |

---

## SFT

SFT is the first model-building step. It teaches the model the chat template,
tool-call format, local conventions, and the rhythm of verified coding work.

Use SFT when:

- You have gold traces or curated JSONL examples.
- The model does not yet follow the tool-call format reliably.
- Terminal RL pass@k is zero or near zero.

Start with:

```text
strategy=sft
epochs=1-3
batch_size=1
gradient_accumulation_steps=8
max_seq_length=2048 or 4096
use_lora=true
lora_rank=16 or 32
load_in_4bit=true
```

Watch:

- Training and eval loss.
- Grad norm spikes.
- Dataset inspection warnings.
- Heldout trace score and executable environment pass@k.

Stop or adjust when:

- Eval loss rises while train loss falls: reduce epochs, lower LR, add dropout,
  or improve data quality.
- Loss improves but pass@k is flat: the model is imitating text but not learning
  outcome-producing behavior. Add verifier-backed examples or move to a
  curriculum/RL step after it can pass something.

---

## DPO

DPO teaches preference: for the same prompt, choose the better continuation.

Use DPO when:

- You have chosen/rejected pairs for the same prompt.
- Rejected examples are plausible mistakes, not unrelated prompts.
- You already have an SFT checkpoint to refine.

Start with:

```text
strategy=dpo
dpo_beta=0.1
epochs=1-2
learning_rate=5e-6 to 1e-5
max_seq_length=long enough for both chosen and rejected completions
```

Watch:

- `rewards/chosen` and `rewards/rejected`.
- Reward margin.
- Preference accuracy.
- Heldout behavior, not just preference metrics.

Stop or adjust when:

- Margin grows but heldout quality drops: beta or LR may be too aggressive.
- Accuracy is noisy: pairs may not share the same prompt or may be mislabeled.

---

## GRPO and RLVR

GRPO samples a group of attempts for the same prompt, scores them, and updates
the policy from the relative advantage inside that group. RLVR is the verifier
locked version: the reward comes from tests, pass/fail status, or another hard
verifier.

Use GRPO/RLVR when:

- A verifier can score attempts automatically.
- At least some attempts pass and some fail.
- You can afford multiple attempts per prompt.

Start with:

```text
strategy=grpo
training_profile=terminal_rl_tmax_like
grpo_group_size=8 to 32
grpo_loss_type=dapo
filter_zero_std_groups=true
active_sampling=true
token_level_loss=true
lm_head_fp32=true
```

The `terminal_rl_tmax_like` profile uses group size `32` by default. Override to
`8` or `16` for cheaper smoke runs, then raise once the reward signal is real.

Watch:

- Reward and reward trend.
- `reward_std`.
- `frac_reward_zero_std`.
- KL/entropy if the backend logs them.
- Timeout rate, verifier error rate, tamper rate.
- pass@1/pass@k on heldout environments.

Stop or adjust when:

- `frac_reward_zero_std` stays near 1.0 and `reward_std` is near zero: the
  reward has no contrast. Use active sampling, improve the reward, make tasks
  easier, or go back to SFT/distillation.
- pass@k is all zero: the model is not ready for RL on that environment. Build
  a curriculum or distill from a teacher first.
- pass@k is all one: the environment is too easy. Keep a few easy cases for
  regression, but train on harder tasks.

---

## Distillation

Distillation moves teacher behavior into a smaller student. It is useful when
the student is too weak for RL but a teacher can solve the task.

Use distillation when:

- You can afford teacher calls.
- The student cannot produce enough passing attempts for RL.
- You want a small local model to imitate a frontier model's style or domain
  knowledge.

Start with:

```text
strategy=distillation
teacher_temperature=0.7 to 2.0
distillation_alpha=0.5
```

Use higher teacher temperature when you want softer distributions and more
information per example. Use lower temperature when you want crisp task behavior.

Watch:

- Student pass@k against teacher-created tasks.
- Regressions on basic tool-call format.
- Over-imitation of teacher verbosity or unsafe habits.

---

## Cascade RL

Cascade RL trains domain stages in order, such as file operations, bash commands,
search/navigation, and multi-step reasoning. Each stage can use a domain-specific
reward, then the specialists can be merged or distilled.

Use cascade when:

- You have enough examples per domain.
- One broad run keeps improving one domain while hurting another.
- You want a curriculum from easier terminal skills to harder multi-step work.

Start with simulate mode to validate stage composition, then run short real
stages with verifier-backed domains.

Watch:

- Per-domain pass@k.
- Stage-to-stage forgetting.
- Final generalist holdout.

---

## DPPO replay path

DPPO replay is for terminal-rollout optimization with behavior/train logprob
comparison and trust-region masking. It is separate from the simple single-turn
code-generation GRPO script.

Use DPPO when:

- You can export served-model environment rollouts as replay JSONL.
- Behavior policy logprobs are captured or can be replayed.
- A backend such as verl, SkyRL, or open-instruct is installed.

Start with:

```text
export replay JSONL
enrich train-policy logprobs if needed
plan a one-step smoke run
run with max_steps=1 before any long job
```

Watch:

- `behavior_logprobs_ready_records`.
- `train_logprobs_ready_records`.
- `train_logprob_replay_required_records`.
- DPPO mask telemetry.
- pass@k after the smoke run.

World-model replay enrichment is optional and additive. It does not change the
base `bashgym.dppo_replay.v1` record semantics.

---

## Source references

- [../../frontend/src/components/training/TrainingConfig.tsx](../../frontend/src/components/training/TrainingConfig.tsx)
- [../../bashgym/api/schemas.py](../../bashgym/api/schemas.py)
- [../../bashgym/gym/trainer.py](../../bashgym/gym/trainer.py)
- [../../bashgym/gym/terminal_rl.py](../../bashgym/gym/terminal_rl.py)
- [../../bashgym/gym/dppo_launcher.py](../../bashgym/gym/dppo_launcher.py)
- [../../tests/gym/test_terminal_rl_profile.py](../../tests/gym/test_terminal_rl_profile.py)
