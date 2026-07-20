# Training Glossary

Compact definitions for BashGym training, terminal RL, and world-model docs.

## Core training

| Term         | Meaning                                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| SFT          | Supervised fine-tuning. The model imitates successful examples. Use it first to teach format, tool use, and local conventions.            |
| DPO          | Direct Preference Optimization. The model learns from chosen/rejected continuations for the same prompt.                                  |
| GRPO         | Group Relative Policy Optimization. The model samples multiple attempts for one prompt and learns from rewards relative to the group.     |
| RLVR         | Reinforcement learning with verifiable rewards. A verifier, test, or pass/fail gate supplies the reward.                                  |
| Distillation | Training a smaller student from a stronger teacher's outputs or distributions. Useful before RL when the student cannot pass tasks yet.   |
| Cascade RL   | Domain-staged training. BashGym trains easier or narrower domains first, then merges or distills specialists.                             |
| MOPD         | Multi-objective policy distillation. The merge/distillation step that combines domain specialists into one student.                       |
| DPPO         | A terminal-rollout optimization path using behavior/train logprob replay plus trust-region masks. Separate from single-turn GRPO scripts. |

## Data and evaluation

| Term                    | Meaning                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Trace                   | A captured AI coding session with tool calls, outputs, metadata, and outcome signals.                                    |
| Gold trace              | A high-quality successful trace used for SFT.                                                                            |
| Chosen/rejected pair    | A DPO pair where both answers respond to the same prompt and one is clearly better.                                      |
| Terminal environment    | An executable task with an instruction, workspace, rollout policy, verifier, and pass/fail reward.                       |
| Verifier                | A command or harness that decides whether a rollout solved the task. Tests and `verify.sh` scripts are common verifiers. |
| pass@1                  | Fraction of tasks solved by the first attempt.                                                                           |
| pass@k                  | Fraction of tasks solved by any of k attempts. Higher k measures exploration plus capability.                            |
| Holdout gate            | Evaluation on unseen tasks or groups, often split by repo, task family, or source.                                       |
| Spurious-reward control | A negative-control check that catches reward signals that look good by chance or shortcut.                               |
| Tamper canary           | A task designed to reveal whether the model edits tests, fixtures, verifiers, or manifests instead of solving the task.  |

## Training knobs

| Term                  | Meaning                                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Effective batch       | `per-device batch * gradient accumulation * devices`. This is optimizer batch size, not GRPO group size.                |
| Gradient accumulation | Multiple forward/backward passes before one optimizer step. Increases effective batch without increasing per-step VRAM. |
| Max sequence length   | Token budget for a training example. Too short can cut off verifier output, recovery steps, or final answers.           |
| LoRA                  | Low-Rank Adaptation. Trains a small adapter instead of all model weights.                                               |
| QLoRA                 | LoRA with the base model loaded in 4-bit precision to reduce VRAM.                                                      |
| LoRA rank             | Adapter capacity. Rank 16 or 32 is a normal starting range.                                                             |
| LoRA alpha            | Scaling factor for adapter updates. Common starters are rank or `2 * rank`.                                             |
| DPO beta              | DPO divergence strength. Lower values keep the model closer to the reference/SFT model. `0.1` is the starter.           |
| GRPO group size       | Number of attempts sampled for the same prompt and compared against each other. Terminal RL usually wants 8-32.         |
| DAPO                  | A GRPO loss variant used by the terminal RL profile for long, variable terminal rollouts.                               |
| Token-level loss      | Policy loss applied at token granularity. Useful for long terminal trajectories.                                        |
| FP32 LM head          | Keeping the output head in float32 for stability during RL.                                                             |

## Reward and sampling

| Term                   | Meaning                                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reward group           | The set of rewards for attempts sampled from one prompt.                                                                                                   |
| reward_std             | Standard deviation of rewards inside a group. Near zero means no relative RL signal.                                                                       |
| zero-std group         | A reward group where all attempts got the same reward. GRPO advantage is zero or nearly zero.                                                              |
| `frac_reward_zero_std` | Fraction of groups in a step with zero reward variance. Near 1.0 means the reward signal is mostly degenerate.                                             |
| Zero-std filtering     | Dropping zero-variance groups from terminal RL updates.                                                                                                    |
| Active sampling        | Sampling extra groups to replace dropped zero-std groups and keep the batch full.                                                                          |
| SFT warm start         | Using an SFT checkpoint or SFT-style behavior before RL so the model can produce attempts worth scoring.                                                   |
| KL                     | Divergence from the reference or previous policy, depending on backend. Useful for training health, not a release gate by itself.                          |
| Entropy                | Policy randomness/diversity signal. Falling entropy can indicate collapse; high entropy can indicate unfocused sampling. Interpret with reward and pass@k. |
| Verifier error rate    | Fraction of environment attempts where the verifier infrastructure failed rather than the model simply failing the task.                                   |
| Tokens per second      | Throughput metric for sizing batch, context, backend, and hardware choices.                                                                                |
| Peak GPU memory        | Maximum observed GPU memory use during the run. Use with OOM count before scaling.                                                                         |
| OOM count              | Out-of-memory events reported by the run. Any non-zero count means the run shape is not stable enough to scale.                                            |

## World models

| Term                        | Meaning                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JEPA                        | Joint Embedding Predictive Architecture. A family of approaches that learn predictive latent structure instead of reconstructing every raw input detail. |
| ECHO                        | An auxiliary loss that trains the policy to predict terminal observation tokens caused by its actions. Starter `echo_aux_lambda` is `0.05`.              |
| RWML                        | Reinforcement World Model Learning. Rewards predicted next states when they are close to actual next states in embedding space.                          |
| RWML distance threshold     | Cosine-distance cutoff for a correct RWML prediction. Starter is `0.2`.                                                                                  |
| Easy RWML sample            | A transition predicted correctly often enough to be considered easy. Starter easy pass-rate threshold is `0.8`.                                          |
| Easy keep probability       | Probability of retaining easy RWML samples. Starter is `0.1`, so hard samples dominate training.                                                         |
| RWML history window         | Number of prior command/observation pairs included in a transition. Starter is `4`.                                                                      |
| World-model replay coverage | Counts of replay records, RWML transitions, history depth, ECHO segments, and char coverage. This is not prediction quality.                             |

## Backend and artifacts

| Term                          | Meaning                                                                                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| DPPO replay JSONL             | Line-delimited replay records containing environment, trajectory, reward, policy logprobs, optimizer readiness, and optional world-model payloads. |
| Behavior policy               | The model that generated the rollout attempt.                                                                                                      |
| Train policy                  | The model being optimized or replay-scored.                                                                                                        |
| Train-logprob replay required | A replay record has behavior logprobs but not train-policy logprobs yet.                                                                           |
| Binary-TV mask                | DPPO token mask based on total-variation divergence threshold.                                                                                     |
| Binary-KL mask                | DPPO token mask based on KL divergence threshold.                                                                                                  |
| Backend smoke                 | A tiny one-step external-backend run used to prove the launcher, replay schema, env vars, and trainer entrypoint before a real job.                |

## Read next

- [overview.md](overview.md)
- [strategy-guide.md](strategy-guide.md)
- [world-models.md](world-models.md)
- [metrics-runbook.md](metrics-runbook.md)
