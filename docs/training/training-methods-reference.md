# Training Methods Reference

This document is the method-by-method reference for BashGym's open-model
training platform. It is written for operators and outside AI/ML reviewers who
need to understand what each method optimizes, what data it needs, what BashGym
currently supports, and what evidence should be required before trusting a
trained model.

Use it with:

```bash
bashgym training capabilities --json
bashgym training plan --strategy sft --json
bashgym training plan --strategy grpo --data terminal_envs --hardware private_compute --json
bashgym training plan --strategy world-model --json
```

For platform surfaces and release gates, read
[capability-map.md](capability-map.md). For diagnosis, read
[metrics-runbook.md](metrics-runbook.md). For terminal RL, read
[tmax-terminal-rl-recipe.md](tmax-terminal-rl-recipe.md). For ECHO/RWML, read
[world-models.md](world-models.md). For targeted self-distillation from failed
trace spans, read [session-distillation.md](session-distillation.md).

---

## Status Key

| Status              | Meaning                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Ready               | BashGym has implemented user-facing controls, API/CLI support, docs, and local tests or smoke evidence.                    |
| Ready with evidence | The method is usable, but only when the user supplies valid data, verifiers, or release evidence.                          |
| Backend-dependent   | BashGym has contracts, adapters, replay, or launch planning, but an installed external backend must still prove execution. |
| Diagnostic          | Useful for investigation, curriculum, or auxiliary training, but not enough to approve routing by itself.                  |
| Compute-gated       | Local contracts exist; the remaining proof requires a configured private/cloud backend run.                                |

---

## Method Selection Rule

Start with the least exotic method that can teach the missing behavior.

| Situation                                                    | First method to try            | Why                                                                              |
| ------------------------------------------------------------ | ------------------------------ | -------------------------------------------------------------------------------- |
| The student cannot follow the tool-call/chat format.         | SFT                            | It teaches format, conventions, and basic action structure.                      |
| You have same-prompt better/worse examples.                  | DPO                            | It sharpens preferences after SFT without needing an executable reward.          |
| The student cannot solve tasks at all.                       | SFT or distillation            | RL needs attempts worth scoring; all-zero pass@k is usually not ready for RL.    |
| Executable tasks sometimes pass and sometimes fail.          | GRPO/RLVR                      | Reward groups have contrast, so verifier-backed RL can learn.                    |
| Failed traces contain a local mistake and later recovery.    | Session Distillation           | It repairs the same target action under a hint without replacing the trajectory. |
| You have multi-step terminal rollouts with logprobs.         | DPPO replay/backend path       | Replay and trust-region masks can hand terminal trajectories to an RL backend.   |
| You want terminal-dynamics diagnostics or curriculum mining. | ECHO/RWML                      | World-model signals can explain or mine failures, but remain diagnostic.         |
| A broad run forgets domains.                                 | Cascade/domain-staged training | Per-domain stages can isolate regressions and later distill/merge.               |

---

## SFT: Supervised Fine-Tuning

**What it optimizes:** next-token likelihood on successful examples.

**BashGym role:** first student baseline. SFT teaches the model the shape of
verified coding work: tool calls, command style, repo conventions, final
verification, and recovery behavior.

**Inputs and artifacts:**

- Gold traces.
- Curated `messages` JSONL.
- `training_examples.jsonl` with source metadata and quality score.
- Config snapshot and `metrics.jsonl`.

**BashGym capability:** Ready.

- Training API and UI expose SFT settings.
- CLI exposes starter plans and setting explanations.
- Backends include Unsloth where supported and plain Transformers/PEFT fallback.
- Local/remote/managed paths exist, depending on hardware and provider.

**Key settings:**

| Setting                     | Meaning                            | Starter                                                                                              |
| --------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `learning_rate`             | Optimizer step size.               | Local LoRA SFT can start higher than DPO/RL; remote/full runs should be more conservative.           |
| `epochs`                    | Passes over the dataset.           | Start with one smoke epoch, then increase only after heldout behavior is healthy.                    |
| `max_seq_length`            | Token budget per example.          | Use the shortest value that preserves prompt, actions, observations, final fix, and verifier output. |
| `use_lora` / `load_in_4bit` | Adapter and QLoRA memory controls. | Use for most local iteration.                                                                        |
| `lora_rank` / `lora_alpha`  | Adapter capacity and scaling.      | Rank 16-32 is a practical first range.                                                               |

**Metrics to watch:**

- Train loss and eval loss.
- Grad norm.
- Truncation count.
- Tool-call/schema validity.
- Heldout trace behavior.
- Executable environment pass@k.

**Risks:**

- Loss improves but behavior does not.
- Failed traces accidentally become SFT success examples.
- Long traces are truncated before the final verifier/fix.
- The model learns style without outcome improvement.

**Reviewer questions:**

- Are BashGym's gold-trace filters strong enough to avoid teaching incomplete or misleading work?
- Should SFT examples include explicit verifier/output masking beyond the current assistant/tool-call masking?
- What minimum heldout pass@k change should be required before routing an SFT student?

---

## DPO: Direct Preference Optimization

**What it optimizes:** preference between chosen and rejected completions for
the same prompt, without training a separate reward model.

**BashGym role:** preference refinement after SFT. DPO should make the student
prefer better actions or completions, but it does not replace executable eval.

**Inputs and artifacts:**

- `dpo_pairs.jsonl`.
- Same prompt identity for chosen/rejected.
- Chosen/rejected quality labels.
- SFT checkpoint or reference model.

**BashGym capability:** Ready.

- DPO config is exposed through training schema/config.
- Decision-DPO data generation exists in the factory surfaces.
- Analyzer summarizes reward margin and preference accuracy when emitted.

**Key settings:**

| Setting          | Meaning                              | Starter                                          |
| ---------------- | ------------------------------------ | ------------------------------------------------ |
| `dpo_beta`       | Strength of reference-policy pull.   | Start around `0.1`; lower if behavior regresses. |
| `learning_rate`  | Step size.                           | Lower than SFT in most cases.                    |
| `max_seq_length` | Pair prompt/completion token budget. | Must preserve the decisive difference.           |
| Pair filters     | Remove bad pairs.                    | Require same prompt and meaningful difference.   |

**Metrics to watch:**

- Chosen/rejected reward.
- Reward margin.
- Preference accuracy.
- Chosen/rejected logprobs.
- Heldout behavior versus the SFT base.

**Risks:**

- Chosen/rejected are not actually comparable.
- Rejected examples are trivially bad, so the model learns a shallow distinction.
- Reward margin improves while task behavior regresses.

**Reviewer questions:**

- What metadata should be mandatory for coding-agent DPO pairs?
- Should BashGym add pair-difficulty or length-ratio gates before training?
- Are ORPO/KTO/IPO/SimPO useful as import/export recipes, or should DPO stay first-class?

---

## Reward Models, ORM, and PRM

**What they optimize:** a learned reward or scoring function. Preference reward
models rank chosen over rejected completions. Outcome reward models score whole
answers or trajectories. Process reward models score steps inside a trajectory.

**BashGym role:** first-class planning and evidence lane for learned reward
models, with training backend proof still evidence-gated. Reward models should
start as audit, best-of-N, rejection-sampling, or trajectory-scoring tools before
they are trusted as RL rewards.

**Inputs and artifacts:**

- `reward_examples.jsonl`.
- `reward_eval.json`.
- Declared `reward_type`: `preference_reward`, `outcome_reward`, or
  `process_reward`.
- Label source and reward scale.
- Source manifest and split/decontamination metadata.
- Step-level reward labels for PRM/process-reward examples.

**BashGym capability:** Ready with evidence.

- Strict reward-example validation exists through
  `bashgym training reward-examples validate <path> --strict`.
- RunCards can require reward examples for `reward_model`, `rm`, `orm`, `prm`,
  and process-reward methods.
- `bashgym training plan --strategy reward-model --json` provides starter
  settings, readiness ladders, and metric guidance.
- `bashgym training reward-model smoke <reward_examples.jsonl> --output-dir <dir> --json`
  runs a dependency-free fixture scorer and writes predictions, metrics, and
  `reward_eval.json` as a contract smoke before real backend training.
- `bashgym training reward-eval evaluate <path> --output reward_eval.json --json`
  emits heldout accuracy, calibration, bias, variance, and eval-only leakage
  evidence for RunCards.
- Held-out Gate release evidence accepts `learned_reward_evidence` and surfaces
  it as diagnostic reward-model evidence until thresholds are chosen.
- RewardBench and CUARewardBench result JSON can be attached through external
  benchmark ingest as eval-only release evidence.

**Key settings:**

| Setting                      | Meaning                                                         | Starter                                                           |
| ---------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| `reward_artifact`            | JSONL reward examples to train/evaluate against.                | `reward_examples.jsonl`, strict-validated.                        |
| `reward_type`                | Preference, outcome, or process reward target.                  | Preference/outcome first; process only with step labels.          |
| `reward_loss`                | Pairwise, regression, or classification loss shape.             | `pairwise_or_regression`.                                         |
| `reward_scale`               | Explicit score range or label schema.                           | Keep declared in the artifact.                                    |
| `eval_split_required`        | Heldout split requirement before serious claims.                | `true`.                                                           |
| `train_split` / `eval_split` | Which records the fixture smoke can learn from and evaluate on. | `train` and `eval`; never train from eval-only benchmark sources. |

**Metrics to watch:**

- Heldout pair accuracy.
- Calibration error.
- Reward margin.
- Length bias.
- Task-family breakdown.
- Reward variance.
- Eval-only leakage.

**Risks:**

- Reward accuracy can improve on easy labels while behavior gets worse.
- Length bias can select verbose or looping trajectories.
- Eval-only benchmark data can leak into the reward artifact.
- RewardBench/CUARewardBench scores are external evidence, not training data by
  default.
- A learned reward can be over-optimized unless matched controls and heldout
  behavior gates are attached.

**Reviewer questions:**

- Which reward benchmark best matches terminal/coding trajectories?
- Should BashGym keep reward models audit-only until selected-vs-random controls
  improve heldout pass@k?
- What calibration threshold should be mandatory before broad claims?

---

## ORPO, KTO, IPO, and SimPO

**What they optimize:** preference alignment variants that reduce or alter the
need for a separate supervised/reference setup, depending on method.

**BashGym role:** ecosystem references, not first-class BashGym workflows today.
They may be useful as future recipe import/export targets through stacks such as
Unsloth or Axolotl.

**Inputs and artifacts:**

- Preference-style pairs or accept/reject labels, depending on method.
- Strong prompt identity and source metadata.
- Heldout behavior evidence after training.

**BashGym capability:** Ecosystem reference.

- BashGym's canonical preference path is DPO.
- Capability docs list ORPO/KTO/IPO/SimPO as methods to track, not stable product paths.

**Metrics to watch:**

- Method-specific preference objective.
- Heldout trace behavior.
- pass@k.
- Forgetting versus SFT base.

**Risks:**

- Easy to overstate support because external libraries expose the method.
- Preference metrics can improve while executable behavior regresses.
- Method choice may distract from data quality and pair quality.

**Reviewer questions:**

- Which preference method is most appropriate for terminal-agent traces with tool calls?
- Should BashGym support multiple preference objectives, or keep DPO until pair quality is proven?

---

## PPO / RLHF

**What it optimizes:** policy improvement against a reward model or verifier,
usually with KL regularization against a reference policy.

**BashGym role:** backend candidate, not the primary built-in terminal RL path
today. PPO is relevant through TRL, verl, and OpenRLHF, but BashGym keeps its
canonical product contract environment/replay/eval first.

**Inputs and artifacts:**

- Prompts/environments.
- Reward function or reward model.
- Reference policy.
- Rollout logs.
- KL, entropy, reward, and behavior evidence.

**BashGym capability:** Backend candidate.

- PPO is not advertised as a first-class BashGym workflow.
- External backends may consume BashGym artifacts if wrapped behind the same eval/replay contracts.

**Metrics to watch:**

- Reward.
- KL.
- Entropy.
- pass@k.
- Timeout/tamper/verifier-error rate.

**Risks:**

- Reward model or verifier shortcuts.
- KL/entropy interpreted as product quality.
- Harder multi-turn terminal trajectories require careful environment integration.

**Reviewer questions:**

- Should BashGym prioritize PPO through verl/OpenRLHF, or keep GRPO/DPPO as the first terminal-agent RL paths?
- What PPO telemetry would be mandatory before scaling?

---

## GRPO / RLVR

**What it optimizes:** relative policy improvement from groups of sampled
attempts scored by verifiable rewards.

**BashGym role:** first-class verifier-backed terminal RL path.

**Inputs and artifacts:**

- Executable terminal environments.
- Attempts grouped by prompt/environment.
- Verifier reward.
- pass@k report.
- Holdout and tamper evidence.

**BashGym capability:** Ready with evidence.

- Terminal RL profile exists.
- Active sampling and zero-std filtering are exposed.
- Environment pass@k, holdout, comparison, spurious controls, and tamper canaries exist.
- DPPO replay can be exported when rollouts/logprobs are available.

**Key settings:**

| Setting                                  | Meaning                               | Starter                                                  |
| ---------------------------------------- | ------------------------------------- | -------------------------------------------------------- |
| `training_profile=terminal_rl_tmax_like` | Terminal RL defaults.                 | Use for verifier-backed shell environments.              |
| `grpo_group_size`                        | Attempts sampled for one prompt.      | 8 locally, 16-32 when hardware allows.                   |
| `active_sampling`                        | Replaces zero-variance groups.        | Enable for sparse rewards.                               |
| `filter_zero_std_groups`                 | Drops groups with no reward contrast. | Enable for policy updates.                               |
| `max_tool_calls_per_episode`             | Action budget.                        | Enough for valid solutions; low enough to prevent loops. |

**Metrics to watch:**

- Reward.
- `reward_std`.
- `frac_reward_zero_std`.
- pass@1/pass@k.
- Verifier error rate.
- Timeout rate.
- Tamper rate.

**Risks:**

- All attempts fail or all pass, producing no useful relative signal.
- Verifier errors masquerade as model failures.
- Reward hacking through verifier/test/fixture tampering.
- pass@k improves only through exploration while pass@1 stays weak.

**Reviewer questions:**

- Are reward-contrast gates sufficient before scaling RL?
- What additional canaries should be mandatory for coding-agent verifiers?
- Should BashGym require pass@1 improvement, pass@k improvement, or both for routing?

---

## RLOO / REINFORCE-Style Methods

**What they optimize:** policy-gradient objectives using leave-one-out or
REINFORCE-style estimators, often exposed by external RLHF stacks.

**BashGym role:** possible future backend algorithm family.

**Inputs and artifacts:**

- Rollouts.
- Rewards.
- Group or batch baselines depending on method.
- Behavior/eval evidence.

**BashGym capability:** Backend candidate only.

- BashGym does not expose RLOO/REINFORCE as a first-class workflow.
- These methods should enter through external backend integration only after replay/eval contracts are stable.

**Risks:**

- High variance.
- Easy to confuse backend availability with product maturity.
- Requires the same anti-hacking and heldout gates as GRPO/RLVR.

**Reviewer questions:**

- Would RLOO/REINFORCE-style methods be simpler than DPPO for terminal rollouts?
- What replay fields would those methods require beyond current DPPO records?

---

## Distillation

**What it optimizes:** student imitation of a stronger teacher's outputs or
distributions.

**BashGym role:** bridge when the student is too weak for RL, or when we want a
smaller routable model for narrow domains.

**Inputs and artifacts:**

- Teacher model/config.
- Teacher outputs or on-policy teacher traces.
- Student checkpoint.
- Heldout pass@k and behavior comparison.

**BashGym capability:** Ready.

- Teacher/student config exists.
- Training path supports distillation-style settings.
- Docs position it before RL when pass@k is all zero.

**Metrics to watch:**

- Student loss.
- Teacher agreement.
- Tool-format regression.
- Student pass@k.
- Heldout behavior versus teacher/base.

**Risks:**

- Copies teacher mistakes.
- Produces plausible traces without independent verifier success.
- Overfits to teacher style rather than executable outcomes.

**Reviewer questions:**

- When should BashGym prefer distillation over SFT warm start?
- What teacher-output metadata is necessary for reproducible student training?

---

## Session Distillation

**What it optimizes:** a masked objective over the student's own target action
tokens. BashGym inserts a short local hint before a failed span, scores the same
tokens under original and hinted context, and trains the original context toward
the hinted distribution while preserving hard-label CE.

**BashGym role:** repair lane for trace-local mistakes. It sits between SFT/DPO
and verifier-backed RL: narrower than whole-answer preference learning, but more
behavioral than plain imitation.

**Inputs and artifacts:**

- Failed or recovery-rich coding traces.
- `session_distillation_records.jsonl`.
- Original context without hints.
- Hinted context with `[Session Distillation Hint]`.
- Exact target action text and `target_span_only` loss mask.
- Reader confidence, verifier outcome, and source metadata.

**BashGym capability:** Ready with evidence.

- The factory can build and validate records from failed trace steps.
- A heuristic reader creates auditable hints and skips clean traces.
- Training API, schema, frontend config, Data Designer pipeline registration,
  and trainer script generation include `session_distillation`.
- RunCards require records, reader/mask metadata, masked-loss metrics, and
  release evidence before promotion.

**Key settings:**

| Setting                               | Meaning                                           | Starter            |
| ------------------------------------- | ------------------------------------------------- | ------------------ |
| `session_distillation_alpha`          | Weight on hinted-context KL versus hard-label CE. | `0.7`              |
| `session_distillation_temperature`    | Softness for the hinted-context distribution.     | `1.0`              |
| `session_distillation_min_confidence` | Minimum reader confidence for accepted records.   | `0.6`              |
| `session_distillation_mask_policy`    | Which tokens receive loss.                        | `target_span_only` |
| `session_distillation_context_mode`   | How hints enter the context.                      | `hint_injected`    |
| `session_distillation_reader`         | Heuristic or model reader for hint generation.    | `heuristic`        |

**Metrics to watch:**

- `session_distillation_loss`
- `session_distillation_kl`
- `session_distillation_ce`
- `session_distillation_masked_tokens`
- Reader confidence distribution.
- Heldout recovery-decision accuracy.
- Tool-call validity and executable pass@k where environments exist.

**Risks:**

- Noisy hints can teach the wrong local correction.
- Loss can fall without recovery behavior improving.
- Overweighting the hinted distribution can weaken ordinary action quality.
- Public trajectory datasets need source manifests and decontamination before
  use; they are fixture/eval resources before they are training defaults.

**Reviewer questions:**

- Should a model reader replace or audit the heuristic reader after the first
  local smoke?
- What heldout recovery-decision threshold should promote a Session
  Distillation run?
- Should target spans include only command/tool tokens, or also brief
  surrounding rationale when the model family emits explicit reasoning tags?

---

## Cascade / Domain-Staged Training

**What it optimizes:** staged capability acquisition across domains, followed by
merge, distillation, or routing.

**BashGym role:** curriculum and anti-forgetting strategy for broad coding-agent
work.

**Inputs and artifacts:**

- Domain labels.
- Per-domain datasets/environments.
- Per-stage configs and checkpoints.
- Per-domain holdout gates.
- Final generalist eval.

**BashGym capability:** Ready with evidence.

- Cascade routes and staged configuration exist.
- Docs require per-domain gates and final generalist holdout.

**Metrics to watch:**

- Per-stage loss/reward.
- Per-domain pass@k.
- Forgetting across earlier domains.
- Final heldout generalist behavior.

**Risks:**

- Domain stages improve local specialists while harming generalist behavior.
- Poorly chosen domain boundaries hide regressions.
- Merge/distillation can wash out specialist gains.

**Reviewer questions:**

- How should BashGym define domain boundaries for coding-agent traces?
- What forgetting threshold should block stage promotion?

---

## DPPO Replay / Backend Path

**What it optimizes:** terminal rollout policy updates using replayed behavior
and train-policy logprobs with trust-region masks.

**BashGym role:** backend handoff path for multi-step terminal rollouts.

**Inputs and artifacts:**

- `EnvironmentSpec`.
- Served-model rollout attempts.
- Commands and terminal observations.
- Verifier reward.
- Behavior-policy logprobs.
- Train-policy logprobs when optimizer updates need them.
- Binary-TV/Binary-KL mask telemetry.
- Optional ECHO/RWML payload.
- `backend_smoke_readiness.json`.

**BashGym capability:** Backend-dependent and compute-gated.

- Replay schema exists.
- Replay summary exists.
- Train-logprob enrichment exists.
- DPPO mask math exists.
- Smoke bundle and launch env exist.
- Installed-backend smoke is still required before claiming runnable DPPO training.

**Key settings:**

| Setting                  | Meaning                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `backend`                | External backend target such as verl, SkyRL, OpenRLHF, or TMax/open-instruct style integration. |
| `dppo_divergence`        | `binary_tv` or `binary_kl` trust-region mask mode.                                              |
| Thresholds               | Token mask cutoffs for divergence.                                                              |
| Replay path              | JSONL artifact carrying trajectories.                                                           |
| Train-logprob enrichment | Adds train-policy logprobs before optimizer update.                                             |

**Metrics to watch:**

- Replay records.
- Behavior-logprob ready records.
- Train-logprob ready records.
- Required train-logprob replay count.
- Mask keep/drop counts.
- Reward/pass@k before and after backend smoke.

**Risks:**

- Contract looks ready locally but external backend cannot consume it.
- Logprobs are misaligned with tokenization.
- Trust-region thresholds are arbitrary until calibrated.
- Backend smoke proves import/execution but not behavior improvement.

**Reviewer questions:**

- Is the DPPO replay contract sufficient for real RL backends?
- Should BashGym pick one canonical backend first, probably SkyRL or verl?
- What minimum before/after pass@k proof should follow a one-step smoke?

---

## JEPA-Style World Models: ECHO and RWML

**What they optimize:** predictive structure around terminal dynamics.

BashGym uses "JEPA-style" in the narrow sense of learning useful predictive
state structure rather than reconstructing every byte of raw terminal output.
The target transition is:

```text
instruction + prior command/observation history + next command
    -> next terminal observation / diff / test state / verifier state
```

### ECHO

ECHO adds an auxiliary observation-token prediction loss:

```text
total loss = policy loss + echo_aux_lambda * environment_prediction_loss
```

### RWML

RWML rewards or scores next-state prediction in embedding space:

```text
reward = 1 if predicted next state is close to actual next state
```

**BashGym role:** auxiliary diagnostics, curriculum mining, and future reranking.

**Inputs and artifacts:**

- DPPO replay with `world_model` payload.
- ECHO role-tagged action/observation spans.
- RWML transition triplets.
- Embedding model id.
- Backend quality metrics.

**BashGym capability:** Backend-dependent and diagnostic.

- Training config fields exist.
- Replay payloads exist.
- Replay summary coverage exists.
- Backend adapter and reward factories exist.
- World-Model Quality panel parses backend metric names.
- Real installed backend loop remains pending.

**Metrics to watch:**

- `world_model_records`.
- `rwml_transitions`.
- `echo_observation_chars`.
- `echo_loss`.
- `rwml_pass_rate`.
- `embedding_distance_mean` / `embedding_distance_p95`.
- `exit_code_accuracy`.
- `test_result_accuracy`.
- Correlation with heldout pass@k and safety.

**Risks:**

- Coverage is mistaken for quality.
- Low ECHO loss does not imply better agent behavior.
- RWML threshold is too easy and rewards trivial predictions.
- World-model metrics become release gates before correlation is proven.

**Reviewer questions:**

- Are ECHO/RWML the right objectives for terminal-agent dynamics?
- Should BashGym learn latent state over diffs/tests/verifier state instead of terminal text?
- What evidence would justify using world-model metrics for routing or release decisions?

---

## Evaluation and Promotion

BashGym should not promote a model from training metrics alone.

| Signal                                                  | Role                                                    |
| ------------------------------------------------------- | ------------------------------------------------------- |
| Loss, KL, entropy, grad norm                            | Training health.                                        |
| Reward, reward_std, preference accuracy                 | Learning signal quality.                                |
| Reward-model heldout accuracy, calibration, length bias | Learned-reward evidence.                                |
| pass@1/pass@k, heldout trace delta, holdout comparison  | Behavior evidence.                                      |
| Spurious controls, tamper canaries, verifier error rate | Safety/release gates.                                   |
| ECHO/RWML quality                                       | Diagnostic until correlated with behavior.              |
| External benchmark ingest                               | Evidence for broad claims when leakage manifests exist. |

Minimum promotion evidence:

- Heldout trace eval is not worse than baseline.
- Environment pass@k improves or meets a declared threshold.
- Grouped holdout gate passes.
- Base-vs-candidate comparison passes when available.
- Spurious controls stay clear.
- Reward-hacking/tamper canaries fail closed.
- External benchmark evidence is attached for broad claims.
- DPPO/ECHO/RWML runs preserve smoke readiness and backend logs.
- World-model quality remains diagnostic.

---

## Source References

- Hugging Face TRL overview: https://huggingface.co/docs/trl/en/index
- TRL SFT Trainer: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO Trainer: https://huggingface.co/docs/trl/en/dpo_trainer
- TRL GRPO Trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- Unsloth docs: https://unsloth.ai/docs
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- verl docs: https://verl.readthedocs.io/
- SkyRL docs: https://docs.skyrl.ai/
- OpenRLHF docs: https://openrlhf.readthedocs.io/
- Axolotl RLHF docs: https://docs.axolotl.ai/docs/rlhf.html
- torchtune DPO recipe: https://meta-pytorch.org/torchtune/0.6/recipes/dpo.html
- LeCun, A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA paper: https://arxiv.org/abs/2301.08243
- Meta V-JEPA research: https://ai.meta.com/research/vjepa/
