# Training methods

Choose a method from the learning signal available in the dataset. BashGym's
direct training request supports `sft`, `dpo`, `grpo`, `rlvr`, `distillation`,
and `session_distillation`. Other workflows in the repository are not aliases
for those six strategies.

Every comparison needs a baseline evaluated on a fixed heldout suite. Training
loss describes optimization; it does not establish behavioral improvement.

## Shared configuration

The direct request schema defines the common parameters: base model, dataset
path, epochs, batch size, gradient accumulation, learning rate, warmup, maximum
sequence length, checkpoint retention, and LoRA settings. Strategy-specific
fields are validated in the same request. See
[`TrainingRequest`](../../bashgym/api/schemas.py).

The effective batch size for one training process is

\[
B_{\mathrm{effective}} = B_{\mathrm{device}} \times N_{\mathrm{accumulation}}.
\]

For distributed trainers, include the number of data-parallel workers as an
additional factor. Keep the effective batch size fixed when comparing learning
rates unless batch size is the variable under study.

Maximum sequence length determines both truncation and memory use. Measure the
token-length distribution after applying the target model's chat template.
Silent truncation can remove the verifier outcome or the action being trained.

LoRA updates low-rank adapters while leaving most base weights frozen. QLoRA
also quantizes the loaded base model. These are parameter and memory choices,
not separate learning objectives.

## SFT

Supervised fine-tuning learns the reference response tokens in structured
conversations. For target tokens \(y_1,\ldots,y_T\) conditioned on context
\(x\), the token cross-entropy is

\[
\mathcal{L}_{\mathrm{SFT}}
= -\sum_{t=1}^{T}\log \pi_\theta(y_t \mid x, y_{<t}).
\]

Use SFT when the dataset contains responses or trajectories worth imitating. It
is also a practical format warm start before preference optimization or RL.

Data contract: JSONL rows with a `messages` list. Tool-use rows may include
structured tool calls and tool-result messages. The direct trainer renders the
messages with the model's chat template.

Evaluate:

- heldout task or trace behavior;
- valid structured outputs and tool calls;
- recovery after failed actions;
- verifier-backed pass@k for terminal-facing models;
- regressions on capabilities the base model should retain.

## DPO

Direct Preference Optimization learns from a chosen and rejected response to
the same prompt. With policy \(\pi_\theta\), reference policy
\(\pi_{\mathrm{ref}}\), chosen response \(y_w\), rejected response \(y_l\), and
temperature-like coefficient \(\beta\):

\[
\mathcal{L}_{\mathrm{DPO}} =
-\mathbb{E}\left[
\log \sigma\left(
\beta\left(
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-

\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right)
\right].
\]

Use DPO when preferences are meaningful and attributable. A collection of gold
responses without rejected alternatives is SFT data, not DPO data.

Data contract: `prompt`, `chosen`, and `rejected`, plus provenance, label
source, split, quality, and decontamination metadata for serious runs. Validate
pairs with
[`dpo_validation.py`](../../bashgym/preferences/dpo_validation.py).

Evaluate preference accuracy and reward margin on heldout pairs, then check the
same behavioral suite used before DPO. An increasing preference margin does not
justify a candidate if task success or reliability regresses.

## GRPO and RLVR

Group Relative Policy Optimization samples several responses for a prompt and
normalizes their rewards within the group. A common group-relative advantage is

\[
A_i = \frac{r_i - \operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}.
\]

The policy update uses these advantages with a clipped probability-ratio
objective. Groups with no reward variance provide no relative learning signal;
the direct request exposes zero-standard-deviation filtering and active
resampling controls.

Use GRPO when multiple sampled answers can be ranked by a reward. Use RLVR when
that reward is produced by a deterministic or executable verifier. The direct
trainer supports syntax, execution, and test-verification reward modes. The
request also exposes group size, sampling temperature, loss variant, ratio
clipping, rollout batch size, and terminal-step limits.

For general terminal tasks, use `EnvironmentSpec`: instruction, fixtures,
build/setup, verifier, reward definition, and rollout limits. See
[`contracts.py`](../../bashgym/environments/contracts.py). A model-judge score
is not equivalent to an executable task-completion verifier.

Evaluate:

- mean reward and reward standard deviation;
- fraction of zero-variance groups and accepted/resampled groups;
- verifier-backed pass@k on training-disjoint tasks;
- timeouts, invalid tool calls, tampering, and spurious-reward canaries;
- behavior relative to the fixed baseline suite.

If \(n\) sampled attempts contain \(c\) successes, BashGym uses the unbiased
estimator

\[
\operatorname{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}.
\]

The implementation is in [`passk.py`](../../bashgym/eval/passk.py).

## Teacher distillation

Teacher distillation normally trains a student from teacher-produced targets or
teacher distributions. BashGym exposes `strategy: "distillation"` and requires
a teacher model in the direct request.

The current generated trainer is not sufficient evidence of full logit
distillation: it declares a KL-plus-cross-entropy function and teacher settings,
but the executed `SFTTrainer` path does not attach that loss or load teacher
logits. Treat the current direct route as an offline teacher-output/SFT
compatibility path until the custom loss is wired and tested end to end. This
boundary is visible in
[`Trainer._generate_distillation_script`](../../bashgym/gym/trainer.py).

For an offline teacher-output experiment, preserve the teacher model and
revision, generation settings, prompts, raw teacher outputs, filtering rules,
and student dataset digest. Compare the student with both its starting model and
the teacher on the same heldout suite.

## Session Distillation

Session Distillation uses a model's own failed or recoverable decision as a
targeted training example. The same target action is scored under:

- the original context, which supplies the student distribution; and
- a hint-injected context, which supplies a detached teacher distribution.

The implemented trainer applies masked cross-entropy and KL only to the target
tokens. Records must include the two contexts, the hint, target text, a
full-target span, `target_span_only` mask policy, reader confidence, verifier
outcome, and source metadata. Partial target-span masking is not implemented.
See
[`session_distillation.py`](../../bashgym/factory/session_distillation.py) and
[`SessionDistillationTrainer`](../../bashgym/gym/trainer.py).

Use this method when the learning question is whether targeted feedback helps
the model make a better decision in similar contexts. Evaluate heldout recovery
decisions, target-token metrics, tool validity, and terminal pass@k where
applicable.

## DPPO replay and backend checks

DPPO is not a value accepted by the direct training strategy enum. The
repository contains replay enrichment, divergence masks, backend capability
checks, smoke plans, and smoke bundles. These prove that rollout data and a
selected external backend can satisfy a launch contract; they do not prove that
BashGym itself executed a DPPO optimizer.

DPPO evidence needs trajectory actions, action masks, old-policy log
probabilities, rewards, replay validation, divergence statistics, the selected
backend's launch evidence, and before/after heldout behavior. Do not label a
GRPO fallback or a generated replay bundle as DPPO training.

Relevant implementations include
[`dppo_replay.py`](../../bashgym/eval/dppo_replay.py) and
[`dppo_backend.py`](../../bashgym/gym/dppo_backend.py).

## Reward-model workflows

Reward-model, outcome-reward-model, and process-reward-model work uses separate
validators and evaluation artifacts; `reward_model` is not a direct
`TrainingStrategy`. Reward examples require a prompt, response or trajectory,
reward value or step rewards, label source, reward scale, provenance, split,
and decontamination metadata. See
[`reward_validation.py`](../../bashgym/preferences/reward_validation.py).

Before using a learned reward for policy optimization, measure heldout pair
accuracy, reward margin, calibration, leakage, and agreement with executable
success where available. A fixture smoke test is not a trained reward model.

## AutoResearch iteration

The agent should begin with the baseline result and one failure hypothesis. A
candidate may change one supported input:

- data source, mixture, threshold, or generated dataset;
- SFT/DPO/RL recipe and bounded hyperparameters;
- reward or verifier;
- evaluator implementation;
- approved training or environment code.

The durable campaign path permits evaluation-only candidates,
training-then-evaluation candidates, and
data-build-then-training-then-evaluation candidates. Each candidate is compared
on the unchanged suite and recorded as kept or discarded before the next
hypothesis. The stage-sequence validation is in
[`proposals.py`](../../bashgym/campaigns/proposals.py).

Do not change the dataset, optimizer, evaluator, and metric in one iteration if
the goal is to attribute the result. When a change is intentionally compound,
record it as such and do not claim a single causal variable.

## Method selection

- Use **SFT** for demonstrated responses or trajectories.
- Use **DPO** for same-prompt chosen/rejected preferences.
- Use **GRPO** for group-sampled responses with a meaningful reward.
- Use **RLVR** when the reward is deterministic or executable.
- Use **teacher-output distillation** only with recorded teacher lineage; treat
  the current direct path as the compatibility boundary described above.
- Use **Session Distillation** for hint-conditioned recovery targets from
  sessions.
- Use **DPPO tooling** for replay and backend validation until an optimizer is
  wired and proven.

## Related references

- [Training data](../TRAINING_DATA_GUIDE.md)
- [Platform architecture](../PLATFORM_OVERVIEW.md)
- [AutoResearch campaign](autoresearch-campaign.md)
