# BashGym Training Overview

This guide explains the training gym from first principles: what data enters the
system, what each training strategy is trying to teach, and which evidence proves
that a trained model is actually better.

For the full capability spread, read [capability-map.md](capability-map.md). For
exact knobs and recipes, read [strategy-guide.md](strategy-guide.md) and
[tmax-terminal-rl-recipe.md](tmax-terminal-rl-recipe.md). For world-model
objectives, read [world-models.md](world-models.md). For targeted
self-distillation from failed trace spans, read
[session-distillation.md](session-distillation.md). For diagnosis during and
after a run, read [metrics-runbook.md](metrics-runbook.md).

---

## The mental model

BashGym trains coding agents from verified work. A useful training example is
not just a prompt and an answer. It is a trajectory:

```text
task + repo/context + commands/tool calls -> outputs/diffs/tests/verifier result
```

The gym keeps the whole loop visible:

```text
capture/import -> classify -> generate examples -> train -> evaluate -> deploy
       ^                                                        |
       |                                                        v
       +---------------------- collect new traces <-------------+
```

This trace-to-training loop is the core BashGym flywheel. Source discovery,
reward modeling, terminal RL, JEPA-style diagnostics, AutoResearch, compute
targets, and education are supporting flywheels around it, not replacements for
it. Use [platform-flywheels.md](platform-flywheels.md) when explaining how these
loops fit together without blurring their jobs.

The key rule is simple: loss curves are not release evidence. Verifiers, tests,
pass@k, holdout gates, tamper checks, and external benchmarks decide whether the
student is good enough to route traffic to.

---

## What BashGym trains from

| Source | What it contains | Best first use |
|---|---|---|
| Gold traces | Successful AI coding sessions captured from Claude Code, Codex, Gemini, Copilot, OpenCode, and similar tools. | SFT baseline and DPO chosen examples. |
| Failed or lower-tier traces | Attempts with bad choices, failed commands, weak verification, or low quality scores. | DPO rejected examples and failure analysis. |
| Custom JSONL | Prebuilt NeMo/TRL-style `messages` datasets. | Importing curated external or hand-authored data. |
| Security datasets | Malware, phishing, or other labeled security corpora converted into examples. | Security-specialist behavior or classification tasks. |
| Terminal environments | Executable tasks with a prompt, workspace, verifier, rollout attempts, and pass/fail reward. | GRPO/RLVR, DPPO replay, pass@k, holdout gates, and world-model replay data. |

Gold traces teach the model what good behavior looks like. Terminal
environments prove whether the behavior survives interaction with a real shell,
repo, and verifier.

### AutoResearch across models and backends

AutoResearch is BashGym's shared research control plane, not a feature of one
model, environment, or trainer. Every registered open model uses the same agent
intake, baseline-first hypotheses, budgets, attempts, leases, cancellation,
restart recovery, artifact sealing, heldout evaluation, experiment ledger,
keep/discard decision, promotion gates, and workspace canvas.

| Layer | All registered BashGym models | Optional NeMo RL/Gym extension |
|---|---|---|
| Research controller and operator skills | Shared | Reused unchanged |
| Local/private-SSH execution authority | Shared | Reused unchanged |
| Evaluation, evidence, ledger, and promotion | Shared | Reused unchanged |
| Model loader and training recipe | Selected per registered model/backend | NeMo RL recipe adapter |
| Distributed rollout/generation topology | Backend-dependent | Ray plus async vLLM |
| Multi-turn environment isolation | Backend-dependent | NeMo Gym servers/sessions |
| Training-to-generation refit | Backend-dependent | NeMo RL refit contract |

A model appearing in a local cache is not enough to activate training. Each new
trainable base must resolve an immutable model revision, compatible installed
trainer recipe, approved data/evaluator binding, and compute profile. Inference
quants and served deployment artifacts cannot silently satisfy that contract.
Inspect an operator-selected snapshot without scanning caches or downloading a
substitute model:

```bash
bashgym campaign inspect-model-artifact \
  --artifact-dir /path/to/selected/snapshot \
  --model-id organization/model \
  --model-revision <immutable-commit> \
  --json
```

The secret-free plan distinguishes a trainable base from an adapter or
inference quant, hashes the complete selected artifact, and identifies candidate
standard, Unsloth, and optional NeMo backends. Pass the same directory as
`--model-artifact-dir` to `campaign setup-autoresearch` to require the inspected
model identity, task, and training readiness before the installation definition
is written. `campaign doctor` remains authoritative for installed runtime,
data, evaluator, credential-material, and compute readiness.

For a new installation, graduate the no-GPU control smoke through one canonical
activation sequence:

1. Inspect the operator-selected snapshot with
   `campaign inspect-model-artifact`.
2. Create the portable definition with `campaign setup-autoresearch`.
3. Run `campaign activate-autoresearch` without `--apply` to preflight the
   registered SSH device, source scope, dataset, evaluator, launch material, and
   identity conflicts.
4. Review the plan and repeat with `--apply`.
5. Require `campaign doctor` to become `materializable`, bring the resident
   controller online through `--install-worker` or an existing service, re-run
   doctor, and require `launch_ready` before a bounded real baseline. Only then
   launch a one-variable candidate.

Registered SSH is the protected execution boundary for both private hardware
and hardware on the BashGym machine via localhost SSH. See
[autoresearch-campaign.md](autoresearch-campaign.md) for the exact flags,
receipts, and evidence requirements.

Named multi-reward environments can bind their declared component order and
weights through `NamedRewardGDPOAdapter`. The adapter emits NeMo's stable
`reward1`, `reward2`, ... and `total_reward` columns, selects the GDPO advantage
estimator with per-component normalization, and records deterministic batch,
configuration, and advantage digests. Its consumed advantages are computed by
the same dependency-free `gdpo_advantages` reference used in BashGym parity
tests, so trainer-side component/order/weight drift fails before launch.

### Optional NeMo Gym environment export

BashGym can export its deterministic star-count environment into the current
NeMo Gym resources-server, simple-agent, and Responses API dataset layout
without installing NeMo Gym into BashGym's core Python environment:

```python
from bashgym.environments import (
    create_nemo_gym_bundle_archive,
    export_star_count_nemo_gym_bundle,
)

manifest = export_star_count_nemo_gym_bundle(
    "star-count-dataset",
    "nemo-gym-bundle",
    nemo_gym_revision="<40-character NeMo Gym commit>",
    bashgym_revision="<40-character BashGym commit>",
    dataset_license="MIT",
)
archive = create_nemo_gym_bundle_archive(
    "nemo-gym-bundle",
    "nemo-gym-bundle.zip",
)
```

The exported directory can be turned into a deterministic, content-validated
single-file transport artifact with `create_nemo_gym_bundle_archive`. A
Gym-enabled executor binds that archive digest, requires NVIDIA's
`examples/nemo_gym/run_grpo_nemo_gym.py` entrypoint, verifies the embedded Gym
source revision, mounts only the approved resources server, includes
`vllm_model_for_training.yaml`, and enables async vLLM HTTP generation.
`no_update` smoke stages use Gym trajectory-collection mode instead of taking an
optimizer step.

The bundle embeds portable image data, BashGym's exact component verifier,
immutable source revisions, and content hashes. Its resources server imports
BashGym only when launched inside the operator's separately pinned NeMo Gym
runtime. Rollout evidence validation requires model-server message-level prompt
IDs, generation IDs, and generation logprobs to pass through unchanged, unique
session IDs for concurrent rollouts, exact component totals, and a synchronized
policy-to-generation refit receipt.

For campaign execution, the registered runner returns the exact bundle manifest
and environment contract alongside runtime logs. The Gym runtime integration
must also emit `nemo_gym_trajectories.jsonl` and an independently observed
`nemo_gym_refit_receipt.json`. When all four raw companions are downloaded, the
remote sealer automatically converts them into the canonical
`nemo_gym_campaign_evidence.json` receipt:

```python
from bashgym.campaigns.nemo_gym_ingestion import convert_nemo_gym_outputs

convert_nemo_gym_outputs(
    attempt,
    bundle_manifest="nemo_gym_bundle_manifest.json",
    environment_contract="nemo_gym_environment_contract.json",
    trajectories="logs/nemo_gym_trajectories.jsonl",
    refit_receipt_path="logs/nemo_gym_refit_receipt.json",
    output_directory=output_directory,
)
```

Each trajectory must carry the same full exact refit receipt as the separate
receipt file. Missing, ambiguous, stale, unsynchronized, or cross-refit output
is rejected; process completion and checkpoint presence are never interpreted
as synchronization evidence.

The receipt binds the exact workspace, campaign, study, action, attempt,
candidate, manifest revision, and claim generation to the bundle and environment
digests. It also preserves the full message-level token arrays, named component
rewards, weighted totals, ordered rollout identities, checkpoint digest, and
policy/generation refit revision. The remote output sealer validates that binding
before assigning the artifact schema. AutoResearch outcome evidence then includes
the sealed artifact ID, while planner snapshots expose only bounded digests,
counts, mean reward, training step, and policy revision—not raw rollouts or
artifact paths.

This is an adapter and evidence boundary, not a claim that a live NeMo RL refit
has run. The live proof remains gated on a dedicated NeMo executor and an
already approved compatible campaign model. See NVIDIA's
[environment model](https://docs.nvidia.com/nemo/gym/main/about/concepts/environments/),
[on-policy token contract](https://docs.nvidia.com/nemo/gym/main/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction),
and [NeMo RL integration flow](https://docs.nvidia.com/nemo/rl/nightly/design-docs/nemo-gym-integration.html).

---

## What the training strategies teach

| Strategy | Teaches | Needs | Proves itself with |
|---|---|---|---|
| SFT | Imitation: reproduce successful trace format, tool use, and problem-solving style. | Gold examples. | Eval loss plus heldout pass@k. |
| DPO | Preference: choose the better response for the same prompt. | Chosen/rejected pairs. | Preference accuracy, reward margin, heldout behavior. |
| GRPO/RLVR | Outcome optimization: improve completions using verifier rewards. | Reward variation across sampled attempts. | Reward, reward_std, pass@1/pass@k, verifier status. |
| Distillation | Compression: move teacher behavior into a smaller student. | Teacher outputs or teacher-on-policy budget. | Student pass@k and quality against teacher baseline. |
| Session Distillation | Local repair: train the original context toward the same action rescored under a targeted hint. | `session_distillation_records.jsonl` with failed spans and target masks. | Masked KL/CE plus heldout decision and terminal-task behavior. |
| Cascade RL | Curriculum: train domain specialists in stages, then merge or distill. | Enough examples per domain. | Per-domain gates and final generalist holdout. |
| DPPO replay | Terminal rollout optimization with behavior/train logprob replay and trust-region masks. | Served-model rollouts with logprobs and a backend such as verl/SkyRL/open-instruct. | Mask telemetry, reward, pass@k, backend smoke artifacts. |

Start with SFT unless you already have a verifier-rich terminal environment and
reward variation. SFT teaches the student the language and shape of the task.
RL improves outcomes only after the model can produce attempts worth comparing.

---

## The basic training path

1. Import or capture traces.

   Use the existing agent history first. Hooks are for future sessions; the
   backlog gives the gym data on day one.

2. Classify and curate.

   Promote verified, complete sessions to gold. Keep failed or weak sessions for
   DPO negatives, but do not mix them into SFT as if they were success examples.

3. Generate examples.

   SFT examples use structured tool-call chat messages. DPO examples must pair a
   chosen and rejected answer for the same prompt.

4. Train the first student with SFT.

   Use LoRA/QLoRA on constrained local hardware only when the selected
   model/backend supports it. Use a registered private compute target for larger
   models, longer sequences, or full fine-tunes; hosted compute is optional.

5. Evaluate before routing.

   Run heldout trace evals, executable environment pass@k, spurious-reward
   controls, tamper canaries, and any relevant public benchmark ingest before
   treating a model as shippable.

6. Route conservatively.

   The student does not need to replace the teacher everywhere. Route narrow
   tasks it passes, fall back to the teacher when confidence or gates are weak,
   then collect the new traces for the next cycle.

---

## First-run tutorial

For a new operator, use this order before changing advanced knobs:

1. Read the plan, not just the settings.

   ```bash
   bashgym training plan --strategy sft --hardware local_24gb --json
   ```

   Start with `starting_settings`, then read `settings_help`, `metric_guide`,
   `readiness_ladder`, and `adjustment_rules`.

2. Make one small SFT baseline.

   Keep the first run short. The goal is to prove data loading, chat template,
   loss masking, metrics logging, and checkpoint writing.

3. Analyze the run.

   ```bash
   bashgym training analyze --run-id <run-id> --json
   ```

   Fix missing metrics, truncation, OOM, or verifier issues before trying RL.

4. Attach behavior evidence.

   Run heldout trace eval or executable environment pass@k. Do not use loss as
   the only success signal.

5. Move to GRPO/DPPO only when the baseline can produce attempts worth scoring.

   For terminal RL, the first question is whether reward groups have contrast.
   If `reward_std` is zero or pass@k is all zero, improve curriculum or SFT
   before scaling RL.

6. Use ECHO/RWML as auxiliary diagnostics.

   World-model quality metrics are useful for curriculum and platform learning,
   but they are not release gates until correlated with pass@k and safety.

7. Save the private compute step for finalization.

   Generate a smoke bundle locally first. Then run the installed-backend
   compute-target smoke only after replay, logprobs, and backend-launch artifacts
   are ready.

---

## Where world models fit

BashGym's JEPA-style world-model work is about predicting useful latent terminal
dynamics, not every raw terminal byte. The transition of interest is:

```text
task + command/history -> next observation, diff, test state, verifier state
```

Two objectives are wired today:

| Objective | Adds | Current status |
|---|---|---|
| ECHO | Auxiliary observation-token prediction loss for terminal outputs. | Config and replay contract are wired; trainer backend integration is the next step. |
| RWML | Embedding-space reward for predicting next terminal state. | Pure reward/transition layer and replay contract are wired; real backend loop is pending. |

Treat ECHO/RWML metrics as diagnostics and auxiliary learning signals until they
correlate with heldout pass@k and safety metrics. Do not use them as release
gates by themselves.

---

## Read next

- [capability-map.md](capability-map.md) - full training/eval capability map and stable vs backend-dependent status.
- [platform-flywheels.md](platform-flywheels.md) - segmented product flywheels from coding traces to training, eval, rewards, source library, compute, and education.
- [training-methods-reference.md](training-methods-reference.md) - method-by-method training reference for operators and AI/ML reviewers.
- [external-review-packet.md](external-review-packet.md) - shareable reviewer packet with capabilities, limits, risks, and feedback questions.
- [rlhf-handbook-comparison.md](rlhf-handbook-comparison.md) - RLHF Book comparison with BashGym strengths, gaps, answered reviewer questions, and action plan.
- [strategy-guide.md](strategy-guide.md) - concrete starting settings and when to use each strategy.
- [session-distillation.md](session-distillation.md) - targeted self-distillation from failed trace spans.
- [tmax-terminal-rl-recipe.md](tmax-terminal-rl-recipe.md) - environment-to-replay-to-backend recipe for terminal RL.
- [private-compute-eval-checklist.md](private-compute-eval-checklist.md) - local/private compute backend-smoke and eval checklist.
- [world-models.md](world-models.md) - ECHO/RWML contracts, defaults, replay telemetry, and boundaries.
- [metrics-runbook.md](metrics-runbook.md) - how to diagnose flat pass@k, zero reward variance, timeouts, verifier errors, and tamper attempts.
- [glossary.md](glossary.md) - compact definitions for the training vocabulary.
- [agent-cli.md](agent-cli.md) - machine-readable CLI commands agents can call for setup and replay analysis.
- [autoresearch-campaign.md](autoresearch-campaign.md) - durable AutoResearch setup, control smoke, bindings, and real-evidence path.
- [../TRAINING_DATA_GUIDE.md](../TRAINING_DATA_GUIDE.md) - trace format and data pipeline reference.
- [../training-config-guide.md](../training-config-guide.md) - existing Training Config panel reference.

## Source references

- [../../bashgym/gym/trainer.py](../../bashgym/gym/trainer.py)
- [../../bashgym/gym/terminal_rl.py](../../bashgym/gym/terminal_rl.py)
- [../../bashgym/eval/dppo_replay.py](../../bashgym/eval/dppo_replay.py)
