# Session Distillation

Session Distillation is BashGym's implementation lane for targeted on-policy
self-distillation. It is designed for the core BashGym flywheel: coding traces
leave behind mistakes, verifier failures, retry loops, and recovery decisions;
those local failure spans become training records.

The mechanism is intentionally narrow:

```text
student trace
  -> reader finds a local mistake span
  -> BashGym inserts a short hint before that span
  -> the same target action tokens are scored under original and hinted context
  -> masked KL/CE trains the original context toward the hinted distribution
```

This is not teacher distillation. A stronger teacher does not rewrite the whole
trajectory. The model learns from its own action under better local context.

## When To Use It

Use Session Distillation when traces contain:

- failed commands or missing paths
- failed tests followed by a recovery pivot
- repeated retries of nearly the same action
- verifier failures where the next step reveals the local fix
- tool-choice mistakes that are too small for DPO and too local for GRPO

Do not use it as a broad replacement for SFT, DPO, or terminal RL. It is best
for repairing local decisions inside otherwise useful traces.

## Method Comparison

| Method               | Learns from                                          | Best fit                                               | BashGym artifact                     |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------ |
| SFT                  | Clean successful examples                            | Teach format, tool use, and project style              | `messages` JSONL                     |
| DPO                  | Chosen/rejected pairs for the same prompt            | Teach preferences between alternate answers            | `dpo_pairs.jsonl`                    |
| Teacher distillation | A stronger model's outputs or logits                 | Compress teacher behavior into a smaller model         | teacher/student training data        |
| Session Distillation | The student's own action with an inserted local hint | Repair failed actions without replacing the trajectory | `session_distillation_records.jsonl` |
| GRPO/RLVR            | Multiple sampled attempts scored by a verifier       | Improve outcomes when reward groups have contrast      | terminal environment rollouts        |
| ECHO/RWML            | Terminal transition prediction and embedding rewards | Diagnostics and auxiliary world-model learning         | DPPO/world-model replay payloads     |

TRL remains the stable external reference for SFT, DPO, reward modeling, PPO,
and GRPO trainers. Unsloth is useful when local or private GPU training needs
lower VRAM and faster SFT/DPO/GRPO runs. Session Distillation uses those stacks
only after BashGym has produced the record contract and verified the mask.

## Artifact Contract

The first-class artifact is `session_distillation_records.jsonl`. Each row must
include:

| Field               | Meaning                                                          |
| ------------------- | ---------------------------------------------------------------- |
| `original_context`  | Context before the target action, without the hint.              |
| `hinted_context`    | Same context with `[Session Distillation Hint]` inserted.        |
| `hint_text`         | The short local correction.                                      |
| `target_text`       | The exact action tokens to train.                                |
| `target_span`       | Character span of the target inside `target_text`.               |
| `loss_mask`         | Currently `target_span_only`.                                    |
| `reader_model`      | `heuristic-session-distillation-reader-v1` or a model reader id. |
| `reader_confidence` | Confidence used to filter weak hints.                            |
| `verifier_outcome`  | Failed verifier/trace signal that justified the record.          |
| `source_metadata`   | Trace, split, mistake type, and provenance details.              |

The original context must not contain the hint tag. The hinted context must
contain it. The target text stays identical between both contexts.

## Default Settings

| Setting                               |         Start here | Why                                                                  |
| ------------------------------------- | -----------------: | -------------------------------------------------------------------- |
| `session_distillation_alpha`          |              `0.7` | Put more weight on hinted-context KL while preserving hard-label CE. |
| `session_distillation_temperature`    |              `1.0` | Avoid over-softening until we have run-level calibration evidence.   |
| `session_distillation_min_confidence` |              `0.6` | Keep obvious failed-action hints and drop weak reader guesses.       |
| `session_distillation_mask_policy`    | `target_span_only` | Prevent the loss from updating unrelated transcript tokens.          |
| `session_distillation_context_mode`   |    `hint_injected` | Matches the one-rollout mechanism.                                   |
| `session_distillation_reader`         |        `heuristic` | Deterministic, cheap, and auditable for the first implementation.    |

Raise the confidence threshold if hints are noisy. Lower alpha if the model
overfits to hints and loses ordinary action quality. Increase data only after
held-out decision accuracy moves, not just because loss falls.

## Metrics To Track

Training metrics:

- `session_distillation_loss`
- `session_distillation_kl`
- `session_distillation_ce`
- `session_distillation_masked_tokens`
- accepted vs skipped records
- reader confidence distribution

Behavior metrics:

- held-out decision accuracy on failed-action/recovery cases
- tool-call validity
- command/recovery-choice accuracy
- terminal-task pass@1/pass@k where environments exist
- timeout, tamper, verifier-error, and forgetting checks

Promotion should fail closed when records, masked-loss metrics, or held-out
release evidence are missing.

## Data Sources For Feasibility

Use BashGym's own traces first because they preserve repo context, commands,
outputs, edits, tests, and recovery decisions. Public agent-trajectory datasets
are useful for fixtures, stress tests, and schema mapping, not as blind defaults.

Candidate external sources:

| Source                             | Useful for                                       | Guardrail                                                                    |
| ---------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| SWE-agent trajectories             | Failure/recovery trajectories and command traces | Re-map into BashGym records and hold out by repo/session.                    |
| OpenHands/SWE-rebench trajectories | Multi-turn agent traces with observations        | Treat as source-library input with provenance and split manifests.           |
| SWE-smith trajectories             | Fine-tuning examples from agent issue solving    | Use for SFT/DPO comparisons, not as proof of Session Distillation by itself. |
| SWE-Zero/OpenHands trajectories    | Large-scale execution-style traces               | Cap pulls, inspect schema, and decontaminate before training.                |
| SWE-bench                          | Evaluation and issue/PR grounding                | Prefer eval/holdout use unless converted records pass source policy.         |

## Running The Loop

The full experiment loop is: build records from traces, validate them, smoke a
tiny local run, smoke on a private compute target, then compare on a held-out
slice. Build records straight from a directory of trace files:

```bash
# Build session_distillation_records.jsonl from failed/gold traces.
# Clean sessions are skipped; hints below --min-confidence are dropped.
bashgym training session-records build data/failed_traces \
  --out data/session_distillation/records.jsonl \
  --min-confidence 0.6 --limit 200 --split train --json
```

The command runs the heuristic reader per session, validates every record, and
only writes the JSONL when all records pass (the report's `validation_errors`
must be empty). `trace_id` is the trace file stem, since traces carry no
`trace_id` on disk.

Then point a training run at that JSONL:

```bash
# Local tiny smoke (LoRA, bf16 on GPU / fp32 on CPU). Keep it small first.
bashgym training … (strategy=session_distillation, dataset=data/session_distillation/records.jsonl)
```

The generated script trains LoRA adapters in bf16 on CUDA (full fine-tune and
fp16 AMP are avoided so it fits consumer/unified-memory GPUs and does not crash
on the fp16 gradient-unscale path). Remote launches upload the
session-distillation script, run it with remote-relative paths, and are not
gated on Unsloth (the backend is plain transformers).

## Implementation Status

Code-backed today:

- `bashgym.factory.session_distillation` builds and validates records, and
  `build_session_distillation_records_from_traces` / the
  `training session-records build` CLI produce the JSONL from trace files.
- The heuristic reader creates records for failed command/tool traces and skips
  clean traces (successful steps are never flagged as mistakes, even when their
  output contains failure keywords).
- `strategy=session_distillation` is wired through trainer/API schemas.
- The generated trainer script loads bf16 + LoRA on CUDA and applies masked
  hinted-context KL plus hard CE; the remote path runs the correct script with
  remote-relative paths and without an Unsloth gate.
- Training Config exposes the method and starter knobs.
- RunCards require records, reader/mask metadata, masked-loss metrics, and
  release evidence for promotion.

Contexts are re-scored through the tokenizer's chat template when one exists
(falling back to plain text for template-less smoke models), and the
`session_distillation_loss/kl/ce/masked_tokens` metrics are parsed from training
stdout into the run metrics so run-analysis safety gates (e.g. the
zero-masked-tokens check) see real data. RunCard evidence can be attached from
the CLI: `bashgym training runcard create --training-method session_distillation
--session-distillation-records … --session-distillation-metrics … --reader-model
… --confidence-threshold … --hint-policy … --mask-policy target_span_only
--target-token-count …` (and `runcard attach-evidence --session-distillation-*`).

Records must currently mask the whole `target_text` (the validator rejects a
partial `target_span` until sub-span masking is implemented).

Still needs runtime proof:

- local tiny LoRA smoke with a small fixture model on GPU (bf16)
- private compute preflight only after local contracts pass
- held-out decision/eval slices that compare SFT, teacher distillation, and
  Session Distillation on the same trace families

## References

- [Hugging Face TRL docs](https://huggingface.co/docs/trl/index)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth reinforcement learning guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
- [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench)
