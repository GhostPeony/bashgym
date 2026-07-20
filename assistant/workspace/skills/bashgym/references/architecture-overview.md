# BashGym Architecture Overview

BashGym is a task-general ML workspace with specialized data, training, evaluation, model, artifact, research, runtime, campaign, agent, and reporting surfaces. Canvas nodes are views/tools over shared durable state; they are not separate orchestration systems.

## Responsibility split

| Layer                     | Owns                                                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| BashGym operational state | Dataset/model revisions, exact configs, jobs, metrics, evals, artifacts, budgets, reports, and live runtime evidence |
| Agent operator            | Context reconciliation, questions/gates, approved actions, monitoring, comparisons, reports, and bounded iteration   |
| GBrain                    | Durable project context, decisions, conclusions, follow-ups, and resolvable artifact references                      |
| Interaction surface       | Discord, canvas agent node, terminal, Codex, or Claude Code access to the same underlying work                       |

## Training lanes

- Direct causal-LM training: SFT, DPO, GRPO, RLVR, teacher distillation, and Session Distillation through the validated training request.
- Terminal RL extensions: DPPO replay/backend readiness plus optional ECHO/RWML diagnostics.
- Multi-stage work: cascade/MOPD with per-stage configs and gates.
- Reward modeling: strict reward artifacts, selected backend, heldout calibration/bias evaluation, and downstream use-site tests.
- Task profiles: embedding retrieval and other future model types plug into shared session/evidence/reporting contracts without redefining BashGym around one experiment.

## Artifact lifecycle

The base-model cache can be shared, but adapters, checkpoints, merged weights, GGUF exports, and reports are per-run artifacts. `artifact_retention` decides which artifacts survive a successful run, `checkpoint_limit` bounds retained resumable checkpoints, and Hugging Face upload can preserve an adapter or merged artifact off-device. See the training skill for executable policy.

## Agent flow

1. Reconcile live BashGym state with relevant GBrain context.
2. Confirm objective, model, dataset, method, eval contract, compute, budget, storage, publication, and stop/promotion authority.
3. Persist and launch the exact config through a supported BashGym surface.
4. Monitor metrics and runtime health without confusing milestones with quality claims.
5. Run method-appropriate evals against a pinned baseline.
6. Retain/upload/clean artifacts according to the declared policy.
7. Generate reports and curate concise outcomes/references into GBrain.
8. Continue only within the approved budget and decision gates.
