---
name: bashgym
description: Understand and operate the BashGym ML workspace across datasets, direct LLM training, terminal RL, evaluations, models, artifacts, campaigns, reports, and agent coordination. Use as the stable platform router; load the training or operator skill for executable work.
version: 2.0.0
tags: [bashgym, training, evaluation, agents, artifacts]
---

# BashGym

BashGym is the shared ML workspace. It stores the authoritative operational record for datasets, model/config revisions, runs, metrics, evaluations, artifacts, budgets, and reports. Hermes, Codex, Claude Code, Discord, and canvas nodes are operator/interface choices over that workspace, not separate training systems.

The project-isolated experiment ledger extends the existing campaign SQLite
database. Every official run carries `workspace_id`, `project_id`,
`experiment_id`, `run_id`, `attempt_id`, exact model/dataset/environment version
IDs, evaluation/artifact IDs, source identity, and correlation ID. Query one
project at a time; never infer a project from the most recent conversation.

## Route the task

- For session planning, authority, monitoring, reporting, resumption, and GBrain curation, read [../bashgym-operator/SKILL.md](../bashgym-operator/SKILL.md).
- For selecting, launching, monitoring, stopping, or evaluating a training method, read [../training/SKILL.md](../training/SKILL.md).
- Before a direct LLM launch, read [../training/references/bashgym-launch-recipes.md](../training/references/bashgym-launch-recipes.md).
- Before activating local, SSH, or cloud compute, read [../training/references/compute-target-activation.md](../training/references/compute-target-activation.md).
- For the general architecture boundary, read [references/architecture-overview.md](references/architecture-overview.md).
- For evidence and promotion gates, read [references/eval-capabilities.md](references/eval-capabilities.md).
- For a multi-iteration baseline/hypothesis loop, use the durable campaign API,
  `bashgym campaign doctor`, and campaign ledger surfaces; do not start new
  research on the prototype `/api/autoresearch/*` surface.

Do not treat this router as live run state. Inspect the API/CLI, run manifests, runtime processes, and current GBrain sources before stating what is active or what happened most recently.

## Direct training boundary

`POST /api/training/start`, `bashgym training start`, and the `start_training` agent tool directly support:

- SFT;
- DPO;
- GRPO;
- RLVR;
- teacher distillation;
- Session Distillation.

DPPO replay/backends, ECHO/RWML diagnostics, reward-model training, and cascade/MOPD use separate method-specific workflows. Embedding retriever training is one project/profile lane, not the default product architecture and not a causal-LM training alias.

`compute_target` must select a real execution lane, not just label provenance.
Direct BashGym runs accept `local`, `ssh:<device_id>`, and
`cloud:nemo-customizer`. NeMo RL runs through registered private compute.
Hugging Face Jobs, managed provider fine-tunes, and SkyPilot/dstack plans use
separate surfaces and retain their upstream job identity.

## Artifact defaults

Routine direct runs should begin with:

```json
{
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto"
}
```

Choose `adapter_checkpoint` only for completed runs that must remain resumable, `deployable` only for an intentional standalone serving artifact, and `full_run` only for audited/promoted work. Public Hugging Face upload requires explicit release authority.

## Non-negotiable evidence

- Save the exact strategy/config and dataset/model lineage before launch.
- Use `list_experiment_projects`, `get_experiment_context`, and
  `get_experiment_run` (or `bashgym ledger ...`) to recover current project state.
- Pass a complete training tracking context for official work. Missing context is
  recorded under `unassigned`, not attached to a guessed project.
- A smoke run proves the runtime path, not model quality.
- A loss curve is training evidence, not a promotion decision.
- Use heldout, environment, reward/replay, safety, baseline-comparison, and RunCard evidence appropriate to the method.
- Keep raw logs, high-volume metric series, datasets, checkpoints, and model files in BashGym artifact storage. Curate concise decisions, milestones, findings, and artifact references into GBrain.
- Consume `ledger events` by durable cursor when curating GBrain or an optional
  cloud sink. Do not make a product Supabase database a required BashGym backend.
- Do not merge product repositories to share a model. Exchange versioned artifacts and explicit integration contracts.
