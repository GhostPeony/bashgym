# BashGym Agent Launch Recipes

Use this reference immediately before a direct training launch. Copy exact field names into the run request, replace placeholders, and save the final request with the run evidence. These recipes are starting envelopes, not permission to skip dataset, compute, evaluation, cost, or publication gates.

Pair this method reference with `compute-target-activation.md`. The target must
activate a real backend: `local`, `ssh:<device_id>`, or
`cloud:nemo-customizer` for the
direct BashGym endpoint. Hugging Face Jobs and managed providers use their own
submission surfaces.

## Common launch contract

Every direct run must identify:

- strategy, base model, approved dataset revision, and compute target;
- measurable objective, baseline, development evaluation, stop rule, and promotion gate;
- LoRA/QLoRA versus full fine-tune intent and hardware fit;
- `checkpoint_limit` and `artifact_retention`;
- local/remote artifact destination and cleanup owner;
- whether Hugging Face upload is enabled, the repository, visibility, and selected artifact.

Use this storage block for routine experiments:

```json
{
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto"
}
```

`adapter_only` still writes resumable checkpoints during training; it removes them only after the final adapter saves successfully. Use `adapter_checkpoint` for completed runs that must remain resumable, `deployable` for a serving candidate that needs merged weights, and `full_run` only when checkpoints plus deployable artifacts must all survive.

For off-device durability without a second full-model copy, enable `auto_push_hf`, provide `hf_repo_name`, keep `hf_private: true`, and use `hf_upload_artifact: "adapter"`. Public visibility requires explicit release approval.

## SFT

Dataset: verified messages/text or gold traces. Failed attempts must not be labeled as successful SFT targets.

```json
{
  "strategy": "sft",
  "dataset_path": "/path/to/sft-train.jsonl",
  "base_model": "<student-model-id>",
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.00002,
  "max_seq_length": 4096,
  "use_lora": true,
  "lora_rank": 16,
  "lora_alpha": 32,
  "load_in_4bit": false,
  "sft_backend": "unsloth",
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
  "use_remote_ssh": true,
  "device_id": "<device-id>"
}
```

Required after training: saved metrics, heldout behavior, and environment pass@k/holdout gate for terminal-facing models.

## DPO

Dataset: same-prompt `chosen`/`rejected` preference pairs. Start from an SFT checkpoint when the student does not yet reliably produce the desired format.

```json
{
  "strategy": "dpo",
  "dataset_path": "/path/to/dpo-pairs.jsonl",
  "base_model": "<sft-or-base-model-id>",
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.00001,
  "max_seq_length": 4096,
  "dpo_beta": 0.1,
  "dpo_backend": "auto",
  "use_lora": true,
  "load_in_4bit": false,
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
  "use_remote_ssh": true,
  "device_id": "<device-id>"
}
```

Required after training: strict pair validation, preference metrics, heldout preference behavior, and no regression against the SFT/base checkpoint.

## GRPO or RLVR

Dataset: prompts or environment specifications with an executable/verifiable reward. Use `strategy: "rlvr"` when the reward is explicitly a deterministic verifier; the remaining fields can stay the same.

```json
{
  "strategy": "grpo",
  "dataset_path": "/path/to/prompts-or-env-specs.jsonl",
  "base_model": "<sft-warm-start-model-id>",
  "training_profile": "terminal_rl_tmax_like",
  "grpo_backend": "auto",
  "grpo_loss_type": "dapo",
  "grpo_reward_mode": "verification",
  "grpo_group_size": 32,
  "grpo_num_generations": 32,
  "grpo_temperature": 0.7,
  "filter_zero_std_groups": true,
  "active_sampling": true,
  "token_level_loss": true,
  "lm_head_fp32": true,
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
  "use_remote_ssh": true,
  "device_id": "<device-id>"
}
```

Required after training: rollout pass@k, reward variance/zero-standard-deviation diagnostics, environment holdout gate, reward-hacking canaries, and release evidence.

## Teacher distillation

Dataset: approved teacher outputs or training records compatible with the selected distillation path. Pin teacher identity and generation/config provenance.

```json
{
  "strategy": "distillation",
  "dataset_path": "/path/to/distillation-train.jsonl",
  "base_model": "<student-model-id>",
  "teacher_model": "<teacher-model-id>",
  "teacher_temperature": 0.7,
  "distillation_alpha": 0.5,
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.00002,
  "max_seq_length": 4096,
  "use_lora": true,
  "load_in_4bit": false,
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
  "use_remote_ssh": true,
  "device_id": "<device-id>"
}
```

Required after training: student loss/divergence evidence plus heldout comparison against both the base student and teacher.

## Session Distillation

Dataset: validated `session_distillation_records.jsonl` built from real failed or recovery-rich spans. This is a local mistake-repair lane, not general teacher distillation.

```json
{
  "strategy": "session_distillation",
  "dataset_path": "/path/to/session_distillation_records.jsonl",
  "base_model": "<student-model-id>",
  "session_distillation_alpha": 0.7,
  "session_distillation_temperature": 1.0,
  "session_distillation_min_confidence": 0.6,
  "session_distillation_mask_policy": "target_span_only",
  "session_distillation_context_mode": "hint_injected",
  "session_distillation_reader": "heuristic",
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 4096,
  "use_lora": true,
  "load_in_4bit": false,
  "checkpoint_limit": 1,
  "artifact_retention": "adapter_only",
  "auto_push_hf": false,
  "hf_private": true,
  "hf_upload_artifact": "auto",
  "use_remote_ssh": true,
  "device_id": "<device-id>"
}
```

Required after training: record validation, masked-token loss/KL/CE metrics, heldout recovery behavior, and terminal pass@k when applicable.

## CLI and direct agent tool

Save only method-specific overrides in a bounded JSON object, then launch:

```bash
bashgym training start \
  --strategy dpo \
  --model <model-id> \
  --dataset-path /path/to/dpo-pairs.jsonl \
  --compute-target ssh:<device-id> \
  --config dpo-config.json \
  --checkpoint-limit 1 \
  --artifact-retention adapter_only \
  --json
```

The `start_training` agent tool accepts the same direct strategies. Put method and storage overrides under its `config` object. The tool validates field names against `TrainingRequest` and rejects attempts to hide strategy, model, dataset target, correlation, or origin inside the nested config.

`--compute-target ssh:<device_id>` and the equivalent agent-tool field now set
`use_remote_ssh: true` plus the matching `device_id`; they are not provenance-only
labels. An ambiguous `cloud` or `hf-jobs` value is rejected by the direct launch
path so an agent cannot accidentally run locally while claiming cloud compute.

## Workflows that are not direct strategies

- DPPO uses rollout capture, replay enrichment, smoke-plan/smoke-bundle, and an installed external backend. Do not send `strategy: "dppo"` to `/api/training/start`.
- ECHO/RWML are auxiliary diagnostics enabled inside a compatible terminal-RL/backend workflow. They are not standalone promotion evidence.
- Reward-model/ORM/PRM work uses reward artifact validation, fixture smoke/eval, and a selected real reward-model backend. Do not call fixture smoke a trained production model.
- Cascade/MOPD uses the cascade endpoints and per-stage evidence. Each stage must preserve its own method config and artifact policy.
