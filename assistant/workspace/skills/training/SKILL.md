---
name: training
description: "Start, stop, pause, resume, and monitor model training runs. Check training progress, view metrics, and get GPU utilization during training. Use when asked to train a model, check training status, view loss curves, stop a run, or see training logs."
---

# Training

Manage SFT/DPO/GRPO training runs on local hardware.

## Start Training

```
scripts/api.sh POST /training/start '{
  "strategy": "sft",
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "selected_repos": [],
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 2e-5
}'
```

Strategy options: `sft`, `dpo`, `grpo`
Empty `selected_repos` means train on all gold traces.

## Check Training Status

```
scripts/api.sh GET /training/{run_id}
```

Returns: epoch, loss, learning rate, elapsed time, estimated completion.

## List Training Runs

```
scripts/api.sh GET /training
```

## Pause Training

```
scripts/api.sh POST /training/{run_id}/pause '{}'
```

## Resume Training

```
scripts/api.sh POST /training/{run_id}/resume '{}'
```

## Stop Training

**Destructive — confirm with user first.**
```
scripts/api.sh POST /training/{run_id}/stop '{}'
```

## GPU Utilization During Training

```
scripts/api.sh GET /system/info
```

Report GPU utilization %, memory used/total, temperature if available.

## Example Interactions

User: "Start training on my gold traces"
→ Call POST /training/start with defaults. Report run_id and initial status.

User: "How's training going?"
→ Call GET /training to find active run. Call GET /training/{run_id} for metrics.
   Also call GET /system/info for GPU stats. Report epoch, loss, GPU %.

User: "Stop the current training"
→ Confirm with user. Find active run_id. Call POST /training/{run_id}/stop.
