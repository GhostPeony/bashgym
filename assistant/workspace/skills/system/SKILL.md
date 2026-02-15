---
name: system
description: "Check BashGym system status, GPU utilization, hardware info, and aggregate statistics. Use when asked about system health, what's running, GPU usage, memory, disk space, or general system stats."
---

# System Status

Query BashGym system health and hardware information.

## Health Check

Quick service health check:
```
scripts/api.sh GET /health
```

Returns: `{"status": "ok", "version": "..."}`

## System Information

Full hardware report (CPU, GPU, memory, disk):
```
scripts/api.sh GET /system/info
```

Returns GPU utilization, CUDA availability, memory usage, disk space.

## GPU Details

Dedicated GPU endpoint:
```
scripts/api.sh GET /system/gpus
```

## Aggregate Statistics

Cross-subsystem stats (trace counts, model counts, training runs):
```
scripts/api.sh GET /stats
```

## Model Recommendations

System-recommended models based on available hardware:
```
scripts/api.sh GET /system/recommendations
```

## Example Interactions

User: "How's the system doing?"
→ Call /health, then /system/info for GPU/memory. Summarize key numbers.

User: "Is my GPU being used?"
→ Call /system/gpus. Report utilization %, memory used/total.

User: "Give me a status report"
→ Call /stats for counts, /system/info for hardware. Present as a concise dashboard.
