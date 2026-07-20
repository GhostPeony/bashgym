---
name: system
description: "Check BashGym system status, GPU utilization, hardware info, and aggregate statistics. Use when asked about system health, what's running, GPU usage, memory, disk space, or general system stats."
---

# System Status

Query BashGym system health and hardware information.

## Health Check

Quick service health check:

```text
bashgym api GET /api/health
```

Returns: `{"status": "ok", "version": "..."}`

## System Information

Full hardware report (CPU, GPU, memory, disk):

```text
bashgym api GET /api/system/info
```

Returns GPU utilization, CUDA availability, memory usage, disk space.

## GPU Details

Dedicated GPU endpoint:

```text
bashgym api GET /api/system/gpus
```

## Aggregate Statistics

Cross-subsystem stats (trace counts, model counts, training runs):

```text
bashgym api GET /api/stats
```

## Model Recommendations

System-recommended models based on available hardware:

```text
bashgym api GET /api/system/recommendations
```

## Example Interactions

User: "How's the system doing?"
→ Call `bashgym api GET /api/health`, then `bashgym api GET /api/system/info` for GPU/memory. Summarize key numbers.

User: "Is my GPU being used?"
→ Call `bashgym api GET /api/system/gpus`. Report utilization %, memory used/total.

User: "Give me a status report"
→ Call `bashgym api GET /api/stats` for counts and `bashgym api GET /api/system/info` for hardware. Present as a concise dashboard.
