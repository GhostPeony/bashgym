# Bash Gym API Reference

The Bash Gym API server provides REST and WebSocket endpoints for all platform operations.

**Interactive docs:** When the server is running, visit [`http://localhost:8003/docs`](http://localhost:8003/docs) for auto-generated Swagger UI with request/response schemas.

---

## Starting the Server

```bash
# Development (hot reload)
python run_backend.py

# Production
python start_api.py
# Or: uvicorn bashgym.api.routes:app --host 0.0.0.0 --port 8003 --workers 4
```

---

## REST Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | System statistics |
| `GET` | `/api/system/info` | Hardware & GPU info |

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/tasks` | Submit new task |
| `GET` | `/api/tasks/{task_id}` | Get task status |
| `GET` | `/api/tasks` | List all tasks |
| `POST` | `/api/tasks/{task_id}/cancel` | Cancel task |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/training/start` | Start training run |
| `GET` | `/api/training/{run_id}` | Get training status |
| `GET` | `/api/training` | List training runs |
| `POST` | `/api/training/{run_id}/pause` | Pause training |
| `POST` | `/api/training/{run_id}/cancel` | Cancel training |
| `GET` | `/api/training/{run_id}/logs` | Stream training logs |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List trained models |
| `GET` | `/api/models/{model_id}` | Get model profile |
| `POST` | `/api/models/{model_id}` | Update model profile |
| `DELETE` | `/api/models/{model_id}` | Delete model |
| `POST` | `/api/models/{model_id}/star` | Star/unstar model |
| `POST` | `/api/models/{model_id}/evaluate` | Run evaluation |
| `GET` | `/api/models/{model_id}/artifacts` | Get model artifacts |
| `POST` | `/api/models/{model_id}/rescan` | Rescan artifacts |
| `POST` | `/api/models/{model_id}/run-eval` | Run custom eval |
| `POST` | `/api/models/{model_id}/deploy-ollama` | Deploy to Ollama |
| `GET` | `/api/models/{model_id}/download` | Download model |
| `GET` | `/api/models/leaderboard` | Model leaderboard |
| `GET` | `/api/models/trends` | Performance trends |
| `POST` | `/api/models/compare` | Compare multiple models |
| `GET` | `/api/models/eval-sets` | List evaluation sets |
| `POST` | `/api/models/eval-sets/generate` | Generate eval set from traces |
| `DELETE` | `/api/models/eval-sets/{eval_set_id}` | Delete eval set |

### Traces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/traces` | List traces (filter by status, repo) |
| `GET` | `/api/traces/{trace_id}` | Get trace details |
| `POST` | `/api/traces/{trace_id}/promote` | Promote to gold |
| `POST` | `/api/traces/{trace_id}/demote` | Demote to failed |
| `POST` | `/api/traces/{trace_id}/generate-examples` | Generate training examples |
| `POST` | `/api/traces/import` | Import new sessions from `~/.claude/projects/` |
| `GET` | `/api/traces/import-since` | Count traces imported since a given ISO timestamp (`?since=`) |
| `POST` | `/api/traces/auto-classify` | Auto-classify pending traces into quality tiers |
| `POST` | `/api/traces/sync` | Sync traces from `~/.bashgym/` into the project data directory |

### Factory

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/factory/synthetic/generate` | Start synthetic generation |
| `GET` | `/api/factory/synthetic/jobs/{job_id}` | Get generation progress |
| `GET` | `/api/factory/synthetic/jobs` | List all synthesis jobs |
| `GET` | `/api/factory/synthetic/presets` | List generation presets |
| `POST` | `/api/factory/designer/preview` | Preview synthetic data |
| `POST` | `/api/factory/designer/create` | Create designer job |
| `POST` | `/api/factory/designer/validate` | Validate schema |
| `POST` | `/api/factory/designer/from-hf` | Load from HuggingFace |
| `POST` | `/api/factory/designer/push-to-hub` | Push to HuggingFace Hub |

### Benchmarks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/benchmarks` | List available benchmarks |
| `POST` | `/api/benchmarks/run` | Run benchmark |
| `GET` | `/api/benchmarks/{run_id}` | Get results |

### Agent (Chat Assistant)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agent/chat` | Send chat message to Peony (returns response + optional `pending_action`) |
| `POST` | `/api/agent/confirm-action` | Approve or deny a pending shell command |
| `GET` | `/api/agent/sessions` | List chat sessions |
| `GET` | `/api/agent/sessions/{session_id}` | Get session details |
| `POST` | `/api/agent/sessions` | Save session |
| `DELETE` | `/api/agent/sessions/{session_id}` | Delete session |

### HuggingFace

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/hf/status` | Check HF integration |
| `POST` | `/api/hf/configure` | Configure HF credentials |
| `DELETE` | `/api/hf/configure` | Remove HF configuration |
| `GET` | `/api/hf/jobs` | List cloud training jobs |
| `POST` | `/api/hf/jobs` | Submit cloud training job (SFT/DPO/Distillation) |
| `GET` | `/api/hf/jobs/hardware` | Get hardware tiers with pricing |
| `GET` | `/api/hf/jobs/{job_id}` | Get job status |
| `GET` | `/api/hf/jobs/{job_id}/logs` | Stream job logs |
| `DELETE` | `/api/hf/jobs/{job_id}` | Cancel job |
| `GET` | `/api/hf/models/search` | Search HuggingFace Hub (`?task=&sort=downloads&limit=10`) |
| `POST` | `/api/hf/evaluate` | Evaluate a model with accuracy, F1, BLEU, or ROUGE metrics |
| `POST` | `/api/hf/inference/generate` | Use HF inference API |
| `POST` | `/api/hf/inference/embed` | Generate embeddings |
| `GET` | `/api/hf/spaces` | List Spaces |
| `POST` | `/api/hf/spaces` | Create Space |
| `DELETE` | `/api/hf/spaces/{space_name}` | Delete Space |
| `GET` | `/api/hf/datasets` | List datasets |
| `POST` | `/api/hf/datasets` | Upload dataset |
| `DELETE` | `/api/hf/datasets/{repo_name}` | Delete dataset |

### Integration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/integration/status` | Bashbros status |
| `GET` | `/api/integration/settings` | Get settings |
| `PUT` | `/api/integration/settings` | Update settings |
| `POST` | `/api/integration/link` | Link bashbros |
| `POST` | `/api/integration/unlink` | Unlink bashbros |
| `GET` | `/api/integration/traces/pending` | List pending traces |
| `POST` | `/api/integration/traces/process` | Process traces |
| `GET` | `/api/integration/models/versions` | List model versions |
| `POST` | `/api/integration/models/export` | Export model |
| `POST` | `/api/integration/models/rollback` | Rollback model |
| `POST` | `/api/integration/watcher/start` | Start trace watcher |
| `POST` | `/api/integration/watcher/stop` | Stop watcher |

### Observability

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/observability/traces` | List profiler traces |
| `GET` | `/api/observability/traces/{trace_id}` | Get trace details |
| `GET` | `/api/observability/guardrails/events` | Get guardrail events |
| `GET` | `/api/observability/guardrails/stats` | Get guardrail statistics |
| `GET` | `/api/observability/guardrails/dpo-negatives` | Get DPO negatives from blocks |
| `GET` | `/api/observability/metrics` | Get system metrics |
| `POST` | `/api/observability/settings/guardrails` | Configure guardrails |
| `POST` | `/api/observability/settings/profiler` | Configure profiler |

### Achievements

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/achievements` | List achievements |
| `GET` | `/api/achievements/stats` | Get achievement statistics |
| `GET` | `/api/achievements/recent` | Get recent achievements |
| `POST` | `/api/achievements/refresh` | Refresh achievements |

### Security

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/security/datasets` | List security datasets |
| `POST` | `/api/security/ingest` | Ingest security dataset |
| `GET` | `/api/security/jobs/{job_id}` | Get security job status |
| `GET` | `/api/security/jobs` | List security jobs |

---

## WebSocket

Connect to `ws://localhost:8003/ws` for real-time updates.

### Message Types

| Category | Message Types |
|----------|--------------|
| **Training** | `training:progress`, `training:complete`, `training:failed`, `training:log` |
| **Tasks** | `task:status`, `task:complete` |
| **Traces** | `trace:added`, `trace:promoted`, `trace:demoted` |
| **Router** | `router:stats`, `router:decision` |
| **Verification** | `verification:result` |
| **Guardrails** | `guardrail:blocked`, `guardrail:warn`, `guardrail:pii_redacted` |
| **HuggingFace** | `hf:job:started`, `hf:job:log`, `hf:job:completed`, `hf:job:failed`, `hf:job:metrics`, `hf:space:ready`, `hf:space:error` |
| **Integration** | `integration:trace:synced`, `integration:model:exported` |
