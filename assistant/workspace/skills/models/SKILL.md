---
name: models
description: 'Browse, compare, evaluate, and manage trained models. View leaderboard, trends, lineage, and artifacts. Deploy models to Ollama. Use when asked about models, model comparison, which model is best, model evaluation, deploying a model, or downloading a model.'
---

# Models

Browse and manage the model registry.

## List Models

```text
bashgym api GET /api/models
```

## Leaderboard

Ranked models by evaluation score:

```text
bashgym api GET /api/models/leaderboard
```

## Trends

Model performance over time:

```text
bashgym api GET /api/models/trends
```

## Compare Models

Save `{"model_ids":["model-a","model-b"]}` as `compare-request.json`, then run:

```text
bashgym api POST /api/models/compare --data-file compare-request.json
```

## Get Model Details

```text
bashgym api GET /api/models/{model_id}
```

## Evaluate a Model

Run evaluation benchmarks:

```text
bashgym api POST /api/models/{model_id}/evaluate
```

## Deploy to Ollama

Make a model available locally via Ollama:
Save `{}` as `deploy-request.json`, then run:

```text
bashgym api POST /api/models/{model_id}/deploy-ollama --data-file deploy-request.json
```

## Delete a Model

**Destructive — confirm with user first.**

```text
bashgym api DELETE /api/models/{model_id}
```

## Model Artifacts

View training artifacts (configs, checkpoints):

```text
bashgym api GET /api/models/{model_id}/artifacts
```

## Example Interactions

User: "Which model is best?"
→ Call `bashgym api GET /api/models/leaderboard`. Report top 3 with scores.

User: "Compare model-a and model-b"
→ Call `bashgym api POST /api/models/compare --data-file compare-request.json`. Present side-by-side metrics.

User: "Deploy my latest model to Ollama"
→ Call `bashgym api GET /api/models` to find latest. Call `bashgym api POST /api/models/{id}/deploy-ollama --data-file deploy-request.json`.
