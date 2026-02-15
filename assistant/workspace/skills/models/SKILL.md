---
name: models
description: "Browse, compare, evaluate, and manage trained models. View leaderboard, trends, lineage, and artifacts. Deploy models to Ollama. Use when asked about models, model comparison, which model is best, model evaluation, deploying a model, or downloading a model."
---

# Models

Browse and manage the model registry.

## List Models

```
scripts/api.sh GET /models
```

## Leaderboard

Ranked models by evaluation score:
```
scripts/api.sh GET /models/leaderboard
```

## Trends

Model performance over time:
```
scripts/api.sh GET /models/trends
```

## Compare Models

```
scripts/api.sh POST /models/compare '{"model_ids": ["model-a", "model-b"]}'
```

## Get Model Details

```
scripts/api.sh GET /models/{model_id}
```

## Evaluate a Model

Run evaluation benchmarks:
```
scripts/api.sh POST /models/{model_id}/evaluate '{}'
```

## Deploy to Ollama

Make a model available locally via Ollama:
```
scripts/api.sh POST /models/{model_id}/deploy-ollama '{}'
```

## Delete a Model

**Destructive — confirm with user first.**
```
scripts/api.sh DELETE /models/{model_id}
```

## Model Artifacts

View training artifacts (configs, checkpoints):
```
scripts/api.sh GET /models/{model_id}/artifacts
```

## Example Interactions

User: "Which model is best?"
→ Call GET /models/leaderboard. Report top 3 with scores.

User: "Compare model-a and model-b"
→ Call POST /models/compare. Present side-by-side metrics.

User: "Deploy my latest model to Ollama"
→ Call GET /models to find latest. Call POST /models/{id}/deploy-ollama.
