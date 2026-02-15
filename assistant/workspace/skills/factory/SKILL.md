---
name: factory
description: "Generate synthetic training data, manage seeds, preview synthesis jobs, and work with the data factory. Use when asked to generate synthetic data, create training examples, augment a dataset, manage seeds, or check factory job status."
---

# Factory

Synthetic data generation and training data management.

## Seeds

List existing seeds:
```
scripts/api.sh GET /factory/seeds
```

Create a new seed:
```
scripts/api.sh POST /factory/seeds '{"content": "...", "tags": ["python", "refactor"]}'
```

Create seeds from gold traces:
```
scripts/api.sh POST /factory/seeds/from-traces '{}'
```

Delete a seed:
```
scripts/api.sh DELETE /factory/seeds/{seed_id}
```

## Synthesis

Preview what synthesis would produce:
```
scripts/api.sh POST /factory/preview '{"seed_ids": ["seed-1"], "count": 5}'
```

Start synthesis job:
```
scripts/api.sh POST /factory/synthesize '{"seed_ids": ["seed-1"], "count": 10}'
```

## Synthetic Data Generation

Generate synthetic training examples:
```
scripts/api.sh POST /factory/synthetic/generate '{
  "preset": "code_completion",
  "count": 50
}'
```

List synthetic generation jobs:
```
scripts/api.sh GET /factory/synthetic/jobs
```

Check job status:
```
scripts/api.sh GET /factory/synthetic/jobs/{job_id}
```

Available presets:
```
scripts/api.sh GET /factory/synthetic/presets
```

## Factory Configuration

```
scripts/api.sh GET /factory/config
scripts/api.sh PUT /factory/config '{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}'
```

## Example Interactions

User: "Generate 50 synthetic training examples"
→ Call GET /factory/synthetic/presets to show options. Ask which preset.
   Call POST /factory/synthetic/generate. Report job_id.

User: "Create seeds from my gold traces"
→ Call POST /factory/seeds/from-traces. Report how many seeds created.

User: "What seeds do I have?"
→ Call GET /factory/seeds. List with tags and content previews.
