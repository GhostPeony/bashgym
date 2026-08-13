---
name: factory
description: 'Generate synthetic training data, manage seeds, preview synthesis jobs, and work with the data factory. Use when asked to generate synthetic data, create training examples, augment a dataset, manage seeds, or check factory job status.'
---

# Factory

Synthetic data generation and training data management.

## Seeds

List existing seeds:

```text
bashgym api GET /api/factory/seeds
```

Create a new seed:
Save `{"data":{"content":"..."},"source":"manual"}` as `seed-request.json`, then run:

```text
bashgym api POST /api/factory/seeds --data-file seed-request.json
```

Create seeds from gold traces:
Save `{}` as `empty-request.json`, then run:

```text
bashgym api POST /api/factory/seeds/from-traces --data-file empty-request.json
```

Delete a seed:

```text
bashgym api DELETE /api/factory/seeds/{seed_id}
```

## Synthesis

Preview what synthesis would produce:
Save `{"row_count":50}` as `preview-request.json`, then run:

```text
bashgym api POST /api/factory/preview --data-file preview-request.json
```

Start synthesis job:
Save `{"row_count":10}` as `synthesis-request.json`, then run:

```text
bashgym api POST /api/factory/synthesize --data-file synthesis-request.json
```

## Synthetic Data Generation

Generate synthetic training examples:
Save `{"preset":"custom","target_examples":50}` as
`synthetic-request.json`, then run:

```text
bashgym api POST /api/factory/synthetic/generate --data-file synthetic-request.json
```

List synthetic generation jobs:

```text
bashgym api GET /api/factory/synthetic/jobs
```

Check job status:

```text
bashgym api GET /api/factory/synthetic/jobs/{job_id}
```

Available presets:

```text
bashgym api GET /api/factory/synthetic/presets
```

## Factory Configuration

Read the current full configuration, edit that schema into `factory-config.json`,
then update it:

```text
bashgym api GET /api/factory/config
bashgym api PUT /api/factory/config --data-file factory-config.json
```

## Example Interactions

User: "Generate 50 synthetic training examples"
→ Call `bashgym api GET /api/factory/synthetic/presets` to show options. Ask which preset.
Call `bashgym api POST /api/factory/synthetic/generate --data-file synthetic-request.json`. Report job_id.

User: "Create seeds from my gold traces"
→ Call `bashgym api POST /api/factory/seeds/from-traces --data-file empty-request.json`. Report how many seeds created.

User: "What seeds do I have?"
→ Call `bashgym api GET /api/factory/seeds`. List source and data previews.
