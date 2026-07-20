---
name: traces
description: 'Browse, promote, demote, classify, and generate training examples from Claude Code execution traces. Use when asked about traces, gold traces, trace counts, promoting or demoting traces, generating examples, exporting training data, or syncing traces.'
---

# Traces

Manage Claude Code execution traces and convert them to training examples.

## List Traces

Filter by status (gold, pending, failed):

```text
bashgym api GET /api/traces --query status=gold
bashgym api GET /api/traces --query status=pending
bashgym api GET /api/traces
```

## Trace Statistics

Quick counts by status:

```text
bashgym api GET /api/traces/stats
```

## List Repositories

See which repos have traces:

```text
bashgym api GET /api/traces/repos
```

## Get Gold Traces

```text
bashgym api GET /api/traces/gold
```

## Promote a Trace

Move a pending trace to gold status:

```text
bashgym api POST /api/traces/{trace_id}/promote
```

## Demote a Trace

**Destructive — confirm with user first.**
Move a gold trace back to failed:

```text
bashgym api POST /api/traces/{trace_id}/demote
```

## Generate Training Examples

Convert a trace into training examples:

```text
bashgym api POST /api/traces/{trace_id}/generate-examples
```

## List Training Examples

```text
bashgym api GET /api/training/examples
```

## Export to NeMo Format

Export generated examples for training:
Save `{}` as `export-request.json`, then run:

```text
bashgym api POST /api/training/export --data-file export-request.json
```

## Sync Traces

Import new traces from Claude Code history:

```text
bashgym api POST /api/traces/sync
```

## Auto-Classify Traces

Automatically classify pending traces:

```text
bashgym api POST /api/traces/auto-classify
```

## Example Interactions

User: "How many gold traces do I have?"
→ Call `bashgym api GET /api/traces/stats`. Report counts per status.

User: "Promote trace abc123"
→ Call `bashgym api POST /api/traces/abc123/promote`. Confirm success.

User: "Generate training data from my gold traces"
→ Call `bashgym api GET /api/traces/gold` to list them. For each, call `bashgym api POST /api/traces/{id}/generate-examples`.
Then call `bashgym api POST /api/training/export --data-file export-request.json` to create NeMo JSONL.

User: "Sync new traces"
→ Call `bashgym api POST /api/traces/sync`. Report how many new traces were imported.
