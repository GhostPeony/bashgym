---
name: traces
description: "Browse, promote, demote, classify, and generate training examples from Claude Code execution traces. Use when asked about traces, gold traces, trace counts, promoting or demoting traces, generating examples, exporting training data, or syncing traces."
---

# Traces

Manage Claude Code execution traces and convert them to training examples.

## List Traces

Filter by status (gold, pending, failed):
```
scripts/api.sh GET "/traces?status=gold"
scripts/api.sh GET "/traces?status=pending"
scripts/api.sh GET /traces
```

## Trace Statistics

Quick counts by status:
```
scripts/api.sh GET /traces/stats
```

## List Repositories

See which repos have traces:
```
scripts/api.sh GET /traces/repos
```

## Get Gold Traces

```
scripts/api.sh GET /traces/gold
```

## Promote a Trace

Move a pending trace to gold status:
```
scripts/api.sh POST /traces/{trace_id}/promote '{}'
```

## Demote a Trace

**Destructive — confirm with user first.**
Move a gold trace back to failed:
```
scripts/api.sh POST /traces/{trace_id}/demote '{}'
```

## Generate Training Examples

Convert a trace into training examples:
```
scripts/api.sh POST /traces/{trace_id}/generate-examples '{}'
```

## List Training Examples

```
scripts/api.sh GET /training/examples
```

## Export to NeMo Format

Export generated examples for training:
```
scripts/api.sh POST /training/export '{}'
```

## Sync Traces

Import new traces from Claude Code history:
```
scripts/api.sh POST /traces/sync '{}'
```

## Auto-Classify Traces

Automatically classify pending traces:
```
scripts/api.sh POST /traces/auto-classify '{}'
```

## Example Interactions

User: "How many gold traces do I have?"
→ Call GET /traces/stats. Report counts per status.

User: "Promote trace abc123"
→ Call POST /traces/abc123/promote. Confirm success.

User: "Generate training data from my gold traces"
→ Call GET /traces/gold to list them. For each, call POST /traces/{id}/generate-examples.
   Then call POST /training/export to create NeMo JSONL.

User: "Sync new traces"
→ Call POST /traces/sync. Report how many new traces were imported.
