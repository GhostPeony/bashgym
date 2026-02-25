# Comprehensive .claude Data Collection Design

> Capturing everything from `~/.claude/` for training, analytics, and session reconstruction.

**Date:** 2026-02-25
**Status:** Approved
**Approach:** Data Lake (Phase 1) with Session Graph evolution (Phase 2)

---

## Problem Statement

BashGym's trace importer currently reads **one data source**: `~/.claude/projects/*/*.jsonl` (session transcripts). This captures tool_use/tool_result pairs with metadata.

But `~/.claude/` contains **~550MB+ of structured agent behavior data** across 10+ data sources that are completely ignored. This includes subagent conversations, versioned file edit history, implementation plans, task decompositions, full debug logs, user prompt history, and environment snapshots.

### Data Audit (from local `.claude/` analysis)

| Source | Location | Count | Size | Currently Imported? |
|---|---|---|---|---|
| Session transcripts | `projects/*/*.jsonl` | 53+ per project | Large | YES |
| Subagent conversations | `projects/*/subagents/*.jsonl` | Per-session | Variable | NO |
| Large tool outputs | `projects/*/tool-results/` | Per-session | Variable | NO |
| User prompt history | `history.jsonl` | 8,767 lines | 4.6MB | NO |
| Debug logs (API traffic) | `debug/` | 355 files | 449MB | NO |
| File edit history | `file-history/` | 244 sessions | 93MB | NO |
| Todo lists | `todos/` | 378 files | ~1.4MB | NO |
| Task state | `tasks/` | 198 sessions | ~1.4MB | NO |
| Implementation plans | `plans/` | 82 files | Variable | NO |
| Pasted content | `paste-cache/` | 127 files | Variable | NO |
| Shell snapshots | `shell-snapshots/` | 297 files | Small | NO |
| Session environments | `session-env/` | 338 sessions | Small | NO |
| Usage statistics | `stats-cache.json` | 1 file | 10KB | NO |

---

## Design: Modular Collector Architecture

### Phase 1 — Data Lake (Modular Collectors)

Create independent collectors per data source, each producing typed records into shared storage. All records carry `session_id` and `timestamp` for cross-linking.

### Phase 2 — Session Graph (Future)

Add a linking layer that reads collector outputs and builds a session knowledge graph. No collector rewrites needed — Phase 2 is purely additive.

---

## Module Structure

```
bashgym/trace_capture/collectors/
  __init__.py
  base.py              # BaseCollector abstract class
  session.py           # SessionCollector (wraps existing ClaudeSessionImporter)
  subagent.py          # SubagentCollector -- parses subagents/*.jsonl
  edit.py              # EditCollector -- file-history/ diffs
  plan.py              # PlanCollector -- plans/*.md
  todo.py              # TodoCollector -- todos/*.json
  prompt.py            # PromptCollector -- history.jsonl + paste-cache/
  debug.py             # DebugCollector -- debug/*.txt (API traffic)
  environment.py       # EnvironmentCollector -- session-env/ + shell-snapshots/
  scanner.py           # ClaudeDataScanner -- orchestrates all collectors
```

### BaseCollector Interface

```python
class BaseCollector(ABC):
    """Abstract base for all .claude data collectors."""

    @abstractmethod
    def scan(self, since=None, project_filter=None) -> List[CollectorScanResult]:
        """Find all uncollected records from this source (dry-run)."""

    @abstractmethod
    def collect(self, session_id: str) -> List[CollectorRecord]:
        """Collect records for a specific session."""

    @abstractmethod
    def collect_all(self, since=None, project_filter=None) -> CollectorBatchResult:
        """Bulk collection with filtering."""
```

### Record Types

| Collector | Record Dataclass | Key Fields |
|---|---|---|
| SubagentCollector | `SubagentRecord` | session_id, agent_id, parent_tool_use_id, steps[], model, tokens, slug |
| EditCollector | `EditRecord` | session_id, file_path, versions[], content_hash, diff |
| PlanCollector | `PlanRecord` | session_id, plan_name, content, created_at |
| TodoCollector | `TodoRecord` | session_id, tasks[], status_distribution |
| PromptCollector | `PromptRecord` | timestamp, project, prompt_text, pasted_content |
| DebugCollector | `DebugRecord` | session_id, api_calls[], system_prompts[], latencies[], full_thinking[] |
| EnvironmentCollector | `EnvironmentRecord` | session_id, env_vars, shell_state, cwd, git_branch |

---

## Storage Layout

```
~/.bashgym/collected/
  sessions/          # Existing trace format (unchanged)
  subagents/         # SubagentRecord JSONs
  edits/             # EditRecord JSONs (with diffs)
  plans/             # PlanRecord JSONs
  todos/             # TodoRecord JSONs
  prompts/           # PromptRecord JSONs
  debug/             # DebugRecord JSONs (processed, not raw)
  environments/      # EnvironmentRecord JSONs
  index.json         # Cross-reference index: session_id -> all related records
  scan_state.json    # Deduplication tracking
```

### Cross-Reference Index

```json
{
  "06ef5435-...": {
    "session": "sessions/imported_claude_06ef5435_20260225.json",
    "subagents": ["subagents/agent-a65889e7.json"],
    "edits": ["edits/06ef5435_trainer.py_v1v2.json"],
    "plan": "plans/06ef5435_crystalline-exploring-sonnet.json",
    "todos": ["todos/06ef5435_tasks.json"],
    "environment": "environments/06ef5435_env.json",
    "project": "my-project",
    "timestamp": "2026-02-24T22:54:00Z"
  }
}
```

---

## Training Value Per Collector

| Collector | Training Use | Priority |
|---|---|---|
| **SubagentCollector** | **Direct SFT examples.** Subagents are focused, pre-segmented task-completion pairs. No additional segmentation needed. | P0 -- Highest value, lowest effort |
| **EditCollector** | **DPO pairs from edit history.** File v1->v2 (rejected if v3 exists), final version = chosen. 93MB untouched. | P0 -- Natural DPO pair source |
| **PlanCollector** | **Chain-of-thought training.** Plan-then-execute pairs teach planning before action. | P1 -- Simple to implement, high value |
| **PromptCollector** | **Real prompt seeds for augmentation.** Feeds `data_factory.py` synthetic generation with real user intents. | P1 -- Improves augmentation quality |
| **TodoCollector** | **Task decomposition training.** Which decompositions led to successful sessions? | P2 -- Enrichment data |
| **DebugCollector** | **Full thinking blocks** (not truncated to 2000 chars), system prompt recovery, latency data. Fills "thinking blocks not in training data" gap. | P2 -- Richest but most complex, needs PII filtering |
| **EnvironmentCollector** | **Conditional training.** Tags examples with platform/shell context for environment-aware behavior. | P3 -- Metadata enrichment |

---

## Peony Tool Expansion

### Expanded `import_traces` tool

```python
{
    "name": "import_traces",
    "description": "Import all available data from Claude Code sessions",
    "parameters": {
        "sources": {
            "type": "string",
            "enum": ["all", "sessions", "subagents", "edits", "plans", "prompts", "todos", "debug", "environments"],
            "default": "all"
        },
        "days": {"type": "integer", "default": 60},
        "project_filter": {"type": "string", "description": "Only import from matching projects"},
        "dry_run": {"type": "boolean", "default": false}
    }
}
```

### New tools

```python
{
    "name": "scan_claude_data",
    "description": "Scan ~/.claude and show what data is available but not yet imported. Shows counts and sizes per source type."
}

{
    "name": "get_collection_status",
    "description": "Show current collection stats: sessions, subagents, edits, plans collected, storage usage, last import time."
}

{
    "name": "collect_source",
    "description": "Collect a specific data source type (e.g., just plans, or just edits)."
}
```

### `bashgym-setup` CLI expansion

The existing `bashgym-setup import-recent` command calls `ClaudeDataScanner` instead of just `ClaudeSessionImporter`:

```bash
bashgym-setup import-recent               # Import everything from last 60 days
bashgym-setup import-recent --source edits # Just file edit history
bashgym-setup scan                         # Dry-run: show what's available
bashgym-setup status                       # Show collection stats
```

---

## Integration with Existing Pipeline

### Example Generation

The `ExampleGenerator` in `bashgym/factory/example_generator.py` gains new methods:

- `subagent_to_example(SubagentRecord)` -- Convert subagent record directly to v2 structured format
- `edits_to_dpo_pair(EditRecord)` -- Generate DPO chosen/rejected from edit versions
- `plan_to_cot_example(PlanRecord, session)` -- Plan + execution as chain-of-thought example

### Quality Scoring

The `TraceProcessor` can use collector data to improve quality scoring:

- Sessions with plans score higher on "complexity"
- Sessions with successful subagents score higher on "tool diversity"
- Sessions with clean edit histories (no v3+ revisions) score higher on "efficiency"

### Training Data Format

All new example types use the existing v2 structured tool-call format from TRAINING_DATA_GUIDE.md. No format changes needed.

---

## Implementation Priority

1. **BaseCollector + ClaudeDataScanner** -- scaffolding
2. **SubagentCollector** -- P0, direct SFT value
3. **EditCollector** -- P0, DPO pair value
4. **PlanCollector** -- P1, simple markdown parsing
5. **PromptCollector** -- P1, augmentation pipeline feed
6. **TodoCollector** -- P2, enrichment
7. **DebugCollector** -- P2, needs PII filtering design
8. **EnvironmentCollector** -- P3, metadata
9. **Peony tool expansion** -- wire up scan/collect/status tools
10. **bashgym-setup CLI expansion** -- expose via shell commands

---

## .claude Folder Format Reference

### Session Subfolder Structure

Each session in `projects/<slug>/<session-id>/` contains:
- `subagents/agent-<id>.jsonl` -- Full subagent conversation JSONL (same format as parent session)
- `tool-results/<tool-use-id>.txt` -- Large tool outputs stored separately

### Subagent JSONL Format

```json
{"parentUuid":null,"isSidechain":true,"userType":"external","cwd":"...","sessionId":"parent-session-id","version":"2.1.51","gitBranch":"feat/...","agentId":"a65889e7ce51c98b9","slug":"iridescent-wiggling-adleman","type":"user","message":{"role":"user","content":"..."}}
```

### File History Format

Path: `file-history/<session-id>/<content-hash>@v<version-number>`
- `@v1` = file state before edit
- `@v2` = file state after edit
- Multiple versions if edited multiple times

### Plans Format

Path: `plans/<alliterative-name>.md`
Structured markdown with problem statements, architecture decisions, implementation steps.

### Todos Format

Path: `todos/<session-id>-agent-<agent-id>.json`
JSON arrays of task objects with statuses.

### History Format

Path: `history.jsonl`
```json
{"display":"user prompt text","pastedContents":{},"timestamp":1759817571796,"project":"C:\\Users\\...\\project-name"}
```

### Debug Log Format

Path: `debug/<session-id>.txt`
Timestamped debug output including API request/response payloads.

### Shell Snapshot Format

Path: `shell-snapshots/snapshot-bash-<timestamp>-<id>.sh`
Shell initialization scripts with aliases, PATH, and tool availability.
