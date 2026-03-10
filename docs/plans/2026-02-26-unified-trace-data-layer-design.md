# Unified Trace Data Layer — Design

## Goal

Unify the trace import pipeline across all 5 AI coding tools (Claude Code, Gemini CLI, Copilot CLI, OpenCode, Codex) with consistent API endpoints, bring importers to feature parity, add frontend import/filtering UI, and surface trace analytics.

## Architecture

Four pillars executed in dependency order:

```
Pillar 1: Unified Import API        → backend endpoints for all tools
Pillar 2: Importer Parity           → consistent interfaces, richer metadata
Pillar 3: Frontend Import & Filter  → buttons, source_tool filter, sync
Pillar 4: Trace Analytics Dashboard → quality, cost, readiness, source breakdown
```

## Pillar 1: Unified Import API

### Problem
Only Claude Code has a REST endpoint (`POST /api/traces/import`). The other 4 importers exist as Python code but are unreachable from the frontend.

### Design

Add to `bashgym/api/routes.py`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /api/traces/import/{source}` | POST | Import from specific tool. `source` = `claude`, `gemini`, `copilot`, `opencode`, `codex` |
| `POST /api/traces/import/all` | POST | Import from all detected tools |

**Request body (shared):**
```json
{
  "days": 60,
  "limit": 100,
  "force": false
}
```

**Response (shared):**
```json
{
  "source": "gemini",
  "imported": 12,
  "skipped": 3,
  "errors": 0,
  "total": 15,
  "new_trace_ids": ["imported_gemini_abc_20260226.json"]
}
```

For `/import/all`, response is an array of the above per source.

### Refactor existing Claude endpoint
The existing `POST /api/traces/import` stays as-is for backwards compat but internally delegates to the same handler as `/api/traces/import/claude`.

---

## Pillar 2: Importer Parity

### Problem
Claude's importer captures 24+ metadata fields. Others capture 3-4. The result classes are different. No shared interface.

### Design

**Base interface** — all importers share:

```python
class BaseImportResult:
    session_id: str
    source_tool: str
    source_file: Optional[Path]
    steps_imported: int
    destination_file: Optional[Path]
    error: Optional[str]
    skipped: bool
    skip_reason: Optional[str]
```

**Metadata enrichment** — add where source data supports it:

| Field | Claude | Gemini | Copilot | OpenCode | Codex |
|-------|--------|--------|---------|----------|-------|
| `models_used` | Yes (has it) | Add (from response.modelVersion) | Add (from metadata) | Add (from metadata) | Add (from metadata) |
| `conversation_turns` | Yes | Add | Add | Add | Add |
| `total_input_tokens` | Yes | Add (if available) | Skip (not in data) | Skip | Skip |
| `total_output_tokens` | Yes | Add (if available) | Skip | Skip | Skip |
| `user_initial_prompt` | Yes | Has it | Has it | Has it | Add |
| `all_user_prompts` | Yes | Has it | Has it | Has it | Add |

**What we DON'T add** (tool doesn't provide it):
- Thinking blocks → Claude-specific (extended thinking)
- Cache tokens → Claude-specific (prompt caching)
- Cognitive extraction → Claude-specific (thinking block content)
- Subagent metadata → Claude-specific
- PII filtering → Keep as Claude async-only for now
- Cost estimation → Only meaningful where we have token counts + known pricing

**Codex importer enrichment:**
Currently `import_codex_sessions()` is minimal. Add proper `user_initial_prompt`, `all_user_prompts`, `conversation_turns`, `models_used` extraction from transcript data.

---

## Pillar 3: Frontend Import & Filter UI

### Problem
No import buttons. No source_tool filter. TraceBrowser only shows repo + status filters.

### Design

**Import Panel** — collapsible section at top of TraceBrowser:

```
┌─────────────────────────────────────────────────────────┐
│ Import Traces                                    [▾]    │
│                                                         │
│ [Import All]  [Claude] [Gemini] [Copilot] [OpenCode]   │
│                                        [Codex] [Sync]   │
│                                                         │
│ Last import: 12 traces from 3 sources, 2 min ago        │
└─────────────────────────────────────────────────────────┘
```

Each button shows a spinner while importing, then a result toast.

**Source Tool Filter** — new filter bar alongside existing status/repo:

```
Status: [All] [Gold] [Silver] [Bronze] [Failed] [Pending]
Source: [All] [Claude Code] [Gemini CLI] [Copilot CLI] [OpenCode] [Codex]
Repo:   [All repos ▾]
```

Backend: Add `source_tool` query param to `GET /api/traces`.

**Sync Button** — calls existing `/api/traces/sync` to pull `~/.bashgym/traces/` into project data dirs.

---

## Pillar 4: Trace Analytics Dashboard

### Problem
`/api/traces/analytics` returns rich data (quality distribution, tool stats, training readiness, token totals) but nothing renders it.

### Design

**New component: `TraceAnalytics.tsx`** — lives in TraceBrowser as a toggleable view or dedicated tab.

**Sections:**

1. **Summary Cards** (top row, 4 cards):
   - Total Traces (with trend)
   - Gold Traces / Training Ready
   - Total Steps Captured
   - Estimated Cost (sum of `api_equivalent_cost_usd`)

2. **Quality Distribution** (bar or donut chart):
   - Gold / Silver / Bronze / Rejected / Pending segments
   - Click to filter TraceBrowser

3. **Source Breakdown** (horizontal bar chart):
   - Traces per source_tool
   - Steps per source_tool

4. **Training Readiness** (progress indicator):
   - SFT ready: X gold traces (target: 30)
   - DPO pairs possible: min(silver, bronze)
   - Progress bar toward training threshold

5. **Tool Usage** (table/chart):
   - Most-used tools across all traces
   - Success rate per tool
   - Tokens per tool

Data source: existing `/api/traces/analytics` + `/api/traces/stats` endpoints. May need to add `source_tool` grouping to analytics endpoint.

### Analytics endpoint enhancement

Add to `/api/traces/analytics` response:
```json
{
  "source_breakdown": [
    {"source": "claude_code", "traces": 45, "steps": 1200, "tokens": 500000},
    {"source": "gemini_cli", "traces": 12, "steps": 300, "tokens": 0}
  ],
  "cost_total_usd": 12.34,
  "avg_quality_score": 0.72
}
```

---

## Tech Decisions

- **No new dependencies** — all frontend charts use existing patterns from TraceBrowser (it already has area charts)
- **Botanical Brutalist styling** — cards with `border-brutal`, monospace labels, offset shadows
- **Incremental delivery** — each pillar is independently useful. Pillar 1 unblocks Pillar 3.
- **Backwards compatible** — existing `/api/traces/import` endpoint stays

---

## Out of Scope

- Async/instrumentation for non-Claude importers (PII, injection detection)
- Cognitive extraction for non-Claude tools (they don't expose thinking)
- Cost estimation for non-Claude tools (no token data in most)
- Aider/Continue/Cursor adapters (still "coming soon")
