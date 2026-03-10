# Unified Trace Data Layer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify trace import across all 5 tools with consistent API endpoints, enrich non-Claude importers, add frontend import/filter UI, and surface trace analytics.

**Architecture:** 4 pillars — unified backend API, importer parity, frontend import UI + source filter, analytics dashboard. Each pillar builds on the previous.

**Tech Stack:** FastAPI (backend), React + Recharts + Lucide (frontend), existing tracesApi patterns, Botanical Brutalist styling.

---

## Task 1: Unified Import API Endpoints

**Files:**
- Modify: `bashgym/api/routes.py` (add new endpoints near existing import endpoint ~line 2058)
- Test: `tests/api/test_import_routes.py`

**Step 1: Write the failing test**

```python
# tests/api/test_import_routes.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from bashgym.api.routes import app

client = TestClient(app)


class TestImportEndpoints:
    """Test unified import API endpoints."""

    @patch("bashgym.api.routes.import_recent")
    def test_import_claude(self, mock_import):
        mock_import.return_value = []
        resp = client.post("/api/traces/import/claude", json={"days": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert "source" in data
        assert data["source"] == "claude"
        assert "imported" in data
        assert "skipped" in data
        assert "errors" in data

    @patch("bashgym.api.routes.import_gemini_sessions")
    def test_import_gemini(self, mock_import):
        mock_import.return_value = []
        resp = client.post("/api/traces/import/gemini", json={"days": 30})
        assert resp.status_code == 200
        assert resp.json()["source"] == "gemini"

    @patch("bashgym.api.routes.import_copilot_sessions")
    def test_import_copilot(self, mock_import):
        mock_import.return_value = []
        resp = client.post("/api/traces/import/copilot", json={"days": 30})
        assert resp.status_code == 200
        assert resp.json()["source"] == "copilot"

    @patch("bashgym.api.routes.import_opencode_sessions")
    def test_import_opencode(self, mock_import):
        mock_import.return_value = []
        resp = client.post("/api/traces/import/opencode", json={"days": 30})
        assert resp.status_code == 200
        assert resp.json()["source"] == "opencode"

    def test_import_invalid_source(self):
        resp = client.post("/api/traces/import/invalid", json={"days": 30})
        assert resp.status_code == 400

    @patch("bashgym.api.routes.import_recent")
    @patch("bashgym.api.routes.import_gemini_sessions")
    @patch("bashgym.api.routes.import_copilot_sessions")
    @patch("bashgym.api.routes.import_opencode_sessions")
    def test_import_all(self, mock_oc, mock_cop, mock_gem, mock_claude):
        mock_claude.return_value = []
        mock_gem.return_value = []
        mock_cop.return_value = []
        mock_oc.return_value = []
        resp = client.post("/api/traces/import/all", json={"days": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert isinstance(data["results"], list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_import_routes.py -v`
Expected: FAIL — endpoints don't exist yet.

**Step 3: Write implementation**

In `bashgym/api/routes.py`, add imports at top:
```python
from bashgym.trace_capture.importers import (
    import_recent,
    import_gemini_sessions,
    import_copilot_sessions,
    import_opencode_sessions,
)
```

Add Pydantic schema:
```python
class TraceImportRequest(BaseModel):
    days: int = 60
    limit: int = 100
    force: bool = False

class TraceImportResponse(BaseModel):
    source: str
    imported: int
    skipped: int
    errors: int
    total: int
    new_trace_ids: List[str] = []
```

Add endpoints:
```python
IMPORT_HANDLERS = {
    "claude": lambda req: import_recent(days=req.days, verbose=False, force=req.force),
    "gemini": lambda req: import_gemini_sessions(days=req.days, limit=req.limit, verbose=False),
    "copilot": lambda req: import_copilot_sessions(days=req.days, limit=req.limit, verbose=False),
    "opencode": lambda req: import_opencode_sessions(days=req.days, limit=req.limit, verbose=False),
    "codex": lambda req: import_codex_sessions(limit=req.limit, verbose=False),
}

@app.post("/api/traces/import/{source}", tags=["Traces"])
async def import_traces_by_source(source: str, request: TraceImportRequest = TraceImportRequest()):
    if source not in IMPORT_HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}. Valid: {list(IMPORT_HANDLERS.keys())}")
    try:
        results = IMPORT_HANDLERS[source](request)
        return _build_import_response(source, results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/traces/import/all", tags=["Traces"])
async def import_traces_all(request: TraceImportRequest = TraceImportRequest()):
    all_results = []
    for source, handler in IMPORT_HANDLERS.items():
        try:
            results = handler(request)
            all_results.append(_build_import_response(source, results))
        except Exception:
            all_results.append(TraceImportResponse(source=source, imported=0, skipped=0, errors=1, total=0))
    return {"results": all_results, "total_imported": sum(r.imported for r in all_results)}
```

Helper to normalize results from different importers:
```python
def _build_import_response(source: str, results) -> TraceImportResponse:
    if not results:
        return TraceImportResponse(source=source, imported=0, skipped=0, errors=0, total=0)
    imported = sum(1 for r in results if not getattr(r, 'skipped', False) and not getattr(r, 'error', None))
    skipped = sum(1 for r in results if getattr(r, 'skipped', False))
    errors = sum(1 for r in results if getattr(r, 'error', None))
    new_ids = [str(getattr(r, 'destination_file', '')) for r in results if getattr(r, 'destination_file', None)]
    return TraceImportResponse(source=source, imported=imported, skipped=skipped, errors=errors, total=len(results), new_trace_ids=new_ids)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_import_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bashgym/api/routes.py tests/api/test_import_routes.py
git commit -m "feat(api): add unified trace import endpoints for all tools"
```

---

## Task 2: Add source_tool Filter to GET /api/traces

**Files:**
- Modify: `bashgym/api/routes.py` (the `list_traces` endpoint ~line 1035)
- Test: `tests/api/test_import_routes.py` (add to same file)

**Step 1: Write the failing test**

```python
class TestSourceToolFilter:
    def test_list_traces_accepts_source_filter(self):
        resp = client.get("/api/traces?source_tool=claude_code")
        assert resp.status_code == 200

    def test_list_traces_no_source_returns_all(self):
        resp = client.get("/api/traces")
        assert resp.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_import_routes.py::TestSourceToolFilter -v`

**Step 3: Write implementation**

Add `source_tool` query parameter to existing `list_traces`:
```python
@app.get("/api/traces", tags=["Traces"])
async def list_traces(
    status: Optional[TraceStatus] = None,
    repo: Optional[str] = None,
    source_tool: Optional[str] = None,  # NEW: filter by source tool
    limit: int = 50,
    offset: int = 0
):
```

In the filtering logic, add:
```python
if source_tool and data.get("source_tool") != source_tool:
    continue
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_import_routes.py -v`

**Step 5: Commit**

```bash
git add bashgym/api/routes.py tests/api/test_import_routes.py
git commit -m "feat(api): add source_tool filter to GET /api/traces"
```

---

## Task 3: Enhance Analytics Endpoint with Source Breakdown

**Files:**
- Modify: `bashgym/api/routes.py` (the `get_trace_analytics` endpoint ~line 1294)
- Test: `tests/api/test_import_routes.py`

**Step 1: Write the failing test**

```python
class TestAnalyticsSourceBreakdown:
    def test_analytics_has_source_breakdown(self):
        resp = client.get("/api/traces/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert "source_breakdown" in data

    def test_analytics_has_cost_total(self):
        resp = client.get("/api/traces/analytics")
        data = resp.json()
        assert "cost_total_usd" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_import_routes.py::TestAnalyticsSourceBreakdown -v`

**Step 3: Write implementation**

In the `get_trace_analytics` endpoint, add source breakdown aggregation:
```python
source_counts = {}
cost_total = 0.0

for trace_file in all_trace_files:
    data = load_trace(trace_file)
    source = data.get("source_tool", "unknown")
    if source not in source_counts:
        source_counts[source] = {"source": source, "traces": 0, "steps": 0, "tokens": 0}
    source_counts[source]["traces"] += 1
    source_counts[source]["steps"] += data.get("summary", {}).get("total_steps", 0)
    source_counts[source]["tokens"] += data.get("metadata", {}).get("total_input_tokens", 0) + data.get("metadata", {}).get("total_output_tokens", 0)
    cost_total += data.get("metadata", {}).get("api_equivalent_cost_usd", 0.0)
```

Add to response:
```python
return {
    # ... existing fields ...
    "source_breakdown": list(source_counts.values()),
    "cost_total_usd": round(cost_total, 4),
    "avg_quality_score": round(avg_quality, 2) if total_sessions > 0 else 0,
}
```

**Step 4: Run test**

Run: `pytest tests/api/test_import_routes.py -v`

**Step 5: Commit**

```bash
git add bashgym/api/routes.py tests/api/test_import_routes.py
git commit -m "feat(api): add source breakdown and cost total to analytics endpoint"
```

---

## Task 4: Importer Parity — Enrich Gemini Importer

**Files:**
- Modify: `bashgym/trace_capture/importers/gemini_history.py`
- Test: `tests/trace_capture/test_importers.py` (add new tests)

**Step 1: Write the failing test**

```python
class TestGeminiImporterEnrichment:
    def test_trace_has_conversation_turns(self, tmp_path):
        # Create a fixture with multiple user/model turns
        session = [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": [{"text": "help me"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "bash", "args": {"command": "ls"}}}]},
        ]
        session_file = tmp_path / "session-test.json"
        session_file.write_text(json.dumps(session))

        importer = GeminiSessionImporter()
        steps, metadata = importer.parse_session_file(session_file)
        assert metadata.get("conversation_turns", 0) >= 2
        assert isinstance(metadata.get("all_user_prompts", []), list)
        assert len(metadata.get("all_user_prompts", [])) >= 2

    def test_trace_has_models_used(self, tmp_path):
        session = [
            {"role": "model", "parts": [{"text": "ok"}], "modelVersion": "gemini-2.5-pro"},
        ]
        session_file = tmp_path / "session-test2.json"
        session_file.write_text(json.dumps(session))

        importer = GeminiSessionImporter()
        steps, metadata = importer.parse_session_file(session_file)
        assert "models_used" in metadata
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/trace_capture/test_importers.py::TestGeminiImporterEnrichment -v`

**Step 3: Write implementation**

In `gemini_history.py`, modify `parse_session_file()` to track:
- `conversation_turns` — count user messages
- `all_user_prompts` — collect user text parts
- `user_initial_prompt` — first user message
- `models_used` — collect from `modelVersion` fields on model turns

Return as second value from parse: `(steps, metadata_dict)`.

Update `import_session()` to include enriched metadata in the saved trace JSON under `metadata`.

**Step 4: Run test**

Run: `pytest tests/trace_capture/test_importers.py -v`

**Step 5: Commit**

```bash
git add bashgym/trace_capture/importers/gemini_history.py tests/trace_capture/test_importers.py
git commit -m "feat(importers): enrich Gemini importer with conversation turns, prompts, models"
```

---

## Task 5: Importer Parity — Enrich Copilot Importer

**Files:**
- Modify: `bashgym/trace_capture/importers/copilot_history.py`
- Test: `tests/trace_capture/test_importers.py`

**Step 1: Write the failing test**

```python
class TestCopilotImporterEnrichment:
    def test_trace_has_conversation_turns(self, tmp_path):
        session = {
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "Sure", "tool_calls": [{"function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
                {"role": "user", "content": "thanks"},
            ]
        }
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps(session))

        importer = CopilotSessionImporter()
        steps, metadata = importer.parse_session_file(session_file)
        assert metadata.get("conversation_turns", 0) >= 2
        assert len(metadata.get("all_user_prompts", [])) >= 2

    def test_trace_has_dpo_metadata(self, tmp_path):
        session = {
            "suggestions": [
                {"command": "git status", "accepted": True},
                {"command": "rm -rf /", "accepted": False},
            ]
        }
        session_file = tmp_path / "session2.json"
        session_file.write_text(json.dumps(session))

        importer = CopilotSessionImporter()
        steps, metadata = importer.parse_session_file(session_file)
        assert "accepted_commands" in metadata
        assert "rejected_commands" in metadata
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/trace_capture/test_importers.py::TestCopilotImporterEnrichment -v`

**Step 3: Write implementation**

Enrich `parse_session_file()` with:
- `conversation_turns`, `all_user_prompts`, `user_initial_prompt`
- `accepted_commands`, `rejected_commands` counts (already partially there in result class, move to metadata)
- `models_used` if available in session data

**Step 4: Run test**

Run: `pytest tests/trace_capture/test_importers.py -v`

**Step 5: Commit**

```bash
git add bashgym/trace_capture/importers/copilot_history.py tests/trace_capture/test_importers.py
git commit -m "feat(importers): enrich Copilot importer with conversation metadata and DPO fields"
```

---

## Task 6: Importer Parity — Enrich OpenCode and Codex Importers

**Files:**
- Modify: `bashgym/trace_capture/importers/opencode_history.py`
- Modify: `bashgym/trace_capture/adapters/codex.py` (codex import function)
- Test: `tests/trace_capture/test_importers.py`

**Step 1: Write the failing tests**

```python
class TestOpenCodeImporterEnrichment:
    def test_trace_has_conversation_turns(self, tmp_path):
        # ... fixture with user/assistant messages
        importer = OpenCodeSessionImporter()
        steps, metadata = importer.parse_session_from_files(session_file)
        assert "conversation_turns" in metadata
        assert "all_user_prompts" in metadata

class TestCodexImporterEnrichment:
    def test_codex_trace_has_user_prompts(self, tmp_path):
        # ... fixture with codex transcript format
        # assert metadata has user_initial_prompt, all_user_prompts, conversation_turns
```

**Step 2: Run test to verify it fails**

**Step 3: Write implementation**

Same pattern: extract conversation_turns, all_user_prompts, user_initial_prompt, models_used from each source's native data format.

For Codex, the import function in `codex.py` needs `import_codex_sessions()` to be a proper function that returns result objects (currently returns List[Dict]).

**Step 4: Run test**

Run: `pytest tests/trace_capture/test_importers.py -v`

**Step 5: Commit**

```bash
git add bashgym/trace_capture/importers/opencode_history.py bashgym/trace_capture/adapters/codex.py tests/trace_capture/test_importers.py
git commit -m "feat(importers): enrich OpenCode and Codex importers with conversation metadata"
```

---

## Task 7: Frontend — API Client for Import Endpoints

**Files:**
- Modify: `frontend/src/services/api.ts`

**Step 1: Add API methods**

In the `tracesApi` object, add:
```typescript
importBySource: (source: string, options?: { days?: number; limit?: number; force?: boolean }) =>
  request<{
    source: string;
    imported: number;
    skipped: number;
    errors: number;
    total: number;
    new_trace_ids: string[];
  }>(`/traces/import/${source}`, {
    method: 'POST',
    body: JSON.stringify(options || {}),
  }),

importAll: (options?: { days?: number; limit?: number }) =>
  request<{
    results: Array<{
      source: string;
      imported: number;
      skipped: number;
      errors: number;
      total: number;
    }>;
    total_imported: number;
  }>('/traces/import/all', {
    method: 'POST',
    body: JSON.stringify(options || {}),
  }),
```

Update `list` method to accept `source_tool`:
```typescript
list: (options?: {
  status?: 'gold' | 'silver' | 'bronze' | 'failed' | 'pending';
  repo?: string;
  source_tool?: string;  // NEW
  limit?: number;
  offset?: number;
}) => {
  const params = new URLSearchParams()
  // ... existing params ...
  if (options?.source_tool) params.append('source_tool', options.source_tool)
  return request<...>(`/traces?${params}`)
}
```

Update `getAnalytics` response type:
```typescript
getAnalytics: () => request<TraceAnalytics & {
  source_breakdown: Array<{ source: string; traces: number; steps: number; tokens: number }>;
  cost_total_usd: number;
  avg_quality_score: number;
}>('/traces/analytics'),
```

**Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat(frontend): add import-by-source and source_tool filter to API client"
```

---

## Task 8: Frontend — Import Panel in TraceBrowser

**Files:**
- Modify: `frontend/src/components/traces/TraceBrowser.tsx`

**Step 1: Add import panel state**

```typescript
const [importPanelOpen, setImportPanelOpen] = useState(false)
const [importingSource, setImportingSource] = useState<string | null>(null)
const [importResults, setImportResults] = useState<Record<string, { imported: number; errors: number } | null>>({})
```

**Step 2: Add import panel component**

Create a collapsible panel below the header/filters area:

```tsx
const IMPORT_SOURCES = [
  { key: 'all', label: 'Import All', icon: Layers },
  { key: 'claude', label: 'Claude Code', icon: Terminal },
  { key: 'gemini', label: 'Gemini CLI', icon: Sparkles },
  { key: 'copilot', label: 'Copilot CLI', icon: GitBranch },
  { key: 'opencode', label: 'OpenCode', icon: FileCode },
  { key: 'codex', label: 'Codex', icon: Hash },
]
```

Each button:
- Shows spinner while importing
- Shows checkmark + count on success
- Shows X + error on failure
- Calls `tracesApi.importBySource(key)` or `tracesApi.importAll()`

Add a "Sync" button that calls `tracesApi.sync()`.

Style with Botanical Brutalist: `border-brutal`, monospace labels, `transition-press` hover.

**Step 3: Add source_tool filter**

Add a new filter row below status filters:

```tsx
const SOURCE_TOOLS = [
  { value: '', label: 'All Sources' },
  { value: 'claude_code', label: 'Claude Code' },
  { value: 'gemini_cli', label: 'Gemini CLI' },
  { value: 'copilot_cli', label: 'Copilot CLI' },
  { value: 'opencode', label: 'OpenCode' },
  { value: 'codex', label: 'Codex' },
]
```

Wire to `tracesApi.list({ source_tool: selectedSource })`.

**Step 4: Verify visually**

Run frontend, check Import panel renders, buttons work, filter narrows results.

**Step 5: Commit**

```bash
git add frontend/src/components/traces/TraceBrowser.tsx
git commit -m "feat(frontend): add import panel and source_tool filter to TraceBrowser"
```

---

## Task 9: Frontend — Trace Analytics Dashboard

**Files:**
- Create: `frontend/src/components/traces/TraceAnalytics.tsx`
- Modify: `frontend/src/components/traces/TraceBrowser.tsx` (add analytics toggle)

**Step 1: Create TraceAnalytics component**

```tsx
// Summary cards row (4 cards)
// Quality distribution chart (horizontal bar or donut)
// Source breakdown chart (horizontal bars)
// Training readiness section (progress bars)
// Tool usage table
```

**Section 1 — Summary Cards:**
```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Total Traces │ │  Gold Ready  │ │ Total Steps  │ │  Est. Cost   │
│     142      │ │     34       │ │    4,521     │ │   $12.34     │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

Uses `border-brutal`, `shadow-brutal-sm`, monospace numbers, serif/brand label.

**Section 2 — Quality Distribution:**
Horizontal stacked bar using existing Recharts patterns from TraceBrowser's AreaChart.
Segments: Gold (accent), Silver (text-secondary), Bronze (status-warning), Failed (status-error), Pending (text-muted).

**Section 3 — Source Breakdown:**
Horizontal bars from `source_breakdown` data. One bar per source tool. Shows trace count and step count.

**Section 4 — Training Readiness:**
```
SFT Ready:  ████████████░░░░░░░░  34/30 traces (ready!)
DPO Pairs:  ██████░░░░░░░░░░░░░░  12 possible pairs
```

Progress bars with thresholds. Green when threshold met.

**Section 5 — Tool Usage:**
Table with columns: Tool, Calls, Sessions, Success Rate, Tokens.
From `tool_stats` in analytics response. Sorted by calls descending.

**Step 2: Wire into TraceBrowser**

Add a toggle button (BarChart3 icon) in the TraceBrowser header that switches between list view and analytics view.

```tsx
const [view, setView] = useState<'list' | 'analytics'>('list')

// In header:
<button onClick={() => setView(v => v === 'list' ? 'analytics' : 'list')}>
  {view === 'list' ? <BarChart3 /> : <List />}
</button>

// In body:
{view === 'analytics' ? <TraceAnalytics /> : </* existing list view */>}
```

**Step 3: Verify visually**

Run frontend, toggle analytics view, verify charts render with real or empty data.

**Step 4: Commit**

```bash
git add frontend/src/components/traces/TraceAnalytics.tsx frontend/src/components/traces/TraceBrowser.tsx
git commit -m "feat(frontend): add trace analytics dashboard with quality, source, and readiness views"
```

---

## Task 10: Integration Test — Full Import → View → Analytics Flow

**Files:**
- Create: `tests/api/test_trace_flow.py`

**Step 1: Write integration test**

```python
class TestTraceImportFlow:
    """End-to-end: import → list with filter → analytics."""

    @patch("bashgym.api.routes.import_recent")
    def test_import_then_list_by_source(self, mock_import):
        # Import
        mock_import.return_value = [mock_result(source_tool="claude_code")]
        resp = client.post("/api/traces/import/claude")
        assert resp.status_code == 200

        # List with filter
        resp = client.get("/api/traces?source_tool=claude_code")
        assert resp.status_code == 200

    def test_analytics_includes_all_fields(self):
        resp = client.get("/api/traces/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert "source_breakdown" in data
        assert "cost_total_usd" in data
        assert "quality_distribution" in data
        assert "training_readiness" in data
```

**Step 2: Run test**

Run: `pytest tests/api/test_trace_flow.py -v`

**Step 3: Commit**

```bash
git add tests/api/test_trace_flow.py
git commit -m "test: add integration test for full import → filter → analytics flow"
```

---

## Summary

| Task | What | Pillar |
|------|------|--------|
| 1 | Unified import API endpoints (per-source + all) | Backend |
| 2 | source_tool filter on GET /api/traces | Backend |
| 3 | Source breakdown + cost in analytics endpoint | Backend |
| 4 | Enrich Gemini importer (turns, prompts, models) | Parity |
| 5 | Enrich Copilot importer (turns, prompts, DPO) | Parity |
| 6 | Enrich OpenCode + Codex importers | Parity |
| 7 | Frontend API client methods | Frontend |
| 8 | Import panel + source filter in TraceBrowser | Frontend |
| 9 | Trace Analytics dashboard component | Frontend |
| 10 | Integration test for full flow | Testing |
