# Terminal-Dispatch Orchestrator Design

**Date:** 2026-02-23
**Status:** Approved

---

## Problem

The existing Orchestrator was built around spawning Claude Code CLI subprocesses in isolated git worktrees — a model that requires a specific repository path, a base branch, and careful process management. This doesn't fit the way the workspace is actually used: terminals are already open, already checked out on the right branch, already running Claude Code. The old execution model also had ToS concerns that required it to be feature-flagged off by default.

**The fix:** make the Orchestrator a spec decomposition + terminal dispatch system. The LLM breaks the spec into tasks; the user edits and approves them; the frontend fires `claude "[prompt]"\n` into idle terminals. No subprocess spawning, no worktrees, no flags.

---

## Architecture

```
User fills simplified SpecForm
         │
         ▼
Backend LLM decomposes spec → TaskDAG
         │
         ▼
User reviews & edits task prompts inline
         │
         ▼
User clicks "Dispatch"
         │
         ▼
Frontend writes claude "[prompt]"\r to idle terminals
(one task per terminal, round-robin if more tasks than terminals)
         │
         ▼
TaskDAGView shows task → terminal assignment
```

---

## Changes

### 1. SpecForm simplification

**Remove:**
- `repository` field
- `base_branch` field
- `max_workers` field (subprocess count — meaningless for terminal dispatch)

**Keep:**
- Title, description, constraints, acceptance criteria (the *what*)
- Max budget USD (caps LLM spend during decomposition)
- LLM config (provider/model for decomposition — optional, defaults to Anthropic)

**Result:** the form is about describing the work, not configuring execution infrastructure.

---

### 2. Inline task editing before dispatch

After decomposition, the awaiting-approval state shows the TaskDAG with each task prompt **editable inline**. The LLM output is a starting point, not gospel.

Each task card has:
- Editable title (single line)
- Editable worker prompt (multi-line textarea, expands on focus)
- Estimated turns / budget (read-only, from LLM)
- Dependency indicators

A "Reset to original" button per task restores the LLM's original text if the user over-edits.

---

### 3. Approval = terminal dispatch

The "Approve & Execute" button:
1. Collects the (possibly edited) task list from the DAG
2. Finds idle terminals from `terminalStore.sessions` (status === 'idle')
3. Assigns tasks to terminals round-robin
4. Calls `window.bashgym.terminal.write(terminalId, `claude "${escapePrompt(task.worker_prompt)}"\r`)` for each task
5. Updates `orchestratorStore` with `taskAssignments: Record<taskId, terminalId>`

If there are more tasks than idle terminals, remaining tasks are queued. When a terminal becomes idle again (status transitions from running → idle), the next queued task is dispatched automatically.

---

### 4. TaskDAGView — terminal assignment display

Each task node gains a terminal badge: `→ T2` or `→ T4 (queued)`. Clicking the badge focuses that terminal in the grid.

Status mapping:
- `pending` → not yet dispatched (waiting for idle terminal)
- `running` → terminal is active (status: running/tool_calling)
- `completed` → terminal returned to idle after being active
- `failed` → terminal returned to error/idle with error attention state

---

### 5. Backend changes

**SpecRequest:** `repository` and `base_branch` become fully ignored on the backend (already optional). `max_workers` defaults to the number of terminals the frontend will use (also ignored server-side for execution).

**`approve_plan` endpoint:** stops starting `_execute_dag` as a background task. Instead it just transitions the job status to `"dispatched"` and returns the full task list so the frontend can drive dispatch.

**New job status:** `"dispatched"` — job is approved and terminals are executing. The backend no longer tracks per-task progress (that's frontend state via terminal status watching).

**Feature gate:** `_check_orchestration_enabled()` is removed from all routes. Terminal dispatch has no ToS concerns. The flag is retained in config for potential future worktree mode but is not enforced anywhere.

---

### 6. Queue management (frontend)

`orchestratorStore` gains:
```ts
taskQueue: string[]           // task IDs waiting for an idle terminal
taskAssignments: Record<string, string>  // taskId → terminalId
```

A `useEffect` in `OrchestratorDashboard` watches `terminalStore.sessions` for status transitions. When a terminal becomes idle and there are queued tasks, the next task is dispatched.

---

## What does NOT change

- The spec form UX structure (tabs: Submit / Active Job / History)
- The LLM decomposition call (backend `POST /orchestrate/submit`)
- The TaskDAGView component (extended, not replaced)
- Budget tracking (still useful for LLM decomposition cost)
- Job history

---

## Acceptance criteria

1. SpecForm has no repo/branch/workers fields
2. After decomposition, every task prompt is editable inline with a reset button
3. Clicking "Dispatch" writes `claude "[prompt]"\r` to idle terminals
4. Each task in the DAG shows which terminal it was sent to
5. Tasks with no idle terminal are queued and auto-dispatched when a terminal frees up
6. The orchestration feature flag is removed from route handlers
7. The disabled state page in OrchestratorDashboard is removed (no longer needed)
