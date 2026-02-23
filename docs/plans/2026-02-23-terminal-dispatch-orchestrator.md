# Terminal-Dispatch Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the Orchestrator from git-worktree subprocess execution to terminal-native dispatch — LLM decomposes the spec, user edits task prompts inline, approval writes `claude "[prompt]"\r` to idle terminals.

**Architecture:** The backend handles only spec decomposition (`POST /orchestrate/submit`). The `approve_plan` endpoint marks the job as dispatched and returns tasks; the frontend drives all terminal dispatch using `window.bashgym.terminal.write()`. Task status is tracked in the orchestratorStore by watching terminal status transitions.

**Tech Stack:** FastAPI (backend), React + Zustand (frontend), Electron IPC via `window.bashgym.terminal.write()`

**Design doc:** `docs/plans/2026-02-23-terminal-dispatch-orchestrator-design.md`

---

### Task 1: Backend — remove the feature gate entirely

**Files:**
- Modify: `bashgym/api/orchestrator_routes.py`

The `_check_orchestration_enabled()` function and all 7 calls to it must be removed. Terminal dispatch has no ToS concerns — the user is running Claude in their own terminals.

**Step 1: Remove the gate function and all call sites**

In `bashgym/api/orchestrator_routes.py`, delete lines 96–111 (the `_check_orchestration_enabled` function) and the `_check_orchestration_enabled()` call at the top of each handler body: `submit_spec`, `approve_plan`, `get_status`, `retry_task`, `cancel_job`, `list_providers`, `list_jobs`.

The function to delete:
```python
def _check_orchestration_enabled() -> None:
    """Raise 503 if orchestration is disabled via feature flag."""
    from bashgym.config import get_settings
    if not get_settings().orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "orchestration_disabled",
                "message": (
                    "Orchestration is temporarily unavailable. "
                    "Set ORCHESTRATION_ENABLED=true to enable."
                ),
            },
        )
```

Also remove the `# Feature gate` comment block header.

**Step 2: Change `approve_plan` to return tasks without spawning execution**

Replace the body of `approve_plan` (lines ~228–246) with this — it marks the job as `"dispatched"` and returns the full task list for frontend-driven dispatch. No `background_tasks.add_task(_execute_dag, ...)` call:

```python
@router.post("/{job_id}/approve")
async def approve_plan(job_id: str, background_tasks: BackgroundTasks):
    """Approve the decomposed plan. Returns task list for frontend dispatch.

    The frontend writes 'claude "[prompt]"\\r' to idle terminals directly.
    No background subprocess execution is started here.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', expected 'awaiting_approval'"
        )

    job["status"] = "dispatched"

    dag = job["dag"]
    tasks = []
    if dag:
        for task_id, task in dag.nodes.items():
            tasks.append({
                "id": task_id,
                "title": task.title,
                "worker_prompt": task.worker_prompt,
                "priority": task.priority.value if hasattr(task.priority, 'value') else task.priority,
                "dependencies": list(task.dependencies),
            })

    return {
        "status": "dispatched",
        "task_count": len(tasks),
        "tasks": tasks,
    }
```

Note: `ApproveRequest` model with `base_branch` is no longer used by this endpoint — remove it from the signature.

**Step 3: Remove `ApproveRequest` model (it's no longer used)**

Delete the `ApproveRequest` class (lines ~44–47):
```python
class ApproveRequest(BaseModel):
    """Request to approve a decomposed plan."""
    base_branch: str = "main"
```

**Step 4: Verify the server restarts cleanly**

```bash
curl -s http://localhost:8003/api/orchestrate/jobs
```

Expected: `{"jobs":[]}` with HTTP 200 (not 503). The gate is gone.

**Step 5: Commit**

```bash
git add bashgym/api/orchestrator_routes.py
git commit -m "feat(orchestrator): remove feature gate, approve returns tasks for terminal dispatch"
```

---

### Task 2: Backend — add `dispatched` status to `get_status`

**Files:**
- Modify: `bashgym/api/orchestrator_routes.py`

The `get_status` endpoint should return `"dispatched"` naturally since it just reads `job["status"]`. No change needed to the logic — just verify the response shape matches what the frontend expects.

**Step 1: Verify get_status returns dispatched**

After approving a test job (or manually setting `_jobs[id]["status"] = "dispatched"` in the Python shell), confirm `GET /api/orchestrate/{id}/status` returns `{"status": "dispatched", ...}`.

No code change needed here — `job["status"]` is already returned verbatim.

**Step 2: Commit**

```bash
git commit --allow-empty -m "chore(orchestrator): verify dispatched status passthrough (no-op)"
```

---

### Task 3: Frontend — update orchestratorStore

**Files:**
- Modify: `frontend/src/stores/orchestratorStore.ts`

**Step 1: Add new state fields to the interface**

In the `OrchestratorState` interface, add after `activeTab`:

```typescript
isDisabled: boolean  // REMOVE THIS — no longer needed
taskAssignments: Record<string, string>  // taskId → terminalId
taskQueue: string[]                       // taskIds waiting for idle terminal
editedPrompts: Record<string, string>     // taskId → user-edited prompt
```

Remove `isDisabled: boolean` from the interface entirely.

Add actions:
```typescript
setEditedPrompt: (taskId: string, prompt: string) => void
resetEditedPrompt: (taskId: string) => void
dispatchToTerminals: () => void
dispatchNextQueued: (terminalId: string) => void
```

**Step 2: Add initial state**

In `create<OrchestratorState>((set, get) => ({`, add alongside other initial values:
```typescript
taskAssignments: {},
taskQueue: [],
editedPrompts: {},
```

Remove `isDisabled: false`.

**Step 3: Add `setEditedPrompt` and `resetEditedPrompt`**

```typescript
setEditedPrompt: (taskId, prompt) =>
  set((state) => ({
    editedPrompts: { ...state.editedPrompts, [taskId]: prompt },
  })),

resetEditedPrompt: (taskId) =>
  set((state) => {
    const next = { ...state.editedPrompts }
    delete next[taskId]
    return { editedPrompts: next }
  }),
```

**Step 4: Add `dispatchToTerminals`**

This is the core dispatch logic. Add it to the store:

```typescript
dispatchToTerminals: () => {
  const { currentJob, editedPrompts } = get()
  if (!currentJob) return

  // Import terminalStore lazily to avoid circular deps
  const { sessions } = (window as any).__terminalStoreSnapshot__ ??
    require('../stores').useTerminalStore.getState()

  const idleTerminalIds = Array.from(
    (sessions as Map<string, { status: string }>).entries()
  )
    .filter(([, s]) => s.status === 'idle')
    .map(([id]) => id)

  const tasks = Object.values(currentJob.tasks)
    .filter((t) => t.status === 'pending')
    .sort((a, b) => {
      const order = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']
      return order.indexOf(a.priority) - order.indexOf(b.priority)
    })

  const assignments: Record<string, string> = { ...get().taskAssignments }
  const queue: string[] = []

  tasks.forEach((task, i) => {
    const terminalId = idleTerminalIds[i]
    if (terminalId) {
      const prompt = editedPrompts[task.id] ?? task.worker_prompt ?? task.description
      const escaped = prompt.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
      window.bashgym?.terminal.write(terminalId, `claude "${escaped}"\r`)
      assignments[task.id] = terminalId
    } else {
      queue.push(task.id)
    }
  })

  set({
    taskAssignments: assignments,
    taskQueue: queue,
  })
},
```

**Step 5: Add `dispatchNextQueued`**

Called when a terminal becomes idle and there are queued tasks:

```typescript
dispatchNextQueued: (terminalId: string) => {
  const { taskQueue, currentJob, editedPrompts } = get()
  if (taskQueue.length === 0 || !currentJob) return

  const [nextTaskId, ...remaining] = taskQueue
  const task = currentJob.tasks[nextTaskId]
  if (!task) {
    set({ taskQueue: remaining })
    return
  }

  const prompt = editedPrompts[nextTaskId] ?? task.worker_prompt ?? task.description
  const escaped = prompt.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
  window.bashgym?.terminal.write(terminalId, `claude "${escaped}"\r`)

  set((state) => ({
    taskQueue: remaining,
    taskAssignments: { ...state.taskAssignments, [nextTaskId]: terminalId },
  }))
},
```

**Step 6: Update `approveJob` to call `dispatchToTerminals`**

The existing `approveJob` calls `orchestratorApi.approveJob()` and updates status. Change it to call `dispatchToTerminals()` after the API call succeeds:

```typescript
approveJob: async (jobId) => {
  const result = await orchestratorApi.approveJob(jobId)
  if (result.ok) {
    set((state) => ({
      currentJob: state.currentJob
        ? { ...state.currentJob, status: 'executing' }
        : null,
    }))
    // Dispatch tasks to idle terminals
    get().dispatchToTerminals()
  }
},
```

**Step 7: Fix `fetchJobs` — remove 503 disabled detection**

Replace the existing `fetchJobs` with the clean version (no `isDisabled` logic):

```typescript
fetchJobs: async () => {
  const result = await orchestratorApi.listJobs()
  if (result.ok && result.data) {
    const jobsList = (result.data.jobs || result.data || []) as any[]
    set({
      jobs: jobsList.map((j: any) => ({
        jobId: j.job_id,
        status: j.status,
        title: j.title || j.job_id,
        taskCount: j.task_count,
      })),
    })
  }
},
```

**Step 8: Update `orchestratorApi.approveJob` in api.ts**

`approveJob` no longer needs `baseBranch`. In `frontend/src/services/api.ts`, change:

```typescript
// Before:
approveJob: (jobId: string, baseBranch?: string) =>
  request(`/orchestrate/${jobId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ base_branch: baseBranch || 'main' })
  }),

// After:
approveJob: (jobId: string) =>
  request(`/orchestrate/${jobId}/approve`, { method: 'POST', body: '{}' }),
```

**Step 9: Commit**

```bash
git add frontend/src/stores/orchestratorStore.ts frontend/src/services/api.ts
git commit -m "feat(orchestrator): add terminal dispatch to store, remove isDisabled"
```

---

### Task 4: Frontend — simplify SpecForm

**Files:**
- Modify: `frontend/src/components/orchestrator/SpecForm.tsx`

**Step 1: Remove state variables**

Delete these `useState` declarations:
```typescript
const [repository, setRepository] = useState('')
const [baseBranch, setBaseBranch] = useState('main')
const [maxWorkers, setMaxWorkers] = useState(5)
```

**Step 2: Remove from `handleSubmit`**

Change the `submitSpec` call to not include `repository`, `base_branch`, or `max_workers`:

```typescript
await submitSpec({
  title: title.trim(),
  description: description.trim(),
  constraints: constraints.length > 0 ? constraints : undefined,
  acceptance_criteria: acceptanceCriteria.length > 0 ? acceptanceCriteria : undefined,
  max_budget_usd: maxBudget,
  llm_config: { provider },
})
```

**Step 3: Remove the Repository + Base Branch grid section**

Delete the entire block (lines ~100–126):
```tsx
{/* Two-column: Repository + Base Branch */}
<div className="grid grid-cols-2 gap-4">
  ...
</div>
```

**Step 4: Update the Execution Config fieldset**

Remove the `Max Workers` field. Rename the fieldset legend. Change from 3-column grid to 2-column:

```tsx
<fieldset className="border-brutal border-border rounded-brutal p-4 bg-background-card">
  <legend className="flex items-center gap-2 px-2">
    <span className="font-brand text-lg text-text-primary">Decomposition Config</span>
  </legend>
  <div className="grid grid-cols-2 gap-4">
    {/* LLM Provider — unchanged */}
    {/* Max Budget — unchanged */}
  </div>
</fieldset>
```

**Step 5: Verify the form renders cleanly**

Open the Orchestrator → Submit tab. Confirm: Title, Description, Constraints, Acceptance Criteria, Decomposition Config (Provider + Budget). No repo/branch/workers fields.

**Step 6: Commit**

```bash
git add frontend/src/components/orchestrator/SpecForm.tsx
git commit -m "feat(orchestrator): simplify spec form — remove repo/branch/workers"
```

---

### Task 5: Frontend — inline prompt editing in TaskDAGView

**Files:**
- Modify: `frontend/src/components/orchestrator/TaskDAGView.tsx`

When `currentJob.status === 'awaiting_approval'` or `'dispatched'`, each expanded task card shows an editable prompt textarea. After dispatch it shows the assigned terminal badge.

**Step 1: Import new store actions and terminalStore**

```tsx
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { useTerminalStore } from '../../stores/terminalStore'
import { Terminal } from 'lucide-react'
```

**Step 2: Pull new state from orchestratorStore**

```tsx
const {
  currentJob,
  retryTask,
  editedPrompts,
  setEditedPrompt,
  resetEditedPrompt,
  taskAssignments,
} = useOrchestratorStore()
const { sessions, panels } = useTerminalStore()
```

**Step 3: Add a helper to get terminal display name**

```tsx
function getTerminalLabel(terminalId: string, sessions: Map<string, any>, panels: any[]): string {
  const panel = panels.find(p => p.terminalId === terminalId)
  return panel?.title ?? terminalId.slice(0, 6)
}
```

**Step 4: Add terminal badge to task card header**

Inside the task card header row (next to the priority tag), add:

```tsx
{taskAssignments[task.id] && (
  <span className="inline-flex items-center gap-1 font-mono text-[10px] bg-background-secondary border border-border px-1.5 py-0.5 rounded-brutal text-text-muted">
    <Terminal className="w-2.5 h-2.5" />
    {getTerminalLabel(taskAssignments[task.id], sessions, panels)}
  </span>
)}
```

**Step 5: Add editable prompt in expanded view**

In the expanded section, after the description `<p>`, add the editable prompt block when the job is in an editable state (`awaiting_approval`):

```tsx
{currentJob?.status === 'awaiting_approval' && (
  <div className="mb-3">
    <div className="flex items-center justify-between mb-1">
      <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
        Claude Prompt
      </span>
      {editedPrompts[task.id] !== undefined && (
        <button
          onClick={() => resetEditedPrompt(task.id)}
          className="font-mono text-[10px] text-text-muted hover:text-accent underline"
        >
          Reset to original
        </button>
      )}
    </div>
    <textarea
      value={editedPrompts[task.id] ?? task.worker_prompt ?? task.description}
      onChange={(e) => setEditedPrompt(task.id, e.target.value)}
      rows={4}
      className="input w-full text-xs font-mono resize-y"
      placeholder="Claude will receive this as its task prompt..."
    />
    {editedPrompts[task.id] !== undefined && (
      <p className="font-mono text-[10px] text-accent mt-1">Edited — original preserved for reset</p>
    )}
  </div>
)}
```

**Step 6: Add `dispatched` to statusConfig**

```typescript
const statusConfig = {
  ...existingStatuses,
  dispatched: { icon: Terminal, color: 'text-accent', bg: 'border-accent' },
  retrying: { icon: RefreshCw, color: 'text-status-warning', bg: 'border-status-warning' },
}
```

**Step 7: Commit**

```bash
git add frontend/src/components/orchestrator/TaskDAGView.tsx
git commit -m "feat(orchestrator): add inline prompt editing and terminal badges to TaskDAGView"
```

---

### Task 6: Frontend — update OrchestratorDashboard

**Files:**
- Modify: `frontend/src/components/orchestrator/OrchestratorDashboard.tsx`

**Step 1: Remove `isDisabled` import and disabled state UI**

Remove `Lock` from the lucide import.

Remove the entire `if (isDisabled)` block and the large disabled state JSX (lines ~31–49 in the current file).

Remove `isDisabled` from the store destructuring.

**Step 2: Add queue watcher effect**

This effect fires when terminal sessions change. If a terminal just became idle and there are queued tasks, dispatch the next one:

```tsx
const { dispatchNextQueued, taskQueue } = useOrchestratorStore()
const { sessions } = useTerminalStore()
const prevSessionStatuses = useRef<Map<string, string>>(new Map())

useEffect(() => {
  if (taskQueue.length === 0) return

  sessions.forEach((session, id) => {
    const prev = prevSessionStatuses.current.get(id)
    if (prev !== 'idle' && session.status === 'idle') {
      // Terminal just became idle — dispatch next queued task
      dispatchNextQueued(id)
    }
  })

  // Update ref for next comparison
  const next = new Map<string, string>()
  sessions.forEach((s, id) => next.set(id, s.status))
  prevSessionStatuses.current = next
}, [sessions, taskQueue, dispatchNextQueued])
```

Add `import { useRef } from 'react'` and `import { useTerminalStore } from '../../stores/terminalStore'`.

**Step 3: Update `ActiveJobActions` — remove `base_branch` argument**

```tsx
function ActiveJobActions() {
  const { currentJob, approveJob, cancelJob } = useOrchestratorStore()
  if (!currentJob) return null
  return (
    <>
      <button
        onClick={() => approveJob(currentJob.jobId)}  // no baseBranch arg
        className="btn-primary font-mono text-xs"
      >
        Dispatch to Terminals
      </button>
      <button
        onClick={() => cancelJob(currentJob.jobId)}
        className="btn-secondary font-mono text-xs"
      >
        Reject
      </button>
    </>
  )
}
```

**Step 4: Verify OrchestratorDashboard renders with no disabled state**

Open Orchestrator in the app. It should go straight to the Submit tab with no locked state or warning.

**Step 5: Commit**

```bash
git add frontend/src/components/orchestrator/OrchestratorDashboard.tsx
git commit -m "feat(orchestrator): remove disabled state, add queue watcher, rename approve button"
```

---

### Task 7: Frontend — clean up Sidebar

**Files:**
- Modify: `frontend/src/components/layout/Sidebar.tsx`

**Step 1: Remove orchestrator's `disabled` and `disabledTitle` props**

In `SecondarySections`, change the orchestrator item back to a plain item:

```typescript
{ id: 'orchestrator', icon: <Network className="w-4 h-4" />, label: 'Orchestrator' }
```

Remove `disabled: orchestratorDisabled` and `disabledTitle: '...'`.

**Step 2: Remove `useOrchestratorStore` import if no longer used**

If `isDisabled` was the only thing pulled from orchestratorStore in the Sidebar, remove:
```typescript
import { useOrchestratorStore } from '../../stores/orchestratorStore'
// and
const { isDisabled: orchestratorDisabled } = useOrchestratorStore()
```

**Step 3: Remove the disabled item rendering from `CollapsibleSection`**

In `CollapsibleSection`, remove the `item.disabled ? <div ...> : <MenuItem ...>` ternary — just always render `<MenuItem>`. Remove the `disabled` and `disabledTitle` fields from the item type.

Clean interface:
```typescript
interface CollapsibleSectionProps {
  title: string
  items: Array<{ id: SecondaryViewId; icon: React.ReactNode; label: string }>
  defaultExpanded?: boolean
}
```

**Step 4: Verify the Sidebar renders the Orchestrator item as a normal clickable link**

**Step 5: Commit**

```bash
git add frontend/src/components/layout/Sidebar.tsx
git commit -m "feat(orchestrator): restore orchestrator nav item, remove disabled state"
```

---

### Task 8: TypeScript — declare `window.bashgym` type if not already typed

**Files:**
- Check: `frontend/src/types/` or `frontend/src/electron.d.ts` (or similar)
- Possibly modify: `frontend/src/vite-env.d.ts` or create `frontend/src/types/electron.d.ts`

**Step 1: Find where window.bashgym is typed**

```bash
grep -r "bashgym" frontend/src --include="*.d.ts" -l
grep -r "interface.*bashgym\|window\.bashgym" frontend/src --include="*.ts" --include="*.tsx" -l | head -5
```

**Step 2: Ensure `terminal.write` is typed**

If `window.bashgym` isn't typed yet, add to `frontend/src/vite-env.d.ts` or a new `frontend/src/types/electron.d.ts`:

```typescript
interface BashGymTerminalBridge {
  write: (id: string, data: string) => void
  kill: (id: string) => void
  resize: (id: string, cols: number, rows: number) => void
}

interface Window {
  bashgym?: {
    terminal: BashGymTerminalBridge
    // other namespaces as needed
  }
}
```

**Step 3: Fix any TypeScript errors introduced by removing `isDisabled`**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

Fix any remaining type errors from the store changes.

**Step 4: Commit**

```bash
git add frontend/src
git commit -m "fix(types): ensure window.bashgym terminal bridge is typed"
```

---

### Task 9: End-to-end verification

**Step 1: Restart the backend**

```bash
# Kill existing uvicorn and restart
.\kill_api.ps1
python run_backend.py
```

**Step 2: Verify orchestration API is ungated**

```bash
curl -s -w "\nHTTP:%{http_code}" http://localhost:8003/api/orchestrate/jobs
```

Expected: `{"jobs":[]}` with HTTP 200.

**Step 3: Test spec submission**

In the UI:
1. Open Orchestrator → Submit tab
2. Confirm form has no repository/branch/workers fields
3. Fill in Title + Description
4. Click "Submit Spec"
5. Auto-switch to Active Job tab — loading state shows decomposition in progress

**Step 4: Test inline editing**

1. Wait for decomposition to complete (status: awaiting_approval)
2. Expand any task card
3. Confirm the editable "Claude Prompt" textarea appears
4. Edit the prompt — confirm "Edited" indicator appears and "Reset to original" button shows
5. Click reset — confirm prompt returns to original

**Step 5: Test terminal dispatch**

1. Open the Workspace (terminal grid) and ensure at least one terminal is idle
2. Return to Orchestrator → Active Job
3. Click "Dispatch to Terminals"
4. Switch to the terminal grid — confirm `claude "[prompt]"` was written to an idle terminal
5. Confirm the task badge in the DAG shows `→ Terminal N`

**Step 6: Test queue behavior**

1. Submit a spec that decomposes into more tasks than idle terminals
2. Dispatch — first N tasks go immediately, rest show as queued
3. Wait for a terminal to finish (return to idle)
4. Confirm the next queued task auto-dispatches

**Step 7: Final commit if any fixups**

```bash
git add -p
git commit -m "fix(orchestrator): terminal dispatch e2e fixups"
```

---

## Summary of changed files

| File | Change |
|---|---|
| `bashgym/api/orchestrator_routes.py` | Remove feature gate, change approve to return tasks |
| `frontend/src/stores/orchestratorStore.ts` | Add dispatch state + actions, remove isDisabled |
| `frontend/src/services/api.ts` | Remove baseBranch from approveJob |
| `frontend/src/components/orchestrator/SpecForm.tsx` | Remove repo/branch/workers |
| `frontend/src/components/orchestrator/TaskDAGView.tsx` | Add inline editing, terminal badges |
| `frontend/src/components/orchestrator/OrchestratorDashboard.tsx` | Remove disabled state, add queue watcher |
| `frontend/src/components/layout/Sidebar.tsx` | Restore normal orchestrator nav item |
| `frontend/src/types/electron.d.ts` (possibly new) | Type window.bashgym.terminal.write |
