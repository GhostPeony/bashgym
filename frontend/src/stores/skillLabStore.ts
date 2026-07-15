import { create } from 'zustand'
import {
  skillLabApi,
  type SkillLabRun,
  type SkillLabRunRequest,
} from '../services/api'
import { useActivityStore } from './activityStore'

export interface SkillRunTransition {
  type: 'skill-eval:started' | 'skill-eval:completed' | 'skill-eval:failed'
  run: SkillLabRun
}

export function skillRunTransitions(
  previous: readonly SkillLabRun[],
  next: readonly SkillLabRun[],
): SkillRunTransition[] {
  const prior = new Map(previous.map((run) => [run.run_id, run]))
  const transitions: SkillRunTransition[] = []
  for (const run of next) {
    const before = prior.get(run.run_id)
    if (!before && (run.status === 'queued' || run.status === 'running')) {
      transitions.push({ type: 'skill-eval:started', run })
    } else if (before && before.status !== run.status && run.status === 'completed') {
      transitions.push({ type: 'skill-eval:completed', run })
    } else if (before && before.status !== run.status && run.status === 'failed') {
      transitions.push({ type: 'skill-eval:failed', run })
    }
  }
  return transitions
}

interface SkillLabState {
  runsByWorkspace: Record<string, SkillLabRun[]>
  loadingByWorkspace: Record<string, boolean>
  errorByWorkspace: Record<string, string | null>
  refresh: (workspaceId: string) => Promise<SkillLabRun[]>
  launch: (payload: SkillLabRunRequest) => Promise<SkillLabRun>
}

function publishTransition(transition: SkillRunTransition) {
  const { run } = transition
  useActivityStore.getState().addEvent(transition.type, {
    run_id: run.run_id,
    skill_id: run.skill_id,
    skill_name: run.skill_name,
    endpoint_id: run.endpoint_id,
    verdict: run.kpis?.verdict,
    uplift: run.kpis?.success_uplift,
    error: run.error,
  })
}

export const useSkillLabStore = create<SkillLabState>((set, get) => ({
  runsByWorkspace: {},
  loadingByWorkspace: {},
  errorByWorkspace: {},

  refresh: async (workspaceId) => {
    set((state) => ({
      loadingByWorkspace: { ...state.loadingByWorkspace, [workspaceId]: true },
    }))
    const response = await skillLabApi.listRuns(workspaceId, 30)
    if (!response.ok || !response.data) {
      const message = response.error || 'Unable to load skill eval runs'
      set((state) => ({
        loadingByWorkspace: { ...state.loadingByWorkspace, [workspaceId]: false },
        errorByWorkspace: { ...state.errorByWorkspace, [workspaceId]: message },
      }))
      return get().runsByWorkspace[workspaceId] || []
    }

    const previous = get().runsByWorkspace[workspaceId] || []
    for (const transition of skillRunTransitions(previous, response.data)) {
      publishTransition(transition)
    }
    set((state) => ({
      runsByWorkspace: { ...state.runsByWorkspace, [workspaceId]: response.data! },
      loadingByWorkspace: { ...state.loadingByWorkspace, [workspaceId]: false },
      errorByWorkspace: { ...state.errorByWorkspace, [workspaceId]: null },
    }))
    return response.data
  },

  launch: async (payload) => {
    const response = await skillLabApi.launch(payload)
    if (!response.ok || !response.data) throw new Error(response.error || 'Unable to start skill eval')
    const run = response.data
    set((state) => ({
      runsByWorkspace: {
        ...state.runsByWorkspace,
        [payload.workspace_id]: [
          run,
          ...(state.runsByWorkspace[payload.workspace_id] || []).filter((item) => item.run_id !== run.run_id),
        ],
      },
      errorByWorkspace: { ...state.errorByWorkspace, [payload.workspace_id]: null },
    }))
    publishTransition({ type: 'skill-eval:started', run })
    return run
  },
}))
