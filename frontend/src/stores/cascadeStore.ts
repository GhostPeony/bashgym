import { create } from 'zustand'
import { cascadeApi } from '../services/api'

export type CascadeStatus = 'idle' | 'running' | 'completed' | 'failed'
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped'

export interface CascadeStage {
  stageNumber: number
  domain: string
  status: StageStatus
  runId?: string
  metrics?: Record<string, number>
  error?: string
  startedAt?: number
  completedAt?: number
}

interface CascadePayload {
  stage_number?: number
  stage?: number
  domain?: string
  run_id?: string
  runId?: string
  metrics?: Record<string, number>
  error?: string
  progress?: number
  current_stage?: number
  total_stages?: number
  [key: string]: unknown
}

interface CascadeState {
  status: CascadeStatus
  error: string | null
  stages: CascadeStage[]
  currentStageNumber: number | null
  totalStages: number | null
  progress: number
  lastUpdatedAt: number | null

  handleStageStarted: (payload: CascadePayload) => void
  handleStageCompleted: (payload: CascadePayload) => void
  handleStageFailed: (payload: CascadePayload) => void
  handleStageSkipped: (payload: CascadePayload) => void
  handleCompleted: (payload: CascadePayload) => void
  handleProgress: (payload: CascadePayload) => void
  setPreflightError: (message: string | null) => void
  reset: () => void

  startCascade: (
    config: Parameters<typeof cascadeApi.start>[0]
  ) => Promise<{ ok: boolean; error?: string }>
  stopCascade: () => Promise<void>
}

const emptyState = {
  status: 'idle' as CascadeStatus,
  error: null as string | null,
  stages: [] as CascadeStage[],
  currentStageNumber: null as number | null,
  totalStages: null as number | null,
  progress: 0,
  lastUpdatedAt: null as number | null
}

function stageNum(p: CascadePayload): number {
  return Number(p.stage_number ?? p.stage ?? p.current_stage ?? 0)
}

function upsertStage(
  stages: CascadeStage[],
  stageNumber: number,
  patch: Partial<CascadeStage>
): CascadeStage[] {
  const idx = stages.findIndex((s) => s.stageNumber === stageNumber)
  if (idx === -1) {
    const fresh: CascadeStage = {
      stageNumber,
      domain: '',
      status: 'pending',
      ...patch
    }
    return [...stages, fresh].sort((a, b) => a.stageNumber - b.stageNumber)
  }
  const next = stages.slice()
  next[idx] = { ...next[idx], ...patch }
  return next
}

export const useCascadeStore = create<CascadeState>((set, get) => ({
  ...emptyState,

  handleStageStarted: (payload) => {
    const num = stageNum(payload)
    set((state) => ({
      status: 'running',
      currentStageNumber: num,
      stages: upsertStage(state.stages, num, {
        domain: payload.domain ?? '',
        status: 'running',
        runId: (payload.run_id ?? payload.runId) as string | undefined,
        startedAt: Date.now()
      }),
      lastUpdatedAt: Date.now()
    }))
  },

  handleStageCompleted: (payload) => {
    const num = stageNum(payload)
    set((state) => ({
      stages: upsertStage(state.stages, num, {
        status: 'completed',
        metrics: payload.metrics,
        completedAt: Date.now()
      }),
      lastUpdatedAt: Date.now()
    }))
  },

  handleStageFailed: (payload) => {
    const num = stageNum(payload)
    const err = (payload.error as string | undefined) ?? 'Stage failed'
    set((state) => ({
      status: 'failed',
      error: err,
      stages: upsertStage(state.stages, num, {
        status: 'failed',
        error: err,
        completedAt: Date.now()
      }),
      lastUpdatedAt: Date.now()
    }))
  },

  handleStageSkipped: (payload) => {
    const num = stageNum(payload)
    set((state) => ({
      stages: upsertStage(state.stages, num, {
        status: 'skipped',
        completedAt: Date.now()
      }),
      lastUpdatedAt: Date.now()
    }))
  },

  handleCompleted: (_payload) => {
    set({
      status: 'completed',
      progress: 1,
      lastUpdatedAt: Date.now()
    })
  },

  handleProgress: (payload) => {
    const num = stageNum(payload)
    set((state) => ({
      currentStageNumber: num > 0 ? num : state.currentStageNumber,
      totalStages:
        typeof payload.total_stages === 'number' ? payload.total_stages : state.totalStages,
      progress: typeof payload.progress === 'number' ? payload.progress : state.progress,
      lastUpdatedAt: Date.now()
    }))
  },

  setPreflightError: (message) => {
    set({ error: message })
  },

  reset: () => {
    set({ ...emptyState })
  },

  startCascade: async (config) => {
    set({ ...emptyState, status: 'running' })
    try {
      const response = await cascadeApi.start(config)
      if (!response.ok) {
        const err = response.error || 'Failed to start cascade'
        set({ status: 'failed', error: err })
        return { ok: false, error: err }
      }
    } catch (error) {
      const err = error instanceof Error ? error.message : 'Failed to start cascade'
      set({ status: 'failed', error: err })
      return { ok: false, error: err }
    }

    // Poll status for preflight errors that surface after the start call returns.
    // The backend's /cascade/status returns { status, error, ... }.
    const deadline = Date.now() + 15_000
    const poll = async () => {
      if (get().status !== 'running') return
      if (Date.now() > deadline) return
      try {
        const res = await cascadeApi.getStatus()
        if (res.ok && res.data) {
          const data = res.data as Record<string, unknown>
          const backendStatus = data.status as string | undefined
          const backendError = data.error as string | undefined
          if (backendStatus === 'failed' && backendError) {
            set({ status: 'failed', error: backendError })
            return
          }
          if (backendStatus === 'running' || backendStatus === 'completed') {
            // Live events take over from here — stop polling for preflight errors.
            return
          }
        }
      } catch {
        /* ignore transient poll errors */
      }
      setTimeout(poll, 1500)
    }
    setTimeout(poll, 1500)

    return { ok: true }
  },

  stopCascade: async () => {
    try {
      await cascadeApi.stop()
      set({ status: 'idle' })
    } catch {
      /* */
    }
  }
}))
