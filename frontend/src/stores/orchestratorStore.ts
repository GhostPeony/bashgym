import { create } from 'zustand'
import { orchestratorApi } from '../services/api'

export interface TaskNode {
  id: string
  title: string
  description: string
  priority: 'CRITICAL' | 'HIGH' | 'NORMAL' | 'LOW'
  status: 'pending' | 'assigned' | 'running' | 'completed' | 'failed' | 'blocked' | 'cancelled' | 'retrying' | 'dispatched'
  dependencies: string[]
  files_touched: string[]
  estimated_turns: number
  budget_usd: number
  retry_count: number
  worker_prompt?: string
  worker_id?: string
  cost_usd?: number
  duration_seconds?: number
  error?: string
}

export interface OrchestratorJob {
  jobId: string
  status: 'decomposing' | 'awaiting_approval' | 'executing' | 'dispatched' | 'completed' | 'failed' | 'cancelled'
  title: string
  tasks: Record<string, TaskNode>
  stats: { pending: number; in_progress: number; completed: number; failed: number }
  budget: { limit_usd: number; spent_usd: number; remaining_usd: number; exceeded: boolean }
  totalCost: number
  totalTime: number
  mergeSuccesses: number
  mergeFailures: number
  error?: string
}

export interface SpecInput {
  title: string
  description: string
  constraints?: string[]
  acceptance_criteria?: string[]
  max_budget_usd?: number
  llm_config?: {
    provider?: string
    model?: string
    temperature?: number
  }
}

interface OrchestratorState {
  currentJob: OrchestratorJob | null
  jobs: Array<{ jobId: string; status: string; title: string; taskCount?: number }>
  providers: Array<{ provider: string; default_model: string; env_key: string; base_url?: string }>
  activeTab: 'submit' | 'active' | 'history'
  taskAssignments: Record<string, string>   // taskId → terminalId
  taskQueue: string[]                        // taskIds waiting for an idle terminal
  editedPrompts: Record<string, string>      // taskId → user-edited prompt

  // Actions
  setActiveTab: (tab: 'submit' | 'active' | 'history') => void
  submitSpec: (spec: SpecInput) => Promise<string | null>
  approveJob: (jobId: string) => Promise<void>
  cancelJob: (jobId: string) => Promise<void>
  retryTask: (jobId: string, taskId: string, prompt?: string) => Promise<void>
  fetchStatus: (jobId: string) => Promise<void>
  fetchJobs: () => Promise<void>
  fetchProviders: () => Promise<void>
  setEditedPrompt: (taskId: string, prompt: string) => void
  resetEditedPrompt: (taskId: string) => void
  dispatchToTerminals: () => void
  dispatchNextQueued: (terminalId: string) => void

  // WS handlers
  handleTaskStarted: (payload: any) => void
  handleTaskCompleted: (payload: any) => void
  handleTaskFailed: (payload: any) => void
  handleBudgetUpdate: (payload: any) => void
  handleComplete: (payload: any) => void
  handleDecomposing: (payload: any) => void
  handleReady: (payload: any) => void
  handleCancelled: (payload: any) => void
  handleTaskRetrying: (payload: any) => void
  handleMergeResult: (payload: any) => void
}

function escapePrompt(prompt: string): string {
  return prompt
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/`/g, '\\`')
    .replace(/\$/g, '\\$')
    .replace(/!/g, '\\!')
}

export const useOrchestratorStore = create<OrchestratorState>((set, get) => ({
  currentJob: null,
  jobs: [],
  providers: [],
  activeTab: 'submit',
  taskAssignments: {},
  taskQueue: [],
  editedPrompts: {},

  setActiveTab: (tab) => set({ activeTab: tab }),

  submitSpec: async (spec) => {
    const result = await orchestratorApi.submitSpec(spec)
    if (result.ok && result.data) {
      const jobId = result.data.job_id
      set({
        currentJob: {
          jobId,
          status: 'decomposing',
          title: spec.title,
          tasks: {},
          stats: { pending: 0, in_progress: 0, completed: 0, failed: 0 },
          budget: {
            limit_usd: spec.max_budget_usd || 10.0,
            spent_usd: 0,
            remaining_usd: spec.max_budget_usd || 10.0,
            exceeded: false,
          },
          totalCost: 0,
          totalTime: 0,
          mergeSuccesses: 0,
          mergeFailures: 0,
        },
        activeTab: 'active',
        taskAssignments: {},
        taskQueue: [],
        editedPrompts: {},
      })
      return jobId
    }
    return null
  },

  approveJob: async (jobId) => {
    const result = await orchestratorApi.approveJob(jobId)
    if (result.ok) {
      set((state) => ({
        currentJob: state.currentJob
          ? { ...state.currentJob, status: 'dispatched' }
          : null,
      }))
      // Dispatch tasks to idle terminals
      get().dispatchToTerminals()
    }
  },

  cancelJob: async (jobId) => {
    const result = await orchestratorApi.cancelJob(jobId)
    if (result.ok) {
      set((state) => ({
        currentJob: state.currentJob
          ? { ...state.currentJob, status: 'cancelled' }
          : null,
      }))
    }
  },

  retryTask: async (jobId, taskId, prompt) => {
    await orchestratorApi.retryTask(jobId, taskId, prompt)
  },

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

  dispatchToTerminals: () => {
    const { currentJob, editedPrompts } = get()
    if (!currentJob) return

    // Lazy-import terminalStore to avoid circular dependency
    const { useTerminalStore } = require('./terminalStore') as typeof import('./terminalStore')
    const { sessions, panels } = useTerminalStore.getState()

    const idleTerminalIds = Array.from(sessions.entries())
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
        window.bashgym?.terminal.write(terminalId, `claude "${escapePrompt(prompt)}"\r`)
        assignments[task.id] = terminalId
      } else {
        queue.push(task.id)
      }
    })

    set({ taskAssignments: assignments, taskQueue: queue })
  },

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
    window.bashgym?.terminal.write(terminalId, `claude "${escapePrompt(prompt)}"\r`)

    set((state) => ({
      taskQueue: remaining,
      taskAssignments: { ...state.taskAssignments, [nextTaskId]: terminalId },
    }))
  },

  fetchStatus: async (jobId) => {
    const result = await orchestratorApi.getStatus(jobId)
    if (result.ok && result.data) {
      const data = result.data
      const tasks: Record<string, TaskNode> = {}
      if (data.dag?.tasks && Array.isArray(data.dag.tasks)) {
        for (const t of data.dag.tasks) {
          tasks[t.id] = {
            id: t.id,
            title: t.title || t.id,
            description: t.description || '',
            priority: t.priority || 'NORMAL',
            status: t.status || 'pending',
            dependencies: t.dependencies || [],
            files_touched: t.files_touched || [],
            estimated_turns: t.estimated_turns || 20,
            budget_usd: t.budget_usd || 2.0,
            retry_count: t.retry_count || 0,
            worker_prompt: t.worker_prompt,
            worker_id: t.worker_id,
            cost_usd: t.result?.cost_usd,
            duration_seconds: t.result?.duration_seconds,
            error: t.result?.error,
          }
        }
      }
      const rawStats = data.dag?.stats || data.stats || {}
      const stats = {
        pending: rawStats.pending || 0,
        in_progress: rawStats.in_progress || rawStats.running || 0,
        completed: rawStats.completed || 0,
        failed: rawStats.failed || 0,
      }
      set({
        currentJob: {
          jobId,
          status: data.status || 'dispatched',
          title: data.title || get().currentJob?.title || '',
          tasks,
          stats,
          budget: data.budget || {
            limit_usd: 10,
            spent_usd: data.total_cost || 0,
            remaining_usd: 10 - (data.total_cost || 0),
            exceeded: false,
          },
          totalCost: data.total_cost || 0,
          totalTime: data.total_time || 0,
          mergeSuccesses: data.synthesis?.merge_successes ?? 0,
          mergeFailures: data.synthesis?.merge_failures ?? 0,
        },
      })
    }
  },

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

  fetchProviders: async () => {
    const result = await orchestratorApi.listProviders()
    if (result.ok && result.data) {
      set({ providers: result.data.providers || [] })
    }
  },

  // WebSocket handlers
  handleDecomposing: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      return { currentJob: { ...state.currentJob, status: 'decomposing' } }
    })
  },

  handleReady: (payload) => {
    const { job_id } = payload
    get().fetchStatus(job_id).then(() => {
      set((state) => {
        if (!state.currentJob || state.currentJob.jobId !== job_id) return state
        return { currentJob: { ...state.currentJob, status: 'awaiting_approval' } }
      })
    })
  },

  handleTaskStarted: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      const taskId = payload.task_id
      const tasks = { ...state.currentJob.tasks }
      if (tasks[taskId]) {
        tasks[taskId] = { ...tasks[taskId], status: 'running', worker_id: payload.worker_id }
      }
      return {
        currentJob: {
          ...state.currentJob,
          tasks,
          stats: {
            ...state.currentJob.stats,
            in_progress: (state.currentJob.stats.in_progress || 0) + 1,
            pending: Math.max(0, (state.currentJob.stats.pending || 0) - 1),
          },
        },
      }
    })
  },

  handleTaskCompleted: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      const taskId = payload.task_id
      const tasks = { ...state.currentJob.tasks }
      if (tasks[taskId]) {
        tasks[taskId] = {
          ...tasks[taskId],
          status: 'completed',
          cost_usd: payload.cost_usd,
          duration_seconds: payload.duration_seconds,
        }
      }
      return {
        currentJob: {
          ...state.currentJob,
          tasks,
          stats: {
            ...state.currentJob.stats,
            completed: (state.currentJob.stats.completed || 0) + 1,
            in_progress: Math.max(0, (state.currentJob.stats.in_progress || 0) - 1),
          },
        },
      }
    })
  },

  handleTaskFailed: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      const taskId = payload.task_id
      const tasks = { ...state.currentJob.tasks }
      if (tasks[taskId]) {
        tasks[taskId] = { ...tasks[taskId], status: 'failed', error: payload.error }
      }
      return {
        currentJob: {
          ...state.currentJob,
          tasks,
          stats: {
            ...state.currentJob.stats,
            failed: (state.currentJob.stats.failed || 0) + 1,
            in_progress: Math.max(0, (state.currentJob.stats.in_progress || 0) - 1),
          },
        },
      }
    })
  },

  handleBudgetUpdate: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      return {
        currentJob: {
          ...state.currentJob,
          budget: {
            limit_usd: payload.budget_usd || state.currentJob.budget.limit_usd,
            spent_usd: payload.spent_usd || 0,
            remaining_usd: payload.remaining_usd || 0,
            exceeded: (payload.spent_usd || 0) > (payload.budget_usd || state.currentJob.budget.limit_usd),
          },
          totalCost: payload.spent_usd || state.currentJob.totalCost,
        },
      }
    })
  },

  handleComplete: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      return {
        currentJob: {
          ...state.currentJob,
          status: 'completed',
          totalCost: payload.total_cost_usd || state.currentJob.totalCost,
          totalTime: payload.total_time_seconds || state.currentJob.totalTime,
          mergeSuccesses: payload.merge_successes ?? state.currentJob.mergeSuccesses,
          mergeFailures: payload.merge_failures ?? state.currentJob.mergeFailures,
        },
      }
    })
  },

  handleCancelled: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      return { currentJob: { ...state.currentJob, status: 'cancelled' } }
    })
  },

  handleTaskRetrying: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      if (!state.currentJob.tasks[payload.task_id]) return state
      const job = state.currentJob
      return {
        currentJob: {
          ...job,
          tasks: {
            ...job.tasks,
            [payload.task_id]: {
              ...job.tasks[payload.task_id],
              status: 'retrying',
              retry_count: (job.tasks[payload.task_id].retry_count || 0) + 1,
            },
          },
        },
      }
    })
  },

  handleMergeResult: (payload) => {
    set((state) => {
      if (!state.currentJob || state.currentJob.jobId !== payload.job_id) return state
      const job = state.currentJob
      return {
        currentJob: {
          ...job,
          mergeSuccesses: job.mergeSuccesses + (payload.success ? 1 : 0),
          mergeFailures: job.mergeFailures + (payload.success ? 0 : 1),
        },
      }
    })
  },
}))
