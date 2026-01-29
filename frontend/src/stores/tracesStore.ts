import { create } from 'zustand'

export type TraceStatus = 'gold' | 'silver' | 'bronze' | 'failed' | 'pending'

// Quality tier based on NVIDIA NeMo research
// Gold: ≥90% success, ≥0.75 quality → SFT training
// Silver: ≥75% success, ≥0.55 quality → DPO chosen
// Bronze: ≥60% success, ≥0.40 quality → DPO rejected
// Rejected: <60% success → Not suitable for training
export type TraceQualityTier = 'gold' | 'silver' | 'bronze' | 'rejected'

export interface TraceStep {
  index: number
  action: string
  tool?: string
  input?: string
  output?: string
  timestamp: number
}

export interface QualityMetrics {
  successRate: number
  verificationScore: number
  complexityScore: number
  lengthScore: number
  toolDiversity: number
  efficiencyScore: number
  totalScore: number
}

export interface RepoInfo {
  name: string
  path?: string
  git_remote?: string
  git_branch?: string
  is_git_repo: boolean
  trace_count?: number
}

export interface Trace {
  id: string
  taskId: string
  taskDescription: string
  status: TraceStatus
  qualityTier?: TraceQualityTier
  steps: TraceStep[]
  quality: QualityMetrics
  repo?: RepoInfo
  reposCount: number
  createdAt: number
  promotedAt?: number
}

interface TracesState {
  // Data
  traces: Trace[]
  selectedTraceId: string | null
  availableRepos: RepoInfo[]

  // Filters
  statusFilter: TraceStatus | 'all'
  tierFilter: TraceQualityTier | 'all'  // Filter by quality tier
  repoFilter: string | null  // repo name to filter by, null = all repos
  searchQuery: string

  // Stats (tiered counts)
  goldCount: number
  silverCount: number
  bronzeCount: number
  failedCount: number
  pendingCount: number

  // Actions
  setTraces: (traces: Trace[]) => void
  addTrace: (trace: Trace) => void
  selectTrace: (id: string | null) => void
  promoteTrace: (id: string) => void
  demoteTrace: (id: string) => void
  setStatusFilter: (status: TraceStatus | 'all') => void
  setTierFilter: (tier: TraceQualityTier | 'all') => void
  setRepoFilter: (repo: string | null) => void
  setSearchQuery: (query: string) => void
  setAvailableRepos: (repos: RepoInfo[]) => void

  // Computed
  filteredTraces: () => Trace[]
}

export const useTracesStore = create<TracesState>((set, get) => ({
  traces: [],
  selectedTraceId: null,
  availableRepos: [],
  statusFilter: 'all',
  tierFilter: 'all',
  repoFilter: null,
  searchQuery: '',
  goldCount: 0,
  silverCount: 0,
  bronzeCount: 0,
  failedCount: 0,
  pendingCount: 0,

  setTraces: (traces) => {
    // Count by status (includes tiered statuses)
    const goldCount = traces.filter((t) => t.status === 'gold').length
    const silverCount = traces.filter((t) => t.status === 'silver').length
    const bronzeCount = traces.filter((t) => t.status === 'bronze').length
    const failedCount = traces.filter((t) => t.status === 'failed').length
    const pendingCount = traces.filter((t) => t.status === 'pending').length

    set({ traces, goldCount, silverCount, bronzeCount, failedCount, pendingCount })
  },

  addTrace: (trace) => {
    set((state) => {
      const newTraces = [...state.traces, trace]
      return {
        traces: newTraces,
        goldCount: trace.status === 'gold' ? state.goldCount + 1 : state.goldCount,
        silverCount: trace.status === 'silver' ? state.silverCount + 1 : state.silverCount,
        bronzeCount: trace.status === 'bronze' ? state.bronzeCount + 1 : state.bronzeCount,
        failedCount: trace.status === 'failed' ? state.failedCount + 1 : state.failedCount,
        pendingCount: trace.status === 'pending' ? state.pendingCount + 1 : state.pendingCount
      }
    })
  },

  selectTrace: (id) => set({ selectedTraceId: id }),

  promoteTrace: (id) => {
    set((state) => {
      const trace = state.traces.find((t) => t.id === id)
      if (!trace || trace.status === 'gold') return state

      const wasStatus = trace.status
      const newTraces = state.traces.map((t) =>
        t.id === id ? { ...t, status: 'gold' as TraceStatus, promotedAt: Date.now() } : t
      )

      return {
        traces: newTraces,
        goldCount: state.goldCount + 1,
        failedCount: wasStatus === 'failed' ? state.failedCount - 1 : state.failedCount,
        pendingCount: wasStatus === 'pending' ? state.pendingCount - 1 : state.pendingCount
      }
    })
  },

  demoteTrace: (id) => {
    set((state) => {
      const trace = state.traces.find((t) => t.id === id)
      if (!trace || trace.status !== 'gold') return state

      // Backend demotes to 'failed' status
      const newTraces = state.traces.map((t) =>
        t.id === id ? { ...t, status: 'failed' as TraceStatus, promotedAt: undefined } : t
      )

      return {
        traces: newTraces,
        goldCount: state.goldCount - 1,
        failedCount: state.failedCount + 1
      }
    })
  },

  setStatusFilter: (status) => set({ statusFilter: status }),

  setTierFilter: (tier) => set({ tierFilter: tier }),

  setRepoFilter: (repo) => set({ repoFilter: repo }),

  setSearchQuery: (query) => set({ searchQuery: query }),

  setAvailableRepos: (repos) => set({ availableRepos: repos }),

  filteredTraces: () => {
    const { traces, statusFilter, tierFilter, repoFilter, searchQuery } = get()

    return traces.filter((trace) => {
      const matchesStatus = statusFilter === 'all' || trace.status === statusFilter
      const matchesTier = tierFilter === 'all' || trace.qualityTier === tierFilter
      const matchesRepo = !repoFilter || trace.repo?.name === repoFilter
      const matchesSearch =
        !searchQuery ||
        trace.taskDescription.toLowerCase().includes(searchQuery.toLowerCase()) ||
        trace.id.includes(searchQuery) ||
        trace.repo?.name.toLowerCase().includes(searchQuery.toLowerCase())

      return matchesStatus && matchesTier && matchesRepo && matchesSearch
    })
  }
}))
