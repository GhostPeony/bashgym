import { create } from 'zustand'

import { tracesApi } from '../services/api'
import { traceCountsResource } from './appResources'

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
  cognitiveQuality: number
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
  toolBreakdown?: Record<string, number>
  createdAt: number
  promotedAt?: number
}

export interface TraceListQuery {
  status: TraceStatus | 'all'
  repo: string | null
  sourceTool: string
}

export const TRACE_PAGE_SIZE = 50

const queryKeyOf = (query: TraceListQuery): string =>
  `${query.status}|${query.repo ?? ''}|${query.sourceTool}`

// Maps an API trace payload to the store Trace shape.
export function mapApiTrace(t: any): Trace {
  return {
    id: t.trace_id,
    taskId: t.task_id,
    taskDescription: t.task_description,
    status: t.status,
    qualityTier: t.quality_tier || undefined,
    steps: [],
    quality: {
      successRate: t.quality.success_rate,
      verificationScore: t.quality.verification_score,
      complexityScore: t.quality.complexity_score,
      lengthScore: t.quality.length_score,
      toolDiversity: t.quality.tool_diversity || 0,
      efficiencyScore: t.quality.efficiency_score || 0,
      cognitiveQuality: t.quality.cognitive_quality || 0,
      totalScore: t.quality.total_score,
    },
    repo: t.repo
      ? {
          name: t.repo.name,
          path: t.repo.path,
          is_git_repo: t.repo.is_git_repo,
          git_branch: t.repo.git_branch,
          git_remote: t.repo.git_remote,
        }
      : undefined,
    reposCount: t.repos_count || 1,
    toolBreakdown: t.tool_breakdown,
    createdAt: t.created_at ? new Date(t.created_at).getTime() : Date.now(),
    promotedAt: t.promoted_at ? new Date(t.promoted_at).getTime() : undefined,
  }
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

  // List fetch lifecycle — survives page remounts so tab switches render
  // cached traces instantly instead of refetching behind a spinner.
  listLoading: boolean      // fetching with nothing usable cached for this query
  listRefreshing: boolean   // background refetch; cached traces stay visible
  listError: string | null
  listLoadedAt: number | null
  lastQuery: TraceListQuery | null
  totalTraces: number
  hasMore: boolean
  loadingMore: boolean

  // Actions
  setTraces: (traces: Trace[]) => void
  setCounts: (counts: { gold: number; silver: number; bronze: number; failed: number; pending: number }) => void
  addTrace: (trace: Trace) => void
  selectTrace: (id: string | null) => void
  promoteTrace: (id: string) => void
  demoteTrace: (id: string) => void
  setStatusFilter: (status: TraceStatus | 'all') => void
  setTierFilter: (tier: TraceQualityTier | 'all') => void
  setRepoFilter: (repo: string | null) => void
  setSearchQuery: (query: string) => void
  setAvailableRepos: (repos: RepoInfo[]) => void

  /** Fetch the first page only when the query changed or nothing is cached. */
  ensureTraces: (query: TraceListQuery) => Promise<void>
  /** Refetch the current query in the background without blanking the list. */
  refreshTraces: () => Promise<void>
  loadMoreTraces: () => Promise<void>

  // Computed
  filteredTraces: () => Trace[]
}

let listGeneration = 0

export const useTracesStore = create<TracesState>((set, get) => {
  const fetchList = async (query: TraceListQuery, background: boolean): Promise<void> => {
    const generation = ++listGeneration
    set({
      listLoading: !background,
      listRefreshing: background,
      listError: null,
      lastQuery: query,
    })
    try {
      const result = await tracesApi.list({
        limit: TRACE_PAGE_SIZE,
        offset: 0,
        status: query.status !== 'all' ? query.status : undefined,
        repo: query.repo || undefined,
        source_tool: query.sourceTool || undefined,
      })
      if (generation !== listGeneration) return
      if (result.ok && result.data) {
        const data = result.data
        const traceList = Array.isArray(data) ? data : data.traces
        const total = Array.isArray(data) ? data.length : data.total
        set({
          traces: (traceList || []).map(mapApiTrace),
          totalTraces: total,
          hasMore: !Array.isArray(data) && data.offset + data.limit < total,
          listLoadedAt: Date.now(),
        })
        if (!Array.isArray(data) && data.counts) {
          get().setCounts(data.counts)
          traceCountsResource.getState().setData(data.counts)
        }
      } else {
        set({ listError: result.error || 'Failed to fetch traces' })
      }
    } catch (e) {
      if (generation === listGeneration) set({ listError: String(e) })
    } finally {
      if (generation === listGeneration) set({ listLoading: false, listRefreshing: false })
    }
  }

  return {
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
  listLoading: false,
  listRefreshing: false,
  listError: null,
  listLoadedAt: null,
  lastQuery: null,
  totalTraces: 0,
  hasMore: false,
  loadingMore: false,

  ensureTraces: async (query) => {
    const { lastQuery, listLoadedAt, listLoading, listRefreshing, traces } = get()
    const sameQuery = lastQuery !== null && queryKeyOf(lastQuery) === queryKeyOf(query)
    if (sameQuery && (listLoadedAt !== null || listLoading || listRefreshing)) return
    await fetchList(query, traces.length > 0)
  },

  refreshTraces: async () => {
    const { lastQuery } = get()
    if (!lastQuery) return
    await fetchList(lastQuery, true)
  },

  loadMoreTraces: async () => {
    const { lastQuery, loadingMore, hasMore, traces } = get()
    if (!lastQuery || loadingMore || !hasMore) return
    set({ loadingMore: true })
    try {
      const result = await tracesApi.list({
        limit: TRACE_PAGE_SIZE,
        offset: traces.length,
        status: lastQuery.status !== 'all' ? lastQuery.status : undefined,
        repo: lastQuery.repo || undefined,
        source_tool: lastQuery.sourceTool || undefined,
      })
      if (result.ok && result.data) {
        const data = result.data
        const traceList = Array.isArray(data) ? data : data.traces
        const total = Array.isArray(data) ? data.length : data.total
        set((state) => ({
          traces: [...state.traces, ...(traceList || []).map(mapApiTrace)],
          totalTraces: total,
          hasMore: !Array.isArray(data) && data.offset + data.limit < total,
        }))
        if (!Array.isArray(data) && data.counts) {
          get().setCounts(data.counts)
        }
      }
    } catch (e) {
      console.error('Failed to load more traces:', e)
    } finally {
      set({ loadingMore: false })
    }
  },

  setTraces: (traces) => {
    // Only recount from loaded traces if setCounts hasn't been called with server data
    // This keeps the store working for callers that don't provide server counts
    const goldCount = traces.filter((t) => t.status === 'gold').length
    const silverCount = traces.filter((t) => t.status === 'silver').length
    const bronzeCount = traces.filter((t) => t.status === 'bronze').length
    const failedCount = traces.filter((t) => t.status === 'failed').length
    const pendingCount = traces.filter((t) => t.status === 'pending').length

    set({ traces, goldCount, silverCount, bronzeCount, failedCount, pendingCount })
  },

  setCounts: (counts) => {
    set({
      goldCount: counts.gold,
      silverCount: counts.silver,
      bronzeCount: counts.bronze,
      failedCount: counts.failed,
      pendingCount: counts.pending
    })
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
    const { traces, tierFilter, searchQuery } = get()

    // Status and repo filtering is done server-side via API params.
    // Only tier and search are applied client-side.
    return traces.filter((trace) => {
      const matchesTier = tierFilter === 'all' || trace.qualityTier === tierFilter
      const matchesSearch =
        !searchQuery ||
        trace.taskDescription.toLowerCase().includes(searchQuery.toLowerCase()) ||
        trace.id.includes(searchQuery) ||
        trace.repo?.name.toLowerCase().includes(searchQuery.toLowerCase())

      return matchesTier && matchesSearch
    })
  }
  }
})
