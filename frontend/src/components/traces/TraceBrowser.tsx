import { useState, useEffect, useCallback } from 'react'
import {
  Search,
  Star,
  StarOff,
  ChevronRight,
  XCircle,
  Clock,
  BarChart3,
  RefreshCw,
  FolderGit2,
  ChevronDown,
  ChevronUp,
  FileCode,
  Download,
  Layers,
  Sparkles,
  Medal,
  Award,
  Trophy,
  Wand2,
  X,
  GitBranch,
  Hash,
  Wrench,
  Loader2,
  Check,
  List
} from 'lucide-react'
import TraceUpload from './TraceUpload'
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts'
import { useTracesStore, useThemeStore } from '../../stores'
import type { Trace, TraceStatus, TraceQualityTier } from '../../stores'
import { tracesApi, RepoInfo, TraceDetailInfo } from '../../services/api'
import { clsx } from 'clsx'
import { TraceAnalytics } from './TraceAnalytics'

const TRACES_VISIT_KEY = 'bashgym-last-traces-visit'

// Empty state placeholder when no traces exist
const emptyTracePlaceholder: Trace = {
  id: 'no-traces',
  taskId: '',
  taskDescription: 'No traces captured yet. Start coding with Claude Code or OpenCode to capture traces.',
  status: 'pending',
  steps: [],
  quality: {
    successRate: 0,
    verificationScore: 0,
    complexityScore: 0,
    lengthScore: 0,
    toolDiversity: 0,
    efficiencyScore: 0,
    cognitiveQuality: 0,
    totalScore: 0
  },
  reposCount: 0,
  createdAt: Date.now()
}

// Source tools available for import
const IMPORT_SOURCES = [
  { key: 'claude', label: 'Claude', value: 'claude_code' },
  { key: 'gemini', label: 'Gemini', value: 'gemini_cli' },
  { key: 'copilot', label: 'Copilot', value: 'copilot_cli' },
  { key: 'opencode', label: 'OpenCode', value: 'opencode' },
  { key: 'codex', label: 'Codex', value: 'codex' },
] as const

/** Sanitize raw trace description into a clean short title */
function sanitizeTraceTitle(raw: string): string {
  if (!raw) return 'Untitled Trace'
  let s = raw
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/[\r\n]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim()
  const end = s.search(/[.!?]\s/)
  if (end > 0 && end < 120) s = s.slice(0, end + 1)
  else if (s.length > 100) s = s.slice(0, 100).replace(/\s+\S*$/, '') + '...'
  return s || 'Untitled Trace'
}

/** Format timestamp to relative time */
function formatRelativeTime(ts: number): string {
  const d = Math.floor((Date.now() - ts) / 1000)
  if (d < 60) return 'Just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  const days = Math.floor(d / 86400)
  if (days <= 30) return `${days}d ago`
  return new Date(ts).toLocaleDateString()
}

// Training example type
interface TrainingExample {
  example_id: string
  user_prompt: string
  assistant_response: string
  step_count: number
  success_rate: number
  confidence: number
}

type DetailTab = 'overview' | 'examples'

export function TraceBrowser() {
  const {
    traces, setTraces, setCounts, statusFilter, setStatusFilter, searchQuery, setSearchQuery,
    filteredTraces, promoteTrace, demoteTrace,
    goldCount, silverCount, bronzeCount, failedCount, pendingCount,
    selectTrace, selectedTraceId, repoFilter, setRepoFilter, availableRepos, setAvailableRepos
  } = useTracesStore()
  const { theme } = useThemeStore()
  const [view, setView] = useState<'list' | 'analytics'>('list')
  const [repoDropdownOpen, setRepoDropdownOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [detailTab, setDetailTab] = useState<DetailTab>('overview')
  const [trainingExamples, setTrainingExamples] = useState<TrainingExample[]>([])
  const [examplesLoading, setExamplesLoading] = useState(false)
  const [examplesError, setExamplesError] = useState<string | null>(null)
  const [expandedExamples, setExpandedExamples] = useState<Set<string>>(new Set())
  const [examplesFlash, setExamplesFlash] = useState<string | null>(null)
  const [traceDetail, setTraceDetail] = useState<TraceDetailInfo | null>(null)
  const [timelineData, setTimelineData] = useState<{ time: string; gold: number; failed: number; pending: number }[]>([])
  const [statsLoading, setStatsLoading] = useState(false)
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | 'all'>('7d')
  const [totalTraces, setTotalTraces] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const PAGE_SIZE = 50

  // Import notification banner state
  const [importedSinceLastVisit, setImportedSinceLastVisit] = useState(0)

  // Import panel state
  const [importPanelOpen, setImportPanelOpen] = useState(false)
  const [importingSource, setImportingSource] = useState<string | null>(null)
  const [importResults, setImportResults] = useState<Record<string, { imported: number; errors: number } | null>>({})
  const [lastImportTime, setLastImportTime] = useState<number | null>(null)

  // Source tool filter
  const [selectedSource, setSelectedSource] = useState<string>('')

  // Auto-classify state
  const [classifyLoading, setClassifyLoading] = useState(false)
  const [classifyResult, setClassifyResult] = useState<{
    summary: { gold: number; silver: number; bronze: number; rejected: number; failed: number; pending: number };
    dpo_pairs_count: number;
  } | null>(null)
  const [showClassifyModal, setShowClassifyModal] = useState(false)

  // Map API trace to store Trace format
  const mapTrace = useCallback((t: any): Trace => ({
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
      totalScore: t.quality.total_score
    },
    repo: t.repo ? {
      name: t.repo.name,
      path: t.repo.path,
      is_git_repo: t.repo.is_git_repo,
      git_branch: t.repo.git_branch,
      git_remote: t.repo.git_remote
    } : undefined,
    reposCount: t.repos_count || 1,
    toolBreakdown: t.tool_breakdown,
    createdAt: t.created_at ? new Date(t.created_at).getTime() : Date.now(),
    promotedAt: t.promoted_at ? new Date(t.promoted_at).getTime() : undefined
  }), [])

  // Fetch first page of traces from API (respects current filters)
  const fetchTraces = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const result = await tracesApi.list({
        limit: PAGE_SIZE,
        offset: 0,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        repo: repoFilter || undefined,
        source_tool: selectedSource || undefined,
      })
      if (result.ok && result.data) {
        // Handle both paginated { traces, total, counts } and legacy array responses
        const traceList = Array.isArray(result.data) ? result.data : result.data.traces
        const total = Array.isArray(result.data) ? result.data.length : result.data.total
        const mappedTraces = (traceList || []).map(mapTrace)
        setTraces(mappedTraces)
        setTotalTraces(total)
        setHasMore(!Array.isArray(result.data) && result.data.offset + result.data.limit < total)
        // Use server-side counts for accurate totals across all traces
        if (!Array.isArray(result.data) && result.data.counts) {
          setCounts(result.data.counts)
        }
      } else {
        setError(result.error || 'Failed to fetch traces')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setIsLoading(false)
    }
  }, [setTraces, setCounts, mapTrace, statusFilter, repoFilter, selectedSource])

  // Load next page of traces
  const loadMore = useCallback(async () => {
    setLoadingMore(true)
    try {
      const result = await tracesApi.list({
        limit: PAGE_SIZE,
        offset: traces.length,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        repo: repoFilter || undefined,
        source_tool: selectedSource || undefined,
      })
      if (result.ok && result.data) {
        const traceList = Array.isArray(result.data) ? result.data : result.data.traces
        const total = Array.isArray(result.data) ? result.data.length : result.data.total
        const mappedTraces = (traceList || []).map(mapTrace)
        setTraces([...traces, ...mappedTraces])
        setTotalTraces(total)
        setHasMore(!Array.isArray(result.data) && result.data.offset + result.data.limit < total)
        if (!Array.isArray(result.data) && result.data.counts) {
          setCounts(result.data.counts)
        }
      }
    } catch (e) {
      console.error('Failed to load more traces:', e)
    } finally {
      setLoadingMore(false)
    }
  }, [traces, setTraces, setCounts, mapTrace, statusFilter, repoFilter, selectedSource])

  // Fetch trace stats for the timeline chart
  const fetchStats = useCallback(async () => {
    setStatsLoading(true)
    try {
      const result = await tracesApi.stats({ range: timeRange })
      if (result.ok && result.data) {
        setTimelineData(result.data.timeline)
      }
    } catch (e) {
      console.error('Failed to fetch trace stats:', e)
    } finally {
      setStatsLoading(false)
    }
  }, [timeRange])

  // Mount-only: repos, import check
  useEffect(() => {
    tracesApi.listRepos().then((result) => {
      if (result.ok && result.data) {
        setAvailableRepos(result.data)
      }
    })

    const lastVisit = localStorage.getItem(TRACES_VISIT_KEY)
    if (lastVisit) {
      tracesApi.importSince(lastVisit).then((result) => {
        if (result.ok && result.data && result.data.count > 0) {
          setImportedSinceLastVisit(result.data.count)
        }
      }).catch(() => {})
    }
    localStorage.setItem(TRACES_VISIT_KEY, new Date().toISOString())
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setAvailableRepos])

  // Fetch stats when timeRange changes
  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  // Re-fetch traces when filters change (statusFilter, repoFilter are in fetchTraces deps)
  useEffect(() => {
    fetchTraces()
  }, [fetchTraces])

  // Generate training examples for selected trace
  const generateExamples = useCallback(async () => {
    if (!selectedTraceId) return
    setExamplesLoading(true)
    setExamplesError(null)
    setExamplesFlash(null)
    try {
      const result = await tracesApi.generateExamples(selectedTraceId, { min_success_rate: 0.3 })
      if (result.ok && result.data) {
        const examples = result.data.examples || []
        setTrainingExamples(examples)
        if (examples.length > 0) {
          setExamplesFlash(`Generated ${examples.length} example${examples.length !== 1 ? 's' : ''}`)
          setTimeout(() => setExamplesFlash(null), 3000)
        }
      } else {
        throw new Error(result.error || 'Failed to generate examples')
      }
    } catch (e) {
      setExamplesError(String(e))
      setTrainingExamples([])
    } finally {
      setExamplesLoading(false)
    }
  }, [selectedTraceId])

  // Export training examples as JSONL
  const handleExport = useCallback(() => {
    if (trainingExamples.length === 0) return
    const jsonl = trainingExamples.map(ex => JSON.stringify({
      messages: [
        { role: 'system', content: 'You are an expert software development agent...' },
        { role: 'user', content: ex.user_prompt },
        { role: 'assistant', content: ex.assistant_response }
      ]
    })).join('\n')
    const blob = new Blob([jsonl], { type: 'application/jsonl' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training_examples_${selectedTraceId}.jsonl`
    a.click()
    URL.revokeObjectURL(url)
  }, [trainingExamples, selectedTraceId])

  // Load examples when switching to examples tab
  useEffect(() => {
    if (detailTab === 'examples' && selectedTraceId) {
      generateExamples()
    }
  }, [detailTab, selectedTraceId, generateExamples])

  // Reset examples when trace changes
  useEffect(() => {
    setTrainingExamples([])
    setExamplesError(null)
    setExpandedExamples(new Set())
    setExamplesFlash(null)
  }, [selectedTraceId])

  // Fetch enriched detail when trace selection changes
  useEffect(() => {
    if (!selectedTraceId) {
      setTraceDetail(null)
      return
    }
    tracesApi.get(selectedTraceId).then((result) => {
      if (result.ok && result.data) {
        setTraceDetail(result.data)
      } else {
        setTraceDetail(null)
      }
    }).catch(() => setTraceDetail(null))
  }, [selectedTraceId])

  // Auto-classify pending traces into quality tiers
  const handleAutoClassify = useCallback(async (dryRun: boolean = true) => {
    setClassifyLoading(true)
    try {
      const result = await tracesApi.autoClassify({
        dry_run: dryRun,
        auto_promote: !dryRun
      })
      if (result.ok && result.data) {
        setClassifyResult({
          summary: result.data.summary,
          dpo_pairs_count: result.data.dpo_pairs_count
        })
        if (!dryRun) {
          // Refresh traces after classification
          fetchTraces()
          setShowClassifyModal(false)
        } else {
          setShowClassifyModal(true)
        }
      }
    } catch (e) {
      console.error('Auto-classify failed:', e)
    } finally {
      setClassifyLoading(false)
    }
  }, [fetchTraces])

  // Import traces from a specific source or all sources
  const handleImport = useCallback(async (source: string | 'all') => {
    setImportingSource(source)
    try {
      if (source === 'all') {
        const result = await tracesApi.importAll()
        if (result.ok && result.data) {
          const newResults: Record<string, { imported: number; errors: number }> = {}
          result.data.results.forEach((r) => {
            newResults[r.source] = { imported: r.imported, errors: r.errors }
          })
          setImportResults(newResults)
          setLastImportTime(Date.now())
        }
      } else {
        const result = await tracesApi.importBySource(source)
        if (result.ok && result.data) {
          setImportResults(prev => ({ ...prev, [source]: { imported: result.data!.imported, errors: result.data!.errors } }))
          setLastImportTime(Date.now())
        }
      }
      // Refresh trace list after import
      fetchTraces()
    } finally {
      setImportingSource(null)
    }
  }, [fetchTraces])

  // Sync traces
  const handleSync = useCallback(async () => {
    setImportingSource('sync')
    try {
      await tracesApi.sync()
      fetchTraces()
    } finally {
      setImportingSource(null)
    }
  }, [fetchTraces])

  // Compute import summary for status line
  const importSummary = (() => {
    const entries = Object.entries(importResults)
    if (entries.length === 0) return null
    const totalImported = entries.reduce((sum, [, r]) => sum + (r?.imported || 0), 0)
    const sourcesWithImports = entries.filter(([, r]) => r && r.imported > 0).length
    return { totalImported, sourcesWithImports }
  })()

  const colors = {
    primary: theme === 'dark' ? '#76B900' : '#0066CC',
    success: '#34C759',
    error: '#FF3B30',
    warning: '#FF9500',
    grid: theme === 'dark' ? '#2C2C2E' : '#E5E5EA',
    text: theme === 'dark' ? '#A1A1A6' : '#6E6E73'
  }

  // Quality breakdown data for selected trace (6 metrics)
  const selectedTrace = traces.find((t) => t.id === selectedTraceId) || traces[0] || emptyTracePlaceholder
  const qualityData = [
    { name: 'Success', value: selectedTrace.quality.successRate * 100, fill: colors.success },
    { name: 'Verify', value: selectedTrace.quality.verificationScore * 100, fill: colors.primary },
    { name: 'Complex', value: selectedTrace.quality.complexityScore * 100, fill: colors.warning },
    { name: 'Tools', value: selectedTrace.quality.toolDiversity * 100, fill: '#5AC8FA' },
    { name: 'Efficiency', value: selectedTrace.quality.efficiencyScore * 100, fill: '#30D158' },
    { name: 'Cognitive', value: selectedTrace.quality.cognitiveQuality * 100, fill: '#FF6B9D' },
    { name: 'Length', value: selectedTrace.quality.lengthScore * 100, fill: '#BF5AF2' }
  ]

  const displayTraces = filteredTraces()

  const getStatusIcon = (status: TraceStatus) => {
    switch (status) {
      case 'gold':
        return <Trophy className="w-4 h-4 text-yellow-500 fill-yellow-500" />
      case 'silver':
        return <Medal className="w-4 h-4 text-slate-400" />
      case 'bronze':
        return <Award className="w-4 h-4 text-amber-600" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-status-error" />
      case 'pending':
        return <Clock className="w-4 h-4 text-status-warning" />
    }
  }

  // Get tier badge styling - solid backgrounds, no opacity patterns
  const getTierBadge = (status: TraceStatus) => {
    const badges: Record<TraceStatus, { className: string; label: string }> = {
      gold: { className: 'bg-accent-light text-yellow-600 border-border', label: 'Gold - SFT' },
      silver: { className: 'bg-background-secondary text-slate-500 border-border', label: 'Silver - DPO+' },
      bronze: { className: 'bg-background-secondary text-amber-700 border-border', label: 'Bronze - DPO-' },
      failed: { className: 'bg-background-secondary text-status-error border-border', label: 'Failed' },
      pending: { className: 'bg-background-secondary text-status-warning border-border', label: 'Pending' }
    }
    return badges[status]
  }

  // Get count for a status
  const getStatusCount = (status: TraceStatus | 'all'): number => {
    switch (status) {
      case 'gold': return goldCount
      case 'silver': return silverCount
      case 'bronze': return bronzeCount
      case 'failed': return failedCount
      case 'pending': return pendingCount
      case 'all': return totalTraces || traces.length
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Import notification banner */}
      {importedSinceLastVisit > 0 && (
        <div className="border-b-2 border-border bg-accent-light px-4 py-2.5 flex items-center justify-between shrink-0">
          <span className="font-mono text-sm text-accent-dark font-semibold">
            {importedSinceLastVisit} new trace{importedSinceLastVisit !== 1 ? 's' : ''} imported since your last visit
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => { setStatusFilter('pending'); setImportedSinceLastVisit(0) }}
              className="font-mono text-xs uppercase tracking-wide border-brutal border-accent-dark/30 bg-white/50 text-accent-dark px-3 py-1 rounded-brutal hover:bg-white/80 transition-colors"
            >
              Review
            </button>
            <button
              onClick={() => setImportedSinceLastVisit(0)}
              className="font-mono text-xs uppercase tracking-wide text-accent-dark/60 hover:text-accent-dark transition-colors px-2"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Import Panel — collapsible */}
      <div className="border-b border-brutal border-border bg-background-card shrink-0">
        <button
          onClick={() => setImportPanelOpen(!importPanelOpen)}
          className="w-full flex items-center justify-between px-4 py-2 hover:bg-background-secondary transition-colors"
        >
          <span className="font-mono text-xs font-semibold uppercase tracking-wider text-text-primary flex items-center gap-2">
            <Download className="w-3.5 h-3.5" />
            Import Traces
          </span>
          {importPanelOpen ? (
            <ChevronUp className="w-4 h-4 text-text-muted" />
          ) : (
            <ChevronDown className="w-4 h-4 text-text-muted" />
          )}
        </button>

        {importPanelOpen && (
          <div className="px-4 pb-3 space-y-2.5">
            {/* Import buttons row */}
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                onClick={() => handleImport('all')}
                disabled={importingSource !== null}
                className={clsx(
                  'px-3 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal transition-press flex items-center gap-1.5',
                  'bg-accent-light text-accent-dark hover-press',
                  importingSource !== null && 'opacity-50 cursor-not-allowed'
                )}
              >
                {importingSource === 'all' ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : importResults['all'] ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Download className="w-3 h-3" />
                )}
                Import All
              </button>

              {IMPORT_SOURCES.map((src) => (
                <button
                  key={src.key}
                  onClick={() => handleImport(src.value)}
                  disabled={importingSource !== null}
                  className={clsx(
                    'px-2.5 py-1.5 text-xs font-mono font-semibold border-brutal border-border rounded-brutal transition-press flex items-center gap-1',
                    importResults[src.value]
                      ? importResults[src.value]!.errors > 0
                        ? 'bg-background-secondary text-status-error hover-press'
                        : 'bg-background-secondary text-status-success hover-press'
                      : 'bg-background-card text-text-secondary hover-press',
                    importingSource !== null && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {importingSource === src.value ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : importResults[src.value] ? (
                    importResults[src.value]!.errors > 0 ? (
                      <XCircle className="w-3 h-3" />
                    ) : (
                      <Check className="w-3 h-3" />
                    )
                  ) : null}
                  {src.label}
                </button>
              ))}

              <button
                onClick={handleSync}
                disabled={importingSource !== null}
                className={clsx(
                  'px-2.5 py-1.5 text-xs font-mono font-semibold border-brutal border-border rounded-brutal transition-press flex items-center gap-1 bg-background-card text-text-secondary hover-press',
                  importingSource !== null && 'opacity-50 cursor-not-allowed'
                )}
              >
                {importingSource === 'sync' ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <RefreshCw className="w-3 h-3" />
                )}
                Sync
              </button>
            </div>

            {/* Import status line */}
            {importSummary && lastImportTime && (
              <p className="text-[11px] font-mono text-text-muted">
                Last: {importSummary.totalImported} trace{importSummary.totalImported !== 1 ? 's' : ''} from{' '}
                {importSummary.sourcesWithImports} source{importSummary.sourcesWithImports !== 1 ? 's' : ''},{' '}
                {formatRelativeTime(lastImportTime)}
              </p>
            )}

            {/* File upload for ChatGPT / MCP imports */}
            <TraceUpload onImportComplete={() => fetchTraces()} />
          </div>
        )}
      </div>

      <div className="flex-1 flex min-h-0">
      {/* Left Panel - Trace List */}
      <div className="w-96 border-r border-brutal border-border flex flex-col bg-background-primary">
        {/* Header */}
        <div className="px-3 py-2.5 border-b border-brutal border-border space-y-2">
          <div className="flex items-center justify-between">
            <h1 className="font-brand text-xl text-text-primary">Traces</h1>
            <div className="flex items-center border-brutal border-border rounded-brutal overflow-hidden">
              <button
                onClick={() => setView('list')}
                className={clsx(
                  'p-1.5 transition-colors',
                  view === 'list'
                    ? 'bg-accent-light text-accent-dark'
                    : 'bg-background-card text-text-muted hover:text-text-primary'
                )}
                title="List view"
              >
                <List className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setView('analytics')}
                className={clsx(
                  'p-1.5 border-l border-brutal border-border transition-colors',
                  view === 'analytics'
                    ? 'bg-accent-light text-accent-dark'
                    : 'bg-background-card text-text-muted hover:text-text-primary'
                )}
                title="Analytics view"
              >
                <BarChart3 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Search + Auto-Classify on one row */}
          <div className="flex items-center gap-1.5">
            <div className="flex-1 flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search traces..."
                className="input w-full text-xs py-1 px-2"
              />
            </div>
            {pendingCount > 0 && (
              <button
                onClick={() => handleAutoClassify(true)}
                disabled={classifyLoading}
                className="btn-icon w-7 h-7 min-w-0 min-h-0 shrink-0"
                title={`Auto-classify ${pendingCount} pending`}
              >
                {classifyLoading ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Wand2 className="w-3.5 h-3.5 text-accent-dark" />
                )}
              </button>
            )}
          </div>

          {/* Filter Tabs — 3x2 grid */}
          <div className="grid grid-cols-3 gap-1">
            {(['all', 'gold', 'silver', 'bronze', 'failed', 'pending'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={clsx(
                  'px-1.5 py-1 text-[10px] font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal transition-press capitalize flex items-center justify-center gap-1',
                  statusFilter === status
                    ? 'bg-accent-light text-accent-dark border-border shadow-brutal-sm'
                    : 'bg-background-card text-text-secondary hover-press'
                )}
              >
                {status !== 'all' && getStatusIcon(status)}
                {status}
                <span className="font-bold">({getStatusCount(status)})</span>
              </button>
            ))}
          </div>

          {/* Source Filter — chip buttons */}
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted mr-0.5">Source:</span>
            <button
              onClick={() => setSelectedSource('')}
              className={clsx(
                'px-1.5 py-0.5 text-[10px] font-mono font-semibold border-brutal border-border rounded-brutal transition-press',
                !selectedSource
                  ? 'bg-accent-light text-accent-dark shadow-brutal-sm'
                  : 'bg-background-card text-text-secondary hover-press'
              )}
            >
              All
            </button>
            {IMPORT_SOURCES.map((src) => (
              <button
                key={src.key}
                onClick={() => setSelectedSource(selectedSource === src.value ? '' : src.value)}
                className={clsx(
                  'px-1.5 py-0.5 text-[10px] font-mono font-semibold border-brutal border-border rounded-brutal transition-press',
                  selectedSource === src.value
                    ? 'bg-accent-light text-accent-dark shadow-brutal-sm'
                    : 'bg-background-card text-text-secondary hover-press'
                )}
              >
                {src.label}
              </button>
            ))}
          </div>

          {/* Repo Filter — slimmer */}
          <div className="relative">
            <button
              onClick={() => setRepoDropdownOpen(!repoDropdownOpen)}
              className={clsx(
                'w-full flex items-center justify-between gap-1.5 px-2 py-1 border-brutal rounded-brutal transition-press text-xs',
                repoFilter
                  ? 'border-border bg-accent-light text-accent-dark shadow-brutal-sm'
                  : 'border-border bg-background-card text-text-secondary hover-press'
              )}
            >
              <div className="flex items-center gap-1.5 min-w-0">
                <FolderGit2 className="w-3 h-3 shrink-0" />
                <span className="font-mono text-xs truncate">{repoFilter || 'All Repos'}</span>
              </div>
              <ChevronDown className={clsx('w-3 h-3 transition-transform shrink-0', repoDropdownOpen && 'rotate-180')} />
            </button>

            {repoDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-background-card border-brutal border-border rounded-brutal shadow-brutal-sm z-10 max-h-48 overflow-y-auto">
                <button
                  onClick={() => { setRepoFilter(null); setRepoDropdownOpen(false) }}
                  className={clsx(
                    'w-full px-2 py-1.5 text-left text-xs font-mono border-b border-border-subtle transition-theme',
                    !repoFilter && 'bg-accent-light text-accent-dark'
                  )}
                >
                  All Repos
                </button>
                {availableRepos.map((repo) => (
                  <button
                    key={repo.name}
                    onClick={() => { setRepoFilter(repo.name); setRepoDropdownOpen(false) }}
                    className={clsx(
                      'w-full px-2 py-1.5 text-left text-xs font-mono border-b border-border-subtle transition-theme flex items-center justify-between',
                      repoFilter === repo.name && 'bg-accent-light text-accent-dark'
                    )}
                  >
                    <span className="truncate">{repo.name}</span>
                    {repo.trace_count && (
                      <span className="font-mono font-bold text-[10px] text-text-muted ml-1">{repo.trace_count}</span>
                    )}
                  </button>
                ))}
                {availableRepos.length === 0 && (
                  <div className="px-2 py-1.5 text-xs text-text-muted font-mono">No repos found</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Trace List */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-32">
              <RefreshCw className="w-5 h-5 animate-spin text-text-muted" />
              <span className="ml-2 text-sm font-mono text-text-muted">Loading traces...</span>
            </div>
          ) : error ? (
            <div className="p-4 text-center">
              <p className="text-sm text-status-error font-mono mb-2">{error}</p>
              <button onClick={fetchTraces} className="btn-secondary text-sm">
                <RefreshCw className="w-4 h-4" /> Retry
              </button>
            </div>
          ) : displayTraces.length === 0 ? (
            <div className="p-4 text-center text-text-muted">
              <Clock className="w-8 h-8 mx-auto mb-2" />
              <p className="text-sm font-mono">No traces found</p>
              <p className="text-xs font-mono mt-1">Start coding with Claude Code to capture traces</p>
            </div>
          ) : displayTraces.map((trace) => (
            <button
              key={trace.id}
              onClick={() => selectTrace(trace.id)}
              className={clsx(
                'w-full px-3 py-2.5 text-left border-b border-brutal border-border transition-press',
                selectedTraceId === trace.id
                  ? 'bg-accent-light'
                  : 'bg-background-card hover-press'
              )}
            >
              {/* Row 1: Repo + time */}
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  {getStatusIcon(trace.status)}
                  {trace.repo ? (
                    <span className="text-xs font-mono font-bold text-accent-dark truncate">{trace.repo.name}</span>
                  ) : (
                    <span className="text-xs font-mono text-text-muted">unknown</span>
                  )}
                </div>
                <span className="text-[10px] font-mono text-text-muted shrink-0 ml-2">
                  {formatRelativeTime(trace.createdAt)}
                </span>
              </div>
              {/* Row 2: Sanitized title — fixed 2-line height */}
              <p className="text-sm text-text-primary line-clamp-2 h-[2.5rem] leading-tight mb-1">
                {sanitizeTraceTitle(trace.taskDescription)}
              </p>
              {/* Row 3: Quality bar */}
              <div className="flex items-center gap-2">
                <div className="progress-bar flex-1 h-2">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${trace.quality.totalScore * 100}%`,
                      backgroundColor:
                        trace.quality.successRate >= 0.9
                          ? '#FFD700'
                          : trace.quality.successRate >= 0.75
                          ? '#C0C0C0'
                          : trace.quality.successRate >= 0.6
                          ? '#CD7F32'
                          : colors.error
                    }}
                  />
                </div>
                <span className="text-[10px] font-mono font-bold text-text-muted">
                  {(trace.quality.totalScore * 100).toFixed(0)}%
                </span>
              </div>
            </button>
          ))}
          {hasMore && (
            <button
              onClick={loadMore}
              disabled={loadingMore}
              className="w-full py-2 text-xs font-mono font-semibold text-accent-dark hover:bg-accent-light transition-colors border-b border-brutal border-border"
            >
              <span className="flex items-center justify-center gap-1.5">
                {loadingMore ? (
                  <RefreshCw className="w-3 h-3 animate-spin" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
                {loadingMore ? 'Loading...' : `Load more (${traces.length} of ${totalTraces})`}
              </span>
            </button>
          )}
        </div>
      </div>

      {/* Right Panel - Details or Analytics */}
      {view === 'analytics' ? (
      <div className="flex-1 overflow-y-auto bg-background-primary">
        <TraceAnalytics />
      </div>
      ) : selectedTrace.id !== 'no-traces' ? (
      <div className="flex-1 p-4 overflow-y-auto bg-background-primary">
        <div>
          {/* Header — no card wrapper, dense info */}
          <div className="mb-3 pb-3 border-b border-brutal border-border">
            {/* Row 1: Repo / branch + status + action */}
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2 min-w-0">
                {selectedTrace.repo ? (
                  <>
                    <FolderGit2 className="w-4 h-4 text-accent-dark shrink-0" />
                    <span className="font-mono text-sm font-bold text-accent-dark truncate">{selectedTrace.repo.name}</span>
                  </>
                ) : (
                  <span className="font-mono text-sm text-text-muted">No repository</span>
                )}
                {selectedTrace.repo?.git_branch && (
                  <span className="flex items-center gap-1 text-text-muted">
                    <GitBranch className="w-3 h-3 shrink-0" />
                    <span className="font-mono text-xs truncate max-w-[140px]">{selectedTrace.repo.git_branch}</span>
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="tag text-[10px]">
                  <span className="flex items-center gap-1">
                    {getStatusIcon(selectedTrace.status)}
                    {getTierBadge(selectedTrace.status).label}
                  </span>
                </span>
                {selectedTrace.status === 'gold' ? (
                  <button
                    onClick={() => demoteTrace(selectedTrace.id)}
                    className="px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal bg-background-card text-status-warning hover-press flex items-center gap-1"
                  >
                    <StarOff className="w-3 h-3" />
                    Demote
                  </button>
                ) : (
                  <button
                    onClick={() => promoteTrace(selectedTrace.id)}
                    className="px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal bg-accent-light text-accent-dark hover-press flex items-center gap-1"
                  >
                    <Star className="w-3 h-3" />
                    Promote
                  </button>
                )}
              </div>
            </div>

            {/* Row 2: Title — sanitized, fixed 2-line height */}
            <h2 className="font-brand text-lg text-text-primary leading-snug h-[2.8rem] line-clamp-2 overflow-hidden" title={selectedTrace.taskDescription}>
              {sanitizeTraceTitle(selectedTrace.taskDescription)}
            </h2>

            {/* Row 3: Metadata chips */}
            <div className="flex items-center gap-3 mt-1 text-[11px] font-mono text-text-muted">
              <span className="flex items-center gap-1">
                <Hash className="w-3 h-3" />
                {selectedTrace.id.slice(0, 12)}
              </span>
              <span>{formatRelativeTime(selectedTrace.createdAt)}</span>
              {selectedTrace.promotedAt && (
                <span>promoted {formatRelativeTime(selectedTrace.promotedAt)}</span>
              )}
              {selectedTrace.reposCount > 1 && (
                <span>{selectedTrace.reposCount} repos</span>
              )}
              <span className="font-bold text-accent-dark">
                {(selectedTrace.quality.totalScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Detail Tabs */}
          <div className="flex items-center gap-1 mb-3 border-b border-brutal border-border">
            <button
              onClick={() => setDetailTab('overview')}
              className={clsx(
                'px-3 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal rounded-brutal rounded-b-none -mb-px transition-press',
                detailTab === 'overview'
                  ? 'bg-accent-light text-accent-dark border-border border-b-transparent'
                  : 'bg-background-secondary text-text-secondary border-transparent hover-press'
              )}
            >
              <BarChart3 className="w-3.5 h-3.5 inline mr-1.5" />
              Overview
            </button>
            <button
              onClick={() => setDetailTab('examples')}
              className={clsx(
                'px-3 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal rounded-brutal rounded-b-none -mb-px transition-press',
                detailTab === 'examples'
                  ? 'bg-accent-light text-accent-dark border-border border-b-transparent'
                  : 'bg-background-secondary text-text-secondary border-transparent hover-press'
              )}
            >
              <Layers className="w-3.5 h-3.5 inline mr-1.5" />
              Examples
              {trainingExamples.length > 0 && (
                <span className="ml-1.5 tag text-[10px]"><span>{trainingExamples.length}</span></span>
              )}
            </button>
          </div>

          {/* Overview Tab Content */}
          {detailTab === 'overview' && (
            <>
              {/* Info + Scores | Tools — side by side */}
              <div className="grid grid-cols-2 gap-3 mb-3">
                {/* Info + Scores combined card */}
                <div className="card p-3">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-2 block">Info</span>
                  <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs font-mono mb-2">
                    <span className="text-text-muted">Status</span>
                    <span className="text-text-primary font-bold capitalize flex items-center gap-1">
                      {getStatusIcon(selectedTrace.status)}
                      {selectedTrace.status}
                    </span>
                    <span className="text-text-muted">Created</span>
                    <span className="text-text-primary">
                      {new Date(selectedTrace.createdAt).toLocaleDateString()} {new Date(selectedTrace.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    {selectedTrace.promotedAt && (<>
                      <span className="text-text-muted">Promoted</span>
                      <span className="text-text-primary">{new Date(selectedTrace.promotedAt).toLocaleDateString()}</span>
                    </>)}
                    {selectedTrace.repo && (<>
                      <span className="text-text-muted">Repo</span>
                      <span className="text-text-primary flex items-center gap-1 min-w-0">
                        <FolderGit2 className="w-3 h-3 shrink-0" />
                        <span className="truncate">{selectedTrace.repo.name}</span>
                      </span>
                    </>)}
                    {selectedTrace.repo?.git_branch && (<>
                      <span className="text-text-muted">Branch</span>
                      <span className="text-text-primary flex items-center gap-1 min-w-0">
                        <GitBranch className="w-3 h-3 shrink-0" />
                        <span className="truncate">{selectedTrace.repo.git_branch}</span>
                      </span>
                    </>)}
                    {selectedTrace.reposCount > 1 && (<>
                      <span className="text-text-muted">Repos</span>
                      <span className="text-text-primary font-bold">{selectedTrace.reposCount}</span>
                    </>)}
                  </div>

                  {/* Scores section */}
                  <div className="border-t border-border pt-2">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Scores</span>
                      <span className="font-mono text-xs font-bold text-accent-dark">{(selectedTrace.quality.totalScore * 100).toFixed(0)}%</span>
                    </div>
                    <div className="space-y-1">
                      {qualityData.map((d) => (
                        <div key={d.name} className="flex items-center gap-2">
                          <span className="font-mono text-[11px] text-text-muted w-16 shrink-0">{d.name}</span>
                          <div className="flex-1 h-3 bg-background-secondary rounded-sm overflow-hidden">
                            <div className="h-full rounded-sm" style={{ width: `${d.value}%`, background: d.fill }} />
                          </div>
                          <span className="font-mono text-[11px] font-bold text-text-primary w-8 text-right shrink-0">{d.value.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Tools — this trace's breakdown */}
                <div className="card p-3">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-2 block">Tools</span>
                  <div className="space-y-1 max-h-64 overflow-y-auto">
                    {(() => {
                      const breakdown = selectedTrace.toolBreakdown || {}
                      const sorted = Object.entries(breakdown).sort((a, b) => b[1] - a[1])
                      const maxCount = sorted[0]?.[1] || 1
                      const totalCalls = sorted.reduce((sum, [, c]) => sum + c, 0)

                      if (sorted.length === 0) {
                        return <p className="text-[11px] text-text-muted font-mono">No tool data</p>
                      }

                      return (
                        <>
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-mono text-[11px] text-text-muted">{sorted.length} types</span>
                            <span className="font-mono text-xs font-bold text-text-primary">{totalCalls} calls</span>
                          </div>
                          {sorted.map(([tool, count]) => (
                            <div key={tool} className="flex items-center gap-2">
                              <span className="font-mono text-[11px] text-text-muted w-14 text-right truncate">{tool}</span>
                              <div className="flex-1 h-3 bg-background-secondary rounded-sm overflow-hidden">
                                <div className="h-full rounded-sm" style={{ width: `${(count / maxCount) * 100}%`, background: 'var(--accent)' }} />
                              </div>
                              <span className="font-mono text-[11px] text-text-primary w-6 text-right">{count}</span>
                            </div>
                          ))}
                        </>
                      )
                    })()}
                  </div>
                </div>
              </div>

              {/* Session Summary + Step Outcomes + Cognitive — from enriched detail */}
              {traceDetail && (
                <div className="space-y-2 mb-3">
                  {/* Session Summary row */}
                  {traceDetail.raw_metrics && (
                    <div className="flex items-center gap-4 text-[11px] font-mono text-text-muted px-1">
                      <span>
                        <span className="text-status-success font-bold">{traceDetail.raw_metrics.successful_steps}</span>
                        {' / '}
                        <span className="text-status-error font-bold">{traceDetail.raw_metrics.failed_steps}</span>
                        {' / '}
                        <span className="font-bold text-text-primary">{traceDetail.raw_metrics.total_steps}</span>
                        {' steps'}
                      </span>
                      {traceDetail.duration_seconds != null && (
                        <span>
                          {traceDetail.duration_seconds >= 3600
                            ? `${Math.floor(traceDetail.duration_seconds / 3600)}h ${Math.floor((traceDetail.duration_seconds % 3600) / 60)}m`
                            : traceDetail.duration_seconds >= 60
                            ? `${Math.floor(traceDetail.duration_seconds / 60)}m ${Math.floor(traceDetail.duration_seconds % 60)}s`
                            : `${Math.floor(traceDetail.duration_seconds)}s`
                          }
                        </span>
                      )}
                      <span>{traceDetail.raw_metrics.unique_tools} tools</span>
                      <span>{traceDetail.raw_metrics.unique_commands} cmds</span>
                    </div>
                  )}

                  {/* Step Outcomes spark bar */}
                  {traceDetail.step_outcomes && traceDetail.step_outcomes.length > 0 && (
                    <div className="px-1">
                      <div className="flex h-3 rounded-sm overflow-hidden border-brutal border-border">
                        {traceDetail.step_outcomes.map((outcome, i) => (
                          <div
                            key={i}
                            className="h-full"
                            style={{
                              flex: 1,
                              backgroundColor: outcome === true ? '#34C759' : outcome === false ? '#FF3B30' : '#8E8E93',
                              borderRight: i < traceDetail.step_outcomes!.length - 1 ? '0.5px solid rgba(0,0,0,0.1)' : 'none',
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Cognitive indicator row */}
                  {traceDetail.cognitive_summary && (
                    <div className="flex items-center gap-4 text-[11px] font-mono text-text-muted px-1">
                      <span className="text-[#FF6B9D] font-bold">{traceDetail.cognitive_summary.thinking_steps} thinking</span>
                      <span className="text-[#5AC8FA] font-bold">{traceDetail.cognitive_summary.planning_phases} planning</span>
                      <span className="text-[#BF5AF2] font-bold">{traceDetail.cognitive_summary.reflections} reflections</span>
                      <span>{(traceDetail.cognitive_summary.cognitive_coverage * 100).toFixed(0)}% cognitive</span>
                    </div>
                  )}
                </div>
              )}

              {/* Timeline */}
              {timelineData.length > 0 && (
                <div className="card p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Timeline</span>
                    <div className="flex items-center gap-1">
                      {(['24h', '7d', '30d', 'all'] as const).map((r) => (
                        <button
                          key={r}
                          onClick={() => setTimeRange(r)}
                          className={clsx(
                            'px-2 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wide rounded-brutal border-brutal transition-press',
                            timeRange === r
                              ? 'bg-accent-light text-accent-dark border-border shadow-brutal-sm'
                              : 'bg-background-secondary text-text-muted border-transparent hover-press'
                          )}
                        >
                          {r}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="h-36">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={timelineData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
                        <XAxis dataKey="time" stroke={colors.text} fontSize={9} tickLine={false} />
                        <YAxis stroke={colors.text} fontSize={9} width={24} tickLine={false} axisLine={false} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'var(--bg-card)',
                            border: '2px solid var(--border-color)',
                            borderRadius: 'var(--radius)',
                            boxShadow: 'var(--shadow-sm)',
                            fontSize: 11
                          }}
                        />
                        <Area type="monotone" dataKey="gold" stackId="1" stroke={colors.success} fill={colors.success} fillOpacity={0.3} name="Gold" />
                        <Area type="monotone" dataKey="failed" stackId="2" stroke={colors.error} fill={colors.error} fillOpacity={0.3} name="Failed" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Training Examples Tab Content */}
          {detailTab === 'examples' && (
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-brand text-xl text-text-primary">Training Examples</h3>
                  <p className="text-sm font-mono text-text-muted">
                    Sessions segmented into individual training examples for fine-tuning
                  </p>
                  {examplesFlash && (
                    <span className="text-xs font-mono font-bold text-status-success mt-1 inline-block">{examplesFlash}</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {trainingExamples.length > 0 && (
                    <button
                      onClick={handleExport}
                      className="btn-secondary flex items-center gap-2"
                    >
                      <Download className="w-4 h-4" />
                      Export JSONL
                    </button>
                  )}
                  <button
                    onClick={generateExamples}
                    disabled={examplesLoading}
                    className="btn-secondary flex items-center gap-2"
                  >
                    {examplesLoading ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    {examplesLoading ? 'Generating...' : 'Regenerate'}
                  </button>
                </div>
              </div>

              {/* Error state */}
              {examplesError && (
                <div className="card p-4 border-status-error">
                  <p className="text-sm font-mono text-status-error">{examplesError}</p>
                </div>
              )}

              {/* Loading state */}
              {examplesLoading && (
                <div className="flex items-center justify-center h-32">
                  <RefreshCw className="w-5 h-5 animate-spin text-text-muted" />
                  <span className="ml-2 text-sm font-mono text-text-muted">Generating training examples...</span>
                </div>
              )}

              {/* Empty state */}
              {!examplesLoading && !examplesError && trainingExamples.length === 0 && (
                <div className="card p-8 text-center">
                  <FileCode className="w-8 h-8 mx-auto mb-2 text-text-muted" />
                  <p className="text-sm font-mono text-text-muted">No training examples generated yet</p>
                  <p className="text-xs font-mono text-text-muted mt-1">Click "Regenerate" to create examples from this session</p>
                </div>
              )}

              {/* Examples list */}
              {!examplesLoading && trainingExamples.length > 0 && (
                <div className="space-y-3">
                  {trainingExamples.map((example, idx) => (
                    <div key={example.example_id} className="card p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="tag text-xs">
                            <span>Example {idx + 1}</span>
                          </span>
                          <span className="text-xs font-mono font-bold text-text-muted">
                            {example.step_count} steps
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={clsx(
                            'text-xs font-mono font-bold px-2 py-0.5 border-brutal rounded-brutal',
                            example.success_rate >= 0.8 ? 'bg-background-secondary text-status-success border-border' :
                            example.success_rate >= 0.5 ? 'bg-background-secondary text-status-warning border-border' :
                            'bg-background-secondary text-status-error border-border'
                          )}>
                            {(example.success_rate * 100).toFixed(0)}% success
                          </span>
                          <span className="text-xs font-mono font-bold text-text-muted">
                            {(example.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                      </div>

                      <div className="mb-3">
                        <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Task Prompt</span>
                        <p className="text-sm text-text-primary mt-1 line-clamp-2">
                          {example.user_prompt}
                        </p>
                      </div>

                      <details className="group">
                        <summary className="cursor-pointer text-xs font-mono font-bold text-accent-dark flex items-center gap-1">
                          <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                          View response ({example.assistant_response.length} chars)
                        </summary>
                        <div className="terminal-chrome mt-2">
                          <pre className={clsx(
                            'p-3 text-xs font-mono text-text-secondary overflow-x-auto whitespace-pre-wrap',
                            expandedExamples.has(example.example_id) ? 'max-h-96 overflow-y-auto' : 'max-h-48 overflow-hidden'
                          )}>
                            {expandedExamples.has(example.example_id)
                              ? example.assistant_response
                              : example.assistant_response.slice(0, 1000)
                            }
                          </pre>
                          {!expandedExamples.has(example.example_id) && example.assistant_response.length > 1000 && (
                            <button
                              onClick={() => setExpandedExamples(prev => new Set(prev).add(example.example_id))}
                              className="w-full py-1.5 text-[10px] font-mono font-semibold text-accent-dark hover:bg-accent-light transition-colors border-t border-border"
                            >
                              Show full response ({example.assistant_response.length.toLocaleString()} chars)
                            </button>
                          )}
                        </div>
                      </details>
                    </div>
                  ))}

                </div>
              )}
            </div>
          )}

        </div>
      </div>
      ) : null}

      </div>{/* end flex-1 flex */}

      {/* Auto-Classify Modal */}
      {showClassifyModal && classifyResult && (
        <div className="fixed inset-0 bg-background-primary flex items-center justify-center z-50" style={{ backgroundColor: 'rgba(27, 32, 64, 0.85)' }}>
          <div className="card-elevated bg-background-card border-brutal border-border max-w-md w-full mx-4">
            <div className="flex items-center justify-between p-4 border-b border-brutal border-border">
              <div className="flex items-center gap-2">
                <Wand2 className="w-5 h-5 text-accent-dark" />
                <h3 className="font-brand text-xl text-text-primary">Auto-Classification Preview</h3>
              </div>
              <button
                onClick={() => setShowClassifyModal(false)}
                className="btn-icon"
              >
                <X className="w-4 h-4 text-text-muted" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              <p className="text-sm font-mono text-text-secondary">
                Based on NVIDIA NeMo thresholds, traces will be classified into:
              </p>

              <div className="grid grid-cols-2 gap-3">
                <div className="card p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Trophy className="w-4 h-4 text-yellow-500" />
                    <span className="font-mono text-xs uppercase tracking-widest text-yellow-600">Gold</span>
                  </div>
                  <p className="font-brand text-xl text-text-primary">{classifyResult.summary.gold}</p>
                  <p className="font-mono text-xs uppercase tracking-widest text-text-muted">SFT Training</p>
                </div>

                <div className="card p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Medal className="w-4 h-4 text-slate-400" />
                    <span className="font-mono text-xs uppercase tracking-widest text-slate-500">Silver</span>
                  </div>
                  <p className="font-brand text-xl text-text-primary">{classifyResult.summary.silver}</p>
                  <p className="font-mono text-xs uppercase tracking-widest text-text-muted">DPO Chosen</p>
                </div>

                <div className="card p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Award className="w-4 h-4 text-amber-600" />
                    <span className="font-mono text-xs uppercase tracking-widest text-amber-700">Bronze</span>
                  </div>
                  <p className="font-brand text-xl text-text-primary">{classifyResult.summary.bronze}</p>
                  <p className="font-mono text-xs uppercase tracking-widest text-text-muted">DPO Rejected</p>
                </div>

                <div className="card p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <XCircle className="w-4 h-4 text-status-error" />
                    <span className="font-mono text-xs uppercase tracking-widest text-status-error">Rejected</span>
                  </div>
                  <p className="font-brand text-xl text-text-primary">{classifyResult.summary.rejected}</p>
                  <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Not Suitable</p>
                </div>
              </div>

              {classifyResult.dpo_pairs_count > 0 && (
                <div className="card card-accent p-3">
                  <p className="text-sm font-mono font-bold text-accent-dark">
                    {classifyResult.dpo_pairs_count} DPO training pairs can be generated
                  </p>
                  <p className="text-xs font-mono text-text-muted mt-1">
                    Pairing Gold traces with Bronze traces for contrastive learning
                  </p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-2 p-4 border-t border-brutal border-border">
              <button
                onClick={() => setShowClassifyModal(false)}
                className="btn-ghost"
              >
                Cancel
              </button>
              <button
                onClick={() => handleAutoClassify(false)}
                disabled={classifyLoading}
                className="btn-primary flex items-center gap-2"
              >
                {classifyLoading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Wand2 className="w-4 h-4" />
                )}
                Apply Classification
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
