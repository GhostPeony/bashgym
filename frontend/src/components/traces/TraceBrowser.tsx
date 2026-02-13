import { useState, useEffect, useCallback } from 'react'
import {
  Search,
  Filter,
  Star,
  StarOff,
  ChevronRight,
  CheckCircle,
  XCircle,
  Clock,
  BarChart3,
  RefreshCw,
  FolderGit2,
  ChevronDown,
  FileCode,
  Download,
  Layers,
  Sparkles,
  Medal,
  Award,
  Trophy,
  Wand2,
  X
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Cell
} from 'recharts'
import { useTracesStore, useThemeStore } from '../../stores'
import type { Trace, TraceStatus, TraceQualityTier } from '../../stores'
import { tracesApi, RepoInfo } from '../../services/api'
import { clsx } from 'clsx'

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
    totalScore: 0
  },
  reposCount: 0,
  createdAt: Date.now()
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
    traces, setTraces, statusFilter, setStatusFilter, searchQuery, setSearchQuery,
    filteredTraces, promoteTrace, demoteTrace,
    goldCount, silverCount, bronzeCount, failedCount, pendingCount,
    selectTrace, selectedTraceId, repoFilter, setRepoFilter, availableRepos, setAvailableRepos
  } = useTracesStore()
  const { theme } = useThemeStore()
  const [repoDropdownOpen, setRepoDropdownOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [detailTab, setDetailTab] = useState<DetailTab>('overview')
  const [trainingExamples, setTrainingExamples] = useState<TrainingExample[]>([])
  const [examplesLoading, setExamplesLoading] = useState(false)
  const [examplesError, setExamplesError] = useState<string | null>(null)
  const [timelineData, setTimelineData] = useState<{ time: string; gold: number; failed: number; pending: number }[]>([])
  const [statsLoading, setStatsLoading] = useState(false)

  // Auto-classify state
  const [classifyLoading, setClassifyLoading] = useState(false)
  const [classifyResult, setClassifyResult] = useState<{
    summary: { gold: number; silver: number; bronze: number; rejected: number; failed: number; pending: number };
    dpo_pairs_count: number;
  } | null>(null)
  const [showClassifyModal, setShowClassifyModal] = useState(false)

  // Fetch traces from API
  const fetchTraces = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const result = await tracesApi.list()
      if (result.ok && result.data) {
        // Map API response to Trace format (snake_case to camelCase)
        const mappedTraces: Trace[] = result.data.map((t) => ({
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
          createdAt: t.created_at ? new Date(t.created_at).getTime() : Date.now(),
          promotedAt: t.promoted_at ? new Date(t.promoted_at).getTime() : undefined
        }))
        setTraces(mappedTraces)
      } else {
        setError(result.error || 'Failed to fetch traces')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setIsLoading(false)
    }
  }, [setTraces])

  // Fetch trace stats for the timeline chart
  const fetchStats = useCallback(async () => {
    setStatsLoading(true)
    try {
      const result = await tracesApi.stats()
      if (result.ok && result.data) {
        setTimelineData(result.data.timeline)
      }
    } catch (e) {
      console.error('Failed to fetch trace stats:', e)
    } finally {
      setStatsLoading(false)
    }
  }, [])

  // Load traces, repos, and stats on mount
  useEffect(() => {
    fetchTraces()
    fetchStats()

    // Fetch available repos from API
    tracesApi.listRepos().then((result) => {
      if (result.ok && result.data) {
        setAvailableRepos(result.data)
      }
    })
  }, [fetchTraces, fetchStats, setAvailableRepos])

  // Generate training examples for selected trace
  const generateExamples = useCallback(async () => {
    if (!selectedTraceId) return

    setExamplesLoading(true)
    setExamplesError(null)
    try {
      const result = await tracesApi.generateExamples(selectedTraceId, { min_success_rate: 0.3 })
      if (result.ok && result.data) {
        setTrainingExamples(result.data.examples || [])
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

  // Get quality CSS class for triangular indicators
  const getQualityClass = (status: TraceStatus): string => {
    switch (status) {
      case 'gold':
      case 'silver':
      case 'bronze':
        return 'quality-gold'
      case 'failed':
        return 'quality-failed'
      case 'pending':
        return 'quality-pending'
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
      case 'all': return traces.length
    }
  }

  return (
    <div className="h-full flex">
      {/* Left Panel - Trace List */}
      <div className="w-96 border-r border-brutal border-border flex flex-col bg-background-primary">
        {/* Header */}
        <div className="p-4 border-b border-brutal border-border">
          <div className="flex items-center gap-3 mb-3">
            <h1 className="font-brand text-2xl text-text-primary">Traces</h1>
            <span className="tag"><span>Browser</span></span>
          </div>

          {/* Search Input */}
          <div className="flex items-center gap-2 mb-3">
            <div className="flex-1 flex items-center gap-2">
              <Search className="w-4 h-4 text-text-muted flex-shrink-0" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search traces..."
                className="input w-full"
              />
            </div>
          </div>

          {/* Filter Tabs - Tiered Quality Classification */}
          <div className="flex flex-wrap items-center gap-1 mb-3">
            {(['all', 'gold', 'silver', 'bronze', 'failed', 'pending'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={clsx(
                  'px-2.5 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal transition-press capitalize flex items-center gap-1',
                  statusFilter === status
                    ? 'bg-accent-light text-accent-dark border-border shadow-brutal-sm'
                    : 'bg-background-card text-text-secondary hover-press'
                )}
              >
                {status !== 'all' && getStatusIcon(status)}
                {status}
                <span className="font-mono font-bold">
                  ({getStatusCount(status)})
                </span>
              </button>
            ))}
          </div>

          {/* Auto-Classify Button */}
          {pendingCount > 0 && (
            <button
              onClick={() => handleAutoClassify(true)}
              disabled={classifyLoading}
              className="btn-primary w-full flex items-center justify-center gap-2 mb-3"
            >
              {classifyLoading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Wand2 className="w-4 h-4" />
              )}
              Auto-Classify {pendingCount} Pending
            </button>
          )}

          {/* Repo Filter */}
          <div className="relative">
            <button
              onClick={() => setRepoDropdownOpen(!repoDropdownOpen)}
              className={clsx(
                'w-full flex items-center justify-between gap-2 px-3 py-2 border-brutal rounded-brutal transition-press text-sm',
                repoFilter
                  ? 'border-border bg-accent-light text-accent-dark shadow-brutal-sm'
                  : 'border-border bg-background-card text-text-secondary hover-press'
              )}
            >
              <div className="flex items-center gap-2">
                <FolderGit2 className="w-4 h-4" />
                <span className="font-mono text-sm">{repoFilter || 'All Repositories'}</span>
              </div>
              <ChevronDown className={clsx('w-4 h-4 transition-transform', repoDropdownOpen && 'rotate-180')} />
            </button>

            {repoDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-background-card border-brutal border-border rounded-brutal shadow-brutal-sm z-10 max-h-48 overflow-y-auto">
                <button
                  onClick={() => { setRepoFilter(null); setRepoDropdownOpen(false) }}
                  className={clsx(
                    'w-full px-3 py-2 text-left text-sm font-mono border-b border-border-subtle transition-theme',
                    !repoFilter && 'bg-accent-light text-accent-dark'
                  )}
                >
                  All Repositories
                </button>
                {availableRepos.map((repo) => (
                  <button
                    key={repo.name}
                    onClick={() => { setRepoFilter(repo.name); setRepoDropdownOpen(false) }}
                    className={clsx(
                      'w-full px-3 py-2 text-left text-sm font-mono border-b border-border-subtle transition-theme flex items-center justify-between',
                      repoFilter === repo.name && 'bg-accent-light text-accent-dark'
                    )}
                  >
                    <span>{repo.name}</span>
                    {repo.trace_count && (
                      <span className="font-mono font-bold text-xs text-text-muted">{repo.trace_count}</span>
                    )}
                  </button>
                ))}
                {availableRepos.length === 0 && (
                  <div className="px-3 py-2 text-sm text-text-muted font-mono">No repositories found</div>
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
                'w-full p-4 text-left border-b border-brutal border-border transition-press',
                selectedTraceId === trace.id
                  ? 'bg-accent-light'
                  : 'bg-background-card hover-press'
              )}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getStatusIcon(trace.status)}
                  <span className="text-xs font-mono font-bold text-text-muted">{trace.id}</span>
                </div>
                <span className="text-xs font-mono text-text-muted">
                  {new Date(trace.createdAt).toLocaleTimeString()}
                </span>
              </div>
              <p className="text-sm text-text-primary line-clamp-2 mb-2">
                {trace.taskDescription}
              </p>
              <div className="flex items-center gap-2 mb-2">
                {trace.repo && (
                  <div className="flex items-center gap-1">
                    <FolderGit2 className="w-3 h-3 text-text-muted" />
                    <span className="text-xs font-mono text-text-muted">{trace.repo.name}</span>
                  </div>
                )}
                {/* Tier badge */}
                {(() => {
                  const badge = getTierBadge(trace.status)
                  return (
                    <span className="tag text-xs">
                      <span>{badge.label}</span>
                    </span>
                  )
                })()}
              </div>
              <div className="flex items-center gap-2">
                {/* Progress bar - brutalist style with triangular end cap */}
                <div className="progress-bar flex-1">
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
                <span className="text-xs font-mono font-bold text-text-muted">
                  {(trace.quality.successRate * 100).toFixed(0)}%
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Right Panel - Details */}
      {selectedTrace.id !== 'no-traces' && (
      <div className="flex-1 p-6 overflow-y-auto bg-background-primary">
        <div className="max-w-4xl">
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className={getQualityClass(selectedTrace.status)} />
                {getStatusIcon(selectedTrace.status)}
                <span className="text-sm font-mono font-bold text-text-muted">{selectedTrace.id}</span>
              </div>
              <h2 className="font-brand text-2xl text-text-primary">
                {selectedTrace.taskDescription}
              </h2>
            </div>

            <div className="flex items-center gap-2">
              {selectedTrace.status === 'gold' ? (
                <button
                  onClick={() => demoteTrace(selectedTrace.id)}
                  className="btn-secondary flex items-center gap-2 text-status-warning"
                >
                  <StarOff className="w-4 h-4" />
                  Demote
                </button>
              ) : (
                <button
                  onClick={() => promoteTrace(selectedTrace.id)}
                  className="btn-primary flex items-center gap-2"
                >
                  <Star className="w-4 h-4" />
                  Promote to Gold
                </button>
              )}
            </div>
          </div>

          {/* Section Divider */}
          <div className="section-divider mb-4" />

          {/* Detail Tabs */}
          <div className="flex items-center gap-1 mb-4 border-b border-brutal border-border">
            <button
              onClick={() => setDetailTab('overview')}
              className={clsx(
                'px-4 py-2 text-sm font-mono font-semibold uppercase tracking-wide border-brutal rounded-brutal rounded-b-none -mb-px transition-press',
                detailTab === 'overview'
                  ? 'bg-accent-light text-accent-dark border-border border-b-transparent'
                  : 'bg-background-secondary text-text-secondary border-transparent hover-press'
              )}
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Overview
            </button>
            <button
              onClick={() => setDetailTab('examples')}
              className={clsx(
                'px-4 py-2 text-sm font-mono font-semibold uppercase tracking-wide border-brutal rounded-brutal rounded-b-none -mb-px transition-press',
                detailTab === 'examples'
                  ? 'bg-accent-light text-accent-dark border-border border-b-transparent'
                  : 'bg-background-secondary text-text-secondary border-transparent hover-press'
              )}
            >
              <Layers className="w-4 h-4 inline mr-2" />
              Examples
              {trainingExamples.length > 0 && (
                <span className="ml-2 tag text-xs"><span>{trainingExamples.length}</span></span>
              )}
            </button>
          </div>

          {/* Overview Tab Content */}
          {detailTab === 'overview' && (
            <>
              {/* Quality Score Breakdown */}
              <div className="card card-accent p-4 mb-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-brand text-xl text-text-primary">Quality Breakdown</h3>
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-text-muted" />
                    <span className="font-brand text-xl text-accent-dark">
                      {(selectedTrace.quality.totalScore * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={qualityData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} stroke={colors.text} fontSize={11} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        stroke={colors.text}
                        fontSize={11}
                        width={70}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--bg-card)',
                          border: '2px solid var(--border-color)',
                          borderRadius: 'var(--radius)',
                          boxShadow: 'var(--shadow-sm)'
                        }}
                        formatter={(value: number) => [`${value.toFixed(0)}%`, 'Score']}
                      />
                      <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={20}>
                        {qualityData.map((entry, index) => (
                          <Cell key={index} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Trace Timeline */}
              <div className="card p-4 mb-4">
                <h3 className="font-brand text-xl text-text-primary mb-4">Traces Over Time</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
                      <XAxis dataKey="time" stroke={colors.text} fontSize={11} />
                      <YAxis stroke={colors.text} fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--bg-card)',
                          border: '2px solid var(--border-color)',
                          borderRadius: 'var(--radius)',
                          boxShadow: 'var(--shadow-sm)'
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="gold"
                        stackId="1"
                        stroke={colors.success}
                        fill={colors.success}
                        fillOpacity={0.3}
                        name="Gold"
                      />
                      <Area
                        type="monotone"
                        dataKey="failed"
                        stackId="2"
                        stroke={colors.error}
                        fill={colors.error}
                        fillOpacity={0.3}
                        name="Failed"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
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
                </div>
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
                          <div className="terminal-header">
                            <div className="terminal-dot terminal-dot-red" />
                            <div className="terminal-dot terminal-dot-yellow" />
                            <div className="terminal-dot terminal-dot-green" />
                          </div>
                          <pre className="p-3 text-xs font-mono text-text-secondary overflow-x-auto max-h-48">
                            {example.assistant_response.slice(0, 1000)}
                            {example.assistant_response.length > 1000 && '...'}
                          </pre>
                        </div>
                      </details>
                    </div>
                  ))}

                  {/* Export button */}
                  <div className="section-divider" />
                  <div className="pt-4">
                    <button
                      onClick={() => {
                        // Download examples as JSONL
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
                        a.download = `training_examples_${selectedTrace.id}.jsonl`
                        a.click()
                        URL.revokeObjectURL(url)
                      }}
                      className="btn-cta"
                    >
                      <Download className="w-4 h-4 mr-2 inline" />
                      Export as JSONL
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Metadata (always show) */}
          <div className="card p-4 mt-4">
            <h3 className="font-brand text-xl text-text-primary mb-4">Metadata</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Task ID</span>
                <p className="font-mono font-bold text-text-primary">{selectedTrace.taskId}</p>
              </div>
              <div>
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Created</span>
                <p className="text-text-primary">
                  {new Date(selectedTrace.createdAt).toLocaleString()}
                </p>
              </div>
              {selectedTrace.promotedAt && (
                <div>
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Promoted</span>
                  <p className="text-text-primary">
                    {new Date(selectedTrace.promotedAt).toLocaleString()}
                  </p>
                </div>
              )}
              <div>
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Status</span>
                <p className={clsx('capitalize flex items-center gap-1', getQualityClass(selectedTrace.status))}>
                  {selectedTrace.status}
                </p>
              </div>
              {selectedTrace.repo && (
                <>
                  <div>
                    <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Repository</span>
                    <p className="text-text-primary flex items-center gap-1">
                      <FolderGit2 className="w-3 h-3" />
                      {selectedTrace.repo.name}
                    </p>
                  </div>
                  {selectedTrace.repo.git_branch && (
                    <div>
                      <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Branch</span>
                      <p className="text-text-primary font-mono text-xs">{selectedTrace.repo.git_branch}</p>
                    </div>
                  )}
                </>
              )}
              {selectedTrace.reposCount > 1 && (
                <div>
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Repos Touched</span>
                  <p className="font-brand text-xl text-text-primary">{selectedTrace.reposCount}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      )}

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
