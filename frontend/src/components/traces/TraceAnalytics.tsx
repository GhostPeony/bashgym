import { useState, useEffect } from 'react'
import { RefreshCw, TrendingUp, Database, DollarSign, Zap } from 'lucide-react'
import { tracesApi, TraceAnalytics as TraceAnalyticsData } from '../../services/api'
import { clsx } from 'clsx'

/** Format source_tool slug into human-friendly name */
function formatSourceName(source: string): string {
  const map: Record<string, string> = {
    claude_code: 'Claude Code',
    gemini_cli: 'Gemini CLI',
    copilot: 'GitHub Copilot',
    opencode: 'OpenCode',
    codex: 'Codex CLI',
    aider: 'Aider',
    cursor: 'Cursor',
    unknown: 'Unknown',
  }
  return map[source] || source.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

/** Quality tier color mapping */
const QUALITY_COLORS: Record<string, { bg: string; label: string }> = {
  gold: { bg: 'bg-amber-500', label: 'Gold' },
  silver: { bg: 'bg-gray-400', label: 'Silver' },
  bronze: { bg: 'bg-orange-600', label: 'Bronze' },
  rejected: { bg: 'bg-red-500', label: 'Rejected' },
  failed: { bg: 'bg-red-500', label: 'Failed' },
  pending: { bg: 'bg-gray-300', label: 'Pending' },
}

export function TraceAnalytics() {
  const [data, setData] = useState<TraceAnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAnalytics = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await tracesApi.getAnalytics()
      if (result.ok && result.data) {
        setData(result.data)
      } else {
        setError(result.error || 'Failed to fetch analytics')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalytics()
  }, [])

  // Loading state
  if (loading) {
    return (
      <div className="p-6 space-y-4">
        <div className="flex items-center gap-2 mb-6">
          <RefreshCw className="w-5 h-5 animate-spin text-text-muted" />
          <span className="font-mono text-sm text-text-muted">Loading analytics...</span>
        </div>
        {/* Skeleton cards */}
        <div className="grid grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="card p-4 animate-pulse">
              <div className="h-3 w-24 bg-[var(--bg-secondary)] rounded mb-3" />
              <div className="h-8 w-16 bg-[var(--bg-secondary)] rounded" />
            </div>
          ))}
        </div>
        <div className="card p-4 animate-pulse h-24" />
        <div className="card p-4 animate-pulse h-32" />
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="p-6">
        <div className="card p-8 text-center">
          <p className="text-sm text-[var(--status-error)] font-mono mb-3">{error}</p>
          <button onClick={fetchAnalytics} className="btn-secondary text-sm">
            <RefreshCw className="w-4 h-4" /> Retry
          </button>
        </div>
      </div>
    )
  }

  // Empty state
  if (!data || data.totals.sessions === 0) {
    return (
      <div className="p-6">
        <div className="card p-8 text-center">
          <Database className="w-8 h-8 mx-auto mb-2 text-[var(--text-muted)]" />
          <p className="text-sm font-mono text-[var(--text-muted)]">No trace data available</p>
          <p className="text-xs font-mono text-[var(--text-muted)] mt-1">Import traces to see analytics</p>
        </div>
      </div>
    )
  }

  // Compute derived values
  const qualityEntries = Object.entries(data.quality_distribution)
  const qualityTotal = qualityEntries.reduce((sum, [, count]) => sum + count, 0)

  const sortedToolStats = [...data.tool_stats]
    .sort((a, b) => b.calls - a.calls)
    .slice(0, 10)

  const maxSourceTraces = Math.max(...data.source_breakdown.map(s => s.traces), 1)

  const SFT_THRESHOLD = 30

  return (
    <div className="p-6 space-y-6 overflow-y-auto">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-brand text-2xl text-[var(--text-primary)]">Trace Analytics</h2>
          <p className="text-xs font-mono text-[var(--text-secondary)] mt-0.5">
            Avg quality: {(data.avg_quality_score * 100).toFixed(1)}%
          </p>
        </div>
        <button
          onClick={fetchAnalytics}
          className="btn-icon w-8 h-8"
          title="Refresh analytics"
        >
          <RefreshCw className="w-4 h-4 text-[var(--text-secondary)]" />
        </button>
      </div>

      {/* Section 1: Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <SummaryCard
          label="Total Traces"
          value={data.totals.sessions.toLocaleString()}
          icon={<Database className="w-4 h-4" />}
        />
        <SummaryCard
          label="Gold Ready"
          value={data.training_readiness.sft_ready.toLocaleString()}
          icon={<TrendingUp className="w-4 h-4" />}
          accent={data.training_readiness.sft_ready >= SFT_THRESHOLD}
        />
        <SummaryCard
          label="Total Steps"
          value={data.totals.steps.toLocaleString()}
          icon={<Zap className="w-4 h-4" />}
        />
        <SummaryCard
          label="Est. Cost"
          value={`$${data.cost_total_usd.toFixed(2)}`}
          icon={<DollarSign className="w-4 h-4" />}
        />
      </div>

      {/* Section 2: Quality Distribution */}
      <div className="card p-4">
        <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-muted)] mb-3 block">
          Quality Distribution
        </span>

        {/* Stacked bar */}
        <div className="flex h-8 border-2 border-[var(--text-primary)] rounded-sm overflow-hidden">
          {qualityEntries.map(([tier, count]) => {
            const pct = qualityTotal > 0 ? (count / qualityTotal) * 100 : 0
            if (pct === 0) return null
            const color = QUALITY_COLORS[tier] || QUALITY_COLORS.pending
            return (
              <div
                key={tier}
                className={clsx(color.bg, 'flex items-center justify-center text-xs font-mono font-bold text-white relative')}
                style={{ width: `${pct}%` }}
                title={`${color.label}: ${count} (${pct.toFixed(1)}%)`}
              >
                {pct >= 8 && (
                  <span className="truncate px-1">{pct.toFixed(0)}%</span>
                )}
              </div>
            )
          })}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 mt-3">
          {qualityEntries.map(([tier, count]) => {
            const color = QUALITY_COLORS[tier] || QUALITY_COLORS.pending
            return (
              <div key={tier} className="flex items-center gap-2">
                <div className={clsx('w-3 h-3 border border-[var(--text-primary)]', color.bg)} />
                <span className="font-mono text-xs text-[var(--text-secondary)]">
                  {color.label}: <span className="font-bold text-[var(--text-primary)]">{count.toLocaleString()}</span>
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Section 3: Source Breakdown */}
      {data.source_breakdown.length > 0 && (
        <div className="card p-4">
          <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-muted)] mb-3 block">
            Source Breakdown
          </span>
          <div className="space-y-2.5">
            {data.source_breakdown.map(source => {
              const barWidth = (source.traces / maxSourceTraces) * 100
              return (
                <div key={source.source} className="flex items-center gap-3">
                  <span className="font-mono text-xs text-[var(--text-primary)] w-28 shrink-0 truncate font-semibold">
                    {formatSourceName(source.source)}
                  </span>
                  <div className="flex-1 h-5 bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-sm overflow-hidden">
                    <div
                      className="h-full bg-[var(--accent)] transition-all duration-300"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs font-bold text-[var(--text-primary)] w-12 text-right shrink-0">
                    {source.traces.toLocaleString()}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Section 4: Training Readiness */}
      <div className="card p-4">
        <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-muted)] mb-3 block">
          Training Readiness
        </span>
        <div className="space-y-3">
          <ReadinessBar
            label="SFT Ready"
            value={data.training_readiness.sft_ready}
            max={Math.max(data.totals.sessions, SFT_THRESHOLD)}
            threshold={SFT_THRESHOLD}
            suffix="traces"
          />
          <ReadinessBar
            label="DPO Pairs"
            value={data.training_readiness.dpo_pairs_possible}
            max={Math.max(data.totals.sessions, SFT_THRESHOLD)}
            threshold={10}
            suffix="possible"
          />
          <ReadinessBar
            label="Trainable"
            value={data.training_readiness.total_trainable}
            max={Math.max(data.totals.sessions, SFT_THRESHOLD)}
            threshold={SFT_THRESHOLD}
            suffix="total"
          />
        </div>
      </div>

      {/* Section 5: Tool Usage Table */}
      {sortedToolStats.length > 0 && (
        <div className="card overflow-hidden">
          <div className="px-4 py-3 border-b-2 border-[var(--text-primary)]">
            <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-muted)]">
              Tool Usage (Top 10)
            </span>
          </div>
          <table className="w-full">
            <thead>
              <tr className="bg-[var(--accent-light)]">
                <th className="text-left px-4 py-2 font-mono text-xs uppercase tracking-wider text-[var(--accent-dark)]">
                  Tool
                </th>
                <th className="text-right px-4 py-2 font-mono text-xs uppercase tracking-wider text-[var(--accent-dark)]">
                  Calls
                </th>
                <th className="text-right px-4 py-2 font-mono text-xs uppercase tracking-wider text-[var(--accent-dark)]">
                  Sessions
                </th>
                <th className="text-right px-4 py-2 font-mono text-xs uppercase tracking-wider text-[var(--accent-dark)]">
                  Success Rate
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedToolStats.map((tool, idx) => (
                <tr
                  key={tool.tool}
                  className={clsx(
                    'border-b border-[var(--border-subtle)] transition-colors',
                    idx % 2 === 0 ? 'bg-[var(--bg-card)]' : 'bg-[var(--bg-primary)]'
                  )}
                >
                  <td className="px-4 py-2 font-mono text-xs font-semibold text-[var(--text-primary)]">
                    {tool.tool}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-right text-[var(--text-primary)]">
                    {tool.calls.toLocaleString()}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-right text-[var(--text-secondary)]">
                    {tool.sessions.toLocaleString()}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-right">
                    <span className={clsx(
                      'font-bold',
                      tool.success_rate >= 0.8 ? 'text-[var(--status-success)]' :
                      tool.success_rate >= 0.5 ? 'text-[var(--status-warning)]' :
                      'text-[var(--status-error)]'
                    )}>
                      {(tool.success_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

/** Summary card with brutalist styling */
function SummaryCard({ label, value, icon, accent }: {
  label: string
  value: string
  icon: React.ReactNode
  accent?: boolean
}) {
  return (
    <div className={clsx(
      'card p-4 hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-all duration-150',
      accent && 'border-[var(--status-success)]'
    )}>
      <div className="flex items-center gap-2 mb-2">
        <span className={clsx(
          'text-[var(--text-muted)]',
          accent && 'text-[var(--status-success)]'
        )}>
          {icon}
        </span>
        <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </span>
      </div>
      <span className={clsx(
        'font-brand text-2xl',
        accent ? 'text-[var(--status-success)]' : 'text-[var(--text-primary)]'
      )}>
        {value}
      </span>
    </div>
  )
}

/** Progress bar with threshold indicator */
function ReadinessBar({ label, value, max, threshold, suffix }: {
  label: string
  value: number
  max: number
  threshold: number
  suffix: string
}) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  const thresholdPct = max > 0 ? Math.min((threshold / max) * 100, 100) : 0
  const meetsThreshold = value >= threshold

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-xs uppercase tracking-wider text-[var(--text-secondary)]">
          {label}
        </span>
        <span className={clsx(
          'font-mono text-xs font-bold',
          meetsThreshold ? 'text-[var(--status-success)]' : 'text-[var(--text-primary)]'
        )}>
          {value.toLocaleString()} {suffix}
        </span>
      </div>
      <div className="relative h-4 bg-[var(--bg-secondary)] border-2 border-[var(--text-primary)] rounded-sm overflow-hidden">
        <div
          className={clsx(
            'h-full transition-all duration-500',
            meetsThreshold ? 'bg-[var(--status-success)]' : 'bg-[var(--accent)]'
          )}
          style={{ width: `${pct}%` }}
        />
        {/* Threshold marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-[var(--text-primary)] opacity-50"
          style={{ left: `${thresholdPct}%` }}
          title={`Threshold: ${threshold}`}
        />
      </div>
    </div>
  )
}
