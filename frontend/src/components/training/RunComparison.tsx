import { useMemo, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import { GitCompareArrows, RefreshCw } from 'lucide-react'
import { metricRunsResource, runMetricsResource } from '../../stores/opsResources'
import { useSessionResource } from '../../stores/sessionResource'
import { clsx } from 'clsx'

// Distinct line colors per selected run (accent first, then stable contrasts)
const RUN_COLORS = ['#76B900', '#3B8EEA', '#CD3131', '#B58900', '#D33682', '#2AA198']
const MAX_SELECTED = 6

export function RunComparison() {
  const {
    data: runsData,
    loading,
    refreshing,
    error: runsError,
    refresh
  } = useSessionResource(metricRunsResource)
  const runs = runsData ?? []
  const [selected, setSelected] = useState<string[]>([])
  const metricEntries = runMetricsResource((s) => s.entries)
  const isFetching = loading || refreshing

  const metricsError = selected.map((id) => metricEntries[id]?.error).find(Boolean) ?? null
  const error = runsError || metricsError

  const toggleRun = (runId: string) => {
    setSelected((prev) => {
      if (prev.includes(runId)) return prev.filter((r) => r !== runId)
      if (prev.length >= MAX_SELECTED) return prev
      return [...prev, runId]
    })
    void runMetricsResource.getState().ensureLoaded(runId)
  }

  // Merge selected runs into one recharts dataset keyed by step
  const data = useMemo(() => {
    const byStep = new Map<number, Record<string, number>>()
    for (const runId of selected) {
      for (const point of metricEntries[runId]?.data ?? []) {
        const row = byStep.get(point.step) ?? { step: point.step }
        row[runId] = point.loss
        byStep.set(point.step, row)
      }
    }
    return Array.from(byStep.values()).sort((a, b) => a.step - b.step)
  }, [selected, metricEntries])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="flex items-center gap-2 text-sm font-mono font-semibold text-text-primary">
          <GitCompareArrows className="w-4 h-4 text-accent" />
          Compare Runs
        </h3>
        <button
          onClick={() => void refresh()}
          className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
          title="Refresh run list"
        >
          <RefreshCw className={clsx('w-3.5 h-3.5', isFetching && 'animate-spin')} />
        </button>
      </div>

      {error && <div className="mb-2 text-xs font-mono text-status-error">{error}</div>}

      <div className="flex flex-wrap gap-1 mb-3">
        {runs.length === 0 && !loading && (
          <span className="text-xs text-text-muted">
            No persisted runs yet — metrics are recorded automatically for new training runs.
          </span>
        )}
        {runs.map((run) => (
          <button
            key={run.run_id}
            onClick={() => toggleRun(run.run_id)}
            className={clsx(
              'tag !text-[11px] !py-0.5 !px-2 font-mono transition-press',
              selected.includes(run.run_id) ? '!border-accent !text-accent' : 'text-text-muted'
            )}
            title={new Date(run.modified * 1000).toLocaleString()}
          >
            {run.run_id}
          </button>
        ))}
      </div>

      {selected.length > 0 && data.length > 0 && (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis
                dataKey="step"
                stroke="var(--text-muted)"
                fontSize={11}
                fontFamily="'JetBrains Mono', monospace"
                tickLine={false}
                axisLine={{ stroke: 'var(--border-color)' }}
              />
              <YAxis
                stroke="var(--text-muted)"
                fontSize={11}
                fontFamily="'JetBrains Mono', monospace"
                tickLine={false}
                axisLine={{ stroke: 'var(--border-color)' }}
                domain={['auto', 'auto']}
                tickFormatter={(value: number) => value.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-card)',
                  border: 'var(--border-weight) solid var(--border-color)',
                  borderRadius: 'var(--radius)',
                  boxShadow: 'var(--shadow-sm)',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '12px'
                }}
                labelStyle={{ color: 'var(--text-secondary)', marginBottom: '4px' }}
                labelFormatter={(label) => `Step ${label}`}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', fontFamily: "'JetBrains Mono', monospace" }}
              />
              {selected.map((runId, i) => (
                <Line
                  key={runId}
                  type="monotone"
                  dataKey={runId}
                  stroke={RUN_COLORS[i % RUN_COLORS.length]}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
