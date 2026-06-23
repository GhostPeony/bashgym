import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { BrainCircuit } from 'lucide-react'
import { useTrainingStore } from '../../stores'
import type { GrpoMetric } from '../../stores'

const AXIS_STYLE = {
  stroke: 'var(--text-muted)',
  fontSize: 10,
  fontFamily: "'JetBrains Mono', monospace",
  tickLine: false,
} as const

const TOOLTIP_STYLE = {
  backgroundColor: 'var(--bg-card)',
  border: 'var(--border-weight) solid var(--border-color)',
  borderRadius: 'var(--radius)',
  boxShadow: 'var(--shadow-sm)',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: '11px',
} as const

function metric(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '-'
  if (Math.abs(v) <= 1) return v.toFixed(3)
  return v.toFixed(2)
}

function percent(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '-'
  return `${(v * 100).toFixed(1)}%`
}

function hasWorldModelMetric(point: GrpoMetric): boolean {
  return (
    point.echoLoss !== undefined ||
    point.rwmlPassRate !== undefined ||
    point.embeddingDistanceMean !== undefined ||
    point.exitCodeAccuracy !== undefined ||
    point.testResultAccuracy !== undefined
  )
}

function StatTile({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
      <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
        {label}
      </p>
      <p className="font-brand text-xl text-text-primary mt-1">{value}</p>
      <p className="font-mono text-[10px] text-text-muted mt-1">{hint}</p>
    </div>
  )
}

export function WorldModelMetricsPanel() {
  const grpoMetrics = useTrainingStore((s) => s.grpoMetrics)
  const series = grpoMetrics.filter(hasWorldModelMetric)

  if (series.length === 0) return null

  const latest = series[series.length - 1]
  const hasPredictionAccuracy =
    latest.exitCodeAccuracy !== undefined || latest.testResultAccuracy !== undefined

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-3 gap-3">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-5 h-5 text-accent" />
          <div>
            <h3 className="font-brand text-lg text-text-primary">World-Model Quality</h3>
            <p className="font-mono text-xs text-text-muted">
              ECHO/RWML diagnostics parsed from trainer logs
            </p>
          </div>
        </div>
        <div className="font-mono text-xs text-text-secondary">
          step <span className="text-text-primary">{latest.step}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-3">
        <StatTile label="ECHO loss" value={metric(latest.echoLoss)} hint="lower is better" />
        <StatTile label="RWML pass" value={percent(latest.rwmlPassRate)} hint="one-step match" />
        <StatTile
          label="Distance"
          value={metric(latest.embeddingDistanceMean)}
          hint="mean embedding gap"
        />
        <StatTile
          label="Exit code"
          value={percent(latest.exitCodeAccuracy)}
          hint="prediction accuracy"
        />
        <StatTile
          label="Test result"
          value={percent(latest.testResultAccuracy)}
          hint="prediction accuracy"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="h-44 border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-2">
            ECHO loss
          </p>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => metric(v)} />
              <Line
                type="monotone"
                dataKey="echoLoss"
                stroke="var(--accent)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="h-44 border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-2">
            RWML quality
          </p>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => percent(v)} />
              <Line
                type="monotone"
                dataKey="rwmlPassRate"
                stroke="var(--status-success)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              {hasPredictionAccuracy ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="exitCodeAccuracy"
                    stroke="var(--status-warning)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="testResultAccuracy"
                    stroke="var(--status-info)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </>
              ) : null}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
