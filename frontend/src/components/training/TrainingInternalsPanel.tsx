import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import type { TrainingMetrics } from '../../stores/trainingStore'

interface TrainingInternalsPanelProps {
  data: TrainingMetrics[]
}

const AXIS = {
  stroke: 'var(--text-muted)',
  fontSize: 11,
  fontFamily: "'JetBrains Mono', monospace",
  tickLine: false as const
}

const TOOLTIP_STYLE = {
  backgroundColor: 'var(--bg-card)',
  border: 'var(--border-weight) solid var(--border-color)',
  borderRadius: 'var(--radius)',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: '12px'
}

function MiniChart({
  title,
  data,
  dataKey,
  stroke,
  format
}: {
  title: string
  data: Array<Record<string, number>>
  dataKey: string
  stroke: string
  format: (v: number) => string
}) {
  return (
    <div className="card p-4">
      <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">{title}</p>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis
              dataKey="step"
              {...AXIS}
              axisLine={{ stroke: 'var(--border-color)' }}
              tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)}
            />
            <YAxis {...AXIS} axisLine={{ stroke: 'var(--border-color)' }} width={54} tickFormatter={format} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelStyle={{ color: 'var(--text-secondary)' }}
              formatter={(v: number) => [format(v), title]}
              labelFormatter={(l) => `Step ${l}`}
            />
            <Line type="monotone" dataKey={dataKey} stroke={stroke} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function TrainingInternalsPanel({ data }: TrainingInternalsPanelProps) {
  const points = data
    .filter((m) => typeof m.step === 'number')
    .map((m) => ({ step: m.step, gradNorm: m.gradNorm, learningRate: m.learningRate }))

  if (points.length < 2) return null

  const fmtLR = (v: number) => {
    if (v === undefined || v === null) return '—'
    if (v !== 0 && v < 0.0001) {
      const exp = Math.floor(Math.log10(v))
      return `${(v / Math.pow(10, exp)).toFixed(1)}e${exp}`
    }
    return v.toFixed(5)
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <MiniChart
        title="Gradient norm"
        data={points}
        dataKey="gradNorm"
        stroke="var(--status-warning)"
        format={(v) => v.toFixed(2)}
      />
      <MiniChart
        title="Learning rate"
        data={points}
        dataKey="learningRate"
        stroke="var(--accent)"
        format={fmtLR}
      />
    </div>
  )
}
