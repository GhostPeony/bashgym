import {
  LineChart,
  Line,
  Area,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { useTrainingStore } from '../../stores'
import type { GrpoMetric } from '../../stores'

interface ChartShellProps {
  title: string
  subtitle?: string
  children: React.ReactNode
}

function ChartShell({ title, subtitle, children }: ChartShellProps) {
  return (
    <div className="card p-3 h-48 flex flex-col">
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="font-mono text-xs uppercase tracking-widest text-text-secondary">
          {title}
        </h4>
        {subtitle && (
          <span className="font-mono text-xs text-text-muted">{subtitle}</span>
        )}
      </div>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  )
}

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

function formatNum(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '—'
  if (Math.abs(v) >= 100) return v.toFixed(1)
  return v.toFixed(4)
}

function formatCount(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '—'
  return Math.round(v).toString()
}

export function GrpoMetricsPanel() {
  const grpoMetrics = useTrainingStore((s) => s.grpoMetrics)

  if (grpoMetrics.length === 0) return null

  // Pre-compute derived series for reward band chart.
  const rewardSeries = grpoMetrics.map((m) => ({
    step: m.step,
    reward: m.reward,
    rewardHigh:
      m.reward !== undefined && m.rewardStd !== undefined
        ? m.reward + m.rewardStd
        : undefined,
    rewardLow:
      m.reward !== undefined && m.rewardStd !== undefined
        ? m.reward - m.rewardStd
        : undefined,
  }))

  const latest: GrpoMetric | undefined = grpoMetrics[grpoMetrics.length - 1]
  const hasSamplingTelemetry =
    latest?.activeSamplingRefills !== undefined ||
    latest?.zeroStdGroupsDropped !== undefined ||
    latest?.effectivePromptGroups !== undefined

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <h3 className="font-brand text-lg text-text-primary">GRPO Metrics</h3>
          <p className="font-mono text-xs text-text-muted">
            Live per-step stats parsed from training logs
          </p>
        </div>
        {latest && (
          <div className="font-mono text-xs text-text-secondary flex gap-4">
            <span>step <span className="text-text-primary">{latest.step}</span></span>
            <span>kl <span className="text-text-primary">{formatNum(latest.kl)}</span></span>
            <span>reward <span className="text-text-primary">{formatNum(latest.reward)}</span></span>
            {hasSamplingTelemetry ? (
              <>
                <span>refills <span className="text-text-primary">{formatCount(latest.activeSamplingRefills)}</span></span>
                <span>dropped <span className="text-text-primary">{formatCount(latest.zeroStdGroupsDropped)}</span></span>
                <span>groups <span className="text-text-primary">{formatCount(latest.effectivePromptGroups)}</span></span>
              </>
            ) : null}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Loss */}
        <ChartShell title="Loss" subtitle={formatNum(latest?.loss)}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={grpoMetrics} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(4)} />
              <Line type="monotone" dataKey="loss" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartShell>

        {/* Reward with std band */}
        <ChartShell title="Reward ± Std" subtitle={formatNum(latest?.reward)}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rewardSeries} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—')} />
              <Area
                type="monotone"
                dataKey="rewardHigh"
                stroke="none"
                fill="var(--accent)"
                fillOpacity={0.18}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="rewardLow"
                stroke="none"
                fill="var(--bg-card)"
                fillOpacity={1}
                isAnimationActive={false}
              />
              <Line type="monotone" dataKey="reward" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartShell>

        {/* frac_reward_zero_std with red threshold */}
        <ChartShell
          title="Frac Reward Zero Std"
          subtitle={formatNum(latest?.fracRewardZeroStd)}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={grpoMetrics} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(3)} />
              <ReferenceLine
                y={0.95}
                stroke="var(--status-error)"
                strokeDasharray="4 4"
                label={{ value: 'degenerate 0.95', position: 'insideTopRight', fill: 'var(--status-error)', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
              />
              <Line
                type="monotone"
                dataKey="fracRewardZeroStd"
                stroke="var(--accent)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartShell>

        {/* KL divergence */}
        <ChartShell title="KL Divergence" subtitle={formatNum(latest?.kl)}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={grpoMetrics} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="step" {...AXIS_STYLE} />
              <YAxis {...AXIS_STYLE} tickFormatter={(v) => v.toFixed(3)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(5)} />
              <Line type="monotone" dataKey="kl" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartShell>
      </div>
    </div>
  )
}
