import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'

interface LossCurveProps {
  data: Array<{ step: number; loss: number; evalLoss?: number }>
  smoothed?: boolean
}

// Exponential moving average (weight on history), matching the ~0.6 default of
// W&B / TensorBoard / Unsloth Studio smoothing.
function ema(values: number[], alpha = 0.6): number[] {
  const out: number[] = []
  let prev: number | null = null
  for (const v of values) {
    prev = prev === null ? v : alpha * prev + (1 - alpha) * v
    out.push(prev)
  }
  return out
}

export function LossCurve({ data, smoothed = false }: LossCurveProps) {
  const hasEval = data.some((d) => d.evalLoss !== undefined && d.evalLoss !== null)

  const smoothedLoss = smoothed ? ema(data.map((d) => d.loss)) : null
  const chartData = data.map((d, i) => ({
    ...d,
    lossSmooth: smoothedLoss ? smoothedLoss[i] : d.loss
  }))

  const allLosses = data.flatMap((d) =>
    d.evalLoss !== undefined && d.evalLoss !== null ? [d.loss, d.evalLoss] : [d.loss]
  )
  const minLoss = Math.min(...allLosses)
  const maxLoss = Math.max(...allLosses)

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--border-subtle)"
          vertical={false}
        />
        <XAxis
          dataKey="step"
          stroke="var(--text-muted)"
          fontSize={11}
          fontFamily="'JetBrains Mono', monospace"
          tickLine={false}
          axisLine={{ stroke: 'var(--border-color)' }}
          tickFormatter={(value) => (value >= 1000 ? `${value / 1000}k` : value)}
        />
        <YAxis
          stroke="var(--text-muted)"
          fontSize={11}
          fontFamily="'JetBrains Mono', monospace"
          tickLine={false}
          axisLine={{ stroke: 'var(--border-color)' }}
          domain={[Math.floor(minLoss * 0.9 * 10) / 10, Math.ceil(maxLoss * 1.1 * 10) / 10]}
          tickFormatter={(value) => value.toFixed(2)}
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
          formatter={(value: number, name: string) => [
            value.toFixed(4),
            name === 'evalLoss' ? 'Eval loss' : name === 'loss' ? 'Train loss (raw)' : 'Train loss'
          ]}
          labelFormatter={(label) => `Step ${label}`}
        />
        <ReferenceLine
          y={minLoss}
          stroke="var(--accent)"
          strokeDasharray="5 5"
          strokeOpacity={0.5}
        />
        {smoothed && (
          <Line type="monotone" dataKey="loss" stroke="var(--accent)" strokeWidth={1} strokeOpacity={0.25} dot={false} />
        )}
        <Line
          type="monotone"
          dataKey={smoothed ? 'lossSmooth' : 'loss'}
          stroke="var(--accent)"
          strokeWidth={2}
          dot={false}
          activeDot={{
            r: 4,
            fill: 'var(--accent)',
            stroke: 'var(--bg-card)',
            strokeWidth: 2
          }}
        />
        {hasEval && (
          <Line
            type="monotone"
            dataKey="evalLoss"
            stroke="var(--text-secondary)"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
            connectNulls
            activeDot={{ r: 4, fill: 'var(--text-secondary)', stroke: 'var(--bg-card)', strokeWidth: 2 }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
