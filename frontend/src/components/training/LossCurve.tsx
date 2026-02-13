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
  data: Array<{ step: number; loss: number }>
}

export function LossCurve({ data }: LossCurveProps) {
  const minLoss = Math.min(...data.map((d) => d.loss))
  const maxLoss = Math.max(...data.map((d) => d.loss))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
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
          formatter={(value: number) => [value.toFixed(4), 'Loss']}
          labelFormatter={(label) => `Step ${label}`}
        />
        <ReferenceLine
          y={minLoss}
          stroke="var(--accent)"
          strokeDasharray="5 5"
          strokeOpacity={0.5}
        />
        <Line
          type="monotone"
          dataKey="loss"
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
      </LineChart>
    </ResponsiveContainer>
  )
}
