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
import { useThemeStore } from '../../stores'

interface LossCurveProps {
  data: Array<{ step: number; loss: number }>
}

export function LossCurve({ data }: LossCurveProps) {
  const { theme } = useThemeStore()

  const colors = {
    line: theme === 'dark' ? '#76B900' : '#0066CC',
    grid: theme === 'dark' ? '#2C2C2E' : '#E5E5EA',
    text: theme === 'dark' ? '#A1A1A6' : '#6E6E73',
    background: theme === 'dark' ? '#1A1A1A' : '#FFFFFF',
    tooltip: theme === 'dark' ? '#242424' : '#F5F5F7'
  }

  const minLoss = Math.min(...data.map((d) => d.loss))
  const maxLoss = Math.max(...data.map((d) => d.loss))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={colors.line} stopOpacity={0.3} />
            <stop offset="95%" stopColor={colors.line} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
        <XAxis
          dataKey="step"
          stroke={colors.text}
          fontSize={11}
          tickLine={false}
          axisLine={{ stroke: colors.grid }}
          tickFormatter={(value) => (value >= 1000 ? `${value / 1000}k` : value)}
        />
        <YAxis
          stroke={colors.text}
          fontSize={11}
          tickLine={false}
          axisLine={{ stroke: colors.grid }}
          domain={[Math.floor(minLoss * 0.9 * 10) / 10, Math.ceil(maxLoss * 1.1 * 10) / 10]}
          tickFormatter={(value) => value.toFixed(2)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: colors.tooltip,
            border: `1px solid ${colors.grid}`,
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            fontSize: '12px'
          }}
          labelStyle={{ color: colors.text, marginBottom: '4px' }}
          formatter={(value: number) => [value.toFixed(4), 'Loss']}
          labelFormatter={(label) => `Step ${label}`}
        />
        <ReferenceLine
          y={minLoss}
          stroke={colors.line}
          strokeDasharray="5 5"
          strokeOpacity={0.5}
        />
        <Line
          type="monotone"
          dataKey="loss"
          stroke={colors.line}
          strokeWidth={2}
          dot={false}
          activeDot={{
            r: 4,
            fill: colors.line,
            stroke: colors.background,
            strokeWidth: 2
          }}
          fill="url(#lossGradient)"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
