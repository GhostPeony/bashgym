import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from 'recharts'
import { useThemeStore } from '../../stores'

interface EpochProgressProps {
  currentEpoch: number
  totalEpochs: number
  currentStep: number
  totalSteps: number
}

export function EpochProgress({
  currentEpoch,
  totalEpochs,
  currentStep,
  totalSteps
}: EpochProgressProps) {
  const { theme } = useThemeStore()

  // Guard against division by zero
  const epochProgress = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0
  const stepProgress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0
  const isWaiting = totalEpochs === 0 && totalSteps === 0

  const colors = {
    primary: theme === 'dark' ? '#76B900' : '#0066CC',
    secondary: theme === 'dark' ? '#00A6FF' : '#76B900',
    background: theme === 'dark' ? '#2C2C2E' : '#E5E5EA',
    text: theme === 'dark' ? '#FFFFFF' : '#1D1D1F'
  }

  const data = [
    {
      name: 'Epoch',
      value: epochProgress,
      fill: colors.primary
    }
  ]

  return (
    <div className="flex flex-col items-center">
      {/* Radial Progress */}
      <div className="relative w-48 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="70%"
            outerRadius="100%"
            barSize={12}
            data={data}
            startAngle={90}
            endAngle={-270}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: colors.background }}
              dataKey="value"
              cornerRadius={6}
              animationDuration={1000}
            />
          </RadialBarChart>
        </ResponsiveContainer>

        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {isWaiting ? (
            <>
              <span className="text-2xl font-bold text-text-muted">—</span>
              <span className="text-sm text-text-muted uppercase tracking-wide">Waiting</span>
            </>
          ) : (
            <>
              <span className="text-3xl font-bold text-text-primary">
                {currentEpoch}/{totalEpochs}
              </span>
              <span className="text-sm text-text-muted uppercase tracking-wide">Epochs</span>
            </>
          )}
        </div>
      </div>

      {/* Step Progress */}
      <div className="w-full mt-4">
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-text-muted">Steps</span>
          <span className="font-medium text-text-primary">
            {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}
          </span>
        </div>
        <div className="h-2 bg-background-tertiary rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${stepProgress}%`,
              background: `linear-gradient(90deg, ${colors.primary}, ${colors.secondary})`
            }}
          />
        </div>
        <div className="flex items-center justify-end mt-1">
          <span className="text-xs text-text-muted">{stepProgress.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}
