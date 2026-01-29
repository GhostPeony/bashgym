import { useThemeStore } from '../../stores'
import { clsx } from 'clsx'

export type FlywheelStage = 'act' | 'verify' | 'synthesize' | 'train' | 'deploy'

interface FlywheelVisualizationProps {
  currentStage?: FlywheelStage
  progress?: number
  showLabels?: boolean
}

const stages: { id: FlywheelStage; label: string; shortLabel: string; angle: number }[] = [
  { id: 'act', label: 'Act (Arena)', shortLabel: 'ACT', angle: -90 },
  { id: 'verify', label: 'Verify (Judge)', shortLabel: 'VER', angle: -18 },
  { id: 'synthesize', label: 'Synthesize (Factory)', shortLabel: 'SYN', angle: 54 },
  { id: 'train', label: 'Train (Gym)', shortLabel: 'TRN', angle: 126 },
  { id: 'deploy', label: 'Deploy', shortLabel: 'DEP', angle: 198 }
]

export function FlywheelVisualization({
  currentStage = 'act',
  progress = 0,
  showLabels = true
}: FlywheelVisualizationProps) {
  const { theme } = useThemeStore()

  const colors = {
    active: theme === 'dark' ? '#76B900' : '#0066CC',
    inactive: theme === 'dark' ? '#2C2C2E' : '#E5E5EA',
    text: theme === 'dark' ? '#FFFFFF' : '#1D1D1F',
    mutedText: theme === 'dark' ? '#6E6E73' : '#86868B',
    glow: theme === 'dark' ? 'rgba(118, 185, 0, 0.4)' : 'rgba(0, 102, 204, 0.3)'
  }

  const radius = 120
  const centerX = 150
  const centerY = 150
  const nodeRadius = 32

  // Calculate position on circle
  const getPosition = (angle: number) => {
    const rad = (angle * Math.PI) / 180
    return {
      x: centerX + radius * Math.cos(rad),
      y: centerY + radius * Math.sin(rad)
    }
  }

  // Get current stage index
  const currentIndex = stages.findIndex((s) => s.id === currentStage)

  return (
    <div className="relative">
      <svg width="300" height="300" viewBox="0 0 300 300">
        {/* Background Circle */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={colors.inactive}
          strokeWidth="2"
          strokeDasharray="5,5"
        />

        {/* Progress Arc */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={colors.active}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={`${(progress / 100) * 2 * Math.PI * radius} ${2 * Math.PI * radius}`}
          transform={`rotate(-90 ${centerX} ${centerY})`}
          style={{
            filter: theme === 'dark' ? `drop-shadow(0 0 8px ${colors.glow})` : 'none'
          }}
        />

        {/* Connection Lines */}
        {stages.map((stage, i) => {
          const pos1 = getPosition(stage.angle)
          const pos2 = getPosition(stages[(i + 1) % stages.length].angle)
          const isActive = i <= currentIndex || (currentIndex === stages.length - 1 && i === 0)

          return (
            <line
              key={`line-${i}`}
              x1={pos1.x}
              y1={pos1.y}
              x2={pos2.x}
              y2={pos2.y}
              stroke={isActive ? colors.active : colors.inactive}
              strokeWidth="2"
              strokeOpacity={isActive ? 0.6 : 0.3}
            />
          )
        })}

        {/* Stage Nodes */}
        {stages.map((stage, i) => {
          const pos = getPosition(stage.angle)
          const isActive = stage.id === currentStage
          const isPast = i < currentIndex

          return (
            <g key={stage.id}>
              {/* Glow effect for active node */}
              {isActive && (
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={nodeRadius + 8}
                  fill={colors.glow}
                  className="animate-pulse"
                />
              )}

              {/* Node circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={nodeRadius}
                fill={isActive ? colors.active : isPast ? colors.active : colors.inactive}
                fillOpacity={isActive ? 1 : isPast ? 0.6 : 1}
                stroke={isActive ? colors.active : 'none'}
                strokeWidth="3"
                className={isActive ? 'stage-active' : ''}
              />

              {/* Node text */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isActive || isPast ? '#FFFFFF' : colors.mutedText}
                fontSize="11"
                fontWeight="bold"
                fontFamily="SF Mono, monospace"
              >
                {stage.shortLabel}
              </text>
            </g>
          )
        })}

        {/* Center Text */}
        <text
          x={centerX}
          y={centerY - 8}
          textAnchor="middle"
          fill={colors.text}
          fontSize="14"
          fontWeight="600"
        >
          OUROBOROS
        </text>
        <text
          x={centerX}
          y={centerY + 10}
          textAnchor="middle"
          fill={colors.mutedText}
          fontSize="11"
        >
          {progress.toFixed(0)}% Complete
        </text>
      </svg>

      {/* Labels */}
      {showLabels && (
        <div className="absolute inset-0 pointer-events-none">
          {stages.map((stage) => {
            const pos = getPosition(stage.angle)
            const isActive = stage.id === currentStage

            // Position label outside the circle
            const labelOffset = 55
            const labelPos = getPosition(stage.angle)
            const labelX =
              labelPos.x < centerX
                ? labelPos.x - labelOffset
                : labelPos.x > centerX
                ? labelPos.x + labelOffset - 100
                : labelPos.x - 50
            const labelY = labelPos.y < centerY ? labelPos.y - 20 : labelPos.y + 10

            return (
              <div
                key={`label-${stage.id}`}
                className={clsx(
                  'absolute text-xs whitespace-nowrap transition-opacity',
                  isActive ? 'text-primary font-medium' : 'text-text-muted'
                )}
                style={{
                  left: `${(labelX / 300) * 100}%`,
                  top: `${(labelY / 300) * 100}%`,
                  width: '100px',
                  textAlign: labelPos.x < centerX ? 'right' : labelPos.x > centerX ? 'left' : 'center'
                }}
              >
                {stage.label}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
