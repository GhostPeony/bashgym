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
    active: 'var(--accent)',
    activeDark: 'var(--accent-dark)',
    inactive: 'var(--border-subtle)',
    border: 'var(--border-color)',
    text: 'var(--text-primary)',
    mutedText: 'var(--text-muted)',
    card: 'var(--bg-card)',
    bg: 'var(--bg-secondary)',
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
    <div className="relative flywheel-animate">
      <svg width="300" height="300" viewBox="0 0 300 300">
        {/* Background Circle — dashed brutalist */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={colors.inactive}
          strokeWidth="2"
          strokeDasharray="8,4"
        />

        {/* Progress Arc — bold accent stroke */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={colors.active}
          strokeWidth="4"
          strokeLinecap="butt"
          strokeDasharray={`${(progress / 100) * 2 * Math.PI * radius} ${2 * Math.PI * radius}`}
          transform={`rotate(-90 ${centerX} ${centerY})`}
        />

        {/* Connection Lines — bold straight lines */}
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
              strokeWidth={isActive ? 3 : 2}
            />
          )
        })}

        {/* Stage Nodes — hard-bordered with triangular markers */}
        {stages.map((stage, i) => {
          const pos = getPosition(stage.angle)
          const isActive = stage.id === currentStage
          const isPast = i < currentIndex

          return (
            <g key={stage.id}>
              {/* Triangular marker for active node */}
              {isActive && (
                <polygon
                  points={`${pos.x},${pos.y - nodeRadius - 12} ${pos.x - 6},${pos.y - nodeRadius - 4} ${pos.x + 6},${pos.y - nodeRadius - 4}`}
                  fill={colors.active}
                />
              )}

              {/* Node circle — hard border */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={nodeRadius}
                fill={isActive ? colors.active : isPast ? colors.activeDark : colors.card}
                stroke={colors.border}
                strokeWidth={isActive ? 3 : 2}
              />

              {/* Node text — mono */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                fill={isActive || isPast ? '#FFFFFF' : colors.mutedText}
                fontSize="11"
                fontWeight="bold"
                fontFamily="'JetBrains Mono', monospace"
              >
                {stage.shortLabel}
              </text>
            </g>
          )
        })}

        {/* Center Text — serif brand font */}
        <text
          x={centerX}
          y={centerY - 8}
          textAnchor="middle"
          fill={colors.text}
          fontSize="14"
          fontWeight="600"
          fontFamily="'Cormorant Garamond', serif"
        >
          OUROBOROS
        </text>
        <text
          x={centerX}
          y={centerY + 10}
          textAnchor="middle"
          fill={colors.mutedText}
          fontSize="11"
          fontFamily="'JetBrains Mono', monospace"
        >
          {progress.toFixed(0)}%
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
                  'absolute text-xs font-mono whitespace-nowrap',
                  isActive ? 'text-accent-dark font-semibold' : 'text-text-muted'
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
