import { clsx } from 'clsx'

export type FlywheelStage = 'act' | 'verify' | 'synthesize' | 'train' | 'deploy'

interface FlywheelMiniProps {
  currentStage?: FlywheelStage
  progress?: number
}

const stages: { id: FlywheelStage; label: string; icon: string }[] = [
  { id: 'act', label: 'Act', icon: '▶' },
  { id: 'verify', label: 'Verify', icon: '✓' },
  { id: 'synthesize', label: 'Synth', icon: '⚡' },
  { id: 'train', label: 'Train', icon: '◉' },
  { id: 'deploy', label: 'Deploy', icon: '↗' }
]

function StageButton({ stage, isActive, isComplete }: {
  stage: typeof stages[0]
  isActive: boolean
  isComplete: boolean
}) {
  return (
    <div
      className={clsx(
        'flex items-center gap-1.5 px-2 py-1.5 rounded text-xs transition-all',
        isActive && 'bg-primary/10'
      )}
    >
      <span
        className={clsx(
          'w-5 h-5 rounded flex items-center justify-center text-[10px] font-medium',
          isActive
            ? 'bg-primary text-white'
            : isComplete
            ? 'bg-primary/30 text-primary'
            : 'bg-background-tertiary text-text-muted'
        )}
      >
        {stage.icon}
      </span>
      <span
        className={clsx(
          isActive
            ? 'text-primary font-medium'
            : isComplete
            ? 'text-text-secondary'
            : 'text-text-muted'
        )}
      >
        {stage.label}
      </span>
    </div>
  )
}

export function FlywheelMini({ currentStage = 'act', progress = 0 }: FlywheelMiniProps) {
  const currentIndex = stages.findIndex((s) => s.id === currentStage)

  const getStageProps = (index: number) => ({
    stage: stages[index],
    isActive: currentStage === stages[index].id,
    isComplete: index < currentIndex
  })

  return (
    <div className="space-y-1">
      {/* Row 1: Act, Verify */}
      <div className="grid grid-cols-2 gap-1">
        <StageButton {...getStageProps(0)} />
        <StageButton {...getStageProps(1)} />
      </div>
      {/* Row 2: Synthesize, Train */}
      <div className="grid grid-cols-2 gap-1">
        <StageButton {...getStageProps(2)} />
        <StageButton {...getStageProps(3)} />
      </div>
      {/* Row 3: Deploy (centered) */}
      <div className="flex justify-center">
        <StageButton {...getStageProps(4)} />
      </div>
      {/* Progress indicator */}
      {progress > 0 && (
        <div className="pt-1">
          <div className="h-1 bg-background-tertiary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
