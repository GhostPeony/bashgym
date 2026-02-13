import { clsx } from 'clsx'

export type FlywheelStage = 'act' | 'verify' | 'synthesize' | 'train' | 'deploy'

interface FlywheelMiniProps {
  currentStage?: FlywheelStage
  progress?: number
}

const stages: { id: FlywheelStage; label: string; icon: string }[] = [
  { id: 'act', label: 'Act', icon: '\u25B6' },
  { id: 'verify', label: 'Verify', icon: '\u2713' },
  { id: 'synthesize', label: 'Synth', icon: '\u26A1' },
  { id: 'train', label: 'Train', icon: '\u25C9' },
  { id: 'deploy', label: 'Deploy', icon: '\u2197' }
]

function StageButton({ stage, isActive, isComplete }: {
  stage: typeof stages[0]
  isActive: boolean
  isComplete: boolean
}) {
  return (
    <div
      className={clsx(
        'flex items-center gap-1.5 px-2 py-1.5 text-xs transition-press border-brutal rounded-brutal',
        isActive
          ? 'bg-accent-light border-accent'
          : 'bg-background-card border-border-subtle'
      )}
    >
      <span
        className={clsx(
          'w-5 h-5 flex items-center justify-center text-[10px] font-mono font-semibold border-brutal rounded-brutal',
          isActive
            ? 'bg-accent text-white border-accent'
            : isComplete
            ? 'bg-accent-light text-accent-dark border-border'
            : 'bg-background-secondary text-text-muted border-border'
        )}
      >
        {stage.icon}
      </span>
      <span
        className={clsx(
          'font-mono text-xs',
          isActive
            ? 'text-accent-dark font-semibold'
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
      {/* Progress indicator — brutalist progress bar */}
      {progress > 0 && (
        <div className="pt-1">
          <div className="progress-bar" style={{ height: '8px' }}>
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
