import { ChevronRight, X, HelpCircle } from 'lucide-react'
import { useTutorialStore, TutorialStep } from '../../stores/tutorialStore'
import { useUIStore } from '../../stores'
import { clsx } from 'clsx'

interface StepConfig {
  id: TutorialStep
  label: string
  description: string
}

const STEPS: StepConfig[] = [
  { id: 'welcome', label: 'Welcome to Bash Gym', description: 'Introduction complete' },
  { id: 'install_hooks', label: 'Install capture hooks', description: 'Enable automatic session capture' },
  { id: 'import_traces', label: 'Import your traces', description: 'Bring in your Claude Code history' },
  { id: 'generate_examples', label: 'Generate training examples', description: 'Transform traces into training data' },
  { id: 'start_training', label: 'Start a training run', description: 'Fine-tune your first model' },
  { id: 'view_model', label: 'View your trained model', description: 'See the results' }
]

function ChecklistItem({ step, isCompleted, isCurrent, onClick }: {
  step: StepConfig
  isCompleted: boolean
  isCurrent: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full flex items-start gap-3 p-2 text-left transition-press',
        'border-b border-border-subtle last:border-b-0',
        isCurrent && !isCompleted && 'bg-accent-light',
        !isCurrent && 'hover:bg-background-secondary'
      )}
    >
      <div className="flex-shrink-0 mt-0.5">
        {isCompleted ? (
          /* Triangle checkmark pointing up — success */
          <div
            className="w-5 h-5 flex items-center justify-center"
          >
            <div
              className="w-0 h-0"
              style={{
                borderLeft: '6px solid transparent',
                borderRight: '6px solid transparent',
                borderBottom: '10px solid var(--status-success)',
              }}
            />
          </div>
        ) : isCurrent ? (
          /* Active — filled triangle pointing right */
          <div className="w-5 h-5 flex items-center justify-center">
            <div
              className="w-0 h-0"
              style={{
                borderTop: '6px solid transparent',
                borderBottom: '6px solid transparent',
                borderLeft: '10px solid var(--accent)',
              }}
            />
          </div>
        ) : (
          /* Inactive — border-only square */
          <div className="w-5 h-5 border-brutal border-border rounded-brutal" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className={clsx(
          'text-sm font-mono',
          isCompleted ? 'text-text-muted line-through' : isCurrent ? 'text-text-primary font-semibold' : 'text-text-secondary'
        )}>
          {step.label}
        </p>
        {isCurrent && !isCompleted && (
          <p className="text-xs text-text-muted mt-0.5">{step.description}</p>
        )}
      </div>
      {isCurrent && !isCompleted && (
        <ChevronRight className="w-4 h-4 text-accent flex-shrink-0" />
      )}
    </button>
  )
}

export function TutorialChecklist() {
  const {
    isTutorialActive,
    currentStep,
    completedSteps,
    isChecklistMinimized,
    toggleChecklist,
    goToStep,
    skipTutorial
  } = useTutorialStore()
  const { setSettingsOpen, openOverlay, closeOverlay } = useUIStore()

  if (!isTutorialActive) return null

  // Minimized state - just a floating button
  if (isChecklistMinimized) {
    return (
      <button
        onClick={toggleChecklist}
        className="fixed bottom-20 right-4 z-50 btn-icon w-12 h-12 bg-accent text-white"
        title="Show tutorial progress"
      >
        <HelpCircle className="w-6 h-6" />
      </button>
    )
  }

  const handleStepClick = (step: TutorialStep) => {
    // Navigate to appropriate view based on step
    switch (step) {
      case 'install_hooks':
        setSettingsOpen(true)
        break
      case 'import_traces':
      case 'generate_examples':
        openOverlay('factory')
        break
      case 'start_training':
        openOverlay('training')
        break
      case 'view_model':
        openOverlay('models')
        break
      default:
        closeOverlay()
    }
    goToStep(step)
  }

  const completedCount = completedSteps.length
  const totalSteps = STEPS.length

  return (
    <div className="fixed bottom-20 right-4 z-50 w-72 border-brutal border-border rounded-brutal shadow-brutal bg-background-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border bg-background-secondary">
        <div>
          <h3 className="font-brand text-sm text-text-primary">Getting Started</h3>
          <p className="text-xs font-mono text-text-muted">{completedCount} of {totalSteps} complete</p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleChecklist}
            className="btn-icon w-7 h-7 text-text-muted"
            title="Minimize"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={skipTutorial}
            className="btn-icon w-7 h-7 text-text-muted"
            title="Close tutorial"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="progress-bar" style={{ height: '8px', borderLeft: 'none', borderRight: 'none', borderRadius: 0 }}>
        <div
          className="progress-fill"
          style={{ width: `${(completedCount / totalSteps) * 100}%` }}
        />
      </div>

      {/* Steps */}
      <div className="max-h-80 overflow-y-auto">
        {STEPS.map((step) => (
          <ChecklistItem
            key={step.id}
            step={step}
            isCompleted={completedSteps.includes(step.id)}
            isCurrent={currentStep === step.id}
            onClick={() => handleStepClick(step.id)}
          />
        ))}
      </div>

      {/* Footer hint */}
      <div className="p-3 border-t border-border bg-background-secondary">
        <p className="text-xs text-text-muted text-center font-mono">
          Click any step to navigate there
        </p>
      </div>
    </div>
  )
}
