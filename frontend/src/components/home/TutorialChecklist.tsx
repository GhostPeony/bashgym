import { CheckCircle2, Circle, ChevronRight, X, HelpCircle } from 'lucide-react'
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
        'w-full flex items-start gap-3 p-2 rounded-lg text-left transition-colors',
        isCurrent && !isCompleted && 'bg-primary/10',
        !isCurrent && 'hover:bg-background-tertiary'
      )}
    >
      <div className="flex-shrink-0 mt-0.5">
        {isCompleted ? (
          <CheckCircle2 className="w-5 h-5 text-status-success" />
        ) : isCurrent ? (
          <div className="w-5 h-5 rounded-full border-2 border-primary flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-primary" />
          </div>
        ) : (
          <Circle className="w-5 h-5 text-text-muted" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className={clsx(
          'text-sm font-medium',
          isCompleted ? 'text-text-muted line-through' : isCurrent ? 'text-text-primary' : 'text-text-secondary'
        )}>
          {step.label}
        </p>
        {isCurrent && !isCompleted && (
          <p className="text-xs text-text-muted mt-0.5">{step.description}</p>
        )}
      </div>
      {isCurrent && !isCompleted && (
        <ChevronRight className="w-4 h-4 text-primary flex-shrink-0" />
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
        className="fixed bottom-20 right-4 z-50 w-12 h-12 rounded-full bg-primary text-white shadow-lg hover:bg-primary/90 transition-colors flex items-center justify-center"
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
    <div className="fixed bottom-20 right-4 z-50 w-72 bg-background-secondary border border-border-subtle rounded-xl shadow-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border-subtle bg-background-tertiary">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Getting Started</h3>
          <p className="text-xs text-text-muted">{completedCount} of {totalSteps} complete</p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleChecklist}
            className="p-1.5 rounded-lg hover:bg-background-primary transition-colors text-text-muted hover:text-text-secondary"
            title="Minimize"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={skipTutorial}
            className="p-1.5 rounded-lg hover:bg-background-primary transition-colors text-text-muted hover:text-text-secondary"
            title="Close tutorial"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-background-tertiary">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${(completedCount / totalSteps) * 100}%` }}
        />
      </div>

      {/* Steps */}
      <div className="p-2 max-h-80 overflow-y-auto">
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
      <div className="p-3 border-t border-border-subtle bg-background-tertiary">
        <p className="text-xs text-text-muted text-center">
          Click any step to navigate there
        </p>
      </div>
    </div>
  )
}
