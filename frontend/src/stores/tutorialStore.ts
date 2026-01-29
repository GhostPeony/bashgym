import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type TutorialStep =
  | 'welcome'
  | 'install_hooks'
  | 'import_traces'
  | 'generate_examples'
  | 'start_training'
  | 'view_model'
  | 'completed'

interface TutorialState {
  // Whether user has seen the intro screen
  hasSeenIntro: boolean
  // Whether tutorial is active (user clicked "Let's Go")
  isTutorialActive: boolean
  // Current step in the tutorial
  currentStep: TutorialStep
  // Completed steps
  completedSteps: TutorialStep[]
  // Whether the checklist panel is minimized
  isChecklistMinimized: boolean
  // Whether to show tooltip for current step
  showTooltip: boolean

  // Actions
  setHasSeenIntro: (seen: boolean) => void
  startTutorial: () => void
  skipTutorial: () => void
  completeStep: (step: TutorialStep) => void
  goToStep: (step: TutorialStep) => void
  toggleChecklist: () => void
  dismissTooltip: () => void
  showTooltipForStep: () => void
  resetTutorial: () => void
}

const STEP_ORDER: TutorialStep[] = [
  'welcome',
  'install_hooks',
  'import_traces',
  'generate_examples',
  'start_training',
  'view_model',
  'completed'
]

export const useTutorialStore = create<TutorialState>()(
  persist(
    (set, get) => ({
      hasSeenIntro: false,
      isTutorialActive: false,
      currentStep: 'welcome',
      completedSteps: [],
      isChecklistMinimized: false,
      showTooltip: false,

      setHasSeenIntro: (seen) => set({ hasSeenIntro: seen }),

      startTutorial: () => set({
        isTutorialActive: true,
        hasSeenIntro: true,
        currentStep: 'install_hooks',
        completedSteps: ['welcome'],
        showTooltip: true
      }),

      skipTutorial: () => set({
        hasSeenIntro: true,
        isTutorialActive: false
      }),

      completeStep: (step) => {
        const { completedSteps, currentStep } = get()
        if (completedSteps.includes(step)) return

        const newCompleted = [...completedSteps, step]
        const currentIndex = STEP_ORDER.indexOf(currentStep)
        const nextStep = STEP_ORDER[currentIndex + 1] || 'completed'

        set({
          completedSteps: newCompleted,
          currentStep: nextStep,
          showTooltip: nextStep !== 'completed',
          isTutorialActive: nextStep !== 'completed'
        })
      },

      goToStep: (step) => set({
        currentStep: step,
        showTooltip: true
      }),

      toggleChecklist: () => set((state) => ({
        isChecklistMinimized: !state.isChecklistMinimized
      })),

      dismissTooltip: () => set({ showTooltip: false }),

      showTooltipForStep: () => set({ showTooltip: true }),

      resetTutorial: () => set({
        hasSeenIntro: false,
        isTutorialActive: false,
        currentStep: 'welcome',
        completedSteps: [],
        isChecklistMinimized: false,
        showTooltip: false
      })
    }),
    {
      name: 'bashgym-tutorial'
    }
  )
)
