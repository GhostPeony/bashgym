import { useCallback } from 'react'
import { useTutorialStore, TutorialStep } from '../stores/tutorialStore'

/**
 * Hook for completing tutorial steps from any component.
 * Call the returned function when the user completes an action
 * that corresponds to a tutorial step.
 */
export function useTutorialComplete() {
  const { isTutorialActive, completeStep, currentStep } = useTutorialStore()

  const complete = useCallback((step: TutorialStep) => {
    if (isTutorialActive) {
      completeStep(step)
    }
  }, [isTutorialActive, completeStep])

  const isCurrentStep = useCallback((step: TutorialStep) => {
    return isTutorialActive && currentStep === step
  }, [isTutorialActive, currentStep])

  return { complete, isCurrentStep, isTutorialActive }
}
