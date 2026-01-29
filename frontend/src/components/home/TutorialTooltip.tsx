import { useEffect, useState } from 'react'
import { X, ArrowRight } from 'lucide-react'
import { useTutorialStore, TutorialStep } from '../../stores/tutorialStore'
import { clsx } from 'clsx'

interface TooltipConfig {
  step: TutorialStep
  targetSelector: string
  title: string
  description: string
  position: 'top' | 'bottom' | 'left' | 'right'
}

const TOOLTIP_CONFIGS: TooltipConfig[] = [
  {
    step: 'install_hooks',
    targetSelector: '[data-tutorial="hooks-section"]',
    title: 'Install Capture Hooks',
    description: 'Hooks capture your Claude Code sessions automatically. Click Install to enable them.',
    position: 'left'
  },
  {
    step: 'import_traces',
    targetSelector: '[data-tutorial="import-button"]',
    title: 'Import Your Traces',
    description: 'You already have session history! Click Import to bring in your Claude Code traces.',
    position: 'bottom'
  },
  {
    step: 'generate_examples',
    targetSelector: '[data-tutorial="generate-button"]',
    title: 'Generate Training Examples',
    description: 'Select traces and click Generate to transform them into training data.',
    position: 'bottom'
  },
  {
    step: 'start_training',
    targetSelector: '[data-tutorial="start-training-button"]',
    title: 'Start Training',
    description: 'Your examples are ready. Configure your settings and start a training run.',
    position: 'left'
  },
  {
    step: 'view_model',
    targetSelector: '[data-tutorial="model-card"]',
    title: 'Your Model is Ready!',
    description: 'Congratulations! View your trained model and test it out.',
    position: 'bottom'
  }
]

export function TutorialTooltip() {
  const { isTutorialActive, currentStep, showTooltip, dismissTooltip, completeStep } = useTutorialStore()
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null)
  const [tooltipConfig, setTooltipConfig] = useState<TooltipConfig | null>(null)

  useEffect(() => {
    if (!isTutorialActive || !showTooltip) {
      setPosition(null)
      setTooltipConfig(null)
      return
    }

    const config = TOOLTIP_CONFIGS.find(c => c.step === currentStep)
    if (!config) {
      setPosition(null)
      setTooltipConfig(null)
      return
    }

    setTooltipConfig(config)

    // Find the target element
    const findTarget = () => {
      const target = document.querySelector(config.targetSelector)
      if (!target) return null

      const rect = target.getBoundingClientRect()
      const tooltipWidth = 280
      const tooltipHeight = 120
      const offset = 12

      let top = 0
      let left = 0

      switch (config.position) {
        case 'top':
          top = rect.top - tooltipHeight - offset
          left = rect.left + rect.width / 2 - tooltipWidth / 2
          break
        case 'bottom':
          top = rect.bottom + offset
          left = rect.left + rect.width / 2 - tooltipWidth / 2
          break
        case 'left':
          top = rect.top + rect.height / 2 - tooltipHeight / 2
          left = rect.left - tooltipWidth - offset
          break
        case 'right':
          top = rect.top + rect.height / 2 - tooltipHeight / 2
          left = rect.right + offset
          break
      }

      // Keep within viewport
      left = Math.max(16, Math.min(left, window.innerWidth - tooltipWidth - 16))
      top = Math.max(16, Math.min(top, window.innerHeight - tooltipHeight - 16))

      return { top, left }
    }

    // Initial position
    const pos = findTarget()
    if (pos) setPosition(pos)

    // Update position on scroll/resize
    const handleUpdate = () => {
      const pos = findTarget()
      if (pos) setPosition(pos)
    }

    window.addEventListener('scroll', handleUpdate, true)
    window.addEventListener('resize', handleUpdate)

    // Poll for element appearance (in case it's rendered after)
    const interval = setInterval(() => {
      if (!position) {
        const pos = findTarget()
        if (pos) setPosition(pos)
      }
    }, 500)

    return () => {
      window.removeEventListener('scroll', handleUpdate, true)
      window.removeEventListener('resize', handleUpdate)
      clearInterval(interval)
    }
  }, [isTutorialActive, currentStep, showTooltip])

  if (!isTutorialActive || !showTooltip || !position || !tooltipConfig) {
    return null
  }

  const handleGotIt = () => {
    dismissTooltip()
  }

  const handleSkipStep = () => {
    completeStep(currentStep)
  }

  return (
    <>
      {/* Backdrop - subtle overlay */}
      <div
        className="fixed inset-0 z-[60] bg-black/10"
        onClick={handleGotIt}
      />

      {/* Tooltip */}
      <div
        className="fixed z-[61] w-70 bg-background-secondary border border-primary/30 rounded-xl shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200"
        style={{ top: position.top, left: position.left, width: 280 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-3 bg-primary/10 border-b border-primary/20">
          <h4 className="text-sm font-semibold text-text-primary">{tooltipConfig.title}</h4>
          <button
            onClick={handleGotIt}
            className="p-1 rounded hover:bg-background-tertiary transition-colors text-text-muted"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-3">
          <p className="text-sm text-text-secondary">{tooltipConfig.description}</p>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between p-3 border-t border-border-subtle bg-background-tertiary">
          <button
            onClick={handleSkipStep}
            className="text-xs text-text-muted hover:text-text-secondary transition-colors"
          >
            Skip this step
          </button>
          <button
            onClick={handleGotIt}
            className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors"
          >
            Got it
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {/* Arrow pointer */}
        <div
          className={clsx(
            'absolute w-3 h-3 bg-background-secondary border-primary/30 transform rotate-45',
            tooltipConfig.position === 'top' && 'bottom-[-7px] left-1/2 -translate-x-1/2 border-b border-r',
            tooltipConfig.position === 'bottom' && 'top-[-7px] left-1/2 -translate-x-1/2 border-t border-l',
            tooltipConfig.position === 'left' && 'right-[-7px] top-1/2 -translate-y-1/2 border-t border-r',
            tooltipConfig.position === 'right' && 'left-[-7px] top-1/2 -translate-y-1/2 border-b border-l'
          )}
        />
      </div>
    </>
  )
}
