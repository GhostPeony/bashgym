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
      {/* Backdrop — solid overlay, no blur */}
      <div
        className="fixed inset-0 z-[60]"
        style={{ backgroundColor: 'rgba(27, 32, 64, 0.15)' }}
        onClick={handleGotIt}
      />

      {/* Tooltip — brutalist card with accent border */}
      <div
        className="fixed z-[61] border-brutal border-accent rounded-brutal shadow-brutal bg-background-card overflow-hidden"
        style={{ top: position.top, left: position.left, width: 280 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-3 bg-accent-light border-b border-border">
          <h4 className="font-brand text-sm text-text-primary">{tooltipConfig.title}</h4>
          <button
            onClick={handleGotIt}
            className="btn-icon w-7 h-7 text-text-muted"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-3">
          <p className="text-sm text-text-secondary">{tooltipConfig.description}</p>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between p-3 border-t border-border bg-background-secondary">
          <button
            onClick={handleSkipStep}
            className="text-xs font-mono text-text-muted hover:text-text-secondary transition-press"
          >
            Skip this step
          </button>
          <button
            onClick={handleGotIt}
            className="btn-primary flex items-center gap-1 px-3 py-1.5 text-xs"
          >
            Got it
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {/* Arrow pointer — CSS triangle */}
        <div
          className={clsx(
            'absolute w-0 h-0',
            tooltipConfig.position === 'top' && 'bottom-[-8px] left-1/2 -translate-x-1/2',
            tooltipConfig.position === 'bottom' && 'top-[-8px] left-1/2 -translate-x-1/2',
            tooltipConfig.position === 'left' && 'right-[-8px] top-1/2 -translate-y-1/2',
            tooltipConfig.position === 'right' && 'left-[-8px] top-1/2 -translate-y-1/2'
          )}
          style={
            tooltipConfig.position === 'top'
              ? { borderLeft: '8px solid transparent', borderRight: '8px solid transparent', borderTop: '8px solid var(--accent)' }
              : tooltipConfig.position === 'bottom'
              ? { borderLeft: '8px solid transparent', borderRight: '8px solid transparent', borderBottom: '8px solid var(--accent)' }
              : tooltipConfig.position === 'left'
              ? { borderTop: '8px solid transparent', borderBottom: '8px solid transparent', borderLeft: '8px solid var(--accent)' }
              : { borderTop: '8px solid transparent', borderBottom: '8px solid transparent', borderRight: '8px solid var(--accent)' }
          }
        />
      </div>
    </>
  )
}
