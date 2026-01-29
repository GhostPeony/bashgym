import { useState, useEffect } from 'react'
import { Modal } from '../common/Modal'
import { Button } from '../common/Button'
import { useUIStore } from '../../stores'
import { hooksApi, systemApi, tracesApi } from '../../services/api'
import { clsx } from 'clsx'
import {
  CheckCircle2,
  Circle,
  Sparkles,
  Terminal,
  FileStack,
  BarChart3,
  Rocket,
  Copy,
  Check,
  ExternalLink,
  Play,
  ArrowRight,
  AlertCircle,
  RefreshCw
} from 'lucide-react'

interface Step {
  id: number
  title: string
  description: string
  icon: React.ReactNode
  completed: boolean
  action?: () => void
  actionLabel?: string
}

export function OnboardingModal() {
  const { isOnboardingOpen, setOnboardingOpen, openOverlay } = useUIStore()
  const [currentStep, setCurrentStep] = useState(0)
  const [hooksInstalled, setHooksInstalled] = useState(false)
  const [traceCount, setTraceCount] = useState(0)
  const [isCheckingStatus, setIsCheckingStatus] = useState(false)
  const [copied, setCopied] = useState(false)
  const [installing, setInstalling] = useState(false)

  // Check status on open
  useEffect(() => {
    if (isOnboardingOpen) {
      checkStatus()
    }
  }, [isOnboardingOpen])

  const checkStatus = async () => {
    setIsCheckingStatus(true)
    try {
      // Check hooks
      const hooksResult = await hooksApi.getStatus()
      if (hooksResult.ok && hooksResult.data) {
        setHooksInstalled(hooksResult.data.all_installed)
        if (hooksResult.data.all_installed && currentStep === 0) {
          setCurrentStep(1)
        }
      }

      // Check trace count
      const tracesResult = await tracesApi.listRepos()
      if (tracesResult.ok && tracesResult.data) {
        const total = tracesResult.data.reduce((sum, r) => sum + (r.trace_count || 0), 0)
        setTraceCount(total)
        if (total > 0 && currentStep === 1) {
          setCurrentStep(2)
        }
      }
    } finally {
      setIsCheckingStatus(false)
    }
  }

  const handleInstallHooks = async () => {
    setInstalling(true)
    try {
      const result = await hooksApi.install()
      if (result.ok) {
        setHooksInstalled(true)
        setCurrentStep(1)
      }
    } finally {
      setInstalling(false)
    }
  }

  const copyCommand = () => {
    const cmd = navigator.platform.toLowerCase().includes('win')
      ? 'xcopy /E /I bashgym\\hooks %USERPROFILE%\\.claude\\hooks'
      : 'cp -r bashgym/hooks/* ~/.claude/hooks/'
    navigator.clipboard.writeText(cmd)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const steps: Step[] = [
    {
      id: 1,
      title: 'Install Observability Hooks',
      description: 'Enable automatic trace capture from your Claude Code sessions',
      icon: <Terminal className="w-5 h-5" />,
      completed: hooksInstalled,
      action: handleInstallHooks,
      actionLabel: installing ? 'Installing...' : 'Install Hooks'
    },
    {
      id: 2,
      title: 'Collect Traces',
      description: 'Work with Claude Code on real tasks to generate training data',
      icon: <FileStack className="w-5 h-5" />,
      completed: traceCount >= 10
    },
    {
      id: 3,
      title: 'Review & Curate',
      description: 'Mark successful traces as "gold" for training data',
      icon: <Sparkles className="w-5 h-5" />,
      completed: false,
      action: () => {
        openOverlay('traces')
        setOnboardingOpen(false)
      },
      actionLabel: 'Open Traces'
    },
    {
      id: 4,
      title: 'Start Training',
      description: 'Train your personal AI coding assistant',
      icon: <Rocket className="w-5 h-5" />,
      completed: false,
      action: () => {
        openOverlay('training')
        setOnboardingOpen(false)
      },
      actionLabel: 'Open Training'
    }
  ]

  return (
    <Modal
      isOpen={isOnboardingOpen}
      onClose={() => setOnboardingOpen(false)}
      title="Getting Started"
      description="Set up BashGym in a few simple steps"
      size="lg"
    >
      <div className="space-y-6">
        {/* Progress indicator */}
        <div className="flex items-center justify-between px-4">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <div
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center transition-colors',
                  step.completed
                    ? 'bg-status-success text-white'
                    : idx === currentStep
                    ? 'bg-primary text-white'
                    : 'bg-background-tertiary text-text-muted'
                )}
              >
                {step.completed ? <Check className="w-4 h-4" /> : step.id}
              </div>
              {idx < steps.length - 1 && (
                <div
                  className={clsx(
                    'w-16 h-0.5 mx-2',
                    idx < currentStep ? 'bg-status-success' : 'bg-border-subtle'
                  )}
                />
              )}
            </div>
          ))}
        </div>

        {/* Current step content */}
        <div className="bg-background-tertiary rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div
              className={clsx(
                'p-3 rounded-xl',
                steps[currentStep].completed ? 'bg-status-success/20 text-status-success' : 'bg-primary/20 text-primary'
              )}
            >
              {steps[currentStep].icon}
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-text-primary">
                {steps[currentStep].title}
              </h3>
              <p className="text-sm text-text-secondary mt-1">
                {steps[currentStep].description}
              </p>

              {/* Step-specific content */}
              <div className="mt-4">
                {currentStep === 0 && (
                  <div className="space-y-4">
                    {hooksInstalled ? (
                      <div className="flex items-center gap-2 text-status-success">
                        <CheckCircle2 className="w-5 h-5" />
                        <span>Hooks are installed and ready</span>
                      </div>
                    ) : (
                      <>
                        <p className="text-sm text-text-muted">
                          BashGym uses Claude Code hooks to capture your coding sessions automatically.
                          Click the button below to install them, or copy the command to run manually.
                        </p>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="primary"
                            onClick={handleInstallHooks}
                            disabled={installing}
                          >
                            {installing ? (
                              <>
                                <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                                Installing...
                              </>
                            ) : (
                              'Install Hooks'
                            )}
                          </Button>
                          <span className="text-text-muted text-sm">or</span>
                          <button
                            onClick={copyCommand}
                            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-background-secondary border border-border-subtle hover:border-border-color transition-colors text-sm"
                          >
                            {copied ? (
                              <>
                                <Check className="w-4 h-4 text-status-success" />
                                Copied!
                              </>
                            ) : (
                              <>
                                <Copy className="w-4 h-4" />
                                Copy Command
                              </>
                            )}
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {currentStep === 1 && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-background-secondary">
                      <span className="text-sm text-text-secondary">Traces collected</span>
                      <span className={clsx(
                        'text-lg font-semibold',
                        traceCount >= 10 ? 'text-status-success' : 'text-text-primary'
                      )}>
                        {traceCount} / 10 minimum
                      </span>
                    </div>

                    {traceCount < 10 ? (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-status-warning/10 border border-status-warning/20">
                        <AlertCircle className="w-4 h-4 text-status-warning flex-shrink-0 mt-0.5" />
                        <div className="text-sm text-text-secondary">
                          <p className="font-medium text-text-primary mb-1">Keep working with Claude Code</p>
                          <p>
                            Each completed task generates a trace. Aim for at least 10-20 diverse traces
                            before training for best results.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-status-success">
                        <CheckCircle2 className="w-5 h-5" />
                        <span>You have enough traces to start training!</span>
                      </div>
                    )}

                    <Button variant="secondary" onClick={checkStatus} disabled={isCheckingStatus}>
                      <RefreshCw className={clsx('w-4 h-4 mr-2', isCheckingStatus && 'animate-spin')} />
                      Refresh Count
                    </Button>
                  </div>
                )}

                {currentStep === 2 && (
                  <div className="space-y-4">
                    <p className="text-sm text-text-muted">
                      Review your collected traces and mark successful sessions as "gold" traces.
                      Gold traces are used for training your model.
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-background-secondary">
                        <p className="text-xs text-text-muted">Total Traces</p>
                        <p className="text-xl font-semibold text-text-primary">{traceCount}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-background-secondary">
                        <p className="text-xs text-text-muted">Gold Traces</p>
                        <p className="text-xl font-semibold text-status-success">0</p>
                      </div>
                    </div>
                    <Button
                      variant="primary"
                      onClick={() => {
                        openOverlay('traces')
                        setOnboardingOpen(false)
                      }}
                    >
                      Open Traces Dashboard
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                )}

                {currentStep === 3 && (
                  <div className="space-y-4">
                    <p className="text-sm text-text-muted">
                      Configure your training run and start fine-tuning your personal coding assistant.
                      Choose from SFT, DPO, GRPO, or Knowledge Distillation strategies.
                    </p>
                    <div className="grid grid-cols-4 gap-2">
                      {['SFT', 'DPO', 'GRPO', 'KD'].map((strategy) => (
                        <div key={strategy} className="p-2 text-center rounded-lg bg-background-secondary border border-border-subtle">
                          <span className="text-sm font-medium text-text-primary">{strategy}</span>
                        </div>
                      ))}
                    </div>
                    <Button
                      variant="primary"
                      onClick={() => {
                        openOverlay('training')
                        setOnboardingOpen(false)
                      }}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Start Training
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between pt-4 border-t border-border-subtle">
          <Button
            variant="ghost"
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
          >
            Back
          </Button>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => setOnboardingOpen(false)}
            >
              Skip for now
            </Button>
            <Button
              variant="primary"
              onClick={() => {
                if (currentStep < steps.length - 1) {
                  setCurrentStep(currentStep + 1)
                } else {
                  setOnboardingOpen(false)
                }
              }}
            >
              {currentStep < steps.length - 1 ? 'Next' : 'Get Started'}
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>

        {/* Quick links */}
        <div className="flex items-center justify-center gap-6 pt-2 text-xs text-text-muted">
          <a
            href="https://github.com/anthropics/claude-code"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 hover:text-primary transition-colors"
          >
            <ExternalLink className="w-3 h-3" />
            Claude Code Docs
          </a>
          <a
            href="#"
            className="flex items-center gap-1 hover:text-primary transition-colors"
            onClick={(e) => {
              e.preventDefault()
              setOnboardingOpen(false)
              useUIStore.getState().setSettingsOpen(true)
            }}
          >
            Settings
          </a>
        </div>
      </div>
    </Modal>
  )
}
