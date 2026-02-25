import { useState, useEffect, useCallback } from 'react'
import {
  Eye, Filter, Sparkles, GraduationCap,
  Play, ArrowRight, RefreshCw,
  CheckCircle2, XCircle, Clock
} from 'lucide-react'
import { pipelineApi, PipelineConfig, PipelineStatus } from '../../services/api'
import { clsx } from 'clsx'

interface StageCardProps {
  title: string
  icon: React.ReactNode
  enabled: boolean
  onToggle: (enabled: boolean) => void
  count: number
  threshold?: number
  onThresholdChange?: (value: number) => void
  description: string
  color: string // accent class like 'text-emerald-500'
  showThreshold: boolean
  triggerable?: boolean
  onTrigger?: () => void
  isTriggering?: boolean
}

function StageCard({ title, icon, enabled, onToggle, count, threshold, onThresholdChange, description, color, showThreshold, triggerable, onTrigger, isTriggering }: StageCardProps) {
  return (
    <div className={clsx(
      'flex-1 border-brutal rounded-brutal p-4 transition-all',
      enabled ? 'bg-background-card shadow-brutal' : 'bg-background-primary opacity-60'
    )}>
      {/* Header with toggle */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={color}>{icon}</span>
          <h3 className="font-mono text-sm font-semibold uppercase tracking-wider">{title}</h3>
        </div>
        <button
          onClick={() => onToggle(!enabled)}
          className={clsx(
            'w-10 h-5 rounded-full border-2 border-border relative transition-colors',
            enabled ? 'bg-accent' : 'bg-background-primary'
          )}
        >
          <span className={clsx(
            'absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white border border-border transition-all',
            enabled ? 'left-5' : 'left-0.5'
          )} />
        </button>
      </div>

      {/* Description */}
      <p className="text-xs text-text-muted mb-3">{description}</p>

      {/* Count display */}
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-2xl font-bold text-text-primary">{count}</span>
        {showThreshold && threshold !== undefined && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-text-muted">/ </span>
            <input
              type="number"
              value={threshold}
              onChange={(e) => onThresholdChange?.(parseInt(e.target.value) || 0)}
              className="w-16 font-mono text-sm bg-background-primary border border-border rounded-brutal px-2 py-0.5 text-text-primary"
              min={1}
            />
          </div>
        )}
      </div>

      {/* Progress bar for threshold stages */}
      {showThreshold && threshold !== undefined && threshold > 0 && (
        <div className="w-full h-1.5 bg-background-primary rounded-full overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all"
            style={{ width: `${Math.min((count / threshold) * 100, 100)}%` }}
          />
        </div>
      )}

      {/* Manual trigger button */}
      {triggerable && (
        <button
          onClick={onTrigger}
          disabled={isTriggering || !enabled}
          className={clsx(
            'mt-3 w-full flex items-center justify-center gap-2 px-3 py-1.5 font-mono text-xs uppercase tracking-wider border-brutal rounded-brutal transition-all',
            enabled && !isTriggering
              ? 'bg-accent text-white hover-press shadow-brutal-sm'
              : 'bg-background-primary text-text-muted cursor-not-allowed'
          )}
        >
          {isTriggering ? (
            <RefreshCw className="w-3 h-3 animate-spin" />
          ) : (
            <Play className="w-3 h-3" />
          )}
          Run Now
        </button>
      )}
    </div>
  )
}

export function PipelineDashboard() {
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [config, setConfig] = useState<PipelineConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [triggeringStage, setTriggeringStage] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    const [statusRes, configRes] = await Promise.all([
      pipelineApi.getStatus(),
      pipelineApi.getConfig(),
    ])
    if (statusRes.ok && statusRes.data) setStatus(statusRes.data)
    if (configRes.ok && configRes.data) setConfig(configRes.data)
    setIsLoading(false)
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  const updateConfig = async (updates: Partial<PipelineConfig>) => {
    const result = await pipelineApi.updateConfig(updates)
    if (result.ok && result.data) setConfig(result.data)
  }

  const triggerStage = async (stage: 'import' | 'classify') => {
    setTriggeringStage(stage)
    await pipelineApi.triggerStage(stage)
    await fetchData()
    setTriggeringStage(null)
  }

  if (isLoading || !config || !status) {
    return (
      <div className="h-full flex items-center justify-center">
        <RefreshCw className="w-6 h-6 text-text-muted animate-spin" />
      </div>
    )
  }

  return (
    <div className="h-full p-6 overflow-auto">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-brand font-semibold text-text-primary">Auto-Import Pipeline</h1>
            <p className="text-sm text-text-secondary mt-1">
              Watch → Classify → Generate → Train
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Watcher status indicator */}
            <div className={clsx(
              'flex items-center gap-2 px-3 py-1.5 border-brutal rounded-brutal font-mono text-xs uppercase tracking-wider',
              status.watcher_running
                ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
                : 'bg-background-primary text-text-muted'
            )}>
              <span className={clsx(
                'w-2 h-2 rounded-full',
                status.watcher_running ? 'bg-emerald-500 animate-pulse' : 'bg-text-muted'
              )} />
              {status.watcher_running ? 'Watching' : 'Stopped'}
            </div>
            <button
              onClick={fetchData}
              className="p-2 border-brutal rounded-brutal hover-press transition-press shadow-brutal-sm"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Pipeline Flow - 4 stages with arrows */}
        <div className="flex items-stretch gap-2">
          <StageCard
            title="Watch"
            icon={<Eye className="w-5 h-5" />}
            enabled={config.watch_enabled}
            onToggle={(v) => updateConfig({ watch_enabled: v })}
            count={status.gold_count + status.pending_count + status.failed_count}
            description={`Watches ~/.claude/projects/ for new sessions. Debounce: ${config.watch_debounce_seconds}s`}
            color="text-sky-500"
            showThreshold={false}
            triggerable
            onTrigger={() => triggerStage('import')}
            isTriggering={triggeringStage === 'import'}
          />

          <div className="flex items-center text-text-muted">
            <ArrowRight className="w-5 h-5" />
          </div>

          <StageCard
            title="Classify"
            icon={<Filter className="w-5 h-5" />}
            enabled={config.classify_enabled}
            onToggle={(v) => updateConfig({ classify_enabled: v })}
            count={status.gold_count}
            description={`Gold: ≥${config.classify_gold_min_success_rate * 100}% success + ≥${config.classify_gold_min_steps} steps`}
            color="text-amber-500"
            showThreshold={false}
            triggerable
            onTrigger={() => triggerStage('classify')}
            isTriggering={triggeringStage === 'classify'}
          />

          <div className="flex items-center text-text-muted">
            <ArrowRight className="w-5 h-5" />
          </div>

          <StageCard
            title="Generate"
            icon={<Sparkles className="w-5 h-5" />}
            enabled={config.generate_enabled}
            onToggle={(v) => updateConfig({ generate_enabled: v })}
            count={status.gold_count}
            threshold={config.generate_gold_threshold}
            onThresholdChange={(v) => updateConfig({ generate_gold_threshold: v })}
            description="Auto-generate training examples when gold traces reach threshold"
            color="text-violet-500"
            showThreshold
          />

          <div className="flex items-center text-text-muted">
            <ArrowRight className="w-5 h-5" />
          </div>

          <StageCard
            title="Train"
            icon={<GraduationCap className="w-5 h-5" />}
            enabled={config.train_enabled}
            onToggle={(v) => updateConfig({ train_enabled: v })}
            count={0}
            threshold={config.train_examples_threshold}
            onThresholdChange={(v) => updateConfig({ train_examples_threshold: v })}
            description="Auto-start training run when generated examples reach threshold"
            color="text-rose-500"
            showThreshold
          />
        </div>

        {/* Stats summary */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          <div className="border-brutal rounded-brutal p-4 bg-background-card shadow-brutal-sm">
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              <span className="font-mono text-xs uppercase tracking-wider text-text-muted">Gold Traces</span>
            </div>
            <span className="font-mono text-3xl font-bold text-text-primary">{status.gold_count}</span>
          </div>
          <div className="border-brutal rounded-brutal p-4 bg-background-card shadow-brutal-sm">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-4 h-4 text-amber-500" />
              <span className="font-mono text-xs uppercase tracking-wider text-text-muted">Pending Review</span>
            </div>
            <span className="font-mono text-3xl font-bold text-text-primary">{status.pending_count}</span>
          </div>
          <div className="border-brutal rounded-brutal p-4 bg-background-card shadow-brutal-sm">
            <div className="flex items-center gap-2 mb-1">
              <XCircle className="w-4 h-4 text-red-500" />
              <span className="font-mono text-xs uppercase tracking-wider text-text-muted">Failed</span>
            </div>
            <span className="font-mono text-3xl font-bold text-text-primary">{status.failed_count}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
