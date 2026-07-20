import { useState, useEffect } from 'react'
import {
  Eye, Filter, Sparkles, GraduationCap,
  Play, ArrowRight, RefreshCw,
  CheckCircle2, XCircle, Clock, Layers
} from 'lucide-react'
import { pipelineApi, PipelineConfig } from '../../services/api'
import { pipelineOverviewResource } from '../../stores/opsResources'
import { useSessionResource } from '../../stores/sessionResource'
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
  const { data, refreshing } = useSessionResource(pipelineOverviewResource)
  const status = data?.status ?? null
  const config = data?.config ?? null
  const [triggeringStage, setTriggeringStage] = useState<string | null>(null)

  // Poll every 5s while mounted; refreshes land in the shared cache
  useEffect(() => {
    const interval = setInterval(() => {
      void pipelineOverviewResource.getState().refresh()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const updateConfig = async (updates: Partial<PipelineConfig>) => {
    const result = await pipelineApi.updateConfig(updates)
    const { data: current, setData } = pipelineOverviewResource.getState()
    if (result.ok && result.data && current) {
      setData({ ...current, config: result.data })
    }
  }

  const triggerStage = async (stage: 'import' | 'classify' | 'cascade') => {
    setTriggeringStage(stage)
    await pipelineApi.triggerStage(stage)
    await pipelineOverviewResource.getState().refresh()
    setTriggeringStage(null)
  }

  if (!config || !status) {
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
              onClick={() => void pipelineOverviewResource.getState().refresh()}
              className="p-2 border-brutal rounded-brutal hover-press transition-press shadow-brutal-sm"
            >
              <RefreshCw className={clsx('w-4 h-4', refreshing && 'animate-spin')} />
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

          <StageCard
            title="Cascade RL"
            icon={<Layers className="w-5 h-5" />}
            enabled={config.cascade_enabled}
            onToggle={(v) => updateConfig({ cascade_enabled: v })}
            count={status.gold_count}
            threshold={config.cascade_gold_threshold}
            onThresholdChange={(v) => updateConfig({ cascade_gold_threshold: v })}
            description="Auto-trigger a domain Cascade RL run when new gold traces reach threshold"
            color="text-violet-500"
            showThreshold
            triggerable
            onTrigger={() => triggerStage('cascade')}
            isTriggering={triggeringStage === 'cascade'}
          />
        </div>

        {config.cascade_enabled && (
          <div className="mt-6 border-brutal rounded-brutal p-4 bg-background-card shadow-brutal-sm">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="font-brand text-lg text-text-primary">Cascade trigger recipe</h3>
                <p className="text-xs text-text-muted">
                  The watcher queues this request when gold traces cross the Cascade RL threshold.
                </p>
              </div>
              {status.cascade_auto_trigger ? (
                <span className="font-mono text-xs uppercase tracking-widest text-accent">
                  Last trigger recorded
                </span>
              ) : null}
            </div>

            <div className="grid grid-cols-4 gap-3">
              <div className="col-span-2">
                <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                  Base model
                </label>
                <input
                  className="input w-full text-sm font-mono"
                  value={config.cascade_base_model}
                  onChange={(e) => updateConfig({ cascade_base_model: e.target.value })}
                  placeholder="Qwen/Qwen3-8B"
                />
              </div>
              <div>
                <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                  Mode
                </label>
                <select
                  className="input w-full text-sm"
                  value={config.cascade_mode}
                  onChange={(e) =>
                    updateConfig({ cascade_mode: e.target.value as PipelineConfig['cascade_mode'] })
                  }
                >
                  <option value="simulate">Simulate</option>
                  <option value="real">Real</option>
                </select>
              </div>
              <div>
                <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                  Stage steps
                </label>
                <input
                  type="number"
                  min={10}
                  max={5000}
                  className="input w-full text-sm font-mono"
                  value={config.cascade_train_steps_per_stage}
                  onChange={(e) =>
                    updateConfig({ cascade_train_steps_per_stage: parseInt(e.target.value) || 10 })
                  }
                />
              </div>
              <div>
                <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                  Min examples
                </label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  className="input w-full text-sm font-mono"
                  value={config.cascade_min_domain_examples}
                  onChange={(e) =>
                    updateConfig({ cascade_min_domain_examples: parseInt(e.target.value) || 1 })
                  }
                />
              </div>
              <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
                <input
                  type="checkbox"
                  checked={config.cascade_use_remote_ssh}
                  onChange={(e) => updateConfig({ cascade_use_remote_ssh: e.target.checked })}
                />
                Remote SSH
              </label>
              <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
                <input
                  type="checkbox"
                  checked={config.cascade_repo_domains_enabled}
                  onChange={(e) => updateConfig({ cascade_repo_domains_enabled: e.target.checked })}
                />
                Repo domains
              </label>
            </div>
          </div>
        )}

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
