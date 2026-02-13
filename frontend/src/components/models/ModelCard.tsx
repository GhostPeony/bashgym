import { useState } from 'react'
import {
  Star,
  MoreVertical,
  TrendingUp,
  TrendingDown,
  Clock,
  Cpu,
  Zap,
  GitBranch,
  CheckCircle,
  AlertCircle,
  Archive,
  Trash2,
  BarChart3
} from 'lucide-react'
import { clsx } from 'clsx'
import { ModelSummary, modelsApi } from '../../services/api'

interface ModelCardProps {
  model: ModelSummary
  onSelect: (modelId: string) => void
  onCompare: (modelId: string) => void
  onRefresh: () => void
}

const STATUS_STYLES: Record<string, { className: string; icon: React.ElementType }> = {
  ready: { className: 'quality-gold text-status-success', icon: CheckCircle },
  needs_eval: { className: 'quality-pending text-status-warning', icon: AlertCircle },
  training: { className: 'text-status-info', icon: Cpu },
  archived: { className: 'text-text-muted', icon: Archive },
  regression_detected: { className: 'quality-failed text-status-error', icon: AlertCircle },
}

const STRATEGY_LABELS: Record<string, string> = {
  sft: 'SFT',
  dpo: 'DPO',
  grpo: 'GRPO',
  distillation: 'Distill',
}

export function ModelCard({ model, onSelect, onCompare, onRefresh }: ModelCardProps) {
  const [showMenu, setShowMenu] = useState(false)
  const [isStarring, setIsStarring] = useState(false)

  const statusStyle = STATUS_STYLES[model.status] || STATUS_STYLES.ready
  const StatusIcon = statusStyle.icon

  const handleStar = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsStarring(true)
    await modelsApi.star(model.model_id, !model.starred)
    setIsStarring(false)
    onRefresh()
  }

  const handleArchive = async () => {
    await modelsApi.delete(model.model_id, true)
    setShowMenu(false)
    onRefresh()
  }

  // Format base model name (take last part after /)
  const baseModelShort = model.base_model.split('/').pop() || model.base_model

  return (
    <div
      className="card card-accent p-4 cursor-pointer relative group"
      onClick={() => onSelect(model.model_id)}
      data-tutorial="model-card"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-brand text-lg text-text-primary truncate">{model.display_name}</h3>
            {model.starred && (
              <Star className="w-4 h-4 text-status-warning fill-current flex-shrink-0" />
            )}
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="font-mono text-xs text-text-muted">{baseModelShort}</span>
            <span className="text-text-muted">|</span>
            <span className="tag">
              <span>{STRATEGY_LABELS[model.training_strategy] || model.training_strategy.toUpperCase()}</span>
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-150">
          <button
            onClick={handleStar}
            disabled={isStarring}
            className={clsx(
              'btn-icon w-7 h-7 flex items-center justify-center',
              model.starred ? 'text-status-warning' : 'text-text-muted'
            )}
          >
            <Star className={clsx('w-3.5 h-3.5', model.starred && 'fill-current')} />
          </button>
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu) }}
              className="btn-icon w-7 h-7 flex items-center justify-center text-text-muted"
            >
              <MoreVertical className="w-3.5 h-3.5" />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-9 z-10 bg-background-card border-brutal border-border shadow-brutal rounded-brutal py-1 min-w-[140px]">
                <button
                  onClick={(e) => { e.stopPropagation(); onCompare(model.model_id); setShowMenu(false) }}
                  className="menu-item w-full text-sm"
                >
                  <BarChart3 className="w-4 h-4" />
                  Compare
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleArchive() }}
                  className="menu-item w-full text-sm text-status-error"
                >
                  <Archive className="w-4 h-4" />
                  Archive
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {/* Custom Eval */}
        <div className="p-2 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-0.5">Custom Eval</div>
          <div className="flex items-center gap-1">
            <span className="font-brand text-xl text-text-primary">
              {model.custom_eval_pass_rate !== null
                ? `${model.custom_eval_pass_rate.toFixed(1)}%`
                : '\u2014'}
            </span>
            {model.custom_eval_pass_rate !== null && model.custom_eval_pass_rate > 0 && (
              <TrendingUp className="w-3 h-3 text-status-success" />
            )}
          </div>
        </div>

        {/* Benchmark Avg */}
        <div className="p-2 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-0.5">Benchmark</div>
          <div className="flex items-center gap-1">
            <span className="font-brand text-xl text-text-primary">
              {model.benchmark_avg_score !== null
                ? `${model.benchmark_avg_score.toFixed(1)}%`
                : '\u2014'}
            </span>
          </div>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex items-center gap-3 font-mono text-xs text-text-muted">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {model.training_duration_display || '\u2014'}
        </span>
        <span className="flex items-center gap-1">
          <Cpu className="w-3 h-3" />
          {model.model_size_display}
        </span>
        {model.inference_latency_ms && (
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3" />
            {model.inference_latency_ms.toFixed(0)}ms
          </span>
        )}
      </div>

      {/* Status & Tags */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-border-subtle">
        <div className={clsx('flex items-center gap-1 font-mono text-xs uppercase tracking-widest', statusStyle.className)}>
          <StatusIcon className="w-3 h-3" />
          {model.status.replace('_', ' ')}
        </div>
        <div className="flex items-center gap-1">
          {model.tags.slice(0, 2).map(tag => (
            <span
              key={tag}
              className="tag"
            >
              <span>{tag}</span>
            </span>
          ))}
          {model.tags.length > 2 && (
            <span className="font-mono text-xs text-text-muted">+{model.tags.length - 2}</span>
          )}
        </div>
      </div>
    </div>
  )
}
