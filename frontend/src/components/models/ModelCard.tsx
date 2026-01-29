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

const STATUS_STYLES: Record<string, { bg: string; text: string; icon: React.ElementType }> = {
  ready: { bg: 'bg-status-success/20', text: 'text-status-success', icon: CheckCircle },
  needs_eval: { bg: 'bg-status-warning/20', text: 'text-status-warning', icon: AlertCircle },
  training: { bg: 'bg-status-info/20', text: 'text-status-info', icon: Cpu },
  archived: { bg: 'bg-text-muted/20', text: 'text-text-muted', icon: Archive },
  regression_detected: { bg: 'bg-status-error/20', text: 'text-status-error', icon: AlertCircle },
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
      className="card-elevated p-4 hover:border-primary/50 transition-colors cursor-pointer relative group"
      onClick={() => onSelect(model.model_id)}
      data-tutorial="model-card"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-text-primary truncate">{model.display_name}</h3>
            {model.starred && (
              <Star className="w-4 h-4 text-yellow-500 fill-yellow-500 flex-shrink-0" />
            )}
          </div>
          <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
            <span>{baseModelShort}</span>
            <span className="text-text-muted/50">•</span>
            <span className="px-1.5 py-0.5 rounded bg-primary/20 text-primary">
              {STRATEGY_LABELS[model.training_strategy] || model.training_strategy.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleStar}
            disabled={isStarring}
            className={clsx(
              'p-1.5 rounded hover:bg-background-tertiary transition-colors',
              model.starred ? 'text-yellow-500' : 'text-text-muted'
            )}
          >
            <Star className={clsx('w-4 h-4', model.starred && 'fill-current')} />
          </button>
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu) }}
              className="p-1.5 rounded hover:bg-background-tertiary text-text-muted transition-colors"
            >
              <MoreVertical className="w-4 h-4" />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-8 z-10 bg-background-secondary border border-border-subtle rounded-lg shadow-lg py-1 min-w-[140px]">
                <button
                  onClick={(e) => { e.stopPropagation(); onCompare(model.model_id); setShowMenu(false) }}
                  className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-background-tertiary flex items-center gap-2"
                >
                  <BarChart3 className="w-4 h-4" />
                  Compare
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleArchive() }}
                  className="w-full px-3 py-2 text-left text-sm text-status-error hover:bg-background-tertiary flex items-center gap-2"
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
        <div className="p-2 bg-background-tertiary rounded">
          <div className="text-xs text-text-muted mb-0.5">Custom Eval</div>
          <div className="flex items-center gap-1">
            <span className="text-lg font-semibold text-text-primary">
              {model.custom_eval_pass_rate !== null
                ? `${model.custom_eval_pass_rate.toFixed(1)}%`
                : '—'}
            </span>
            {model.custom_eval_pass_rate !== null && model.custom_eval_pass_rate > 0 && (
              <TrendingUp className="w-3 h-3 text-status-success" />
            )}
          </div>
        </div>

        {/* Benchmark Avg */}
        <div className="p-2 bg-background-tertiary rounded">
          <div className="text-xs text-text-muted mb-0.5">Benchmark</div>
          <div className="flex items-center gap-1">
            <span className="text-lg font-semibold text-text-primary">
              {model.benchmark_avg_score !== null
                ? `${model.benchmark_avg_score.toFixed(1)}%`
                : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex items-center gap-3 text-xs text-text-muted">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {model.training_duration_display || '—'}
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
        <div className={clsx('flex items-center gap-1 px-2 py-0.5 rounded text-xs', statusStyle.bg, statusStyle.text)}>
          <StatusIcon className="w-3 h-3" />
          {model.status.replace('_', ' ')}
        </div>
        <div className="flex items-center gap-1">
          {model.tags.slice(0, 2).map(tag => (
            <span
              key={tag}
              className="px-1.5 py-0.5 text-xs rounded bg-background-tertiary text-text-muted"
            >
              {tag}
            </span>
          ))}
          {model.tags.length > 2 && (
            <span className="text-xs text-text-muted">+{model.tags.length - 2}</span>
          )}
        </div>
      </div>
    </div>
  )
}
