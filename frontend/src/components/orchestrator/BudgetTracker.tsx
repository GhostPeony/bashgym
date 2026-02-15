import { DollarSign, AlertTriangle, Clock, GitMerge } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

export function BudgetTracker() {
  const { currentJob } = useOrchestratorStore()

  if (!currentJob) return null

  const { budget, totalCost, totalTime, mergeSuccesses, mergeFailures } = currentJob
  const percentUsed = budget.limit_usd > 0 ? (budget.spent_usd / budget.limit_usd) * 100 : 0
  const isWarning = percentUsed > 80
  const isExceeded = budget.exceeded

  // Per-task cost breakdown sorted by highest cost first
  const taskCosts = Object.values(currentJob.tasks)
    .filter(t => t.cost_usd !== undefined && t.cost_usd > 0)
    .sort((a, b) => (b.cost_usd || 0) - (a.cost_usd || 0))

  return (
    <div className="border-brutal border-border rounded-brutal bg-background-card p-4 shadow-brutal">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted">Budget</h3>
        {isWarning && !isExceeded && (
          <AlertTriangle className="w-4 h-4 text-status-warning" />
        )}
        {isExceeded && (
          <span className="tag text-[10px] bg-status-error/20 text-status-error border-status-error">EXCEEDED</span>
        )}
      </div>

      {/* Progress Bar */}
      <div className="mb-3">
        <div className="flex justify-between mb-1">
          <span className="font-mono text-xs text-text-secondary">
            ${budget.spent_usd.toFixed(2)} / ${budget.limit_usd.toFixed(2)}
          </span>
          <span className={clsx(
            'font-mono text-xs',
            isExceeded ? 'text-status-error' : isWarning ? 'text-status-warning' : 'text-text-muted'
          )}>
            {percentUsed.toFixed(0)}%
          </span>
        </div>
        <div className="h-2.5 bg-background-secondary rounded-brutal overflow-hidden border border-border">
          <div
            className={clsx(
              'h-full rounded-brutal transition-all duration-500',
              isExceeded ? 'bg-status-error' : isWarning ? 'bg-status-warning' : 'bg-accent'
            )}
            style={{ width: `${Math.min(percentUsed, 100)}%` }}
          />
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="flex items-center gap-2">
          <DollarSign className="w-3.5 h-3.5 text-text-muted" />
          <div>
            <span className="font-mono text-xs text-text-muted block">Total Cost</span>
            <span className="font-mono text-sm text-text-primary">${totalCost.toFixed(3)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="w-3.5 h-3.5 text-text-muted" />
          <div>
            <span className="font-mono text-xs text-text-muted block">Elapsed</span>
            <span className="font-mono text-sm text-text-primary">
              {totalTime > 60
                ? `${Math.floor(totalTime / 60)}m ${Math.round(totalTime % 60)}s`
                : `${Math.round(totalTime)}s`
              }
            </span>
          </div>
        </div>
      </div>

      {/* Merge Stats */}
      {(mergeSuccesses > 0 || mergeFailures > 0) && (
        <div className="flex items-center gap-3 mb-3 pt-3 border-t border-border">
          <GitMerge className="w-3.5 h-3.5 text-text-muted" />
          <span className="font-mono text-xs text-text-muted">Merges:</span>
          <span className="font-mono text-xs text-status-success">{mergeSuccesses} ok</span>
          {mergeFailures > 0 && (
            <span className="font-mono text-xs text-status-error">{mergeFailures} failed</span>
          )}
        </div>
      )}

      {/* Per-task Cost Breakdown */}
      {taskCosts.length > 0 && (
        <div className="pt-3 border-t border-border">
          <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted block mb-2">
            Cost Breakdown
          </span>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {taskCosts.map((task) => (
              <div key={task.id} className="flex items-center justify-between">
                <span className="text-xs text-text-secondary truncate flex-1 mr-2">{task.title}</span>
                <span className="font-mono text-xs text-text-primary flex-shrink-0">
                  ${(task.cost_usd || 0).toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
