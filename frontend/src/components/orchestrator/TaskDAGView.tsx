import { useState } from 'react'
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Lock,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  FileCode
} from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

const priorityColors: Record<string, string> = {
  CRITICAL: 'bg-status-error/20 text-status-error border-status-error',
  HIGH: 'bg-status-warning/20 text-status-warning border-status-warning',
  NORMAL: 'bg-accent-light text-accent border-accent',
  LOW: 'bg-background-secondary text-text-muted border-border',
}

const statusConfig: Record<string, { icon: typeof CheckCircle2; color: string; bg: string }> = {
  running: { icon: Loader2, color: 'text-status-warning', bg: 'border-status-warning' },
  pending: { icon: Clock, color: 'text-text-muted', bg: 'border-border' },
  assigned: { icon: Clock, color: 'text-accent', bg: 'border-accent' },
  blocked: { icon: Lock, color: 'text-text-muted', bg: 'border-border' },
  completed: { icon: CheckCircle2, color: 'text-status-success', bg: 'border-status-success' },
  failed: { icon: XCircle, color: 'text-status-error', bg: 'border-status-error' },
  cancelled: { icon: XCircle, color: 'text-text-muted', bg: 'border-border' },
}

const statusOrder = ['running', 'pending', 'assigned', 'blocked', 'completed', 'failed', 'cancelled']

export function TaskDAGView() {
  const { currentJob, retryTask } = useOrchestratorStore()
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const [retryPrompt, setRetryPrompt] = useState('')
  const [retryingTask, setRetryingTask] = useState<string | null>(null)

  if (!currentJob) return null

  const tasks = Object.values(currentJob.tasks)

  // Group and sort by status, then by priority within each status
  const sortedTasks = [...tasks].sort((a, b) => {
    const aOrder = statusOrder.indexOf(a.status)
    const bOrder = statusOrder.indexOf(b.status)
    if (aOrder !== bOrder) return aOrder - bOrder
    const priorityOrder = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']
    return priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority)
  })

  const handleRetry = async (taskId: string) => {
    setRetryingTask(taskId)
    try {
      await retryTask(currentJob.jobId, taskId, retryPrompt || undefined)
      setRetryPrompt('')
      setExpandedTask(null)
    } finally {
      setRetryingTask(null)
    }
  }

  if (tasks.length === 0) {
    return (
      <div className="border-brutal border-border rounded-brutal bg-background-card p-6">
        <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted mb-4">Task Graph</h3>
        <div className="flex items-center justify-center py-8 text-text-muted">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          <span className="font-mono text-sm">Decomposing spec into tasks...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="border-brutal border-border rounded-brutal bg-background-card p-4 shadow-brutal">
      <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted mb-4">
        Task Graph &mdash; {tasks.length} tasks
      </h3>

      <div className="space-y-3">
        {sortedTasks.map((task) => {
          const config = statusConfig[task.status] || statusConfig.pending
          const StatusIcon = config.icon
          const isExpanded = expandedTask === task.id

          return (
            <div
              key={task.id}
              className={clsx(
                'border-brutal rounded-brutal bg-background-primary transition-colors',
                config.bg
              )}
            >
              {/* Task Card Header */}
              <button
                onClick={() => setExpandedTask(isExpanded ? null : task.id)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left"
              >
                <StatusIcon
                  className={clsx(
                    'w-4 h-4 flex-shrink-0',
                    config.color,
                    task.status === 'running' && 'animate-spin'
                  )}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-text-primary truncate">{task.title}</span>
                    <span className={clsx('tag text-[10px] py-0 px-1.5', priorityColors[task.priority])}>
                      {task.priority}
                    </span>
                  </div>
                  {task.dependencies.length > 0 && (
                    <span className="font-mono text-[10px] text-text-muted">
                      depends on: {task.dependencies.join(', ')}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {task.cost_usd !== undefined && (
                    <span className="font-mono text-xs text-text-secondary">${task.cost_usd.toFixed(3)}</span>
                  )}
                  {task.duration_seconds !== undefined && (
                    <span className="font-mono text-xs text-text-muted">{Math.round(task.duration_seconds)}s</span>
                  )}
                  {isExpanded
                    ? <ChevronUp className="w-4 h-4 text-text-muted" />
                    : <ChevronDown className="w-4 h-4 text-text-muted" />
                  }
                </div>
              </button>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="px-4 pb-4 border-t border-border">
                  <p className="text-sm text-text-secondary mt-3 mb-3">{task.description}</p>

                  {task.files_touched.length > 0 && (
                    <div className="mb-3">
                      <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Files:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {task.files_touched.map((f, i) => (
                          <span
                            key={i}
                            className="inline-flex items-center gap-1 font-mono text-[11px] bg-background-secondary px-2 py-0.5 rounded-brutal border border-border"
                          >
                            <FileCode className="w-3 h-3" />
                            {f}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-4 font-mono text-xs text-text-muted">
                    <span>Budget: ${task.budget_usd.toFixed(2)}</span>
                    <span>Est. turns: {task.estimated_turns}</span>
                    {task.retry_count > 0 && <span>Retries: {task.retry_count}</span>}
                  </div>

                  {task.error && (
                    <div className="mt-3 p-3 bg-status-error/10 border border-status-error rounded-brutal">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 text-status-error flex-shrink-0 mt-0.5" />
                        <p className="font-mono text-xs text-status-error">{task.error}</p>
                      </div>
                    </div>
                  )}

                  {task.status === 'failed' && (
                    <div className="mt-3 space-y-2">
                      <textarea
                        value={retryPrompt}
                        onChange={(e) => setRetryPrompt(e.target.value)}
                        placeholder="Optional: modify the prompt for retry..."
                        rows={2}
                        className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2 text-xs font-mono text-text-primary focus:outline-none focus:border-accent resize-none"
                      />
                      <button
                        onClick={() => handleRetry(task.id)}
                        disabled={retryingTask === task.id}
                        className="btn-secondary font-mono text-xs flex items-center gap-1.5"
                      >
                        <RefreshCw className={clsx('w-3 h-3', retryingTask === task.id && 'animate-spin')} />
                        {retryingTask === task.id ? 'Retrying...' : 'Retry Task'}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
