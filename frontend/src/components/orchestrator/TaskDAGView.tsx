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
  FileCode,
  Terminal,
  RotateCcw
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
  dispatched: { icon: Terminal, color: 'text-accent', bg: 'border-accent' },
  blocked: { icon: Lock, color: 'text-text-muted', bg: 'border-border' },
  completed: { icon: CheckCircle2, color: 'text-status-success', bg: 'border-status-success' },
  failed: { icon: XCircle, color: 'text-status-error', bg: 'border-status-error' },
  retrying: { icon: RefreshCw, color: 'text-status-warning', bg: 'border-status-warning' },
  cancelled: { icon: XCircle, color: 'text-text-muted', bg: 'border-border' },
}

const statusOrder = ['running', 'dispatched', 'pending', 'assigned', 'blocked', 'retrying', 'completed', 'failed', 'cancelled']

function shortTerminalId(terminalId: string): string {
  // Extract a short display name: "terminal-3" → "T3", UUID → last 4 chars
  const match = terminalId.match(/(\d+)$/)
  if (match) return `T${match[1]}`
  return terminalId.slice(-4).toUpperCase()
}

export function TaskDAGView() {
  const {
    currentJob,
    retryTask,
    taskAssignments,
    taskQueue,
    editedPrompts,
    setEditedPrompt,
    resetEditedPrompt,
  } = useOrchestratorStore()
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const [retryPrompt, setRetryPrompt] = useState('')
  const [retryingTask, setRetryingTask] = useState<string | null>(null)

  if (!currentJob) return null

  const tasks = Object.values(currentJob.tasks)
  const isAwaitingApproval = currentJob.status === 'awaiting_approval'

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
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted">Task Graph</h3>
        </div>
        <div className="section-divider mb-3" />
        <div className="flex items-center justify-center py-8 text-text-muted">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          <span className="font-mono text-sm">Decomposing spec into tasks...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="card p-4">
      {/* Panel header */}
      <div className="flex items-center gap-2 mb-3">
        <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted">Task Graph</h3>
        <span className="tag text-[10px] py-0 px-1.5">
          <span>{tasks.length} tasks</span>
        </span>
        {isAwaitingApproval && (
          <span className="tag text-[10px] py-0 px-1.5 border-status-warning/60 bg-status-warning/10 text-status-warning">
            <span>Edit prompts before dispatching</span>
          </span>
        )}
      </div>
      <div className="section-divider mb-3" />

      {/* Scrollable task list */}
      <div className="space-y-2 max-h-[calc(100vh-320px)] overflow-y-auto">
        {sortedTasks.map((task) => {
          const config = statusConfig[task.status] || statusConfig.pending
          const StatusIcon = config.icon
          const isExpanded = expandedTask === task.id
          const assignedTerminalId = taskAssignments[task.id]
          const isQueued = taskQueue.includes(task.id)
          const currentPrompt = editedPrompts[task.id] ?? task.worker_prompt ?? task.description
          const hasEdits = !!editedPrompts[task.id]

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
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left"
              >
                <StatusIcon
                  className={clsx(
                    'w-4 h-4 flex-shrink-0',
                    config.color,
                    (task.status === 'running' || task.status === 'retrying') && 'animate-spin'
                  )}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-text-primary truncate">{task.title}</span>
                    <span className={clsx('tag text-[10px] py-0 px-1.5', priorityColors[task.priority])}>
                      <span>{task.priority}</span>
                    </span>
                    {hasEdits && (
                      <span className="tag text-[10px] py-0 px-1.5 border-accent/60 bg-accent/10 text-accent">
                        <span>edited</span>
                      </span>
                    )}
                  </div>
                  {task.dependencies.length > 0 && (
                    <span className="font-mono text-[10px] text-text-muted">
                      depends on: {task.dependencies.join(', ')}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {/* Terminal assignment badge */}
                  {assignedTerminalId && (
                    <span className="font-mono text-[10px] border border-accent bg-accent/10 text-accent px-1.5 py-0.5 rounded-brutal flex items-center gap-1">
                      <Terminal className="w-2.5 h-2.5" />
                      {shortTerminalId(assignedTerminalId)}
                    </span>
                  )}
                  {isQueued && !assignedTerminalId && (
                    <span className="font-mono text-[10px] border border-border bg-background-secondary text-text-muted px-1.5 py-0.5 rounded-brutal">
                      queued
                    </span>
                  )}
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
                <div className="px-3 pb-3 border-t border-border">
                  <p className="text-sm text-text-secondary mt-3 mb-3">{task.description}</p>

                  {/* Editable prompt — shown when awaiting approval */}
                  {isAwaitingApproval && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
                          Worker Prompt
                        </span>
                        {hasEdits && (
                          <button
                            onClick={() => resetEditedPrompt(task.id)}
                            className="flex items-center gap-1 font-mono text-[10px] text-text-muted hover:text-accent transition-colors"
                          >
                            <RotateCcw className="w-2.5 h-2.5" />
                            Reset to original
                          </button>
                        )}
                      </div>
                      <textarea
                        value={currentPrompt}
                        onChange={(e) => setEditedPrompt(task.id, e.target.value)}
                        rows={4}
                        className="input w-full text-xs font-mono resize-y"
                        placeholder="Prompt that will be sent to the Claude agent..."
                      />
                    </div>
                  )}

                  {/* Worker prompt read-only after dispatch */}
                  {!isAwaitingApproval && task.worker_prompt && (
                    <div className="mb-3 p-2 bg-background-secondary rounded-brutal border border-border">
                      <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted block mb-1">Prompt</span>
                      <p className="font-mono text-[11px] text-text-secondary leading-relaxed">{task.worker_prompt}</p>
                    </div>
                  )}

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
                        className="input w-full text-xs font-mono resize-none"
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
