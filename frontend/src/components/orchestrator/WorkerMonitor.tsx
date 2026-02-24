import { Terminal, Clock } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

function shortTerminalId(terminalId: string): string {
  const match = terminalId.match(/(\d+)$/)
  if (match) return `Terminal ${match[1]}`
  return terminalId.slice(-6).toUpperCase()
}

export function WorkerMonitor() {
  const { currentJob, taskAssignments, taskQueue } = useOrchestratorStore()

  if (!currentJob) return null

  const assignedTasks = Object.values(currentJob.tasks).filter(
    t => taskAssignments[t.id] && t.status !== 'completed' && t.status !== 'failed' && t.status !== 'cancelled'
  )
  const queuedCount = taskQueue.length

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted">Terminals</h3>
        <div className="flex items-center gap-2">
          <span className="tag text-[10px]">
            <span>{assignedTasks.length} active</span>
          </span>
          {queuedCount > 0 && (
            <span className="tag text-[10px] border-border text-text-muted">
              <span>{queuedCount} queued</span>
            </span>
          )}
        </div>
      </div>

      {assignedTasks.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-4 font-mono">
          {currentJob.status === 'completed' ? 'All done' :
           currentJob.status === 'awaiting_approval' ? 'Awaiting dispatch' :
           'No active terminals'}
        </p>
      ) : (
        <div className="space-y-3">
          {assignedTasks.map((task) => {
            const terminalId = taskAssignments[task.id]
            return (
              <div key={task.id} className="border border-border rounded-brutal p-3 bg-background-primary">
                <div className="flex items-center gap-2 mb-1">
                  <Terminal className="w-3.5 h-3.5 text-status-warning" />
                  <span className="text-sm text-text-primary truncate flex-1">{task.title}</span>
                </div>
                <div className="flex items-center gap-3 mt-2">
                  {terminalId && (
                    <span className="font-mono text-[10px] text-accent">
                      {shortTerminalId(terminalId)}
                    </span>
                  )}
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-text-muted" />
                    <span className="font-mono text-[10px] text-text-muted">running...</span>
                  </div>
                </div>
                <div className="mt-2 h-1 bg-background-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-status-warning rounded-full animate-pulse" style={{ width: '60%' }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {queuedCount > 0 && (
        <div className="mt-3 pt-3 border-t border-border">
          <p className="font-mono text-[10px] text-text-muted">
            {queuedCount} task{queuedCount !== 1 ? 's' : ''} waiting for an idle terminal
          </p>
        </div>
      )}
    </div>
  )
}
