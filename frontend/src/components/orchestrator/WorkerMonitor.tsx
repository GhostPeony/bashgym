import { Cpu, Clock } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

export function WorkerMonitor() {
  const { currentJob } = useOrchestratorStore()

  if (!currentJob) return null

  const runningTasks = Object.values(currentJob.tasks).filter(t => t.status === 'running')

  return (
    <div className="border-brutal border-border rounded-brutal bg-background-card p-4 shadow-brutal">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-xs uppercase tracking-widest text-text-muted">Workers</h3>
        <span className="tag text-[10px]">
          {runningTasks.length} active
        </span>
      </div>

      {runningTasks.length === 0 ? (
        <p className="text-sm text-text-muted text-center py-4 font-mono">
          {currentJob.status === 'completed' ? 'All done' :
           currentJob.status === 'awaiting_approval' ? 'Awaiting approval' :
           'No active workers'}
        </p>
      ) : (
        <div className="space-y-3">
          {runningTasks.map((task) => (
            <div key={task.id} className="border border-border rounded-brutal p-3 bg-background-primary">
              <div className="flex items-center gap-2 mb-1">
                <Cpu className="w-3.5 h-3.5 text-status-warning" />
                <span className="text-sm text-text-primary truncate flex-1">{task.title}</span>
              </div>
              <div className="flex items-center gap-3 mt-2">
                {task.worker_id && (
                  <span className="font-mono text-[10px] text-text-muted">
                    worker: {task.worker_id.slice(0, 8)}
                  </span>
                )}
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3 text-text-muted" />
                  <span className="font-mono text-[10px] text-text-muted">running...</span>
                </div>
              </div>
              {/* Activity indicator */}
              <div className="mt-2 h-1 bg-background-secondary rounded-full overflow-hidden">
                <div className="h-full bg-status-warning rounded-full animate-pulse" style={{ width: '60%' }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
