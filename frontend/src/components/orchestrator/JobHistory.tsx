import { useEffect, useState } from 'react'
import { Clock, CheckCircle2, XCircle, Loader2, Ban, Send } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

const statusConfig: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
  completed: { icon: CheckCircle2, color: 'text-status-success', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-status-error', label: 'Failed' },
  executing: { icon: Loader2, color: 'text-status-warning', label: 'Executing' },
  decomposing: { icon: Loader2, color: 'text-status-warning', label: 'Decomposing' },
  awaiting_approval: { icon: Clock, color: 'text-status-warning', label: 'Awaiting Approval' },
  cancelled: { icon: Ban, color: 'text-text-muted', label: 'Cancelled' },
}

export function JobHistory() {
  const { jobs, fetchJobs, fetchStatus, setActiveTab } = useOrchestratorStore()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchJobs().finally(() => setLoading(false))
  }, [fetchJobs])

  const handleViewJob = async (jobId: string) => {
    await fetchStatus(jobId)
    setActiveTab('active')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-text-muted" />
      </div>
    )
  }

  if (jobs.length === 0) {
    return (
      <div className="card p-12 text-center">
        <div className="w-16 h-16 border-brutal border-border rounded-brutal bg-background-secondary flex items-center justify-center mx-auto mb-4">
          <Clock className="w-8 h-8 text-text-muted" />
        </div>
        <h3 className="font-brand text-xl text-text-primary mb-2">No orchestration jobs yet</h3>
        <p className="text-text-muted mb-4">Submit a spec to start multi-agent execution</p>
        <button
          onClick={() => setActiveTab('submit')}
          className="btn-primary"
        >
          <Send className="w-4 h-4 mr-2" />
          Submit your first spec
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Section header */}
      <div className="flex items-center gap-3 mb-1">
        <Clock className="w-5 h-5 text-accent" />
        <h2 className="font-brand text-lg text-text-primary">Job History</h2>
        <span className="tag text-[10px] py-0 px-1.5">
          <span>{jobs.length} jobs</span>
        </span>
      </div>
      <div className="section-divider mb-4" />

      {/* Table in card */}
      <div className="card p-0 overflow-hidden">
        {/* Table Header */}
        <div className="grid grid-cols-12 gap-4 px-4 py-3 bg-background-secondary border-b border-border">
          <div className="col-span-5 font-mono text-xs uppercase tracking-widest text-text-muted">Title</div>
          <div className="col-span-2 font-mono text-xs uppercase tracking-widest text-text-muted">Status</div>
          <div className="col-span-2 font-mono text-xs uppercase tracking-widest text-text-muted text-center">Tasks</div>
          <div className="col-span-3 font-mono text-xs uppercase tracking-widest text-text-muted text-right">Actions</div>
        </div>

        {/* Table Rows */}
        {jobs.map((job) => {
          const config = statusConfig[job.status] || statusConfig.cancelled
          const StatusIcon = config.icon
          return (
            <div
              key={job.jobId}
              className="grid grid-cols-12 gap-4 px-4 py-3 border-b border-border last:border-b-0 bg-background-card hover:bg-background-secondary transition-colors"
            >
              <div className="col-span-5">
                <p className="text-sm text-text-primary truncate">{job.title}</p>
                <p className="font-mono text-xs text-text-muted">{job.jobId.slice(0, 12)}...</p>
              </div>
              <div className="col-span-2 flex items-center gap-2">
                <StatusIcon className={clsx('w-4 h-4', config.color, job.status === 'executing' && 'animate-spin')} />
                <span className={clsx('font-mono text-xs', config.color)}>{config.label}</span>
              </div>
              <div className="col-span-2 flex items-center justify-center">
                <span className="font-mono text-sm text-text-secondary">{job.taskCount ?? '\u2014'}</span>
              </div>
              <div className="col-span-3 flex items-center justify-end">
                <button
                  onClick={() => handleViewJob(job.jobId)}
                  className="btn-secondary font-mono text-xs px-3 py-1.5"
                >
                  View
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
