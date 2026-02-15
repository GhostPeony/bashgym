import { useEffect } from 'react'
import { Network, Send, Activity, Clock } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { SpecForm } from './SpecForm'
import { TaskDAGView } from './TaskDAGView'
import { WorkerMonitor } from './WorkerMonitor'
import { BudgetTracker } from './BudgetTracker'
import { JobHistory } from './JobHistory'
import { clsx } from 'clsx'

const tabs = [
  { id: 'submit' as const, label: 'Submit', icon: Send },
  { id: 'active' as const, label: 'Active Job', icon: Activity },
  { id: 'history' as const, label: 'History', icon: Clock },
]

export function OrchestratorDashboard() {
  const { activeTab, setActiveTab, currentJob, fetchJobs } = useOrchestratorStore()

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  // Auto-switch to active tab if there's a running job
  useEffect(() => {
    if (currentJob && (currentJob.status === 'executing' || currentJob.status === 'decomposing' || currentJob.status === 'awaiting_approval')) {
      setActiveTab('active')
    }
  }, [currentJob?.jobId])

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center shadow-brutal">
            <Network className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h1 className="font-brand text-2xl text-text-primary">Orchestrator</h1>
            <p className="text-sm text-text-secondary">Multi-agent task decomposition & parallel execution</p>
          </div>
        </div>
        {currentJob && (
          <div className={clsx(
            'tag',
            currentJob.status === 'executing' && 'bg-status-success/20 text-status-success border-status-success',
            currentJob.status === 'completed' && 'bg-accent-light text-accent border-accent',
            currentJob.status === 'failed' && 'bg-status-error/20 text-status-error border-status-error',
            currentJob.status === 'decomposing' && 'bg-status-warning/20 text-status-warning border-status-warning',
            currentJob.status === 'awaiting_approval' && 'bg-status-warning/20 text-status-warning border-status-warning',
            currentJob.status === 'cancelled' && 'bg-background-secondary text-text-muted border-border',
          )}>
            {currentJob.status.replace('_', ' ')}
          </div>
        )}
      </div>

      {/* Tab Bar */}
      <div className="flex gap-1 mb-6 border-b border-border">
        {tabs.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2.5 font-mono text-xs uppercase tracking-wider transition-colors border-b-2 -mb-px',
                isActive
                  ? 'border-accent text-accent'
                  : 'border-transparent text-text-muted hover:text-text-secondary'
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
              {tab.id === 'active' && currentJob && (
                <span className="w-2 h-2 rounded-full bg-status-success animate-pulse" />
              )}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      {activeTab === 'submit' && <SpecForm />}

      {activeTab === 'active' && (
        currentJob ? (
          <ActiveJobView />
        ) : (
          <div className="flex flex-col items-center justify-center py-20 text-text-muted">
            <Network className="w-12 h-12 mb-4 opacity-30" />
            <p className="font-mono text-sm">No active orchestration job</p>
            <button
              onClick={() => setActiveTab('submit')}
              className="btn-secondary mt-4 font-mono text-xs"
            >
              Submit a Spec
            </button>
          </div>
        )
      )}

      {activeTab === 'history' && <JobHistory />}
    </div>
  )
}

function ActiveJobView() {
  const { currentJob, approveJob, cancelJob, fetchStatus } = useOrchestratorStore()

  useEffect(() => {
    if (!currentJob) return
    // Poll for status updates every 5 seconds
    const interval = setInterval(() => {
      fetchStatus(currentJob.jobId)
    }, 5000)
    return () => clearInterval(interval)
  }, [currentJob?.jobId, fetchStatus])

  if (!currentJob) return null

  return (
    <div className="space-y-6">
      {/* Job Header */}
      <div className="border-brutal border-border rounded-brutal bg-background-card p-4 shadow-brutal">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-brand text-lg text-text-primary">{currentJob.title}</h2>
            <p className="font-mono text-xs text-text-muted mt-1">
              Job: {currentJob.jobId.slice(0, 8)}...
            </p>
          </div>
          <div className="flex items-center gap-3">
            {currentJob.status === 'awaiting_approval' && (
              <>
                <button
                  onClick={() => approveJob(currentJob.jobId)}
                  className="btn-primary font-mono text-xs"
                >
                  Approve & Execute
                </button>
                <button
                  onClick={() => cancelJob(currentJob.jobId)}
                  className="btn-secondary font-mono text-xs"
                >
                  Reject
                </button>
              </>
            )}
            {currentJob.status === 'executing' && (
              <button
                onClick={() => cancelJob(currentJob.jobId)}
                className="btn-secondary font-mono text-xs text-status-error border-status-error"
              >
                Cancel
              </button>
            )}
          </div>
        </div>

        {/* Stats Row */}
        <div className="flex gap-4 mt-4">
          {[
            { label: 'Pending', value: currentJob.stats.pending, color: 'text-text-muted' },
            { label: 'Running', value: currentJob.stats.in_progress, color: 'text-status-warning' },
            { label: 'Completed', value: currentJob.stats.completed, color: 'text-status-success' },
            { label: 'Failed', value: currentJob.stats.failed, color: 'text-status-error' },
          ].map(stat => (
            <div key={stat.label} className="flex items-center gap-2">
              <span className={clsx('font-mono text-lg font-bold', stat.color)}>{stat.value}</span>
              <span className="font-mono text-xs text-text-muted uppercase">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Grid: Task DAG + Workers */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-8">
          <TaskDAGView />
        </div>
        <div className="col-span-4 space-y-6">
          <WorkerMonitor />
          <BudgetTracker />
        </div>
      </div>
    </div>
  )
}
