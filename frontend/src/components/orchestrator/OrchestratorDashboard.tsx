import { useEffect, useRef } from 'react'
import { Network, Send, Activity, Clock, Loader2 } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { useTerminalStore } from '../../stores/terminalStore'
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
  const { activeTab, setActiveTab, currentJob, jobs, fetchJobs } = useOrchestratorStore()

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  // Auto-switch to active tab if there's a running job
  useEffect(() => {
    if (currentJob && (
      currentJob.status === 'dispatched' ||
      currentJob.status === 'decomposing' ||
      currentJob.status === 'awaiting_approval'
    )) {
      setActiveTab('active')
    }
  }, [currentJob?.jobId])

  // Queue watcher: dispatch next queued task when a terminal becomes idle
  const prevStatusesRef = useRef<Map<string, string>>(new Map())
  useEffect(() => {
    const unsub = useTerminalStore.subscribe((state, prevState) => {
      state.sessions.forEach((session, terminalId) => {
        const prevSession = prevState.sessions.get(terminalId)
        if (session.status === 'idle' && prevSession?.status !== 'idle') {
          const { taskQueue, dispatchNextQueued } = useOrchestratorStore.getState()
          if (taskQueue.length > 0) {
            dispatchNextQueued(terminalId)
          }
        }
      })
    })
    return unsub
  }, [])

  return (
    <div className="h-full flex flex-col bg-background-primary">
      {/* Fixed Header */}
      <div className="p-6 border-b-2 border-border bg-background-card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center shadow-brutal">
              <Network className="w-5 h-5 text-accent" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="font-brand text-2xl text-text-primary">Orchestrator</h1>
                <span className="tag"><span>MULTI-AGENT</span></span>
              </div>
              <p className="text-sm text-text-secondary mt-1">
                Decompose specs into tasks & dispatch to terminal agents
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Status tag */}
            {currentJob && (
              <span className={clsx(
                'tag',
                currentJob.status === 'dispatched' && 'bg-status-success/20 text-status-success border-status-success',
                currentJob.status === 'completed' && 'bg-accent-light text-accent border-accent',
                currentJob.status === 'failed' && 'bg-status-error/20 text-status-error border-status-error',
                currentJob.status === 'decomposing' && 'bg-status-warning/20 text-status-warning border-status-warning',
                currentJob.status === 'awaiting_approval' && 'bg-status-warning/20 text-status-warning border-status-warning',
                currentJob.status === 'cancelled' && 'bg-background-secondary text-text-muted border-border',
              )}>
                <span>{currentJob.status.replace(/_/g, ' ')}</span>
              </span>
            )}
            {/* Contextual buttons for active tab */}
            {activeTab === 'active' && currentJob && (
              <>
                {currentJob.status === 'awaiting_approval' && (
                  <ActiveJobActions />
                )}
                {currentJob.status === 'dispatched' && (
                  <CancelJobButton />
                )}
              </>
            )}
          </div>
        </div>

        {/* Brutalist tab bar */}
        <div className="flex gap-1 mt-6">
          {tabs.map(tab => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            const historyCount = tab.id === 'history' && jobs.length > 0 ? jobs.length : undefined
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 text-sm font-mono font-semibold tracking-wide uppercase transition-all whitespace-nowrap border-brutal',
                  'rounded-brutal',
                  isActive
                    ? 'bg-accent-light text-accent-dark border-border shadow-brutal-sm'
                    : 'bg-transparent text-text-secondary border-transparent hover:text-text-primary hover:border-border hover:bg-background-secondary'
                )}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
                {tab.id === 'active' && currentJob && (
                  <span className="w-2 h-2 rounded-full bg-status-success animate-pulse" />
                )}
                {historyCount !== undefined && (
                  <span className="tag text-[10px] py-0 px-1.5">
                    <span>{historyCount}</span>
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'submit' && (
          <div className="p-6 max-w-3xl mx-auto">
            <SpecForm />
          </div>
        )}

        {activeTab === 'active' && (
          currentJob ? (
            <ActiveJobView />
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-text-muted">
              <div className="w-16 h-16 border-brutal border-border rounded-brutal bg-background-secondary flex items-center justify-center mx-auto mb-4">
                <Network className="w-8 h-8 text-text-muted" />
              </div>
              <h3 className="font-brand text-xl text-text-primary mb-2">No active orchestration job</h3>
              <p className="text-text-muted mb-4">Submit a spec to start multi-agent execution</p>
              <button
                onClick={() => setActiveTab('submit')}
                className="btn-primary"
              >
                Submit a Spec
              </button>
            </div>
          )
        )}

        {activeTab === 'history' && (
          <div className="p-6 max-w-6xl mx-auto">
            <JobHistory />
          </div>
        )}
      </div>
    </div>
  )
}

function ActiveJobActions() {
  const { currentJob, approveJob, cancelJob } = useOrchestratorStore()
  if (!currentJob) return null
  return (
    <>
      <button
        onClick={() => approveJob(currentJob.jobId)}
        className="btn-primary font-mono text-xs"
      >
        Dispatch to Terminals
      </button>
      <button
        onClick={() => cancelJob(currentJob.jobId)}
        className="btn-secondary font-mono text-xs"
      >
        Reject
      </button>
    </>
  )
}

function CancelJobButton() {
  const { currentJob, cancelJob } = useOrchestratorStore()
  if (!currentJob) return null
  return (
    <button
      onClick={() => cancelJob(currentJob.jobId)}
      className="btn-secondary font-mono text-xs text-status-error border-status-error"
    >
      Cancel
    </button>
  )
}

function ActiveJobView() {
  const { currentJob, fetchStatus } = useOrchestratorStore()

  useEffect(() => {
    if (!currentJob) return
    const interval = setInterval(() => {
      fetchStatus(currentJob.jobId)
    }, 5000)
    return () => clearInterval(interval)
  }, [currentJob?.jobId, fetchStatus])

  if (!currentJob) return null

  return (
    <div className="h-full flex flex-col">
      {/* Pinned stats bar */}
      <div className="px-6 py-3 border-b border-border-subtle bg-background-secondary">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h2 className="font-brand text-lg text-text-primary">{currentJob.title}</h2>
            <p className="font-mono text-xs text-text-muted">
              Job: {currentJob.jobId.slice(0, 8)}...
            </p>
          </div>
          <div className="flex items-center gap-6">
            {[
              { label: 'Pending', value: currentJob.stats.pending, color: 'text-text-muted' },
              { label: 'Running', value: currentJob.stats.in_progress, color: 'text-status-warning' },
              { label: 'Done', value: currentJob.stats.completed, color: 'text-status-success' },
              { label: 'Failed', value: currentJob.stats.failed, color: 'text-status-error' },
            ].map(stat => (
              <div key={stat.label} className="flex items-center gap-2">
                <span className={clsx('font-mono text-lg font-bold', stat.color)}>{stat.value}</span>
                <span className="font-mono text-xs text-text-muted uppercase">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Scrollable grid */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-6xl mx-auto grid grid-cols-12 gap-6">
          <div className="col-span-8">
            <TaskDAGView />
          </div>
          <div className="col-span-4 space-y-6">
            <WorkerMonitor />
            <BudgetTracker />
          </div>
        </div>
      </div>
    </div>
  )
}
