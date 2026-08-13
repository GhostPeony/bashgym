import {
  AlertCircle,
  CheckCircle2,
  Clock,
  ExternalLink,
  Loader2,
  RefreshCw,
  Server,
  Square
} from 'lucide-react'
import { clsx } from 'clsx'

import type { HFJob } from '../../services/api'
import { hfJobsResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'

interface CloudTrainingProps {
  className?: string
}

export interface CloudTrainingViewProps extends CloudTrainingProps {
  jobs: HFJob[]
  loading: boolean
  refreshing: boolean
  error: string | null
  onRefresh: () => void
}

function statusIcon(status: HFJob['status']) {
  switch (status) {
    case 'pending':
      return <Clock className="w-4 h-4 text-status-warning" />
    case 'running':
      return <Loader2 className="w-4 h-4 text-accent animate-spin" />
    case 'completed':
      return <CheckCircle2 className="w-4 h-4 text-status-success" />
    case 'failed':
      return <AlertCircle className="w-4 h-4 text-status-error" />
    case 'cancelled':
      return <Square className="w-4 h-4 text-text-secondary" />
  }
}

export function CloudTrainingView({
  jobs,
  loading,
  refreshing,
  error,
  onRefresh,
  className
}: CloudTrainingViewProps) {
  if (loading) {
    return (
      <div className={clsx('p-6', className)}>
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-accent" />
        </div>
      </div>
    )
  }

  return (
    <div className={clsx('p-6', className)}>
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <h2 className="text-lg font-brand text-text-primary">Hugging Face Jobs</h2>
          <p className="text-sm text-text-secondary mt-1 font-mono">
            Observe jobs launched through Hugging Face Jobs.
          </p>
        </div>
        <button onClick={onRefresh} className="btn-icon" title="Refresh provider jobs">
          <RefreshCw
            className={clsx('w-4 h-4 text-text-secondary', refreshing && 'animate-spin')}
          />
        </button>
      </div>

      <div className="mb-5 p-3 border border-border-primary rounded-brutal bg-background-secondary">
        <p className="text-sm text-text-secondary">
          Launch and cancellation stay with the workflow that owns the provider job. This view reads
          provider status and links to provider logs.
        </p>
      </div>

      {error ? (
        <div className="p-3 border-2 border-status-error rounded-brutal bg-background-card flex items-start gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span className="text-sm font-mono">{error}</span>
        </div>
      ) : jobs.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <Server className="w-12 h-12 mx-auto mb-3 text-text-muted" />
          <p className="font-brand text-lg">No jobs returned</p>
          <p className="text-sm mt-1 font-mono">
            No provider jobs were returned for this account namespace.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => (
            <article key={`${job.namespace ?? ''}:${job.job_id}`} className="card p-4">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3 min-w-0">
                  {statusIcon(job.status)}
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-mono text-text-primary break-all">
                        {job.job_id}
                      </span>
                      <span className="tag">{job.hardware}</span>
                      {job.namespace && <span className="tag">{job.namespace}</span>}
                    </div>
                    <span className="text-xs text-text-secondary font-mono">
                      {new Date(job.created_at).toLocaleString()}
                    </span>
                  </div>
                </div>
                {job.logs_url && (
                  <a
                    href={job.logs_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-icon flex-shrink-0"
                    title="Open provider job"
                  >
                    <ExternalLink className="w-4 h-4 text-text-secondary" />
                  </a>
                )}
              </div>
              {job.error_message && (
                <div className="mt-2 p-2 border-2 border-status-error rounded-brutal bg-background-card text-sm text-status-error font-mono">
                  {job.error_message}
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  )
}

export function CloudTraining({ className }: CloudTrainingProps) {
  const { data, loading, refreshing, error, refresh } = useSessionResource(hfJobsResource)

  return (
    <CloudTrainingView
      className={className}
      jobs={data ?? []}
      loading={loading}
      refreshing={refreshing}
      error={error}
      onRefresh={() => void refresh()}
    />
  )
}
