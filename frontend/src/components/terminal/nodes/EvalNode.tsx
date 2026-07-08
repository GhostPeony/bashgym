import { memo, useEffect, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { FlaskConical, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'
import { evalAdvancedApi, type HeldoutJobResponse } from '../../../services/api'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import type { DataNodeData } from './types'

export type EvalNodeType = Node<DataNodeData, 'evals'>

const POLL_MS = 10_000

export const EvalNode = memo(function EvalNode({ data, selected }: NodeProps<EvalNodeType>) {
  const [jobs, setJobs] = useState<HeldoutJobResponse[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const res = await evalAdvancedApi.heldoutList(5)
      if (cancelled) return
      setLoaded(true)
      if (res.ok && res.data) {
        setJobs(res.data)
        setError(null)
      } else {
        setError(res.error || 'API unreachable')
      }
    }
    void load()
    const t = setInterval(load, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [])

  const latest = jobs.find((j) => j.report)

  const buildContext = () => {
    if (!latest?.report) return '## Held-out eval\n\nNo completed eval reports yet.'
    const r = latest.report
    return [
      `## Held-out eval — ${latest.model_id}`,
      `- verdict: ${r.ship ? 'SHIP' : 'NO-SHIP'}`,
      `- candidate pass rate: ${(r.candidate_pass_rate * 100).toFixed(1)}%`,
      `- base pass rate: ${(r.base_pass_rate * 100).toFixed(1)}%`,
      `- n: ${r.n} (${r.n_clusters} clusters), metric: ${r.metric}`,
      ...r.reasons.map((reason) => `- reason: ${reason}`)
    ].join('\n')
  }

  const hasFailure = jobs.some((j) => j.status === 'failed')

  return (
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={FlaskConical}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={
        hasFailure ? 'bg-status-error' :
        jobs.some((j) => j.status === 'running') ? 'bg-accent animate-pulse' :
        latest?.report?.ship ? 'bg-status-success' : 'bg-background-tertiary'
      }
      hue={hueFor('evals')}
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      {!loaded ? (
        <div className="flex justify-center py-3">
          <Loader2 className="w-4 h-4 animate-spin text-text-muted" />
        </div>
      ) : error ? (
        <div className="text-[10px] font-mono text-status-error text-center py-2">{error}</div>
      ) : jobs.length === 0 ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-3">
          No held-out eval jobs yet
        </div>
      ) : (
        <div className="space-y-1.5">
          {jobs.map((job) => (
            <div key={job.job_id} className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5">
              <div className="flex items-center gap-1.5 text-[10px] font-mono">
                <span className={clsx(
                  'w-1.5 h-1.5 rounded-full flex-shrink-0',
                  job.status === 'running' && 'bg-accent animate-pulse',
                  job.status === 'completed' && 'bg-status-success',
                  job.status === 'failed' && 'bg-status-error'
                )} />
                <span className="flex-1 truncate text-text-secondary" title={job.model_id}>{job.model_id}</span>
                {job.report && (
                  <span className={clsx(
                    'px-1 py-px border-brutal rounded-brutal text-[8px] font-bold uppercase',
                    job.report.ship
                      ? 'border-status-success/60 bg-status-success/10 text-status-success'
                      : 'border-status-error/60 bg-status-error/10 text-status-error'
                  )}>
                    {job.report.ship ? 'ship' : 'no-ship'}
                  </span>
                )}
              </div>
              {job.report && (
                <div className="text-[9px] font-mono text-text-muted mt-0.5">
                  {(job.report.candidate_pass_rate * 100).toFixed(1)}% vs base {(job.report.base_pass_rate * 100).toFixed(1)}%
                </div>
              )}
              {job.error && (
                <div className="text-[9px] font-mono text-status-error mt-0.5 truncate" title={job.error}>{job.error}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </DataNodeShell>
  )
})
