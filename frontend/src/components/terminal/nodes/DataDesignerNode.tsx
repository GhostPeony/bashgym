import { memo, useEffect, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Factory, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'
import { designerApi, type DesignerJobStatus } from '../../../services/api'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import type { DataNodeData } from './types'

export type DataDesignerNodeType = Node<DataNodeData, 'designer'>

const POLL_MS = 10_000

export const DataDesignerNode = memo(function DataDesignerNode({ data, selected }: NodeProps<DataDesignerNodeType>) {
  const [jobs, setJobs] = useState<DesignerJobStatus[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const res = await designerApi.listJobs(6)
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

  const latestDone = jobs.find((j) => j.status === 'completed')

  const buildContext = () => {
    if (jobs.length === 0) return '## Data Designer\n\nNo generation jobs yet.'
    return [
      '## Data Designer jobs',
      ...jobs.slice(0, 5).map((j) => {
        const prog = j.progress ? ` ${j.progress.current}/${j.progress.total}` : ''
        const out = j.output_dir ? ` → ${j.output_dir}` : ''
        return `- ${j.job_id} [${j.status}] ${j.pipeline}${prog}${out}`
      }),
      latestDone?.output_dir ? `\nLatest completed output: ${latestDone.output_dir}` : ''
    ].filter(Boolean).join('\n')
  }

  return (
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Factory}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={
        jobs.some((j) => j.status === 'failed') ? 'bg-status-error' :
        jobs.some((j) => j.status === 'running' || j.status === 'queued') ? 'bg-accent animate-pulse' :
        latestDone ? 'bg-status-success' : 'bg-background-tertiary'
      }
      hue={hueFor('designer')}
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
          No generation jobs yet
          <span className="block mt-1 text-text-secondary">Create one in Data Factory → Designer</span>
        </div>
      ) : (
        <div className="space-y-1.5">
          {jobs.map((job) => {
            const pct = job.progress && job.progress.total > 0
              ? Math.min(100, (job.progress.current / job.progress.total) * 100)
              : 0
            return (
              <div key={job.job_id} className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5">
                <div className="flex items-center gap-1.5 text-[10px] font-mono">
                  <span className={clsx(
                    'w-1.5 h-1.5 rounded-full flex-shrink-0',
                    (job.status === 'running' || job.status === 'queued') && 'bg-accent animate-pulse',
                    job.status === 'completed' && 'bg-status-success',
                    job.status === 'failed' && 'bg-status-error'
                  )} />
                  <span className="flex-1 truncate text-text-secondary" title={job.pipeline}>{job.pipeline}</span>
                  <span className="text-text-muted flex-shrink-0">
                    {job.progress ? `${job.progress.current}/${job.progress.total}` : job.num_records}
                  </span>
                </div>
                {job.status === 'running' && (
                  <div className="h-1 mt-1 bg-background-tertiary rounded-brutal overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                  </div>
                )}
                {job.error && (
                  <div className="text-[9px] font-mono text-status-error mt-0.5 truncate" title={job.error}>{job.error}</div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </DataNodeShell>
  )
})
