import { memo, useEffect, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Factory, Loader2, Send, SlidersHorizontal } from 'lucide-react'
import { clsx } from 'clsx'
import { API_BASE, designerApi, type DesignerJobStatus } from '../../../services/api'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type DataDesignerNodeType = Node<DataNodeData, 'designer'>

const POLL_MS = 10_000

function jobContext(job: DesignerJobStatus): string {
  return [
    `## Data Designer job ${job.job_id}`,
    `- pipeline: ${job.pipeline}`,
    `- status: ${job.status}`,
    `- records: ${job.progress ? `${job.progress.current}/${job.progress.total}` : job.num_records}`,
    job.output_dir ? `- output dir: ${job.output_dir}` : null,
    job.error ? `- error: ${job.error}` : null,
    '',
    'Inspect the output above (JSONL records) before using it for training.'
  ].filter(Boolean).join('\n')
}

export const DataDesignerNode = memo(function DataDesignerNode({ data, selected }: NodeProps<DataDesignerNodeType>) {
  const [jobs, setJobs] = useState<DesignerJobStatus[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [sentJobId, setSentJobId] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)

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
  const latest = jobs[0]

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

  const sendJob = async (job: DesignerJobStatus) => {
    const result = await routeToLinkedTerminals(data.panelId, jobContext(job), 'md')
    if (result.routed > 0) {
      setSentJobId(job.job_id)
      setTimeout(() => setSentJobId(null), 1500)
    }
  }

  return (
    <>
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
      headerRight={
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setConfigOpen(true)
          }}
          className="nodrag node-btn node-btn-accent"
          title="Configure Data Designer node"
        >
          <SlidersHorizontal className="w-3 h-3" />
        </button>
      }
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
          <span className="block mt-1 text-text-secondary">Waiting for job state</span>
        </div>
      ) : (
        <div className="space-y-1.5">
          {jobs.map((job) => {
            const pct = job.progress && job.progress.total > 0
              ? Math.min(100, (job.progress.current / job.progress.total) * 100)
              : 0
            const canSend = data.hasConnections && job.status === 'completed' && !!job.output_dir
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
                  {canSend && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        void sendJob(job)
                      }}
                      className={clsx('nodrag node-btn', sentJobId === job.job_id ? 'node-btn-success' : 'node-btn-accent')}
                      title="Send this job's output path to linked terminals"
                    >
                      <Send className="w-2.5 h-2.5" />
                    </button>
                  )}
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
    <NodeConfigModal
      isOpen={configOpen}
      onClose={() => setConfigOpen(false)}
      title={`${data.title} Config`}
      description="Data Designer job monitor"
      size="lg"
    >
      <ConfigSection title="Designer State">
        <div className="flex flex-wrap gap-1.5">
          <ConfigPill tone={error ? 'error' : jobs.some((j) => j.status === 'running' || j.status === 'queued') ? 'accent' : latestDone ? 'success' : 'neutral'}>
            {error ? 'error' : jobs.some((j) => j.status === 'running' || j.status === 'queued') ? 'active' : latestDone ? 'ready' : 'idle'}
          </ConfigPill>
          <ConfigPill tone="neutral">{jobs.length} jobs</ConfigPill>
          {data.hasConnections ? <ConfigPill tone="accent">linked</ConfigPill> : null}
        </div>
        <ConfigRows>
          <ConfigRow label="Poll interval" value={`${POLL_MS / 1000}s`} />
          <ConfigRow label="Latest job" value={latest?.job_id} />
          <ConfigRow label="Latest pipeline" value={latest?.pipeline} />
          <ConfigRow label="Latest status" value={latest?.status} />
          <ConfigRow
            label="Latest progress"
            value={latest?.progress ? `${latest.progress.current}/${latest.progress.total}` : latest?.num_records}
          />
          <ConfigRow label="Latest output" value={latestDone?.output_dir} />
          <ConfigRow label="Error" value={error || latest?.error} />
        </ConfigRows>
      </ConfigSection>

      <ConfigSection title="Live Handles">
        <ConfigRows>
          <ConfigRow label="Recent jobs" value={`${API_BASE}/factory/designer/jobs?limit=6`} />
          <ConfigRow label="Latest job" value={latest ? `${API_BASE}/factory/designer/jobs/${encodeURIComponent(latest.job_id)}` : undefined} />
          <ConfigRow label="Pipelines" value={`${API_BASE}/factory/designer/pipelines`} />
          <ConfigRow label="Workspace API" value={`${API_BASE}/workspace/context?format=json`} />
        </ConfigRows>
      </ConfigSection>
    </NodeConfigModal>
    </>
  )
})
