import { memo, useEffect, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { ChevronDown, ChevronRight, Loader2, SlidersHorizontal } from 'lucide-react'
import { clsx } from 'clsx'
import { API_BASE, evalAdvancedApi, type HeldoutJobResponse, type HeldoutReport } from '../../../services/api'
import { useSkillLabStore, useWorkspaceStore } from '../../../stores'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type EvalNodeType = Node<DataNodeData, 'evals'>

const POLL_MS = 10_000

/** Shorten metric keys for the compact table: local_window_recall_at_10 → local_window r@10 */
function metricLabel(key: string): string {
  return key
    .replace(/_recall_at_(\d+)/, ' r@$1')
    .replace(/_mrr$/, ' mrr')
    .replace(/_mean_rank$/, ' rank')
}

function formatMetric(value: number): string {
  return (Math.abs(value) < 1 ? value.toFixed(3) : value.toFixed(1)).replace(/^(-?)0\./, '$1.')
}

function DeltaTable({ report }: { report: HeldoutReport }) {
  const deltas = report.delta_metrics
  if (!deltas) return null
  const base = report.base_metrics ?? {}
  const candidate = report.candidate_metrics ?? {}
  return (
    <div className="mt-1 space-y-px">
      {Object.entries(deltas).map(([key, delta]) => (
        <div key={key} className="flex items-center gap-1.5 text-[9px] font-mono">
          <span className="flex-1 truncate text-text-muted" title={key}>{metricLabel(key)}</span>
          {key in base && key in candidate && (
            <span className="text-text-secondary flex-shrink-0">
              {formatMetric(base[key])} → {formatMetric(candidate[key])}
            </span>
          )}
          <span
            className={clsx(
              'w-12 text-right flex-shrink-0 font-bold',
              delta > 0 ? 'text-status-success' : delta < 0 ? 'text-status-error' : 'text-text-muted'
            )}
          >
            {delta > 0 ? '+' : ''}{formatMetric(delta)}
          </span>
        </div>
      ))}
    </div>
  )
}

export const EvalNode = memo(function EvalNode({ data, selected }: NodeProps<EvalNodeType>) {
  const workspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const skillRunMap = useSkillLabStore((state) => state.runsByWorkspace)
  const refreshSkillRuns = useSkillLabStore((state) => state.refresh)
  const skillRuns = skillRunMap[workspaceId] || []
  const [jobs, setJobs] = useState<HeldoutJobResponse[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)

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

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!cancelled) await refreshSkillRuns(workspaceId)
    }
    void load()
    const timer = window.setInterval(() => void load(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [refreshSkillRuns, workspaceId])

  const latest = jobs.find((j) => j.report)

  const buildContext = () => {
    if (!latest?.report) return '## Held-out eval\n\nNo completed eval reports yet.'
    const r = latest.report
    const lines = [
      `## Held-out eval — ${latest.model_id}`,
      `- verdict: ${r.ship ? 'SHIP' : 'NO-SHIP'}`,
      `- candidate pass rate: ${(r.candidate_pass_rate * 100).toFixed(1)}%`,
      `- base pass rate: ${(r.base_pass_rate * 100).toFixed(1)}%`,
      `- n: ${r.n} (${r.n_clusters} clusters), metric: ${r.metric}`,
      ...r.reasons.map((reason) => `- reason: ${reason}`)
    ]
    if (r.delta_metrics) {
      lines.push('', '### Per-metric deltas (base → candidate)')
      for (const [key, delta] of Object.entries(r.delta_metrics)) {
        const base = r.base_metrics?.[key]
        const cand = r.candidate_metrics?.[key]
        const pair = base !== undefined && cand !== undefined ? `${base.toFixed(4)} → ${cand.toFixed(4)}` : ''
        lines.push(`- ${key}: ${pair} (Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(4)})`)
      }
    }
    if (r.candidate_model_path) lines.push('', `Candidate model: ${r.candidate_model_path}`)
    if (r.remote_eval_manifest) lines.push(`Eval manifest: ${r.remote_eval_manifest}`)
    return lines.join('\n')
  }

  const hasFailure = jobs.some((j) => j.status === 'failed')

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      flowerVariant="evals"
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={data.hasTerminalConnections ? buildContext : undefined}
      statusBarClass={
        hasFailure ? 'bg-status-error' :
        jobs.some((j) => j.status === 'running') ? 'bg-accent animate-pulse' :
        latest?.report?.ship ? 'bg-status-success' : 'bg-background-tertiary'
      }
      hue={hueFor('evals')}
      headerRight={
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setConfigOpen(true)
          }}
          className="nodrag node-btn node-btn-accent"
          title="Configure eval node"
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
      ) : jobs.length === 0 && skillRuns.length === 0 ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-3">
          No eval jobs yet
        </div>
      ) : (
        <div className="space-y-1.5">
          {jobs.map((job) => {
            const expandable = !!job.report?.delta_metrics
            const expanded = expandedId === job.job_id
            return (
              <div key={job.job_id} className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5">
                <div
                  className={clsx(
                    'flex items-center gap-1.5 text-[10px] font-mono',
                    expandable && 'nodrag cursor-pointer'
                  )}
                  onClick={expandable ? (e) => {
                    e.stopPropagation()
                    setExpandedId(expanded ? null : job.job_id)
                  } : undefined}
                  title={expandable ? 'Show per-metric deltas' : undefined}
                >
                  <span className={clsx(
                    'w-1.5 h-1.5 rounded-full flex-shrink-0',
                    job.status === 'running' && 'bg-accent animate-pulse',
                    job.status === 'completed' && 'bg-status-success',
                    job.status === 'failed' && 'bg-status-error'
                  )} />
                  {expandable && (
                    expanded
                      ? <ChevronDown className="w-2.5 h-2.5 text-text-muted flex-shrink-0" />
                      : <ChevronRight className="w-2.5 h-2.5 text-text-muted flex-shrink-0" />
                  )}
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
                {expanded && job.report && <DeltaTable report={job.report} />}
                {job.error && (
                  <div className="text-[9px] font-mono text-status-error mt-0.5 truncate" title={job.error}>{job.error}</div>
                )}
              </div>
            )
          })}
          {skillRuns.slice(0, 3).map((run) => (
            <div key={run.run_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
              <div className="flex min-w-0 items-center gap-1.5 font-mono text-[10px]">
                <span className={clsx(
                  'h-1.5 w-1.5 flex-shrink-0 rounded-full',
                  (run.status === 'queued' || run.status === 'running') && 'bg-accent animate-pulse',
                  run.status === 'completed' && 'bg-status-success',
                  run.status === 'failed' && 'bg-status-error',
                )} />
                <span className="min-w-0 flex-1 truncate text-text-secondary">{run.skill_name}</span>
                <span className="flex-shrink-0 text-[8px] font-bold uppercase text-text-muted">
                  {run.kpis?.verdict || run.status}
                </span>
              </div>
              {run.kpis ? (
                <div className="mt-0.5 font-mono text-[9px] text-text-muted">
                  uplift {run.kpis.success_uplift >= 0 ? '+' : ''}{Math.round(run.kpis.success_uplift * 100)}% · route F1 {Math.round(run.kpis.routing_f1 * 100)}%
                </div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </DataNodeShell>
    <NodeConfigModal
      isOpen={configOpen}
      onClose={() => setConfigOpen(false)}
      title={`${data.title} Config`}
      description="Held-out eval monitor"
      size="lg"
    >
      <ConfigSection title="Eval State">
        <div className="flex flex-wrap gap-1.5">
          <ConfigPill tone={error ? 'error' : jobs.some((j) => j.status === 'running') ? 'accent' : latest?.report?.ship ? 'success' : 'neutral'}>
            {error ? 'error' : jobs.some((j) => j.status === 'running') ? 'running' : latest?.report ? 'report' : 'idle'}
          </ConfigPill>
          <ConfigPill tone="neutral">{jobs.length} jobs</ConfigPill>
          {latest?.report ? (
            <ConfigPill tone={latest.report.ship ? 'success' : 'error'}>
              {latest.report.ship ? 'ship' : 'no-ship'}
            </ConfigPill>
          ) : null}
        </div>
        <ConfigRows>
          <ConfigRow label="Poll interval" value={`${POLL_MS / 1000}s`} />
          <ConfigRow label="Latest model" value={latest?.model_id} />
          <ConfigRow label="Latest job" value={latest?.job_id} />
          <ConfigRow label="Metric" value={latest?.report?.metric} />
          <ConfigRow
            label="Candidate"
            value={latest?.report ? `${(latest.report.candidate_pass_rate * 100).toFixed(1)}%` : undefined}
          />
          <ConfigRow
            label="Base"
            value={latest?.report ? `${(latest.report.base_pass_rate * 100).toFixed(1)}%` : undefined}
          />
          <ConfigRow label="Eval rows" value={latest?.report?.n} />
          <ConfigRow label="Clusters" value={latest?.report?.n_clusters} />
          <ConfigRow label="Error" value={error} />
        </ConfigRows>
      </ConfigSection>

      <ConfigSection title="Live Handles">
        <ConfigRows>
          <ConfigRow label="Recent evals" value={`${API_BASE}/eval/heldout?limit=5`} />
          <ConfigRow label="Latest eval" value={latest ? `${API_BASE}/eval/heldout/${latest.job_id}` : undefined} />
          <ConfigRow label="Workspace API" value={`${API_BASE}/workspace/context?format=json`} />
        </ConfigRows>
      </ConfigSection>

      <ConfigSection title={`Skill evals (${skillRuns.length})`}>
        <ConfigRows>
          <ConfigRow label="Running" value={skillRuns.filter((run) => run.status === 'queued' || run.status === 'running').length} />
          <ConfigRow label="Latest skill" value={skillRuns[0]?.skill_name} />
          <ConfigRow label="Latest verdict" value={skillRuns[0]?.kpis?.verdict || skillRuns[0]?.status} />
          <ConfigRow label="Latest uplift" value={skillRuns[0]?.kpis ? `${Math.round(skillRuns[0].kpis.success_uplift * 100)}%` : undefined} />
        </ConfigRows>
      </ConfigSection>
    </NodeConfigModal>
    </>
  )
})
