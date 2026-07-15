import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Cloud, CloudUpload, Cpu, Database, ExternalLink, Loader2, Monitor, Play, Save, Send, SlidersHorizontal } from 'lucide-react'
import { clsx } from 'clsx'
import {
  designerApi,
  hfApi,
  type DesignerCreateRequest,
  type DesignerJobStatus,
  type DesignerPipelineInfo,
  type HFDataset,
} from '../../../services/api'
import { useActivityStore } from '../../../stores/activityStore'
import { useCanvasOrchestratorStore } from '../../../stores/canvasOrchestratorStore'
import { useRuntimeStore } from '../../../stores/runtimeStore'
import { useTerminalStore } from '../../../stores/terminalStore'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import { datasetRepoNameForJob } from './designerHuggingFace'
import type { DataNodeData } from './types'

export type DataDesignerNodeType = Node<DataNodeData, 'designer'>

const POLL_MS = 10_000

interface DesignerNodeConfig {
  pipeline?: string
  numRecords?: number
  seedSource?: string
  seedType?: string
  seedFormat?: string
  textModel?: string
  codeModel?: string
  judgeModel?: string
  mcpBackend?: string
  keepOnlyPassing?: boolean
  lastJobId?: string
  designerJobId?: string
  runtimeJobId?: string
  status?: string
  jobName?: string
  dataset?: string
  model?: string
  provider?: string
  execution?: 'local' | 'private' | 'cloud' | 'unknown' | null
  outputDir?: string
  progress?: { current: number; total: number; phase?: string }
  publishedDatasets?: Record<string, HFDataset>
  error?: string
}

interface PublishDraft {
  repoName: string
  private: boolean
}

interface DesignerDraft {
  pipeline: string
  numRecords: number
  seedSource: string
  seedType: string
  seedFormat: string
  textModel: string
  codeModel: string
  judgeModel: string
  mcpBackend: string
  keepOnlyPassing: boolean
}

function draftFromConfig(config: DesignerNodeConfig): DesignerDraft {
  return {
    pipeline: config.pipeline || '',
    numRecords: config.numRecords ?? 100,
    seedSource: config.seedSource || '',
    seedType: config.seedType || 'traces',
    seedFormat: config.seedFormat || 'claude_code',
    textModel: config.textModel || '',
    codeModel: config.codeModel || '',
    judgeModel: config.judgeModel || '',
    mcpBackend: config.mcpBackend || 'auto',
    keepOnlyPassing: config.keepOnlyPassing ?? true,
  }
}

function requestFromDraft(draft: DesignerDraft): DesignerCreateRequest {
  return {
    pipeline: draft.pipeline,
    num_records: draft.numRecords,
    seed_source: draft.seedSource.trim() || undefined,
    seed_type: draft.seedType,
    seed_format: draft.seedType === 'agent_rollouts' ? draft.seedFormat : undefined,
    text_model: draft.textModel.trim() || undefined,
    code_model: draft.codeModel.trim() || undefined,
    judge_model: draft.judgeModel.trim() || undefined,
    mcp_backend: draft.pipeline === 'mcp_tool_use' ? draft.mcpBackend : undefined,
    keep_only_passing: draft.keepOnlyPassing,
  }
}

function compactPath(value?: string | null): string {
  if (!value) return 'Not reported'
  const parts = value.split(/[\\/]/).filter(Boolean)
  return parts.at(-1) || value
}

function executionLabel(value?: string): string {
  if (value === 'private') return 'Private compute'
  if (value === 'local') return 'Local'
  if (value === 'cloud') return 'Cloud'
  return 'Compute unknown'
}

function statusTone(status?: string): 'neutral' | 'accent' | 'success' | 'error' {
  if (status === 'running' || status === 'queued') return 'accent'
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'error'
  return 'neutral'
}

function formattedDate(value?: string | null): string | undefined {
  if (!value) return undefined
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function jobContext(job: DesignerJobStatus): string {
  return [
    `## Data Designer job ${job.job_id}`,
    `- pipeline: ${job.pipeline}`,
    `- status: ${job.status}`,
    job.progress?.phase ? `- phase: ${job.progress.phase}` : null,
    `- records: ${job.progress ? `${job.progress.current}/${job.progress.total}` : job.num_records}`,
    job.dataset ? `- dataset: ${job.dataset}` : null,
    job.model ? `- model: ${job.model}` : null,
    job.execution ? `- compute: ${executionLabel(job.execution)}` : null,
    job.output_dir ? `- output dir: ${job.output_dir}` : null,
    job.error ? `- error: ${job.error}` : null,
    '',
    'Inspect the output above (JSONL records) before using it for training.'
  ].filter(Boolean).join('\n')
}

export const DataDesignerNode = memo(function DataDesignerNode({ data, selected }: NodeProps<DataDesignerNodeType>) {
  const nodeConfig = (data.adapterConfig || {}) as DesignerNodeConfig
  const [jobs, setJobs] = useState<DesignerJobStatus[]>([])
  const [pipelines, setPipelines] = useState<DesignerPipelineInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [sentJobId, setSentJobId] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [publishJobId, setPublishJobId] = useState<string | null>(null)
  const [publishDraft, setPublishDraft] = useState<PublishDraft>({ repoName: '', private: true })
  const [publishing, setPublishing] = useState(false)
  const [publishError, setPublishError] = useState<string | null>(null)
  const [publishResult, setPublishResult] = useState<HFDataset | null>(null)
  const [generating, setGenerating] = useState(false)
  const [draft, setDraft] = useState<DesignerDraft>(() => draftFromConfig(nodeConfig))
  const observedStatusesRef = useRef<Map<string, string>>(new Map())
  const statusHydratedRef = useRef(false)
  const updatePanelConfig = useTerminalStore((state) => state.updatePanelConfig)
  const runtimeJobs = useRuntimeStore((state) => state.jobs)
  const configuredJobId = nodeConfig.designerJobId || nodeConfig.runtimeJobId || nodeConfig.lastJobId
  const displayJobs = useMemo(() => {
    const observed = runtimeJobs
      .filter((job) => job.kind === 'designer')
      .map((job): DesignerJobStatus => ({
        job_id: job.job_id,
        status: job.status,
        pipeline: job.pipeline || job.script,
        num_records: job.progress?.total ?? 0,
        progress: job.progress?.total
          ? { current: job.progress.current, total: job.progress.total }
          : undefined,
        job_name: job.job_name,
        dataset: job.dataset,
        model: job.model,
        provider: job.provider,
        execution: job.execution,
        started_at: job.started_at,
        output_dir: job.output_dir || undefined,
      }))
    const observedIds = new Set(observed.map((job) => job.job_id))
    const merged = [...observed, ...jobs.filter((job) => !observedIds.has(job.job_id))]
    if (!configuredJobId || !nodeConfig.status || merged.some((job) => job.job_id === configuredJobId)) {
      return merged
    }
    return [...merged, {
      job_id: configuredJobId,
      status: nodeConfig.status,
      pipeline: nodeConfig.pipeline || 'data_designer',
      num_records: nodeConfig.numRecords ?? nodeConfig.progress?.total ?? 0,
      progress: nodeConfig.progress,
      job_name: nodeConfig.jobName,
      dataset: nodeConfig.dataset,
      model: nodeConfig.model,
      provider: nodeConfig.provider,
      execution: nodeConfig.execution,
      output_dir: nodeConfig.outputDir,
      error: nodeConfig.error,
    }]
  }, [
    configuredJobId,
    jobs,
    nodeConfig.dataset,
    nodeConfig.error,
    nodeConfig.execution,
    nodeConfig.jobName,
    nodeConfig.model,
    nodeConfig.numRecords,
    nodeConfig.outputDir,
    nodeConfig.pipeline,
    nodeConfig.progress,
    nodeConfig.provider,
    nodeConfig.status,
    runtimeJobs,
  ])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const res = await designerApi.listJobs(6)
      if (cancelled) return
      setLoaded(true)
      if (res.ok && res.data) {
        if (statusHydratedRef.current) {
          for (const job of res.data) {
            const previous = observedStatusesRef.current.get(job.job_id)
            if (previous !== job.status && (job.status === 'completed' || job.status === 'failed')) {
              useActivityStore.getState().addEvent(`designer:${job.status}`, {
                job_id: job.job_id,
                pipeline: job.pipeline,
                output_dir: job.output_dir,
                error: job.error,
              })
            }
          }
        }
        observedStatusesRef.current = new Map(res.data.map((job) => [job.job_id, job.status]))
        statusHydratedRef.current = true
        setJobs(res.data)
        if (configuredJobId) {
          const configured = res.data.find((job) => job.job_id === configuredJobId)
          if (configured) {
            useCanvasOrchestratorStore.getState().handleDesignerJob(configured, {
              originPanelId: data.panelId,
            })
          }
        }
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
  }, [configuredJobId, data.panelId])

  useEffect(() => {
    let cancelled = false
    void designerApi.listPipelines().then((res) => {
      if (cancelled || !res.ok || !res.data) return
      setPipelines(res.data.pipelines)
      setDraft((current) => ({
        ...current,
        pipeline: current.pipeline || res.data!.pipelines[0]?.name || '',
      }))
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!configOpen) return
    setDraft((current) => {
      const configured = draftFromConfig((data.adapterConfig || {}) as DesignerNodeConfig)
      return {
        ...configured,
        pipeline: configured.pipeline || current.pipeline || pipelines[0]?.name || '',
      }
    })
    setActionError(null)
  }, [configOpen, data.adapterConfig, pipelines])

  const latestDone = displayJobs.find((j) => j.status === 'completed')
  const selectedJob = selectedJobId
    ? displayJobs.find((job) => job.job_id === selectedJobId) || null
    : null
  const selectedRuntimeJob = selectedJob
    ? runtimeJobs.find((job) => job.job_id === selectedJob.job_id)
    : undefined
  const publishJob = publishJobId
    ? displayJobs.find((job) => job.job_id === publishJobId) || null
    : null
  const linkedHuggingFace = data.linkedHuggingFace || []
  const selectedPct = selectedJob?.progress && selectedJob.progress.total > 0
    ? Math.min(100, (selectedJob.progress.current / selectedJob.progress.total) * 100)
    : 0
  const selectedIndeterminate = Boolean(
    selectedJob?.status === 'running'
    && selectedJob.progress?.total
    && selectedJob.progress.current <= 0,
  )
  const selectedCanSend = Boolean(
    selectedJob
    && data.hasTerminalConnections
    && selectedJob.status === 'completed'
    && selectedJob.output_dir,
  )
  const selectedCanPublish = Boolean(
    selectedJob
    && selectedJob.status === 'completed'
    && selectedJob.output_dir,
  )

  const buildContext = () => {
    if (displayJobs.length === 0) return '## Data Designer\n\nNo generation jobs yet.'
    return [
      '## Data Designer jobs',
      ...displayJobs.slice(0, 5).map((j) => {
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

  const openPublish = (job: DesignerJobStatus) => {
    setPublishJobId(job.job_id)
    setSelectedJobId(null)
    setPublishDraft({ repoName: datasetRepoNameForJob(job), private: true })
    setPublishError(null)
    setPublishResult(null)
  }

  const closePublish = () => {
    const jobId = publishJobId
    setPublishJobId(null)
    setPublishError(null)
    if (jobId) setSelectedJobId(jobId)
  }

  const publishDataset = async () => {
    if (!publishJob?.output_dir || !publishDraft.repoName.trim() || publishing) return
    setPublishing(true)
    setPublishError(null)
    try {
      const response = await hfApi.uploadDataset({
        local_path: publishJob.output_dir,
        repo_name: publishDraft.repoName.trim(),
        private: publishDraft.private,
        metadata: {
          bashgym_job_id: publishJob.job_id,
          job_name: publishJob.job_name,
          pipeline: publishJob.pipeline,
          source_dataset: publishJob.dataset,
          generator_model: publishJob.model,
        },
      })
      if (!response.ok || !response.data) {
        throw new Error(response.error || 'Unable to publish dataset to Hugging Face.')
      }

      const result = response.data
      setPublishResult(result)
      const terminalState = useTerminalStore.getState()
      const designerPanel = terminalState.panels.find((panel) => panel.id === data.panelId)
      terminalState.updatePanelConfig(data.panelId, {
        ...(designerPanel?.adapterConfig || {}),
        publishedDatasets: {
          ...((designerPanel?.adapterConfig?.publishedDatasets as Record<string, HFDataset>) || {}),
          [publishJob.job_id]: result,
        },
      })
      for (const target of linkedHuggingFace) {
        const targetPanel = terminalState.panels.find((panel) => panel.id === target.panelId)
        terminalState.updatePanelConfig(target.panelId, {
          ...(targetPanel?.adapterConfig || {}),
          refreshRequestedAt: Date.now(),
          lastPublishedRepo: result.repo_id,
        })
      }
      useActivityStore.getState().addEvent('hf:dataset:completed', {
        job_id: publishJob.job_id,
        repo_id: result.repo_id,
        url: result.url,
      })
    } catch (caught) {
      setPublishError(caught instanceof Error ? caught.message : 'Unable to publish dataset.')
    } finally {
      setPublishing(false)
    }
  }

  const saveDraft = (job?: DesignerJobStatus) => {
    const next = {
      ...nodeConfig,
      ...draft,
      ...(job ? {
        designerJobId: job.job_id,
        lastJobId: job.job_id,
        status: job.status,
      } : {}),
    }
    updatePanelConfig(data.panelId, next)
    return next
  }

  const generateDataset = async () => {
    if (!draft.pipeline || generating) {
      if (!draft.pipeline) setActionError('Choose a Data Designer pipeline first.')
      return
    }
    setGenerating(true)
    setActionError(null)
    saveDraft()
    try {
      const res = await designerApi.create({
        ...requestFromDraft(draft),
        origin: { kind: 'panel', panel_id: data.panelId },
      })
      if (!res.ok || !res.data) throw new Error(res.error || 'Unable to start Data Designer job.')
      const job = res.data
      observedStatusesRef.current.set(job.job_id, job.status)
      setJobs((current) => [job, ...current.filter((candidate) => candidate.job_id !== job.job_id)].slice(0, 6))
      saveDraft(job)
      useCanvasOrchestratorStore.getState().handleDesignerJob(job, {
        originPanelId: data.panelId,
        config: { ...draft },
      })
      useActivityStore.getState().addEvent('designer:queued', {
        job_id: job.job_id,
        pipeline: job.pipeline,
        num_records: job.num_records,
      })
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : 'Unable to start Data Designer job.')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      flowerVariant="designer"
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={data.hasTerminalConnections ? buildContext : undefined}
      statusBarClass={
        displayJobs.some((j) => j.status === 'failed') ? 'bg-status-error' :
        displayJobs.some((j) => j.status === 'running' || j.status === 'queued') ? 'bg-accent animate-pulse' :
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
          title="Configure generation inputs"
        >
          <SlidersHorizontal className="w-3 h-3" />
        </button>
      }
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      {!loaded && displayJobs.length === 0 ? (
        <div className="flex justify-center py-3">
          <Loader2 className="w-4 h-4 animate-spin text-text-muted" />
        </div>
      ) : error && displayJobs.length === 0 ? (
        <div className="text-[10px] font-mono text-status-error text-center py-2">{error}</div>
      ) : displayJobs.length === 0 ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-3">
          No generation jobs yet
          <span className="block mt-1 text-text-secondary">Waiting for job state</span>
        </div>
      ) : (
        <div className="space-y-1.5">
          {displayJobs.map((job) => {
            const observedJob = runtimeJobs.find((candidate) => candidate.job_id === job.job_id)
            const pct = job.progress && job.progress.total > 0
              ? Math.min(100, (job.progress.current / job.progress.total) * 100)
              : 0
            const indeterminate = (
              job.status === 'running'
              && Boolean(job.progress?.total)
              && job.progress!.current <= 0
            )
            const canSend = data.hasTerminalConnections && job.status === 'completed' && !!job.output_dir
            const canPublish = job.status === 'completed' && !!job.output_dir
            return (
              <div
                key={job.job_id}
                role="button"
                tabIndex={0}
                className="nodrag cursor-pointer rounded-brutal border-brutal border-border-subtle px-2 py-1.5 transition-colors hover:border-accent hover:bg-background-tertiary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                title="View run details"
                onClick={(event) => {
                  event.stopPropagation()
                  setSelectedJobId(job.job_id)
                }}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return
                  event.preventDefault()
                  event.stopPropagation()
                  setSelectedJobId(job.job_id)
                }}
              >
                <div className="flex items-center gap-1.5 text-[10px] font-mono">
                  <span className={clsx(
                    'w-1.5 h-1.5 rounded-full flex-shrink-0',
                    (job.status === 'running' || job.status === 'queued') && 'bg-accent animate-pulse',
                    job.status === 'completed' && 'bg-status-success',
                    job.status === 'failed' && 'bg-status-error'
                  )} />
                  <span
                    className="flex-1 truncate font-semibold text-text-primary"
                    title={job.job_name || observedJob?.job_name || job.pipeline}
                  >
                    {job.job_name || observedJob?.job_name || job.pipeline}
                  </span>
                  <span className="text-text-primary flex-shrink-0 font-semibold">
                    {indeterminate
                      ? job.progress?.phase || 'working'
                      : job.progress ? `${pct.toFixed(0)}%` : job.num_records}
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
                  {canPublish && linkedHuggingFace.length > 0 && (
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        openPublish(job)
                      }}
                      className="nodrag node-btn node-btn-accent"
                      title="Publish dataset to linked Hugging Face node"
                    >
                      <CloudUpload className="h-2.5 w-2.5" />
                    </button>
                  )}
                </div>
                {job.dataset || job.model || job.provider || job.execution ? (
                  <div className="mt-1.5 grid grid-cols-1 gap-1 border-t border-border-subtle pt-1.5 font-mono text-[9px] text-text-muted">
                    <span className="flex min-w-0 items-center gap-1" title={job.dataset || undefined}>
                      <Database className="h-2.5 w-2.5 flex-shrink-0 text-accent" />
                      <span className="truncate">{compactPath(job.dataset)}</span>
                    </span>
                    <span className="flex min-w-0 items-center gap-1" title={job.model || undefined}>
                      <Cpu className="h-2.5 w-2.5 flex-shrink-0 text-accent" />
                      <span className="truncate">{job.model || job.provider || 'Provider default'}</span>
                    </span>
                    <span className="flex min-w-0 items-center gap-1">
                      {job.execution === 'cloud'
                        ? <Cloud className="h-2.5 w-2.5 flex-shrink-0 text-accent" />
                        : <Monitor className="h-2.5 w-2.5 flex-shrink-0 text-accent" />}
                      <span>{executionLabel(job.execution || undefined)}</span>
                    </span>
                  </div>
                ) : null}
                {job.status === 'running' && (
                  <div className="mt-1.5 space-y-1">
                    <div className="h-2.5 overflow-hidden border-brutal border-border-subtle bg-background-tertiary">
                      {indeterminate
                        ? <div className="canvas-progress-indeterminate h-full bg-accent" />
                        : <div className="h-full bg-accent transition-[width] duration-500 ease-out" style={{ width: `${pct}%` }} />}
                    </div>
                    <div className="flex justify-between font-mono text-[9px] text-text-muted">
                      <span>{job.progress?.phase || 'running'}</span>
                      <span>
                        {indeterminate
                          ? 'working'
                          : job.progress ? `${job.progress.current} / ${job.progress.total}` : 'working'}
                      </span>
                    </div>
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
      title={`${data.title} Setup`}
      description="Configure and launch synthetic dataset jobs"
      size="lg"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="min-w-0 font-mono text-[10px] text-status-error">
            {actionError}
          </span>
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              type="button"
              className="btn-secondary flex items-center gap-2 text-xs"
              onClick={() => {
                saveDraft()
                setConfigOpen(false)
              }}
            >
              <Save className="h-3.5 w-3.5" />
              Save
            </button>
            <button
              type="button"
              className="btn-primary flex items-center gap-2 text-xs"
              onClick={() => void generateDataset()}
              disabled={generating || !draft.pipeline}
            >
              {generating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Generate
            </button>
          </div>
        </div>
      }
    >
      <ConfigSection title="Generation Inputs">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="node-field">
            <span className="node-field-label">Pipeline</span>
            <select
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.pipeline}
              onChange={(event) => setDraft((current) => ({ ...current, pipeline: event.target.value }))}
            >
              {pipelines.length === 0 ? <option value="">No pipelines available</option> : null}
              {pipelines.map((pipeline) => (
                <option key={pipeline.name} value={pipeline.name}>{pipeline.name}</option>
              ))}
            </select>
          </label>
          <label className="node-field">
            <span className="node-field-label">Records</span>
            <input
              type="number"
              min={1}
              max={100000}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.numRecords}
              onChange={(event) => setDraft((current) => ({ ...current, numRecords: Number(event.target.value) }))}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Seed type</span>
            <select
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.seedType}
              onChange={(event) => setDraft((current) => ({ ...current, seedType: event.target.value }))}
            >
              <option value="traces">Gold traces</option>
              <option value="agent_rollouts">Agent rollouts</option>
            </select>
          </label>
          {draft.seedType === 'agent_rollouts' ? (
            <label className="node-field">
              <span className="node-field-label">Rollout format</span>
              <select
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={draft.seedFormat}
                onChange={(event) => setDraft((current) => ({ ...current, seedFormat: event.target.value }))}
              >
                <option value="claude_code">Claude Code</option>
                <option value="codex">Codex</option>
                <option value="hermes_agent">Hermes</option>
                <option value="pi_coding_agent">Pi</option>
                <option value="atif">ATIF</option>
              </select>
            </label>
          ) : null}
          <label className="node-field md:col-span-2">
            <span className="node-field-label">Seed source</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.seedSource}
              onChange={(event) => setDraft((current) => ({ ...current, seedSource: event.target.value }))}
              placeholder="Optional trace, dataset, or rollout path"
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Text model</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.textModel}
              onChange={(event) => setDraft((current) => ({ ...current, textModel: event.target.value }))}
              placeholder="Provider default"
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Code model</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.codeModel}
              onChange={(event) => setDraft((current) => ({ ...current, codeModel: event.target.value }))}
              placeholder="Provider default"
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Judge model</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.judgeModel}
              onChange={(event) => setDraft((current) => ({ ...current, judgeModel: event.target.value }))}
              placeholder="Provider default"
            />
          </label>
          {draft.pipeline === 'mcp_tool_use' ? (
            <label className="node-field">
              <span className="node-field-label">MCP backend</span>
              <select
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={draft.mcpBackend}
                onChange={(event) => setDraft((current) => ({ ...current, mcpBackend: event.target.value }))}
              >
                <option value="auto">Auto</option>
                <option value="bashgym">BashGym</option>
                <option value="hermes">Hermes</option>
              </select>
            </label>
          ) : null}
          <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2 md:col-span-2">
            <input
              type="checkbox"
              checked={draft.keepOnlyPassing}
              onChange={(event) => setDraft((current) => ({ ...current, keepOnlyPassing: event.target.checked }))}
            />
            <span className="font-mono text-[11px] text-text-secondary">Keep only passing generated records</span>
          </label>
        </div>
      </ConfigSection>

    </NodeConfigModal>
    <NodeConfigModal
      isOpen={Boolean(selectedJob)}
      onClose={() => setSelectedJobId(null)}
      title={selectedJob?.job_name || selectedRuntimeJob?.job_name || selectedJob?.pipeline || 'Data Designer Run'}
      description="Generation run details"
      size="md"
      footer={
        <div className="flex w-full justify-end gap-2">
          <button
            type="button"
            className="btn-secondary text-xs"
            onClick={() => setSelectedJobId(null)}
          >
            Close
          </button>
          {selectedCanSend && selectedJob ? (
            <button
              type="button"
              className="btn-secondary flex items-center gap-2 text-xs"
              onClick={() => void sendJob(selectedJob)}
            >
              <Send className="h-3.5 w-3.5" />
              Send output
            </button>
          ) : null}
          {selectedCanPublish && selectedJob ? (
            <button
              type="button"
              className="btn-primary flex items-center gap-2 text-xs"
              onClick={() => openPublish(selectedJob)}
            >
              <CloudUpload className="h-3.5 w-3.5" />
              {linkedHuggingFace.length ? 'Publish to linked HF' : 'Publish to HF'}
            </button>
          ) : null}
        </div>
      }
    >
      {selectedJob ? (
        <>
          <ConfigSection>
            <div className="flex flex-wrap gap-1.5">
              <ConfigPill tone={statusTone(selectedJob.status)}>{selectedJob.status}</ConfigPill>
              <ConfigPill tone="neutral">{selectedJob.pipeline}</ConfigPill>
              <ConfigPill tone="accent">
                {executionLabel(selectedJob.execution || selectedRuntimeJob?.execution)}
              </ConfigPill>
            </div>
          </ConfigSection>
          {selectedJob.progress?.total ? (
            <ConfigSection title="Progress">
              <div className="space-y-2">
                <div className="flex items-end justify-between gap-3">
                  <p className="font-mono text-lg font-semibold text-text-primary">
                    {selectedJob.progress.current} / {selectedJob.progress.total}
                  </p>
                  <span className="font-mono text-2xl font-semibold text-accent">
                    {selectedIndeterminate
                      ? selectedJob.progress.phase || 'Working'
                      : `${selectedPct.toFixed(0)}%`}
                  </span>
                </div>
                <div className="h-3 overflow-hidden border-brutal border-border bg-background-tertiary">
                  {selectedIndeterminate
                    ? <div className="canvas-progress-indeterminate h-full bg-accent" />
                    : <div className="h-full bg-accent transition-[width] duration-500 ease-out" style={{ width: `${selectedPct}%` }} />}
                </div>
              </div>
            </ConfigSection>
          ) : null}
          <ConfigSection title="Run Details">
            <ConfigRows>
              <ConfigRow label="Dataset" value={selectedJob.dataset || selectedRuntimeJob?.dataset} />
              <ConfigRow label="Model" value={selectedJob.model || selectedRuntimeJob?.model || selectedJob.provider || selectedRuntimeJob?.provider} />
              <ConfigRow label="Provider" value={selectedJob.provider || selectedRuntimeJob?.provider} />
              <ConfigRow label="Started" value={formattedDate(selectedJob.started_at || selectedRuntimeJob?.started_at)} />
              <ConfigRow label="Completed" value={formattedDate(selectedRuntimeJob?.completed_at)} />
              <ConfigRow label="Output" value={selectedJob.output_dir || selectedRuntimeJob?.output_dir} />
              <ConfigRow
                label="Hugging Face"
                value={nodeConfig.publishedDatasets?.[selectedJob.job_id] ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-accent hover:text-accent-dark"
                    onClick={() => window.open(nodeConfig.publishedDatasets?.[selectedJob.job_id].url, '_blank')}
                  >
                    {nodeConfig.publishedDatasets[selectedJob.job_id].repo_id}
                    <ExternalLink className="h-3 w-3" />
                  </button>
                ) : undefined}
              />
              <ConfigRow label="Error" value={selectedJob.error} />
            </ConfigRows>
          </ConfigSection>
        </>
      ) : null}
    </NodeConfigModal>
    <NodeConfigModal
      isOpen={Boolean(publishJob)}
      onClose={closePublish}
      title="Publish Dataset"
      description={linkedHuggingFace.length
        ? `Send this completed run to ${linkedHuggingFace.map((target) => target.title).join(', ')}`
        : 'Send this completed run to the configured Hugging Face account'}
      size="md"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="min-w-0 font-mono text-[10px] text-status-error">{publishError}</span>
          <div className="flex flex-shrink-0 gap-2">
            <button type="button" className="btn-secondary text-xs" onClick={closePublish}>
              {publishResult ? 'Done' : 'Cancel'}
            </button>
            {!publishResult ? (
              <button
                type="button"
                className="btn-primary flex items-center gap-2 text-xs"
                disabled={publishing || !publishDraft.repoName.trim()}
                onClick={() => void publishDataset()}
              >
                {publishing
                  ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  : <CloudUpload className="h-3.5 w-3.5" />}
                Publish dataset
              </button>
            ) : null}
          </div>
        </div>
      }
    >
      {publishJob ? (
        <>
          <ConfigSection title="Destination">
            <div className="flex flex-wrap gap-1.5">
              {linkedHuggingFace.length ? linkedHuggingFace.map((target) => (
                <ConfigPill key={target.panelId} tone="accent">{target.title}</ConfigPill>
              )) : <ConfigPill tone="neutral">Configured Hugging Face account</ConfigPill>}
              <ConfigPill tone={publishDraft.private ? 'neutral' : 'warning'}>
                {publishDraft.private ? 'private' : 'public'}
              </ConfigPill>
            </div>
            <label className="node-field">
              <span className="node-field-label">Dataset repository</span>
              <input
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={publishDraft.repoName}
                disabled={publishing || Boolean(publishResult)}
                onChange={(event) => setPublishDraft((current) => ({
                  ...current,
                  repoName: event.target.value,
                }))}
              />
            </label>
            <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2">
              <input
                type="checkbox"
                checked={publishDraft.private}
                disabled={publishing || Boolean(publishResult)}
                onChange={(event) => setPublishDraft((current) => ({
                  ...current,
                  private: event.target.checked,
                }))}
              />
              <span className="font-mono text-[11px] text-text-secondary">Private dataset</span>
            </label>
          </ConfigSection>
          <ConfigSection title={publishResult ? 'Published' : 'Source'}>
            <ConfigRows>
              <ConfigRow label="Run" value={publishJob.job_name || publishJob.pipeline} />
              <ConfigRow label="Records" value={publishJob.progress?.current || publishJob.num_records} />
              <ConfigRow label="Output" value={publishJob.output_dir} />
              <ConfigRow
                label="Repository"
                value={publishResult ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-accent hover:text-accent-dark"
                    onClick={() => window.open(publishResult.url, '_blank')}
                  >
                    {publishResult.repo_id}
                    <ExternalLink className="h-3 w-3" />
                  </button>
                ) : undefined}
              />
            </ConfigRows>
          </ConfigSection>
        </>
      ) : null}
    </NodeConfigModal>
    </>
  )
})
