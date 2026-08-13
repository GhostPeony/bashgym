import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Sparkles,
  Eye,
  Play,
  RefreshCw,
  CheckCircle2,
  ArrowRight,
  Cpu,
  GitBranch,
  ShieldCheck
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  designerApi,
  type DesignerModel,
  type DesignerPipelineInfo,
  type DesignerJobStatus
} from '../../services/api'
import { useTrainingStore } from '../../stores'
import { useActivityStore } from '../../stores/activityStore'
import { useCanvasOrchestratorStore } from '../../stores/canvasOrchestratorStore'
import { useSessionResource } from '../../stores/sessionResource'
import { designerModelsResource, designerPipelinesResource } from '../../stores/factoryResources'

const EMPTY_PIPELINES: DesignerPipelineInfo[] = []
const EMPTY_MODELS: DesignerModel[] = []

const ROLLOUT_FORMATS = [
  { value: 'claude_code', label: 'Claude Code' },
  { value: 'codex', label: 'Codex' },
  { value: 'hermes_agent', label: 'Hermes' },
  { value: 'pi_coding_agent', label: 'Pi' },
  { value: 'atif', label: 'ATIF' }
]

/** A model picker grouped by provider, drawn from the live catalog. */
function ModelSelect({
  label,
  value,
  onChange,
  models,
  codeOnly
}: {
  label: string
  value: string
  onChange: (v: string) => void
  models: DesignerModel[]
  codeOnly?: boolean
}) {
  const opts = codeOnly ? models.filter((m) => m.is_code_model) : models
  const byProvider = useMemo(() => {
    const groups: Record<string, DesignerModel[]> = {}
    for (const m of opts) (groups[m.provider] ||= []).push(m)
    return groups
  }, [opts])

  return (
    <label className="block">
      <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
        {label}
      </span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="input w-full">
        <option value="">— provider default —</option>
        {Object.entries(byProvider).map(([prov, ms]) => (
          <optgroup key={prov} label={prov}>
            {ms.map((m) => (
              <option key={m.id} value={m.name}>
                {m.name}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  )
}

export function DataDesignerTab() {
  const { data: pipelinesData, loading: loadingPipelines } =
    useSessionResource(designerPipelinesResource)
  const {
    data: modelsData,
    loading: modelsInitialLoading,
    refreshing: modelsRefreshing,
    refresh: refreshModels
  } = useSessionResource(designerModelsResource)
  const pipelines = pipelinesData?.pipelines ?? EMPTY_PIPELINES
  const available = pipelinesData?.available ?? true
  const models = modelsData?.models ?? EMPTY_MODELS
  const modelsLoading = modelsInitialLoading || modelsRefreshing
  const [selected, setSelected] = useState<string | null>(null)

  const [textModel, setTextModel] = useState('')
  const [codeModel, setCodeModel] = useState('')
  const [judgeModel, setJudgeModel] = useState('')

  const [numRecords, setNumRecords] = useState(10)
  const [seedSource, setSeedSource] = useState('')
  const [seedType, setSeedType] = useState('traces')
  const [seedFormat, setSeedFormat] = useState('claude_code')
  const [keepOnlyPassing, setKeepOnlyPassing] = useState(true)
  const [mcpBackend, setMcpBackend] = useState('auto')

  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewRecords, setPreviewRecords] = useState<Record<string, unknown>[] | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const [job, setJob] = useState<DesignerJobStatus | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  const setDatasetPathOverride = useTrainingStore((s) => s.setDatasetPathOverride)

  const selectedPipeline = pipelines.find((p) => p.name === selected) || null
  const isRollout = seedType === 'agent_rollouts'
  const isToolPipeline = selected === 'mcp_tool_use'

  useEffect(() => {
    if (pipelines.length > 0) setSelected((s) => s ?? pipelines[0].name)
  }, [pipelines])

  const clearJobPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearTimeout(pollRef.current)
      pollRef.current = null
    }
  }, [])
  useEffect(() => () => clearJobPolling(), [clearJobPolling])

  const handlePreview = useCallback(async () => {
    if (!selected) return
    setPreviewLoading(true)
    setPreviewError(null)
    setPreviewRecords(null)
    const res = await designerApi.preview({
      pipeline: selected,
      num_records: Math.min(numRecords, 5),
      text_model: textModel || undefined,
      code_model: codeModel || undefined,
      judge_model: judgeModel || undefined
    })
    setPreviewLoading(false)
    if (res.ok && res.data) setPreviewRecords(res.data.records)
    else setPreviewError(res.error || 'Preview failed')
  }, [selected, numRecords, textModel, codeModel, judgeModel])

  const pollJob = useCallback(
    async (jobId: string) => {
      const res = await designerApi.getJob(jobId)
      if (res.ok && res.data) {
        setJob(res.data)
        useCanvasOrchestratorStore.getState().handleDesignerJob(res.data)
        if (res.data.status === 'completed' || res.data.status === 'failed') {
          useActivityStore.getState().addEvent(`designer:${res.data.status}`, {
            job_id: res.data.job_id,
            pipeline: res.data.pipeline,
            output_dir: res.data.output_dir,
            error: res.data.error
          })
          clearJobPolling()
          if (res.data.status === 'failed') setJobError(res.data.error || 'Job failed')
          return
        }
      }
      pollRef.current = window.setTimeout(() => pollJob(jobId), 2000)
    },
    [clearJobPolling]
  )

  const handleGenerate = useCallback(async () => {
    if (!selected) return
    clearJobPolling()
    setJobError(null)
    setJob(null)
    const res = await designerApi.create({
      pipeline: selected,
      num_records: numRecords,
      seed_source: seedSource || undefined,
      seed_type: seedType,
      seed_format: isRollout ? seedFormat : undefined,
      text_model: textModel || undefined,
      code_model: codeModel || undefined,
      judge_model: judgeModel || undefined,
      mcp_backend: isToolPipeline ? mcpBackend : undefined,
      keep_only_passing: keepOnlyPassing
    })
    if (res.ok && res.data) {
      setJob(res.data)
      useCanvasOrchestratorStore.getState().handleDesignerJob(res.data)
      useActivityStore.getState().addEvent('designer:queued', {
        job_id: res.data.job_id,
        pipeline: res.data.pipeline,
        num_records: res.data.num_records
      })
      pollRef.current = window.setTimeout(() => pollJob(res.data!.job_id), 2000)
    } else {
      setJobError(res.error || 'Failed to start job')
    }
  }, [
    selected,
    numRecords,
    seedSource,
    seedType,
    seedFormat,
    isRollout,
    textModel,
    codeModel,
    judgeModel,
    isToolPipeline,
    mcpBackend,
    keepOnlyPassing,
    pollJob,
    clearJobPolling
  ])

  const handleUseDataset = useCallback(() => {
    if (job?.output_dir) setDatasetPathOverride(job.output_dir)
  }, [job, setDatasetPathOverride])

  const jobRunning = !!job && job.status !== 'completed' && job.status !== 'failed'

  if (!available && !loadingPipelines) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="card p-4 border-l-4 border-l-status-warning bg-status-warning/10">
          <p className="font-mono text-xs text-text-primary">
            Data Designer is not installed. Run{' '}
            <code className="text-accent-dark">pip install "bashgym[data-designer]"</code> to enable
            synthetic data generation.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-end justify-between">
        <div>
          <h2 className="font-brand text-3xl text-text-primary flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-accent" />
            Data Designer
          </h2>
          <p className="font-mono text-xs text-text-muted mt-1">
            Synthesize training data — traces, agent rollouts, distillation & real tool-use
          </p>
        </div>
        <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
          {pipelines.length} pipelines · {models.length} models
        </span>
      </div>

      {/* Pipeline selector */}
      <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary mb-2">
        Pipeline
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        {loadingPipelines && (
          <div className="col-span-3 font-mono text-xs text-text-muted text-center py-6">
            Loading pipelines…
          </div>
        )}
        {pipelines.map((p) => (
          <button
            key={p.name}
            onClick={() => setSelected(p.name)}
            className={clsx(
              'card p-3 text-left transition-press',
              selected === p.name ? 'border-accent bg-accent-light' : 'hover:border-border'
            )}
          >
            <p
              className={clsx(
                'font-mono text-sm font-bold truncate',
                selected === p.name ? 'text-accent-dark' : 'text-text-primary'
              )}
            >
              {p.name}
            </p>
            <p className="text-xs text-text-secondary mt-1 line-clamp-2">{p.description}</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-text-muted mt-2">
              {p.columns.length} columns
            </p>
          </button>
        ))}
      </div>

      {/* Column DAG of the selected pipeline */}
      {selectedPipeline && selectedPipeline.columns.length > 0 && (
        <div className="card p-4 mb-5">
          <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary mb-3 flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 text-accent" />
            Column Pipeline
          </h3>
          <div className="flex flex-wrap items-center gap-y-2">
            {selectedPipeline.columns.map((c, i) => (
              <div key={c} className="flex items-center">
                <span className="font-mono text-[11px] px-2 py-1 border-2 border-border bg-background-secondary text-text-secondary">
                  {c}
                </span>
                {i < selectedPipeline.columns.length - 1 && (
                  <ArrowRight className="w-3.5 h-3.5 mx-1 text-accent/50 shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {selected && (
        <>
          {/* Models */}
          <div className="card p-4 mb-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-accent" />
                Teacher Models
              </h3>
              <button
                onClick={() => void refreshModels()}
                className="font-mono text-[10px] uppercase tracking-[0.12em] text-text-muted hover:text-accent-dark flex items-center gap-1 transition-colors"
                title="Refresh the live model catalog"
              >
                <RefreshCw className={clsx('w-3 h-3', modelsLoading && 'animate-spin')} />
                {models.length} available
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ModelSelect label="Text" value={textModel} onChange={setTextModel} models={models} />
              <ModelSelect
                label="Code"
                value={codeModel}
                onChange={setCodeModel}
                models={models}
                codeOnly
              />
              <ModelSelect
                label="Judge"
                value={judgeModel}
                onChange={setJudgeModel}
                models={models}
              />
            </div>
            <p className="font-mono text-[10px] text-text-muted mt-2">
              Discovered across NVIDIA NIM, Ollama & local sources — leave as default to use the
              provider's pick.
            </p>
          </div>

          {/* Generation config */}
          <div className="card p-4 mb-5">
            <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary mb-3">
              Generation
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Records
                </span>
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={numRecords}
                  onChange={(e) => setNumRecords(Number(e.target.value))}
                  className="input w-full"
                />
              </label>
              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Seed Type
                </span>
                <select
                  value={seedType}
                  onChange={(e) => setSeedType(e.target.value)}
                  className="input w-full"
                >
                  <option value="traces">Gold Traces</option>
                  <option value="agent_rollouts">Agent Rollouts</option>
                  <option value="huggingface">HuggingFace Dataset</option>
                  <option value="file">File</option>
                  <option value="unstructured">Unstructured Text</option>
                </select>
              </label>
              {isRollout ? (
                <label className="block">
                  <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                    Rollout Format
                  </span>
                  <select
                    value={seedFormat}
                    onChange={(e) => setSeedFormat(e.target.value)}
                    className="input w-full"
                  >
                    {ROLLOUT_FORMATS.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="block">
                  <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                    Seed Source
                  </span>
                  <input
                    type="text"
                    value={seedSource}
                    onChange={(e) => setSeedSource(e.target.value)}
                    placeholder="Path or HF repo id (optional)"
                    className="input w-full"
                  />
                </label>
              )}
            </div>

            {isRollout && (
              <label className="block mt-4">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Rollout Path
                </span>
                <input
                  type="text"
                  value={seedSource}
                  onChange={(e) => setSeedSource(e.target.value)}
                  placeholder="Default: ~/.claude/projects (per format) — override optional"
                  className="input w-full"
                />
              </label>
            )}

            {/* Toggles */}
            <div className="flex flex-wrap items-center gap-3 mt-4">
              <button
                type="button"
                onClick={() => setKeepOnlyPassing((v) => !v)}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 border-2 transition-press font-mono text-[11px] uppercase tracking-[0.12em]',
                  keepOnlyPassing
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'border-border text-text-muted'
                )}
                title="Drop rows below the pipeline's quality bar at export"
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                Quality Gate {keepOnlyPassing ? 'On' : 'Off'}
              </button>

              {isToolPipeline && (
                <label className="flex items-center gap-2">
                  <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-text-muted">
                    Tool Backend
                  </span>
                  <select
                    value={mcpBackend}
                    onChange={(e) => setMcpBackend(e.target.value)}
                    className="input py-1 text-xs"
                  >
                    <option value="auto">Auto</option>
                    <option value="docker">Docker</option>
                    <option value="local">Local</option>
                  </select>
                </label>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 mt-5">
              <button
                onClick={handlePreview}
                disabled={previewLoading}
                className="btn-secondary flex items-center gap-2"
              >
                {previewLoading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
                Preview
              </button>
              <button
                onClick={handleGenerate}
                disabled={jobRunning}
                className="btn-primary flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                Generate
              </button>
            </div>
          </div>
        </>
      )}

      {/* Preview */}
      {previewError && (
        <div className="card p-3 mb-4 border-l-4 border-l-status-error bg-status-error/10">
          <p className="font-mono text-xs text-status-error">{previewError}</p>
        </div>
      )}
      {previewRecords && previewRecords.length > 0 && (
        <div className="card p-4 mb-6">
          <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary mb-3">
            Preview · {previewRecords.length} records
          </h3>
          <div className="space-y-3">
            {previewRecords.map((record, idx) => (
              <pre
                key={idx}
                className="card p-3 bg-background-secondary font-mono text-xs overflow-auto max-h-64"
              >
                {JSON.stringify(record, null, 2)}
              </pre>
            ))}
          </div>
        </div>
      )}

      {/* Job status */}
      {job && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="font-brand text-lg text-text-primary">Generation Job</h3>
              <p className="font-mono text-xs text-text-muted">
                {job.job_id} · {job.pipeline}
              </p>
            </div>
            <span
              className={clsx(
                'tag text-xs',
                job.status === 'completed' && 'text-status-success',
                job.status === 'failed' && 'text-status-error'
              )}
            >
              <span>{job.status}</span>
            </span>
          </div>

          {job.progress && (
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs text-text-muted">Progress</span>
                <span className="font-mono text-xs text-text-secondary">
                  {job.progress.current} / {job.progress.total}
                </span>
              </div>
              <div className="h-2 bg-background-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent transition-all"
                  style={{
                    width: `${
                      job.progress.total > 0 ? (job.progress.current / job.progress.total) * 100 : 0
                    }%`
                  }}
                />
              </div>
            </div>
          )}

          {job.status === 'completed' && job.output_dir && (
            <div className="flex items-center justify-between mt-4 p-3 card bg-status-success/10 border-l-4 border-l-status-success">
              <div className="font-mono text-xs">
                <CheckCircle2 className="w-3 h-3 inline mr-1 text-status-success" />
                Output: <span className="text-text-primary">{job.output_dir}</span>
              </div>
              <button
                onClick={handleUseDataset}
                className="btn-secondary flex items-center gap-1 text-xs"
              >
                Use this dataset
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          )}

          {jobError && <p className="font-mono text-xs text-status-error mt-3">{jobError}</p>}
        </div>
      )}
    </div>
  )
}
