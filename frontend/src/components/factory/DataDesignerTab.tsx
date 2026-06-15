import { useCallback, useEffect, useRef, useState } from 'react'
import { Sparkles, Eye, Play, RefreshCw, CheckCircle2, ArrowRight } from 'lucide-react'
import { clsx } from 'clsx'
import {
  designerApi,
  type DesignerPipelineInfo,
  type DesignerJobStatus,
} from '../../services/api'
import { useTrainingStore } from '../../stores'

export function DataDesignerTab() {
  const [pipelines, setPipelines] = useState<DesignerPipelineInfo[]>([])
  const [available, setAvailable] = useState(true)
  const [loadingPipelines, setLoadingPipelines] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [numRecords, setNumRecords] = useState(10)
  const [seedSource, setSeedSource] = useState('')
  const [seedType, setSeedType] = useState('traces')

  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewRecords, setPreviewRecords] = useState<Record<string, unknown>[] | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const [job, setJob] = useState<DesignerJobStatus | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  const setDatasetPathOverride = useTrainingStore((s) => s.setDatasetPathOverride)

  useEffect(() => {
    setLoadingPipelines(true)
    designerApi.listPipelines().then((res) => {
      setLoadingPipelines(false)
      if (res.ok && res.data) {
        setPipelines(res.data.pipelines)
        setAvailable(res.data.available)
        if (res.data.pipelines.length > 0 && !selected) {
          setSelected(res.data.pipelines[0].name)
        }
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
    })
    setPreviewLoading(false)
    if (res.ok && res.data) {
      setPreviewRecords(res.data.records)
    } else {
      setPreviewError(res.error || 'Preview failed')
    }
  }, [selected, numRecords])

  const pollJob = useCallback(
    async (jobId: string) => {
      const res = await designerApi.getJob(jobId)
      if (res.ok && res.data) {
        setJob(res.data)
        const status = res.data.status
        if (status === 'completed' || status === 'failed') {
          clearJobPolling()
          if (status === 'failed') {
            setJobError(res.data.error || 'Job failed')
          }
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
    })
    if (res.ok && res.data) {
      setJob(res.data)
      pollRef.current = window.setTimeout(() => pollJob(res.data!.job_id), 2000)
    } else {
      setJobError(res.error || 'Failed to start job')
    }
  }, [selected, numRecords, seedSource, seedType, pollJob, clearJobPolling])

  const handleUseDataset = useCallback(() => {
    if (job?.output_dir) {
      setDatasetPathOverride(job.output_dir)
    }
  }, [job, setDatasetPathOverride])

  if (!available && !loadingPipelines) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="card p-4 border-l-4 border-l-status-warning bg-status-warning/10">
          <p className="font-mono text-xs text-text-primary">
            DataDesigner pipelines are not installed. Install the factory extras to enable this tab.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h2 className="font-brand text-2xl text-text-primary flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-accent" />
          Data Designer
        </h2>
        <p className="font-mono text-xs text-text-muted mt-1">
          Generate synthetic training datasets from registered pipelines
        </p>
      </div>

      {/* Pipeline cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
        {loadingPipelines && (
          <div className="col-span-2 font-mono text-xs text-text-muted text-center py-6">
            Loading pipelines...
          </div>
        )}
        {pipelines.map((p) => (
          <button
            key={p.name}
            onClick={() => setSelected(p.name)}
            className={clsx(
              'card p-4 text-left transition-press',
              selected === p.name
                ? 'border-accent bg-accent-light'
                : 'hover:border-border'
            )}
          >
            <div className="flex items-start gap-2">
              <Sparkles
                className={clsx(
                  'w-4 h-4 mt-0.5',
                  selected === p.name ? 'text-accent-dark' : 'text-text-muted'
                )}
              />
              <div className="flex-1">
                <p
                  className={clsx(
                    'font-mono text-sm font-bold',
                    selected === p.name ? 'text-accent-dark' : 'text-text-primary'
                  )}
                >
                  {p.name}
                </p>
                <p className="text-xs text-text-secondary mt-1">{p.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Config form */}
      {selected && (
        <div className="card p-4 mb-6">
          <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
            Generation Config
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">
                Num Records
              </label>
              <input
                type="number"
                min={1}
                max={10000}
                value={numRecords}
                onChange={(e) => setNumRecords(Number(e.target.value))}
                className="input w-full"
              />
            </div>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">
                Seed Type
              </label>
              <select
                value={seedType}
                onChange={(e) => setSeedType(e.target.value)}
                className="input w-full"
              >
                <option value="traces">Gold Traces</option>
                <option value="huggingface">HuggingFace Dataset</option>
                <option value="file">File</option>
                <option value="unstructured">Unstructured Text</option>
              </select>
            </div>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">
                Seed Source
              </label>
              <input
                type="text"
                value={seedSource}
                onChange={(e) => setSeedSource(e.target.value)}
                placeholder="Path or HF repo id (optional)"
                className="input w-full"
              />
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4">
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
              disabled={!!job && job.status !== 'completed' && job.status !== 'failed'}
              className="btn-primary flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Generate
            </button>
          </div>
        </div>
      )}

      {/* Preview output */}
      {previewError && (
        <div className="card p-3 mb-4 border-l-4 border-l-status-error bg-status-error/10">
          <p className="font-mono text-xs text-status-error">{previewError}</p>
        </div>
      )}
      {previewRecords && previewRecords.length > 0 && (
        <div className="card p-4 mb-6">
          <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
            Preview ({previewRecords.length} records)
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
                      job.progress.total > 0
                        ? (job.progress.current / job.progress.total) * 100
                        : 0
                    }%`,
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

          {jobError && (
            <p className="font-mono text-xs text-status-error mt-3">{jobError}</p>
          )}
        </div>
      )}
    </div>
  )
}
