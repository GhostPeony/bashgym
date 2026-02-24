import { useState, useEffect, useCallback } from 'react'
import {
  Play,
  Square,
  RefreshCw,
  Loader2,
  ExternalLink,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Clock,
  Server,
  ChevronDown,
  ChevronRight,
  DollarSign,
} from 'lucide-react'
import { hfApi, HFJob, HFJobSubmitRequest } from '../../services/api'
import { wsService, MessageTypes } from '../../services/websocket'
import { clsx } from 'clsx'

const HARDWARE_OPTIONS = [
  { value: 't4-small', label: 'T4 Small', vram: '16GB', costPerHour: 0.60, description: 'Small models (<3B)' },
  { value: 'a10g-small', label: 'A10G Small', vram: '24GB', costPerHour: 1.05, description: 'Medium models (3-7B)' },
  { value: 'a10g-large', label: 'A10G Large', vram: '24GB', costPerHour: 1.80, description: 'Longer training' },
  { value: 'a100-large', label: 'A100 Large', vram: '80GB', costPerHour: 4.50, description: 'Large models (7B+)' },
  { value: 'h100', label: 'H100', vram: '80GB', costPerHour: 10.00, description: 'Maximum performance' },
]

const STRATEGY_OPTIONS = [
  { value: 'sft', label: 'SFT', description: 'Supervised Fine-Tuning — standard instruction tuning' },
  { value: 'dpo', label: 'DPO', description: 'Direct Preference Optimization — preference-based alignment' },
  { value: 'distillation', label: 'Distillation', description: 'Knowledge Distillation — learn from a larger teacher model' },
]

interface CloudTrainingProps {
  className?: string
}

export function CloudTraining({ className }: CloudTrainingProps) {
  const [jobs, setJobs] = useState<HFJob[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobMetrics, setJobMetrics] = useState<Record<string, Record<string, number>>>({})

  // Form state
  const [formData, setFormData] = useState<HFJobSubmitRequest>({
    dataset_repo: '',
    output_repo: '',
    hardware: 'a10g-small',
    base_model: 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    num_epochs: 3,
    learning_rate: 2e-5,
    strategy: 'sft',
    batch_size: 1,
    lora_r: 16,
    lora_alpha: 32,
    max_seq_length: 2048,
  })

  const fetchJobs = useCallback(async () => {
    const result = await hfApi.listJobs()
    if (result.ok && result.data) {
      setJobs(result.data)
      setError(null)
    } else if (result.error?.includes('403')) {
      setError('HuggingFace Pro subscription required for cloud training')
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchJobs()

    // Subscribe to job events
    const unsubscribeStarted = wsService.subscribe(MessageTypes.HF_JOB_STARTED, () => {
      fetchJobs()
    })
    const unsubscribeCompleted = wsService.subscribe(MessageTypes.HF_JOB_COMPLETED, () => {
      fetchJobs()
    })
    const unsubscribeFailed = wsService.subscribe(MessageTypes.HF_JOB_FAILED, () => {
      fetchJobs()
    })
    const unsubscribeMetrics = wsService.subscribe(MessageTypes.HF_JOB_METRICS, (payload: { job_id: string; metrics: Record<string, number> }) => {
      setJobMetrics((prev) => ({
        ...prev,
        [payload.job_id]: { ...prev[payload.job_id], ...payload.metrics },
      }))
    })

    return () => {
      unsubscribeStarted()
      unsubscribeCompleted()
      unsubscribeFailed()
      unsubscribeMetrics()
    }
  }, [fetchJobs])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)

    const result = await hfApi.submitJob(formData)
    if (result.ok) {
      setShowForm(false)
      fetchJobs()
    } else {
      setError(result.error || 'Failed to submit job')
    }
    setSubmitting(false)
  }

  const handleCancel = async (jobId: string) => {
    const result = await hfApi.cancelJob(jobId)
    if (result.ok) {
      fetchJobs()
    }
  }

  const getStatusIcon = (status: string) => {
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
      default:
        return null
    }
  }

  const selectedHardware = HARDWARE_OPTIONS.find((h) => h.value === formData.hardware)
  const estimatedCostPerHour = selectedHardware?.costPerHour ?? 0
  // Rough estimate: ~1 hour per epoch for small models
  const estimatedTotalCost = estimatedCostPerHour * (formData.num_epochs ?? 3)

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
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-brand text-text-primary">Cloud Training</h2>
          <p className="text-sm text-text-secondary mt-1 font-mono">
            Train models on HuggingFace GPUs via Unsloth Jobs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchJobs}
            className="btn-icon"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-text-secondary" />
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="btn-primary flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            New Job
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-background-card border-2 border-status-error rounded-brutal flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm font-mono">{error}</span>
        </div>
      )}

      {/* Job Submission Form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="mb-6 p-4 border-brutal shadow-brutal-sm rounded-brutal bg-background-card">
          <h3 className="text-sm font-brand text-text-primary mb-4">Submit Training Job</h3>

          {/* Training Strategy */}
          <div className="mb-4">
            <label className="block text-xs font-mono text-text-secondary mb-2 uppercase tracking-widest">Training Strategy</label>
            <div className="flex gap-2">
              {STRATEGY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setFormData({ ...formData, strategy: opt.value as HFJobSubmitRequest['strategy'] })}
                  className={clsx(
                    'flex-1 p-3 border-2 rounded-brutal text-left transition-all duration-150',
                    formData.strategy === opt.value
                      ? 'border-accent bg-accent/10 shadow-brutal-sm'
                      : 'border-border-primary hover:border-accent/50'
                  )}
                >
                  <span className="block text-sm font-mono font-semibold text-text-primary">{opt.label}</span>
                  <span className="block text-xs text-text-secondary mt-0.5">{opt.description}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Dataset Repo</label>
              <input
                type="text"
                value={formData.dataset_repo}
                onChange={(e) => setFormData({ ...formData, dataset_repo: e.target.value })}
                placeholder="username/dataset-name"
                className="input w-full text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Output Repo</label>
              <input
                type="text"
                value={formData.output_repo}
                onChange={(e) => setFormData({ ...formData, output_repo: e.target.value })}
                placeholder="username/model-name"
                className="input w-full text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Hardware</label>
              <select
                value={formData.hardware}
                onChange={(e) => setFormData({ ...formData, hardware: e.target.value })}
                className="input w-full text-sm"
              >
                {HARDWARE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label} ({opt.vram}) — ${opt.costPerHour.toFixed(2)}/hr — {opt.description}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Base Model</label>
              <input
                type="text"
                value={formData.base_model}
                onChange={(e) => setFormData({ ...formData, base_model: e.target.value })}
                className="input w-full text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Epochs</label>
              <input
                type="number"
                value={formData.num_epochs}
                onChange={(e) => setFormData({ ...formData, num_epochs: parseInt(e.target.value) || 1 })}
                min={1}
                max={100}
                className="input w-full text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Learning Rate</label>
              <input
                type="number"
                value={formData.learning_rate}
                onChange={(e) => setFormData({ ...formData, learning_rate: parseFloat(e.target.value) || 2e-5 })}
                step={0.00001}
                className="input w-full text-sm"
              />
            </div>
          </div>

          {/* Cost Estimate */}
          <div className="mt-4 p-3 border-2 border-border-primary rounded-brutal bg-background-page flex items-center gap-3">
            <DollarSign className="w-4 h-4 text-accent flex-shrink-0" />
            <div className="text-sm">
              <span className="font-mono text-text-secondary">Estimated cost: </span>
              <span className="font-mono font-semibold text-text-primary">
                ${estimatedCostPerHour.toFixed(2)}/hr
              </span>
              <span className="font-mono text-text-secondary"> · ~${estimatedTotalCost.toFixed(2)} total</span>
              <span className="text-xs text-text-muted ml-2">(~1hr/epoch estimate)</span>
            </div>
          </div>

          {/* Advanced Settings */}
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-xs font-mono text-text-secondary hover:text-text-primary uppercase tracking-widest"
            >
              {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-border-primary">
                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Batch Size</label>
                  <input
                    type="number"
                    value={formData.batch_size}
                    onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) || 1 })}
                    min={1}
                    max={64}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Max Seq Length</label>
                  <input
                    type="number"
                    value={formData.max_seq_length}
                    onChange={(e) => setFormData({ ...formData, max_seq_length: parseInt(e.target.value) || 2048 })}
                    min={256}
                    max={8192}
                    step={256}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">LoRA Rank (r)</label>
                  <input
                    type="number"
                    value={formData.lora_r}
                    onChange={(e) => setFormData({ ...formData, lora_r: parseInt(e.target.value) || 16 })}
                    min={4}
                    max={256}
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">LoRA Alpha</label>
                  <input
                    type="number"
                    value={formData.lora_alpha}
                    onChange={(e) => setFormData({ ...formData, lora_alpha: parseInt(e.target.value) || 32 })}
                    min={4}
                    max={512}
                    className="input w-full text-sm"
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 mt-4">
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="btn-ghost"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Submit Job
            </button>
          </div>
        </form>
      )}

      {/* Jobs List */}
      {jobs.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <Server className="w-12 h-12 mx-auto mb-3 text-text-muted" />
          <p className="font-brand text-lg">No training jobs yet</p>
          <p className="text-sm mt-1 font-mono">Submit a job to train on HuggingFace GPUs</p>
        </div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => {
            const metrics = jobMetrics[job.job_id] || job.metrics
            return (
              <div
                key={job.job_id}
                className="card p-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-mono text-text-primary">
                          {job.job_id}
                        </span>
                        <span className="tag">
                          <span>{job.hardware}</span>
                        </span>
                      </div>
                      <span className="text-xs text-text-secondary font-mono">
                        {new Date(job.created_at).toLocaleString()}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {metrics && metrics.loss !== undefined && (
                      <span className="text-xs font-mono text-accent">
                        loss: {metrics.loss.toFixed(4)}
                      </span>
                    )}
                    {job.logs_url && (
                      <a
                        href={job.logs_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-icon"
                        title="View Logs"
                      >
                        <ExternalLink className="w-4 h-4 text-text-secondary" />
                      </a>
                    )}
                    {(job.status === 'pending' || job.status === 'running') && (
                      <button
                        onClick={() => handleCancel(job.job_id)}
                        className="btn-icon"
                        title="Cancel Job"
                      >
                        <Trash2 className="w-4 h-4 text-status-error" />
                      </button>
                    )}
                  </div>
                </div>
                {job.error_message && (
                  <div className="mt-2 p-2 border-2 border-status-error rounded-brutal bg-background-card text-sm text-status-error font-mono">
                    {job.error_message}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
