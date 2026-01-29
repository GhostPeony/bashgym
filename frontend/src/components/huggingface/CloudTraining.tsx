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
  Server
} from 'lucide-react'
import { hfApi, HFJob, HFJobSubmitRequest } from '../../services/api'
import { wsService, MessageTypes } from '../../services/websocket'
import { clsx } from 'clsx'

const HARDWARE_OPTIONS = [
  { value: 't4-small', label: 'T4 Small', vram: '16GB', description: 'Small models (<3B)' },
  { value: 'a10g-small', label: 'A10G Small', vram: '24GB', description: 'Medium models (3-7B)' },
  { value: 'a10g-large', label: 'A10G Large', vram: '24GB', description: 'Longer training' },
  { value: 'a100-large', label: 'A100 Large', vram: '80GB', description: 'Large models (7B+)' },
]

interface CloudTrainingProps {
  className?: string
}

export function CloudTraining({ className }: CloudTrainingProps) {
  const [jobs, setJobs] = useState<HFJob[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [formData, setFormData] = useState<HFJobSubmitRequest>({
    dataset_repo: '',
    output_repo: '',
    hardware: 'a10g-small',
    base_model: 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    num_epochs: 3,
    learning_rate: 2e-5,
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

    return () => {
      unsubscribeStarted()
      unsubscribeCompleted()
      unsubscribeFailed()
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
        return <Loader2 className="w-4 h-4 text-accent-primary animate-spin" />
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

  if (loading) {
    return (
      <div className={clsx('p-6', className)}>
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-accent-primary" />
        </div>
      </div>
    )
  }

  return (
    <div className={clsx('p-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-text-primary">Cloud Training</h2>
          <p className="text-sm text-text-secondary mt-1">
            Train models on HuggingFace GPUs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchJobs}
            className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-text-secondary" />
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center gap-2 px-3 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 transition-colors"
          >
            <Play className="w-4 h-4" />
            New Job
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-status-error/10 border border-status-error/30 rounded-lg flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Job Submission Form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="mb-6 p-4 bg-background-secondary rounded-lg border border-border">
          <h3 className="text-sm font-medium text-text-primary mb-4">Submit Training Job</h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-text-secondary mb-1">Dataset Repo</label>
              <input
                type="text"
                value={formData.dataset_repo}
                onChange={(e) => setFormData({ ...formData, dataset_repo: e.target.value })}
                placeholder="username/dataset-name"
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Output Repo</label>
              <input
                type="text"
                value={formData.output_repo}
                onChange={(e) => setFormData({ ...formData, output_repo: e.target.value })}
                placeholder="username/model-name"
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Hardware</label>
              <select
                value={formData.hardware}
                onChange={(e) => setFormData({ ...formData, hardware: e.target.value })}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
              >
                {HARDWARE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label} ({opt.vram}) - {opt.description}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Base Model</label>
              <input
                type="text"
                value={formData.base_model}
                onChange={(e) => setFormData({ ...formData, base_model: e.target.value })}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Epochs</label>
              <input
                type="number"
                value={formData.num_epochs}
                onChange={(e) => setFormData({ ...formData, num_epochs: parseInt(e.target.value) })}
                min={1}
                max={100}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Learning Rate</label>
              <input
                type="number"
                value={formData.learning_rate}
                onChange={(e) => setFormData({ ...formData, learning_rate: parseFloat(e.target.value) })}
                step={0.00001}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
              />
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-4">
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="px-4 py-2 text-text-secondary hover:text-text-primary transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex items-center gap-2 px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 transition-colors"
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
          <Server className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No training jobs yet</p>
          <p className="text-sm mt-1">Submit a job to train on HuggingFace GPUs</p>
        </div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job) => (
            <div
              key={job.job_id}
              className="p-4 bg-background-secondary rounded-lg border border-border"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(job.status)}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text-primary">
                        {job.job_id}
                      </span>
                      <span className="px-2 py-0.5 text-xs bg-background-tertiary text-text-secondary rounded">
                        {job.hardware}
                      </span>
                    </div>
                    <span className="text-xs text-text-secondary">
                      {new Date(job.created_at).toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {job.logs_url && (
                    <a
                      href={job.logs_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
                      title="View Logs"
                    >
                      <ExternalLink className="w-4 h-4 text-text-secondary" />
                    </a>
                  )}
                  {(job.status === 'pending' || job.status === 'running') && (
                    <button
                      onClick={() => handleCancel(job.job_id)}
                      className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
                      title="Cancel Job"
                    >
                      <Trash2 className="w-4 h-4 text-status-error" />
                    </button>
                  )}
                </div>
              </div>
              {job.error_message && (
                <div className="mt-2 p-2 bg-status-error/10 rounded text-sm text-status-error">
                  {job.error_message}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
