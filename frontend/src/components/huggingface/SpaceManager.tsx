import { useState, useEffect, useCallback } from 'react'
import {
  Plus,
  Trash2,
  RefreshCw,
  Loader2,
  ExternalLink,
  AlertCircle,
  CheckCircle2,
  Clock,
  Layers,
  Rocket
} from 'lucide-react'
import { hfApi, HFSpace, HFSpaceCreateRequest } from '../../services/api'
import { wsService, MessageTypes } from '../../services/websocket'
import { clsx } from 'clsx'

interface SpaceManagerProps {
  className?: string
}

export function SpaceManager({ className }: SpaceManagerProps) {
  const [spaces, setSpaces] = useState<HFSpace[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [formData, setFormData] = useState<HFSpaceCreateRequest>({
    model_repo: '',
    space_name: '',
    private: true,
    gpu_duration: 60,
  })

  const fetchSpaces = useCallback(async () => {
    const result = await hfApi.listSpaces()
    if (result.ok && result.data) {
      setSpaces(result.data)
      setError(null)
    } else if (result.error?.includes('403')) {
      setError('HuggingFace Pro subscription required for ZeroGPU Spaces')
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchSpaces()

    // Subscribe to space events
    const unsubscribeReady = wsService.subscribe(MessageTypes.HF_SPACE_READY, () => {
      fetchSpaces()
    })
    const unsubscribeError = wsService.subscribe(MessageTypes.HF_SPACE_ERROR, () => {
      fetchSpaces()
    })

    return () => {
      unsubscribeReady()
      unsubscribeError()
    }
  }, [fetchSpaces])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setCreating(true)
    setError(null)

    const result = await hfApi.createSpace(formData)
    if (result.ok) {
      setShowForm(false)
      setFormData({
        model_repo: '',
        space_name: '',
        private: true,
        gpu_duration: 60,
      })
      fetchSpaces()
    } else {
      setError(result.error || 'Failed to create Space')
    }
    setCreating(false)
  }

  const handleDelete = async (spaceName: string) => {
    if (!confirm(`Delete Space "${spaceName}"? This cannot be undone.`)) return

    const result = await hfApi.deleteSpace(spaceName)
    if (result.ok) {
      fetchSpaces()
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'building':
        return <Loader2 className="w-4 h-4 text-status-warning animate-spin" />
      case 'running':
        return <CheckCircle2 className="w-4 h-4 text-status-success" />
      case 'stopped':
        return <Clock className="w-4 h-4 text-text-secondary" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-status-error" />
      default:
        return null
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'building':
        return 'Building...'
      case 'running':
        return 'Running'
      case 'stopped':
        return 'Stopped'
      case 'error':
        return 'Error'
      default:
        return status
    }
  }

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
          <h2 className="text-lg font-brand text-text-primary">ZeroGPU Spaces</h2>
          <p className="text-sm text-text-secondary mt-1 font-mono">
            Deploy models to HuggingFace Spaces with free GPU
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchSpaces}
            className="btn-icon"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-text-secondary" />
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="btn-primary flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Space
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-background-card border-2 border-status-error rounded-brutal flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm font-mono">{error}</span>
        </div>
      )}

      {/* Space Creation Form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="mb-6 p-4 border-brutal shadow-brutal-sm rounded-brutal bg-background-card">
          <h3 className="text-sm font-brand text-text-primary mb-4">Create Inference Space</h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Model Repo</label>
              <input
                type="text"
                value={formData.model_repo}
                onChange={(e) => setFormData({ ...formData, model_repo: e.target.value })}
                placeholder="username/model-name"
                className="input w-full text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">Space Name</label>
              <input
                type="text"
                value={formData.space_name}
                onChange={(e) => setFormData({ ...formData, space_name: e.target.value })}
                placeholder="my-inference-space"
                className="input w-full text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-secondary mb-1 uppercase tracking-widest">GPU Duration (seconds)</label>
              <input
                type="number"
                value={formData.gpu_duration}
                onChange={(e) => setFormData({ ...formData, gpu_duration: parseInt(e.target.value) })}
                min={30}
                max={300}
                className="input w-full text-sm"
              />
              <p className="text-xs text-text-secondary mt-1 font-mono">30-300 seconds per request</p>
            </div>
            <div className="flex items-center">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.private}
                  onChange={(e) => setFormData({ ...formData, private: e.target.checked })}
                  className="w-4 h-4 border-brutal rounded-brutal"
                />
                <span className="text-sm text-text-primary font-mono">Private Space</span>
              </label>
            </div>
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
              disabled={creating}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              {creating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Rocket className="w-4 h-4" />
              )}
              Create Space
            </button>
          </div>
        </form>
      )}

      {/* Spaces List */}
      {spaces.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <Layers className="w-12 h-12 mx-auto mb-3 text-text-muted" />
          <p className="font-brand text-lg">No Spaces deployed</p>
          <p className="text-sm mt-1 font-mono">Create a Space to deploy your model for inference</p>
        </div>
      ) : (
        <div className="space-y-3">
          {spaces.map((space) => (
            <div
              key={space.space_name}
              className="card p-4"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(space.status)}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono text-text-primary">
                        {space.space_name}
                      </span>
                      <span className={clsx(
                        'tag',
                        space.status === 'running' ? 'text-status-success' :
                        space.status === 'building' ? 'text-status-warning' :
                        space.status === 'error' ? 'text-status-error' :
                        'text-text-secondary'
                      )}>
                        <span>{getStatusText(space.status)}</span>
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href={space.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-icon"
                    title="Open Space"
                  >
                    <ExternalLink className="w-4 h-4 text-text-secondary" />
                  </a>
                  <button
                    onClick={() => handleDelete(space.space_name)}
                    className="btn-icon"
                    title="Delete Space"
                  >
                    <Trash2 className="w-4 h-4 text-status-error" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
