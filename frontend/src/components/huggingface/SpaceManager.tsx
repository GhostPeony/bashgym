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
          <h2 className="text-lg font-semibold text-text-primary">ZeroGPU Spaces</h2>
          <p className="text-sm text-text-secondary mt-1">
            Deploy models to HuggingFace Spaces with free GPU
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchSpaces}
            className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-text-secondary" />
          </button>
          <button
            onClick={() => setShowForm(!showForm)}
            className="flex items-center gap-2 px-3 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Space
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-status-error/10 border border-status-error/30 rounded-lg flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Space Creation Form */}
      {showForm && (
        <form onSubmit={handleSubmit} className="mb-6 p-4 bg-background-secondary rounded-lg border border-border">
          <h3 className="text-sm font-medium text-text-primary mb-4">Create Inference Space</h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-text-secondary mb-1">Model Repo</label>
              <input
                type="text"
                value={formData.model_repo}
                onChange={(e) => setFormData({ ...formData, model_repo: e.target.value })}
                placeholder="username/model-name"
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">Space Name</label>
              <input
                type="text"
                value={formData.space_name}
                onChange={(e) => setFormData({ ...formData, space_name: e.target.value })}
                placeholder="my-inference-space"
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-text-secondary mb-1">GPU Duration (seconds)</label>
              <input
                type="number"
                value={formData.gpu_duration}
                onChange={(e) => setFormData({ ...formData, gpu_duration: parseInt(e.target.value) })}
                min={30}
                max={300}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
              />
              <p className="text-xs text-text-secondary mt-1">30-300 seconds per request</p>
            </div>
            <div className="flex items-center">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.private}
                  onChange={(e) => setFormData({ ...formData, private: e.target.checked })}
                  className="w-4 h-4 rounded border-border"
                />
                <span className="text-sm text-text-primary">Private Space</span>
              </label>
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
              disabled={creating}
              className="flex items-center gap-2 px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 transition-colors"
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
          <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No Spaces deployed</p>
          <p className="text-sm mt-1">Create a Space to deploy your model for inference</p>
        </div>
      ) : (
        <div className="space-y-3">
          {spaces.map((space) => (
            <div
              key={space.space_name}
              className="p-4 bg-background-secondary rounded-lg border border-border"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(space.status)}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text-primary">
                        {space.space_name}
                      </span>
                      <span className={clsx(
                        'px-2 py-0.5 text-xs rounded',
                        space.status === 'running' ? 'bg-status-success/20 text-status-success' :
                        space.status === 'building' ? 'bg-status-warning/20 text-status-warning' :
                        space.status === 'error' ? 'bg-status-error/20 text-status-error' :
                        'bg-background-tertiary text-text-secondary'
                      )}>
                        {getStatusText(space.status)}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href={space.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
                    title="Open Space"
                  >
                    <ExternalLink className="w-4 h-4 text-text-secondary" />
                  </a>
                  <button
                    onClick={() => handleDelete(space.space_name)}
                    className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
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
