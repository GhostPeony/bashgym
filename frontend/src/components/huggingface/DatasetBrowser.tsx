import { useState, useEffect, useCallback } from 'react'
import {
  Upload,
  Trash2,
  RefreshCw,
  Loader2,
  ExternalLink,
  AlertCircle,
  Database,
  FolderOpen
} from 'lucide-react'
import { hfApi } from '../../services/api'
import { clsx } from 'clsx'

interface DatasetBrowserProps {
  className?: string
}

export function DatasetBrowser({ className }: DatasetBrowserProps) {
  const [datasets, setDatasets] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [prefix, setPrefix] = useState('bashgym')

  const fetchDatasets = useCallback(async () => {
    setLoading(true)
    const result = await hfApi.listDatasets(prefix)
    if (result.ok && result.data) {
      setDatasets(result.data)
      setError(null)
    } else {
      setError(result.error || 'Failed to fetch datasets')
    }
    setLoading(false)
  }, [prefix])

  useEffect(() => {
    fetchDatasets()
  }, [fetchDatasets])

  const handleDelete = async (repoName: string) => {
    if (!confirm(`Delete dataset "${repoName}"? This cannot be undone.`)) return

    const result = await hfApi.deleteDataset(repoName)
    if (result.ok) {
      fetchDatasets()
    } else {
      setError(result.error || 'Failed to delete dataset')
    }
  }

  const getDatasetUrl = (repoId: string) => {
    return `https://huggingface.co/datasets/${repoId}`
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
          <h2 className="text-lg font-semibold text-text-primary">Datasets</h2>
          <p className="text-sm text-text-secondary mt-1">
            Training datasets stored on HuggingFace Hub
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-secondary">Filter:</span>
            <input
              type="text"
              value={prefix}
              onChange={(e) => setPrefix(e.target.value)}
              placeholder="bashgym"
              className="w-32 px-2 py-1 bg-background-primary border border-border rounded text-text-primary text-sm"
            />
          </div>
          <button
            onClick={fetchDatasets}
            className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-text-secondary" />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-status-error/10 border border-status-error/30 rounded-lg flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Info Box */}
      <div className="mb-6 p-4 bg-background-secondary rounded-lg border border-border">
        <div className="flex items-start gap-3">
          <Upload className="w-5 h-5 text-accent-primary flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-text-primary">Upload from Training Dashboard</h3>
            <p className="text-sm text-text-secondary mt-1">
              Datasets are uploaded automatically when you export training examples from the Training Dashboard.
              Use the "Export to HuggingFace" option to push your training data to the Hub.
            </p>
          </div>
        </div>
      </div>

      {/* Datasets List */}
      {datasets.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No datasets found</p>
          <p className="text-sm mt-1">
            {prefix ? `No datasets matching "${prefix}"` : 'Export training data to see datasets here'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {datasets.map((datasetId) => (
            <div
              key={datasetId}
              className="flex items-center justify-between p-3 bg-background-secondary rounded-lg border border-border hover:border-border-subtle transition-colors"
            >
              <div className="flex items-center gap-3">
                <FolderOpen className="w-5 h-5 text-accent-primary" />
                <span className="text-sm text-text-primary font-medium">{datasetId}</span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={getDatasetUrl(datasetId)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 px-2 py-1 text-xs text-accent-primary hover:bg-accent-primary/10 rounded transition-colors"
                >
                  <ExternalLink className="w-3 h-3" />
                  Data Studio
                </a>
                <button
                  onClick={() => handleDelete(datasetId)}
                  className="p-1.5 rounded hover:bg-background-tertiary transition-colors"
                  title="Delete Dataset"
                >
                  <Trash2 className="w-4 h-4 text-status-error" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pro Storage Info */}
      <div className="mt-6 p-4 bg-accent-primary/10 rounded-lg border border-accent-primary/30">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-accent-primary" />
          <span className="text-sm text-text-primary">HuggingFace Pro Storage</span>
        </div>
        <p className="text-xs text-text-secondary mt-2">
          Pro subscribers get 1TB of private dataset storage with Data Studio access for exploring and visualizing your training data.
        </p>
      </div>
    </div>
  )
}
