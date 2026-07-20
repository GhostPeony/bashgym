import { useState } from 'react'
import {
  FileText,
  Upload,
  Loader2,
  ExternalLink,
  RefreshCw,
} from 'lucide-react'
import { hfApi } from '../../services/api'
import { hfTraceDatasetsResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'

const DEFAULT_TRACE_DIRS = [
  '~/.bashgym/gold_traces_local',
  '~/.bashgym/failed_traces_local',
  '~/.bashgym/traces',
]

export function TracesTab() {
  const { data, loading, refreshing, refresh } = useSessionResource(hfTraceDatasetsResource)
  const datasets = data ?? []
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Upload form
  const [traceDir, setTraceDir] = useState(DEFAULT_TRACE_DIRS[0])
  const [repoId, setRepoId] = useState('')
  const [isPrivate, setIsPrivate] = useState(true)

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!traceDir.trim() || !repoId.trim()) return

    setUploading(true)
    setError(null)
    setSuccess(null)

    const result = await hfApi.uploadTraces({
      trace_dir: traceDir.trim(),
      repo_id: repoId.trim(),
      private: isPrivate,
    })

    if (result.ok && result.data) {
      setSuccess(`Uploaded ${result.data.num_traces} traces to ${result.data.url}`)
      await refresh()
    } else {
      setError(result.error || 'Upload failed')
    }
    setUploading(false)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-brand text-text-primary">Agent Traces</h2>
          <p className="text-sm text-text-secondary mt-1">
            Upload bashgym traces to HuggingFace Hub. HF auto-detects agent trace formats and provides a specialized viewer.
          </p>
        </div>
        <button onClick={() => refresh()} className="btn-icon" title="Refresh">
          <RefreshCw className={`w-4 h-4${refreshing ? ' animate-spin' : ''}`} />
        </button>
      </div>

      {error && (
        <div className="p-3 border-2 border-status-error rounded-brutal text-sm text-status-error">
          {error}
        </div>
      )}
      {success && (
        <div className="p-3 border-2 border-status-success rounded-brutal text-sm text-status-success">
          {success}
        </div>
      )}

      {/* Upload form */}
      <div className="border-2 border-border rounded-brutal p-4 bg-background-card">
        <h3 className="text-sm font-brand text-text-primary mb-3 flex items-center gap-2">
          <Upload className="w-4 h-4" />
          Upload Traces to Hub
        </h3>
        <form onSubmit={handleUpload} className="space-y-3">
          <div>
            <label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">
              Trace Directory
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={traceDir}
                onChange={(e) => setTraceDir(e.target.value)}
                className="input flex-1 text-sm font-mono"
              />
              <div className="flex gap-1">
                {DEFAULT_TRACE_DIRS.map((dir) => (
                  <button
                    key={dir}
                    type="button"
                    onClick={() => setTraceDir(dir)}
                    className={`text-xs px-2 py-1 rounded border font-mono ${
                      traceDir === dir
                        ? 'border-accent text-accent bg-accent-light'
                        : 'border-border-subtle text-text-muted hover:text-text-secondary'
                    }`}
                  >
                    {dir.split('/').pop()}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div>
            <label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">
              HF Dataset Repo ID
            </label>
            <input
              type="text"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder="username/bashgym-gold-traces"
              className="input w-full text-sm font-mono"
            />
          </div>
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 text-sm text-text-secondary">
              <input
                type="checkbox"
                checked={isPrivate}
                onChange={(e) => setIsPrivate(e.target.checked)}
              />
              Private dataset
            </label>
            <button
              type="submit"
              disabled={uploading || !traceDir.trim() || !repoId.trim()}
              className="btn-primary flex items-center gap-2 text-sm"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Existing trace datasets */}
      <div>
        <h3 className="text-sm font-brand text-text-primary mb-3">Your Trace Datasets on Hub</h3>
        {loading ? (
          <div className="flex justify-center py-6">
            <Loader2 className="w-5 h-5 animate-spin text-text-secondary" />
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-8 text-text-secondary">
            <FileText className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No trace datasets found. Upload your first traces above.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds) => (
              <div key={ds.id} className="border-2 border-border rounded-brutal p-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-text-secondary" />
                  <div>
                    <span className="font-mono text-sm text-text-primary">{ds.id}</span>
                    <div className="flex items-center gap-3 mt-0.5">
                      {ds.private && (
                        <span className="text-xs text-text-muted border border-border-subtle rounded px-1">private</span>
                      )}
                      <span className="text-xs text-text-muted">{ds.downloads} downloads</span>
                    </div>
                  </div>
                </div>
                <a
                  href={`https://huggingface.co/datasets/${ds.id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-icon"
                  title="View on HuggingFace"
                >
                  <ExternalLink className="w-4 h-4 text-text-secondary" />
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
