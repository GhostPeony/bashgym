import { useState } from 'react'
import {
  HardDrive,
  Plus,
  Trash2,
  RefreshCw,
  FolderOpen,
  File,
  Copy,
  Loader2,
  ArrowRight,
} from 'lucide-react'
import { hfApi } from '../../services/api'
import { hfBucketsResource, hfBucketTreeResource } from '../../stores/hfResources'
import { useKeyedSessionResource, useSessionResource } from '../../stores/sessionResource'

export function BucketsTab() {
  const { data: bucketsData, loading, refreshing, refresh } = useSessionResource(hfBucketsResource)
  const buckets = bucketsData ?? []
  const [creating, setCreating] = useState(false)
  const [newBucketId, setNewBucketId] = useState('')
  const [newBucketPrivate, setNewBucketPrivate] = useState(true)
  const [selectedBucket, setSelectedBucket] = useState<string | null>(null)
  const { data: filesData, loading: filesLoading } = useKeyedSessionResource(
    hfBucketTreeResource,
    selectedBucket ?? ''
  )
  const files = filesData ?? []
  const [error, setError] = useState<string | null>(null)

  // Copy dialog
  const [showCopy, setShowCopy] = useState(false)
  const [copySource, setCopySource] = useState('')
  const [copyDest, setCopyDest] = useState('')
  const [copying, setCopying] = useState(false)

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newBucketId.trim()) return
    setCreating(true)
    setError(null)
    const result = await hfApi.createBucket({ bucket_id: newBucketId.trim(), private: newBucketPrivate })
    if (result.ok) {
      setNewBucketId('')
      await refresh()
    } else {
      setError(result.error || 'Failed to create bucket')
    }
    setCreating(false)
  }

  const handleDelete = async (bucketId: string) => {
    if (!confirm(`Delete bucket "${bucketId}"? This cannot be undone.`)) return
    const result = await hfApi.deleteBucket(bucketId)
    if (result.ok) {
      if (selectedBucket === bucketId) {
        setSelectedBucket(null)
      }
      await refresh()
    }
  }

  const handleBrowse = (bucketId: string) => {
    setSelectedBucket(bucketId)
  }

  const handleCopy = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!copySource.trim() || !copyDest.trim()) return
    setCopying(true)
    setError(null)
    const result = await hfApi.copyFiles({ source: copySource.trim(), destination: copyDest.trim() })
    if (result.ok) {
      setShowCopy(false)
      setCopySource('')
      setCopyDest('')
    } else {
      setError(result.error || 'Copy failed')
    }
    setCopying(false)
  }

  const formatSize = (bytes?: number) => {
    if (!bytes) return '—'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`
    return `${(bytes / 1073741824).toFixed(1)} GB`
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-brand text-text-primary">Storage Buckets</h2>
          <p className="text-sm text-text-secondary mt-1">
            Mutable S3-like storage for checkpoints, logs, and data shards. $12/TB/month.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setShowCopy(true)} className="btn-ghost flex items-center gap-2 text-sm">
            <Copy className="w-4 h-4" />
            Instant Copy
          </button>
          <button onClick={() => refresh()} className="btn-icon" title="Refresh">
            <RefreshCw className={`w-4 h-4${refreshing ? ' animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 border-2 border-status-error rounded-brutal text-sm text-status-error">
          {error}
        </div>
      )}

      {/* Create form */}
      <form onSubmit={handleCreate} className="flex gap-2 items-end">
        <div className="flex-1">
          <label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">
            Bucket ID
          </label>
          <input
            type="text"
            value={newBucketId}
            onChange={(e) => setNewBucketId(e.target.value)}
            placeholder="username/my-training-bucket"
            className="input w-full text-sm"
          />
        </div>
        <label className="flex items-center gap-2 text-sm text-text-secondary pb-2">
          <input
            type="checkbox"
            checked={newBucketPrivate}
            onChange={(e) => setNewBucketPrivate(e.target.checked)}
          />
          Private
        </label>
        <button type="submit" disabled={creating} className="btn-primary flex items-center gap-2 text-sm">
          {creating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
          Create
        </button>
      </form>

      {/* Bucket list */}
      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-text-secondary" />
        </div>
      ) : buckets.length === 0 ? (
        <div className="text-center py-8 text-text-secondary">
          <HardDrive className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No buckets yet. Create one to store training artifacts.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {buckets.map((b) => (
            <div
              key={b.id}
              className={`border-2 rounded-brutal p-3 flex items-center justify-between cursor-pointer transition-press hover-press ${
                selectedBucket === b.id ? 'border-accent bg-accent-light' : 'border-border hover:bg-background-secondary'
              }`}
              onClick={() => handleBrowse(b.id)}
            >
              <div className="flex items-center gap-3">
                <HardDrive className="w-5 h-5 text-text-secondary" />
                <div>
                  <span className="font-mono text-sm text-text-primary">{b.id}</span>
                  {b.private && (
                    <span className="ml-2 text-xs text-text-muted border border-border-subtle rounded px-1">
                      private
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(b.id) }}
                  className="btn-icon text-status-error hover:bg-background-secondary"
                  title="Delete bucket"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* File browser */}
      {selectedBucket && (
        <div className="border-2 border-border rounded-brutal">
          <div className="p-3 border-b-2 border-border bg-background-secondary flex items-center gap-2">
            <FolderOpen className="w-4 h-4 text-accent" />
            <span className="font-mono text-sm text-text-primary">{selectedBucket}</span>
          </div>
          {filesLoading ? (
            <div className="flex justify-center py-6">
              <Loader2 className="w-5 h-5 animate-spin text-text-secondary" />
            </div>
          ) : files.length === 0 ? (
            <div className="p-6 text-center text-text-secondary text-sm">
              Empty bucket. Use sync to upload files.
            </div>
          ) : (
            <div className="divide-y divide-border">
              {files.map((f, i) => (
                <div key={i} className="px-3 py-2 flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    {f.type === 'folder' ? (
                      <FolderOpen className="w-4 h-4 text-accent" />
                    ) : (
                      <File className="w-4 h-4 text-text-secondary" />
                    )}
                    <span className="font-mono text-text-primary">{f.name}</span>
                  </div>
                  <span className="text-text-muted font-mono text-xs">{formatSize(f.size)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Copy dialog */}
      {showCopy && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowCopy(false)}>
          <div className="bg-background-card border-2 border-border rounded-brutal p-6 w-full max-w-lg shadow-brutal" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-brand text-text-primary mb-4 flex items-center gap-2">
              <Copy className="w-5 h-5" />
              Instant File Copy
            </h3>
            <p className="text-sm text-text-secondary mb-4">
              Server-side copy with zero bandwidth for Xet-tracked files. Use <code className="text-xs bg-background-secondary px-1 rounded">hf://</code> URIs.
            </p>
            <form onSubmit={handleCopy} className="space-y-3">
              <div>
                <label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">Source</label>
                <input
                  type="text"
                  value={copySource}
                  onChange={(e) => setCopySource(e.target.value)}
                  placeholder="hf://datasets/org/source-repo/path/"
                  className="input w-full text-sm font-mono"
                />
              </div>
              <div className="flex justify-center">
                <ArrowRight className="w-5 h-5 text-text-muted" />
              </div>
              <div>
                <label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">Destination</label>
                <input
                  type="text"
                  value={copyDest}
                  onChange={(e) => setCopyDest(e.target.value)}
                  placeholder="hf://buckets/org/dest-bucket/path/"
                  className="input w-full text-sm font-mono"
                />
              </div>
              <div className="flex gap-2 justify-end pt-2">
                <button type="button" onClick={() => setShowCopy(false)} className="btn-ghost text-sm">Cancel</button>
                <button type="submit" disabled={copying} className="btn-primary flex items-center gap-2 text-sm">
                  {copying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Copy className="w-4 h-4" />}
                  Copy
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
