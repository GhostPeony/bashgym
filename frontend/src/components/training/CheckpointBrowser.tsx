import { useCallback, useEffect, useState } from 'react'
import { RefreshCw, Trash2, ArrowRight, Package } from 'lucide-react'
import { clsx } from 'clsx'
import { trainingApi } from '../../services/api'
import { useTrainingStore } from '../../stores'

export interface CheckpointInfo {
  id: string
  run_id: string
  kind: 'final' | 'merged' | 'intermediate' | 'gguf'
  path: string
  size_mb: number
  created_at: string
  base_model: string | null
}

function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb.toFixed(0)} MB`
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export function CheckpointBrowser() {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [pruningRun, setPruningRun] = useState<string | null>(null)
  const setBaseModelOverride = useTrainingStore((s) => s.setBaseModelOverride)

  const fetchCheckpoints = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await trainingApi.listCheckpoints()
      if (res.ok && res.data) {
        setCheckpoints(res.data)
      } else {
        setError(res.error || 'Failed to load checkpoints')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCheckpoints()
  }, [fetchCheckpoints])

  const handleUseAsBase = (cp: CheckpointInfo) => {
    setBaseModelOverride(cp.path)
  }

  const handleDelete = async (cp: CheckpointInfo) => {
    const confirmed = window.confirm(
      `Delete checkpoint?\n\n${cp.id}\n${formatSize(cp.size_mb)}\n\nThis is permanent.`
    )
    if (!confirmed) return
    setDeleting(cp.id)
    setError(null)
    try {
      const res = await trainingApi.deleteCheckpoint(cp.id)
      if (!res.ok) {
        setError(res.error || 'Delete failed')
      } else {
        setCheckpoints((prev) => prev.filter((c) => c.id !== cp.id))
      }
    } finally {
      setDeleting(null)
    }
  }

  const handleKeepAdapterOnly = async (runId: string) => {
    const removable = checkpoints.filter((cp) => cp.run_id === runId && cp.kind !== 'final')
    if (removable.length === 0) return
    const removableMb = removable.reduce((sum, cp) => sum + cp.size_mb, 0)
    const confirmed = window.confirm(
      `Keep only the final adapter for ${runId}?\n\n` +
      `This permanently deletes ${removable.length} merged/checkpoint/GGUF artifact(s) ` +
      `and frees about ${formatSize(removableMb)}.`,
    )
    if (!confirmed) return
    setPruningRun(runId)
    setError(null)
    try {
      const deleted = new Set<string>()
      for (const cp of removable) {
        const res = await trainingApi.deleteCheckpoint(cp.id)
        if (!res.ok) {
          setError(res.error || `Failed to delete ${cp.id}`)
          break
        }
        deleted.add(cp.id)
      }
      setCheckpoints((prev) => prev.filter((cp) => !deleted.has(cp.id)))
    } finally {
      setPruningRun(null)
    }
  }

  const totalSizeMb = checkpoints.reduce((sum, cp) => sum + cp.size_mb, 0)

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-brand text-xl text-text-primary">Checkpoints</h2>
          <p className="font-mono text-xs text-text-muted">
            {formatSize(totalSizeMb)} across local training artifacts
          </p>
        </div>
        <button
          onClick={fetchCheckpoints}
          disabled={loading}
          className="btn-icon flex items-center justify-center"
          title="Refresh"
        >
          <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
        </button>
      </div>

      {error && (
        <div className="card p-3 border-l-4 border-l-status-error bg-status-error/10 mb-3">
          <p className="font-mono text-xs text-status-error">{error}</p>
        </div>
      )}

      {checkpoints.length === 0 && !loading && !error && (
        <div className="text-center py-12 text-text-muted">
          <Package className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="font-mono text-xs uppercase tracking-widest">No checkpoints found</p>
        </div>
      )}

      {checkpoints.length > 0 && (
        <div className="overflow-auto">
          <table className="w-full font-mono text-xs">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Run
                </th>
                <th className="text-left py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Kind
                </th>
                <th className="text-left py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Base Model
                </th>
                <th className="text-right py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Size
                </th>
                <th className="text-left py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Created
                </th>
                <th className="text-right py-2 px-2 text-text-muted uppercase tracking-widest font-normal">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((cp) => (
                <tr key={cp.id} className="border-b border-border-subtle hover:bg-background-secondary">
                  <td className="py-2 px-2 text-text-primary">{cp.run_id}</td>
                  <td className="py-2 px-2">
                    <span className="tag text-[10px] py-0 px-1.5">
                      <span>{cp.kind}</span>
                    </span>
                  </td>
                  <td className="py-2 px-2 text-text-secondary truncate max-w-xs">
                    {cp.base_model || '—'}
                  </td>
                  <td className="py-2 px-2 text-right text-text-secondary">
                    {formatSize(cp.size_mb)}
                  </td>
                  <td className="py-2 px-2 text-text-secondary">{formatDate(cp.created_at)}</td>
                  <td className="py-2 px-2">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => handleUseAsBase(cp)}
                        className="btn-secondary flex items-center gap-1 py-1 px-2 text-[10px]"
                        title="Use as base model for next run"
                      >
                        <ArrowRight className="w-3 h-3" />
                        Use as base
                      </button>
                      {cp.kind === 'final' && checkpoints.some((candidate) => candidate.run_id === cp.run_id && candidate.kind !== 'final') && (
                        <button
                          onClick={() => handleKeepAdapterOnly(cp.run_id)}
                          disabled={pruningRun === cp.run_id}
                          className="btn-secondary flex items-center gap-1 py-1 px-2 text-[10px]"
                          title="Delete merged weights, GGUF exports, and intermediate checkpoints for this run"
                        >
                          {pruningRun === cp.run_id ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Package className="w-3 h-3" />}
                          Keep adapter only
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(cp)}
                        disabled={deleting === cp.id}
                        className="btn-icon w-7 h-7 flex items-center justify-center text-status-error"
                        title="Delete"
                      >
                        {deleting === cp.id ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          <Trash2 className="w-3 h-3" />
                        )}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
