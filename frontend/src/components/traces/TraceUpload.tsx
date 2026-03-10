import { useState, useCallback, useRef } from 'react'
import { Upload, Loader2, Check, XCircle, AlertTriangle } from 'lucide-react'
import clsx from 'clsx'
import { tracesApi } from '../../services/api'

type UploadSource = 'chatgpt' | 'mcp'

interface TraceUploadProps {
  onImportComplete: () => void
}

interface UploadResult {
  imported_count: number
  skipped_count: number
  failed_count: number
  total_steps: number
  errors: string[]
}

export default function TraceUpload({ onImportComplete }: TraceUploadProps) {
  const [source, setSource] = useState<UploadSource>('chatgpt')
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const acceptedTypes: Record<UploadSource, string> = {
    chatgpt: '.zip,.json',
    mcp: '.json,.jsonl',
  }

  const sourceHints: Record<UploadSource, string> = {
    chatgpt: 'ZIP export or conversations.json',
    mcp: 'JSON-RPC log or simple .json/.jsonl',
  }

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true)
    setResult(null)
    setError(null)

    const resp = await tracesApi.uploadAndImport(file, source)

    if (resp.ok && resp.data) {
      setResult(resp.data)
      if (resp.data.imported_count > 0) {
        onImportComplete()
      }
    } else {
      setError(resp.error || 'Upload failed')
    }

    setUploading(false)
  }, [source, onImportComplete])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [handleUpload])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const onFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [handleUpload])

  return (
    <div className="space-y-2">
      {/* Source selector */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-mono uppercase tracking-wider text-text-muted">Upload:</span>
        {(['chatgpt', 'mcp'] as const).map((s) => (
          <button
            key={s}
            onClick={() => { setSource(s); setResult(null); setError(null) }}
            disabled={uploading}
            className={clsx(
              'px-2 py-1 text-[10px] font-mono font-semibold uppercase tracking-wide border-brutal border-border rounded-brutal transition-press',
              source === s
                ? 'bg-accent-light text-accent-dark shadow-brutal-sm'
                : 'bg-background-card text-text-secondary hover-press',
              uploading && 'opacity-50 cursor-not-allowed'
            )}
          >
            {s === 'chatgpt' ? 'ChatGPT' : 'MCP Logs'}
          </button>
        ))}
      </div>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !uploading && fileInputRef.current?.click()}
        className={clsx(
          'relative flex flex-col items-center justify-center gap-1.5 py-4 px-3 border-2 border-dashed rounded-brutal cursor-pointer transition-all',
          isDragging
            ? 'border-accent bg-accent-light/40'
            : 'border-border hover:border-accent/50 hover:bg-background-secondary/50',
          uploading && 'pointer-events-none opacity-60'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedTypes[source]}
          onChange={onFileSelect}
          className="hidden"
        />

        {uploading ? (
          <Loader2 className="w-5 h-5 text-accent animate-spin" />
        ) : (
          <Upload className="w-5 h-5 text-text-muted" />
        )}

        <span className="text-xs font-mono text-text-secondary text-center">
          {uploading ? 'Importing...' : 'Drop file or click to browse'}
        </span>
        <span className="text-[10px] font-mono text-text-muted">
          {sourceHints[source]}
        </span>
      </div>

      {/* Result */}
      {result && (
        <div className={clsx(
          'flex items-start gap-2 px-2.5 py-1.5 text-[11px] font-mono border-brutal rounded-brutal',
          result.failed_count > 0
            ? 'border-status-error/30 bg-status-error/5 text-status-error'
            : 'border-status-success/30 bg-status-success/5 text-status-success'
        )}>
          {result.failed_count > 0 ? (
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          ) : (
            <Check className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          )}
          <div>
            <span className="font-semibold">
              {result.imported_count} imported
            </span>
            {result.skipped_count > 0 && (
              <span className="text-text-muted"> / {result.skipped_count} skipped</span>
            )}
            {result.total_steps > 0 && (
              <span className="text-text-muted"> / {result.total_steps} steps</span>
            )}
            {result.errors.length > 0 && (
              <div className="mt-1 text-status-error">
                {result.errors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-2.5 py-1.5 text-[11px] font-mono border-brutal border-status-error/30 bg-status-error/5 text-status-error rounded-brutal">
          <XCircle className="w-3.5 h-3.5 shrink-0" />
          {error}
        </div>
      )}
    </div>
  )
}
