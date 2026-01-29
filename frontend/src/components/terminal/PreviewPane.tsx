import { useState } from 'react'
import { FileText, X, RefreshCw } from 'lucide-react'
import { useTerminalStore } from '../../stores'

interface PreviewPaneProps {
  id: string
  title: string
  filePath?: string
  isActive: boolean
}

export function PreviewPane({ id, title, filePath, isActive }: PreviewPaneProps) {
  const { removePanel } = useTerminalStore()
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation()
    removePanel(id)
  }

  const loadFile = async () => {
    if (!filePath) return
    setLoading(true)
    // In real implementation, this would read the file
    // For now, show placeholder
    setTimeout(() => {
      setContent(`// File: ${filePath}\n\n// Content would appear here`)
      setLoading(false)
    }, 500)
  }

  return (
    <div className="h-full flex flex-col bg-background-secondary">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-background-secondary border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-text-muted" />
          <span className="text-sm font-medium text-text-primary truncate max-w-[200px]">
            {filePath || title}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {filePath && (
            <button
              onClick={loadFile}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={handleClose}
            className="p-1 rounded hover:bg-status-error/20 text-text-muted hover:text-status-error"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <RefreshCw className="w-6 h-6 text-text-muted animate-spin" />
          </div>
        ) : content ? (
          <pre className="text-sm font-mono text-text-primary whitespace-pre-wrap">
            {content}
          </pre>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <FileText className="w-12 h-12 text-text-muted mb-4" />
            <p className="text-sm text-text-muted">
              {filePath ? 'Click refresh to load file' : 'No file selected'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
