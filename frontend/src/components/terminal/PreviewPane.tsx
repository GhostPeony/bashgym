import { useState, useEffect, useCallback } from 'react'
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

  const loadFile = useCallback(async () => {
    if (!filePath) return
    setLoading(true)
    try {
      const result = await window.bashgym?.files.readFile(filePath)
      if (result?.success && result.content !== undefined) {
        setContent(result.content)
      } else {
        setContent(`// Could not load file${result?.error ? ':\n// ' + result.error : ''}`)
      }
    } catch {
      setContent(`// Failed to load file`)
    } finally {
      setLoading(false)
    }
  }, [filePath])

  // Auto-load when filePath changes
  useEffect(() => {
    if (filePath) {
      loadFile()
    } else {
      setContent(null)
    }
  }, [filePath, loadFile])

  return (
    <div className="terminal-chrome h-full flex flex-col">
      {/* Header — terminal-header */}
      <div className="terminal-header">
        <div className="flex-1 flex items-center gap-2">
          <FileText className="w-4 h-4 text-text-muted" />
          <span className="text-sm font-mono text-text-primary truncate max-w-[200px]">
            {filePath || title}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {filePath && (
            <button
              onClick={loadFile}
              className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary"
              title="Refresh"
            >
              <RefreshCw className="w-3.5 h-3.5 text-text-muted" />
            </button>
          )}
          <button
            onClick={handleClose}
            className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-status-error/20"
            title="Close"
          >
            <X className="w-3.5 h-3.5 text-text-muted hover:text-status-error" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4 bg-background-terminal">
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
            <div className="w-16 h-16 border-brutal border-border rounded-brutal flex items-center justify-center mb-4 bg-background-secondary">
              <FileText className="w-8 h-8 text-text-muted" />
            </div>
            <p className="text-sm text-text-muted font-mono">
              {filePath ? 'Click refresh to load file' : 'No file selected'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
