import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import {
  FileText,
  FileCode,
  FileJson,
  FileImage,
  File,
  Maximize2,
  X,
  Copy,
  ExternalLink,
  Clock,
  HardDrive,
  Hash
} from 'lucide-react'
import { clsx } from 'clsx'

export interface PreviewNodeData {
  panelId: string
  filePath: string
  fileSize?: number
  lineCount?: number
  language?: string
  previewLines?: string[]
  lastModified?: number
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  onCopyPath?: (path: string) => void
  onOpenExternal?: (path: string) => void
}

export type PreviewNodeType = NodeProps<PreviewNodeData>

// Get language icon based on file extension/language
function getLanguageIcon(language?: string, filePath?: string) {
  const ext = filePath?.split('.').pop()?.toLowerCase()
  const lang = language?.toLowerCase() || ext

  // Code files
  if (['ts', 'tsx', 'typescript'].includes(lang || '')) {
    return <FileCode className="w-4 h-4 text-blue-400" />
  }
  if (['js', 'jsx', 'javascript'].includes(lang || '')) {
    return <FileCode className="w-4 h-4 text-yellow-400" />
  }
  if (['py', 'python'].includes(lang || '')) {
    return <FileCode className="w-4 h-4 text-green-400" />
  }
  if (['rs', 'rust'].includes(lang || '')) {
    return <FileCode className="w-4 h-4 text-orange-400" />
  }
  if (['go', 'golang'].includes(lang || '')) {
    return <FileCode className="w-4 h-4 text-cyan-400" />
  }

  // Data files
  if (['json', 'jsonl'].includes(lang || '')) {
    return <FileJson className="w-4 h-4 text-amber-400" />
  }
  if (['yaml', 'yml'].includes(lang || '')) {
    return <FileJson className="w-4 h-4 text-red-400" />
  }

  // Images
  if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'].includes(ext || '')) {
    return <FileImage className="w-4 h-4 text-purple-400" />
  }

  // Text files
  if (['md', 'markdown', 'txt'].includes(lang || '')) {
    return <FileText className="w-4 h-4 text-text-muted" />
  }

  return <File className="w-4 h-4 text-text-muted" />
}

// Get language name for display
function getLanguageName(language?: string, filePath?: string): string {
  const ext = filePath?.split('.').pop()?.toLowerCase()
  const lang = language?.toLowerCase() || ext

  const names: Record<string, string> = {
    ts: 'TypeScript',
    tsx: 'TypeScript React',
    js: 'JavaScript',
    jsx: 'JavaScript React',
    py: 'Python',
    rs: 'Rust',
    go: 'Go',
    json: 'JSON',
    jsonl: 'JSON Lines',
    yaml: 'YAML',
    yml: 'YAML',
    md: 'Markdown',
    txt: 'Text',
    html: 'HTML',
    css: 'CSS',
    scss: 'SCSS',
    sql: 'SQL',
    sh: 'Shell',
    bash: 'Bash'
  }

  return names[lang || ''] || (ext ? ext.toUpperCase() : 'Unknown')
}

// Format file size
function formatFileSize(bytes?: number): string {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Format relative time
function formatRelativeTime(timestamp?: number): string {
  if (!timestamp) return ''
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

// Get filename from path
function getFileName(path: string): string {
  return path.split(/[/\\]/).pop() || path
}

// Get syntax highlighting color based on language
function getSyntaxColor(language?: string): string {
  const lang = language?.toLowerCase()
  if (['ts', 'tsx', 'typescript'].includes(lang || '')) return 'border-l-blue-400'
  if (['js', 'jsx', 'javascript'].includes(lang || '')) return 'border-l-yellow-400'
  if (['py', 'python'].includes(lang || '')) return 'border-l-green-400'
  if (['rs', 'rust'].includes(lang || '')) return 'border-l-orange-400'
  if (['go'].includes(lang || '')) return 'border-l-cyan-400'
  if (['json', 'yaml', 'yml'].includes(lang || '')) return 'border-l-amber-400'
  return 'border-l-border-subtle'
}

export const PreviewNode = memo(function PreviewNode({ data, selected }: PreviewNodeType) {
  const {
    panelId,
    filePath,
    fileSize,
    lineCount,
    language,
    previewLines,
    lastModified,
    onFocus,
    onClose,
    onCopyPath,
    onOpenExternal
  } = data

  const handleFocus = useCallback(() => {
    onFocus?.(panelId)
  }, [panelId, onFocus])

  const handleClose = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onClose?.(panelId)
  }, [panelId, onClose])

  const handleCopyPath = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onCopyPath?.(filePath)
  }, [filePath, onCopyPath])

  const handleOpenExternal = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onOpenExternal?.(filePath)
  }, [filePath, onOpenExternal])

  const fileName = getFileName(filePath)
  const langName = getLanguageName(language, filePath)
  const relativeTime = formatRelativeTime(lastModified)

  return (
    <div
      className={clsx(
        'w-[300px] bg-background-secondary rounded-lg border-2 shadow-lg transition-all cursor-pointer',
        'border-border-subtle hover:border-border',
        selected && 'ring-2 ring-primary border-primary'
      )}
      onClick={handleFocus}
    >
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-primary !w-2 !h-2"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-primary !w-2 !h-2"
      />

      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 bg-background-tertiary/50 rounded-t-lg">
        <div className="p-1.5 rounded-md bg-background-tertiary">
          {getLanguageIcon(language, filePath)}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-text-primary truncate block" title={filePath}>
            {fileName}
          </span>
          <span className="text-[10px] text-text-muted">{langName}</span>
        </div>
        <div className="flex items-center gap-0.5">
          {onCopyPath && (
            <button
              onClick={handleCopyPath}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
              title="Copy path"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
          )}
          {onOpenExternal && (
            <button
              onClick={handleOpenExternal}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
              title="Open externally"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={handleFocus}
            className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
            title="Focus panel"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleClose}
            className="p-1 rounded hover:bg-status-error/20 text-text-muted hover:text-status-error"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* File metadata */}
      <div className="flex items-center gap-3 px-3 py-1.5 text-[10px] text-text-muted border-b border-border-subtle">
        {fileSize !== undefined && (
          <div className="flex items-center gap-1">
            <HardDrive className="w-2.5 h-2.5" />
            <span>{formatFileSize(fileSize)}</span>
          </div>
        )}
        {lineCount !== undefined && (
          <div className="flex items-center gap-1">
            <Hash className="w-2.5 h-2.5" />
            <span>{lineCount} lines</span>
          </div>
        )}
        {relativeTime && (
          <div className="flex items-center gap-1">
            <Clock className="w-2.5 h-2.5" />
            <span>{relativeTime}</span>
          </div>
        )}
      </div>

      {/* Code preview */}
      <div className={clsx(
        'px-3 py-2 font-mono text-[10px] leading-relaxed',
        'border-l-2',
        getSyntaxColor(language)
      )}>
        {previewLines && previewLines.length > 0 ? (
          <div className="space-y-0.5 max-h-[120px] overflow-hidden">
            {previewLines.slice(0, 8).map((line, idx) => (
              <div key={idx} className="flex">
                <span className="text-text-muted w-6 text-right mr-2 select-none flex-shrink-0">
                  {idx + 1}
                </span>
                <span className="text-text-secondary truncate">{line || ' '}</span>
              </div>
            ))}
            {previewLines.length > 8 && (
              <div className="text-text-muted text-center pt-1">
                +{previewLines.length - 8} more lines
              </div>
            )}
          </div>
        ) : (
          <div className="text-text-muted text-center py-2">
            No preview available
          </div>
        )}
      </div>

      {/* Path display */}
      <div className="px-3 py-1.5 bg-background-tertiary/30 rounded-b-lg">
        <div className="text-[9px] text-text-muted truncate" title={filePath}>
          {filePath}
        </div>
      </div>
    </div>
  )
})
