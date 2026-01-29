import { memo, useCallback, useState } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import {
  Terminal,
  Globe,
  FileText,
  FolderTree,
  Loader2,
  MessageSquare,
  Wrench,
  Coffee,
  Maximize2,
  X,
  GitBranch,
  Clock,
  Folder,
  Cpu,
  Activity,
  ChevronDown,
  ChevronRight,
  Coins,
  Pause,
  Play
} from 'lucide-react'
import { clsx } from 'clsx'
import type { AttentionState, AgentStatus, PanelType, ToolHistoryItem, SessionMetrics } from '../../stores/terminalStore'
import { ToolBreadcrumbs } from './ToolBreadcrumbs'
import { useCanvasControlStore } from '../../stores'

export interface TerminalNodeData {
  panelId: string
  title: string
  type: PanelType
  status?: AgentStatus
  attention?: AttentionState
  gitBranch?: string
  model?: string
  currentTool?: string
  cwd?: string
  lastActivity?: number
  taskSummary?: string
  // New enhanced fields
  toolHistory?: ToolHistoryItem[]
  metrics?: SessionMetrics
  recentFiles?: string[]
  errorMessage?: string
  isExpanded?: boolean
  isPaused?: boolean
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  onTogglePause?: (panelId: string) => void
}

export type TerminalNodeType = NodeProps<TerminalNodeData>

// Get icon based on panel type
function getPanelIcon(type: PanelType) {
  switch (type) {
    case 'terminal':
      return <Terminal className="w-4 h-4" />
    case 'browser':
      return <Globe className="w-4 h-4" />
    case 'preview':
      return <FileText className="w-4 h-4" />
    case 'files':
      return <FolderTree className="w-4 h-4" />
    default:
      return <Terminal className="w-4 h-4" />
  }
}

// Get status icon based on agent status
function getStatusIcon(status?: AgentStatus, isPaused?: boolean) {
  if (isPaused) {
    return <Pause className="w-3 h-3 text-status-warning" />
  }
  switch (status) {
    case 'running':
      return <Loader2 className="w-3 h-3 animate-spin text-primary" />
    case 'waiting_input':
      return <MessageSquare className="w-3 h-3 text-status-warning" />
    case 'tool_calling':
      return <Wrench className="w-3 h-3 text-info animate-pulse" />
    case 'idle':
    default:
      return <Coffee className="w-3 h-3 text-text-muted" />
  }
}

// Get attention border color
function getAttentionStyle(attention?: AttentionState) {
  switch (attention) {
    case 'success':
      return 'border-status-success ring-2 ring-status-success/20'
    case 'error':
      return 'border-status-error ring-2 ring-status-error/20'
    case 'waiting':
      return 'border-status-warning ring-2 ring-status-warning/20 animate-pulse'
    default:
      return 'border-border-subtle hover:border-border'
  }
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

// Shorten path for display
function shortenPath(path?: string): string {
  if (!path) return ''
  const shortened = path.replace(/^(C:\\Users\\[^\\]+|\/home\/[^/]+|\/Users\/[^/]+)/, '~')
  const parts = shortened.split(/[/\\]/)
  if (parts.length > 3) {
    return '.../' + parts.slice(-2).join('/')
  }
  return shortened
}

// Format token count
function formatTokens(count: number): string {
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`
  return count.toString()
}

// Format cost
function formatCost(cost: number): string {
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

export const TerminalNode = memo(function TerminalNode({ data, selected }: TerminalNodeType) {
  const {
    panelId,
    title,
    type,
    status,
    attention,
    gitBranch,
    model,
    currentTool,
    cwd,
    lastActivity,
    taskSummary,
    toolHistory,
    metrics,
    recentFiles,
    errorMessage,
    isExpanded: dataExpanded,
    isPaused,
    onFocus,
    onClose,
    onTogglePause
  } = data

  // Local expanded state (for recent files section)
  const [filesExpanded, setFilesExpanded] = useState(false)

  // Get global control settings
  const { showMetrics, showToolHistory, showRecentFiles } = useCanvasControlStore()

  const handleFocus = useCallback(() => {
    onFocus?.(panelId)
  }, [panelId, onFocus])

  const handleClose = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onClose?.(panelId)
  }, [panelId, onClose])

  const handleTogglePause = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onTogglePause?.(panelId)
  }, [panelId, onTogglePause])

  const relativeTime = formatRelativeTime(lastActivity)
  const shortPath = shortenPath(cwd)
  const totalTokens = metrics ? metrics.inputTokens + metrics.outputTokens : 0

  // Calculate width based on expanded state
  const nodeWidth = dataExpanded ? 400 : 320

  return (
    <div
      className={clsx(
        'bg-background-secondary rounded-lg border-2 shadow-lg transition-all cursor-pointer',
        getAttentionStyle(attention),
        selected && 'ring-2 ring-primary border-primary',
        isPaused && 'opacity-75'
      )}
      style={{ width: nodeWidth }}
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
        <div className={clsx(
          'p-1.5 rounded-md',
          status === 'running' && !isPaused && 'bg-primary/20',
          status === 'waiting_input' && 'bg-status-warning/20',
          status === 'tool_calling' && !isPaused && 'bg-info/20',
          (isPaused || !status || status === 'idle') && 'bg-background-tertiary'
        )}>
          {getPanelIcon(type)}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-text-primary truncate block">
            {title}
          </span>
          {relativeTime && (
            <span className="text-[10px] text-text-muted flex items-center gap-1">
              <Clock className="w-2.5 h-2.5" />
              {relativeTime}
            </span>
          )}
        </div>
        {/* Metrics badge in header */}
        {showMetrics && metrics && totalTokens > 0 && (
          <div
            className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-background-tertiary text-[10px] text-text-muted"
            title={`Input: ${formatTokens(metrics.inputTokens)} | Output: ${formatTokens(metrics.outputTokens)} | Cost: ${formatCost(metrics.cost)}`}
          >
            <Coins className="w-2.5 h-2.5" />
            <span>{formatTokens(totalTokens)}</span>
          </div>
        )}
        <div className="flex items-center gap-0.5">
          {onTogglePause && (
            <button
              onClick={handleTogglePause}
              className={clsx(
                'p-1 rounded text-text-muted',
                isPaused ? 'hover:bg-status-success/20 hover:text-status-success' : 'hover:bg-status-warning/20 hover:text-status-warning'
              )}
              title={isPaused ? 'Resume' : 'Pause'}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
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

      {/* Status bar */}
      <div className={clsx(
        'flex items-center gap-2 px-3 py-1.5 text-xs border-b border-border-subtle',
        !isPaused && status === 'running' && 'bg-primary/5',
        !isPaused && status === 'waiting_input' && 'bg-status-warning/5',
        !isPaused && status === 'tool_calling' && 'bg-info/5',
        isPaused && 'bg-status-warning/10'
      )}>
        {getStatusIcon(status, isPaused)}
        <div className="flex-1 min-w-0">
          <span className={clsx(
            'font-medium block',
            isPaused && 'text-status-warning',
            !isPaused && status === 'running' && 'text-primary',
            !isPaused && status === 'waiting_input' && 'text-status-warning',
            !isPaused && status === 'tool_calling' && 'text-info',
            !isPaused && (!status || status === 'idle') && 'text-text-muted'
          )}>
            {isPaused ? 'Paused' :
             status === 'waiting_input' ? 'Waiting for input' :
             status === 'tool_calling' ? 'Working' :
             status === 'running' ? 'Thinking...' :
             'Idle'}
          </span>
          {currentTool && status === 'tool_calling' && !isPaused && (
            <span className="text-[10px] text-info/80 truncate block">
              {currentTool}
            </span>
          )}
        </div>
        {!isPaused && (status === 'running' || status === 'tool_calling') && (
          <Activity className="w-3 h-3 text-primary animate-pulse flex-shrink-0" />
        )}
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="px-3 py-1.5 bg-status-error/10 border-b border-status-error/20">
          <span className="text-[10px] text-status-error line-clamp-2">{errorMessage}</span>
        </div>
      )}

      {/* Content */}
      <div className="px-3 py-2 space-y-2">
        {/* Task summary */}
        {taskSummary && (
          <div className="bg-background-tertiary/50 rounded px-2 py-1.5">
            <div className="text-[10px] text-text-muted uppercase tracking-wide mb-0.5">Current Task</div>
            <div className="text-xs text-text-primary line-clamp-2">
              {taskSummary}
            </div>
          </div>
        )}

        {/* Info grid */}
        <div className="space-y-1.5 text-xs">
          {shortPath && (
            <div className="flex items-center gap-1.5">
              <Folder className="w-3 h-3 text-text-muted flex-shrink-0" />
              <span className="text-text-secondary truncate flex-1" title={cwd}>
                {shortPath}
              </span>
            </div>
          )}

          <div className="flex items-center gap-3">
            {gitBranch && (
              <div className="flex items-center gap-1.5">
                <GitBranch className="w-3 h-3 text-primary flex-shrink-0" />
                <span className="text-primary truncate">{gitBranch}</span>
              </div>
            )}

            {model && (
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3 h-3 text-text-muted flex-shrink-0" />
                <span className="text-text-secondary truncate">{model}</span>
              </div>
            )}
          </div>
        </div>

        {/* Tool history breadcrumbs */}
        {showToolHistory && toolHistory && toolHistory.length > 0 && (
          <div className="pt-1 border-t border-border-subtle">
            <ToolBreadcrumbs history={toolHistory} maxItems={5} />
          </div>
        )}

        {/* Recent files (collapsible) */}
        {showRecentFiles && recentFiles && recentFiles.length > 0 && (
          <div className="pt-1 border-t border-border-subtle">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                setFilesExpanded(!filesExpanded)
              }}
              className="w-full flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary py-0.5"
            >
              {filesExpanded ? (
                <ChevronDown className="w-2.5 h-2.5" />
              ) : (
                <ChevronRight className="w-2.5 h-2.5" />
              )}
              <FileText className="w-2.5 h-2.5" />
              <span>Recent Files ({recentFiles.length})</span>
            </button>
            {filesExpanded && (
              <div className="mt-1 space-y-0.5">
                {recentFiles.slice(0, 5).map((file, idx) => (
                  <div
                    key={idx}
                    className="text-[10px] text-text-secondary truncate pl-4"
                    title={file}
                  >
                    {shortenPath(file)}
                  </div>
                ))}
                {recentFiles.length > 5 && (
                  <div className="text-[10px] text-text-muted pl-4">
                    +{recentFiles.length - 5} more
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Empty state hint */}
        {!shortPath && !gitBranch && !model && !currentTool && !toolHistory?.length && (
          <div className="text-[10px] text-text-muted text-center py-1">
            Run a command to see details
          </div>
        )}
      </div>

      {/* Visual indicator bar */}
      <div
        className={clsx(
          'h-1.5 rounded-b-lg transition-colors',
          isPaused && 'bg-status-warning/50',
          !isPaused && status === 'running' && 'bg-gradient-to-r from-primary to-primary/50 animate-pulse',
          !isPaused && status === 'waiting_input' && 'bg-status-warning',
          !isPaused && status === 'tool_calling' && 'bg-gradient-to-r from-info to-info/50 animate-pulse',
          !isPaused && (!status || status === 'idle') && 'bg-background-tertiary',
          attention === 'error' && 'bg-status-error',
          attention === 'success' && 'bg-status-success'
        )}
      />
    </div>
  )
})

// Node types for React Flow
export const nodeTypes = {
  terminal: TerminalNode
}
