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
      return <Loader2 className="w-3 h-3 animate-spin text-accent" />
    case 'waiting_input':
      return <MessageSquare className="w-3 h-3 text-status-warning" />
    case 'tool_calling':
      return <Wrench className="w-3 h-3 text-accent animate-pulse" />
    case 'idle':
    default:
      return <Coffee className="w-3 h-3 text-text-muted" />
  }
}

// Get attention border color — brutalist: hard borders, no rings
function getAttentionStyle(attention?: AttentionState) {
  switch (attention) {
    case 'success':
      return 'border-status-success shadow-brutal-sm'
    case 'error':
      return 'border-status-error shadow-brutal-sm'
    case 'waiting':
      return 'border-status-warning shadow-brutal-sm animate-attention-pulse'
    default:
      return 'border-border hover:border-border'
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
        'card !rounded-brutal border-brutal cursor-pointer',
        getAttentionStyle(attention),
        selected && 'border-accent shadow-brutal',
        isPaused && 'opacity-75'
      )}
      style={{ width: nodeWidth }}
      onClick={handleFocus}
    >
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />

      {/* Header — terminal style */}
      <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border rounded-t-brutal">
        <div className={clsx(
          'p-1.5 border-brutal rounded-brutal',
          status === 'running' && !isPaused && 'bg-accent-light border-accent',
          status === 'waiting_input' && 'bg-status-warning/20 border-status-warning',
          status === 'tool_calling' && !isPaused && 'bg-accent-light border-accent',
          (isPaused || !status || status === 'idle') && 'bg-background-tertiary border-border-subtle'
        )}>
          {getPanelIcon(type)}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-mono font-semibold text-text-primary truncate block">
            {title}
          </span>
          {relativeTime && (
            <span className="text-[10px] text-text-muted flex items-center gap-1 font-mono">
              <Clock className="w-2.5 h-2.5" />
              {relativeTime}
            </span>
          )}
        </div>
        {/* Metrics badge in header */}
        {showMetrics && metrics && totalTokens > 0 && (
          <div
            className="tag !text-[9px] !py-0 !px-1.5 !transform-none !bg-background-tertiary !border-border-subtle !text-text-muted"
            title={`Input: ${formatTokens(metrics.inputTokens)} | Output: ${formatTokens(metrics.outputTokens)} | Cost: ${formatCost(metrics.cost)}`}
          >
            <span className="flex items-center gap-1">
              <Coins className="w-2.5 h-2.5" />
              {formatTokens(totalTokens)}
            </span>
          </div>
        )}
        <div className="flex items-center gap-0.5">
          {onTogglePause && (
            <button
              onClick={handleTogglePause}
              className={clsx(
                'p-1 text-text-muted transition-press',
                isPaused ? 'hover:text-status-success' : 'hover:text-status-warning'
              )}
              title={isPaused ? 'Resume' : 'Pause'}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            </button>
          )}
          <button
            onClick={handleFocus}
            className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
            title="Focus panel"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-status-error/20 text-text-muted hover:text-status-error transition-press"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Status bar */}
      <div className={clsx(
        'flex items-center gap-2 px-3 py-1.5 text-xs border-b border-brutal border-border font-mono',
        !isPaused && status === 'running' && 'bg-accent-light',
        !isPaused && status === 'waiting_input' && 'bg-status-warning/10',
        !isPaused && status === 'tool_calling' && 'bg-accent-light',
        isPaused && 'bg-status-warning/10'
      )}>
        {getStatusIcon(status, isPaused)}
        <div className="flex-1 min-w-0">
          <span className={clsx(
            'font-semibold block',
            isPaused && 'text-status-warning',
            !isPaused && status === 'running' && 'text-accent-dark',
            !isPaused && status === 'waiting_input' && 'text-status-warning',
            !isPaused && status === 'tool_calling' && 'text-accent-dark',
            !isPaused && (!status || status === 'idle') && 'text-text-muted'
          )}>
            {isPaused ? 'Paused' :
             status === 'waiting_input' ? 'Waiting for input' :
             status === 'tool_calling' ? 'Working' :
             status === 'running' ? 'Thinking...' :
             'Idle'}
          </span>
          {currentTool && status === 'tool_calling' && !isPaused && (
            <span className="text-[10px] text-accent-dark truncate block">
              {currentTool}
            </span>
          )}
        </div>
        {!isPaused && (status === 'running' || status === 'tool_calling') && (
          <Activity className="w-3 h-3 text-accent animate-pulse flex-shrink-0" />
        )}
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="px-3 py-1.5 bg-status-error/10 border-b border-brutal border-status-error">
          <span className="text-[10px] text-status-error line-clamp-2 font-mono">{errorMessage}</span>
        </div>
      )}

      {/* Content */}
      <div className="px-3 py-2 space-y-2">
        {/* Task summary */}
        {taskSummary && (
          <div className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5 bg-background-secondary">
            <div className="text-[10px] text-text-muted uppercase tracking-wider mb-0.5 font-mono font-semibold">Current Task</div>
            <div className="text-xs text-text-primary line-clamp-2 font-mono">
              {taskSummary}
            </div>
          </div>
        )}

        {/* Info grid */}
        <div className="space-y-1.5 text-xs font-mono">
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
                <GitBranch className="w-3 h-3 text-accent flex-shrink-0" />
                <span className="text-accent truncate">{gitBranch}</span>
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
          <div className="pt-1 border-t border-brutal border-border-subtle">
            <ToolBreadcrumbs history={toolHistory} maxItems={5} />
          </div>
        )}

        {/* Recent files (collapsible) */}
        {showRecentFiles && recentFiles && recentFiles.length > 0 && (
          <div className="pt-1 border-t border-brutal border-border-subtle">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                setFilesExpanded(!filesExpanded)
              }}
              className="w-full flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary py-0.5 font-mono"
            >
              {/* Triangular chevron */}
              <span className="flex-shrink-0 w-2.5 h-2.5 flex items-center justify-center">
                {filesExpanded ? (
                  <span
                    className="block w-0 h-0"
                    style={{
                      borderLeft: '3px solid transparent',
                      borderRight: '3px solid transparent',
                      borderTop: '4px solid currentColor'
                    }}
                  />
                ) : (
                  <span
                    className="block w-0 h-0"
                    style={{
                      borderTop: '3px solid transparent',
                      borderBottom: '3px solid transparent',
                      borderLeft: '4px solid currentColor'
                    }}
                  />
                )}
              </span>
              <FileText className="w-2.5 h-2.5" />
              <span>Recent Files ({recentFiles.length})</span>
            </button>
            {filesExpanded && (
              <div className="mt-1 space-y-0.5">
                {recentFiles.slice(0, 5).map((file, idx) => (
                  <div
                    key={idx}
                    className="text-[10px] text-text-secondary truncate pl-4 font-mono"
                    title={file}
                  >
                    {shortenPath(file)}
                  </div>
                ))}
                {recentFiles.length > 5 && (
                  <div className="text-[10px] text-text-muted pl-4 font-mono">
                    +{recentFiles.length - 5} more
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Empty state hint */}
        {!shortPath && !gitBranch && !model && !currentTool && !toolHistory?.length && (
          <div className="text-[10px] text-text-muted text-center py-1 font-mono">
            Run a command to see details
          </div>
        )}
      </div>

      {/* Visual indicator bar — solid colors, no gradients */}
      <div
        className={clsx(
          'h-1.5 rounded-b-brutal',
          isPaused && 'bg-status-warning',
          !isPaused && status === 'running' && 'bg-accent animate-pulse',
          !isPaused && status === 'waiting_input' && 'bg-status-warning',
          !isPaused && status === 'tool_calling' && 'bg-accent animate-pulse',
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
