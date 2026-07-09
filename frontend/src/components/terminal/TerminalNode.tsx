import { memo, useCallback, useState } from 'react'
import { Handle, Position, NodeProps, Node } from '@xyflow/react'
import {
  FileText,
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
  Coins,
  Pause,
  Play,
  Link2,
  Eye
} from 'lucide-react'
import { clsx } from 'clsx'
import type { AttentionState, AgentStatus, AgentKind, PanelType, ToolHistoryItem, SessionMetrics } from '../../stores/terminalStore'
import { ToolBreadcrumbs } from './ToolBreadcrumbs'
import { AgentBadge } from '../sessions/AgentBadge'
import { useCanvasControlStore } from '../../stores'
import { NodeFlowerMark } from './NodeFlowerMark'

export interface TerminalNodeData extends Record<string, unknown> {
  panelId: string
  title: string
  type: PanelType
  status?: AgentStatus
  attention?: AttentionState
  gitBranch?: string
  model?: string
  agentKind?: AgentKind
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
  lastOutput?: string[]
  hasConnections?: boolean
  /** This terminal is the watched end of a monitor edge */
  isWatched?: boolean
  /** Title of the terminal this one is watching (watcher end of a monitor edge) */
  watchingTitle?: string
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  onTogglePause?: (panelId: string) => void
}

export type TerminalNodeType = Node<TerminalNodeData, 'terminal'>

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

export const TerminalNode = memo(function TerminalNode({ data, selected }: NodeProps<TerminalNodeType>) {
  const {
    panelId,
    title,
    type,
    status,
    attention,
    gitBranch,
    model,
    agentKind,
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
    hasConnections,
    isWatched,
    watchingTitle,
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
        !isPaused && status === 'running' && 'terminal-status-running',
        !isPaused && status === 'tool_calling' && 'terminal-status-tool-calling',
        !isPaused && status === 'waiting_input' && 'terminal-status-waiting-input'
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
        <NodeFlowerMark
          variant={type}
          size="xl"
          active={Boolean(!isPaused && (status === 'running' || status === 'tool_calling' || selected))}
          muted={Boolean(isPaused || !status || status === 'idle')}
          title={`${type} node`}
        />
        <div className="flex-1 min-w-0">
          <span className="text-sm font-mono font-semibold text-text-primary truncate flex items-center gap-1.5">
            <span className="truncate">{title}</span>
            {agentKind && <AgentBadge kind={agentKind} />}
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
        <div className="flex items-center gap-1 nodrag">
          {watchingTitle && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-[hsl(270_45%_60%)]/60 bg-[hsl(270_45%_60%)]/10 rounded-brutal text-[hsl(270_45%_60%)] text-[8px] font-bold uppercase tracking-wider max-w-[110px]"
              title={`Monitoring "${watchingTitle}" — snapshots arrive as a file path in this terminal's input`}
            >
              <Eye className="w-2.5 h-2.5 flex-shrink-0" />
              <span className="truncate">{watchingTitle}</span>
            </div>
          )}
          {isWatched && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-[hsl(270_45%_60%)]/60 bg-[hsl(270_45%_60%)]/10 rounded-brutal text-[hsl(270_45%_60%)] text-[8px] font-bold uppercase tracking-wider"
              title="Being monitored — a connected watcher can receive snapshots of this terminal's output"
            >
              <Eye className="w-2.5 h-2.5" />
              <span>WATCHED</span>
            </div>
          )}
          {hasConnections && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-accent/60 bg-accent/10 rounded-brutal text-accent"
              title="Connected to a data node — content routes into this terminal"
            >
              <Link2 className="w-2.5 h-2.5" />
            </div>
          )}
          {onTogglePause && (
            <button
              onClick={handleTogglePause}
              className={clsx('node-btn', isPaused ? 'node-btn-success' : 'node-btn-warning')}
              title={isPaused ? 'Resume' : 'Pause'}
            >
              {isPaused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
            </button>
          )}
          <button
            onClick={handleFocus}
            className="node-btn node-btn-accent"
            title="Focus panel"
          >
            <Maximize2 className="w-3 h-3" />
          </button>
          <button
            onClick={handleClose}
            className="node-btn node-btn-danger"
            title="Close"
          >
            <X className="w-3 h-3" />
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
// eslint-disable-next-line react-refresh/only-export-components
export const nodeTypes = {
  terminal: TerminalNode
}
