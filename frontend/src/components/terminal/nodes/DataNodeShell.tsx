import { memo, useCallback, useState, type ReactNode } from 'react'
import { Handle, Position } from '@xyflow/react'
import { Link2, Loader2, Send, X } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { clsx } from 'clsx'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'

export interface DataNodeShellProps {
  panelId: string
  title: string
  icon: LucideIcon
  selected?: boolean
  hasConnections?: boolean
  /** Builds the markdown for Send-to-terminal; button shown only when linked. May be async (e.g. to fetch analysis at send time). */
  buildContext?: () => string | Promise<string>
  headerRight?: ReactNode
  /** Tailwind bg-* class for the bottom indicator bar */
  statusBarClass?: string
  /** Identity hue from the platform accent palette; tints strip + icon per node type */
  hue?: number
  visualPhase?: string
  motion?: string
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  children: ReactNode
}

export const DataNodeShell = memo(function DataNodeShell({
  panelId,
  title,
  icon: Icon,
  selected,
  hasConnections,
  buildContext,
  headerRight,
  statusBarClass = 'bg-background-tertiary',
  hue,
  visualPhase,
  motion,
  onFocus,
  onClose,
  children
}: DataNodeShellProps) {
  const [sending, setSending] = useState(false)

  const handleFocus = useCallback(() => onFocus?.(panelId), [panelId, onFocus])

  const handleClose = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onClose?.(panelId)
  }, [panelId, onClose])

  const handleSend = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!buildContext || sending) return
    setSending(true)
    try {
      const content = await buildContext()
      await routeToLinkedTerminals(panelId, content, 'md')
    } finally {
      setSending(false)
    }
  }, [panelId, buildContext, sending])

  return (
    <div
      className={clsx(
        'w-[360px] card !rounded-brutal border-brutal cursor-pointer',
        selected ? 'border-accent shadow-brutal' : 'border-border hover:border-border',
        visualPhase === 'planned' && 'canvas-node-ghost',
        motion === 'enter-from-origin' && 'canvas-node-enter'
      )}
      onClick={handleFocus}
      data-canvas-phase={visualPhase}
    >
      <Handle type="target" position={Position.Left} className="!bg-accent !w-2 !h-2 !border-brutal !border-border" />
      <Handle type="source" position={Position.Right} className="!bg-accent !w-2 !h-2 !border-brutal !border-border" />

      {hue != null && (
        <div
          className="h-1 rounded-t-brutal"
          style={{ background: `hsl(${hue}, 45%, 65%)` }}
        />
      )}
      <div className={clsx(
        'flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border',
        hue == null && 'rounded-t-brutal'
      )}>
        <div
          className="p-1.5 border-brutal border-border-subtle rounded-brutal bg-background-tertiary"
          style={hue != null ? {
            background: `hsla(${hue}, 40%, 60%, 0.16)`,
            borderColor: `hsla(${hue}, 35%, 50%, 0.55)`
          } : undefined}
        >
          <Icon
            className={clsx('w-4 h-4', hue == null && 'text-accent')}
            style={hue != null ? { color: `hsl(${hue}, 45%, 48%)` } : undefined}
          />
        </div>
        <span className="flex-1 min-w-0 text-sm font-mono font-semibold text-text-primary truncate">
          {title}
        </span>
        {headerRight}
        <div className="flex items-center gap-1 nodrag">
          {hasConnections && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-accent/60 bg-accent/10 rounded-brutal text-accent"
              title="Connected to terminal"
            >
              <Link2 className="w-2.5 h-2.5" />
            </div>
          )}
          {hasConnections && buildContext && (
            <button
              type="button"
              onClick={handleSend}
              disabled={sending}
              className="node-btn node-btn-accent"
              title="Send context to linked terminals"
            >
              {sending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
            </button>
          )}
          <button
            type="button"
            onClick={handleClose}
            className="node-btn node-btn-danger"
            title="Close"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      </div>

      <div className="px-3 py-2">{children}</div>

      <div
        className={clsx(
          'h-1.5 rounded-b-brutal',
          statusBarClass,
          visualPhase === 'running' && 'canvas-state-strip-live',
          visualPhase === 'queued' && 'canvas-state-strip-queued'
        )}
      />
    </div>
  )
})
