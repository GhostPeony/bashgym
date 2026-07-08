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
  /** Builds the markdown for Send-to-terminal; button shown only when linked */
  buildContext?: () => string
  headerRight?: ReactNode
  /** Tailwind bg-* class for the bottom indicator bar */
  statusBarClass?: string
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
      await routeToLinkedTerminals(panelId, buildContext(), 'md')
    } finally {
      setSending(false)
    }
  }, [panelId, buildContext, sending])

  return (
    <div
      className={clsx(
        'w-[320px] card !rounded-brutal border-brutal cursor-pointer',
        selected ? 'border-accent shadow-brutal' : 'border-border hover:border-border'
      )}
      onClick={handleFocus}
    >
      <Handle type="target" position={Position.Left} className="!bg-accent !w-2 !h-2 !border-brutal !border-border" />
      <Handle type="source" position={Position.Right} className="!bg-accent !w-2 !h-2 !border-brutal !border-border" />

      <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border rounded-t-brutal">
        <div className="p-1.5 border-brutal border-border-subtle rounded-brutal bg-background-tertiary">
          <Icon className="w-4 h-4 text-accent" />
        </div>
        <span className="flex-1 min-w-0 text-sm font-mono font-semibold text-text-primary truncate">
          {title}
        </span>
        {headerRight}
        <div className="flex items-center gap-0.5 nodrag">
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
              className="p-1 text-text-muted hover:text-accent transition-press"
              title="Send context to linked terminals"
            >
              {sending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            </button>
          )}
          <button
            type="button"
            onClick={handleClose}
            className="p-1 hover:bg-status-error/20 text-text-muted hover:text-status-error transition-press"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="px-3 py-2 nodrag nowheel">{children}</div>

      <div className={clsx('h-1.5 rounded-b-brutal', statusBarClass)} />
    </div>
  )
})
