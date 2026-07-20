import { memo, useCallback, useState, type ReactNode } from 'react'
import { Handle, Position } from '@xyflow/react'
import { Link2, Loader2, Send, X } from 'lucide-react'
import { clsx } from 'clsx'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'
import type { RouteResult } from '../../../utils/edgeRouting'
import { NodeFlowerMark } from '../NodeFlowerMark'
import type { NodeFlowerVariant } from '../nodeFlowerAssets'
import { useNodeSurface } from './nodeSurface'

export interface DataNodeShellProps {
  panelId: string
  title: string
  flowerVariant?: NodeFlowerVariant
  selected?: boolean
  hasConnections?: boolean
  /** Builds the markdown for Send-to-terminal; button shown only when linked. May be async (e.g. to fetch analysis at send time). */
  buildContext?: () => string | Promise<string>
  contextBasename?: string
  onContextRouted?: (result: RouteResult) => void | Promise<void>
  onContextRouteError?: (error: unknown) => void | Promise<void>
  headerRight?: ReactNode
  /** Tailwind bg-* class for the bottom indicator bar */
  statusBarClass?: string
  /** Identity hue from the platform accent palette; tints strip + icon per node type */
  hue?: number
  visualPhase?: string
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  children: ReactNode
}

export const DataNodeShell = memo(function DataNodeShell({
  panelId,
  title,
  flowerVariant = 'integration',
  selected,
  hasConnections,
  buildContext,
  contextBasename,
  onContextRouted,
  onContextRouteError,
  headerRight,
  statusBarClass = 'bg-background-tertiary',
  hue,
  visualPhase,
  onFocus,
  onClose,
  children
}: DataNodeShellProps) {
  const [sending, setSending] = useState(false)
  const surface = useNodeSurface()
  const isGrid = surface === 'grid'

  const handleFocus = useCallback(() => onFocus?.(panelId), [panelId, onFocus])

  const handleClose = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onClose?.(panelId)
    },
    [panelId, onClose]
  )

  const handleSend = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation()
      if (!buildContext || sending) return
      setSending(true)
      try {
        const content = await buildContext()
        const result = await routeToLinkedTerminals(
          panelId,
          content,
          'md',
          contextBasename || 'bashgym_context'
        )
        await onContextRouted?.(result)
      } catch (error) {
        await onContextRouteError?.(error)
      } finally {
        setSending(false)
      }
    },
    [panelId, buildContext, contextBasename, onContextRouted, onContextRouteError, sending]
  )

  return (
    <div
      className={clsx(
        'card !rounded-brutal border-brutal cursor-pointer',
        isGrid ? 'flex h-full w-full min-w-0 flex-col overflow-hidden' : 'w-[360px]',
        selected ? 'border-accent shadow-brutal' : 'border-border hover:border-border',
        visualPhase === 'planned' && 'canvas-node-ghost'
      )}
      onClick={handleFocus}
      data-canvas-phase={visualPhase}
      data-node-surface={surface}
    >
      {!isGrid && (
        <>
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
        </>
      )}

      {hue != null && (
        <div className="h-1 rounded-t-brutal" style={{ background: `hsl(${hue}, 45%, 65%)` }} />
      )}
      <div
        className={clsx(
          'flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border',
          hue == null && 'rounded-t-brutal'
        )}
      >
        <NodeFlowerMark
          variant={flowerVariant}
          hue={hue}
          size="xl"
          active={Boolean(selected || visualPhase === 'running' || visualPhase === 'queued')}
          muted={visualPhase === 'planned'}
          title={`${title} node`}
        />
        <span className="flex-1 min-w-0 text-sm font-mono font-semibold text-text-primary truncate">
          {title}
        </span>
        {headerRight}
        <div className="flex items-center gap-1 nodrag">
          {hasConnections && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-accent/60 bg-accent/10 rounded-brutal text-accent"
              title="Connected on canvas"
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
              {sending ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Send className="w-3 h-3" />
              )}
            </button>
          )}
          {!isGrid && (
            <button
              type="button"
              onClick={handleClose}
              className="node-btn node-btn-danger"
              title="Close"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      <div className={clsx('px-3 py-2', isGrid && 'min-h-0 flex-1 overflow-auto')}>{children}</div>

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
