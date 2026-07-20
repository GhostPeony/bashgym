import { memo, useCallback, useRef, useState, useEffect } from 'react'
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
  type Edge
} from '@xyflow/react'
import { Eye, Send, ArrowLeftRight, Check } from 'lucide-react'
import { clsx } from 'clsx'
import type { MonitorAutoMode } from '../../../stores'

export interface MonitorEdgeData extends Record<string, unknown> {
  auto?: MonitorAutoMode
  onSendSnapshot?: (edgeId: string) => Promise<{ sent: boolean; error?: string }>
  onCycleAuto?: (edgeId: string) => void
  onSwapDirection?: (edgeId: string) => void
}

export type MonitorEdgeType = Edge<MonitorEdgeData, 'monitor'>

// Wisteria — the monitor identity hue, distinct from the data-node palette
const MONITOR_STROKE = 'hsl(270 45% 60%)'

const AUTO_LABEL: Record<MonitorAutoMode, string> = {
  off: 'AUTO: OFF',
  prefill: 'AUTO: PREFILL',
  send: 'AUTO: SEND'
}

export const MonitorEdge = memo(function MonitorEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data
}: EdgeProps<MonitorEdgeType>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition
  })

  const auto = data?.auto ?? 'off'
  const [sendState, setSendState] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(
    () => () => {
      if (flashTimerRef.current) clearTimeout(flashTimerRef.current)
    },
    []
  )

  const handleSend = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation()
      if (!data?.onSendSnapshot || sendState === 'sending') return
      setSendState('sending')
      const result = await data.onSendSnapshot(id)
      setSendState(result.sent ? 'sent' : 'error')
      if (flashTimerRef.current) clearTimeout(flashTimerRef.current)
      flashTimerRef.current = setTimeout(() => setSendState('idle'), 1500)
    },
    [data, id, sendState]
  )

  const handleCycleAuto = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      data?.onCycleAuto?.(id)
    },
    [data, id]
  )

  const handleSwap = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      data?.onSwapDirection?.(id)
    },
    [data, id]
  )

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{ stroke: MONITOR_STROKE, strokeWidth: 2, strokeDasharray: '6 4' }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan flex flex-col items-center gap-1"
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: 'all'
          }}
        >
          <div
            className="flex items-center gap-1 px-1.5 py-0.5 border-brutal rounded-brutal shadow-brutal-sm bg-background-card font-mono text-[9px] font-bold uppercase tracking-wider"
            style={{ borderColor: MONITOR_STROKE, color: MONITOR_STROKE }}
            title="Monitor edge — output flows from the watched terminal (source) to the watcher (target)"
          >
            <Eye className="w-2.5 h-2.5" />
            MONITOR
            {auto !== 'off' && (
              <span className="px-1 border-l" style={{ borderColor: MONITOR_STROKE }}>
                {auto === 'send' ? 'AUTO·SEND' : 'AUTO·PREFILL'}
              </span>
            )}
          </div>
          {selected && (
            <div className="flex items-center gap-1">
              <button
                onClick={handleSend}
                disabled={sendState === 'sending'}
                className={clsx(
                  'node-btn node-btn-wide',
                  sendState === 'sent' && 'node-btn-success',
                  sendState === 'error' && 'node-btn-danger',
                  (sendState === 'idle' || sendState === 'sending') && 'node-btn-accent'
                )}
                title="Prefill a snapshot file path into the watcher's input — nothing runs until you press Enter there"
              >
                {sendState === 'sent' ? (
                  <span className="flex items-center gap-1">
                    <Check className="w-2.5 h-2.5" /> SENT
                  </span>
                ) : sendState === 'error' ? (
                  'NO OUTPUT'
                ) : (
                  <span className="flex items-center gap-1">
                    <Send className="w-2.5 h-2.5" /> SEND SNAPSHOT
                  </span>
                )}
              </button>
              <button
                onClick={handleCycleAuto}
                className={clsx('node-btn node-btn-wide', auto !== 'off' && 'node-btn-success')}
                title="Send a snapshot whenever the watched agent finishes a step (min 20s apart). PREFILL types the path; SEND submits it automatically"
              >
                {AUTO_LABEL[auto]}
              </button>
              <button
                onClick={handleSwap}
                className="node-btn node-btn-wide"
                title="Swap watcher and watched"
              >
                <span className="flex items-center gap-1">
                  <ArrowLeftRight className="w-2.5 h-2.5" /> SWAP
                </span>
              </button>
            </div>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  )
})
