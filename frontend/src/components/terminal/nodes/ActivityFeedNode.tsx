import { memo, useMemo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Activity } from 'lucide-react'
import { clsx } from 'clsx'
import { useActivityStore, type ActivitySeverity } from '../../../stores/activityStore'
import { DataNodeShell } from './DataNodeShell'
import type { DataNodeData } from './types'

export type ActivityFeedNodeType = Node<DataNodeData, 'activity'>

const SEVERITY_DOT: Record<ActivitySeverity, string> = {
  info: 'bg-text-muted',
  success: 'bg-status-success',
  warning: 'bg-status-warning',
  error: 'bg-status-error'
}

const FILTERS = ['training', 'trace', 'pipeline', 'guardrail', 'verification', 'orchestration']

function relTime(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  return `${Math.floor(m / 60)}h`
}

export const ActivityFeedNode = memo(function ActivityFeedNode({ data, selected }: NodeProps<ActivityFeedNodeType>) {
  const events = useActivityStore((s) => s.events)
  const [filter, setFilter] = useState<string | null>(null)

  const visible = useMemo(
    () => (filter ? events.filter((e) => e.category === filter) : events).slice(0, 8),
    [events, filter]
  )

  const buildContext = () =>
    [
      '## Recent BashGym activity',
      ...(filter ? events.filter((e) => e.category === filter) : events)
        .slice(0, 10)
        .map((e) => `- [${e.severity}] ${e.title} (${e.type})`)
    ].join('\n')

  return (
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Activity}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={visible.some((e) => e.severity === 'error') ? 'bg-status-error' : 'bg-background-tertiary'}
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      <div className="flex flex-wrap gap-1 mb-2">
        {FILTERS.map((f) => (
          <button
            key={f}
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setFilter(filter === f ? null : f)
            }}
            className={clsx(
              'px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider border-brutal rounded-brutal transition-press',
              filter === f
                ? 'border-accent bg-accent/10 text-accent'
                : 'border-border text-text-muted hover:text-text-secondary'
            )}
          >
            {f}
          </button>
        ))}
      </div>
      {visible.length === 0 ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-2">
          No events yet — waiting for backend activity
        </div>
      ) : (
        <div className="space-y-1">
          {visible.map((e) => (
            <div key={e.id} className="flex items-center gap-1.5 text-[10px] font-mono">
              <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', SEVERITY_DOT[e.severity])} />
              <span className="flex-1 truncate text-text-secondary" title={e.title}>{e.title}</span>
              <span className="text-text-muted flex-shrink-0">{relTime(e.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </DataNodeShell>
  )
})
