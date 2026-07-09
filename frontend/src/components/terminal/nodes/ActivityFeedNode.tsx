import { memo, useMemo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Pause, Play, SlidersHorizontal } from 'lucide-react'
import { clsx } from 'clsx'
import { useActivityStore, type ActivityEvent, type ActivitySeverity } from '../../../stores/activityStore'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type ActivityFeedNodeType = Node<DataNodeData, 'activity'>

const SEVERITY_DOT: Record<ActivitySeverity, string> = {
  info: 'bg-text-muted',
  success: 'bg-status-success',
  warning: 'bg-status-warning',
  error: 'bg-status-error'
}

const FILTERS = ['training', 'trace', 'pipeline', 'guardrail', 'verification', 'orchestration']

const CATEGORY_LABEL: Record<string, string> = {
  training: 'Training',
  trace: 'Trace',
  pipeline: 'Pipeline',
  guardrail: 'Guardrail',
  verification: 'Verification',
  orchestration: 'Orchestration'
}

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
  const [errorsOnly, setErrorsOnly] = useState(false)
  const [frozen, setFrozen] = useState<ActivityEvent[] | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<ActivityEvent | null>(null)

  const source = frozen ?? events
  const filtered = useMemo(() => {
    let list = filter ? source.filter((e) => e.category === filter) : source
    if (errorsOnly) list = list.filter((e) => e.severity === 'error' || e.severity === 'warning')
    return list
  }, [source, filter, errorsOnly])
  const visible = filtered.slice(0, 8)
  const warningCount = source.filter((event) => event.severity === 'warning' || event.severity === 'error').length

  const buildContext = () =>
    [
      '## Recent BashGym activity',
      ...filtered.slice(0, 10).map((e) => `- [${e.severity}] ${e.title} (${e.type})`)
    ].join('\n')

  const chipClass = (active: boolean) =>
    clsx(
      'nodrag px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider border-brutal rounded-brutal transition-press',
      active
        ? 'border-accent bg-accent/10 text-accent shadow-brutal-sm'
        : 'border-border bg-background-card text-text-muted hover:text-text-secondary hover:border-border'
    )

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      flowerVariant="activity"
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={visible.some((e) => e.severity === 'error') ? 'bg-status-error' : 'bg-background-tertiary'}
      hue={hueFor('activity')}
      headerRight={
        <>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setConfigOpen(true)
            }}
            className="nodrag node-btn node-btn-accent"
            title="Configure activity feed"
          >
            <SlidersHorizontal className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setFrozen(frozen ? null : [...events])
            }}
            className={clsx('nodrag node-btn', frozen ? 'node-btn-warning' : 'node-btn-accent')}
            title={frozen ? 'Resume live feed' : 'Pause feed to read'}
          >
            {frozen ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
          </button>
        </>
      }
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
            className={chipClass(filter === f)}
          >
            {f}
          </button>
        ))}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setErrorsOnly(!errorsOnly)
          }}
          className={chipClass(errorsOnly)}
          title="Only warnings and errors"
        >
          err
        </button>
      </div>
      {visible.length === 0 ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-2">
          {errorsOnly ? 'No warnings or errors' : 'No events yet — waiting for backend activity'}
        </div>
      ) : (
        <div className="space-y-1">
          {visible.map((e) => {
            return (
              <div
                key={e.id}
                className="flex items-center gap-1.5 text-[10px] font-mono nodrag cursor-pointer hover:text-text-primary"
                onClick={(ev) => {
                  ev.stopPropagation()
                  setSelectedEvent(e)
                }}
                title={`${e.title} - event details`}
              >
                <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', SEVERITY_DOT[e.severity])} />
                <span className="flex-1 truncate text-text-secondary">{e.title}</span>
                <span className="text-text-muted flex-shrink-0">{relTime(e.timestamp)}</span>
              </div>
            )
          })}
        </div>
      )}
    </DataNodeShell>
    <NodeConfigModal
      isOpen={configOpen}
      onClose={() => setConfigOpen(false)}
      title={`${data.title} Config`}
      description="Workspace activity feed"
      size="md"
    >
      <ConfigSection title="Feed State">
        <div className="flex flex-wrap gap-1.5">
          <ConfigPill tone={frozen ? 'warning' : 'success'}>{frozen ? 'paused' : 'live'}</ConfigPill>
          <ConfigPill tone={errorsOnly ? 'warning' : 'neutral'}>{errorsOnly ? 'warnings only' : 'all severities'}</ConfigPill>
          {filter ? <ConfigPill tone="accent">{filter}</ConfigPill> : null}
        </div>
        <ConfigRows>
          <ConfigRow label="Source events" value={source.length} />
          <ConfigRow label="Visible events" value={visible.length} />
          <ConfigRow label="Warnings" value={warningCount} />
          <ConfigRow label="Category filter" value={filter || 'all'} />
          <ConfigRow label="Snapshot" value={frozen ? `${frozen.length} frozen events` : 'live stream'} />
        </ConfigRows>
      </ConfigSection>
    </NodeConfigModal>

    <NodeConfigModal
      isOpen={selectedEvent != null}
      onClose={() => setSelectedEvent(null)}
      title="Activity Event"
      description={selectedEvent ? CATEGORY_LABEL[selectedEvent.category] || selectedEvent.category : undefined}
      size="md"
    >
      {selectedEvent ? (
        <ConfigSection title={selectedEvent.title}>
          <div className="flex flex-wrap gap-1.5">
            <ConfigPill tone={selectedEvent.severity === 'error' ? 'error' : selectedEvent.severity === 'warning' ? 'warning' : selectedEvent.severity === 'success' ? 'success' : 'neutral'}>
              {selectedEvent.severity}
            </ConfigPill>
            <ConfigPill tone="neutral">{selectedEvent.category}</ConfigPill>
          </div>
          <ConfigRows>
            <ConfigRow label="Type" value={selectedEvent.type} />
            <ConfigRow label="Category" value={selectedEvent.category} />
            <ConfigRow label="Time" value={new Date(selectedEvent.timestamp).toLocaleString()} />
            <ConfigRow label="Event id" value={selectedEvent.id} />
            <ConfigRow label="Detail" value={selectedEvent.detail} />
          </ConfigRows>
        </ConfigSection>
      ) : null}
    </NodeConfigModal>
    </>
  )
})
