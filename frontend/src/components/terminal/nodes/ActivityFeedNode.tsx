import { memo, useMemo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Copy,
  Info,
  Pause,
  Play,
  Search,
  SlidersHorizontal,
  Trash2,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import { useActivityStore, type ActivityEvent, type ActivitySeverity } from '../../../stores/activityStore'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type ActivityFeedNodeType = Node<DataNodeData, 'activity'>

const SEVERITY_DOT: Record<ActivitySeverity, string> = {
  info: 'bg-text-muted',
  success: 'bg-status-success',
  warning: 'bg-status-warning',
  error: 'bg-status-error',
}

const SEVERITY_ICON = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  error: AlertTriangle,
} satisfies Record<ActivitySeverity, typeof Info>

const COMPACT_FILTERS = ['training', 'orchestration', 'designer', 'guardrail', 'hf']

const CATEGORY_LABEL: Record<string, string> = {
  autoresearch: 'AutoResearch',
  cascade: 'Cascade',
  designer: 'Data Designer',
  guardrail: 'Guardrails',
  hf: 'Hugging Face',
  integration: 'Integrations',
  orchestration: 'Orchestration',
  pipeline: 'Pipeline',
  router: 'Router',
  'schema-research': 'Schema Research',
  'skill-eval': 'Skill Lab',
  trace: 'Traces',
  training: 'Training',
  verification: 'Verification',
}

function relTime(ts: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

function formattedDetail(detail?: string): string {
  if (!detail) return 'No additional event data was recorded.'
  try {
    return JSON.stringify(JSON.parse(detail), null, 2)
  } catch {
    return detail
  }
}

export const ActivityFeedNode = memo(function ActivityFeedNode({ data, selected }: NodeProps<ActivityFeedNodeType>) {
  const events = useActivityStore((state) => state.events)
  const clearEvents = useActivityStore((state) => state.clear)
  const dismissEvent = useActivityStore((state) => state.dismissEvent)
  const setFeedOpen = useActivityStore((state) => state.setOpen)
  const [filter, setFilter] = useState<string | null>(null)
  const [severity, setSeverity] = useState<ActivitySeverity | 'all'>('all')
  const [query, setQuery] = useState('')
  const [frozen, setFrozen] = useState<ActivityEvent[] | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null)
  const [copied, setCopied] = useState(false)

  const source = frozen ?? events
  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const event of source) counts.set(event.category, (counts.get(event.category) ?? 0) + 1)
    return Array.from(counts.entries()).sort((left, right) => right[1] - left[1])
  }, [source])
  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    return source.filter((event) => {
      if (filter && event.category !== filter) return false
      if (severity !== 'all' && event.severity !== severity) return false
      if (!normalizedQuery) return true
      return `${event.title} ${event.type} ${event.detail ?? ''}`.toLowerCase().includes(normalizedQuery)
    })
  }, [source, filter, severity, query])
  const visible = filtered.slice(0, 8)
  const selectedEvent = selectedEventId == null
    ? null
    : source.find((event) => event.id === selectedEventId) ?? null
  const warningCount = useMemo(
    () => source.filter((event) => event.severity === 'warning' || event.severity === 'error').length,
    [source],
  )

  const openCenter = (eventId?: number) => {
    if (eventId != null) setSelectedEventId(eventId)
    setCopied(false)
    setConfigOpen(true)
    setFeedOpen(true)
  }
  const closeCenter = () => {
    setConfigOpen(false)
    setFeedOpen(false)
  }
  const toggleFrozen = () => setFrozen((current) => current ? null : [...events])
  const copySelected = async () => {
    if (!selectedEvent) return
    const text = [
      selectedEvent.title,
      `Type: ${selectedEvent.type}`,
      `Severity: ${selectedEvent.severity}`,
      `Time: ${new Date(selectedEvent.timestamp).toLocaleString()}`,
      '',
      formattedDetail(selectedEvent.detail),
    ].join('\n')
    await navigator.clipboard.writeText(text)
    setCopied(true)
  }

  const buildContext = () => [
    '## Recent BashGym activity',
    ...filtered.slice(0, 10).map((event) => `- [${event.severity}] ${event.title} (${event.type})`),
  ].join('\n')

  const chipClass = (active: boolean) => clsx(
    'nodrag px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider border-brutal rounded-brutal transition-press',
    active
      ? 'border-accent bg-accent/10 text-accent shadow-brutal-sm'
      : 'border-border bg-background-card text-text-muted hover:text-text-secondary hover:border-border',
  )

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={data.title}
        flowerVariant="activity"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? buildContext : undefined}
        statusBarClass={visible.some((event) => event.severity === 'error') ? 'bg-status-error' : 'bg-background-tertiary'}
        hue={hueFor('activity')}
        headerRight={
          <>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                openCenter()
              }}
              className="nodrag node-btn node-btn-accent"
              title="Open Activity Center"
              aria-label="Open Activity Center"
            >
              <SlidersHorizontal className="w-3 h-3" />
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                toggleFrozen()
              }}
              className={clsx('nodrag node-btn', frozen ? 'node-btn-warning' : 'node-btn-accent')}
              title={frozen ? 'Resume live activity' : 'Pause activity updates'}
              aria-pressed={Boolean(frozen)}
            >
              {frozen ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
            </button>
          </>
        }
        onFocus={data.onFocus}
        onClose={data.onClose}
      >
        <div className="flex flex-wrap gap-1 mb-2" aria-label="Activity filters">
          {COMPACT_FILTERS.map((category) => (
            <button
              key={category}
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                setFilter(filter === category ? null : category)
              }}
              className={chipClass(filter === category)}
              aria-pressed={filter === category}
            >
              {category}
            </button>
          ))}
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              setSeverity(severity === 'error' ? 'all' : 'error')
            }}
            className={chipClass(severity === 'error')}
            aria-pressed={severity === 'error'}
            title="Only errors"
          >
            errors
          </button>
        </div>
        {visible.length === 0 ? (
          <button
            type="button"
            className="nodrag w-full text-[10px] font-mono text-text-muted text-center py-2 hover:text-accent"
            onClick={(event) => {
              event.stopPropagation()
              openCenter()
            }}
          >
            {source.length === 0 ? 'No activity yet — open Activity Center' : 'No events match these filters'}
          </button>
        ) : (
          <div className="space-y-1">
            {visible.map((event) => (
              <button
                type="button"
                key={event.id}
                className="nodrag flex w-full items-center gap-1.5 text-left text-[10px] font-mono hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                onClick={(clickEvent) => {
                  clickEvent.stopPropagation()
                  openCenter(event.id)
                }}
                title={`Inspect ${event.title}`}
              >
                <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', SEVERITY_DOT[event.severity])} />
                <span className="flex-1 truncate text-text-secondary">{event.title}</span>
                <span className="text-text-muted flex-shrink-0">{relTime(event.timestamp)}</span>
              </button>
            ))}
          </div>
        )}
      </DataNodeShell>

      <NodeConfigModal
        isOpen={configOpen}
        onClose={closeCenter}
        title="Activity Center"
        description="Search, inspect, and manage activity across this workspace."
        size="xl"
        layout="workspace"
        footer={
          <>
            <button
              type="button"
              className={clsx('btn-secondary mr-auto', frozen && 'text-status-warning border-status-warning')}
              onClick={toggleFrozen}
            >
              {frozen ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              {frozen ? 'Resume live feed' : 'Pause feed'}
            </button>
            <button
              type="button"
              className="btn-secondary text-status-error border-status-error"
              disabled={events.length === 0}
              onClick={() => {
                if (events.length > 0 && window.confirm(`Clear all ${events.length} activity events?`)) {
                  clearEvents()
                  setSelectedEventId(null)
                }
              }}
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear all
            </button>
            <button type="button" className="btn-primary" onClick={closeCenter}>Done</button>
          </>
        }
      >
        <div className="activity-center">
          <div className="activity-center-summary" aria-label="Feed summary">
            <div>
              <span className={clsx('activity-center-live-dot', frozen && 'activity-center-live-dot-paused')} />
              <strong>{frozen ? 'Paused snapshot' : 'Live workspace feed'}</strong>
            </div>
            <span>{source.length} events</span>
            <span>{warningCount} need attention</span>
            <span>{filtered.length} shown</span>
          </div>

          <div className="activity-center-layout">
            <section className="activity-center-feed" aria-label="Activity events">
              <div className="activity-center-toolbar">
                <label className="activity-center-search">
                  <Search className="w-4 h-4" aria-hidden="true" />
                  <span className="sr-only">Search activity</span>
                  <input
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search events, tools, or details"
                  />
                  {query ? (
                    <button type="button" onClick={() => setQuery('')} aria-label="Clear search">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  ) : null}
                </label>

                <div className="activity-center-filter-row" aria-label="Filter by severity">
                  {(['all', 'error', 'warning', 'success', 'info'] as const).map((value) => (
                    <button
                      type="button"
                      key={value}
                      className={clsx('activity-center-filter', severity === value && 'activity-center-filter-active')}
                      onClick={() => setSeverity(value)}
                      aria-pressed={severity === value}
                    >
                      {value}
                    </button>
                  ))}
                </div>

                <div className="activity-center-filter-row" aria-label="Filter by category">
                  <button
                    type="button"
                    className={clsx('activity-center-filter', filter == null && 'activity-center-filter-active')}
                    onClick={() => setFilter(null)}
                    aria-pressed={filter == null}
                  >
                    All sources
                  </button>
                  {categoryCounts.map(([category, count]) => (
                    <button
                      type="button"
                      key={category}
                      className={clsx('activity-center-filter', filter === category && 'activity-center-filter-active')}
                      onClick={() => setFilter(filter === category ? null : category)}
                      aria-pressed={filter === category}
                    >
                      {CATEGORY_LABEL[category] ?? category} <span>{count}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="activity-center-event-list" role="list">
                {filtered.length === 0 ? (
                  <div className="activity-center-empty">
                    <Activity className="w-7 h-7" aria-hidden="true" />
                    <strong>{source.length === 0 ? 'No activity recorded yet' : 'No matching events'}</strong>
                    <span>{source.length === 0 ? 'Task, training, and service updates will appear here.' : 'Try clearing a filter or using a broader search.'}</span>
                  </div>
                ) : filtered.map((event) => {
                  const SeverityIcon = SEVERITY_ICON[event.severity]
                  return (
                    <button
                      type="button"
                      role="listitem"
                      key={event.id}
                      className={clsx('activity-center-event', selectedEventId === event.id && 'activity-center-event-selected')}
                      onClick={() => {
                        setSelectedEventId(event.id)
                        setCopied(false)
                      }}
                      aria-current={selectedEventId === event.id ? 'true' : undefined}
                    >
                      <SeverityIcon className={clsx('w-4 h-4', `activity-severity-${event.severity}`)} aria-hidden="true" />
                      <span className="activity-center-event-copy">
                        <strong>{event.title}</strong>
                        <span>{CATEGORY_LABEL[event.category] ?? event.category} · {relTime(event.timestamp)}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
            </section>

            <aside className="activity-center-detail" aria-label="Selected event details">
              {selectedEvent ? (
                <>
                  <div className="activity-center-detail-header">
                    <div>
                      <span className="activity-center-eyebrow">Event detail</span>
                      <h3>{selectedEvent.title}</h3>
                    </div>
                    <div className="flex gap-1.5">
                      <button type="button" className="node-btn node-btn-wide" onClick={() => void copySelected()}>
                        <Copy className="w-3.5 h-3.5" />
                        {copied ? 'Copied' : 'Copy'}
                      </button>
                      <button
                        type="button"
                        className="node-btn node-btn-danger"
                        onClick={() => {
                          dismissEvent(selectedEvent.id)
                          setSelectedEventId(null)
                        }}
                        title="Dismiss event"
                        aria-label="Dismiss selected event"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  <div className="activity-center-detail-meta">
                    <ConfigPill tone={selectedEvent.severity === 'error' ? 'error' : selectedEvent.severity === 'warning' ? 'warning' : selectedEvent.severity === 'success' ? 'success' : 'neutral'}>
                      {selectedEvent.severity}
                    </ConfigPill>
                    <span>{CATEGORY_LABEL[selectedEvent.category] ?? selectedEvent.category}</span>
                    <time dateTime={new Date(selectedEvent.timestamp).toISOString()}>
                      {new Date(selectedEvent.timestamp).toLocaleString()}
                    </time>
                  </div>
                  <div className="activity-center-detail-type">
                    <span>Event type</span>
                    <code>{selectedEvent.type}</code>
                  </div>
                  <pre className="activity-center-payload">{formattedDetail(selectedEvent.detail)}</pre>
                </>
              ) : (
                <div className="activity-center-empty">
                  <Activity className="w-8 h-8" aria-hidden="true" />
                  <strong>Select an event to inspect it</strong>
                  <span>Details, timestamps, and copy or dismiss actions will appear here.</span>
                </div>
              )}
            </aside>
          </div>
        </div>
      </NodeConfigModal>
    </>
  )
})
