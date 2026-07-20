import { useMemo, useState } from 'react'
import { ArrowUpRight, CircleAlert, CircleCheck, Info, Trash2, TriangleAlert, X } from 'lucide-react'
import { clsx } from 'clsx'

import { useActivityStore } from '../../stores/activityStore'
import type {
  ActivityDestination,
  ActivityEvent,
  ActivitySeverity,
} from '../../stores/activityStore'
import { useUIStore } from '../../stores/uiStore'

const SEVERITY_ICON: Record<ActivitySeverity, typeof Info> = {
  info: Info,
  success: CircleCheck,
  warning: TriangleAlert,
  error: CircleAlert,
}

const SEVERITY_COLOR: Record<ActivitySeverity, string> = {
  info: 'text-text-muted',
  success: 'text-status-success',
  warning: 'text-status-warning',
  error: 'text-status-error',
}

function timeAgo(ts: number): string {
  const seconds = Math.max(0, Math.floor((Date.now() - ts) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

function formattedDetail(detail?: string): string {
  if (!detail) return 'No additional event fields were recorded.'
  try {
    return JSON.stringify(JSON.parse(detail), null, 2)
  } catch {
    return detail
  }
}

interface ActivityFeedPanelProps {
  events: ActivityEvent[]
  enabledCategories: Set<string>
  selectedEventId: number | null
  onSelect: (eventId: number) => void
  onNavigate: (destination: ActivityDestination) => void
  onToggleCategory: (category: string) => void
  onClear: () => void
  onClose: () => void
}

export function ActivityFeedPanel({
  events,
  enabledCategories,
  selectedEventId,
  onSelect,
  onNavigate,
  onToggleCategory,
  onClear,
  onClose,
}: ActivityFeedPanelProps) {
  const categories = useMemo(
    () => Array.from(new Set(events.map((event) => event.category))).sort(),
    [events],
  )
  const visible = useMemo(
    () => enabledCategories.size === 0
      ? events
      : events.filter((event) => enabledCategories.has(event.category)),
    [events, enabledCategories],
  )
  const selectedEvent = selectedEventId == null
    ? null
    : events.find((event) => event.id === selectedEventId) ?? null

  return (
    <aside className="fixed bottom-10 right-0 top-12 z-40 flex w-[30rem] max-w-full flex-col border-l-2 border-border bg-background-card shadow-brutal" aria-label="Notifications">
      <header className="flex items-start justify-between gap-3 border-b-2 border-border px-4 py-3">
        <div>
          <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-accent">Workspace signal</p>
          <h2 className="font-serif text-xl font-semibold text-text-primary">Notifications</h2>
          <p className="mt-0.5 text-xs text-text-secondary">Inspect what changed, then open the relevant workspace.</p>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" onClick={onClear} title="Clear notifications" className="btn-icon text-text-secondary hover:text-status-error">
            <Trash2 className="h-4 w-4" />
          </button>
          <button type="button" onClick={onClose} title="Close notifications" className="btn-icon text-text-secondary hover:text-text-primary">
            <X className="h-4 w-4" />
          </button>
        </div>
      </header>

      {categories.length > 1 ? (
        <div className="flex flex-wrap gap-1 border-b border-border px-4 py-2" aria-label="Notification categories">
          {categories.map((category) => (
            <button
              type="button"
              key={category}
              onClick={() => onToggleCategory(category)}
              aria-pressed={enabledCategories.size === 0 || enabledCategories.has(category)}
              className={clsx(
                'border px-2 py-1 font-mono text-[11px] uppercase tracking-wide transition-colors',
                enabledCategories.size === 0 || enabledCategories.has(category)
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border text-text-muted',
              )}
            >
              {category}
            </button>
          ))}
        </div>
      ) : null}

      <div className={clsx('min-h-0 overflow-y-auto', selectedEvent ? 'max-h-[46%] border-b-2 border-border' : 'flex-1')} role="list">
        {visible.length === 0 ? (
          <div className="px-4 py-10 text-center text-xs leading-5 text-text-muted">
            No notifications yet. Training, campaign, trace, and service changes will appear here.
          </div>
        ) : visible.map((event) => {
          const Icon = SEVERITY_ICON[event.severity]
          return (
            <button
              type="button"
              role="listitem"
              key={event.id}
              aria-label={`Inspect ${event.title}`}
              aria-current={selectedEvent?.id === event.id ? 'true' : undefined}
              onClick={() => onSelect(event.id)}
              className={clsx(
                'flex w-full gap-3 border-b border-border-subtle px-4 py-3 text-left transition-colors hover:bg-background-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-accent',
                selectedEvent?.id === event.id && 'bg-background-secondary',
              )}
            >
              <Icon className={clsx('mt-0.5 h-4 w-4 shrink-0', SEVERITY_COLOR[event.severity])} />
              <span className="min-w-0 flex-1">
                <strong className="block text-xs leading-5 text-text-primary">{event.title}</strong>
                <span className="mt-0.5 flex items-center gap-2 font-mono text-[11px] text-text-muted">
                  <span>{event.category}</span><span aria-hidden="true">·</span><span>{timeAgo(event.timestamp)}</span>
                </span>
              </span>
            </button>
          )
        })}
      </div>

      {selectedEvent ? (
        <section className="min-h-0 flex-1 overflow-y-auto p-4" aria-label="Selected notification detail">
          <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-accent">Event detail</p>
          <h3 className="mt-1 font-serif text-lg font-semibold text-text-primary">{selectedEvent.title}</h3>
          <dl className="mt-3 grid grid-cols-[7rem_minmax(0,1fr)] gap-x-3 gap-y-1 border-y border-border py-2 font-mono text-[11px]">
            <dt className="text-text-muted">Type</dt><dd className="break-all text-text-primary">{selectedEvent.type}</dd>
            <dt className="text-text-muted">Severity</dt><dd className={SEVERITY_COLOR[selectedEvent.severity]}>{selectedEvent.severity}</dd>
            <dt className="text-text-muted">Recorded</dt><dd className="text-text-primary">{new Date(selectedEvent.timestamp).toLocaleString()}</dd>
          </dl>
          <pre className="mt-3 max-h-44 overflow-auto whitespace-pre-wrap break-words bg-background-secondary p-3 font-mono text-[11px] leading-5 text-text-secondary">{formattedDetail(selectedEvent.detail)}</pre>
          {selectedEvent.destination ? (
            <button type="button" className="btn-primary mt-3" onClick={() => onNavigate(selectedEvent.destination!)}>
              {selectedEvent.destination.label}<ArrowUpRight className="h-4 w-4" />
            </button>
          ) : null}
        </section>
      ) : (
        <div className="border-t-2 border-border px-4 py-4 text-xs text-text-muted">Select a notification to inspect its exact event fields.</div>
      )}
    </aside>
  )
}

export function ActivityFeed() {
  const { events, isOpen, enabledCategories, setOpen, toggleCategory, clear } = useActivityStore()
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null)
  const openOverlay = useUIStore((state) => state.openOverlay)
  const openTraining = useUIStore((state) => state.openTraining)

  if (!isOpen) return null

  const navigate = (destination: ActivityDestination) => {
    if (destination.view === 'autoresearch') {
      openTraining('autoresearch', {
        workspaceId: destination.workspaceId ?? null,
        campaignId: destination.campaignId ?? null,
      })
    } else if (destination.view === 'training') {
      openTraining('runs')
    } else {
      openOverlay(destination.view)
    }
    setOpen(false)
  }

  return (
    <ActivityFeedPanel
      events={events}
      enabledCategories={enabledCategories}
      selectedEventId={selectedEventId}
      onSelect={setSelectedEventId}
      onNavigate={navigate}
      onToggleCategory={toggleCategory}
      onClear={() => { clear(); setSelectedEventId(null) }}
      onClose={() => setOpen(false)}
    />
  )
}
