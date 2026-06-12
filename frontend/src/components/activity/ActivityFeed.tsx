import { useMemo } from 'react'
import { X, Trash2, CircleAlert, CircleCheck, Info, TriangleAlert } from 'lucide-react'
import { useActivityStore } from '../../stores/activityStore'
import type { ActivitySeverity } from '../../stores/activityStore'
import { clsx } from 'clsx'

const SEVERITY_ICON: Record<ActivitySeverity, typeof Info> = {
  info: Info,
  success: CircleCheck,
  warning: TriangleAlert,
  error: CircleAlert
}

const SEVERITY_COLOR: Record<ActivitySeverity, string> = {
  info: 'text-text-muted',
  success: 'text-status-success',
  warning: 'text-status-warning',
  error: 'text-status-error'
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.floor(m / 60)}h ago`
}

export function ActivityFeed() {
  const { events, isOpen, enabledCategories, setOpen, toggleCategory, clear } = useActivityStore()

  const categories = useMemo(
    () => Array.from(new Set(events.map((e) => e.category))).sort(),
    [events]
  )

  const visible = useMemo(
    () =>
      enabledCategories.size === 0
        ? events
        : events.filter((e) => enabledCategories.has(e.category)),
    [events, enabledCategories]
  )

  if (!isOpen) return null

  return (
    <div className="fixed right-0 top-12 bottom-10 z-40 flex w-96 flex-col border-l border-border bg-background-card shadow-brutal">
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <span className="font-mono text-xs uppercase tracking-widest text-text-primary">Activity</span>
        <div className="flex items-center gap-1">
          <button
            onClick={clear}
            title="Clear"
            className="rounded-brutal p-1 text-text-secondary hover:bg-background-secondary hover:text-text-primary transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setOpen(false)}
            title="Close"
            className="rounded-brutal p-1 text-text-secondary hover:bg-background-secondary hover:text-text-primary transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
      {categories.length > 1 && (
        <div className="flex flex-wrap gap-1 border-b border-border px-3 py-2">
          {categories.map((c) => (
            <button
              key={c}
              onClick={() => toggleCategory(c)}
              className={clsx(
                'rounded-full border-brutal px-2 py-0.5 font-mono text-xs transition-colors',
                enabledCategories.size === 0 || enabledCategories.has(c)
                  ? 'border-accent text-accent'
                  : 'border-border text-text-muted'
              )}
            >
              {c}
            </button>
          ))}
        </div>
      )}
      <div className="flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <div className="px-3 py-8 text-center font-mono text-xs text-text-muted">
            No activity yet. Agent, training, and pipeline events appear here live.
          </div>
        ) : (
          visible.map((e) => {
            const Icon = SEVERITY_ICON[e.severity]
            return (
              <div key={e.id} className="flex gap-2 border-b border-border-subtle px-3 py-2">
                <Icon className={clsx('mt-0.5 w-4 h-4 shrink-0', SEVERITY_COLOR[e.severity])} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm text-text-primary">{e.title}</div>
                  <div className="flex items-center gap-2 font-mono text-xs text-text-muted">
                    <span className="rounded-brutal bg-background-secondary px-1">{e.category}</span>
                    <span>{timeAgo(e.timestamp)}</span>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
