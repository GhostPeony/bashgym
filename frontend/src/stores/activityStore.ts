/**
 * Unified agent activity feed store.
 *
 * Collects every tracked WebSocket event (training, traces, orchestration,
 * pipeline, guardrails, cascade, HF, autoresearch, integrations) into a single
 * chronological feed with severity classification and category filters.
 */
import { create } from 'zustand'

export type ActivitySeverity = 'info' | 'success' | 'warning' | 'error'

export interface ActivityEvent {
  id: number
  /** Raw WS message type, e.g. "orchestration:task:started" */
  type: string
  /** Top-level category derived from the type prefix, e.g. "orchestration" */
  category: string
  severity: ActivitySeverity
  title: string
  detail?: string
  timestamp: number
}

const MAX_EVENTS = 500

let nextId = 1

const SEVERITY_RULES: Array<[RegExp, ActivitySeverity]> = [
  [/(:failed|:error|guardrail:blocked)$/, 'error'],
  [/(:retrying|guardrail:warn|:cancelled)$/, 'warning'],
  [/(:complete|:completed|:ready|trace:promoted|threshold_reached)$/, 'success'],
]

export function severityFor(type: string): ActivitySeverity {
  for (const [re, sev] of SEVERITY_RULES) {
    if (re.test(type)) return sev
  }
  return 'info'
}

/** Build a human title from a WS message. Falls back to the type itself. */
export function titleFor(type: string, payload: Record<string, unknown>): string {
  const p = payload as Record<string, any>
  switch (true) {
    case type === 'training:progress':
      return `Training step ${p.step ?? '?'} — loss ${typeof p.loss === 'number' ? p.loss.toFixed(4) : '?'}`
    case type === 'training:complete':
      return 'Training run complete'
    case type === 'training:failed':
      return `Training failed${p.error ? `: ${p.error}` : ''}`
    case type.startsWith('orchestration:task'):
      return `${p.task_title ?? p.task_id ?? 'task'} — ${type.split(':').pop()}`
    case type === 'trace:added':
      return `Trace ingested${p.repo ? ` (${p.repo})` : ''}`
    case type === 'trace:promoted':
      return 'Trace promoted to gold'
    case type === 'trace:demoted':
      return 'Trace demoted'
    case type.startsWith('pipeline:'):
      return `Pipeline: ${type.split(':').slice(1).join(' ')}`
    case type.startsWith('guardrail:'):
      return `Guardrail ${type.split(':')[1]}${p.rule ? ` — ${p.rule}` : ''}`
    case type === 'verification:result':
      return `Verification ${(p.passed ?? p.success) ? 'passed' : 'failed'}${p.task_id ? ` — ${p.task_id}` : ''}`
    default:
      return type
  }
}

interface ActivityState {
  events: ActivityEvent[]
  unread: number
  isOpen: boolean
  /** Category filters; empty set = show all */
  enabledCategories: Set<string>
  addEvent: (type: string, payload: Record<string, unknown>) => void
  setOpen: (open: boolean) => void
  toggleCategory: (category: string) => void
  clear: () => void
}

const TRACKED_PREFIXES = [
  'training', 'trace', 'orchestration', 'pipeline', 'guardrail',
  'cascade', 'hf', 'autoresearch', 'integration', 'schema-research',
  'verification'
]

export function isTrackedType(type: string): boolean {
  return TRACKED_PREFIXES.some((p) => type === p || type.startsWith(p + ':'))
}

export const useActivityStore = create<ActivityState>((set) => ({
  events: [],
  unread: 0,
  isOpen: false,
  enabledCategories: new Set(),

  addEvent: (type, payload) => {
    if (!isTrackedType(type)) return
    // training:progress fires every step — keep only the latest one in the feed
    set((state) => {
      const category = type.split(':')[0]
      const event: ActivityEvent = {
        id: nextId++,
        type,
        category,
        severity: severityFor(type),
        title: titleFor(type, payload),
        detail: typeof payload === 'object' ? JSON.stringify(payload).slice(0, 500) : undefined,
        timestamp: Date.now()
      }
      let events = state.events
      if (type === 'training:progress' && events[0]?.type === 'training:progress') {
        events = events.slice(1)
      }
      events = [event, ...events].slice(0, MAX_EVENTS)
      return {
        events,
        unread: state.isOpen ? 0 : state.unread + (type === 'training:progress' ? 0 : 1)
      }
    })
  },

  setOpen: (open) => set((state) => ({ isOpen: open, unread: open ? 0 : state.unread })),

  toggleCategory: (category) =>
    set((state) => {
      const next = new Set(state.enabledCategories)
      if (next.has(category)) next.delete(category)
      else next.add(category)
      return { enabledCategories: next }
    }),

  clear: () => set({ events: [], unread: 0 })
}))
