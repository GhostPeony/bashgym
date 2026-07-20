/**
 * Unified agent activity feed store.
 *
 * Collects every tracked WebSocket event (training, traces, orchestration,
 * pipeline, guardrails, cascade, HF, autoresearch, integrations) into a single
 * chronological feed with severity classification and category filters.
 */
import { create } from 'zustand'

export type ActivitySeverity = 'info' | 'success' | 'warning' | 'error'

export interface ActivityDestination {
  label: string
  view:
    | 'autoresearch'
    | 'training'
    | 'traces'
    | 'guardrails'
    | 'router'
    | 'huggingface'
    | 'factory'
    | 'pipeline'
    | 'orchestrator'
    | 'integration'
  workspaceId?: string
  campaignId?: string
}

export interface ActivityEvent {
  id: number
  /** Stable identity for mutation acknowledgements that may also arrive by WebSocket. */
  key?: string
  /** Raw WS message type, e.g. "orchestration:task:started" */
  type: string
  /** Top-level category derived from the type prefix, e.g. "orchestration" */
  category: string
  severity: ActivitySeverity
  title: string
  detail?: string
  destination?: ActivityDestination
  timestamp: number
}

const MAX_EVENTS = 500

const COMPACTED_EVENT_TYPES = new Set([
  'training:progress',
  'training:log',
  'hf:job:log',
  'hf:job:metrics',
  'router:stats',
  'orchestration:budget:update',
  'cascade:progress',
  'campaign:training-metrics-appended'
])

let nextId = 1

const SEVERITY_RULES: Array<[RegExp, ActivitySeverity]> = [
  [/(:failed|:error|guardrail:blocked)$/, 'error'],
  [/(:retrying|guardrail:warn|:cancelled)$/, 'warning'],
  [/(:complete|:completed|:ready|trace:promoted|threshold_reached)$/, 'success']
]

export function severityFor(type: string): ActivitySeverity {
  if (type === 'hf-context:stale') return 'warning'
  if (type === 'hf-context:discovery-cancelled') return 'warning'
  if (
    [
      'hf-context:discovery-completed',
      'hf-context:pinned',
      'hf-context:activated',
      'hf-context:sent',
      'hf-context:eval-prepared'
    ].includes(type)
  )
    return 'success'
  for (const [re, sev] of SEVERITY_RULES) {
    if (re.test(type)) return sev
  }
  return 'info'
}

/** Build a human title from a WS message. Falls back to the type itself. */
export function titleFor(type: string, payload: Record<string, unknown>): string {
  const p = payload as Record<string, any>
  switch (true) {
    case type === 'training:queued':
      return `${String(p.strategy || 'training').toUpperCase()} run queued`
    case type === 'training:progress':
      return `Training step ${p.step ?? '?'} — loss ${typeof p.loss === 'number' ? p.loss.toFixed(4) : '?'}`
    case type === 'training:complete':
      return 'Training run complete'
    case type === 'training:failed':
      return `Training failed${p.error ? `: ${p.error}` : ''}`
    case type === 'campaign:created':
      return `Campaign created${p.campaign_id ? ` — ${p.campaign_id}` : ''}`
    case type === 'campaign:started':
      return 'Campaign started'
    case type === 'campaign:paused':
      return 'Campaign paused'
    case type === 'campaign:resumed':
      return 'Campaign resumed'
    case type === 'campaign:cancelled':
      return 'Campaign cancelled'
    case type === 'campaign:action-completed':
      return `Campaign action complete${p.action_id ? ` — ${p.action_id}` : ''}`
    case type === 'campaign:action-failed':
      return `Campaign action failed${p.action_id ? ` — ${p.action_id}` : ''}`
    case type === 'campaign:gate-decided':
      return `Campaign gate ${p.verdict || 'decided'}`
    case type === 'campaign:training-metrics-appended':
      return 'Campaign training metrics updated'
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
    case type === 'designer:queued':
      return `Data Designer job queued${p.pipeline ? ` — ${p.pipeline}` : ''}`
    case type === 'designer:completed':
      return `Data Designer job complete${p.pipeline ? ` — ${p.pipeline}` : ''}`
    case type === 'designer:failed':
      return `Data Designer job failed${p.error ? `: ${p.error}` : ''}`
    case type === 'hf:dataset:completed':
      return `Dataset published${p.repo_id ? ` — ${p.repo_id}` : ''}`
    case type === 'hf-context:discovery-completed':
      return `Hugging Face evidence ready${p.evidence_count != null ? ` — ${p.evidence_count} items` : ''}`
    case type === 'hf-context:discovery-started':
      return 'Hugging Face evidence discovery started'
    case type === 'hf-context:discovery-cancelled':
      return `Hugging Face evidence discovery cancelled${p.evidence_count != null ? ` — kept ${p.evidence_count} items` : ''}`
    case type === 'hf-context:pinned':
      return `Hugging Face context pinned${p.version ? ` — v${p.version}` : ''}`
    case type === 'hf-context:activated':
      return `Hugging Face context activated${p.version ? ` — v${p.version}` : ''}`
    case type === 'hf-context:deactivated':
      return 'Hugging Face context deactivated'
    case type === 'hf-context:sent':
      return 'Hugging Face context sent to terminal'
    case type === 'hf-context:eval-prepared':
      return 'Hugging Face Eval preview prepared'
    case type === 'hf-context:stale':
      return 'Hugging Face context is stale'
    case type === 'skill-eval:started':
      return `Skill eval started${p.skill_name ? ` — ${p.skill_name}` : ''}`
    case type === 'skill-eval:prepared':
      return `Skill Lab prepared${p.skill_name ? ` — ${p.skill_name}` : ''}`
    case type === 'skill-eval:skill-saved':
      return `Skill saved${p.skill_name ? ` — ${p.skill_name}` : ''}`
    case type === 'skill-eval:completed':
      return `Skill eval ${p.verdict || 'completed'}${p.skill_name ? ` — ${p.skill_name}` : ''}`
    case type === 'skill-eval:failed':
      return `Skill eval failed${p.skill_name ? ` — ${p.skill_name}` : ''}`
    default:
      return type
  }
}

export function eventKeyFor(type: string, payload: Record<string, unknown>): string | undefined {
  if (type.startsWith('campaign:') && typeof payload.event_id === 'string' && payload.event_id)
    return `${type}:${payload.event_id}`
  const entityId =
    payload.run_id ??
    payload.job_id ??
    payload.task_id ??
    payload.stage_id ??
    payload.attempt_id ??
    payload.action_id ??
    payload.campaign_id
  if (COMPACTED_EVENT_TYPES.has(type)) {
    return `${type}:${typeof entityId === 'string' && entityId ? entityId : 'active'}`
  }
  if (typeof payload.idempotency_key === 'string' && payload.idempotency_key) {
    return payload.idempotency_key
  }
  if (typeof entityId !== 'string' || !entityId) return undefined
  if (
    type === 'training:queued' ||
    type === 'training:complete' ||
    type === 'training:failed' ||
    type.startsWith('designer:') ||
    type.startsWith('skill-eval:') ||
    type.startsWith('campaign:')
  ) {
    return `${type}:${entityId}`
  }
  return undefined
}

const PUBLIC_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/

export function destinationFor(
  type: string,
  payload: Record<string, unknown>
): ActivityDestination | undefined {
  const category = type.startsWith('hf-context:') ? 'hf' : type.split(':')[0]
  if (category === 'campaign' || category === 'autoresearch') {
    const workspaceId =
      typeof payload.workspace_id === 'string' && PUBLIC_ID.test(payload.workspace_id)
        ? payload.workspace_id
        : undefined
    const campaignId =
      typeof payload.campaign_id === 'string' && PUBLIC_ID.test(payload.campaign_id)
        ? payload.campaign_id
        : undefined
    return { label: 'Open AutoResearch', view: 'autoresearch', workspaceId, campaignId }
  }
  const mapping: Record<string, ActivityDestination> = {
    training: { label: 'Open Training', view: 'training' },
    'skill-eval': { label: 'Open Training', view: 'training' },
    trace: { label: 'Open Traces', view: 'traces' },
    guardrail: { label: 'Open Guardrails', view: 'guardrails' },
    router: { label: 'Open Router', view: 'router' },
    cascade: { label: 'Open Router', view: 'router' },
    hf: { label: 'Open Hugging Face', view: 'huggingface' },
    designer: { label: 'Open Data Factory', view: 'factory' },
    'schema-research': { label: 'Open Data Factory', view: 'factory' },
    pipeline: { label: 'Open Pipeline', view: 'pipeline' },
    orchestration: { label: 'Open Orchestrator', view: 'orchestrator' },
    verification: { label: 'Open Orchestrator', view: 'orchestrator' },
    integration: { label: 'Open Integrations', view: 'integration' }
  }
  return mapping[category]
}

interface ActivityState {
  events: ActivityEvent[]
  unread: number
  isOpen: boolean
  /** Category filters; empty set = show all */
  enabledCategories: Set<string>
  addEvent: (type: string, payload: Record<string, unknown>) => void
  removeEvent: (key: string) => void
  dismissEvent: (id: number) => void
  setOpen: (open: boolean) => void
  toggleCategory: (category: string) => void
  clear: () => void
}

const TRACKED_PREFIXES = [
  'training',
  'trace',
  'orchestration',
  'pipeline',
  'guardrail',
  'cascade',
  'hf',
  'hf-context',
  'autoresearch',
  'integration',
  'schema-research',
  'verification',
  'designer',
  'skill-eval',
  'router',
  'campaign'
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
      const category = type.startsWith('hf-context:') ? 'hf' : type.split(':')[0]
      const key = eventKeyFor(type, payload)
      const existingIndex = key ? state.events.findIndex((event) => event.key === key) : -1
      const compacted = COMPACTED_EVENT_TYPES.has(type)
      if (existingIndex >= 0 && !compacted) return state
      const now = Date.now()
      const existing = existingIndex >= 0 ? state.events[existingIndex] : undefined
      // Progress can arrive many times per second. Four feed refreshes per second
      // is enough for legibility while keeping the canvas main thread available.
      if (existing && compacted && now - existing.timestamp < 250) return state
      const event: ActivityEvent = {
        id: existing?.id ?? nextId++,
        key,
        type,
        category,
        severity: severityFor(type),
        title: titleFor(type, payload),
        detail: typeof payload === 'object' ? JSON.stringify(payload).slice(0, 500) : undefined,
        destination: destinationFor(type, payload),
        timestamp: now
      }
      let events = state.events
      if (existingIndex >= 0) events = events.filter((_, index) => index !== existingIndex)
      events = [event, ...events].slice(0, MAX_EVENTS)
      return {
        events,
        unread: state.isOpen ? 0 : state.unread + (compacted ? 0 : 1)
      }
    })
  },

  removeEvent: (key) =>
    set((state) => {
      const events = state.events.filter((event) => event.key !== key)
      const removed = state.events.length - events.length
      if (removed === 0) return state
      return {
        events,
        unread: state.isOpen ? 0 : Math.max(0, state.unread - removed)
      }
    }),

  dismissEvent: (id) =>
    set((state) => {
      const events = state.events.filter((event) => event.id !== id)
      return events.length === state.events.length ? state : { events }
    }),

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
