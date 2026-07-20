import type {
  CampaignControlRoomSnapshotV1,
  CandidateSummaryV1,
  DecisionSurfaceV1,
  HumanWorkSummaryV1,
  JourneyPhaseSummaryV1,
  MetricDescriptorV1,
  ReadinessSummaryV1,
} from '../../services/api'
import type { CampaignFreshness } from '../../stores/campaignFreshness'

export type ControlRoomFreshness = CampaignFreshness

export function resolveControlRoomFreshness(input: {
  detailFreshness?: ControlRoomFreshness
  workspaceFreshness?: ControlRoomFreshness
  workspaceLoading?: boolean
  workspaceError?: string | null
}): ControlRoomFreshness {
  if (input.detailFreshness) return input.detailFreshness
  if (input.workspaceFreshness) return input.workspaceFreshness
  if (input.workspaceLoading) return 'reconciling'
  if (input.workspaceError) return 'error'
  return 'live'
}

export type PresentationTone = 'success' | 'warning' | 'error' | 'info' | 'neutral'

export function campaignStatusTone(status: string): PresentationTone {
  if (status === 'failed') return 'error'
  if (status === 'completed' || status === 'exhausted') return 'success'
  if (status === 'paused' || status === 'awaiting_authority' || status === 'cancelling') return 'warning'
  if (status === 'validating' || status === 'ready' || status === 'active') return 'info'
  return 'neutral'
}

export interface JourneyPhaseViewModel {
  id: string
  label: string
  stateLabel: string
  tone: PresentationTone
  evidenceCount: number
  executionOwner: string
  attentionOwner: string
  blocker: ReturnType<typeof presentBlocker>
}

export interface BlockerViewModel {
  code: string
  summary: string
  evidenceIds: string[]
  secondaryCodes: string[]
}

export interface NeedsYouViewModel {
  sentence: string
  code: string | null
}

export interface MetricViewModel {
  id: string
  label: string
  unit: string | null
  direction: string
  target: number | null
  value: 'Unavailable'
  delta: null
  comparabilityKey: string
}

export type ControlRoomViewModel =
  | { kind: 'loading'; message: string; authoritative: false }
  | { kind: 'empty'; message: string; authoritative: false }
  | { kind: 'error' | 'offline'; message: string; authoritative: false }
  | {
      kind: 'snapshot'
      snapshot: CampaignControlRoomSnapshotV1
      authoritative: boolean
      freshness: ControlRoomFreshness
      freshnessLabel: string
      controllerLabel: string
      error: string | null
      journey: JourneyPhaseViewModel[]
      needsYou: NeedsYouViewModel | null
      metrics: MetricViewModel[]
    }

const PHASE_LABELS: Record<string, string> = {
  setup: 'Setup',
  baseline: 'Baseline',
  experiments: 'Experiments',
  human_review: 'Human review',
  decision: 'Decision',
}

const STATE_TONES: Record<string, PresentationTone> = {
  not_started: 'neutral',
  ready: 'info',
  active: 'info',
  blocked: 'warning',
  complete: 'success',
  failed: 'error',
  skipped: 'neutral',
}

function readable(value: string): string {
  const words = value.replace(/[_-]+/g, ' ').trim()
  return words ? words.replace(/\b\w/g, (letter) => letter.toUpperCase()) : 'Unknown'
}

function ownerLabel(value: string): string {
  if (value === 'bashgym') return 'BashGym'
  return readable(value)
}

export function presentJourney(
  phases: readonly JourneyPhaseSummaryV1[],
): JourneyPhaseViewModel[] {
  return phases.map((phase) => ({
    id: phase.phase_id,
    label: PHASE_LABELS[phase.phase_id] || readable(phase.phase_id),
    stateLabel: readable(phase.state),
    tone: STATE_TONES[phase.state] || 'neutral',
    evidenceCount: phase.evidence_count,
    executionOwner: ownerLabel(phase.execution_owner),
    attentionOwner: ownerLabel(phase.attention_owner),
    blocker: phase.primary_blocker ? {
      code: phase.primary_blocker.code,
      summary: phase.primary_blocker.summary,
      evidenceIds: [...phase.primary_blocker.evidence_ids],
      secondaryCodes: [...phase.primary_blocker.secondary_codes],
    } : null,
  }))
}

export function presentBlocker(surface: DecisionSurfaceV1): BlockerViewModel | null {
  const blocker = surface.blocker
  return blocker ? {
    code: blocker.code,
    summary: blocker.summary,
    evidenceIds: [...blocker.evidence_ids],
    secondaryCodes: [...blocker.secondary_codes],
  } : null
}

/**
 * Distils the one thing (if any) that is waiting on the human into a plain
 * sentence plus the trailing machine code. Combines the server's primary
 * blocker, the launch-readiness block, and pending human review. Returns null
 * when nothing needs the user, so the caller can render nothing at all.
 */
export function presentNeedsYou(input: {
  surface: DecisionSurfaceV1
  readiness: ReadinessSummaryV1
  humanWork: HumanWorkSummaryV1
  status: string
}): NeedsYouViewModel | null {
  const { surface, readiness, humanWork, status } = input
  if (surface.blocker) {
    return { sentence: surface.blocker.summary, code: surface.blocker.code }
  }
  if (humanWork.blocking_count > 0) {
    const noun = humanWork.blocking_count === 1 ? 'review is' : 'reviews are'
    return {
      sentence: `${humanWork.blocking_count} blinded ${noun} waiting on your decision before this campaign can continue.`,
      code: 'human_review_required',
    }
  }
  if (status === 'ready' && !readiness.launch_ready) {
    const code = readiness.blocking_codes[0] ?? null
    const detail = code ? readable(code) : 'a readiness check'
    return {
      sentence: `Readiness is blocked by ${detail}. Fix it and readiness re-verifies automatically before Start unlocks.`,
      code,
    }
  }
  if (humanWork.open_count > 0) {
    const noun = humanWork.open_count === 1 ? 'review is' : 'reviews are'
    return {
      sentence: `${humanWork.open_count} human ${noun} open for you to pick up.`,
      code: 'human_review_open',
    }
  }
  return null
}

export function presentMetrics(
  metrics: readonly MetricDescriptorV1[],
  _champion: CandidateSummaryV1 | null,
  _candidate: CandidateSummaryV1 | null,
): MetricViewModel[] {
  return metrics.map((metric) => ({
    id: metric.metric_id,
    label: metric.display_name,
    unit: metric.unit,
    direction: readable(metric.direction),
    target: metric.target,
    value: 'Unavailable',
    delta: null,
    comparabilityKey: metric.comparability_key,
  }))
}

function freshnessLabel(freshness: ControlRoomFreshness): string {
  return {
    live: 'Live',
    reconciling: 'Reconciling campaign state',
    stale: 'Cached campaign state · stale',
    offline: 'Cached campaign state · offline',
    error: 'Campaign state error',
  }[freshness]
}

export function buildControlRoomModel(input: {
  snapshot: CampaignControlRoomSnapshotV1 | null
  freshness: ControlRoomFreshness
  error: string | null
}): ControlRoomViewModel {
  if (!input.snapshot) {
    if (input.freshness === 'reconciling') {
      return { kind: 'loading', message: 'Loading campaign state…', authoritative: false }
    }
    if (input.freshness === 'offline') {
      return { kind: 'offline', message: input.error || 'BashGym desktop is offline.', authoritative: false }
    }
    if (input.freshness === 'error') {
      return { kind: 'error', message: input.error || 'Unable to load campaign state.', authoritative: false }
    }
    return { kind: 'empty', message: 'No durable campaign is selected.', authoritative: false }
  }

  return {
    kind: 'snapshot',
    snapshot: input.snapshot,
    authoritative: input.freshness === 'live',
    freshness: input.freshness,
    freshnessLabel: freshnessLabel(input.freshness),
    controllerLabel: readable(input.snapshot.controller.state),
    error: input.error,
    journey: presentJourney(input.snapshot.journey),
    needsYou: presentNeedsYou({
      surface: input.snapshot.decision_surface,
      readiness: input.snapshot.readiness,
      humanWork: input.snapshot.human_work,
      status: input.snapshot.campaign.status,
    }),
    metrics: presentMetrics(input.snapshot.metrics, input.snapshot.champion, input.snapshot.candidate),
  }
}
