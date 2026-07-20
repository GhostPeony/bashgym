import { create } from 'zustand'
import {
  toCampaignActivityFields,
  toCampaignPublicArtifact,
  type CampaignArtifact,
  type CampaignEventItem,
} from '../campaignVisibility'
import {
  campaignApi,
  type ApiResponse,
  type CampaignControlRoomSnapshotV1,
} from '../services/api'
import { useActivityStore } from './activityStore'
import type { ControlRoomFreshness } from '../components/autoresearch/controlRoomModel'
import {
  acknowledgeCampaignSubscription,
  applyCampaignHint,
  beginCampaignReconciliation,
  createCampaignFreshness,
  expireCampaignCursor,
  failCampaignReconciliation,
  finishCampaignReconciliation,
  markCampaignDisconnected,
  markCampaignError,
  markCampaignReconnected,
  reconciliationRetryDelay,
  snapshotSemanticIdentity,
  validateCampaignSnapshot,
  type CampaignHintV1,
  type CampaignReconciliationState,
} from './campaignFreshness'

export type { CampaignArtifact, CampaignEventItem } from '../campaignVisibility'

export type CampaignStatus =
  | 'draft'
  | 'validating'
  | 'ready'
  | 'active'
  | 'paused'
  | 'awaiting_authority'
  | 'cancelling'
  | 'completed'
  | 'exhausted'
  | 'failed'
  | 'cancelled'

export interface CampaignRecord {
  schema_version: string
  campaign_id: string
  workspace_id: string
  title: string
  kind: string
  objective: string
  target_model: Record<string, unknown>
  owner_actor_id: string
  manifest_revision: number
  status: CampaignStatus
  prior_scheduling_status?: 'active' | 'paused' | null
  active_study_id?: string | null
  active_action_id?: string | null
  champion_ref?: string | null
  best_development_candidate_ref?: string | null
  stop_reason?: string | null
  version: number
  created_at: string
  updated_at: string
}

export interface CampaignControllerStatus {
  schema_version: string
  online: boolean
  state: 'online' | 'stale' | 'offline'
  code: 'controller_online' | 'controller_stale' | 'controller_offline' | string
  observed_at: string
  heartbeat_age_seconds?: number | null
  guidance?: string | null
  owner_id?: string | null
  generation?: number | null
  heartbeat_at?: string | null
  expires_at?: string | null
}

export interface CampaignAttempt {
  schema_version: 'public_campaign_attempt.v1'
  attempt_id: string
  workspace_id: string
  campaign_id: string
  study_id: string
  action_id: string
  attempt_number: number
  claim_generation: number
  status: string
  input_digest: string
  candidate_digest: string
  manifest_revision: number
  stage: string
  executor_kind: string | null
  created_at: string
  updated_at: string
}

export interface CampaignStudy {
  study_id: string
  workspace_id: string
  campaign_id: string
  proposal_id: string
  status: string
  stage_plan: {
    items: Array<{ stage: string; disposition: string; reason: string }>
  }
  current_stage_index: number
  candidate_digest: string
  version: number
  created_at: string
  updated_at: string
}

export interface CampaignProposalRecord {
  proposal: {
    proposal_id: string
    hypothesis: string
    study_family: string
    primary_variable: string
    expected_outcome: string
    falsification_criterion: string
    estimated_cost: number
    status: string
    created_at: string
  }
  validation: {
    valid: boolean
    reason_codes: string[]
  }
  study_id?: string | null
  updated_at: string
}

export interface CampaignEvidence {
  workspace_id: string
  campaign_id: string
  campaign_version: number
  manifest_revision: number
  status: CampaignStatus
  objective: string
  champion_ref?: string | null
  best_development_candidate_ref?: string | null
  approved_data_scopes: string[]
  compute_profile_id: string
  budget_remaining: Record<string, number>
  proposal_counts: Record<string, number>
  available_executors: string[]
  active_study_id?: string | null
  active_action_id?: string | null
  snapshot_digest: string
  created_at: string
}

export interface CampaignComparison {
  champion_digest: string
  candidate_digest: string
  sample_count: number
  video_count: number
  metrics: Record<string, number | null>
  slice_metrics: Record<string, Record<string, number>>
  verdict: 'passed' | 'failed' | 'insufficient_evidence'
  blocking_reasons: string[]
  warnings: string[]
  comparison_digest: string
  created_at: string
}

export interface CampaignMetricValue {
  step: number
  source: string
  value: number
  observed_at: string
}

export interface CampaignLedgerEvaluation {
  evaluation_result_id: string
  evaluation_suite_id: string
  run_id: string
  status: string
  metrics: Record<string, number>
  created_at: string
}

export interface CampaignLedgerDecision {
  decision_id: string
  experiment_id: string
  run_id?: string | null
  decision_type: string
  outcome: string
  rationale: string
  evidence_refs: string[]
  created_at: string
}

export interface CampaignLedgerArtifact {
  artifact_id: string
  run_id: string
  attempt_id?: string | null
  kind: string
  sha256: string
  size_bytes: number
  media_type: string
  metadata: Record<string, unknown>
  created_at: string
}

export interface CampaignLedgerProject {
  project: { project_id: string; display_name: string; status: string }
  experiments: Array<Record<string, unknown> & { experiment_id: string }>
  runs: Array<Record<string, unknown> & {
    run_id: string
    status: string
    run_kind?: string
    is_simulation?: boolean
    model_version_id?: string | null
    queued_at?: string
  }>
  evaluations: CampaignLedgerEvaluation[]
  artifacts: CampaignLedgerArtifact[]
  decisions: CampaignLedgerDecision[]
  evidence: Record<string, string[]>
}

export interface CampaignAutoResearchState {
  campaign_status: CampaignStatus
  next_action:
    | 'prepare_campaign'
    | 'start_campaign'
    | 'submit_baseline'
    | 'wait_for_result'
    | 'propose_candidate'
    | 'stop'
    | 'blocked'
  reason_code: string
  ready_for_next_proposal: boolean
  baseline_verified: boolean
  pending_proposal_id?: string | null
  best_proposal_id?: string | null
  best_study_id?: string | null
  best_metric?: number | null
  attempts_used: number
  proposals_used: number
  budget_used: number
  budget_remaining: number
  latest_decision?: string | null
}

export interface CampaignAutoResearchOutcome {
  result: {
    result_id: string
    proposal_id: string
    study_id: string
    role: 'baseline' | 'candidate'
    provenance: 'real' | 'simulated'
    outcome: 'completed' | 'crashed'
    metric_name: string
    metric_value?: number | null
    actual_cost: number
    attempt_ids: string[]
    evidence_references: string[]
    recorded_at: string
  }
  decision: {
    proposal_id: string
    decision: 'baseline' | 'keep' | 'discard' | 'crash' | 'ineligible'
    reason_code: string
    eligible_for_best: boolean
    improvement?: number | null
    decided_at: string
  }
}

export interface CampaignCodeLineage {
  lineage_id: string
  campaign_id: string
  proposal_id: string
  mutation_kind: 'trainer' | 'gym' | 'reward' | 'evaluator'
  source_repository_profile_id: string
  state: 'required' | 'prepared' | 'captured'
  base_commit?: string | null
  branch_name?: string | null
  commit_sha?: string | null
  changed_paths: string[]
  patch_sha256?: string | null
  created_at: string
  updated_at: string
  captured_at?: string | null
}

export interface CampaignAutoResearchDiagnostics {
  schema_version: 'autoresearch_diagnostics.v1'
  workspace_id: string
  campaign_id: string
  primary_metric: string
  metric_direction: 'maximize' | 'minimize'
  low_signal: boolean
  signals: Array<{
    code: string
    severity: 'info' | 'warning' | 'critical'
    summary: string
    evidence_references: string[]
  }>
  checkpoint_comparisons: Array<{
    evaluation_result_id: string
    run_id: string
    role: 'checkpoint' | 'final'
    step?: number | null
    metric_name: string
    metric_value: number
    improvement_from_previous?: number | null
    improvement_from_baseline?: number | null
  }>
  error_slices: Array<{
    slice_path: string
    direction: 'maximize' | 'minimize' | 'unknown'
    candidate_value: number
    baseline_value?: number | null
    improvement?: number | null
    status: 'observed' | 'improved' | 'regressed' | 'unchanged'
    evidence_references: string[]
  }>
  ranked_hypotheses: Array<{
    hypothesis_id: string
    rank: number
    action_kind: 'diagnostic' | 'evaluation' | 'candidate'
    changed_variable: string
    hypothesis: string
    rationale: string
    expected_outcome: string
    falsification_criterion: string
    evidence_references: string[]
    eligible_for_submission: boolean
  }>
}

export interface CampaignAutoResearchProjection {
  spec: {
    primary_metric: string
    metric_direction: 'maximize' | 'minimize'
    stop_rules: Record<string, unknown>
  }
  state: CampaignAutoResearchState
  proposals: Array<Record<string, unknown>>
  outcomes: CampaignAutoResearchOutcome[]
  code_lineages?: CampaignCodeLineage[]
  diagnostics?: CampaignAutoResearchDiagnostics
}

export interface CampaignLedgerProjection {
  schema_version: string
  workspace_id: string
  campaign_id: string
  generated_at: string
  linked: boolean
  projects: CampaignLedgerProject[]
  autoresearch?: CampaignAutoResearchProjection | null
}

export interface CampaignDetailState {
  snapshot: CampaignControlRoomSnapshotV1 | null
  freshness: ControlRoomFreshness
  lastVerifiedAt: string | null
  reconciliation: CampaignReconciliationState
  pages: {
    events: CampaignEventItem[]
    artifacts: CampaignArtifact[]
    eventCursor: number
    artifactCursor: string | null
    eventsHasMore: boolean
    artifactsHasMore: boolean
    eventsLoading: boolean
    artifactsLoading: boolean
    eventsLoaded: boolean
    artifactsLoaded: boolean
    eventsError: string | null
    artifactsError: string | null
  }
  campaign: CampaignRecord
  studies: CampaignStudy[]
  proposals: CampaignProposalRecord[]
  evidence: CampaignEvidence | null
  attempts: CampaignAttempt[]
  artifacts: CampaignArtifact[]
  comparisons: CampaignComparison[]
  ledger: CampaignLedgerProjection | null
  events: CampaignEventItem[]
  lossByAttempt: Record<string, CampaignMetricValue[]>
  nextCursor: number
  loading: boolean
  error: string | null
}

interface CampaignWorkspaceState {
  campaigns: CampaignRecord[]
  controller: CampaignControllerStatus | null
  selectedCampaignId: string | null
  details: Record<string, CampaignDetailState>
  freshness: ControlRoomFreshness
  loadGeneration: number
  loading: boolean
  error: string | null
  liveRefs: number
  connected: boolean
  subscribed: boolean
  connectionGeneration: number
}

interface CampaignState {
  workspaces: Record<string, CampaignWorkspaceState>
  load: (workspaceId: string, preferredCampaignId?: string) => Promise<void>
  refresh: (workspaceId: string, campaignId: string) => Promise<void>
  loadEventsPage: (workspaceId: string, campaignId: string) => Promise<void>
  loadArtifactsPage: (workspaceId: string, campaignId: string) => Promise<void>
  loadAttemptMetrics: (workspaceId: string, campaignId: string, attemptId: string) => Promise<void>
  loadLegacyDetail: (workspaceId: string, campaignId: string) => Promise<void>
  refreshController: (workspaceId: string) => Promise<void>
  select: (workspaceId: string, campaignId: string) => Promise<void>
  transition: (
    workspaceId: string,
    campaignId: string,
    action: 'start' | 'pause' | 'resume' | 'cancel' | 'conclude',
    expectedVersion: number,
    reason?: string,
  ) => Promise<boolean>
  handleHint: (hint: CampaignHintV1) => Promise<void>
  handleConnection: (connected: boolean, connectionGeneration?: number) => Promise<void>
  handleSubscription: (
    workspaceId: string,
    subscribed: boolean,
    connectionGeneration: number,
  ) => Promise<void>
  startWorkspaceLive: (workspaceId: string) => void
  stopWorkspaceLive: (workspaceId: string) => void
  reconcileCampaign: (workspaceId: string, campaignId: string, reason: string) => Promise<void>
  reconcileWorkspaceCampaignList: (workspaceId: string) => Promise<void>
}

const emptyWorkspace = (): CampaignWorkspaceState => ({
  campaigns: [],
  controller: null,
  selectedCampaignId: null,
  details: {},
  freshness: 'reconciling',
  loadGeneration: 0,
  loading: false,
  error: null,
  liveRefs: 0,
  connected: false,
  subscribed: false,
  connectionGeneration: 0,
})

const controllerRefreshInFlight = new Map<string, Promise<void>>()
const controllerRefreshAt = new Map<string, number>()
const eventPageInFlight = new Map<string, Promise<void>>()
const artifactPageInFlight = new Map<string, Promise<void>>()
const attemptMetricsInFlight = new Map<string, Promise<void>>()
const snapshotInFlight = new Map<string, Promise<void>>()
const hintReconciliationInFlight = new Map<string, {
  controller: AbortController
  promise: Promise<void>
}>()
const pendingHintTargets = new Map<string, CampaignHintV1>()
const CONTROLLER_REFRESH_DEDUPE_MS = 2_000
const MAX_HINT_RETRIES = 5
const MAX_PENDING_HINT_TARGETS = 256
const CAMPAIGN_SAFETY_RECONCILE_MS = 18_000
const campaignSafetyReconcileEntries = new Map<string, {
  refs: number
  timer: ReturnType<typeof setInterval>
}>()

function campaignKey(workspaceId: string, campaignId: string): string {
  return `${workspaceId}\u0000${campaignId}`
}

export function retainCampaignSafetyReconcile(
  workspaceId: string,
  campaignId: string,
): () => void {
  const key = campaignKey(workspaceId, campaignId)
  const existing = campaignSafetyReconcileEntries.get(key)
  if (existing) {
    existing.refs += 1
  } else {
    const timer = setInterval(() => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void useCampaignStore.getState().reconcileCampaign(
          workspaceId,
          campaignId,
          'safety-reconcile',
        )
      }
    }, CAMPAIGN_SAFETY_RECONCILE_MS)
    campaignSafetyReconcileEntries.set(key, { refs: 1, timer })
  }
  let released = false
  return () => {
    if (released) return
    released = true
    const entry = campaignSafetyReconcileEntries.get(key)
    if (!entry) return
    entry.refs -= 1
    if (entry.refs <= 0) {
      clearInterval(entry.timer)
      campaignSafetyReconcileEntries.delete(key)
    }
  }
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'))
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, milliseconds)
    const abort = () => {
      clearTimeout(timer)
      reject(new DOMException('Aborted', 'AbortError'))
    }
    signal.addEventListener('abort', abort, { once: true })
  })
}

function errorOf<T>(response: ApiResponse<T>, fallback: string): string {
  return response.error || fallback
}

function mergeEvents(
  existing: CampaignEventItem[],
  incoming: CampaignEventItem[],
): CampaignEventItem[] {
  const byId = new Map(existing.map((item) => [item.event.event_id, item]))
  for (const item of incoming) byId.set(item.event.event_id, item)
  return Array.from(byId.values()).sort((a, b) => a.cursor - b.cursor).slice(-500)
}

function emptyDetail(campaign: CampaignRecord): CampaignDetailState {
  const reconciliation = createCampaignFreshness({
    eventCursor: 0,
    aggregateVersion: 0,
    hasSnapshot: false,
    connected: true,
  })
  return {
    snapshot: null,
    freshness: 'reconciling',
    lastVerifiedAt: null,
    reconciliation,
    pages: {
      events: [],
      artifacts: [],
      eventCursor: 0,
      artifactCursor: null,
      eventsHasMore: true,
      artifactsHasMore: true,
      eventsLoading: false,
      artifactsLoading: false,
      eventsLoaded: false,
      artifactsLoaded: false,
      eventsError: null,
      artifactsError: null,
    },
    campaign,
    studies: [],
    proposals: [],
    evidence: null,
    attempts: [],
    artifacts: [],
    comparisons: [],
    ledger: null,
    events: [],
    lossByAttempt: {},
    nextCursor: 0,
    loading: false,
    error: null,
  }
}

function isOfflineFailure(response: ApiResponse<unknown>): boolean {
  if ([
    'campaign_desktop_bridge_required',
    'campaign_backend_unavailable',
    'campaign_backend_root_unavailable',
    'campaign_backend_configuration_invalid',
  ].includes(response.code || '')) return true
  const message = (response.error || '').toLowerCase()
  return ['network', 'offline', 'bridge unavailable', 'failed to fetch'].some((value) =>
    message.includes(value)
  )
}

export const useCampaignStore = create<CampaignState>((set, get) => ({
  workspaces: {},

  load: async (workspaceId, preferredCampaignId) => {
    const generation = (get().workspaces[workspaceId]?.loadGeneration || 0) + 1
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: {
          ...(state.workspaces[workspaceId] || emptyWorkspace()),
            freshness: 'reconciling',
          loadGeneration: generation,
          loading: true,
          error: null,
        },
      },
    }))
    const response = await campaignApi.list(workspaceId)
    if (!response.ok || !response.data) {
      set((state) => {
        const workspace = state.workspaces[workspaceId] || emptyWorkspace()
        if (workspace.loadGeneration !== generation) return state
        const offline = isOfflineFailure(response)
        const error = errorOf(response, 'Unable to load campaigns')
        const details: Record<string, CampaignDetailState> = {}
        for (const [id, detail] of Object.entries(workspace.details)) {
          const freshness: ControlRoomFreshness = offline
            ? 'offline'
            : detail.snapshot ? 'stale' : 'error'
          details[id] = {
            ...detail,
            freshness,
            error,
          }
        }
        return {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              details,
              freshness: offline ? 'offline' : workspace.campaigns.length ? 'stale' : 'error',
              loading: false,
              error,
            },
          },
        }
      })
      return
    }
    const responseData = response.data
    const campaigns = [...responseData.campaigns].sort((a, b) =>
      b.updated_at.localeCompare(a.updated_at)
    )
    let selectedCampaignId: string | null = null
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      if (workspace.loadGeneration !== generation) return state
      const selected = campaigns.find((item) => item.campaign_id === preferredCampaignId)
        || campaigns.find((item) => item.campaign_id === workspace.selectedCampaignId)
        || campaigns[0]
        || null
      selectedCampaignId = selected?.campaign_id || null
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            campaigns,
            controller: responseData.controller,
            selectedCampaignId,
            freshness: workspace.subscribed ? 'reconciling' : 'stale',
            loading: false,
            error: null,
          },
        },
      }
    })
    if (selectedCampaignId) await get().refresh(workspaceId, selectedCampaignId)
  },

  refresh: async (workspaceId, campaignId) => {
    const key = campaignKey(workspaceId, campaignId)
    const existing = snapshotInFlight.get(key)
    if (existing) return existing
    const workspaceBefore = get().workspaces[workspaceId] || emptyWorkspace()
    const campaign = workspaceBefore.campaigns.find((item) => item.campaign_id === campaignId)
    if (!campaign) return
    const previous = workspaceBefore.details[campaignId] || emptyDetail(campaign)
    const authorityBefore = workspaceBefore.subscribed && !previous.reconciliation.subscribed
      ? acknowledgeCampaignSubscription(previous.reconciliation, workspaceBefore.connectionGeneration)
      : previous.reconciliation
    const reconciliation = beginCampaignReconciliation(authorityBefore)
    const generation = reconciliation.generation
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            details: {
              ...workspace.details,
              [campaignId]: {
                ...previous,
                freshness: reconciliation.freshness,
                reconciliation,
                loading: true,
                error: null,
              },
            },
          },
        },
      }
    })
    const request = (async () => {
      const response = await campaignApi.snapshot(workspaceId, campaignId)
      if (!response.ok || !response.data) {
        set((state) => {
          const workspace = state.workspaces[workspaceId] || emptyWorkspace()
          const detail = workspace.details[campaignId]
          if (!detail || detail.reconciliation.generation !== generation) return state
          const offline = isOfflineFailure(response)
          const next = failCampaignReconciliation(detail.reconciliation, {
            generation,
            connected: !offline,
            exhausted: !offline && !detail.snapshot,
            errorCode: response.code || 'campaign_snapshot_failed',
          })
          return {
            workspaces: {
              ...state.workspaces,
              [workspaceId]: {
                ...workspace,
                details: {
                  ...workspace.details,
                  [campaignId]: {
                    ...detail,
                    reconciliation: next,
                    loading: false,
                    freshness: next.freshness,
                    error: errorOf(response, 'Unable to load AutoResearch campaign state'),
                  },
                },
              },
            },
          }
        })
        return
      }
      set((state) => {
        const workspace = state.workspaces[workspaceId] || emptyWorkspace()
        const detail = workspace.details[campaignId]
        if (!detail || detail.reconciliation.generation !== generation) return state
        const incoming = validateCampaignSnapshot(response.data, workspaceId, campaignId)
        if (!incoming) {
          const next = markCampaignError(detail.reconciliation, 'campaign_snapshot_invalid')
          return {
            workspaces: {
              ...state.workspaces,
              [workspaceId]: {
                ...workspace,
                details: {
                  ...workspace.details,
                  [campaignId]: {
                    ...detail,
                    reconciliation: next,
                    freshness: next.freshness,
                    loading: false,
                    error: 'Rejected invalid campaign snapshot schema or identity.',
                  },
                },
              },
            },
          }
        }
        const semanticKey = snapshotSemanticIdentity(incoming)
        const next = finishCampaignReconciliation(detail.reconciliation, {
          generation,
          eventCursor: incoming.latest_event_cursor,
          aggregateVersion: incoming.aggregate_version,
          semanticKey,
          verifiedAt: incoming.snapshot_at,
        })
        const accepted = next.appliedCursor === incoming.latest_event_cursor
          && next.appliedVersion === incoming.aggregate_version
          && next.errorCode !== 'campaign_snapshot_regressed'
        if (!accepted) {
          return {
            workspaces: {
              ...state.workspaces,
              [workspaceId]: {
                ...workspace,
                details: {
                  ...workspace.details,
                  [campaignId]: {
                    ...detail,
                    reconciliation: next,
                    freshness: next.freshness,
                    loading: false,
                    error: 'Rejected regressive campaign state; retained the last verified state.',
                  },
                },
              },
            },
          }
        }
        const semanticallyUnchanged = Boolean(
          detail.snapshot && detail.reconciliation.semanticKey === semanticKey
        )
        const snapshot = semanticallyUnchanged ? detail.snapshot! : incoming
        const updatedCampaign: CampaignRecord = semanticallyUnchanged
          ? detail.campaign
          : {
              ...detail.campaign,
              title: snapshot.campaign.title,
              objective: snapshot.campaign.objective,
              kind: snapshot.campaign.kind,
              status: snapshot.campaign.status,
              manifest_revision: snapshot.manifest_revision,
              active_study_id: snapshot.campaign.active_study_id,
              active_action_id: snapshot.campaign.active_action_id,
              champion_ref: snapshot.campaign.champion_ref,
              stop_reason: snapshot.campaign.stop_reason,
              version: snapshot.aggregate_version,
              updated_at: snapshot.snapshot_at,
            }
        return {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              campaigns: semanticallyUnchanged
                ? workspace.campaigns
                : workspace.campaigns.map((item) =>
                    item.campaign_id === campaignId ? updatedCampaign : item
                  ),
              details: {
                ...workspace.details,
                [campaignId]: {
                  ...detail,
                  campaign: updatedCampaign,
                  snapshot,
                  reconciliation: next,
                  freshness: next.freshness,
                  lastVerifiedAt: next.lastVerifiedAt,
                  loading: false,
                  error: next.errorCode ? 'Awaiting durable campaign state for the latest notification target.' : null,
                },
              },
            },
          },
        }
      })
    })().finally(() => {
      if (snapshotInFlight.get(key) === request) snapshotInFlight.delete(key)
    })
    snapshotInFlight.set(key, request)
    return request
  },

  reconcileCampaign: async (workspaceId, campaignId, _reason) => {
    await get().refresh(workspaceId, campaignId)
  },

  reconcileWorkspaceCampaignList: async (workspaceId) => {
    await get().load(workspaceId, get().workspaces[workspaceId]?.selectedCampaignId || undefined)
  },

  loadEventsPage: (workspaceId, campaignId) => {
    const requestKey = `${workspaceId}\u0000${campaignId}`
    const existing = eventPageInFlight.get(requestKey)
    if (existing) return existing
    const detail = get().workspaces[workspaceId]?.details[campaignId]
    if (!detail || (detail.pages.eventsLoaded && !detail.pages.eventsHasMore)) return Promise.resolve()
    const requestedCursor = detail.pages.eventCursor
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      const current = workspace.details[campaignId]
      if (!current) return state
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            details: {
              ...workspace.details,
              [campaignId]: {
                ...current,
                pages: { ...current.pages, eventsLoading: true, eventsError: null },
              },
            },
          },
        },
      }
    })
    const request = (async () => {
      const response = await campaignApi.events(workspaceId, campaignId, requestedCursor, 100)
      if (!response.ok || !response.data) {
        if (response.status === 410 && response.code === 'campaign_event_cursor_expired') {
          const detailRecord = response.details && typeof response.details === 'object'
            ? response.details as Record<string, unknown>
            : {}
          const resumeCursor = Number.isSafeInteger(detailRecord.resume_cursor)
            && Number(detailRecord.resume_cursor) >= 0
            ? Number(detailRecord.resume_cursor)
            : 0
          set((state) => {
            const workspace = state.workspaces[workspaceId] || emptyWorkspace()
            const current = workspace.details[campaignId]
            if (!current) return state
            const reconciliation = expireCampaignCursor(current.reconciliation)
            return {
              workspaces: {
                ...state.workspaces,
                [workspaceId]: {
                  ...workspace,
                  details: {
                    ...workspace.details,
                    [campaignId]: {
                      ...current,
                      freshness: reconciliation.freshness,
                      reconciliation,
                      events: [],
                      nextCursor: resumeCursor,
                      pages: {
                        ...current.pages,
                        events: [],
                        eventCursor: resumeCursor,
                        eventsHasMore: true,
                        eventsLoading: false,
                        eventsLoaded: false,
                        eventsError: null,
                      },
                    },
                  },
                },
              },
            }
          })
          await get().reconcileCampaign(workspaceId, campaignId, 'event-cursor-expired')
          return
        }
        set((state) => {
          const workspace = state.workspaces[workspaceId] || emptyWorkspace()
          const current = workspace.details[campaignId]
          if (!current) return state
          return {
            workspaces: {
              ...state.workspaces,
              [workspaceId]: {
                ...workspace,
                details: {
                  ...workspace.details,
                  [campaignId]: {
                    ...current,
                    pages: {
                      ...current.pages,
                      eventsLoading: false,
                      eventsError: errorOf(response, 'Unable to load campaign events'),
                    },
                  },
                },
              },
            },
          }
        })
        return
      }
      const currentEvents = get().workspaces[workspaceId]?.details[campaignId]?.pages.events || []
      const acceptedIds = new Set(currentEvents.map((item) => item.event.event_id))
      for (const item of response.data.items) {
        if (acceptedIds.has(item.event.event_id)) continue
        acceptedIds.add(item.event.event_id)
        const fields = toCampaignActivityFields(item.event)
        if (fields) useActivityStore.getState().addEvent(item.event.event_type, fields)
      }
      set((state) => {
        const workspace = state.workspaces[workspaceId] || emptyWorkspace()
        const current = workspace.details[campaignId]
        if (!current) return state
        const events = mergeEvents(current.pages.events, response.data!.items)
        const nextCursor = response.data!.next_cursor
        return {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              details: {
                ...workspace.details,
                [campaignId]: {
                  ...current,
                  events,
                  nextCursor,
                  pages: {
                    ...current.pages,
                    events,
                    eventCursor: nextCursor,
                    eventsHasMore: response.data!.items.length > 0 && nextCursor > requestedCursor,
                    eventsLoading: false,
                    eventsLoaded: true,
                    eventsError: null,
                  },
                },
              },
            },
          },
        }
      })
    })().finally(() => eventPageInFlight.delete(requestKey))
    eventPageInFlight.set(requestKey, request)
    return request
  },

  loadArtifactsPage: (workspaceId, campaignId) => {
    const requestKey = `${workspaceId}\u0000${campaignId}`
    const existing = artifactPageInFlight.get(requestKey)
    if (existing) return existing
    const detail = get().workspaces[workspaceId]?.details[campaignId]
    if (!detail || (detail.pages.artifactsLoaded && !detail.pages.artifactsHasMore)) {
      return Promise.resolve()
    }
    const requestedCursor = detail.pages.artifactCursor
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      const current = workspace.details[campaignId]
      if (!current) return state
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            details: {
              ...workspace.details,
              [campaignId]: {
                ...current,
                pages: { ...current.pages, artifactsLoading: true, artifactsError: null },
              },
            },
          },
        },
      }
    })
    const request = (async () => {
      const response = await campaignApi.artifacts(
        workspaceId,
        campaignId,
        requestedCursor,
        50,
      )
      if (!response.ok || !response.data) {
        set((state) => {
          const workspace = state.workspaces[workspaceId] || emptyWorkspace()
          const current = workspace.details[campaignId]
          if (!current) return state
          return {
            workspaces: {
              ...state.workspaces,
              [workspaceId]: {
                ...workspace,
                details: {
                  ...workspace.details,
                  [campaignId]: {
                    ...current,
                    pages: {
                      ...current.pages,
                      artifactsLoading: false,
                      artifactsError: errorOf(response, 'Unable to load campaign artifacts'),
                    },
                  },
                },
              },
            },
          }
        })
        return
      }
      const incoming = response.data.artifacts
        .map((artifact) => toCampaignPublicArtifact(artifact))
        .filter((artifact): artifact is CampaignArtifact => artifact !== null)
      set((state) => {
        const workspace = state.workspaces[workspaceId] || emptyWorkspace()
        const current = workspace.details[campaignId]
        if (!current) return state
        const byId = new Map(current.pages.artifacts.map((item) => [item.artifact_id, item]))
        for (const artifact of incoming) byId.set(artifact.artifact_id, artifact)
        const artifacts = Array.from(byId.values()).slice(-500)
        return {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              details: {
                ...workspace.details,
                [campaignId]: {
                  ...current,
                  artifacts,
                  pages: {
                    ...current.pages,
                    artifacts,
                    artifactCursor: response.data!.next_cursor,
                    artifactsHasMore: response.data!.has_more,
                    artifactsLoading: false,
                    artifactsLoaded: true,
                    artifactsError: null,
                  },
                },
              },
            },
          },
        }
      })
    })().finally(() => artifactPageInFlight.delete(requestKey))
    artifactPageInFlight.set(requestKey, request)
    return request
  },

  loadAttemptMetrics: (workspaceId, campaignId, attemptId) => {
    const requestKey = `${workspaceId}\u0000${campaignId}\u0000${attemptId}`
    const existing = attemptMetricsInFlight.get(requestKey)
    if (existing) return existing
    const detail = get().workspaces[workspaceId]?.details[campaignId]
    if (!detail || detail.snapshot?.active_work?.attempt_id !== attemptId) return Promise.resolve()
    const requestGeneration = detail.reconciliation.generation
    const currentValues = detail.lossByAttempt[attemptId] || []
    const afterStep = currentValues.reduce((latest, point) => Math.max(latest, point.step), -1)
    const request = (async () => {
      const response = await campaignApi.metrics(
        workspaceId,
        campaignId,
        attemptId,
        'training_metrics.jsonl',
        'loss',
        afterStep,
        2000,
      )
      if (!response.ok || !response.data) return
      const incoming = response.data.values.filter((point) => (
        Number.isSafeInteger(point.step)
        && point.step >= 0
        && point.source === 'training_metrics.jsonl'
        && Number.isFinite(point.value)
        && typeof point.observed_at === 'string'
      ))
      set((state) => {
        const workspace = state.workspaces[workspaceId]
        const current = workspace?.details[campaignId]
        if (
          !workspace
          || !current
          || current.reconciliation.generation !== requestGeneration
          || current.snapshot?.active_work?.attempt_id !== attemptId
        ) return state
        const byPoint = new Map(
          (current.lossByAttempt[attemptId] || []).map((point) => [`${point.source}:${point.step}`, point])
        )
        for (const point of incoming) byPoint.set(`${point.source}:${point.step}`, point)
        const values = Array.from(byPoint.values())
          .sort((left, right) => left.step - right.step || left.source.localeCompare(right.source))
          .slice(-500)
        return {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              details: {
                ...workspace.details,
                [campaignId]: {
                  ...current,
                  lossByAttempt: { ...current.lossByAttempt, [attemptId]: values },
                },
              },
            },
          },
        }
      })
    })().finally(() => {
      if (attemptMetricsInFlight.get(requestKey) === request) attemptMetricsInFlight.delete(requestKey)
    })
    attemptMetricsInFlight.set(requestKey, request)
    return request
  },

  loadLegacyDetail: async (workspaceId, campaignId) => {
    const current = get().workspaces[workspaceId]?.details[campaignId]
    if (!current) return
    const requestGeneration = current.reconciliation.generation
    const snapshotIdentity = current.snapshot
      ? `${current.snapshot.aggregate_version}:${current.snapshot.latest_event_cursor}`
      : null
    const [campaignResponse, studiesResponse, proposalsResponse, evidenceResponse, attemptsResponse, comparisonsResponse, ledgerResponse] = await Promise.all([
      campaignApi.get(workspaceId, campaignId),
      campaignApi.studies(workspaceId, campaignId),
      campaignApi.proposals(workspaceId, campaignId),
      campaignApi.evidence(workspaceId, campaignId),
      campaignApi.attempts(workspaceId, campaignId),
      campaignApi.comparisons(workspaceId, campaignId),
      campaignApi.ledger(workspaceId, campaignId),
    ])
    if (!campaignResponse.data || !studiesResponse.data || !proposalsResponse.data || !evidenceResponse.data || !attemptsResponse.data || !comparisonsResponse.data || !ledgerResponse.data) return
    const attempts = attemptsResponse.data.attempts
    const metricResponses = await Promise.all(attempts.filter((attempt) =>
      ['smoke_training', 'full_training'].includes(attempt.stage)
    ).slice(-4).map(async (attempt) => ({
      attemptId: attempt.attempt_id,
      response: await campaignApi.metrics(workspaceId, campaignId, attempt.attempt_id, 'training_metrics.jsonl', 'loss'),
    })))
    const incomingLossByAttempt: Record<string, CampaignMetricValue[]> = {}
    for (const item of metricResponses) {
      if (item.response.ok && item.response.data) incomingLossByAttempt[item.attemptId] = item.response.data.values
    }
    let accepted = false
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      const detail = workspace.details[campaignId]
      if (!detail) return state
      const currentSnapshotIdentity = detail.snapshot
        ? `${detail.snapshot.aggregate_version}:${detail.snapshot.latest_event_cursor}`
        : null
      if (
        detail.reconciliation.generation !== requestGeneration
        || currentSnapshotIdentity !== snapshotIdentity
      ) return state
      accepted = true
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            details: {
              ...workspace.details,
              [campaignId]: {
                ...detail,
                campaign: detail.snapshot ? detail.campaign : campaignResponse.data!,
                studies: studiesResponse.data!.studies,
                proposals: proposalsResponse.data!.proposals,
                evidence: evidenceResponse.data!,
                attempts,
                comparisons: comparisonsResponse.data!.comparisons,
                ledger: ledgerResponse.data!,
                lossByAttempt: { ...detail.lossByAttempt, ...incomingLossByAttempt },
              },
            },
          },
        },
      }
    })
    if (!accepted) return
    await Promise.all([
      get().loadEventsPage(workspaceId, campaignId),
      get().loadArtifactsPage(workspaceId, campaignId),
    ])
  },

  refreshController: async (workspaceId) => {
    const existing = controllerRefreshInFlight.get(workspaceId)
    if (existing) return existing
    const now = Date.now()
    if (now - (controllerRefreshAt.get(workspaceId) || 0) < CONTROLLER_REFRESH_DEDUPE_MS) {
      return
    }
    const request = (async () => {
      const response = await campaignApi.list(workspaceId)
      if (response.ok && response.data) {
        set((state) => ({
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...(state.workspaces[workspaceId] || emptyWorkspace()),
              controller: response.data!.controller,
            },
          },
        }))
        controllerRefreshAt.set(workspaceId, Date.now())
      }
    })().finally(() => controllerRefreshInFlight.delete(workspaceId))
    controllerRefreshInFlight.set(workspaceId, request)
    return request
  },

  select: async (workspaceId, campaignId) => {
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: {
          ...(state.workspaces[workspaceId] || emptyWorkspace()),
          selectedCampaignId: campaignId,
          loadGeneration: (state.workspaces[workspaceId]?.loadGeneration || 0) + 1,
        },
      },
    }))
    await get().refresh(workspaceId, campaignId)
  },

  transition: async (workspaceId, campaignId, action, expectedVersion, reason) => {
    const authority = get().workspaces[workspaceId]?.details[campaignId]
    if (!authority || authority.freshness !== 'live') return false
    const response = await campaignApi.transition(
      workspaceId, campaignId, action, expectedVersion, reason
    )
    if (!response.ok) {
      set((state) => {
        const workspace = state.workspaces[workspaceId] || emptyWorkspace()
        const detail = workspace.details[campaignId]
        return !detail ? state : {
          workspaces: {
            ...state.workspaces,
            [workspaceId]: {
              ...workspace,
              details: {
                ...workspace.details,
                [campaignId]: {
                  ...detail,
                  error: errorOf(response, `Unable to ${action} campaign`),
                },
              },
            },
          },
        }
      })
      // A rejected mutation can itself prove that the renderer's last verified
      // authority is no longer current (readiness drift, lease loss, or CAS
      // conflict). Reconcile the compact projection before another attempt;
      // the error response never becomes campaign state.
      await get().reconcileCampaign(workspaceId, campaignId, 'mutation-rejected')
      return false
    }
    await get().reconcileCampaign(workspaceId, campaignId, 'mutation-ack')
    return true
  },

  handleHint: async (hint) => {
    let workspace = get().workspaces[hint.workspace_id]
    if (!workspace || !workspace.subscribed) return
    const key = campaignKey(hint.workspace_id, hint.campaign_id)
    let detail = workspace.details[hint.campaign_id]
    const knownCampaign = workspace.campaigns.find((item) => item.campaign_id === hint.campaign_id)
    if (!detail && hint.event_type === 'campaign:created') {
      pendingHintTargets.set(key, hint)
      while (pendingHintTargets.size > MAX_PENDING_HINT_TARGETS) {
        const oldest = pendingHintTargets.keys().next().value as string | undefined
        if (!oldest) break
        pendingHintTargets.delete(oldest)
      }
      await get().reconcileWorkspaceCampaignList(hint.workspace_id)
      workspace = get().workspaces[hint.workspace_id]
      const campaign = workspace?.campaigns.find((item) => item.campaign_id === hint.campaign_id)
      if (!workspace?.subscribed || !campaign) return
      detail = workspace.details[hint.campaign_id] || emptyDetail(campaign)
      const acknowledged = acknowledgeCampaignSubscription(
        detail.reconciliation,
        workspace.connectionGeneration,
      )
      detail = { ...detail, reconciliation: acknowledged, freshness: acknowledged.freshness }
      set((state) => ({
        workspaces: {
          ...state.workspaces,
          [hint.workspace_id]: {
            ...state.workspaces[hint.workspace_id]!,
            details: {
              ...state.workspaces[hint.workspace_id]!.details,
              [hint.campaign_id]: detail!,
            },
          },
        },
      }))
      pendingHintTargets.delete(key)
    } else if (!detail) {
      // Unknown non-created campaigns are not materialized from transport data.
      if (!knownCampaign) return
      const acknowledged = acknowledgeCampaignSubscription(
        emptyDetail(knownCampaign).reconciliation,
        workspace.connectionGeneration,
      )
      detail = { ...emptyDetail(knownCampaign), reconciliation: acknowledged, freshness: acknowledged.freshness }
      set((state) => ({
        workspaces: {
          ...state.workspaces,
          [hint.workspace_id]: {
            ...state.workspaces[hint.workspace_id]!,
            details: { ...state.workspaces[hint.workspace_id]!.details, [hint.campaign_id]: detail! },
          },
        },
      }))
    }
    if (!detail) return
    const decision = applyCampaignHint(detail.reconciliation, hint)
    if (!decision.reconcile) return
    set((state) => {
      const currentWorkspace = state.workspaces[hint.workspace_id]
      const current = currentWorkspace?.details[hint.campaign_id]
      if (!currentWorkspace || !current) return state
      const next = applyCampaignHint(current.reconciliation, hint)
      if (!next.reconcile) return state
      return {
        workspaces: {
          ...state.workspaces,
          [hint.workspace_id]: {
            ...currentWorkspace,
            details: {
              ...currentWorkspace.details,
              [hint.campaign_id]: {
                ...current,
                reconciliation: next.state,
                freshness: next.state.freshness,
                error: null,
              },
            },
          },
        },
      }
    })

    const existing = hintReconciliationInFlight.get(key)
    if (existing) return existing.promise
    const controller = new AbortController()
    const request = (async () => {
      await abortableDelay(50, controller.signal)
      for (let attempt = 0; attempt <= MAX_HINT_RETRIES; attempt += 1) {
        if (controller.signal.aborted) return
        await get().reconcileCampaign(hint.workspace_id, hint.campaign_id, 'hint')
        const current = get().workspaces[hint.workspace_id]?.details[hint.campaign_id]
        if (!current) return
        const authority = current.reconciliation
        const satisfied = authority.subscribed
          && authority.appliedCursor >= authority.targetCursor
          && authority.appliedVersion >= authority.targetVersion
        if (satisfied) return
        if (['campaign_snapshot_invalid', 'campaign_subscription_denied'].includes(authority.errorCode || '')) return
        if (attempt >= MAX_HINT_RETRIES) {
          set((state) => {
            const currentWorkspace = state.workspaces[hint.workspace_id]
            const latest = currentWorkspace?.details[hint.campaign_id]
            if (!currentWorkspace || !latest) return state
            const next = markCampaignError(latest.reconciliation, 'campaign_snapshot_retry_exhausted')
            return {
              workspaces: {
                ...state.workspaces,
                [hint.workspace_id]: {
                  ...currentWorkspace,
                  details: {
                    ...currentWorkspace.details,
                    [hint.campaign_id]: {
                      ...latest,
                      reconciliation: next,
                      freshness: next.freshness,
                      error: 'Awaiting durable campaign state for the latest notification target.',
                    },
                  },
                },
              },
            }
          })
          return
        }
        await abortableDelay(reconciliationRetryDelay(attempt), controller.signal)
      }
    })().catch((error) => {
      if (!(error instanceof DOMException && error.name === 'AbortError')) throw error
    }).finally(() => {
      if (hintReconciliationInFlight.get(key)?.promise === request) {
        hintReconciliationInFlight.delete(key)
      }
    })
    hintReconciliationInFlight.set(key, { controller, promise: request })
    return request
  },

  handleConnection: async (connected, suppliedGeneration) => {
    const nextGeneration = suppliedGeneration ?? Math.max(
      1,
      ...Object.values(get().workspaces).map((workspace) => workspace.connectionGeneration + 1),
    )
    if (!connected) {
      for (const run of hintReconciliationInFlight.values()) run.controller.abort()
      hintReconciliationInFlight.clear()
      // REST cannot be physically aborted through the Electron bridge, but its
      // old generation is invalidated and must not block a fresh reconnect read.
      snapshotInFlight.clear()
      pendingHintTargets.clear()
      set((state) => ({
        workspaces: Object.fromEntries(Object.entries(state.workspaces).map(([workspaceId, workspace]) => [
          workspaceId,
          {
            ...workspace,
            freshness: 'offline',
            connected: false,
            subscribed: false,
            connectionGeneration: nextGeneration,
            loading: false,
            details: Object.fromEntries(Object.entries(workspace.details).map(([campaignId, detail]) => [
              campaignId,
              {
                ...detail,
                freshness: 'offline',
                reconciliation: markCampaignDisconnected(detail.reconciliation, nextGeneration),
                loading: false,
              },
            ])),
          },
        ])),
      }))
      return
    }
    const workspaces = Object.entries(get().workspaces)
    set((state) => ({
      workspaces: Object.fromEntries(Object.entries(state.workspaces).map(([workspaceId, workspace]) => [
        workspaceId,
        {
          ...workspace,
          freshness: 'reconciling',
          connected: true,
          subscribed: false,
          connectionGeneration: nextGeneration,
          details: Object.fromEntries(Object.entries(workspace.details).map(([campaignId, detail]) => [
            campaignId,
            {
              ...detail,
              freshness: 'reconciling',
              reconciliation: markCampaignReconnected(detail.reconciliation, nextGeneration),
            },
          ])),
        },
      ])),
    }))
    // Socket-open is transport only. Campaign authority waits for per-workspace ack.
    void workspaces
  },

  handleSubscription: async (workspaceId, subscribed, connectionGeneration) => {
    const workspaceBefore = get().workspaces[workspaceId]
    if (!workspaceBefore || connectionGeneration < workspaceBefore.connectionGeneration) return
    set((state) => {
      const workspace = state.workspaces[workspaceId]
      if (!workspace || connectionGeneration < workspace.connectionGeneration) return state
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            connected: true,
            subscribed,
            connectionGeneration,
            freshness: subscribed ? 'reconciling' : 'error',
            details: Object.fromEntries(Object.entries(workspace.details).map(([campaignId, detail]) => {
              const reconciliation = subscribed
                ? acknowledgeCampaignSubscription(detail.reconciliation, connectionGeneration)
                : markCampaignError(detail.reconciliation, 'campaign_subscription_denied')
              return [campaignId, {
                ...detail,
                freshness: reconciliation.freshness,
                reconciliation,
                error: subscribed ? null : 'Live campaign subscription was denied.',
              }]
            })),
          },
        },
      }
    })
    if (subscribed) {
      await Promise.all(Object.keys(get().workspaces[workspaceId]?.details || {}).map((campaignId) =>
        get().reconcileCampaign(workspaceId, campaignId, 'subscription-ack')
      ))
    }
  },

  startWorkspaceLive: (workspaceId) => {
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: {
          ...(state.workspaces[workspaceId] || emptyWorkspace()),
          liveRefs: (state.workspaces[workspaceId]?.liveRefs || 0) + 1,
        },
      },
    }))
  },

  stopWorkspaceLive: (workspaceId) => {
    set((state) => {
      const workspace = state.workspaces[workspaceId]
      if (!workspace) return state
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: { ...workspace, liveRefs: Math.max(0, workspace.liveRefs - 1) },
        },
      }
    })
  },
}))
