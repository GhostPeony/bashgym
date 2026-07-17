import { create } from 'zustand'
import {
  toCampaignActivityFields,
  toCampaignPublicArtifact,
  type CampaignArtifact,
  type CampaignEventItem,
} from '../campaignVisibility'
import { campaignApi, type ApiResponse } from '../services/api'
import { useActivityStore } from './activityStore'

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
  executor: Record<string, unknown>
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
  loading: boolean
  error: string | null
}

interface CampaignState {
  workspaces: Record<string, CampaignWorkspaceState>
  load: (workspaceId: string, preferredCampaignId?: string) => Promise<void>
  refresh: (workspaceId: string, campaignId: string) => Promise<void>
  refreshController: (workspaceId: string) => Promise<void>
  select: (workspaceId: string, campaignId: string) => Promise<void>
  transition: (
    workspaceId: string,
    campaignId: string,
    action: 'start' | 'pause' | 'resume' | 'cancel',
    expectedVersion: number,
    reason?: string,
  ) => Promise<boolean>
}

const emptyWorkspace = (): CampaignWorkspaceState => ({
  campaigns: [],
  controller: null,
  selectedCampaignId: null,
  details: {},
  loading: false,
  error: null,
})

const controllerRefreshInFlight = new Map<string, Promise<void>>()
const controllerRefreshAt = new Map<string, number>()
const CONTROLLER_REFRESH_DEDUPE_MS = 2_000

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

export const useCampaignStore = create<CampaignState>((set, get) => ({
  workspaces: {},

  load: async (workspaceId, preferredCampaignId) => {
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: {
          ...(state.workspaces[workspaceId] || emptyWorkspace()),
          loading: true,
          error: null,
        },
      },
    }))
    const response = await campaignApi.list(workspaceId)
    if (!response.ok || !response.data) {
      set((state) => ({
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...(state.workspaces[workspaceId] || emptyWorkspace()),
            loading: false,
            error: errorOf(response, 'Unable to load campaigns'),
          },
        },
      }))
      return
    }
    const responseData = response.data
    const campaigns = [...responseData.campaigns].sort((a, b) =>
      b.updated_at.localeCompare(a.updated_at)
    )
    const prior = get().workspaces[workspaceId]?.selectedCampaignId
    const selected = campaigns.find((item) => item.campaign_id === preferredCampaignId)
      || campaigns.find((item) => item.campaign_id === prior)
      || campaigns[0]
      || null
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: {
          ...(state.workspaces[workspaceId] || emptyWorkspace()),
          campaigns,
          controller: responseData.controller,
          selectedCampaignId: selected?.campaign_id || null,
          loading: false,
          error: null,
        },
      },
    }))
    if (selected) await get().refresh(workspaceId, selected.campaign_id)
  },

  refresh: async (workspaceId, campaignId) => {
    const previous = get().workspaces[workspaceId]?.details[campaignId]
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      const campaign = workspace.campaigns.find((item) => item.campaign_id === campaignId)
      if (!campaign) return state
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            details: {
              ...workspace.details,
              [campaignId]: previous ? {
                ...previous,
                loading: true,
                error: null,
              } : {
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
                loading: true,
                error: null,
              },
            },
          },
        },
      }
    })
    const afterCursor = previous?.nextCursor || 0
    const [campaignResponse, studiesResponse, proposalsResponse, evidenceResponse, attemptsResponse, artifactsResponse, comparisonsResponse, ledgerResponse, eventsResponse] = await Promise.all([
      campaignApi.get(workspaceId, campaignId),
      campaignApi.studies(workspaceId, campaignId),
      campaignApi.proposals(workspaceId, campaignId),
      campaignApi.evidence(workspaceId, campaignId),
      campaignApi.attempts(workspaceId, campaignId),
      campaignApi.artifacts(workspaceId, campaignId),
      campaignApi.comparisons(workspaceId, campaignId),
      campaignApi.ledger(workspaceId, campaignId),
      campaignApi.events(workspaceId, campaignId, afterCursor, 200),
    ])
    const failed = [campaignResponse, studiesResponse, proposalsResponse, evidenceResponse, attemptsResponse, artifactsResponse, comparisonsResponse, ledgerResponse, eventsResponse]
      .find((response) => !response.ok)
    if (
      failed
      || !campaignResponse.data
      || !studiesResponse.data
      || !proposalsResponse.data
      || !evidenceResponse.data
      || !attemptsResponse.data
      || !artifactsResponse.data
      || !comparisonsResponse.data
      || !ledgerResponse.data
      || !eventsResponse.data
    ) {
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
                  loading: false,
                  error: failed?.error || 'Unable to reconcile campaign evidence',
                },
              },
            },
          },
        }
      })
      return
    }
    const attempts = attemptsResponse.data.attempts
    const trainingAttempts = attempts.filter((attempt) =>
      ['smoke_training', 'full_training'].includes(attempt.stage)
    ).slice(-4)
    const metricResponses = await Promise.all(trainingAttempts.map(async (attempt) => ({
      attemptId: attempt.attempt_id,
      response: await campaignApi.metrics(
        workspaceId,
        campaignId,
        attempt.attempt_id,
        'training_metrics.jsonl',
        'loss',
        -1,
        2000,
      ),
    })))
    const lossByAttempt = { ...(previous?.lossByAttempt || {}) }
    for (const item of metricResponses) {
      if (item.response.ok && item.response.data) {
        lossByAttempt[item.attemptId] = item.response.data.values
      }
    }
    const events = mergeEvents(previous?.events || [], eventsResponse.data.items)
    for (const item of eventsResponse.data.items) {
      const fields = toCampaignActivityFields(item.event)
      if (fields) useActivityStore.getState().addEvent(item.event.event_type, fields)
    }
    const artifacts = artifactsResponse.data.artifacts
      .map((artifact) => toCampaignPublicArtifact(artifact))
      .filter((artifact): artifact is CampaignArtifact => artifact !== null)
    const detail: CampaignDetailState = {
      campaign: campaignResponse.data,
      studies: studiesResponse.data.studies,
      proposals: proposalsResponse.data.proposals,
      evidence: evidenceResponse.data,
      attempts,
      artifacts,
      comparisons: comparisonsResponse.data.comparisons,
      ledger: ledgerResponse.data,
      events,
      lossByAttempt,
      nextCursor: eventsResponse.data.next_cursor,
      loading: false,
      error: null,
    }
    set((state) => {
      const workspace = state.workspaces[workspaceId] || emptyWorkspace()
      return {
        workspaces: {
          ...state.workspaces,
          [workspaceId]: {
            ...workspace,
            campaigns: workspace.campaigns.map((item) =>
              item.campaign_id === campaignId ? campaignResponse.data! : item
            ),
            details: { ...workspace.details, [campaignId]: detail },
          },
        },
      }
    })
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
        },
      },
    }))
    await get().refresh(workspaceId, campaignId)
  },

  transition: async (workspaceId, campaignId, action, expectedVersion, reason) => {
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
      return false
    }
    await get().refresh(workspaceId, campaignId)
    return true
  },
}))
