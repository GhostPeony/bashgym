import type { CampaignControlRoomSnapshotV1 } from '../services/api'

export const CAMPAIGN_HINT_SCHEMA_VERSION = 'campaign_hint.v1' as const

export type CampaignFreshness = 'reconciling' | 'live' | 'stale' | 'offline' | 'error'

export interface CampaignHintV1 {
  schema_version: typeof CAMPAIGN_HINT_SCHEMA_VERSION
  workspace_id: string
  campaign_id: string
  event_cursor: number
  aggregate_version: number
  event_type: string
  correlation_id: string
  emitted_at: string
}

/** The sole authority state for one workspace/campaign projection. */
export interface CampaignReconciliationState {
  freshness: CampaignFreshness
  generation: number
  connectionGeneration: number
  subscribed: boolean
  appliedCursor: number
  appliedVersion: number
  targetCursor: number
  targetVersion: number
  semanticKey: string | null
  inFlightGeneration: number | null
  retryCount: number
  lastHintAt: string | null
  lastVerifiedAt: string | null
  errorCode: string | null
}

export interface CampaignHintDecision {
  state: CampaignReconciliationState
  reconcile: boolean
  reason: 'duplicate' | 'out_of_order' | 'hint' | 'crossed'
}

export function createCampaignFreshness(input: {
  eventCursor: number
  aggregateVersion: number
  hasSnapshot: boolean
  connected: boolean
}): CampaignReconciliationState {
  return {
    freshness: input.connected
      ? input.hasSnapshot ? 'stale' : 'reconciling'
      : 'offline',
    generation: 0,
    connectionGeneration: 0,
    subscribed: false,
    appliedCursor: input.eventCursor,
    appliedVersion: input.aggregateVersion,
    targetCursor: input.eventCursor,
    targetVersion: input.aggregateVersion,
    semanticKey: null,
    inFlightGeneration: null,
    retryCount: 0,
    lastHintAt: null,
    lastVerifiedAt: null,
    errorCode: null,
  }
}

export function beginCampaignReconciliation(
  state: CampaignReconciliationState,
  options: { targetCursor?: number; targetVersion?: number } = {},
): CampaignReconciliationState {
  const generation = state.generation + 1
  return {
    ...state,
    freshness: state.freshness === 'offline' ? 'offline' : 'reconciling',
    generation,
    targetCursor: Math.max(state.targetCursor, options.targetCursor ?? 0),
    targetVersion: Math.max(state.targetVersion, options.targetVersion ?? 0),
    inFlightGeneration: generation,
    errorCode: null,
  }
}

export function acknowledgeCampaignSubscription(
  state: CampaignReconciliationState,
  connectionGeneration: number,
): CampaignReconciliationState {
  if (connectionGeneration < state.connectionGeneration) return state
  return {
    ...state,
    freshness: 'reconciling',
    connectionGeneration,
    subscribed: true,
    generation: state.generation + 1,
    inFlightGeneration: null,
    retryCount: 0,
    errorCode: null,
  }
}

export function applyCampaignHint(
  state: CampaignReconciliationState,
  hint: CampaignHintV1,
): CampaignHintDecision {
  if (hint.event_cursor <= state.targetCursor && hint.aggregate_version <= state.targetVersion) {
    const exact = hint.event_cursor === state.targetCursor
      && hint.aggregate_version === state.targetVersion
    return { state, reconcile: false, reason: exact ? 'duplicate' : 'out_of_order' }
  }
  const crossed = (
    hint.event_cursor > state.targetCursor && hint.aggregate_version < state.targetVersion
  ) || (
    hint.aggregate_version > state.targetVersion && hint.event_cursor < state.targetCursor
  )
  return {
    state: {
      ...state,
      freshness: state.inFlightGeneration === null
        ? state.freshness === 'offline' ? 'offline' : 'stale'
        : state.freshness === 'offline' ? 'offline' : 'reconciling',
      targetCursor: Math.max(state.targetCursor, hint.event_cursor),
      targetVersion: Math.max(state.targetVersion, hint.aggregate_version),
      lastHintAt: hint.emitted_at,
      errorCode: null,
    },
    reconcile: true,
    reason: crossed ? 'crossed' : 'hint',
  }
}

export function expireCampaignCursor(state: CampaignReconciliationState): CampaignReconciliationState {
  return {
    ...state,
    freshness: state.freshness === 'offline' ? 'offline' : 'stale',
    errorCode: 'campaign_event_cursor_expired',
  }
}

export function finishCampaignReconciliation(
  state: CampaignReconciliationState,
  result: {
    generation: number
    eventCursor: number
    aggregateVersion: number
    semanticKey?: string
    verifiedAt?: string
  },
): CampaignReconciliationState {
  if (result.generation !== state.generation || state.inFlightGeneration !== result.generation) {
    return state
  }
  if (result.eventCursor < state.appliedCursor || result.aggregateVersion < state.appliedVersion) {
    return {
      ...state,
      freshness: state.freshness === 'offline' ? 'offline' : 'stale',
      inFlightGeneration: null,
      retryCount: state.retryCount + 1,
      errorCode: 'campaign_snapshot_regressed',
    }
  }
  const covered = result.eventCursor >= state.targetCursor
    && result.aggregateVersion >= state.targetVersion
  return {
    ...state,
    freshness: covered && state.subscribed ? 'live' : 'stale',
    appliedCursor: result.eventCursor,
    appliedVersion: result.aggregateVersion,
    targetCursor: Math.max(state.targetCursor, result.eventCursor),
    targetVersion: Math.max(state.targetVersion, result.aggregateVersion),
    semanticKey: result.semanticKey ?? state.semanticKey,
    inFlightGeneration: null,
    retryCount: covered ? 0 : state.retryCount + 1,
    lastVerifiedAt: result.verifiedAt ?? state.lastVerifiedAt,
    errorCode: covered ? null : 'campaign_snapshot_behind_target',
  }
}

export function failCampaignReconciliation(
  state: CampaignReconciliationState,
  result: {
    generation: number
    connected: boolean
    exhausted?: boolean
    errorCode?: string
  },
): CampaignReconciliationState {
  if (result.generation !== state.generation || state.inFlightGeneration !== result.generation) {
    return state
  }
  return {
    ...state,
    subscribed: result.connected ? state.subscribed : false,
    freshness: !result.connected ? 'offline' : result.exhausted ? 'error' : 'stale',
    inFlightGeneration: null,
    retryCount: state.retryCount + 1,
    errorCode: result.errorCode ?? 'campaign_snapshot_failed',
  }
}

export function markCampaignError(
  state: CampaignReconciliationState,
  errorCode: string,
): CampaignReconciliationState {
  return {
    ...state,
    freshness: 'error',
    inFlightGeneration: null,
    errorCode,
  }
}

export function markCampaignDisconnected(
  state: CampaignReconciliationState,
  connectionGeneration = state.connectionGeneration + 1,
): CampaignReconciliationState {
  if (connectionGeneration < state.connectionGeneration) return state
  return {
    ...state,
    freshness: 'offline',
    generation: state.generation + 1,
    connectionGeneration,
    subscribed: false,
    inFlightGeneration: null,
    errorCode: 'campaign_socket_offline',
  }
}

/** Socket-open is not subscription authority; only the ack transition sets subscribed. */
export function markCampaignReconnected(
  state: CampaignReconciliationState,
  connectionGeneration = state.connectionGeneration + 1,
): CampaignReconciliationState {
  if (connectionGeneration < state.connectionGeneration) return state
  return {
    ...state,
    freshness: 'reconciling',
    generation: state.generation + 1,
    connectionGeneration,
    subscribed: false,
    inFlightGeneration: null,
    retryCount: 0,
    errorCode: null,
  }
}

export function reconciliationRetryDelay(attempt: number): number {
  return Math.min(250 * (2 ** Math.max(0, attempt)), 8_000)
}

const HINT_KEYS = [
  'aggregate_version',
  'campaign_id',
  'correlation_id',
  'emitted_at',
  'event_cursor',
  'event_type',
  'schema_version',
  'workspace_id',
] as const

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function hasExactKeys(value: Record<string, unknown>, expectedKeys: readonly string[]): boolean {
  return Object.keys(value).sort().join('\u0000') === [...expectedKeys].sort().join('\u0000')
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isNullableString(value: unknown): value is string | null {
  return value === null || isString(value)
}

function isSafeInteger(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= minimum
}

function isFiniteNumber(value: unknown, minimum?: number, maximum?: number): value is number {
  return typeof value === 'number'
    && Number.isFinite(value)
    && (minimum === undefined || value >= minimum)
    && (maximum === undefined || value <= maximum)
}

function isStringArray(value: unknown, maxLength: number): value is string[] {
  return Array.isArray(value)
    && value.length <= maxLength
    && value.every(isString)
}

function isIdentifierArray(value: unknown, maxLength: number): value is string[] {
  return Array.isArray(value)
    && value.length <= maxLength
    && value.every(isCampaignIdentifier)
}

function isNullableIdentifier(value: unknown): value is string | null {
  return value === null || isCampaignIdentifier(value)
}

function isHexDigest(value: unknown): value is string {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
}

function isDigest(value: unknown): value is string | null {
  return value === null || isHexDigest(value)
}

const CAMPAIGN_STATUSES = new Set([
  'draft', 'validating', 'ready', 'active', 'paused', 'awaiting_authority',
  'cancelling', 'completed', 'exhausted', 'failed', 'cancelled',
])

const CAMPAIGN_CAPABILITIES = new Set([
  'campaign.read',
  'campaign.create_from_template',
  'campaign.create',
  'campaign.revise',
  'campaign.start',
  'campaign.pause',
  'campaign.resume',
  'campaign.cancel',
  'campaign.complete',
  'study.propose',
  'study.retry',
  'study.abandon',
  'data.use_approved',
  'data.approve_external',
  'data.build',
  'compute.smoke',
  'compute.train_within_budget',
  'compute.amend_budget',
  'compute.manage_resident_services',
  'compute.force_stop',
  'eval.development',
  'eval.protected_acquire',
  'eval.protected_execute',
  'experiment.ledger_write',
  'experiment.code_mutate',
  'promotion.decide',
  'promotion.override',
  'artifact.publish_hf',
  'handoff.memexai_prepare',
])

function isDecisionBlocker(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'code', 'summary', 'evidence_ids', 'secondary_codes',
  ])) return false
  return value.schema_version === 'decision_blocker.v1'
    && isCampaignIdentifier(value.code)
    && isString(value.summary)
    && isIdentifierArray(value.evidence_ids, 64)
    && isIdentifierArray(value.secondary_codes, 64)
    && new Set(value.evidence_ids as string[]).size === (value.evidence_ids as string[]).length
    && new Set(value.secondary_codes as string[]).size === (value.secondary_codes as string[]).length
    && !(value.secondary_codes as string[]).includes(value.code as string)
}

function isCampaignSummary(value: unknown, campaignId: string, aggregateVersion: number): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'campaign_id', 'title', 'objective', 'kind', 'status',
    'aggregate_version', 'manifest_revision', 'active_study_id', 'active_action_id',
    'champion_ref', 'stop_reason',
  ])) return false
  return value.schema_version === 'campaign_summary.v1'
    && value.campaign_id === campaignId
    && isCampaignIdentifier(value.campaign_id)
    && isString(value.title)
    && isString(value.objective)
    && (value.kind === 'embedding_retrieval' || value.kind === 'general')
    && typeof value.status === 'string'
    && CAMPAIGN_STATUSES.has(value.status)
    && value.aggregate_version === aggregateVersion
    && isSafeInteger(value.manifest_revision, 1)
    && isNullableIdentifier(value.active_study_id)
    && isNullableIdentifier(value.active_action_id)
    && isNullableString(value.champion_ref)
    && isNullableString(value.stop_reason)
}

function isControllerObservation(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'controller_observation_version', 'state', 'observed_at',
    'heartbeat_age_seconds', 'lease_expires_at', 'controller_instance_id', 'safe_guidance',
  ])) return false
  return value.schema_version === 'controller_observation.v1'
    && isSafeInteger(value.controller_observation_version)
    && ['online', 'stale', 'offline'].includes(String(value.state))
    && isCanonicalUtcTimestamp(value.observed_at)
    && (value.heartbeat_age_seconds === null || isFiniteNumber(value.heartbeat_age_seconds, 0))
    && (value.lease_expires_at === null || isCanonicalUtcTimestamp(value.lease_expires_at))
    && isNullableIdentifier(value.controller_instance_id)
    && isNullableString(value.safe_guidance)
}

function isReadinessSummary(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'materializable', 'launch_ready', 'checked_at',
    'activation_receipt_digest', 'doctor_receipt_digest', 'blocking_codes',
  ])) return false
  return value.schema_version === 'readiness_summary.v1'
    && typeof value.materializable === 'boolean'
    && typeof value.launch_ready === 'boolean'
    && isCanonicalUtcTimestamp(value.checked_at)
    && isDigest(value.activation_receipt_digest)
    && isDigest(value.doctor_receipt_digest)
    && isIdentifierArray(value.blocking_codes, 64)
    && new Set(value.blocking_codes as string[]).size === (value.blocking_codes as string[]).length
}

function isSafeBinding(value: unknown): boolean {
  if (value === null) return true
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'binding_id', 'immutable_digest', 'display_label',
  ])) return false
  return value.schema_version === 'safe_binding_identity.v1'
    && isCampaignIdentifier(value.binding_id)
    && isDigest(value.immutable_digest)
    && isString(value.display_label)
}

function isBindingSummary(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'model', 'data', 'evaluator', 'source', 'compute',
  ])) return false
  return value.schema_version === 'binding_summary.v1'
    && ['model', 'data', 'evaluator', 'source', 'compute'].every((key) => isSafeBinding(value[key]))
}

function isJourneyPhase(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'phase_id', 'state', 'execution_owner', 'attention_owner',
    'primary_blocker', 'evidence_count', 'next_action_ids',
  ])) return false
  return value.schema_version === 'journey_phase_summary.v1'
    && ['setup', 'baseline', 'experiments', 'human_review', 'decision'].includes(String(value.phase_id))
    && ['not_started', 'ready', 'active', 'blocked', 'complete', 'failed', 'skipped'].includes(String(value.state))
    && ['bashgym', 'human', 'none'].includes(String(value.execution_owner))
    && ['bashgym', 'agent', 'human', 'none'].includes(String(value.attention_owner))
    && (value.primary_blocker === null || isDecisionBlocker(value.primary_blocker))
    && isSafeInteger(value.evidence_count)
    && isIdentifierArray(value.next_action_ids, 64)
    && new Set(value.next_action_ids as string[]).size === (value.next_action_ids as string[]).length
}

function isProcessIdentity(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'run_id', 'compute_profile_id', 'state',
  ])) return false
  return value.schema_version === 'opaque_process_identity.v1'
    && isCampaignIdentifier(value.run_id)
    && isCampaignIdentifier(value.compute_profile_id)
    && ['launching', 'running', 'completed', 'failed', 'cancelled', 'unknown'].includes(String(value.state))
}

function isActiveWork(value: unknown): boolean {
  if (value === null) return true
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'study_id', 'proposal_id', 'action_id', 'attempt_id', 'stage',
    'hypothesis_summary', 'primary_variable_summary', 'controlled_variable_summary',
    'progress_fraction', 'eta_seconds', 'executor_type', 'process_identity',
  ])) return false
  const stages = ['data_build', 'contract_evaluation', 'smoke_training', 'full_training', 'development_evaluation', 'comparison', 'recipe_lock', 'protected_evaluation', 'promotion']
  return value.schema_version === 'active_work_summary.v1'
    && ['study_id', 'proposal_id', 'action_id', 'attempt_id'].every((key) => isNullableIdentifier(value[key]))
    && (value.stage === null || stages.includes(String(value.stage)))
    && isNullableString(value.hypothesis_summary)
    && isNullableString(value.primary_variable_summary)
    && isStringArray(value.controlled_variable_summary, 64)
    && (value.progress_fraction === null || isFiniteNumber(value.progress_fraction, 0, 1))
    && (value.eta_seconds === null || isFiniteNumber(value.eta_seconds, 0))
    && (value.executor_type === null || ['fake', 'ssh_remote', 'development_evaluation'].includes(String(value.executor_type)))
    && (value.process_identity === null || isProcessIdentity(value.process_identity))
}

function isCandidate(value: unknown): boolean {
  if (value === null) return true
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'candidate_ref', 'source_attempt_ids', 'source_artifact_ids',
    'latest_comparable_evaluation_id', 'comparison_verdict', 'gate_state',
  ])) return false
  return value.schema_version === 'candidate_summary.v1'
    && isString(value.candidate_ref)
    && isIdentifierArray(value.source_attempt_ids, 100)
    && isIdentifierArray(value.source_artifact_ids, 100)
    && isNullableIdentifier(value.latest_comparable_evaluation_id)
    && (value.comparison_verdict === null || ['passed', 'failed', 'insufficient_evidence'].includes(String(value.comparison_verdict)))
    && ['not_evaluated', 'blocked', 'passed', 'failed', 'promoted'].includes(String(value.gate_state))
}

function isMetric(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'metric_id', 'display_name', 'unit', 'direction', 'target',
    'tolerance', 'evaluator_revision', 'sample_count', 'uncertainty_method', 'comparability_key',
  ])) return false
  return value.schema_version === 'metric_descriptor.v1'
    && isCampaignIdentifier(value.metric_id)
    && isString(value.display_name)
    && isNullableString(value.unit)
    && ['maximize', 'minimize', 'target'].includes(String(value.direction))
    && (value.target === null || isFiniteNumber(value.target))
    && (value.tolerance === null || isFiniteNumber(value.tolerance))
    && isNullableString(value.evaluator_revision)
    && (value.sample_count === null || isSafeInteger(value.sample_count))
    && isNullableString(value.uncertainty_method)
    && isHexDigest(value.comparability_key)
}

function isBudget(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['schema_version', 'resources', 'blocked'])) return false
  if (value.schema_version !== 'budget_summary.v1' || typeof value.blocked !== 'boolean' || !Array.isArray(value.resources) || value.resources.length > 64) return false
  return value.resources.every((resource) => {
    if (!isPlainRecord(resource) || !hasExactKeys(resource, [
      'schema_version', 'unit', 'limit', 'reserved', 'settled', 'remaining', 'blocked', 'blocker_code',
    ])) return false
    return resource.schema_version === 'budget_resource_summary.v1'
      && isCampaignIdentifier(resource.unit)
      && ['limit', 'reserved', 'settled', 'remaining'].every((key) => isFiniteNumber(resource[key]))
      && typeof resource.blocked === 'boolean'
      && (resource.blocker_code === null || isCampaignIdentifier(resource.blocker_code))
  })
}

function isHumanWorkSummary(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['schema_version', 'blocking_count', 'open_count', 'newest'])) return false
  if (value.schema_version !== 'human_work_summary.v1' || !isSafeInteger(value.blocking_count) || !isSafeInteger(value.open_count) || !Array.isArray(value.newest) || value.newest.length > 10) return false
  return value.newest.every((item) => {
    if (!isPlainRecord(item) || !hasExactKeys(item, [
      'schema_version', 'work_item_id', 'kind', 'status', 'blocking_scope', 'assigned_actor_id',
      'required_count', 'completed_count', 'due_at',
    ])) return false
    return item.schema_version === 'human_work_item_summary.v1'
      && isCampaignIdentifier(item.work_item_id)
      && ['blinded_sample_evaluation', 'promotion_decision'].includes(String(item.kind))
      && ['open', 'claimed', 'accepted', 'rejected', 'revision_requested', 'abstained', 'cancelled', 'expired'].includes(String(item.status))
      && isCampaignIdentifier(item.blocking_scope)
      && isNullableIdentifier(item.assigned_actor_id)
      && isSafeInteger(item.required_count)
      && isSafeInteger(item.completed_count)
      && (item.due_at === null || isCanonicalUtcTimestamp(item.due_at))
  })
}

function isAgentSummary(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'session_id', 'actor_id', 'origin_id', 'bundle_id', 'capability_revision',
    'expires_at', 'liveness', 'last_cursor', 'last_request_id',
  ])) return false
  return value.schema_version === 'attached_agent_summary.v1'
    && ['session_id', 'actor_id', 'origin_id', 'bundle_id'].every((key) => isCampaignIdentifier(value[key]))
    && isSafeInteger(value.capability_revision, 1)
    && isCanonicalUtcTimestamp(value.expires_at)
    && ['active', 'disconnected', 'expired', 'revoked'].includes(String(value.liveness))
    && isSafeInteger(value.last_cursor)
    && isNullableIdentifier(value.last_request_id)
}

function isCollectionCursor(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['schema_version', 'count', 'next_cursor', 'has_more'])) return false
  return value.schema_version === 'collection_cursor.v1'
    && isSafeInteger(value.count)
    && isNullableString(value.next_cursor)
    && typeof value.has_more === 'boolean'
    && value.has_more === (value.next_cursor !== null)
}

function isCollectionSummary(value: unknown): boolean {
  const keys = ['events', 'proposals', 'studies', 'attempts', 'artifacts', 'comparisons', 'human_work']
  if (!isPlainRecord(value) || !hasExactKeys(value, ['schema_version', ...keys])) return false
  return value.schema_version === 'collection_summary.v1' && keys.every((key) => isCollectionCursor(value[key]))
}

function isDecisionSurface(value: unknown): boolean {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'schema_version', 'execution_owner', 'attention_owner', 'blocker', 'next_actions',
    'recovery_actions', 'promotion_eligible',
  ])) return false
  if (value.schema_version !== 'decision_surface.v1'
    || !['bashgym', 'human', 'none'].includes(String(value.execution_owner))
    || !['bashgym', 'agent', 'human', 'none'].includes(String(value.attention_owner))
    || (value.blocker !== null && !isDecisionBlocker(value.blocker))
    || !Array.isArray(value.next_actions)
    || value.next_actions.length > 64
    || !isIdentifierArray(value.recovery_actions, 16)
    || typeof value.promotion_eligible !== 'boolean') return false
  if (new Set(value.recovery_actions as string[]).size !== (value.recovery_actions as string[]).length) return false
  if (!value.next_actions.every((action) => {
    if (!isPlainRecord(action) || !hasExactKeys(action, [
      'schema_version', 'action', 'capability', 'freshness_class', 'requires_human_work',
    ])) return false
    return action.schema_version === 'decision_action.v1'
      && isCampaignIdentifier(action.action)
      && typeof action.capability === 'string'
      && CAMPAIGN_CAPABILITIES.has(action.capability)
      && ['read', 'lifecycle', 'privileged'].includes(String(action.freshness_class))
      && typeof action.requires_human_work === 'boolean'
  })) return false
  return new Set((value.next_actions as Record<string, unknown>[]).map((action) => action.action)).size === value.next_actions.length
}

export function isCampaignIdentifier(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(value)
}

function isEventType(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(value)
}

function isCanonicalUtcTimestamp(value: unknown): value is string {
  if (typeof value !== 'string') return false
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{6}))?Z$/.exec(value)
  if (!match) return false
  const parsed = new Date(value)
  if (!Number.isFinite(parsed.getTime())) return false
  return parsed.getUTCFullYear() === Number(match[1])
    && parsed.getUTCMonth() + 1 === Number(match[2])
    && parsed.getUTCDate() === Number(match[3])
    && parsed.getUTCHours() === Number(match[4])
    && parsed.getUTCMinutes() === Number(match[5])
    && parsed.getUTCSeconds() === Number(match[6])
}

export function parseCampaignHint(value: unknown): CampaignHintV1 | null {
  if (!isPlainRecord(value)) return null
  if (Object.keys(value).sort().join('\u0000') !== [...HINT_KEYS].sort().join('\u0000')) return null
  if (value.schema_version !== CAMPAIGN_HINT_SCHEMA_VERSION) return null
  if (!isCampaignIdentifier(value.workspace_id) || !isCampaignIdentifier(value.campaign_id)) return null
  if (!Number.isSafeInteger(value.event_cursor) || Number(value.event_cursor) < 1) return null
  if (!Number.isSafeInteger(value.aggregate_version) || Number(value.aggregate_version) < 1) return null
  if (!isEventType(value.event_type) || !isCampaignIdentifier(value.correlation_id)) return null
  if (!isCanonicalUtcTimestamp(value.emitted_at)) return null
  return value as unknown as CampaignHintV1
}

export function validateCampaignSnapshot(
  value: unknown,
  workspaceId: string,
  campaignId: string,
): CampaignControlRoomSnapshotV1 | null {
  const topLevelKeys = [
    'schema_version', 'workspace_id', 'campaign_id', 'aggregate_version', 'manifest_revision',
    'authorization_revision', 'snapshot_at', 'latest_event_cursor', 'campaign', 'controller',
    'readiness', 'bindings', 'journey', 'active_work', 'champion', 'candidate', 'metrics',
    'budget', 'human_work', 'agents', 'collections', 'decision_surface',
  ]
  if (!isPlainRecord(value) || !hasExactKeys(value, topLevelKeys) || value.schema_version !== 'control_room_snapshot.v1') return null
  if (value.workspace_id !== workspaceId || value.campaign_id !== campaignId) return null
  if (!isCampaignIdentifier(value.workspace_id) || !isCampaignIdentifier(value.campaign_id)) return null
  if (!Number.isSafeInteger(value.aggregate_version) || Number(value.aggregate_version) < 1) return null
  if (!isSafeInteger(value.manifest_revision, 1) || !isSafeInteger(value.authorization_revision, 1)) return null
  if (!isCanonicalUtcTimestamp(value.snapshot_at)) return null
  if (!Number.isSafeInteger(value.latest_event_cursor) || Number(value.latest_event_cursor) < 0) return null
  if (!isCampaignSummary(value.campaign, campaignId, Number(value.aggregate_version))) return null
  if (!isControllerObservation(value.controller) || !isReadinessSummary(value.readiness)) return null
  if (!isBindingSummary(value.bindings)) return null
  if (!Array.isArray(value.journey) || value.journey.length !== 5 || !value.journey.every(isJourneyPhase)) return null
  if (value.journey.map((phase) => (phase as Record<string, unknown>).phase_id).join('|') !== 'setup|baseline|experiments|human_review|decision') return null
  if ((value.campaign as Record<string, unknown>).manifest_revision !== value.manifest_revision) return null
  if (!isActiveWork(value.active_work) || !isCandidate(value.champion) || !isCandidate(value.candidate)) return null
  if (!Array.isArray(value.metrics) || value.metrics.length > 64 || !value.metrics.every(isMetric)) return null
  if (!isBudget(value.budget) || !isHumanWorkSummary(value.human_work)) return null
  if (!Array.isArray(value.agents) || value.agents.length > 32 || !value.agents.every(isAgentSummary)) return null
  if (!isCollectionSummary(value.collections) || !isDecisionSurface(value.decision_surface)) return null
  return value as unknown as CampaignControlRoomSnapshotV1
}

export function snapshotSemanticIdentity(snapshot: {
  workspace_id: string
  campaign_id: string
  snapshot_at?: unknown
  aggregate_version: number
  latest_event_cursor: number
  authorization_revision: number
  controller: { controller_observation_version: number; observed_at?: unknown }
  readiness: {
    schema_version: string
    checked_at?: unknown
    materializable: boolean
    launch_ready: boolean
    activation_receipt_digest: string | null
    doctor_receipt_digest: string | null
    blocking_codes: string[]
  }
}): string {
  return JSON.stringify({
    workspace_id: snapshot.workspace_id,
    campaign_id: snapshot.campaign_id,
    aggregate_version: snapshot.aggregate_version,
    latest_event_cursor: snapshot.latest_event_cursor,
    controller_observation_version: snapshot.controller.controller_observation_version,
    readiness: {
      schema_version: snapshot.readiness.schema_version,
      materializable: snapshot.readiness.materializable,
      launch_ready: snapshot.readiness.launch_ready,
      activation_receipt_digest: snapshot.readiness.activation_receipt_digest,
      doctor_receipt_digest: snapshot.readiness.doctor_receipt_digest,
      blocking_codes: snapshot.readiness.blocking_codes,
    },
    authorization_revision: snapshot.authorization_revision,
  })
}
