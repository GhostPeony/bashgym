export const CAMPAIGN_RECOVERY_SCHEMA_VERSION = 'campaign_recovery.v1' as const

export type RecoveryFreshness = 'live' | 'reconciling' | 'stale' | 'offline' | 'error'
export type RecoveryAction = 'resume' | 'repair' | 'takeover'
export type RecoveryDecision = 'resumable' | 'blocked' | 'read_only' | 'degraded'
export type RecoveryExecutionStatus = 'accepted' | 'executing' | 'completed' | 'failed' | 'blocked'
export type RecoveryOutcomeCode =
  'campaign_resumed' | 'attempt_reconciled' | 'needs_operator' | 'authority_changed'
export type RecoveryBlockerCode =
  | 'missing_binding'
  | 'private_compute_inaccessible'
  | 'checkpoint_missing'
  | 'artifact_missing'
  | 'schema_incompatible'
  | 'revision_incompatible'
  | 'active_foreign_lease'
  | 'eligibility_unverified'
  | 'no_permitted_recovery_action'
  | 'recovery_consumer_unavailable'
  | 'recovery_request_in_flight'
  | 'recovery_execution_needs_operator'

type DoctorCheck =
  | 'binding_verified'
  | 'checkpoint_available'
  | 'compute_reachable'
  | 'schema_compatible'
  | 'lease_observed'

export interface RecoveryScopePublicV1 {
  workspace_id: string
  campaign_id: string
  campaign_revision: number
  event_cursor: number
  aggregate_version: number
}

export interface RecoveryReceiptPublicV1 extends RecoveryScopePublicV1 {
  receipt_id: string
  kind: 'doctor' | 'eligibility' | 'recovery'
  emitted_at: string
  digest: string
}

export interface RecoveryEligibilityPublicV1 extends RecoveryScopePublicV1 {
  receipt_id: string
  receipt_digest: string
  decision: 'eligible' | 'blocked'
  allowed_actions: RecoveryAction[]
  doctor_evidence_id: string
  doctor_evidence_digest: string
  issued_at: string
  expires_at: string
}

export interface RecoveryExecutionPublicV1 {
  schema_version: 'campaign_recovery_execution.v1'
  workspace_id: string
  campaign_id: string
  action: 'resume' | 'repair'
  status: RecoveryExecutionStatus
  outcome_code: RecoveryOutcomeCode | null
  attempt_id: string | null
}

export interface RecoveryExecutionConsumerPublicV1 {
  supported: boolean
  ready: boolean
  reason_code: 'ready' | 'controller_unowned' | 'controller_lease_expired' | 'foreign_controller'
}

export interface CampaignRecoveryPublicV1 {
  schema_version: typeof CAMPAIGN_RECOVERY_SCHEMA_VERSION
  workspace_id: string
  campaign_id: string
  installation: { installation_id: string; lineage_mode: 'clone' | 'fork' }
  bindings: {
    model_id: string | null
    model_display_label: string | null
    data_id: string | null
    evaluator_id: string | null
    compute_id: string | null
  }
  lineage: {
    checkpoint_id: string
    artifact_id: string
    parent_campaign_id: string | null
    campaign_revision: number
    event_cursor: number
    aggregate_version: number
    schema_version: string
  }
  controller: {
    state: 'owned' | 'unowned' | 'foreign_lease' | 'expired'
    owner_status: 'current_controller' | 'unassigned' | 'foreign_controller' | 'lease_expired'
    lease_id: string | null
  }
  compute: {
    availability: 'reachable' | 'inaccessible' | 'unknown'
    integration_label: 'NeMo' | null
  }
  artifacts: { checkpoint: 'available' | 'missing'; artifact: 'available' | 'missing' }
  compatibility: { schema: 'compatible' | 'incompatible'; revision: 'compatible' | 'incompatible' }
  doctor: {
    evidence_id: string
    evidence_digest: string
    checks: Array<{ check: DoctorCheck; status: 'pass' | 'fail' | 'unknown' }>
  }
  receipts: RecoveryReceiptPublicV1[]
  eligibility: RecoveryEligibilityPublicV1 | null
  execution_consumer: RecoveryExecutionConsumerPublicV1
  latest_execution: RecoveryExecutionPublicV1 | null
  consumed_idempotency_keys: string[]
}

export interface RecoveryBlockerViewModel {
  code: RecoveryBlockerCode
  label: string
}

export interface RecoveryRequest {
  action: RecoveryAction
  idempotencyKey: string
  workspaceId: string
  campaignId: string
  eligibilityReceiptId: string
  doctorEvidenceId: string
  expectedCampaignRevision: number
  expectedEventCursor: number
  expectedAggregateVersion: number
  expectedControllerLeaseId: string | null
  checkpointId: string
  artifactId: string
  humanConfirmed: true
}

export interface CampaignRecoveryViewModel {
  snapshot: CampaignRecoveryPublicV1
  freshness: RecoveryFreshness
  decision: RecoveryDecision
  decisionLabel: string
  mutationsEnabled: boolean
  blockers: RecoveryBlockerViewModel[]
  lineage: CampaignRecoveryPublicV1['lineage']
  eligibilityVerified: boolean
  consumerReady: boolean
  consumerGuidance: string
  latestReceipt: RecoveryReceiptPublicV1 | null
  latestExecution: RecoveryExecutionPublicV1 | null
  actions: Record<RecoveryAction, { label: string; enabled: boolean; guidance: string }>
}

const REQUIRED_DOCTOR_CHECKS: DoctorCheck[] = [
  'binding_verified',
  'checkpoint_available',
  'compute_reachable',
  'schema_compatible',
  'lease_observed'
]
const ACTIONS: RecoveryAction[] = ['resume', 'repair', 'takeover']

function hasExactKeys(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  return actual.length === expected.length && actual.every((key, index) => key === expected[index])
}

function matches(value: unknown, pattern: RegExp): value is string {
  return typeof value === 'string' && pattern.test(value)
}

const isDurableId = (value: unknown): value is string =>
  matches(value, /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/)
const isWorkspaceId = isDurableId
const isCampaignId = isDurableId
const isInstallationId = (value: unknown): value is string => matches(value, /^ins_[a-f0-9]{32}$/)
const isModelBindingId = (value: unknown): value is string => matches(value, /^mdl_[a-f0-9]{32}$/)
const isDataBindingId = (value: unknown): value is string => matches(value, /^dat_[a-f0-9]{32}$/)
const isEvaluatorBindingId = (value: unknown): value is string =>
  matches(value, /^evl_[a-f0-9]{32}$/)
const isComputeBindingId = (value: unknown): value is string => matches(value, /^cpu_[a-f0-9]{32}$/)
const isCheckpointId = (value: unknown): value is string => matches(value, /^ckpt_[a-f0-9]{32}$/)
const isArtifactId = (value: unknown): value is string => matches(value, /^art_[a-f0-9]{32}$/)
const isSchemaId = (value: unknown): value is string => matches(value, /^sch_[a-f0-9]{32}$/)
const isLeaseId = (value: unknown): value is string => matches(value, /^lease_[a-f0-9]{32}$/)
const isEvidenceId = (value: unknown): value is string => matches(value, /^evd_[a-f0-9]{32}$/)
const isReceiptId = (value: unknown): value is string => matches(value, /^rcpt_[a-f0-9]{32}$/)
const isDigest = (value: unknown): value is string => matches(value, /^sha256_[a-f0-9]{64}$/)
const isIdempotencyKey = (value: unknown): value is string => matches(value, /^idem_[a-f0-9]{32}$/)
const isAttemptId = (value: unknown): value is string =>
  matches(value, /^attempt-[A-Za-z0-9][A-Za-z0-9_.:-]{0,151}$/)

const CREDENTIAL_LABEL_SIGNATURES = [
  /(?:^|[ ._+-])(?:gh[pousr]_|github_pat_)/i,
  /(?:^|[ ._+-])sk[_-](?:(?:proj|live|test)[_-])?[A-Za-z0-9]/i,
  /(?:^|[ ._+-])AKIA[0-9A-Z]{16}(?=$|[ ._+-])/,
  /(?:^|[ ._+-])xox[baprs]-/i,
  /(?:^|[ ._+-])AIza[0-9A-Za-z_-]{20,}(?=$|[ ._+-])/,
  /(?:^|[ ._+-])(?:begin[ ._+-]+)?private[ ._+-]+key(?:$|[ ._+-])/i,
  /(?:^|[ ._+-])(?:api[ ._+-]*key|client[ ._+-]*secret|access[ ._+-]*token|password)(?:$|[ ._+-])/i
]
const PERSONAL_DEVICE_LABEL_SIGNATURE =
  /(?:macbook(?:[ ._+-]*(?:pro|air))?|imac|iphone|ipad|laptop|desktop|workstation|localhost|hostname)/i

function isPublicModelDisplayLabel(value: unknown): value is string {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9 .+_-]{0,95}$/.test(value)) return false
  if (/[\\/]/.test(value) || /^[A-Za-z]:/.test(value) || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(value))
    return false
  return (
    !CREDENTIAL_LABEL_SIGNATURES.some((signature) => signature.test(value)) &&
    !PERSONAL_DEVICE_LABEL_SIGNATURE.test(value)
  )
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function isTimestamp(value: unknown): value is string {
  if (
    typeof value !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value)
  )
    return false
  const parsed = new Date(value)
  if (!Number.isFinite(parsed.getTime())) return false
  const canonical = parsed.toISOString()
  return value === canonical || value === canonical.replace('.000Z', 'Z')
}

function isAction(value: unknown): value is RecoveryAction {
  return value === 'resume' || value === 'repair' || value === 'takeover'
}

function parseInstallation(value: unknown): CampaignRecoveryPublicV1['installation'] | null {
  if (!hasExactKeys(value, ['installation_id', 'lineage_mode'])) return null
  return isInstallationId(value.installation_id) &&
    (value.lineage_mode === 'clone' || value.lineage_mode === 'fork')
    ? { installation_id: value.installation_id, lineage_mode: value.lineage_mode }
    : null
}

function parseBindings(value: unknown): CampaignRecoveryPublicV1['bindings'] | null {
  if (
    !hasExactKeys(value, [
      'model_id',
      'model_display_label',
      'data_id',
      'evaluator_id',
      'compute_id'
    ])
  )
    return null
  if (value.model_id !== null && !isModelBindingId(value.model_id)) return null
  if (value.model_display_label !== null && !isPublicModelDisplayLabel(value.model_display_label))
    return null
  if (value.model_id === null && value.model_display_label !== null) return null
  if (value.data_id !== null && !isDataBindingId(value.data_id)) return null
  if (value.evaluator_id !== null && !isEvaluatorBindingId(value.evaluator_id)) return null
  if (value.compute_id !== null && !isComputeBindingId(value.compute_id)) return null
  return {
    model_id: value.model_id,
    model_display_label: value.model_display_label,
    data_id: value.data_id,
    evaluator_id: value.evaluator_id,
    compute_id: value.compute_id
  }
}

function parseLineage(value: unknown): CampaignRecoveryPublicV1['lineage'] | null {
  const keys = [
    'checkpoint_id',
    'artifact_id',
    'parent_campaign_id',
    'campaign_revision',
    'event_cursor',
    'aggregate_version',
    'schema_version'
  ]
  if (!hasExactKeys(value, keys)) return null
  if (
    !isCheckpointId(value.checkpoint_id) ||
    !isArtifactId(value.artifact_id) ||
    (value.parent_campaign_id !== null && !isCampaignId(value.parent_campaign_id))
  )
    return null
  if (
    !isPositiveInteger(value.campaign_revision) ||
    !isNonNegativeInteger(value.event_cursor) ||
    !isPositiveInteger(value.aggregate_version) ||
    !isSchemaId(value.schema_version)
  )
    return null
  return {
    checkpoint_id: value.checkpoint_id,
    artifact_id: value.artifact_id,
    parent_campaign_id: value.parent_campaign_id,
    campaign_revision: value.campaign_revision,
    event_cursor: value.event_cursor,
    aggregate_version: value.aggregate_version,
    schema_version: value.schema_version
  }
}

function parseController(value: unknown): CampaignRecoveryPublicV1['controller'] | null {
  if (!hasExactKeys(value, ['state', 'owner_status', 'lease_id'])) return null
  if (
    value.state === 'owned' &&
    value.owner_status === 'current_controller' &&
    isLeaseId(value.lease_id)
  )
    return { state: value.state, owner_status: value.owner_status, lease_id: value.lease_id }
  if (value.state === 'unowned' && value.owner_status === 'unassigned' && value.lease_id === null)
    return { state: value.state, owner_status: value.owner_status, lease_id: null }
  if (
    value.state === 'foreign_lease' &&
    value.owner_status === 'foreign_controller' &&
    isLeaseId(value.lease_id)
  )
    return { state: value.state, owner_status: value.owner_status, lease_id: value.lease_id }
  if (
    value.state === 'expired' &&
    value.owner_status === 'lease_expired' &&
    isLeaseId(value.lease_id)
  )
    return { state: value.state, owner_status: value.owner_status, lease_id: value.lease_id }
  return null
}

function isDoctorCheck(value: unknown): value is DoctorCheck {
  return REQUIRED_DOCTOR_CHECKS.includes(value as DoctorCheck)
}

function parseDoctor(value: unknown): CampaignRecoveryPublicV1['doctor'] | null {
  if (
    !hasExactKeys(value, ['evidence_id', 'evidence_digest', 'checks']) ||
    !isEvidenceId(value.evidence_id) ||
    !isDigest(value.evidence_digest) ||
    !Array.isArray(value.checks) ||
    value.checks.length > REQUIRED_DOCTOR_CHECKS.length
  )
    return null
  const checks = value.checks.map((entry) => {
    if (
      !hasExactKeys(entry, ['check', 'status']) ||
      !isDoctorCheck(entry.check) ||
      (entry.status !== 'pass' && entry.status !== 'fail' && entry.status !== 'unknown')
    )
      return null
    return { check: entry.check, status: entry.status }
  })
  if (
    !checks.every(
      (entry): entry is CampaignRecoveryPublicV1['doctor']['checks'][number] => entry !== null
    )
  )
    return null
  if (new Set(checks.map((entry) => entry.check)).size !== checks.length) return null
  return { evidence_id: value.evidence_id, evidence_digest: value.evidence_digest, checks }
}

function parseScope(value: Record<string, unknown>): RecoveryScopePublicV1 | null {
  if (
    !isWorkspaceId(value.workspace_id) ||
    !isCampaignId(value.campaign_id) ||
    !isPositiveInteger(value.campaign_revision) ||
    !isNonNegativeInteger(value.event_cursor) ||
    !isPositiveInteger(value.aggregate_version)
  )
    return null
  return {
    workspace_id: value.workspace_id,
    campaign_id: value.campaign_id,
    campaign_revision: value.campaign_revision,
    event_cursor: value.event_cursor,
    aggregate_version: value.aggregate_version
  }
}

function parseReceipts(value: unknown): RecoveryReceiptPublicV1[] | null {
  if (!Array.isArray(value) || value.length > 32) return null
  const receipts = value.map((entry) => {
    const keys = [
      'receipt_id',
      'kind',
      'workspace_id',
      'campaign_id',
      'campaign_revision',
      'event_cursor',
      'aggregate_version',
      'emitted_at',
      'digest'
    ]
    if (
      !hasExactKeys(entry, keys) ||
      !isReceiptId(entry.receipt_id) ||
      (entry.kind !== 'doctor' && entry.kind !== 'eligibility' && entry.kind !== 'recovery') ||
      !isTimestamp(entry.emitted_at) ||
      !isDigest(entry.digest)
    )
      return null
    const scope = parseScope(entry)
    return scope
      ? {
          receipt_id: entry.receipt_id,
          kind: entry.kind,
          ...scope,
          emitted_at: entry.emitted_at,
          digest: entry.digest
        }
      : null
  })
  if (!receipts.every((entry): entry is RecoveryReceiptPublicV1 => entry !== null)) return null
  if (new Set(receipts.map((entry) => entry.receipt_id)).size !== receipts.length) return null
  if (
    receipts.some(
      (entry, index) =>
        index > 0 && Date.parse(entry.emitted_at) <= Date.parse(receipts[index - 1]!.emitted_at)
    )
  )
    return null
  return receipts
}

function parseEligibility(value: unknown): RecoveryEligibilityPublicV1 | null | undefined {
  if (value === null) return null
  const keys = [
    'receipt_id',
    'receipt_digest',
    'decision',
    'allowed_actions',
    'workspace_id',
    'campaign_id',
    'campaign_revision',
    'event_cursor',
    'aggregate_version',
    'doctor_evidence_id',
    'doctor_evidence_digest',
    'issued_at',
    'expires_at'
  ]
  if (
    !hasExactKeys(value, keys) ||
    !isReceiptId(value.receipt_id) ||
    !isDigest(value.receipt_digest) ||
    (value.decision !== 'eligible' && value.decision !== 'blocked') ||
    !Array.isArray(value.allowed_actions) ||
    value.allowed_actions.length > ACTIONS.length
  )
    return undefined
  if (
    !value.allowed_actions.every(isAction) ||
    new Set(value.allowed_actions).size !== value.allowed_actions.length ||
    !isEvidenceId(value.doctor_evidence_id) ||
    !isDigest(value.doctor_evidence_digest) ||
    !isTimestamp(value.issued_at) ||
    !isTimestamp(value.expires_at)
  )
    return undefined
  const scope = parseScope(value)
  if (!scope || Date.parse(value.issued_at) >= Date.parse(value.expires_at)) return undefined
  return {
    receipt_id: value.receipt_id,
    receipt_digest: value.receipt_digest,
    decision: value.decision,
    allowed_actions: [...value.allowed_actions],
    ...scope,
    doctor_evidence_id: value.doctor_evidence_id,
    doctor_evidence_digest: value.doctor_evidence_digest,
    issued_at: value.issued_at,
    expires_at: value.expires_at
  }
}

function parseConsumedKeys(value: unknown): string[] | null {
  if (
    !Array.isArray(value) ||
    value.length > 64 ||
    !value.every(isIdempotencyKey) ||
    new Set(value).size !== value.length
  )
    return null
  return [...value]
}

function parseLatestExecution(value: unknown): RecoveryExecutionPublicV1 | null | undefined {
  if (value === null) return null
  const keys = [
    'schema_version',
    'workspace_id',
    'campaign_id',
    'action',
    'status',
    'outcome_code',
    'attempt_id'
  ]
  if (!hasExactKeys(value, keys) || value.schema_version !== 'campaign_recovery_execution.v1')
    return undefined
  if (!isWorkspaceId(value.workspace_id) || !isCampaignId(value.campaign_id)) return undefined
  if (value.action !== 'resume' && value.action !== 'repair') return undefined
  if (!['accepted', 'executing', 'completed', 'failed', 'blocked'].includes(String(value.status)))
    return undefined
  const status = value.status as RecoveryExecutionStatus
  const outcomeCodes: RecoveryOutcomeCode[] = [
    'campaign_resumed',
    'attempt_reconciled',
    'needs_operator',
    'authority_changed'
  ]
  if (
    value.outcome_code !== null &&
    !outcomeCodes.includes(value.outcome_code as RecoveryOutcomeCode)
  )
    return undefined
  if (value.attempt_id !== null && !isAttemptId(value.attempt_id)) return undefined
  if (value.action === 'resume' && value.attempt_id !== null) return undefined
  if ((status === 'accepted' || status === 'executing') && value.outcome_code !== null)
    return undefined
  if (
    (status === 'completed' || status === 'failed' || status === 'blocked') &&
    value.outcome_code === null
  )
    return undefined
  if (
    status === 'completed' &&
    value.outcome_code !== (value.action === 'resume' ? 'campaign_resumed' : 'attempt_reconciled')
  )
    return undefined
  if (
    (status === 'failed' || status === 'blocked') &&
    value.outcome_code !== 'needs_operator' &&
    value.outcome_code !== 'authority_changed'
  )
    return undefined
  return {
    schema_version: 'campaign_recovery_execution.v1',
    workspace_id: value.workspace_id,
    campaign_id: value.campaign_id,
    action: value.action,
    status,
    outcome_code: value.outcome_code as RecoveryOutcomeCode | null,
    attempt_id: value.attempt_id
  }
}

function parseExecutionConsumer(value: unknown): RecoveryExecutionConsumerPublicV1 | null {
  if (!hasExactKeys(value, ['supported', 'ready', 'reason_code'])) return null
  if (typeof value.supported !== 'boolean' || typeof value.ready !== 'boolean') return null
  const reasons: RecoveryExecutionConsumerPublicV1['reason_code'][] = [
    'ready',
    'controller_unowned',
    'controller_lease_expired',
    'foreign_controller'
  ]
  if (!reasons.includes(value.reason_code as RecoveryExecutionConsumerPublicV1['reason_code']))
    return null
  if ((value.reason_code === 'ready') !== (value.supported && value.ready)) return null
  if (value.ready && !value.supported) return null
  return {
    supported: value.supported,
    ready: value.ready,
    reason_code: value.reason_code as RecoveryExecutionConsumerPublicV1['reason_code']
  }
}

/** Accepts only the bounded, backend-issued public recovery projection. */
export function parseCampaignRecovery(value: unknown): CampaignRecoveryPublicV1 | null {
  const keys = [
    'schema_version',
    'workspace_id',
    'campaign_id',
    'installation',
    'bindings',
    'lineage',
    'controller',
    'compute',
    'artifacts',
    'compatibility',
    'doctor',
    'receipts',
    'eligibility',
    'execution_consumer',
    'latest_execution',
    'consumed_idempotency_keys'
  ]
  if (
    !hasExactKeys(value, keys) ||
    value.schema_version !== CAMPAIGN_RECOVERY_SCHEMA_VERSION ||
    !isWorkspaceId(value.workspace_id) ||
    !isCampaignId(value.campaign_id)
  )
    return null
  const installation = parseInstallation(value.installation)
  const bindings = parseBindings(value.bindings)
  const lineage = parseLineage(value.lineage)
  const controller = parseController(value.controller)
  const doctor = parseDoctor(value.doctor)
  const receipts = parseReceipts(value.receipts)
  const eligibility = parseEligibility(value.eligibility)
  const executionConsumer = parseExecutionConsumer(value.execution_consumer)
  const latestExecution = parseLatestExecution(value.latest_execution)
  const consumedKeys = parseConsumedKeys(value.consumed_idempotency_keys)
  if (
    !installation ||
    !bindings ||
    !lineage ||
    !controller ||
    !doctor ||
    !receipts ||
    eligibility === undefined ||
    !executionConsumer ||
    latestExecution === undefined ||
    !consumedKeys
  )
    return null
  const consumerReasonByController: Record<
    CampaignRecoveryPublicV1['controller']['state'],
    RecoveryExecutionConsumerPublicV1['reason_code']
  > = {
    owned: 'ready',
    unowned: 'controller_unowned',
    expired: 'controller_lease_expired',
    foreign_lease: 'foreign_controller'
  }
  if (executionConsumer.reason_code !== consumerReasonByController[controller.state]) return null
  if (
    receipts.some(
      (receipt) =>
        receipt.workspace_id !== value.workspace_id || receipt.campaign_id !== value.campaign_id
    )
  )
    return null
  if (
    latestExecution &&
    (latestExecution.workspace_id !== value.workspace_id ||
      latestExecution.campaign_id !== value.campaign_id)
  )
    return null
  if (
    !hasExactKeys(value.compute, ['availability', 'integration_label']) ||
    (value.compute.availability !== 'reachable' &&
      value.compute.availability !== 'inaccessible' &&
      value.compute.availability !== 'unknown') ||
    (value.compute.integration_label !== null && value.compute.integration_label !== 'NeMo')
  )
    return null
  if (
    !hasExactKeys(value.artifacts, ['checkpoint', 'artifact']) ||
    (value.artifacts.checkpoint !== 'available' && value.artifacts.checkpoint !== 'missing') ||
    (value.artifacts.artifact !== 'available' && value.artifacts.artifact !== 'missing')
  )
    return null
  if (
    !hasExactKeys(value.compatibility, ['schema', 'revision']) ||
    (value.compatibility.schema !== 'compatible' &&
      value.compatibility.schema !== 'incompatible') ||
    (value.compatibility.revision !== 'compatible' &&
      value.compatibility.revision !== 'incompatible')
  )
    return null
  return {
    schema_version: CAMPAIGN_RECOVERY_SCHEMA_VERSION,
    workspace_id: value.workspace_id,
    campaign_id: value.campaign_id,
    installation,
    bindings,
    lineage,
    controller,
    compute: {
      availability: value.compute.availability,
      integration_label: value.compute.integration_label
    },
    artifacts: { checkpoint: value.artifacts.checkpoint, artifact: value.artifacts.artifact },
    compatibility: { schema: value.compatibility.schema, revision: value.compatibility.revision },
    doctor,
    receipts,
    eligibility,
    execution_consumer: executionConsumer,
    latest_execution: latestExecution,
    consumed_idempotency_keys: consumedKeys
  }
}

function scopeMatches(
  snapshot: CampaignRecoveryPublicV1,
  eligibility: RecoveryEligibilityPublicV1
): boolean {
  return (
    eligibility.workspace_id === snapshot.workspace_id &&
    eligibility.campaign_id === snapshot.campaign_id &&
    eligibility.campaign_revision === snapshot.lineage.campaign_revision &&
    eligibility.event_cursor === snapshot.lineage.event_cursor &&
    eligibility.aggregate_version === snapshot.lineage.aggregate_version
  )
}

function hasCompleteDoctorEvidence(snapshot: CampaignRecoveryPublicV1): boolean {
  return REQUIRED_DOCTOR_CHECKS.every((required) =>
    snapshot.doctor.checks.some((entry) => entry.check === required && entry.status === 'pass')
  )
}

function hasCurrentEligibility(snapshot: CampaignRecoveryPublicV1, now: string): boolean {
  const eligibility = snapshot.eligibility
  const latest = snapshot.receipts.at(-1)
  if (
    !isTimestamp(now) ||
    !eligibility ||
    eligibility.decision !== 'eligible' ||
    !scopeMatches(snapshot, eligibility) ||
    !hasCompleteDoctorEvidence(snapshot)
  )
    return false
  if (
    eligibility.doctor_evidence_id !== snapshot.doctor.evidence_id ||
    eligibility.doctor_evidence_digest !== snapshot.doctor.evidence_digest
  )
    return false
  if (
    Date.parse(eligibility.issued_at) > Date.parse(now) ||
    Date.parse(eligibility.expires_at) <= Date.parse(now)
  )
    return false
  return (
    latest?.kind === 'eligibility' &&
    latest.receipt_id === eligibility.receipt_id &&
    latest.digest === eligibility.receipt_digest &&
    latest.emitted_at === eligibility.issued_at &&
    latest.workspace_id === eligibility.workspace_id &&
    latest.campaign_id === eligibility.campaign_id &&
    latest.campaign_revision === eligibility.campaign_revision &&
    latest.event_cursor === eligibility.event_cursor &&
    latest.aggregate_version === eligibility.aggregate_version
  )
}

function blocker(code: RecoveryBlockerCode, label: string): RecoveryBlockerViewModel {
  return { code, label }
}

function controllerPermits(
  action: RecoveryAction,
  state: CampaignRecoveryPublicV1['controller']['state']
): boolean {
  if (action === 'takeover') return state === 'expired'
  return state === 'owned' || state === 'unowned'
}

export function buildCampaignRecoveryModel(input: {
  snapshot: CampaignRecoveryPublicV1
  freshness: RecoveryFreshness
  now: string
}): CampaignRecoveryViewModel {
  const { snapshot, freshness } = input
  const blockers: RecoveryBlockerViewModel[] = []
  const requiredBindingIds = [
    snapshot.bindings.model_id,
    snapshot.bindings.data_id,
    snapshot.bindings.evaluator_id,
    snapshot.bindings.compute_id
  ]
  if (requiredBindingIds.some((value) => value === null))
    blockers.push(
      blocker(
        'missing_binding',
        'An immutable model, data, evaluator, or compute binding is missing.'
      )
    )
  if (snapshot.compute.availability !== 'reachable')
    blockers.push(
      blocker('private_compute_inaccessible', 'The private compute binding is not reachable.')
    )
  if (snapshot.artifacts.checkpoint !== 'available')
    blockers.push(blocker('checkpoint_missing', 'The required checkpoint is unavailable.'))
  if (snapshot.artifacts.artifact !== 'available')
    blockers.push(blocker('artifact_missing', 'The required artifact lineage is unavailable.'))
  if (snapshot.compatibility.schema !== 'compatible')
    blockers.push(
      blocker('schema_incompatible', 'The recovery schema is incompatible with this campaign.')
    )
  if (snapshot.compatibility.revision !== 'compatible')
    blockers.push(
      blocker('revision_incompatible', 'The authoritative campaign revision is incompatible.')
    )
  if (snapshot.controller.state === 'foreign_lease')
    blockers.push(
      blocker(
        'active_foreign_lease',
        'Another controller holds the active lease; it will not be taken over automatically.'
      )
    )
  const eligibilityVerified = hasCurrentEligibility(snapshot, input.now)
  if (!eligibilityVerified)
    blockers.push(
      blocker(
        'eligibility_unverified',
        'A current server-issued recovery eligibility receipt bound to this workspace and campaign is required.'
      )
    )
  const consumerReady =
    freshness === 'live' &&
    snapshot.execution_consumer.supported &&
    snapshot.execution_consumer.ready &&
    snapshot.execution_consumer.reason_code === 'ready'
  if (!consumerReady)
    blockers.push(
      blocker(
        'recovery_consumer_unavailable',
        'The authenticated projection does not show a current resident recovery worker. Requests remain disabled.'
      )
    )
  const executionInFlight =
    snapshot.latest_execution?.status === 'accepted' ||
    snapshot.latest_execution?.status === 'executing'
  if (executionInFlight)
    blockers.push(
      blocker(
        'recovery_request_in_flight',
        'A recovery request is already accepted or executing. Wait for its terminal receipt before requesting another.'
      )
    )
  const executionNeedsOperator =
    snapshot.latest_execution?.status === 'failed' ||
    snapshot.latest_execution?.status === 'blocked'
  if (executionNeedsOperator)
    blockers.push(
      blocker(
        'recovery_execution_needs_operator',
        'The latest recovery execution requires operator reconciliation before another request.'
      )
    )
  const allowed = snapshot.eligibility?.allowed_actions || []
  if (
    eligibilityVerified &&
    !allowed.some((recoveryAction) => controllerPermits(recoveryAction, snapshot.controller.state))
  ) {
    blockers.push(
      blocker(
        'no_permitted_recovery_action',
        'No recovery action in the current eligibility receipt is compatible with the controller state.'
      )
    )
  }

  const liveAndClear = consumerReady && eligibilityVerified && blockers.length === 0
  const action = (label: string, enabled: boolean, guidance: string) => ({
    label,
    enabled,
    guidance
  })
  const actions = {
    resume: action(
      'Resume campaign',
      liveAndClear &&
        allowed.includes('resume') &&
        controllerPermits('resume', snapshot.controller.state),
      'Requires current scope-bound eligibility, confirmation, and a fresh idempotency key.'
    ),
    repair: action(
      'Request repair',
      liveAndClear &&
        allowed.includes('repair') &&
        controllerPermits('repair', snapshot.controller.state),
      'Repairs are explicit requests and preserve the recorded lineage.'
    ),
    takeover: action(
      'Take over expired lease',
      liveAndClear &&
        allowed.includes('takeover') &&
        controllerPermits('takeover', snapshot.controller.state),
      'Only an expired lease can be requested; active foreign leases are never stolen.'
    )
  }
  const mutationsEnabled = Object.values(actions).some((item) => item.enabled)
  const decision: RecoveryDecision =
    freshness === 'offline' || freshness === 'stale'
      ? 'read_only'
      : freshness === 'reconciling' ||
          freshness === 'error' ||
          snapshot.compute.availability !== 'reachable'
        ? 'degraded'
        : mutationsEnabled
          ? 'resumable'
          : 'blocked'
  return {
    snapshot,
    freshness,
    decision,
    decisionLabel: {
      resumable: 'Resumable',
      blocked: 'Blocked',
      read_only: 'Read-only',
      degraded: 'Degraded'
    }[decision],
    mutationsEnabled,
    blockers,
    lineage: { ...snapshot.lineage },
    eligibilityVerified,
    consumerReady,
    consumerGuidance: consumerReady
      ? 'The backend confirms that the resident worker can consume recovery requests.'
      : {
          ready: 'Recovery consumer readiness is unavailable.',
          controller_unowned: 'No resident recovery worker currently owns the controller lease.',
          controller_lease_expired: 'The recovery worker controller lease has expired.',
          foreign_controller: 'A different controller owns the active recovery lease.'
        }[snapshot.execution_consumer.reason_code],
    latestReceipt: snapshot.receipts.at(-1) || null,
    latestExecution: snapshot.latest_execution,
    actions
  }
}

export function buildRecoveryRequest(input: {
  model: CampaignRecoveryViewModel
  action: RecoveryAction
  humanConfirmed: boolean
  idempotencyKey: string
}): RecoveryRequest | null {
  const { model } = input
  const eligibility = model.snapshot.eligibility
  if (
    !model.actions[input.action].enabled ||
    !eligibility ||
    !input.humanConfirmed ||
    !isIdempotencyKey(input.idempotencyKey) ||
    model.snapshot.consumed_idempotency_keys.includes(input.idempotencyKey)
  )
    return null
  return {
    action: input.action,
    idempotencyKey: input.idempotencyKey,
    workspaceId: model.snapshot.workspace_id,
    campaignId: model.snapshot.campaign_id,
    eligibilityReceiptId: eligibility.receipt_id,
    doctorEvidenceId: model.snapshot.doctor.evidence_id,
    expectedCampaignRevision: model.lineage.campaign_revision,
    expectedEventCursor: model.lineage.event_cursor,
    expectedAggregateVersion: model.lineage.aggregate_version,
    expectedControllerLeaseId: model.snapshot.controller.lease_id,
    checkpointId: model.lineage.checkpoint_id,
    artifactId: model.lineage.artifact_id,
    humanConfirmed: true
  }
}
