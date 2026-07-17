/** Pure authority contracts for one campaign-scoped coding/training agent. */
export const supportedCampaignAgentFamilies = ['codex', 'hermes'] as const
export type CampaignAgentFamily = typeof supportedCampaignAgentFamilies[number]

export const campaignAgentCapabilities = [
  'campaign_observe',
  'training_launch',
  'training_pause_self',
  'artifact_read',
  'artifact_propose',
] as const
export type CampaignAgentCapability = typeof campaignAgentCapabilities[number]

/** Capabilities implemented by the current main-owned, read-only Codex bridge. */
export const launchableCampaignAgentCapabilities = [
  'campaign_observe',
  'artifact_read',
] as const satisfies readonly CampaignAgentCapability[]

/** Declared contract capabilities which are not executable through the direct bridge yet. */
export const unavailableDirectCampaignAgentCapabilities = [
  'training_launch',
  'training_pause_self',
  'artifact_propose',
] as const satisfies readonly CampaignAgentCapability[]

export const prohibitedCampaignAgentActions = [
  'human_work_approve',
  'budget_change',
  'promotion_override',
  'publication_publish',
  'campaign_force_stop',
] as const

export type CampaignAgentFreshness = 'live' | 'reconciling' | 'stale' | 'offline' | 'error'

export const campaignAgentAuditMessageCodes = [
  'agent_attached',
  'agent_revoked',
  'credential_renewed',
  'liveness_changed',
] as const
export type CampaignAgentAuditMessageCode = typeof campaignAgentAuditMessageCodes[number]

export interface CampaignAgentScope {
  campaignId: string
  workspaceId: string
}

export interface CampaignAgentGrantConfirmationReceipt {
  readonly schemaVersion: 'campaign_agent_grant_confirmation.v1'
  readonly issuer: 'campaign_authority'
  readonly receiptId: string
  readonly receiptDigest: string
  readonly humanPrincipal: { readonly principalId: string; readonly principalType: 'human' }
  readonly scope: CampaignAgentScope
  readonly agentFamily: CampaignAgentFamily
  readonly agentOrigin: string
  readonly agentPrincipalId: string
  readonly sessionId: string
  readonly requestedCapabilities: readonly CampaignAgentCapability[]
  readonly grantedCapabilities: readonly CampaignAgentCapability[]
  readonly capabilityDigest: string
  readonly grantRevision: number
  readonly issuedAt: string
  readonly expiresAt: string
}

export interface CampaignAgentAttachDraft {
  scope: CampaignAgentScope
  agentFamily: CampaignAgentFamily
  agentOrigin: string
  sessionId: string
  agentPrincipalId: string
  requestedCapabilities: CampaignAgentCapability[]
  grantedCapabilities: CampaignAgentCapability[]
  confirmation: CampaignAgentGrantConfirmationReceipt | null
  idempotencyKey: string
}

export interface CampaignAgentAttachRequest {
  action: 'attach'
  scope: CampaignAgentScope
  agentFamily: CampaignAgentFamily
  agentOrigin: string
  sessionId: string
  agentPrincipalId: string
  requestedCapabilities: CampaignAgentCapability[]
  grantedCapabilities: CampaignAgentCapability[]
  confirmationReceipt: CampaignAgentGrantConfirmationReceipt
  baseAttachmentVersion: number | null
  idempotencyKey: string
}

export type CampaignAgentCredentialStatus = 'active' | 'expired' | 'revoked'
export type CampaignAgentLiveness = 'live' | 'idle' | 'offline' | 'expired' | 'revoked'

export interface CampaignAgentPublicProvenance {
  readonly agentFamily: CampaignAgentFamily
  readonly agentOrigin: string
  readonly agentOriginStatus: 'verified'
  readonly sessionId: string
  readonly agentPrincipalId: string
  readonly attachedAt: string
  readonly attachedBy: string
  readonly grantReceiptId: string
  readonly grantReceiptDigest: string
  readonly credentialIssuedAt: string
  readonly credentialExpiresAt: string
  readonly credentialStatus: CampaignAgentCredentialStatus
  readonly credentialRevocationRevision: number
  readonly credentialStatusSource: 'campaign_authority'
  readonly liveness: CampaignAgentLiveness
  readonly resumeCursor: string | null
  readonly revokedAt: string | null
  readonly revokedBy: string | null
}

export interface CampaignAgentPublicReceipt {
  readonly receiptId: string
  readonly kind: 'attach' | 'revoke'
  readonly actorId: string
  readonly occurredAt: string
  readonly idempotencyKey: string
  readonly attachmentVersion: number
  readonly receiptDigest: string
}

export interface CampaignAgentPublicAttachment {
  readonly schemaVersion: 'campaign_agent_public_attachment.v1'
  readonly attachmentId: string
  readonly attachmentVersion: number
  readonly status: 'attached' | 'revoked'
  readonly scope: Readonly<CampaignAgentScope>
  readonly requestedCapabilities: readonly CampaignAgentCapability[]
  readonly grantedCapabilities: readonly CampaignAgentCapability[]
  readonly receiptWindow: {
    readonly fromVersion: number
    readonly throughVersion: number
    readonly hasEarlier: boolean
  }
  readonly provenance: CampaignAgentPublicProvenance
  readonly receipts: readonly CampaignAgentPublicReceipt[]
}

export interface CampaignAgentPublicAuditEvent {
  readonly eventId: string
  readonly sequence: number
  readonly kind: 'attachment' | 'revocation' | 'credential' | 'liveness'
  readonly occurredAt: string
  readonly messageCode: CampaignAgentAuditMessageCode
}

const publicViewBrand: unique symbol = Symbol('CampaignAgentPublicView')

export interface CampaignAgentPublicView {
  readonly [publicViewBrand]: true
  readonly schemaVersion: 'campaign_agent_public_view.v1'
  readonly observedAt: string
  readonly scope: Readonly<CampaignAgentScope>
  readonly attachment: CampaignAgentPublicAttachment | null
  readonly auditEvents: readonly CampaignAgentPublicAuditEvent[]
}

export interface CampaignAgentRevokeRequest {
  action: 'revoke'
  attachmentId: string
  attachmentVersion: number
  scope: CampaignAgentScope
  actorId: string
  idempotencyKey: string
}

function normalizedCapabilities(values: readonly CampaignAgentCapability[]): CampaignAgentCapability[] {
  return [...values].sort()
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index])
}

function fnv1a(input: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

export function capabilityBindingDigest(
  agentPrincipalId: string,
  requested: readonly CampaignAgentCapability[],
  granted: readonly CampaignAgentCapability[],
): string {
  return `fnv1a32:${fnv1a(`principal=${agentPrincipalId};requested=${normalizedCapabilities(requested).join(',')};granted=${normalizedCapabilities(granted).join(',')}`)}`
}

export function isProhibitedCampaignAgentAction(action: string): boolean {
  return (prohibitedCampaignAgentActions as readonly string[]).includes(action)
}

function isKnownCapability(value: string): value is CampaignAgentCapability {
  return (campaignAgentCapabilities as readonly string[]).includes(value)
}

function isKnownFamily(value: string): value is CampaignAgentFamily {
  return (supportedCampaignAgentFamilies as readonly string[]).includes(value)
}

export function isCanonicalCampaignAgentTimestamp(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(value)) return false
  const epoch = Date.parse(value)
  return Number.isFinite(epoch) && new Date(epoch).toISOString() === `${value.slice(0, -1)}.000Z`
}

export function validateCampaignAgentAuthoritativeClock(authoritativeNow: string, observedAt: string): string | null {
  if (!isCanonicalCampaignAgentTimestamp(authoritativeNow) || !isCanonicalCampaignAgentTimestamp(observedAt)) {
    return 'Authoritative canonical current time and canonical observed time are required.'
  }
  if (Date.parse(authoritativeNow) < Date.parse(observedAt)) {
    return 'Authoritative current time cannot precede the observed server view.'
  }
  return null
}

function scopeError(selected: CampaignAgentScope, supplied: CampaignAgentScope): string | null {
  if (selected.workspaceId !== supplied.workspaceId) return 'Draft does not match the authoritative selected workspace.'
  if (selected.campaignId !== supplied.campaignId) return 'Draft does not match the authoritative selected campaign.'
  return null
}

function confirmationBindingError(draft: CampaignAgentAttachDraft, now: string): string | null {
  if (!draft.confirmation) return 'A current server-issued human grant confirmation receipt is required.'
  let receipt: CampaignAgentGrantConfirmationReceipt
  try {
    receipt = parseCampaignAgentGrantConfirmation(draft.confirmation)
  } catch (error) {
    return `Invalid server confirmation receipt: ${error instanceof Error ? error.message : 'validation failed.'}`
  }
  if (receipt.humanPrincipal.principalType !== 'human' || receipt.humanPrincipal.principalId === draft.agentPrincipalId) {
    return 'Grant confirmation requires a distinct human principal.'
  }
  if (Date.parse(receipt.issuedAt) > Date.parse(now)) return 'The human grant confirmation is not yet issued for the authoritative current time.'
  if (Date.parse(receipt.expiresAt) <= Date.parse(now)) return 'The human grant confirmation receipt has expired.'
  const requested = normalizedCapabilities(draft.requestedCapabilities)
  const granted = normalizedCapabilities(draft.grantedCapabilities)
  const boundRequested = normalizedCapabilities(receipt.requestedCapabilities)
  const boundGranted = normalizedCapabilities(receipt.grantedCapabilities)
  if (
    receipt.scope.workspaceId !== draft.scope.workspaceId
    || receipt.scope.campaignId !== draft.scope.campaignId
    || receipt.agentFamily !== draft.agentFamily
    || receipt.agentOrigin !== draft.agentOrigin
    || receipt.agentPrincipalId !== draft.agentPrincipalId
    || receipt.sessionId !== draft.sessionId
    || !sameStrings(requested, boundRequested)
    || !sameStrings(granted, boundGranted)
    || receipt.capabilityDigest !== capabilityBindingDigest(draft.agentPrincipalId, requested, granted)
  ) return 'The human grant confirmation is not bound to the exact current draft.'
  return null
}

export function validateGrantConfirmationRequest(
  draft: CampaignAgentAttachDraft,
  selectedScope: CampaignAgentScope,
): string[] {
  const mismatch = scopeError(selectedScope, draft.scope)
  if (mismatch) return [mismatch]
  if (!draft.scope.workspaceId.trim() || !draft.scope.campaignId.trim()) return ['Workspace and campaign scope are required.']
  if (!isKnownFamily(draft.agentFamily)) return ['Only Codex and Hermes are conformant executable agent families.']
  if (!draft.agentOrigin.trim() || !draft.sessionId.trim() || !draft.agentPrincipalId.trim() || !draft.idempotencyKey.trim()) {
    return ['Agent origin, session, principal, and idempotency key are required.']
  }
  if (draft.requestedCapabilities.length === 0 || draft.requestedCapabilities.length > campaignAgentCapabilities.length) {
    return ['At least one supported capability must be requested.']
  }
  const unsupported = draft.requestedCapabilities.find((capability) => !isKnownCapability(capability))
  if (unsupported) return [isProhibitedCampaignAgentAction(unsupported) ? `Prohibited authority action requested: ${unsupported}.` : 'Requested capabilities must be explicit and supported.']
  if (new Set(draft.requestedCapabilities).size !== draft.requestedCapabilities.length || new Set(draft.grantedCapabilities).size !== draft.grantedCapabilities.length) {
    return ['Capability declarations cannot contain duplicates.']
  }
  if (!draft.grantedCapabilities.every((capability) => isKnownCapability(capability) && draft.requestedCapabilities.includes(capability))) {
    return ['Granted capabilities must be a subset of requested capabilities.']
  }
  return []
}

export function validateAttachDraft(
  draft: CampaignAgentAttachDraft,
  selectedScope: CampaignAgentScope,
  now: string,
): string[] {
  if (!isCanonicalCampaignAgentTimestamp(now)) return ['Authoritative canonical current time is required.']
  const declarationErrors = validateGrantConfirmationRequest(draft, selectedScope)
  if (declarationErrors.length > 0) return declarationErrors
  const confirmationError = confirmationBindingError(draft, now)
  return confirmationError ? [confirmationError] : []
}

export function buildAttachRequest(
  selectedScope: CampaignAgentScope,
  draft: CampaignAgentAttachDraft,
  now: string,
  currentAttachment: CampaignAgentPublicAttachment | null,
): CampaignAgentAttachRequest {
  const mismatch = scopeError(selectedScope, draft.scope)
  if (mismatch) throw new Error(mismatch)
  if (currentAttachment) {
    const currentMismatch = scopeError(selectedScope, currentAttachment.scope)
    if (currentMismatch) throw new Error(`Current attachment: ${currentMismatch}`)
  }
  if (currentAttachment?.status === 'attached') throw new Error('An agent is already attached to this campaign.')
  const errors = validateAttachDraft(draft, selectedScope, now)
  if (errors.length > 0) throw new Error(errors.join(' '))
  const confirmationReceipt = parseCampaignAgentGrantConfirmation(draft.confirmation)
  return {
    action: 'attach',
    scope: { ...selectedScope },
    agentFamily: draft.agentFamily,
    agentOrigin: draft.agentOrigin,
    sessionId: draft.sessionId,
    agentPrincipalId: draft.agentPrincipalId,
    requestedCapabilities: normalizedCapabilities(draft.requestedCapabilities),
    grantedCapabilities: normalizedCapabilities(draft.grantedCapabilities),
    confirmationReceipt,
    baseAttachmentVersion: currentAttachment?.attachmentVersion ?? null,
    idempotencyKey: draft.idempotencyKey,
  }
}

export function buildRevokeRequest(
  selectedScope: CampaignAgentScope,
  attachment: CampaignAgentPublicAttachment,
  input: { actorId: string; idempotencyKey: string },
): CampaignAgentRevokeRequest {
  const mismatch = scopeError(selectedScope, attachment.scope)
  if (mismatch) throw new Error(mismatch)
  if (attachment.status === 'revoked') throw new Error('This agent attachment is already revoked.')
  if (!input.actorId.trim() || !input.idempotencyKey.trim()) throw new Error('Actor and idempotency key are required.')
  return {
    action: 'revoke',
    attachmentId: attachment.attachmentId,
    attachmentVersion: attachment.attachmentVersion,
    scope: { ...selectedScope },
    actorId: input.actorId,
    idempotencyKey: input.idempotencyKey,
  }
}

function asExactObject(value: unknown, expectedKeys: readonly string[], label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object.`)
  const record = value as Record<string, unknown>
  const keys = Object.keys(record).sort()
  const expected = [...expectedKeys].sort()
  if (!sameStrings(keys, expected)) throw new Error(`${label} must contain the exact public keys.`)
  return record
}

const sensitiveValuePattern = /(?:\b[A-Za-z]:\/|\/\/[^/\s]+\/[^/\s]+|(?:^|[^A-Za-z0-9/])\/(?!\/)\S+|(?:^|[^A-Za-z0-9])(?:\.\.?\/|~\/|\.ssh\/)\S*|\b(?:https?|file|ssh|s3|gs):\/\/|\bbearer\s+[A-Za-z0-9._-]+|(?:secret|token|api[_ -]?key|password|protected[_ -]?row[_ -]?id|private[_ -]?row[_ -]?id)\s*[:=]\s*\S+|\bsk-[A-Za-z0-9]{8,}|\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b|-----BEGIN(?: [A-Z]+)* PRIVATE KEY-----)/i

function normalizedPublicValue(value: string, label: string): string {
  let normalized = value.normalize('NFKC')
  for (let layer = 0; layer < 2; layer += 1) {
    let hasEncodedOctet = false
    for (let index = 0; index < normalized.length; index += 1) {
      if (normalized[index] !== '%') continue
      const octet = normalized.slice(index + 1, index + 3)
      if (/^[a-f0-9]{2}$/i.test(octet)) {
        hasEncodedOctet = true
        index += 2
        continue
      }
      if (index > 0 && /\d/.test(normalized[index - 1])) continue
      throw new Error(`${label} contains a sensitive public value (malformed percent encoding).`)
    }
    if (!hasEncodedOctet) break
    try {
      normalized = decodeURIComponent(normalized)
    } catch {
      throw new Error(`${label} contains a sensitive public value (malformed percent encoding).`)
    }
  }
  if (/%[a-f0-9]{2}/i.test(normalized)) throw new Error(`${label} contains a sensitive public value (percent encoding exceeds the allowed depth).`)
  return normalized.replace(/\\/g, '/')
}

function publicString(value: unknown, label: string, maxLength = 128): string {
  if (typeof value !== 'string') throw new Error(`${label} must be a string.`)
  if (value.trim().length === 0 || value.length > maxLength) throw new Error(`${label} must be between 1 and ${maxLength} characters.`)
  if (sensitiveValuePattern.test(normalizedPublicValue(value, label))) throw new Error(`${label} contains a sensitive public value.`)
  return value
}

function nullablePublicString(value: unknown, label: string): string | null {
  return value === null ? null : publicString(value, label)
}

function isoString(value: unknown, label: string): string {
  const parsed = publicString(value, label, 40)
  if (!isCanonicalCampaignAgentTimestamp(parsed)) throw new Error(`${label} must be a canonical UTC timestamp.`)
  return parsed
}

function publicInteger(value: unknown, label: string, minimum = 0): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) throw new Error(`${label} must be an integer of at least ${minimum}.`)
  return value
}

function boundedArray(value: unknown, label: string, maxLength: number): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`)
  if (value.length > maxLength) throw new Error(`${label} must contain at most ${maxLength} entries.`)
  return value
}

function parseScope(value: unknown, label: string): CampaignAgentScope {
  const record = asExactObject(value, ['workspace_id', 'campaign_id'], label)
  return {
    workspaceId: publicString(record.workspace_id, `${label}.workspace_id`),
    campaignId: publicString(record.campaign_id, `${label}.campaign_id`),
  }
}

function parseCapabilities(value: unknown, label: string): CampaignAgentCapability[] {
  const values = boundedArray(value, label, campaignAgentCapabilities.length).map((entry) => publicString(entry, label))
  if (values.some((entry) => !isKnownCapability(entry))) throw new Error(`${label} contains an unsupported or prohibited capability.`)
  if (new Set(values).size !== values.length) throw new Error(`${label} contains duplicate capabilities.`)
  return normalizedCapabilities(values as CampaignAgentCapability[])
}

export function parseCampaignAgentGrantConfirmation(value: unknown): CampaignAgentGrantConfirmationReceipt {
  const record = asExactObject(value, [
    'schemaVersion', 'issuer', 'receiptId', 'receiptDigest', 'humanPrincipal', 'scope', 'agentFamily',
    'agentOrigin', 'agentPrincipalId', 'sessionId', 'requestedCapabilities', 'grantedCapabilities',
    'capabilityDigest', 'grantRevision', 'issuedAt', 'expiresAt',
  ], 'Campaign agent grant confirmation')
  if (record.schemaVersion !== 'campaign_agent_grant_confirmation.v1' || record.issuer !== 'campaign_authority') {
    throw new Error('The grant confirmation must be issued by the campaign authority.')
  }
  const humanRecord = asExactObject(record.humanPrincipal, ['principalId', 'principalType'], 'Grant confirmation human principal')
  if (humanRecord.principalType !== 'human') throw new Error('Grant confirmation human principal type must be human.')
  const humanPrincipalId = publicString(humanRecord.principalId, 'grant_confirmation.humanPrincipal.principalId')
  if (humanPrincipalId !== humanPrincipalId.trim()) throw new Error('Grant confirmation human principal must be trimmed.')
  const scopeRecord = asExactObject(record.scope, ['campaignId', 'workspaceId'], 'Grant confirmation scope')
  const family = publicString(record.agentFamily, 'grant_confirmation.agentFamily')
  if (!isKnownFamily(family)) throw new Error('grant_confirmation.agentFamily is unsupported.')
  const requestedCapabilities = parseCapabilities(record.requestedCapabilities, 'grant_confirmation.requestedCapabilities')
  const grantedCapabilities = parseCapabilities(record.grantedCapabilities, 'grant_confirmation.grantedCapabilities')
  if (!grantedCapabilities.every((capability) => requestedCapabilities.includes(capability))) {
    throw new Error('Grant confirmation capabilities exceed the requested authority.')
  }
  const agentPrincipalId = publicString(record.agentPrincipalId, 'grant_confirmation.agentPrincipalId')
  const receiptDigest = publicString(record.receiptDigest, 'grant_confirmation.receiptDigest')
  if (!/^sha256:[a-f0-9]{64}$/.test(receiptDigest)) throw new Error('The grant confirmation receipt digest is invalid.')
  const capabilityDigest = publicString(record.capabilityDigest, 'grant_confirmation.capabilityDigest')
  if (capabilityDigest !== capabilityBindingDigest(agentPrincipalId, requestedCapabilities, grantedCapabilities)) {
    throw new Error('The grant confirmation capability digest is invalid.')
  }
  const issuedAt = isoString(record.issuedAt, 'grant_confirmation.issuedAt')
  const expiresAt = isoString(record.expiresAt, 'grant_confirmation.expiresAt')
  if (Date.parse(issuedAt) >= Date.parse(expiresAt)) throw new Error('The human grant confirmation must be issued before it expires.')
  return deepFreeze({
    schemaVersion: 'campaign_agent_grant_confirmation.v1',
    issuer: 'campaign_authority',
    receiptId: publicString(record.receiptId, 'grant_confirmation.receiptId'),
    receiptDigest,
    humanPrincipal: {
      principalId: humanPrincipalId,
      principalType: 'human',
    },
    scope: {
      campaignId: publicString(scopeRecord.campaignId, 'grant_confirmation.scope.campaignId'),
      workspaceId: publicString(scopeRecord.workspaceId, 'grant_confirmation.scope.workspaceId'),
    },
    agentFamily: family,
    agentOrigin: publicString(record.agentOrigin, 'grant_confirmation.agentOrigin'),
    agentPrincipalId,
    sessionId: publicString(record.sessionId, 'grant_confirmation.sessionId'),
    requestedCapabilities,
    grantedCapabilities,
    capabilityDigest,
    grantRevision: publicInteger(record.grantRevision, 'grant_confirmation.grantRevision', 1),
    issuedAt,
    expiresAt,
  })
}

function parseReceipt(value: unknown): CampaignAgentPublicReceipt {
  const record = asExactObject(value, [
    'receipt_id', 'kind', 'actor_id', 'occurred_at', 'idempotency_key', 'attachment_version', 'receipt_digest',
  ], 'Campaign agent receipt')
  const kind = publicString(record.kind, 'receipt.kind')
  if (kind !== 'attach' && kind !== 'revoke') throw new Error('receipt.kind is unsupported.')
  const receiptDigest = publicString(record.receipt_digest, 'receipt.receipt_digest')
  if (!/^sha256:[a-f0-9]{64}$/.test(receiptDigest)) throw new Error('receipt.receipt_digest is invalid.')
  return {
    receiptId: publicString(record.receipt_id, 'receipt.receipt_id'),
    kind,
    actorId: publicString(record.actor_id, 'receipt.actor_id'),
    occurredAt: isoString(record.occurred_at, 'receipt.occurred_at'),
    idempotencyKey: publicString(record.idempotency_key, 'receipt.idempotency_key'),
    attachmentVersion: publicInteger(record.attachment_version, 'receipt.attachment_version', 1),
    receiptDigest,
  }
}

function parseAuditEvent(value: unknown): CampaignAgentPublicAuditEvent {
  const record = asExactObject(value, ['event_id', 'sequence', 'kind', 'occurred_at', 'message_code'], 'Campaign agent audit event')
  const kind = publicString(record.kind, 'audit.kind')
  if (!['attachment', 'revocation', 'credential', 'liveness'].includes(kind)) throw new Error('audit.kind is unsupported.')
  const messageCode = publicString(record.message_code, 'audit.message_code')
  if (!(campaignAgentAuditMessageCodes as readonly string[]).includes(messageCode)) throw new Error('audit.message_code is an unsupported fixed message code.')
  const messageCodeForKind: Record<CampaignAgentPublicAuditEvent['kind'], CampaignAgentAuditMessageCode> = {
    attachment: 'agent_attached',
    revocation: 'agent_revoked',
    credential: 'credential_renewed',
    liveness: 'liveness_changed',
  }
  if (messageCode !== messageCodeForKind[kind as CampaignAgentPublicAuditEvent['kind']]) {
    throw new Error('Audit message code does not match the classified event kind.')
  }
  return {
    eventId: publicString(record.event_id, 'audit.event_id'),
    sequence: publicInteger(record.sequence, 'audit.sequence'),
    kind: kind as CampaignAgentPublicAuditEvent['kind'],
    occurredAt: isoString(record.occurred_at, 'audit.occurred_at'),
    messageCode: messageCode as CampaignAgentAuditMessageCode,
  }
}

function parseAttachment(value: unknown, scope: CampaignAgentScope, observedAt: string): CampaignAgentPublicAttachment | null {
  if (value === null) return null
  const record = asExactObject(value, [
    'schema_version', 'attachment_id', 'attachment_version', 'status', 'requested_capabilities',
    'granted_capabilities', 'receipt_window', 'provenance', 'receipts',
  ], 'Campaign agent attachment')
  if (record.schema_version !== 'campaign_agent_public_attachment.v1') throw new Error('Unsupported campaign agent attachment schema version.')
  const status = publicString(record.status, 'attachment.status')
  if (status !== 'attached' && status !== 'revoked') throw new Error('attachment.status is unsupported.')
  const attachmentId = publicString(record.attachment_id, 'attachment.attachment_id')
  const attachmentVersion = publicInteger(record.attachment_version, 'attachment.attachment_version', 1)
  const requestedCapabilities = parseCapabilities(record.requested_capabilities, 'attachment.requested_capabilities')
  const grantedCapabilities = parseCapabilities(record.granted_capabilities, 'attachment.granted_capabilities')
  if (!grantedCapabilities.every((capability) => requestedCapabilities.includes(capability))) throw new Error('Authoritative grants exceed requested capabilities.')
  const receiptWindowRecord = asExactObject(
    record.receipt_window,
    ['from_version', 'through_version', 'has_earlier'],
    'Campaign agent receipt window',
  )
  const hasEarlier = receiptWindowRecord.has_earlier
  if (typeof hasEarlier !== 'boolean') throw new Error('receipt_window.has_earlier must be a boolean.')
  const receiptWindow = {
    fromVersion: publicInteger(receiptWindowRecord.from_version, 'receipt_window.from_version', 1),
    throughVersion: publicInteger(receiptWindowRecord.through_version, 'receipt_window.through_version', 1),
    hasEarlier,
  }

  const provenanceRecord = asExactObject(record.provenance, [
    'agent_family', 'agent_origin', 'agent_origin_status', 'session_id', 'agent_principal_id', 'attached_at', 'attached_by',
    'grant_receipt_id', 'grant_receipt_digest', 'credential_issued_at', 'credential_expires_at',
    'credential_status', 'credential_revocation_revision', 'credential_status_source', 'liveness',
    'resume_cursor', 'revoked_at', 'revoked_by',
  ], 'Campaign agent provenance')
  const family = publicString(provenanceRecord.agent_family, 'provenance.agent_family')
  if (!isKnownFamily(family)) throw new Error('provenance.agent_family is unsupported.')
  if (provenanceRecord.agent_origin_status !== 'verified') throw new Error('Brokered agent origin must be authority-verified.')
  const credentialStatus = publicString(provenanceRecord.credential_status, 'provenance.credential_status')
  if (!['active', 'expired', 'revoked'].includes(credentialStatus)) throw new Error('provenance.credential_status is unsupported.')
  const liveness = publicString(provenanceRecord.liveness, 'provenance.liveness')
  if (!['live', 'idle', 'offline', 'expired', 'revoked'].includes(liveness)) throw new Error('provenance.liveness is unsupported.')
  if (provenanceRecord.credential_status_source !== 'campaign_authority') throw new Error('Credential status must come from campaign authority.')
  const grantReceiptDigest = publicString(provenanceRecord.grant_receipt_digest, 'provenance.grant_receipt_digest')
  if (!/^sha256:[a-f0-9]{64}$/.test(grantReceiptDigest)) throw new Error('provenance.grant_receipt_digest is invalid.')
  const attachedAt = isoString(provenanceRecord.attached_at, 'provenance.attached_at')
  const attachedBy = publicString(provenanceRecord.attached_by, 'provenance.attached_by')
  const credentialIssuedAt = isoString(provenanceRecord.credential_issued_at, 'provenance.credential_issued_at')
  const credentialExpiresAt = isoString(provenanceRecord.credential_expires_at, 'provenance.credential_expires_at')
  const revocationRevision = publicInteger(provenanceRecord.credential_revocation_revision, 'provenance.credential_revocation_revision')
  const revokedAt = provenanceRecord.revoked_at === null ? null : isoString(provenanceRecord.revoked_at, 'provenance.revoked_at')
  const revokedBy = nullablePublicString(provenanceRecord.revoked_by, 'provenance.revoked_by')
  const observedEpoch = Date.parse(observedAt)
  const issuedEpoch = Date.parse(credentialIssuedAt)
  const attachedEpoch = Date.parse(attachedAt)
  const expiresEpoch = Date.parse(credentialExpiresAt)
  const hasRevocationMetadata = Boolean(revokedAt && revokedBy)
  if (Boolean(revokedAt) !== Boolean(revokedBy)) throw new Error('Revocation actor and time must be present together.')
  if (issuedEpoch > attachedEpoch || attachedEpoch > observedEpoch || issuedEpoch >= expiresEpoch || attachedEpoch >= expiresEpoch) {
    throw new Error('Credential temporal ordering must satisfy issued <= attached <= observed and attached < expiry.')
  }
  if (credentialStatus === 'active') {
    if (expiresEpoch <= observedEpoch) throw new Error('An active credential has expired.')
    if (!['live', 'idle', 'offline'].includes(liveness)) throw new Error('Active credential liveness must be live, idle, or offline.')
    if (hasRevocationMetadata) throw new Error('Active credential cannot carry revocation metadata.')
  }
  if (credentialStatus === 'expired') {
    if (expiresEpoch > observedEpoch) throw new Error('Expired credential status precedes credential expiry.')
    if (liveness !== 'expired') throw new Error('Expired credential liveness must be expired.')
    if (hasRevocationMetadata) throw new Error('Expired credential cannot carry revocation metadata.')
  }
  if (credentialStatus === 'revoked') {
    if (status !== 'revoked' || liveness !== 'revoked' || revocationRevision < 1 || !hasRevocationMetadata) {
      throw new Error('Revoked attachment requires authoritative revocation revision, status, liveness, actor, and time.')
    }
    const revokedEpoch = Date.parse(revokedAt!)
    if (revokedEpoch < attachedEpoch || revokedEpoch > observedEpoch) throw new Error('Revocation temporal ordering must be attached <= revoked <= observed.')
  }
  if (status === 'revoked' && credentialStatus !== 'revoked') throw new Error('Revoked attachment requires revoked credential status.')
  if (status === 'attached' && credentialStatus === 'revoked') throw new Error('Attached record contains inconsistent revocation authority.')

  const receipts = boundedArray(record.receipts, 'attachment.receipts', 20).map(parseReceipt)
    .sort((left, right) => left.attachmentVersion - right.attachmentVersion || left.receiptId.localeCompare(right.receiptId))
  if (receipts.length === 0) throw new Error('Attachment receipt history cannot be empty.')
  if (new Set(receipts.map((receipt) => receipt.receiptId)).size !== receipts.length) throw new Error('Attachment receipts contain duplicate IDs.')
  if (new Set(receipts.map((receipt) => receipt.idempotencyKey)).size !== receipts.length) throw new Error('Attachment receipt history contains duplicate idempotency keys.')
  for (let index = 0; index < receipts.length; index += 1) {
    const receipt = receipts[index]
    if (index === 0 && receipt.kind !== 'attach') throw new Error('Attachment receipt history must begin with attach.')
    if (receipt.attachmentVersion > attachmentVersion) throw new Error('Receipt version exceeds the current attachment version.')
    if (index > 0) {
      const prior = receipts[index - 1]
      if (receipt.attachmentVersion <= prior.attachmentVersion || Date.parse(receipt.occurredAt) <= Date.parse(prior.occurredAt)) {
        throw new Error('Attachment receipt history must be strictly monotonic.')
      }
      if (receipt.attachmentVersion !== prior.attachmentVersion + 1 || receipt.kind === prior.kind) {
        throw new Error('Attachment receipt history must use consecutive versions and alternating attach/revoke transitions.')
      }
    }
  }
  const currentReceipt = receipts[receipts.length - 1]
  const expectedCurrentKind = status === 'attached' ? 'attach' : 'revoke'
  const expectedCurrentTime = status === 'attached' ? attachedAt : revokedAt
  if (currentReceipt.attachmentVersion !== attachmentVersion) throw new Error('Current receipt must match the current attachment version.')
  if (
    receiptWindow.fromVersion !== receipts[0].attachmentVersion
    || receiptWindow.throughVersion !== currentReceipt.attachmentVersion
    || receiptWindow.throughVersion !== attachmentVersion
    || receiptWindow.hasEarlier !== (receiptWindow.fromVersion > 1)
  ) throw new Error('Receipt window metadata does not match the bounded authoritative history.')
  if (currentReceipt.kind !== expectedCurrentKind || currentReceipt.occurredAt !== expectedCurrentTime) {
    throw new Error('Current receipt does not match the authoritative attachment transition.')
  }
  if (status === 'attached' && currentReceipt.actorId !== attachedBy) {
    throw new Error('Current attach receipt actor must match provenance attachedBy.')
  }
  if (status === 'revoked' && currentReceipt.actorId !== revokedBy) {
    throw new Error('Current revoke receipt actor must match provenance revokedBy.')
  }
  const precedingReceipt = receipts.length > 1 ? receipts[receipts.length - 2] : null
  if (status === 'attached' && precedingReceipt?.kind === 'revoke'
    && issuedEpoch < Date.parse(precedingReceipt.occurredAt)) {
    throw new Error('Reattach credential must be issued at or after the preceding revoke transition.')
  }

  return {
    schemaVersion: 'campaign_agent_public_attachment.v1',
    attachmentId,
    attachmentVersion,
    status,
    scope: { ...scope },
    requestedCapabilities,
    grantedCapabilities,
    receiptWindow,
    provenance: {
      agentFamily: family,
      agentOrigin: publicString(provenanceRecord.agent_origin, 'provenance.agent_origin'),
      agentOriginStatus: 'verified',
      sessionId: publicString(provenanceRecord.session_id, 'provenance.session_id'),
      agentPrincipalId: publicString(provenanceRecord.agent_principal_id, 'provenance.agent_principal_id'),
      attachedAt,
      attachedBy,
      grantReceiptId: publicString(provenanceRecord.grant_receipt_id, 'provenance.grant_receipt_id'),
      grantReceiptDigest,
      credentialIssuedAt,
      credentialExpiresAt,
      credentialStatus: credentialStatus as CampaignAgentCredentialStatus,
      credentialRevocationRevision: revocationRevision,
      credentialStatusSource: 'campaign_authority',
      liveness: liveness as CampaignAgentLiveness,
      resumeCursor: nullablePublicString(provenanceRecord.resume_cursor, 'provenance.resume_cursor'),
      revokedAt,
      revokedBy,
    },
    receipts,
  }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const key of Reflect.ownKeys(value)) {
      deepFreeze((value as Record<PropertyKey, unknown>)[key])
    }
    Object.freeze(value)
  }
  return value
}

export function parseCampaignAgentPublicView(value: unknown): CampaignAgentPublicView {
  const record = asExactObject(value, ['schema_version', 'observed_at', 'scope', 'attachment', 'audit_events'], 'Campaign agent public view')
  if (record.schema_version !== 'campaign_agent_public_view.v1') throw new Error('Unsupported campaign agent public view schema version.')
  const observedAt = isoString(record.observed_at, 'public_view.observed_at')
  const scope = parseScope(record.scope, 'public_view.scope')
  const attachment = parseAttachment(record.attachment, scope, observedAt)
  const auditEvents = boundedArray(record.audit_events, 'public_view.audit_events', 50).map(parseAuditEvent)
  if (new Set(auditEvents.map((event) => event.eventId)).size !== auditEvents.length) throw new Error('Audit events contain duplicate IDs.')
  if (new Set(auditEvents.map((event) => event.sequence)).size !== auditEvents.length) throw new Error('Audit events must have unique sequences.')
  if (auditEvents.some((event) => Date.parse(event.occurredAt) > Date.parse(observedAt))) {
    throw new Error('Audit event time cannot occur after the public view was observed.')
  }
  auditEvents.sort((left, right) => right.sequence - left.sequence || left.eventId.localeCompare(right.eventId))
  return deepFreeze({
    [publicViewBrand]: true,
    schemaVersion: 'campaign_agent_public_view.v1',
    observedAt,
    scope,
    attachment,
    auditEvents: auditEvents.slice(0, 20),
  })
}

export function reconcileAuthoritativeAttachment(
  selectedScope: CampaignAgentScope,
  current: CampaignAgentPublicAttachment | null,
  incoming: CampaignAgentPublicAttachment,
  expectedMutation: {
    idempotencyKey: string
    kind: 'attach' | 'revoke'
    actorId: string
    baseAttachmentVersion: number | null
  },
): CampaignAgentPublicAttachment {
  const assertSelectedScope = (attachment: CampaignAgentPublicAttachment, label: 'Current' | 'Incoming') => {
    if (attachment.scope.workspaceId !== selectedScope.workspaceId) throw new Error(`${label} attachment does not match the selected workspace.`)
    if (attachment.scope.campaignId !== selectedScope.campaignId) throw new Error(`${label} attachment does not match the selected campaign.`)
  }
  const assertHistory = (attachment: CampaignAgentPublicAttachment) => {
    if (attachment.receipts.length === 0) throw new Error('Authoritative attachment receipt history cannot be empty.')
    if (new Set(attachment.receipts.map((receipt) => receipt.receiptId)).size !== attachment.receipts.length
      || new Set(attachment.receipts.map((receipt) => receipt.idempotencyKey)).size !== attachment.receipts.length) {
      throw new Error('Authoritative attachment receipt history must have unique IDs and idempotency keys.')
    }
    for (let index = 0; index < attachment.receipts.length; index += 1) {
      const receipt = attachment.receipts[index]
      if (index === 0 && receipt.kind !== 'attach') throw new Error('Authoritative attachment history must begin with attach.')
      if (index > 0) {
        const prior = attachment.receipts[index - 1]
        if (receipt.attachmentVersion <= prior.attachmentVersion || Date.parse(receipt.occurredAt) <= Date.parse(prior.occurredAt)) {
          throw new Error('Authoritative attachment receipt history must be strictly monotonic.')
        }
        if (receipt.attachmentVersion !== prior.attachmentVersion + 1 || receipt.kind === prior.kind) {
          throw new Error('Authoritative attachment history must use consecutive alternating transitions.')
        }
      }
    }
    const latest = attachment.receipts[attachment.receipts.length - 1]
    if (latest.attachmentVersion !== attachment.attachmentVersion) throw new Error('Latest receipt must match the current attachment version.')
    if ((attachment.status === 'attached' && latest.kind !== 'attach') || (attachment.status === 'revoked' && latest.kind !== 'revoke')) {
      throw new Error('Latest receipt kind does not match attachment status.')
    }
    const transitionActor = attachment.status === 'attached' ? attachment.provenance.attachedBy : attachment.provenance.revokedBy
    if (latest.actorId !== transitionActor) throw new Error('Latest receipt actor does not match authoritative transition provenance.')
  }
  assertSelectedScope(incoming, 'Incoming')
  assertHistory(incoming)
  if (current) {
    assertSelectedScope(current, 'Current')
    assertHistory(current)
  }
  const expectedReceipts = incoming.receipts.filter((receipt) => receipt.idempotencyKey === expectedMutation.idempotencyKey)
  if (expectedReceipts.length !== 1) throw new Error('Authoritative response must contain one unique expected idempotency receipt.')
  const expectedReceipt = expectedReceipts[0]
  if (expectedReceipt.kind !== expectedMutation.kind) throw new Error('Expected receipt does not match the mutation kind.')
  if (expectedReceipt.attachmentVersion !== incoming.attachmentVersion) throw new Error('Expected receipt does not represent the incoming version.')
  const expectedActor = publicString(expectedMutation.actorId, 'Expected actor')
  const authoritativeActor = expectedMutation.kind === 'attach' ? incoming.provenance.attachedBy : incoming.provenance.revokedBy
  if (expectedReceipt.actorId !== expectedActor || authoritativeActor !== expectedActor) {
    throw new Error('Expected actor does not match the authoritative mutation receipt and provenance.')
  }
  if (expectedMutation.kind === 'attach') {
    const expectedIndex = incoming.receipts.indexOf(expectedReceipt)
    const priorReceipt = expectedIndex > 0 ? incoming.receipts[expectedIndex - 1] : null
    const authoritativeBase = priorReceipt?.attachmentVersion ?? null
    if (priorReceipt && priorReceipt.kind !== 'revoke') throw new Error('Reattach must follow an authoritative revoke receipt.')
    if (expectedMutation.baseAttachmentVersion !== authoritativeBase) {
      throw new Error('Attach mutation base version does not bind the authoritative transition.')
    }
    if (!current && authoritativeBase !== null) throw new Error('Reattach requires the current revoked authoritative base.')
    if (current && incoming.attachmentVersion > current.attachmentVersion
      && (current.status !== 'revoked' || current.attachmentVersion !== authoritativeBase)) {
      throw new Error('Attach mutation base version does not match the current revoked attachment.')
    }
  } else {
    const baseVersion = expectedMutation.baseAttachmentVersion
    if (baseVersion === null || !Number.isInteger(baseVersion) || baseVersion < 1) {
      throw new Error('Revoke mutation requires a valid base version.')
    }
    const priorReceipt = incoming.receipts[incoming.receipts.length - 2]
    if (!priorReceipt || priorReceipt.attachmentVersion !== baseVersion || expectedReceipt.attachmentVersion !== baseVersion + 1) {
      throw new Error('Revoke mutation base version does not bind the authoritative transition.')
    }
    if (!current || (incoming.attachmentVersion > current.attachmentVersion
      && (current.status !== 'attached' || current.attachmentVersion !== baseVersion))) {
      throw new Error('Revoke mutation base version does not match the current authoritative attachment.')
    }
  }
  if (!current) return incoming
  if (current.attachmentId !== incoming.attachmentId) throw new Error('Authoritative response changed attachment identity.')
  if (incoming.attachmentVersion < current.attachmentVersion) throw new Error('Cannot reconcile an older attachment version.')
  if (current.status === 'revoked' && incoming.status === 'attached'
    && incoming.attachmentVersion > current.attachmentVersion) {
    const currentRevokedAt = current.provenance.revokedAt
    if (!currentRevokedAt || Date.parse(incoming.provenance.credentialIssuedAt) < Date.parse(currentRevokedAt)) {
      throw new Error('Reattach credential must be issued at or after the current revoke time.')
    }
    if (incoming.provenance.credentialRevocationRevision < current.provenance.credentialRevocationRevision) {
      throw new Error('Reattach credential revocation revision cannot rollback and must remain monotonic.')
    }
  }
  if (incoming.attachmentVersion === current.attachmentVersion) {
    if (JSON.stringify(incoming) !== JSON.stringify(current)) throw new Error('Conflicting complete authoritative attachment reused a version.')
    return current
  }
  const currentByVersion = new Map(current.receipts.map((receipt) => [receipt.attachmentVersion, receipt]))
  const overlap = incoming.receipts.filter((receipt) => currentByVersion.has(receipt.attachmentVersion))
  if (overlap.length === 0) throw new Error('Newer attachment window must overlap current authoritative history.')
  for (const receipt of overlap) {
    if (JSON.stringify(receipt) !== JSON.stringify(currentByVersion.get(receipt.attachmentVersion))) {
      throw new Error('Newer attachment version rewrote overlapping authoritative receipt history.')
    }
  }
  return incoming
}

export function isCampaignAgentMutationAllowed(freshness: CampaignAgentFreshness): boolean {
  return freshness === 'live'
}
