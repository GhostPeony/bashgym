import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildAttachRequest,
  buildRevokeRequest,
  campaignAgentCapabilities,
  capabilityBindingDigest,
  isCampaignAgentMutationAllowed,
  isProhibitedCampaignAgentAction,
  parseCampaignAgentPublicView,
  parseCampaignAgentGrantConfirmation,
  reconcileAuthoritativeAttachment,
  validateAttachDraft,
  type CampaignAgentAttachDraft,
  type CampaignAgentGrantConfirmationReceipt,
  type CampaignAgentScope,
} from './campaignAgentModel'

const selectedScope: CampaignAgentScope = { campaignId: 'campaign-a', workspaceId: 'workspace-a' }
const now = '2026-07-16T20:00:00Z'

function confirmationFor(
  input: Omit<CampaignAgentAttachDraft, 'confirmation'>,
  overrides: Partial<CampaignAgentGrantConfirmationReceipt> = {},
): CampaignAgentGrantConfirmationReceipt {
  const requestedCapabilities = [...input.requestedCapabilities].sort()
  const grantedCapabilities = [...input.grantedCapabilities].sort()
  return {
    schemaVersion: 'campaign_agent_grant_confirmation.v1',
    issuer: 'campaign_authority',
    receiptId: 'grant-receipt-7',
    receiptDigest: `sha256:${'a'.repeat(64)}`,
    humanPrincipal: { principalId: 'human-operator-7', principalType: 'human' },
    scope: { ...input.scope },
    agentFamily: input.agentFamily,
    agentOrigin: input.agentOrigin,
    agentPrincipalId: input.agentPrincipalId,
    sessionId: input.sessionId,
    requestedCapabilities,
    grantedCapabilities,
    capabilityDigest: capabilityBindingDigest(input.agentPrincipalId, requestedCapabilities, grantedCapabilities),
    grantRevision: 7,
    issuedAt: '2026-07-16T19:55:00Z',
    expiresAt: '2026-07-16T20:30:00Z',
    ...overrides,
  }
}

function createDraft(overrides: Partial<CampaignAgentAttachDraft> = {}): CampaignAgentAttachDraft {
  const base: Omit<CampaignAgentAttachDraft, 'confirmation'> = {
    scope: selectedScope,
    agentFamily: 'codex',
    agentOrigin: 'broker:codex-local-01',
    sessionId: 'session-42',
    agentPrincipalId: 'agent-principal-42',
    requestedCapabilities: ['campaign_observe', 'training_launch'],
    grantedCapabilities: ['campaign_observe'],
    idempotencyKey: 'attach-campaign-a-01',
  }
  const merged = { ...base, ...overrides }
  return {
    ...merged,
    confirmation: overrides.confirmation === undefined ? confirmationFor(merged) : overrides.confirmation,
  }
}

function publicPayload(): Record<string, unknown> {
  return {
    schema_version: 'campaign_agent_public_view.v1',
    observed_at: '2026-07-16T20:01:00Z',
    scope: { workspace_id: 'workspace-a', campaign_id: 'campaign-a' },
    attachment: {
      schema_version: 'campaign_agent_public_attachment.v1',
      attachment_id: 'attachment-1',
      attachment_version: 3,
      status: 'attached',
      requested_capabilities: ['campaign_observe', 'training_launch'],
      granted_capabilities: ['campaign_observe'],
      receipt_window: { from_version: 3, through_version: 3, has_earlier: true },
      provenance: {
        agent_family: 'codex',
        agent_origin: 'broker:codex-local-01',
        agent_origin_status: 'verified',
        session_id: 'session-42',
        agent_principal_id: 'agent-principal-42',
        attached_at: '2026-07-16T20:00:30Z',
        attached_by: 'human-operator-7',
        grant_receipt_id: 'grant-receipt-7',
        grant_receipt_digest: `sha256:${'a'.repeat(64)}`,
        credential_issued_at: '2026-07-16T20:00:30Z',
        credential_expires_at: '2026-07-16T21:00:30Z',
        credential_status: 'active',
        credential_revocation_revision: 3,
        credential_status_source: 'campaign_authority',
        liveness: 'live',
        resume_cursor: 'cursor-19',
        revoked_at: null,
        revoked_by: null,
      },
      receipts: [{
        receipt_id: 'attach-receipt-1', kind: 'attach', actor_id: 'human-operator-7',
        occurred_at: '2026-07-16T20:00:30Z', idempotency_key: 'attach-campaign-a-01',
        attachment_version: 3, receipt_digest: `sha256:${'b'.repeat(64)}`,
      }],
    },
    audit_events: [{
      event_id: 'event-1', sequence: 19, kind: 'attachment',
      occurred_at: '2026-07-16T20:00:30Z', message_code: 'agent_attached',
    }],
  }
}

function attachmentPayload(payload: Record<string, unknown>): Record<string, unknown> {
  return payload.attachment as Record<string, unknown>
}

function provenancePayload(payload: Record<string, unknown>): Record<string, unknown> {
  return attachmentPayload(payload).provenance as Record<string, unknown>
}

function receiptWindowPayload(payload: Record<string, unknown>): Record<string, unknown> {
  return attachmentPayload(payload).receipt_window as Record<string, unknown>
}

function revokedPublicPayload(): Record<string, unknown> {
  const payload = publicPayload()
  attachmentPayload(payload).status = 'revoked'
  attachmentPayload(payload).attachment_version = 4
  receiptWindowPayload(payload).through_version = 4
  provenancePayload(payload).credential_status = 'revoked'
  provenancePayload(payload).credential_revocation_revision = 4
  provenancePayload(payload).liveness = 'revoked'
  provenancePayload(payload).revoked_at = '2026-07-16T20:00:50Z'
  provenancePayload(payload).revoked_by = 'human-operator-8'
  ;(attachmentPayload(payload).receipts as Array<Record<string, unknown>>).push({
    receipt_id: 'revoke-receipt-1', kind: 'revoke', actor_id: 'human-operator-8',
    occurred_at: '2026-07-16T20:00:50Z', idempotency_key: 'revoke-attachment-1',
    attachment_version: 4, receipt_digest: `sha256:${'c'.repeat(64)}`,
  })
  return payload
}

function reattachedPublicPayload(): Record<string, unknown> {
  const payload = revokedPublicPayload()
  attachmentPayload(payload).status = 'attached'
  attachmentPayload(payload).attachment_version = 5
  receiptWindowPayload(payload).through_version = 5
  provenancePayload(payload).attached_at = '2026-07-16T20:00:55Z'
  provenancePayload(payload).attached_by = 'human-operator-9'
  provenancePayload(payload).grant_receipt_id = 'grant-receipt-9'
  provenancePayload(payload).grant_receipt_digest = `sha256:${'d'.repeat(64)}`
  provenancePayload(payload).credential_issued_at = '2026-07-16T20:00:55Z'
  provenancePayload(payload).credential_expires_at = '2026-07-16T21:00:55Z'
  provenancePayload(payload).credential_status = 'active'
  provenancePayload(payload).credential_revocation_revision = 5
  provenancePayload(payload).liveness = 'live'
  provenancePayload(payload).resume_cursor = 'cursor-20'
  provenancePayload(payload).revoked_at = null
  provenancePayload(payload).revoked_by = null
  ;(attachmentPayload(payload).receipts as Array<Record<string, unknown>>).push({
    receipt_id: 'reattach-receipt-1', kind: 'attach', actor_id: 'human-operator-9',
    occurred_at: '2026-07-16T20:00:55Z', idempotency_key: 'reattach-campaign-a-01',
    attachment_version: 5, receipt_digest: `sha256:${'e'.repeat(64)}`,
  })
  return payload
}

test('uses explicit least-privilege capabilities and rejects every prohibited authority action', () => {
  assert.equal(campaignAgentCapabilities.includes('campaign_write' as never), false)
  for (const action of [
    'human_work_approve', 'budget_change', 'promotion_override', 'publication_publish', 'campaign_force_stop',
  ]) {
    assert.equal(isProhibitedCampaignAgentAction(action), true, action)
    assert.match(validateAttachDraft(createDraft({ requestedCapabilities: [action as never] }), selectedScope, now)[0], /prohibited|supported/i)
  }
})

test('requires a current server grant receipt bound to a distinct human and exact draft authority', () => {
  assert.deepEqual(validateAttachDraft(createDraft(), selectedScope, now), [])
  const mutations: Partial<CampaignAgentAttachDraft>[] = [
    { scope: { ...selectedScope, campaignId: 'campaign-b' } },
    { scope: { ...selectedScope, workspaceId: 'workspace-b' } },
    { agentFamily: 'hermes' },
    { agentOrigin: 'broker:codex-local-02' },
    { sessionId: 'session-43' },
    { agentPrincipalId: 'agent-principal-43' },
    { requestedCapabilities: ['campaign_observe', 'artifact_read'] },
    { grantedCapabilities: ['campaign_observe', 'training_launch'] },
  ]
  for (const mutation of mutations) {
    assert.match(validateAttachDraft({ ...createDraft(), ...mutation }, selectedScope, now).join(' '), /selected campaign|selected workspace|current draft|subset/i)
  }
  const samePrincipal = createDraft()
  samePrincipal.confirmation = confirmationFor(samePrincipal, {
    humanPrincipal: { principalId: samePrincipal.agentPrincipalId, principalType: 'human' },
  })
  assert.match(validateAttachDraft(samePrincipal, selectedScope, now).join(' '), /distinct human/i)
  const expired = createDraft()
  expired.confirmation = confirmationFor(expired, { expiresAt: now })
  assert.match(validateAttachDraft(expired, selectedScope, now).join(' '), /expired/i)
})

test('rejects principal-only substitution that retains a receipt issued for another agent', () => {
  const original = createDraft()
  const substituted = { ...original, agentPrincipalId: 'agent-principal-substitute' }
  assert.match(validateAttachDraft(substituted, selectedScope, now).join(' '), /exact current draft/i)
  assert.throws(() => buildAttachRequest(selectedScope, substituted, now, null), /exact current draft/i)
})

test('runtime-validates the complete server grant receipt and requires a public distinct human identity', () => {
  const valid = createDraft().confirmation!
  assert.equal(parseCampaignAgentGrantConfirmation(valid).humanPrincipal.principalId, 'human-operator-7')
  for (const principalId of ['', '   ', ' human-operator-7 ', 'h'.repeat(129), '../private/human', '//server/private/human']) {
    const draft = createDraft()
    draft.confirmation = { ...valid, humanPrincipal: { principalId, principalType: 'human' } }
    assert.match(validateAttachDraft(draft, selectedScope, now).join(' '), /human principal|public value|between 1 and 128/i, JSON.stringify(principalId))
  }
  const unknownField = { ...valid, unclassifiedNote: 'not public' }
  assert.throws(() => parseCampaignAgentGrantConfirmation(unknownField), /exact public keys/i)
  const malformedNested = { ...valid, humanPrincipal: 'human-operator-7' }
  assert.throws(() => parseCampaignAgentGrantConfirmation(malformedNested), /human principal.*object/i)
})

test('binds attach construction to the authoritative selected scope and blocks a second live attachment', () => {
  const draft = createDraft()
  const request = buildAttachRequest(selectedScope, draft, now, null)
  assert.deepEqual(request.scope, selectedScope)
  assert.equal(request.confirmationReceipt.receiptId, 'grant-receipt-7')
  assert.throws(() => buildAttachRequest({ ...selectedScope, campaignId: 'campaign-b' }, draft, now, null), /selected campaign/i)
  assert.throws(() => buildAttachRequest({ ...selectedScope, workspaceId: 'workspace-b' }, draft, now, null), /selected workspace/i)
  const liveAttachment = parseCampaignAgentPublicView(publicPayload()).attachment
  assert.ok(liveAttachment)
  assert.throws(() => buildAttachRequest(selectedScope, draft, now, liveAttachment), /already attached/i)
})

test('parses only the exact versioned public projection and preserves brokered credential bounds', () => {
  const view = parseCampaignAgentPublicView(publicPayload())
  assert.equal(view.schemaVersion, 'campaign_agent_public_view.v1')
  assert.equal(view.attachment?.provenance.agentOrigin, 'broker:codex-local-01')
  assert.equal(view.attachment?.provenance.agentOriginStatus, 'verified')
  assert.equal(view.attachment?.provenance.credentialExpiresAt, '2026-07-16T21:00:30Z')
  assert.equal(view.attachment?.provenance.credentialRevocationRevision, 3)
  assert.equal(view.attachment?.provenance.credentialStatus, 'active')
  assert.equal(view.attachment?.provenance.liveness, 'live')
  assert.equal(view.attachment?.provenance.resumeCursor, 'cursor-19')

  const unknownTopLevel = publicPayload()
  unknownTopLevel.display_note = 'innocent looking but unclassified'
  assert.throws(() => parseCampaignAgentPublicView(unknownTopLevel), /exact public keys/i)
  const unknownNested = publicPayload()
  provenancePayload(unknownNested).display_note = 'unclassified nested value'
  assert.throws(() => parseCampaignAgentPublicView(unknownNested), /exact public keys/i)
})

test('deep-freezes the complete branded public projection after validation', () => {
  const view = parseCampaignAgentPublicView(publicPayload())
  assert.equal(Object.isFrozen(view), true)
  assert.equal(Object.isFrozen(view.scope), true)
  assert.equal(Object.isFrozen(view.auditEvents), true)
  assert.equal(Object.isFrozen(view.auditEvents[0]), true)
  assert.equal(Object.isFrozen(view.attachment), true)
  assert.equal(Object.isFrozen(view.attachment?.scope), true)
  assert.equal(Object.isFrozen(view.attachment?.provenance), true)
  assert.equal(Object.isFrozen(view.attachment?.receipts), true)
  assert.equal(Object.isFrozen(view.attachment?.receipts[0]), true)
  assert.throws(() => {
    const mutable = view.auditEvents as unknown as Array<{ messageCode: string }>
    mutable[0].messageCode = 'private_arbitrary_text'
  }, /read only|readonly|frozen|assign/i)
  assert.equal(view.auditEvents[0].messageCode, 'agent_attached')
})

test('rejects absolute paths and high-confidence credential canaries after bounded decoding', () => {
  for (const canary of [
    'Read C:\\Users\\private\\credentials.json',
    'Fetched https://internal.example/private',
    'Authorization Bearer abc.def.ghi',
    'api_key=sk-1234567890',
    'protected_row_id=row-7734',
    '../private/model.json',
    '..\\private\\model.json',
    '.ssh/id_ed25519',
    '\\\\server\\share\\private\\model.json',
    'https%3A%2F%2Finternal.example%2Fprivate',
    '..%2Fprivate%2Fmodel.json',
    'artifact=../private/key.pem',
    '//server/share/private/key.pem',
    '.%2e%2fprivate%2fkey.pem',
    '%2e%2e%5cprivate%5ckey.pem',
    'file:%2f%2fserver%2fshare%2fkey.pem',
    '%252e%252e%252fprivate%252fkey.pem',
    '%25252e%25252e%25252fprivate%25252fkey.pem',
    'artifact=%2Gprivate/key.pem',
    'artifact=/home/cade/private/key.pem',
    'artifact=%2fhome%2fcade%2fprivate%2fkey.pem',
    'artifact=/opt/private/key.pem',
    `ghp_${'a'.repeat(36)}`,
    `github_pat_${'a'.repeat(40)}`,
    'AKIAIOSFODNN7EXAMPLE',
    '-----BEGIN PRIVATE KEY-----',
    `ghp%5f${'a'.repeat(36)}`,
    '%2d%2d%2d%2d%2dBEGIN%20PRIVATE%20KEY%2d%2d%2d%2d%2d',
  ]) {
    const payload = publicPayload()
    provenancePayload(payload).resume_cursor = canary
    assert.throws(() => parseCampaignAgentPublicView(payload), /sensitive public value/i)
  }
  const nested = publicPayload()
  provenancePayload(nested).resume_cursor = { text: 'nested' }
  assert.throws(() => parseCampaignAgentPublicView(nested), /must be a string/i)
  const oversized = publicPayload()
  provenancePayload(oversized).resume_cursor = 'x'.repeat(129)
  assert.throws(() => parseCampaignAgentPublicView(oversized), /128 characters/i)
  const tooMany = publicPayload()
  tooMany.audit_events = Array.from({ length: 51 }, (_, sequence) => ({
    event_id: `event-${sequence}`, sequence, kind: 'liveness', occurred_at: '2026-07-16T20:00:30Z', message_code: 'liveness_changed',
  }))
  assert.throws(() => parseCampaignAgentPublicView(tooMany), /at most 50/i)

  for (const cursor of ['cursor/tokenization-19', 'private-set-metric-7', 'training-95-percent']) {
    const ordinary = publicPayload()
    provenancePayload(ordinary).resume_cursor = cursor
    assert.equal(parseCampaignAgentPublicView(ordinary).attachment?.provenance.resumeCursor, cursor)
  }
})

test('accepts only fixed audit message codes with unique observed chronology', () => {
  const legacyText = publicPayload()
  const legacyEvent = (legacyText.audit_events as Array<Record<string, unknown>>)[0]
  legacyEvent.summary = 'arbitrary server text'
  assert.throws(() => parseCampaignAgentPublicView(legacyText), /exact public keys/i)

  const unknownCode = publicPayload()
  ;(unknownCode.audit_events as Array<Record<string, unknown>>)[0].message_code = 'artifact=/private/leak'
  assert.throws(() => parseCampaignAgentPublicView(unknownCode), /message code|sensitive public value/i)

  const mismatchedCode = publicPayload()
  ;(mismatchedCode.audit_events as Array<Record<string, unknown>>)[0].message_code = 'liveness_changed'
  assert.throws(() => parseCampaignAgentPublicView(mismatchedCode), /message code.*kind|kind.*message code/i)

  const duplicateSequence = publicPayload()
  ;(duplicateSequence.audit_events as Array<Record<string, unknown>>).push({
    event_id: 'event-2', sequence: 19, kind: 'liveness', occurred_at: '2026-07-16T20:00:40Z', message_code: 'liveness_changed',
  })
  assert.throws(() => parseCampaignAgentPublicView(duplicateSequence), /duplicate.*sequence|unique.*sequence/i)

  const futureEvent = publicPayload()
  ;(futureEvent.audit_events as Array<Record<string, unknown>>)[0].occurred_at = '2026-07-16T20:01:01Z'
  assert.throws(() => parseCampaignAgentPublicView(futureEvent), /audit.*observed|after.*observed/i)
})

test('accepts only exact canonical UTC timestamps across public and grant inputs', () => {
  for (const timestamp of [
    '2026-7-16T20:01:00Z',
    '2026-07-16 20:01:00',
    'July 16, 2026 20:01:00 UTC',
    '2026-02-30T20:01:00Z',
    '2026-07-16T20:01:00+00:00',
  ]) {
    const payload = publicPayload()
    payload.observed_at = timestamp
    assert.throws(() => parseCampaignAgentPublicView(payload), /canonical UTC timestamp/i, timestamp)
  }
  assert.match(validateAttachDraft(createDraft(), selectedScope, 'not-a-date').join(' '), /canonical current time/i)
})

test('enforces confirmation issue, current-time, and expiry ordering', () => {
  const futureIssued = createDraft()
  futureIssued.confirmation = confirmationFor(futureIssued, {
    issuedAt: '2026-07-16T20:01:00Z', expiresAt: '2026-07-16T20:30:00Z',
  })
  assert.match(validateAttachDraft(futureIssued, selectedScope, now).join(' '), /not yet issued|future/i)

  const zeroWindow = createDraft()
  zeroWindow.confirmation = confirmationFor(zeroWindow, {
    issuedAt: '2026-07-16T20:00:00Z', expiresAt: '2026-07-16T20:00:00Z',
  })
  assert.match(validateAttachDraft(zeroWindow, selectedScope, now).join(' '), /issued before it expires/i)

  const issuedNow = createDraft()
  issuedNow.confirmation = confirmationFor(issuedNow, { issuedAt: now })
  assert.deepEqual(validateAttachDraft(issuedNow, selectedScope, now), [])
})

test('enforces the closed credential, liveness, revocation, and temporal matrix', () => {
  const invalidCases: Array<{ name: string; mutate: (payload: Record<string, unknown>) => void }> = [
    { name: 'active-revoked-liveness', mutate: (payload) => { provenancePayload(payload).liveness = 'revoked' } },
    { name: 'expired-live-liveness', mutate: (payload) => {
      provenancePayload(payload).credential_status = 'expired'
      provenancePayload(payload).credential_expires_at = '2026-07-16T20:00:45Z'
      provenancePayload(payload).liveness = 'live'
    } },
    { name: 'expired-before-observation', mutate: (payload) => {
      provenancePayload(payload).credential_status = 'expired'
      provenancePayload(payload).credential_expires_at = '2026-07-16T21:00:30Z'
      provenancePayload(payload).liveness = 'expired'
    } },
    { name: 'issued-after-attached', mutate: (payload) => { provenancePayload(payload).credential_issued_at = '2026-07-16T20:00:40Z' } },
    { name: 'attached-after-observed', mutate: (payload) => { provenancePayload(payload).attached_at = '2026-07-16T20:02:00Z' } },
    { name: 'expiry-before-issued', mutate: (payload) => { provenancePayload(payload).credential_expires_at = '2026-07-16T20:00:00Z' } },
    { name: 'active-with-revocation-metadata', mutate: (payload) => {
      provenancePayload(payload).revoked_at = '2026-07-16T20:00:50Z'
      provenancePayload(payload).revoked_by = 'human-operator-8'
    } },
    { name: 'revoked-with-live-liveness', mutate: (payload) => {
      attachmentPayload(payload).status = 'revoked'
      provenancePayload(payload).credential_status = 'revoked'
      provenancePayload(payload).liveness = 'live'
      provenancePayload(payload).revoked_at = '2026-07-16T20:00:50Z'
      provenancePayload(payload).revoked_by = 'human-operator-8'
    } },
    { name: 'revoked-before-attached', mutate: (payload) => {
      attachmentPayload(payload).status = 'revoked'
      provenancePayload(payload).credential_status = 'revoked'
      provenancePayload(payload).liveness = 'revoked'
      provenancePayload(payload).revoked_at = '2026-07-16T20:00:00Z'
      provenancePayload(payload).revoked_by = 'human-operator-8'
    } },
    { name: 'revoked-after-observed', mutate: (payload) => {
      attachmentPayload(payload).status = 'revoked'
      provenancePayload(payload).credential_status = 'revoked'
      provenancePayload(payload).liveness = 'revoked'
      provenancePayload(payload).revoked_at = '2026-07-16T20:02:00Z'
      provenancePayload(payload).revoked_by = 'human-operator-8'
    } },
  ]
  for (const invalidCase of invalidCases) {
    const payload = invalidCase.name.startsWith('revoked') ? revokedPublicPayload() : publicPayload()
    invalidCase.mutate(payload)
    assert.throws(() => parseCampaignAgentPublicView(payload), /credential|liveness|temporal|revok|expired/i, invalidCase.name)
  }

  const expired = publicPayload()
  provenancePayload(expired).credential_status = 'expired'
  provenancePayload(expired).credential_expires_at = '2026-07-16T20:00:45Z'
  provenancePayload(expired).liveness = 'expired'
  assert.equal(parseCampaignAgentPublicView(expired).attachment?.provenance.credentialStatus, 'expired')
})

test('rejects inconsistent credential expiry and revocation authority', () => {
  const expiredActive = publicPayload()
  provenancePayload(expiredActive).credential_expires_at = '2026-07-16T20:00:45Z'
  assert.throws(() => parseCampaignAgentPublicView(expiredActive), /active credential has expired/i)

  const revoked = publicPayload()
  attachmentPayload(revoked).status = 'revoked'
  provenancePayload(revoked).credential_status = 'revoked'
  provenancePayload(revoked).credential_revocation_revision = 0
  provenancePayload(revoked).liveness = 'revoked'
  provenancePayload(revoked).revoked_at = '2026-07-16T20:02:00Z'
  provenancePayload(revoked).revoked_by = 'human-operator-8'
  assert.throws(() => parseCampaignAgentPublicView(revoked), /revocation revision/i)
})

test('reconciles duplicate attach and revoke responses by server version and idempotency receipt', () => {
  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  const attachMutation = {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach' as const,
    actorId: 'human-operator-7', baseAttachmentVersion: null,
  }
  assert.equal(reconcileAuthoritativeAttachment(selectedScope, null, attached, attachMutation), attached)
  assert.equal(reconcileAuthoritativeAttachment(selectedScope, attached, attached, attachMutation), attached)

  const revoked = parseCampaignAgentPublicView(revokedPublicPayload()).attachment!
  const revokeMutation = {
    idempotencyKey: 'revoke-attachment-1', kind: 'revoke' as const,
    actorId: 'human-operator-8', baseAttachmentVersion: 3,
  }
  const reconciled = reconcileAuthoritativeAttachment(selectedScope, attached, revoked, revokeMutation)
  assert.equal(reconciled, revoked)
  assert.equal(reconcileAuthoritativeAttachment(selectedScope, reconciled, revoked, revokeMutation), reconciled)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, revoked, attached, attachMutation), /older attachment version/i)
  assert.throws(() => buildRevokeRequest(selectedScope, revoked, {
    actorId: 'human-operator-8', idempotencyKey: 'revoke-attachment-2',
  }), /already revoked/i)
})

test('builds and reconciles one immutable attach-revoke-reattach lifecycle with exact replay', () => {
  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  const revoked = parseCampaignAgentPublicView(revokedPublicPayload()).attachment!
  const reattached = parseCampaignAgentPublicView(reattachedPublicPayload()).attachment!
  assert.equal(reattached.attachmentId, attached.attachmentId)
  assert.deepEqual(reattached.receipts.slice(0, 2), revoked.receipts)
  assert.deepEqual(reattached.receipts.map((receipt) => receipt.kind), ['attach', 'revoke', 'attach'])

  const reattachDraft = createDraft({ idempotencyKey: 'reattach-campaign-a-01' })
  reattachDraft.confirmation = confirmationFor(reattachDraft, {
    receiptId: 'grant-receipt-9',
    receiptDigest: `sha256:${'d'.repeat(64)}`,
    humanPrincipal: { principalId: 'human-operator-9', principalType: 'human' },
    grantRevision: 9,
  })
  const request = buildAttachRequest(selectedScope, reattachDraft, now, revoked)
  assert.equal(request.baseAttachmentVersion, 4)
  assert.equal(buildAttachRequest(selectedScope, createDraft(), now, null).baseAttachmentVersion, null)

  const reconciledAttached = reconcileAuthoritativeAttachment(selectedScope, null, attached, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach',
    actorId: 'human-operator-7', baseAttachmentVersion: null,
  })
  const reconciledRevoked = reconcileAuthoritativeAttachment(selectedScope, reconciledAttached, revoked, {
    idempotencyKey: 'revoke-attachment-1', kind: 'revoke',
    actorId: 'human-operator-8', baseAttachmentVersion: 3,
  })
  const mutation = {
    idempotencyKey: 'reattach-campaign-a-01', kind: 'attach' as const,
    actorId: 'human-operator-9', baseAttachmentVersion: 4,
  }
  assert.equal(reconcileAuthoritativeAttachment(selectedScope, reconciledRevoked, reattached, mutation), reattached)
  assert.equal(reconcileAuthoritativeAttachment(selectedScope, reattached, reattached, mutation), reattached)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, revoked, reattached, {
    ...mutation, baseAttachmentVersion: null,
  }), /attach.*base version/i)
})

test('requires reattach credentials to renew after revoke without revision rollback', () => {
  const issuedBeforeRevoke = reattachedPublicPayload()
  provenancePayload(issuedBeforeRevoke).credential_issued_at = '2026-07-16T20:00:45Z'
  assert.throws(() => parseCampaignAgentPublicView(issuedBeforeRevoke), /credential.*issued.*revoke|revoke.*credential.*issued/i)

  const currentRevoked = parseCampaignAgentPublicView(revokedPublicPayload()).attachment!
  const incoming = parseCampaignAgentPublicView(reattachedPublicPayload()).attachment!
  const mutation = {
    idempotencyKey: 'reattach-campaign-a-01', kind: 'attach' as const,
    actorId: 'human-operator-9', baseAttachmentVersion: 4,
  }
  const staleIssuance = {
    ...incoming,
    provenance: { ...incoming.provenance, credentialIssuedAt: '2026-07-16T20:00:45Z' },
  }
  assert.throws(
    () => reconcileAuthoritativeAttachment(selectedScope, currentRevoked, staleIssuance, mutation),
    /credential.*issued.*current revoke|current revoke.*credential.*issued/i,
  )

  const revisionRollback = {
    ...incoming,
    provenance: { ...incoming.provenance, credentialRevocationRevision: 3 },
  }
  assert.throws(
    () => reconcileAuthoritativeAttachment(selectedScope, currentRevoked, revisionRollback, mutation),
    /revocation revision.*rollback|monotonic.*revocation revision/i,
  )
})

test('reconciliation binds selected scope and exact current mutation receipt before version handling', () => {
  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  const mutation = {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach' as const,
    actorId: 'human-operator-7', baseAttachmentVersion: null,
  }
  for (const field of ['campaign_id', 'workspace_id'] as const) {
    const payload = publicPayload()
    ;(payload.scope as Record<string, unknown>)[field] = field === 'campaign_id' ? 'campaign-b' : 'workspace-b'
    const substituted = parseCampaignAgentPublicView(payload).attachment!
    assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, substituted, mutation), /incoming.*selected (campaign|workspace)/i)
    assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, substituted, attached, mutation), /current.*selected (campaign|workspace)/i)
  }

  const revoked = parseCampaignAgentPublicView(revokedPublicPayload()).attachment!
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, revoked, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach', actorId: 'human-operator-7', baseAttachmentVersion: null,
  }), /incoming version|current attachment version/i)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, revoked, {
    idempotencyKey: 'revoke-attachment-1', kind: 'attach', actorId: 'human-operator-8', baseAttachmentVersion: null,
  }), /mutation kind/i)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, revoked, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'revoke', actorId: 'human-operator-7', baseAttachmentVersion: 3,
  }), /mutation kind|incoming version/i)
})

test('reconciliation rejects incoherent history and compares the complete equal-version attachment', () => {
  const duplicateKey = revokedPublicPayload()
  ;(attachmentPayload(duplicateKey).receipts as Array<Record<string, unknown>>)[1].idempotency_key = 'attach-campaign-a-01'
  assert.throws(() => parseCampaignAgentPublicView(duplicateKey), /idempotency|receipt history/i)

  const wrongVersion = revokedPublicPayload()
  ;(attachmentPayload(wrongVersion).receipts as Array<Record<string, unknown>>)[1].attachment_version = 3
  assert.throws(() => parseCampaignAgentPublicView(wrongVersion), /monotonic|current attachment version/i)

  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  const changedPayload = publicPayload()
  provenancePayload(changedPayload).liveness = 'idle'
  const changed = parseCampaignAgentPublicView(changedPayload).attachment!
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, changed, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach', actorId: 'human-operator-7', baseAttachmentVersion: null,
  }), /complete authoritative attachment|conflicting/i)
})

test('reconciles a newer rolling receipt window by immutable overlap after lifetime compaction', () => {
  const currentPayload = revokedPublicPayload()
  ;(attachmentPayload(currentPayload).receipts as Array<Record<string, unknown>>).unshift(
    {
      receipt_id: 'attach-receipt-old', kind: 'attach', actor_id: 'human-operator-1',
      occurred_at: '2026-07-16T19:59:30Z', idempotency_key: 'attach-old',
      attachment_version: 1, receipt_digest: `sha256:${'1'.repeat(64)}`,
    },
    {
      receipt_id: 'revoke-receipt-old', kind: 'revoke', actor_id: 'human-operator-2',
      occurred_at: '2026-07-16T20:00:00Z', idempotency_key: 'revoke-old',
      attachment_version: 2, receipt_digest: `sha256:${'2'.repeat(64)}`,
    },
  )
  Object.assign(receiptWindowPayload(currentPayload), {
    from_version: 1, through_version: 4, has_earlier: false,
  })
  const current = parseCampaignAgentPublicView(currentPayload).attachment!
  const incoming = parseCampaignAgentPublicView(reattachedPublicPayload()).attachment!

  assert.equal(reconcileAuthoritativeAttachment(selectedScope, current, incoming, {
    idempotencyKey: 'reattach-campaign-a-01',
    kind: 'attach',
    actorId: 'human-operator-9',
    baseAttachmentVersion: 4,
  }), incoming)
})

test('rejects receipt window metadata that hides or invents authoritative history', () => {
  for (const mutation of [
    { from_version: 2 },
    { through_version: 4 },
    { has_earlier: false },
  ]) {
    const payload = publicPayload()
    Object.assign(receiptWindowPayload(payload), mutation)
    assert.throws(() => parseCampaignAgentPublicView(payload), /receipt window/i)
  }
})

test('binds current transition receipt actors to provenance and reconciliation intent', () => {
  const wrongAttachActor = publicPayload()
  ;(attachmentPayload(wrongAttachActor).receipts as Array<Record<string, unknown>>)[0].actor_id = 'human-impostor'
  assert.throws(() => parseCampaignAgentPublicView(wrongAttachActor), /attach receipt actor.*attachedBy/i)

  const wrongRevokeActor = revokedPublicPayload()
  ;(attachmentPayload(wrongRevokeActor).receipts as Array<Record<string, unknown>>)[1].actor_id = 'human-impostor'
  assert.throws(() => parseCampaignAgentPublicView(wrongRevokeActor), /revoke receipt actor.*revokedBy/i)

  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, null, attached, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach', actorId: 'human-impostor', baseAttachmentVersion: null,
  }), /expected actor/i)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, null, attached, {
    idempotencyKey: 'attach-campaign-a-01', kind: 'attach', actorId: 'human-operator-7', baseAttachmentVersion: 2,
  }), /attach.*base version/i)

  const revoked = parseCampaignAgentPublicView(revokedPublicPayload()).attachment!
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, revoked, {
    idempotencyKey: 'revoke-attachment-1', kind: 'revoke', actorId: 'human-impostor', baseAttachmentVersion: 3,
  }), /expected actor/i)
  assert.throws(() => reconcileAuthoritativeAttachment(selectedScope, attached, revoked, {
    idempotencyKey: 'revoke-attachment-1', kind: 'revoke', actorId: 'human-operator-8', baseAttachmentVersion: 2,
  }), /revoke.*base version/i)
})

test('builds exact-scope revoke payloads only for live authoritative attachments', () => {
  const attached = parseCampaignAgentPublicView(publicPayload()).attachment!
  const request = buildRevokeRequest(selectedScope, attached, {
    actorId: 'human-operator-8', idempotencyKey: 'revoke-attachment-1',
  })
  assert.deepEqual(request.scope, selectedScope)
  assert.equal(request.attachmentVersion, 3)
  assert.throws(() => buildRevokeRequest({ ...selectedScope, campaignId: 'campaign-b' }, attached, {
    actorId: 'human-operator-8', idempotencyKey: 'revoke-attachment-1',
  }), /selected campaign/i)
  assert.throws(() => buildRevokeRequest({ ...selectedScope, workspaceId: 'workspace-b' }, attached, {
    actorId: 'human-operator-8', idempotencyKey: 'revoke-attachment-1',
  }), /selected workspace/i)
})

test('allows mutations only in every live freshness case', () => {
  assert.equal(isCampaignAgentMutationAllowed('live'), true)
  for (const freshness of ['reconciling', 'stale', 'offline', 'error'] as const) {
    assert.equal(isCampaignAgentMutationAllowed(freshness), false)
  }
})
