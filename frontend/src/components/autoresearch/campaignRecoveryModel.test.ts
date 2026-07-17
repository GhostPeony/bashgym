import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildCampaignRecoveryModel,
  buildRecoveryRequest,
  parseCampaignRecovery,
  type CampaignRecoveryPublicV1,
  type RecoveryAction,
  type RecoveryFreshness,
} from './campaignRecoveryModel'

const DIGEST_A = `sha256_${'a'.repeat(64)}`
const DIGEST_B = `sha256_${'b'.repeat(64)}`
const NOW = '2026-07-16T20:30:00Z'
const REF = {
  installation: `ins_${'1'.repeat(32)}`,
  model: `mdl_${'2'.repeat(32)}`,
  data: `dat_${'3'.repeat(32)}`,
  evaluator: `evl_${'4'.repeat(32)}`,
  compute: `cpu_${'5'.repeat(32)}`,
  checkpoint: `ckpt_${'6'.repeat(32)}`,
  artifact: `art_${'7'.repeat(32)}`,
  schema: `sch_${'8'.repeat(32)}`,
  lease: `lease_${'9'.repeat(32)}`,
  expiredLease: `lease_${'a'.repeat(32)}`,
  foreignLease: `lease_${'b'.repeat(32)}`,
  evidence: `evd_${'c'.repeat(32)}`,
  doctorReceipt: `rcpt_${'d'.repeat(32)}`,
  eligibilityReceipt: `rcpt_${'e'.repeat(32)}`,
  otherEvidence: `evd_${'f'.repeat(32)}`,
  idempotencyUsed: `idem_${'1'.repeat(32)}`,
  idempotencyResume: `idem_${'2'.repeat(32)}`,
  idempotencyRepair: `idem_${'3'.repeat(32)}`,
  idempotencyTakeover: `idem_${'4'.repeat(32)}`,
}

function recovery(overrides: Partial<CampaignRecoveryPublicV1> = {}): CampaignRecoveryPublicV1 {
  const workspaceId = 'workspace-a'
  const campaignId = 'autoresearch-smoke-1'
  const lineage = { checkpoint_id: REF.checkpoint, artifact_id: REF.artifact, parent_campaign_id: 'autoresearch-smoke-0', campaign_revision: 12, event_cursor: 453, aggregate_version: 19, schema_version: REF.schema }
  const doctor = {
    evidence_id: REF.evidence, evidence_digest: DIGEST_A,
    checks: [
      { check: 'binding_verified', status: 'pass' },
      { check: 'checkpoint_available', status: 'pass' },
      { check: 'compute_reachable', status: 'pass' },
      { check: 'schema_compatible', status: 'pass' },
      { check: 'lease_observed', status: 'pass' },
    ],
  } as CampaignRecoveryPublicV1['doctor']
  const scope = { workspace_id: workspaceId, campaign_id: campaignId, campaign_revision: lineage.campaign_revision, event_cursor: lineage.event_cursor, aggregate_version: lineage.aggregate_version }
  return {
    schema_version: 'campaign_recovery.v1', workspace_id: workspaceId, campaign_id: campaignId,
    installation: { installation_id: REF.installation, lineage_mode: 'fork' },
    bindings: { model_id: REF.model, model_display_label: 'Operator-selected model', data_id: REF.data, evaluator_id: REF.evaluator, compute_id: REF.compute },
    lineage,
    controller: { state: 'owned', owner_status: 'current_controller', lease_id: REF.lease },
    compute: { availability: 'reachable', integration_label: null },
    artifacts: { checkpoint: 'available', artifact: 'available' },
    compatibility: { schema: 'compatible', revision: 'compatible' }, doctor,
    receipts: [
      { receipt_id: REF.doctorReceipt, kind: 'doctor', ...scope, emitted_at: '2026-07-16T20:00:00Z', digest: DIGEST_A },
      { receipt_id: REF.eligibilityReceipt, kind: 'eligibility', ...scope, emitted_at: '2026-07-16T20:15:00Z', digest: DIGEST_B },
    ],
    eligibility: { receipt_id: REF.eligibilityReceipt, receipt_digest: DIGEST_B, decision: 'eligible', allowed_actions: ['resume', 'repair'], ...scope, doctor_evidence_id: doctor.evidence_id, doctor_evidence_digest: doctor.evidence_digest, issued_at: '2026-07-16T20:15:00Z', expires_at: '2026-07-16T21:15:00Z' },
    execution_consumer: { supported: true, ready: true, reason_code: 'ready' },
    latest_execution: null,
    consumed_idempotency_keys: [REF.idempotencyUsed],
    ...overrides,
  }
}

test('accepts canonical durable workspace/campaign IDs and keeps operator model choices in a separate display label', () => {
  assert.deepEqual(parseCampaignRecovery(recovery()), recovery())
  assert.notEqual(parseCampaignRecovery(recovery({ workspace_id: 'workspace-a', campaign_id: 'autoresearch-smoke-1', bindings: { ...recovery().bindings, model_display_label: 'default' } })), null)
  const punctuated = structuredClone(recovery())
  punctuated.workspace_id = 'workspace.branch:dev_1'
  punctuated.campaign_id = 'autoresearch.smoke:run_1'
  for (const receipt of punctuated.receipts) {
    receipt.workspace_id = punctuated.workspace_id
    receipt.campaign_id = punctuated.campaign_id
  }
  punctuated.eligibility!.workspace_id = punctuated.workspace_id
  punctuated.eligibility!.campaign_id = punctuated.campaign_id
  assert.notEqual(parseCampaignRecovery(punctuated), null)
})

test('strictly parses the bounded latest recovery execution lifecycle and binds it to projection scope', () => {
  const source = recovery()
  const executing = {
    ...source,
    latest_execution: {
      schema_version: 'campaign_recovery_execution.v1',
      workspace_id: source.workspace_id,
      campaign_id: source.campaign_id,
      action: 'repair',
      status: 'executing',
      outcome_code: null,
      attempt_id: null,
    },
  }
  assert.deepEqual(parseCampaignRecovery(executing), executing)
  assert.notEqual(parseCampaignRecovery({ ...source, latest_execution: null }), null)

  for (const mutate of [
    (value: typeof executing) => { value.latest_execution.workspace_id = 'workspace-b' },
    (value: typeof executing) => { value.latest_execution.status = 'running' },
    (value: typeof executing) => { Object.assign(value.latest_execution, { outcome_code: 'private_host_failed' }) },
    (value: typeof executing) => { Object.assign(value.latest_execution, { worker_id: 'private-worker' }) },
  ]) {
    const value = structuredClone(executing)
    mutate(value)
    assert.equal(parseCampaignRecovery(value), null)
  }
})

test('requires current resident consumer authority and blocks duplicate in-flight recovery requests', () => {
  const ready = parseCampaignRecovery({ ...recovery(), latest_execution: null })
  assert.notEqual(ready, null)
  const readyModel = buildCampaignRecoveryModel({ snapshot: ready!, freshness: 'live', now: NOW })
  assert.equal(readyModel.consumerReady, true)
  assert.equal(readyModel.actions.resume.enabled, true)

  const unowned = parseCampaignRecovery({
    ...recovery(),
    controller: { state: 'unowned', owner_status: 'unassigned', lease_id: null },
    execution_consumer: { supported: true, ready: false, reason_code: 'controller_unowned' },
    latest_execution: null,
  })
  assert.notEqual(unowned, null)
  const unavailable = buildCampaignRecoveryModel({ snapshot: unowned!, freshness: 'live', now: NOW })
  assert.equal(unavailable.consumerReady, false)
  assert.equal(unavailable.actions.resume.enabled, false)
  assert.equal(unavailable.blockers.some((item) => item.code === 'recovery_consumer_unavailable'), true)

  const accepted = parseCampaignRecovery({
    ...recovery(),
    latest_execution: {
      schema_version: 'campaign_recovery_execution.v1',
      workspace_id: 'workspace-a',
      campaign_id: 'autoresearch-smoke-1',
      action: 'resume',
      status: 'accepted',
      outcome_code: null,
      attempt_id: null,
    },
  })
  assert.notEqual(accepted, null)
  const pending = buildCampaignRecoveryModel({ snapshot: accepted!, freshness: 'live', now: NOW })
  assert.equal(pending.consumerReady, true)
  assert.equal(pending.actions.resume.enabled, false)
  assert.equal(pending.blockers.some((item) => item.code === 'recovery_request_in_flight'), true)

  for (const execution_consumer of [
    { supported: true, ready: false, reason_code: 'ready' },
    { supported: false, ready: true, reason_code: 'controller_unowned' },
    { supported: true, ready: false, reason_code: 'private_worker_offline' },
    { supported: true, ready: true, reason_code: 'ready', worker_id: 'private-worker' },
  ]) {
    assert.equal(parseCampaignRecovery({ ...recovery(), execution_consumer }), null)
  }
  assert.equal(parseCampaignRecovery({
    ...recovery(),
    controller: { state: 'expired', owner_status: 'lease_expired', lease_id: REF.expiredLease },
  }), null)
})

test('accepts bounded public model labels without blacklisting operator model families', () => {
  for (const label of ['Operator-selected model', 'default', 'alternate model', 'optional trainer']) {
    assert.notEqual(parseCampaignRecovery(recovery({ bindings: { ...recovery().bindings, model_display_label: label } })), null, label)
  }
})

test('accepts a registered model without an optional public display label and does not report a missing binding', () => {
  const source = recovery({ bindings: { ...recovery().bindings, model_display_label: null } })
  const parsed = parseCampaignRecovery(source)
  assert.notEqual(parsed, null)
  const model = buildCampaignRecoveryModel({ snapshot: parsed!, freshness: 'live', now: NOW })
  assert.equal(model.blockers.some((item) => item.code === 'missing_binding'), false)
  assert.equal(model.mutationsEnabled, true)
})

test('rejects credential, key, path, URI, and personal-device signatures in the public model label', () => {
  const unsafeLabels = [
    'ghp_0123456789secret', 'github_pat_0123456789secret', 'sk-proj-0123456789secret', 'sk_live_0123456789secret',
    'AKIAIOSFODNN7EXAMPLE', 'xoxb-0123456789-secret', 'AIza0123456789abcdefghijklmnop', 'ssh-private-key', 'BEGIN PRIVATE KEY', 'client-secret',
    'C:\\Users\\Cade\\models', '/Users/cade/models', 'ssh://private-host', 'file:///private/model',
    'cades-macbook', 'Cade MacBook Pro', 'dev-laptop', 'home-workstation', 'localhost',
  ]
  for (const label of unsafeLabels) {
    assert.equal(parseCampaignRecovery(recovery({ bindings: { ...recovery().bindings, model_display_label: label } })), null, label)
  }
})

test('rejects embedded credential tokens and collapsed personal-device names in the public model label', () => {
  const adversarialLabels = [
    'Qwen ghp_0123456789secret', 'Gemma sk-proj-0123456789secret',
    'Model AKIAIOSFODNN7EXAMPLE', 'Qwen xoxb-0123456789-secret',
    'Model AIza0123456789abcdefghijklmnop', 'CadesMacBookPro', 'CadesLaptop', 'HomeWorkstation',
  ]
  for (const label of adversarialLabels) {
    assert.equal(parseCampaignRecovery(recovery({ bindings: { ...recovery().bindings, model_display_label: label } })), null, label)
  }
})

test('accepts only canonical UTC ISO timestamps and rejects Date.parse-normalized variants', () => {
  const canonicalMillis = structuredClone(recovery())
  canonicalMillis.receipts[0]!.emitted_at = '2026-07-16T20:00:00.123Z'
  assert.notEqual(parseCampaignRecovery(canonicalMillis), null)
  for (const timestamp of ['2026-07-16 20:00:00', '07/16/2026 20:00:00Z', '2026-07-16T20:00:00+00:00', '2026-02-30T20:00:00Z', '2026-07-16T20:00:00.1Z']) {
    const value = structuredClone(recovery())
    value.receipts[0]!.emitted_at = timestamp
    assert.equal(parseCampaignRecovery(value), null, timestamp)
  }
})

test('rejects additive secret/path fields and unprefixed values in opaque recovery fields', () => {
  const value = recovery()
  assert.equal(parseCampaignRecovery({ ...value, ssh_private_key: 'secret' }), null)
  const setters: Array<[string, (copy: CampaignRecoveryPublicV1, canary: string) => void]> = [
    ['installation', (copy, canary) => { copy.installation.installation_id = canary }],
    ['model', (copy, canary) => { copy.bindings.model_id = canary }],
    ['data', (copy, canary) => { copy.bindings.data_id = canary }],
    ['evaluator', (copy, canary) => { copy.bindings.evaluator_id = canary }],
    ['compute', (copy, canary) => { copy.bindings.compute_id = canary }],
    ['checkpoint', (copy, canary) => { copy.lineage.checkpoint_id = canary }],
    ['artifact', (copy, canary) => { copy.lineage.artifact_id = canary }],
    ['schema', (copy, canary) => { copy.lineage.schema_version = canary }],
    ['lease', (copy, canary) => { copy.controller.lease_id = canary }],
    ['doctor evidence', (copy, canary) => { copy.doctor.evidence_id = canary }],
    ['receipt', (copy, canary) => { copy.receipts[0]!.receipt_id = canary }],
  ]
  for (const [field, set] of setters) {
    for (const canary of ['ghp_0123456789secret', 'ssh://private-host', 'C:\\Users\\Cade\\.ssh', 'cades-macbook']) {
      const copy = structuredClone(value)
      set(copy, canary)
      assert.equal(parseCampaignRecovery(copy), null, `${field} accepted ${canary}`)
    }
  }
})

test('rejects credential and personal-hardware suffix bypasses for every opaque recovery reference', () => {
  const setters: Array<[string, string, (copy: CampaignRecoveryPublicV1, canary: string) => void]> = [
    ['installation', 'ins_', (copy, canary) => { copy.installation.installation_id = canary }],
    ['model', 'mdl_', (copy, canary) => { copy.bindings.model_id = canary }],
    ['data', 'dat_', (copy, canary) => { copy.bindings.data_id = canary }],
    ['evaluator', 'evl_', (copy, canary) => { copy.bindings.evaluator_id = canary }],
    ['compute', 'cpu_', (copy, canary) => { copy.bindings.compute_id = canary }],
    ['checkpoint', 'ckpt_', (copy, canary) => { copy.lineage.checkpoint_id = canary }],
    ['artifact', 'art_', (copy, canary) => { copy.lineage.artifact_id = canary }],
    ['schema', 'sch_', (copy, canary) => { copy.lineage.schema_version = canary }],
    ['lease', 'lease_', (copy, canary) => { copy.controller.lease_id = canary }],
    ['evidence', 'evd_', (copy, canary) => { copy.doctor.evidence_id = canary }],
    ['receipt', 'rcpt_', (copy, canary) => { copy.receipts[0]!.receipt_id = canary }],
  ]
  for (const [field, prefix, set] of setters) {
    for (const suffix of ['ghp_0123456789secret', 'cades-macbook', 'AKIAIOSFODNN7EXAMPLE']) {
      const copy = structuredClone(recovery())
      set(copy, `${prefix}${suffix}`)
      assert.equal(parseCampaignRecovery(copy), null, `${field} accepted ${prefix}${suffix}`)
    }
  }
})

test('rejects any historical receipt whose workspace or campaign differs from the projection root', () => {
  const wrongWorkspace = structuredClone(recovery())
  wrongWorkspace.receipts[0]!.workspace_id = 'workspace-b'
  assert.equal(parseCampaignRecovery(wrongWorkspace), null)
  const wrongCampaign = structuredClone(recovery())
  wrongCampaign.receipts[0]!.campaign_id = 'autoresearch-smoke-2'
  assert.equal(parseCampaignRecovery(wrongCampaign), null)
  const historicalRevision = structuredClone(recovery())
  historicalRevision.receipts[0]!.campaign_revision = 11
  historicalRevision.receipts[0]!.event_cursor = 400
  assert.notEqual(parseCampaignRecovery(historicalRevision), null)
})

test('parses a null public binding identity and derives the reachable missing-binding block without changing lineage', () => {
  const source = recovery({ bindings: { ...recovery().bindings, data_id: null } })
  const parsed = parseCampaignRecovery(source)
  assert.notEqual(parsed, null)
  const model = buildCampaignRecoveryModel({ snapshot: parsed!, freshness: 'live', now: NOW })
  assert.equal(model.blockers.some((item) => item.code === 'missing_binding'), true)
  assert.equal(model.mutationsEnabled, false)
  assert.deepEqual(model.lineage, source.lineage)
})

test('rejects incompatible controller state, owner, and lease combinations', () => {
  const invalid = [
    { state: 'owned', owner_status: 'unassigned', lease_id: REF.lease },
    { state: 'unowned', owner_status: 'unassigned', lease_id: REF.lease },
    { state: 'foreign_lease', owner_status: 'foreign_controller', lease_id: null },
    { state: 'expired', owner_status: 'current_controller', lease_id: REF.lease },
  ]
  for (const controller of invalid) assert.equal(parseCampaignRecovery(recovery({ controller: controller as CampaignRecoveryPublicV1['controller'] })), null)
})

test('rejects out-of-order, duplicate, and unbounded receipt or evidence arrays', () => {
  const value = recovery()
  assert.equal(parseCampaignRecovery(recovery({ receipts: [...value.receipts].reverse() })), null)
  assert.equal(parseCampaignRecovery(recovery({ receipts: [...value.receipts, value.receipts[1]!] })), null)
  assert.equal(parseCampaignRecovery(recovery({ receipts: Array.from({ length: 33 }, () => value.receipts[0]!) })), null)
  assert.equal(parseCampaignRecovery(recovery({ doctor: { ...value.doctor, checks: [...value.doctor.checks, value.doctor.checks[0]!] } })), null)
})

test('gates mutation for all five freshness states', () => {
  const expected: Array<[RecoveryFreshness, boolean, string]> = [
    ['live', true, 'resumable'], ['reconciling', false, 'degraded'], ['stale', false, 'read_only'],
    ['offline', false, 'read_only'], ['error', false, 'degraded'],
  ]
  for (const [freshness, enabled, decision] of expected) {
    const model = buildCampaignRecoveryModel({ snapshot: recovery(), freshness, now: NOW })
    assert.equal(model.mutationsEnabled, enabled, freshness)
    assert.equal(model.decision, decision, freshness)
  }
})

test('treats an invalid authority time as unverified rather than bypassing receipt expiry', () => {
  const model = buildCampaignRecoveryModel({ snapshot: recovery(), freshness: 'live', now: 'not-a-timestamp' })
  assert.equal(model.mutationsEnabled, false)
  assert.equal(model.blockers.some((item) => item.code === 'eligibility_unverified'), true)
})

test('enables recovery only from a complete, current, latest eligibility receipt bound to exact scope and doctor evidence', () => {
  const base = recovery()
  const cases: Array<[string, CampaignRecoveryPublicV1]> = [
    ['expired', recovery({ eligibility: { ...base.eligibility!, expires_at: NOW } })],
    ['wrong workspace', recovery({ eligibility: { ...base.eligibility!, workspace_id: 'workspace-b' } })],
    ['wrong campaign', recovery({ eligibility: { ...base.eligibility!, campaign_id: 'autoresearch-other-1' } })],
    ['wrong revision', recovery({ eligibility: { ...base.eligibility!, campaign_revision: 11 } })],
    ['wrong cursor', recovery({ eligibility: { ...base.eligibility!, event_cursor: 452 } })],
    ['wrong aggregate', recovery({ eligibility: { ...base.eligibility!, aggregate_version: 18 } })],
    ['wrong evidence', recovery({ eligibility: { ...base.eligibility!, doctor_evidence_id: REF.otherEvidence } })],
    ['wrong evidence digest', recovery({ eligibility: { ...base.eligibility!, doctor_evidence_digest: DIGEST_B } })],
    ['not latest', recovery({ eligibility: { ...base.eligibility!, receipt_id: REF.doctorReceipt, receipt_digest: DIGEST_A, issued_at: '2026-07-16T20:00:00Z' } })],
    ['incomplete doctor', recovery({ doctor: { ...base.doctor, checks: base.doctor.checks.slice(0, 4) } })],
  ]
  for (const [name, snapshot] of cases) {
    const model = buildCampaignRecoveryModel({ snapshot, freshness: 'live', now: NOW })
    assert.equal(model.mutationsEnabled, false, name)
    assert.equal(model.blockers.some((item) => item.code === 'eligibility_unverified'), true, name)
  }
})

test('derives an explicit blocker when current eligibility has no controller-compatible permitted action', () => {
  for (const [name, allowedActions] of [['empty', []], ['owned cannot takeover', ['takeover']]] as Array<[string, RecoveryAction[]]>) {
    const base = recovery()
    const snapshot = recovery({ eligibility: { ...base.eligibility!, allowed_actions: allowedActions } })
    const model = buildCampaignRecoveryModel({ snapshot, freshness: 'live', now: NOW })
    assert.equal(model.mutationsEnabled, false, name)
    assert.equal(model.blockers.some((item) => item.code === 'no_permitted_recovery_action'), true, name)
  }
})

test('constructs exact workspace-scoped resume and repair requests only after confirmation with a fresh unused idempotency key', () => {
  const model = buildCampaignRecoveryModel({ snapshot: recovery(), freshness: 'live', now: NOW })
  const expected = {
    workspaceId: 'workspace-a', campaignId: 'autoresearch-smoke-1', eligibilityReceiptId: REF.eligibilityReceipt, doctorEvidenceId: REF.evidence,
    expectedCampaignRevision: 12, expectedEventCursor: 453, expectedAggregateVersion: 19, expectedControllerLeaseId: REF.lease,
    checkpointId: REF.checkpoint, artifactId: REF.artifact, humanConfirmed: true as const,
  }
  for (const [action, idempotencyKey] of [['resume', REF.idempotencyResume], ['repair', REF.idempotencyRepair]] as Array<[RecoveryAction, string]>) {
    assert.equal(buildRecoveryRequest({ model, action, humanConfirmed: false, idempotencyKey }), null)
    assert.deepEqual(buildRecoveryRequest({ model, action, humanConfirmed: true, idempotencyKey }), { action, idempotencyKey, ...expected })
  }
  assert.equal(buildRecoveryRequest({ model, action: 'resume', humanConfirmed: true, idempotencyKey: 'resume-12' }), null)
  assert.equal(buildRecoveryRequest({ model, action: 'resume', humanConfirmed: true, idempotencyKey: REF.idempotencyUsed }), null)
})

test('keeps takeover visible but disabled when an expired or foreign controller cannot prove a ready consumer', () => {
  const base = recovery()
  const eligibility = { ...base.eligibility!, allowed_actions: ['takeover'] as RecoveryAction[] }
  const expired = recovery({ controller: { state: 'expired', owner_status: 'lease_expired', lease_id: REF.expiredLease }, eligibility, execution_consumer: { supported: true, ready: false, reason_code: 'controller_lease_expired' } })
  const model = buildCampaignRecoveryModel({ snapshot: expired, freshness: 'live', now: NOW })
  assert.equal(model.actions.takeover.enabled, false)
  assert.equal(buildRecoveryRequest({ model, action: 'takeover', humanConfirmed: false, idempotencyKey: REF.idempotencyTakeover }), null)
  assert.equal(buildRecoveryRequest({ model, action: 'takeover', humanConfirmed: true, idempotencyKey: REF.idempotencyTakeover }), null)
  const foreign = recovery({ controller: { state: 'foreign_lease', owner_status: 'foreign_controller', lease_id: REF.foreignLease }, eligibility, execution_consumer: { supported: true, ready: false, reason_code: 'foreign_controller' } })
  assert.equal(buildCampaignRecoveryModel({ snapshot: foreign, freshness: 'live', now: NOW }).actions.takeover.enabled, false)
})
