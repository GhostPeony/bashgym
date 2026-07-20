import assert from 'node:assert/strict'
import test from 'node:test'
import { Children, createElement, isValidElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { CampaignRecoveryPanel, type CampaignRecoveryPanelProps } from './CampaignRecoveryPanel'
import type {
  CampaignRecoveryPublicV1,
  RecoveryAction,
  RecoveryRequest
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
  idempotencyResume: `idem_${'1'.repeat(32)}`,
  idempotencyRepair: `idem_${'2'.repeat(32)}`,
  idempotencyTakeover: `idem_${'3'.repeat(32)}`
}

function recovery(overrides: Partial<CampaignRecoveryPublicV1> = {}): CampaignRecoveryPublicV1 {
  const scope = {
    workspace_id: 'workspace-a',
    campaign_id: 'autoresearch-smoke-1',
    campaign_revision: 12,
    event_cursor: 453,
    aggregate_version: 19
  }
  const doctor = {
    evidence_id: REF.evidence,
    evidence_digest: DIGEST_A,
    checks: [
      { check: 'binding_verified', status: 'pass' },
      { check: 'checkpoint_available', status: 'pass' },
      { check: 'compute_reachable', status: 'pass' },
      { check: 'schema_compatible', status: 'pass' },
      { check: 'lease_observed', status: 'pass' }
    ]
  } as CampaignRecoveryPublicV1['doctor']
  return {
    schema_version: 'campaign_recovery.v1',
    workspace_id: scope.workspace_id,
    campaign_id: scope.campaign_id,
    installation: { installation_id: REF.installation, lineage_mode: 'clone' },
    bindings: {
      model_id: REF.model,
      model_display_label: 'Operator-selected model',
      data_id: REF.data,
      evaluator_id: REF.evaluator,
      compute_id: REF.compute
    },
    lineage: {
      checkpoint_id: REF.checkpoint,
      artifact_id: REF.artifact,
      parent_campaign_id: null,
      campaign_revision: 12,
      event_cursor: 453,
      aggregate_version: 19,
      schema_version: REF.schema
    },
    controller: { state: 'owned', owner_status: 'current_controller', lease_id: REF.lease },
    compute: { availability: 'reachable', integration_label: null },
    artifacts: { checkpoint: 'available', artifact: 'available' },
    compatibility: { schema: 'compatible', revision: 'compatible' },
    doctor,
    receipts: [
      {
        receipt_id: REF.doctorReceipt,
        kind: 'doctor',
        ...scope,
        emitted_at: '2026-07-16T20:00:00Z',
        digest: DIGEST_A
      },
      {
        receipt_id: REF.eligibilityReceipt,
        kind: 'eligibility',
        ...scope,
        emitted_at: '2026-07-16T20:15:00Z',
        digest: DIGEST_B
      }
    ],
    eligibility: {
      receipt_id: REF.eligibilityReceipt,
      receipt_digest: DIGEST_B,
      decision: 'eligible',
      allowed_actions: ['resume', 'repair'],
      ...scope,
      doctor_evidence_id: doctor.evidence_id,
      doctor_evidence_digest: doctor.evidence_digest,
      issued_at: '2026-07-16T20:15:00Z',
      expires_at: '2026-07-16T21:15:00Z'
    },
    execution_consumer: { supported: true, ready: true, reason_code: 'ready' },
    latest_execution: null,
    consumed_idempotency_keys: [],
    ...overrides
  }
}

function props(overrides: Partial<CampaignRecoveryPanelProps> = {}): CampaignRecoveryPanelProps {
  return {
    snapshot: recovery(),
    freshness: 'live',
    now: NOW,
    confirmations: { resume: false, repair: false, takeover: false },
    idempotencyKeys: {
      resume: REF.idempotencyResume,
      repair: REF.idempotencyRepair,
      takeover: REF.idempotencyTakeover
    },
    onConfirmationChange: () => {},
    onRequest: () => {},
    ...overrides
  }
}

function render(overrides: Partial<CampaignRecoveryPanelProps> = {}) {
  return renderToStaticMarkup(createElement(CampaignRecoveryPanel, props(overrides)))
}

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (!isValidElement<{ children?: ReactNode }>(node)) return ''
  return Children.toArray(node.props.children).map(textOf).join('')
}

function click(root: ReactNode, label: string): void {
  if (!isValidElement<{ children?: ReactNode; onClick?: () => void }>(root)) return
  if (typeof root.props.onClick === 'function' && textOf(root).includes(label)) {
    root.props.onClick()
    return
  }
  for (const child of Children.toArray(root.props.children)) click(child, label)
}

test('renders validated workspace scope, operator model binding, authoritative lineage, doctor evidence, and the actual latest receipt', () => {
  const html = render()
  assert.match(html, /Workspace workspace-a · Campaign autoresearch-smoke-1/)
  assert.match(html, /Operator-selected model/)
  assert.match(html, new RegExp(REF.model))
  assert.match(html, /Revision 12 · Event cursor 453 · Aggregate 19/)
  assert.match(html, new RegExp(`Checkpoint ${REF.checkpoint} · Artifact ${REF.artifact}`))
  assert.match(html, /Doctor evidence/)
  assert.match(html, new RegExp(`Latest recovery receipt: ${REF.eligibilityReceipt}`))
  assert.match(html, /permit an explicit recovery request/)
})

test('renders a registered model identity when its optional public display label is absent', () => {
  const source = recovery()
  const html = render({
    snapshot: { ...source, bindings: { ...source.bindings, model_display_label: null } }
  })
  assert.match(html, new RegExp(REF.model))
  assert.doesNotMatch(html, /Recovery projection unavailable/)
  assert.match(html, /permit an explicit recovery request/)
})

test('renders the latest worker-consumed lifecycle and disables duplicate requests while execution is pending', () => {
  const source = recovery()
  const html = render({
    snapshot: {
      ...source,
      latest_execution: {
        schema_version: 'campaign_recovery_execution.v1',
        workspace_id: source.workspace_id,
        campaign_id: source.campaign_id,
        action: 'repair',
        status: 'executing',
        outcome_code: null,
        attempt_id: null
      }
    }
  })
  assert.match(html, /Latest execution/)
  assert.match(html, /Repair · Executing/)
  assert.match(html, /resident recovery worker is processing this request/i)
  assert.match(html, /disabled=""/)
})

test('keeps recovery controls visible-disabled with a connection reason when no backend projection is available', () => {
  const html = render({ snapshot: null, freshness: 'offline' })
  assert.match(html, /Campaign recovery/)
  assert.match(html, /Recovery projection unavailable/)
  assert.match(html, /Resume campaign/)
  assert.match(html, /Request repair/)
  assert.match(html, /Take over expired lease/)
  assert.match(html, /disabled=""/)
})

test('keeps the safe shell visible and all actions disabled for every non-live freshness state', () => {
  for (const freshness of ['reconciling', 'stale', 'offline', 'error'] as const) {
    const html = render({ freshness })
    assert.match(html, /Campaign recovery/, freshness)
    assert.match(html, new RegExp(REF.checkpoint), freshness)
    assert.match(html, /disabled=""/, freshness)
  }
})

test('parses unknown panel input and withholds every unsafe value in a fully visible invalid read-only shell', () => {
  const unsafe = {
    ...recovery(),
    bindings: {
      ...recovery().bindings,
      model_id: 'mdl_ghp_visible_secret',
      compute_id: 'cpu_cades-macbook'
    },
    ssh_private_key: 'ghp_visible_secret'
  }
  const html = render({ snapshot: unsafe })
  assert.match(html, /Recovery projection unavailable/)
  assert.match(html, /Read-only/)
  assert.match(html, /Immutable bindings unavailable/)
  assert.match(html, /disabled=""/)
  assert.doesNotMatch(html, /mdl_ghp_visible_secret|cpu_cades-macbook|ghp_visible_secret/)
})

test('withholds unsafe model display labels at the panel boundary', () => {
  for (const label of [
    'ghp_0123456789secret',
    'sk-proj-0123456789secret',
    'sk_live_0123456789secret',
    'AKIAIOSFODNN7EXAMPLE',
    'xoxb-0123456789-secret',
    'AIza0123456789abcdefghijklmnop',
    'ssh-private-key',
    'BEGIN PRIVATE KEY',
    'C:\\Users\\Cade',
    'ssh://private-host',
    'cades-macbook',
    'dev-laptop',
    'home-workstation'
  ]) {
    const html = render({
      snapshot: recovery({ bindings: { ...recovery().bindings, model_display_label: label } })
    })
    assert.match(html, /Recovery projection unavailable/, label)
    assert.equal(html.includes(label), false, label)
    assert.match(html, /disabled=""/, label)
  }
})

test('withholds embedded credential tokens and collapsed device names from panel rendering', () => {
  for (const label of [
    'Qwen ghp_0123456789secret',
    'Gemma sk-proj-0123456789secret',
    'Model AKIAIOSFODNN7EXAMPLE',
    'Qwen xoxb-0123456789-secret',
    'Model AIza0123456789abcdefghijklmnop',
    'CadesMacBookPro',
    'CadesLaptop',
    'HomeWorkstation'
  ]) {
    const html = render({
      snapshot: recovery({ bindings: { ...recovery().bindings, model_display_label: label } })
    })
    assert.match(html, /Recovery projection unavailable/, label)
    assert.equal(html.includes(label), false, label)
    assert.match(html, /disabled=""/, label)
  }
})

test('renders an explicit no-action blocker and never green success copy for empty or controller-incompatible actions', () => {
  for (const allowedActions of [[], ['takeover'] as RecoveryAction[]]) {
    const base = recovery()
    const html = render({
      snapshot: recovery({ eligibility: { ...base.eligibility!, allowed_actions: allowedActions } })
    })
    assert.match(
      html,
      /No recovery action in the current eligibility receipt is compatible with the controller state/
    )
    assert.doesNotMatch(html, /permit an explicit recovery request/)
    assert.match(html, /disabled=""/)
  }
})

test('invokes repair only after controlled confirmation and emits the exact scope-bound request', () => {
  const requests: RecoveryRequest[] = []
  const unconfirmed = CampaignRecoveryPanel(
    props({ onRequest: (request) => requests.push(request) })
  )
  click(unconfirmed, 'Request repair')
  assert.equal(requests.length, 0)
  const confirmed = CampaignRecoveryPanel(
    props({
      confirmations: { resume: false, repair: true, takeover: false },
      onRequest: (request) => requests.push(request)
    })
  )
  click(confirmed, 'Request repair')
  assert.deepEqual(requests, [
    {
      action: 'repair',
      idempotencyKey: REF.idempotencyRepair,
      workspaceId: 'workspace-a',
      campaignId: 'autoresearch-smoke-1',
      eligibilityReceiptId: REF.eligibilityReceipt,
      doctorEvidenceId: REF.evidence,
      expectedCampaignRevision: 12,
      expectedEventCursor: 453,
      expectedAggregateVersion: 19,
      expectedControllerLeaseId: REF.lease,
      checkpointId: REF.checkpoint,
      artifactId: REF.artifact,
      humanConfirmed: true
    }
  ])
})

test('keeps confirmed takeover disabled when the backend cannot prove a ready recovery consumer', () => {
  const base = recovery()
  const eligibility = { ...base.eligibility!, allowed_actions: ['takeover'] as RecoveryAction[] }
  const requests: RecoveryRequest[] = []
  const expired = recovery({
    controller: { state: 'expired', owner_status: 'lease_expired', lease_id: REF.expiredLease },
    eligibility,
    execution_consumer: { supported: true, ready: false, reason_code: 'controller_lease_expired' }
  })
  click(
    CampaignRecoveryPanel(
      props({
        snapshot: expired,
        confirmations: { resume: false, repair: false, takeover: true },
        onRequest: (request) => requests.push(request)
      })
    ),
    'Take over expired lease'
  )
  assert.equal(requests.length, 0)
  const foreign = recovery({
    controller: {
      state: 'foreign_lease',
      owner_status: 'foreign_controller',
      lease_id: REF.foreignLease
    },
    eligibility,
    execution_consumer: { supported: true, ready: false, reason_code: 'foreign_controller' }
  })
  click(
    CampaignRecoveryPanel(
      props({
        snapshot: foreign,
        confirmations: { resume: false, repair: false, takeover: true },
        onRequest: (request) => requests.push(request)
      })
    ),
    'Take over expired lease'
  )
  assert.equal(requests.length, 0)
})
