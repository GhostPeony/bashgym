import assert from 'node:assert/strict'
import test from 'node:test'

import { controlRoomSnapshot } from './controlRoomFixtures'
import {
  buildControlRoomModel,
  campaignStatusTone,
  presentActions,
  presentBlocker,
  presentJourney,
  presentMetrics,
  presentOwner,
  resolveControlRoomFreshness,
} from './controlRoomModel'

test('inherits workspace authority when no campaign detail is loaded', () => {
  assert.equal(resolveControlRoomFreshness({ workspaceFreshness: 'offline' }), 'offline')
  assert.equal(resolveControlRoomFreshness({ workspaceFreshness: 'stale' }), 'stale')
  assert.equal(resolveControlRoomFreshness({ workspaceLoading: true }), 'reconciling')
  assert.equal(resolveControlRoomFreshness({ workspaceError: 'bad contract' }), 'error')
  assert.equal(resolveControlRoomFreshness({ detailFreshness: 'live', workspaceFreshness: 'offline' }), 'live')
})

test('maps only known campaign statuses to semantic tones and keeps unknown values neutral', () => {
  assert.equal(campaignStatusTone('active'), 'info')
  assert.equal(campaignStatusTone('failed'), 'error')
  assert.equal(campaignStatusTone('completed'), 'success')
  assert.equal(campaignStatusTone('remote_future_status'), 'neutral')
})

test('presents the fixed journey in backend order and tolerates additive states', () => {
  const snapshot = controlRoomSnapshot()
  const phases = presentJourney([
    ...snapshot.journey.slice(0, 2),
    { ...snapshot.journey[2], state: 'new_remote_state' as never },
    ...snapshot.journey.slice(3),
  ])
  assert.deepEqual(phases.map((phase) => phase.id), [
    'setup', 'baseline', 'experiments', 'human_review', 'decision',
  ])
  assert.equal(phases[2]?.tone, 'neutral')
  assert.equal(phases[2]?.stateLabel, 'New Remote State')
})

test('presents execution and attention ownership separately', () => {
  const snapshot = controlRoomSnapshot()
  assert.deepEqual(presentOwner(snapshot.decision_surface), {
    execution: 'BashGym',
    attention: 'Agent',
  })
  assert.notEqual(presentOwner({ ...snapshot.decision_surface, attention_owner: 'agent' }).execution, 'Agent')
})

test('preserves the server primary blocker and evidence IDs', () => {
  const snapshot = controlRoomSnapshot()
  const blocker = {
    schema_version: 'decision_blocker.v1' as const,
    code: 'human_review_required',
    summary: 'A blinded review is required.',
    evidence_ids: ['evidence-2', 'evidence-1'],
    secondary_codes: ['budget_warning'],
  }
  assert.deepEqual(presentBlocker({ ...snapshot.decision_surface, blocker }), {
    code: blocker.code,
    summary: blocker.summary,
    evidenceIds: blocker.evidence_ids,
    secondaryCodes: blocker.secondary_codes,
  })
})

test('presents only server actions and keeps M0 actions disabled for every freshness', () => {
  const surface = controlRoomSnapshot().decision_surface
  for (const freshness of ['live', 'stale', 'offline', 'reconciling', 'error'] as const) {
    const actions = presentActions(surface, freshness)
    assert.deepEqual(actions.map((action) => action.id), ['inspect-active-work', 'reconcile-controller'])
    assert.equal(actions.every((action) => action.enabled === false), true)
  }
})

test('renders metric descriptors without fabricated values or deltas', () => {
  const snapshot = controlRoomSnapshot()
  const metrics = presentMetrics(snapshot.metrics, snapshot.champion, snapshot.candidate)
  assert.equal(metrics[0]?.value, 'Unavailable')
  assert.equal(metrics[0]?.delta, null)
})

test('builds distinct initial and cached load states', () => {
  assert.equal(buildControlRoomModel({ snapshot: null, freshness: 'reconciling', error: null }).kind, 'loading')
  assert.equal(buildControlRoomModel({ snapshot: null, freshness: 'error', error: 'bad contract' }).kind, 'error')
  assert.equal(buildControlRoomModel({ snapshot: null, freshness: 'offline', error: 'bridge down' }).kind, 'offline')
  assert.equal(buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'stale', error: 'timeout' }).kind, 'snapshot')
  assert.equal(buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'offline', error: 'bridge down' }).authoritative, false)
})

test('keeps controller health distinct from renderer freshness', () => {
  const snapshot = controlRoomSnapshot({
    controller: { ...controlRoomSnapshot().controller, state: 'offline' },
  })
  const model = buildControlRoomModel({ snapshot, freshness: 'live', error: null })
  assert.equal(model.kind, 'snapshot')
  assert.equal(model.freshnessLabel, 'Live')
  assert.equal(model.controllerLabel, 'Offline')
  assert.equal(model.authoritative, true)
})
