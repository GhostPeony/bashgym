import assert from 'node:assert/strict'
import test from 'node:test'

import { controlRoomSnapshot } from './controlRoomFixtures'
import {
  buildControlRoomModel,
  campaignStatusTone,
  presentBlocker,
  presentJourney,
  presentMetrics,
  presentNeedsYou,
  resolveControlRoomFreshness
} from './controlRoomModel'

test('inherits workspace authority when no campaign detail is loaded', () => {
  assert.equal(resolveControlRoomFreshness({ workspaceFreshness: 'offline' }), 'offline')
  assert.equal(resolveControlRoomFreshness({ workspaceFreshness: 'stale' }), 'stale')
  assert.equal(resolveControlRoomFreshness({ workspaceLoading: true }), 'reconciling')
  assert.equal(resolveControlRoomFreshness({ workspaceError: 'bad contract' }), 'error')
  assert.equal(
    resolveControlRoomFreshness({ detailFreshness: 'live', workspaceFreshness: 'offline' }),
    'live'
  )
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
    ...snapshot.journey.slice(3)
  ])
  assert.deepEqual(
    phases.map((phase) => phase.id),
    ['setup', 'baseline', 'experiments', 'human_review', 'decision']
  )
  assert.equal(phases[2]?.tone, 'neutral')
  assert.equal(phases[2]?.stateLabel, 'New Remote State')
})

test('needs-you presenter surfaces the server blocker with its machine code and stays silent when idle', () => {
  const snapshot = controlRoomSnapshot()
  const blocked = presentNeedsYou({
    surface: {
      ...snapshot.decision_surface,
      blocker: {
        schema_version: 'decision_blocker.v1',
        code: 'campaign_compute_unreachable',
        summary: 'The registered compute target is unreachable, so readiness is blocked.',
        evidence_ids: [],
        secondary_codes: []
      }
    },
    readiness: snapshot.readiness,
    humanWork: snapshot.human_work,
    status: snapshot.campaign.status
  })
  assert.equal(
    blocked?.sentence,
    'The registered compute target is unreachable, so readiness is blocked.'
  )
  assert.equal(blocked?.code, 'campaign_compute_unreachable')

  const idle = presentNeedsYou({
    surface: snapshot.decision_surface,
    readiness: snapshot.readiness,
    humanWork: snapshot.human_work,
    status: snapshot.campaign.status
  })
  assert.equal(idle, null)
})

test('needs-you presenter derives launch-blocked and pending human work when there is no server blocker', () => {
  const snapshot = controlRoomSnapshot()
  const launchBlocked = presentNeedsYou({
    surface: snapshot.decision_surface,
    readiness: {
      ...snapshot.readiness,
      launch_ready: false,
      blocking_codes: ['controller_offline']
    },
    humanWork: snapshot.human_work,
    status: 'ready'
  })
  assert.match(launchBlocked?.sentence ?? '', /Controller Offline/)
  assert.equal(launchBlocked?.code, 'controller_offline')

  const humanBlocking = presentNeedsYou({
    surface: snapshot.decision_surface,
    readiness: snapshot.readiness,
    humanWork: { ...snapshot.human_work, blocking_count: 2 },
    status: snapshot.campaign.status
  })
  assert.match(humanBlocking?.sentence ?? '', /2 blinded reviews are waiting/)
  assert.equal(humanBlocking?.code, 'human_review_required')
})

test('preserves the server primary blocker and evidence IDs', () => {
  const snapshot = controlRoomSnapshot()
  const blocker = {
    schema_version: 'decision_blocker.v1' as const,
    code: 'human_review_required',
    summary: 'A blinded review is required.',
    evidence_ids: ['evidence-2', 'evidence-1'],
    secondary_codes: ['budget_warning']
  }
  assert.deepEqual(presentBlocker({ ...snapshot.decision_surface, blocker }), {
    code: blocker.code,
    summary: blocker.summary,
    evidenceIds: blocker.evidence_ids,
    secondaryCodes: blocker.secondary_codes
  })
})

test('renders metric descriptors without fabricated values or deltas', () => {
  const snapshot = controlRoomSnapshot()
  const metrics = presentMetrics(snapshot.metrics, snapshot.champion, snapshot.candidate)
  assert.equal(metrics[0]?.value, 'Unavailable')
  assert.equal(metrics[0]?.delta, null)
})

test('builds distinct initial and cached load states', () => {
  assert.equal(
    buildControlRoomModel({ snapshot: null, freshness: 'reconciling', error: null }).kind,
    'loading'
  )
  assert.equal(
    buildControlRoomModel({ snapshot: null, freshness: 'error', error: 'bad contract' }).kind,
    'error'
  )
  assert.equal(
    buildControlRoomModel({ snapshot: null, freshness: 'offline', error: 'bridge down' }).kind,
    'offline'
  )
  assert.equal(
    buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'stale', error: 'timeout' })
      .kind,
    'snapshot'
  )
  assert.equal(
    buildControlRoomModel({
      snapshot: controlRoomSnapshot(),
      freshness: 'offline',
      error: 'bridge down'
    }).authoritative,
    false
  )
})

test('keeps controller health distinct from renderer freshness', () => {
  const snapshot = controlRoomSnapshot({
    controller: { ...controlRoomSnapshot().controller, state: 'offline' }
  })
  const model = buildControlRoomModel({ snapshot, freshness: 'live', error: null })
  assert.equal(model.kind, 'snapshot')
  assert.equal(model.freshnessLabel, 'Live')
  assert.equal(model.controllerLabel, 'Offline')
  assert.equal(model.authoritative, true)
})
