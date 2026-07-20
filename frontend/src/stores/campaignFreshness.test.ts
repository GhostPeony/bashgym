import assert from 'node:assert/strict'
import test from 'node:test'

import {
  acknowledgeCampaignSubscription,
  applyCampaignHint,
  beginCampaignReconciliation,
  createCampaignFreshness,
  expireCampaignCursor,
  failCampaignReconciliation,
  finishCampaignReconciliation,
  markCampaignDisconnected,
  markCampaignReconnected,
  parseCampaignHint,
  reconciliationRetryDelay,
  snapshotSemanticIdentity,
  validateCampaignSnapshot,
  type CampaignHintV1
} from './campaignFreshness'
import { controlRoomSnapshot } from '../components/autoresearch/controlRoomFixtures'
import { campaignApi, type CampaignControlRoomSnapshotV1 } from '../services/api'
import { campaignsMissingPanels } from './campaignCanvasLifecycle'
import { useCampaignStore, type CampaignRecord } from './campaignStore'

const hint = (overrides: Partial<CampaignHintV1> = {}): CampaignHintV1 => ({
  schema_version: 'campaign_hint.v1',
  workspace_id: 'workspace-a',
  campaign_id: 'campaign-a',
  event_cursor: 42,
  aggregate_version: 7,
  event_type: 'campaign:created',
  correlation_id: 'correlation-a',
  emitted_at: '2026-07-16T18:00:00Z',
  ...overrides
})

const controller = {
  schema_version: 'campaign_controller_status.v1',
  online: true,
  state: 'online' as const,
  code: 'controller_online' as const,
  observed_at: '2026-07-16T18:00:00Z'
}

test('duplicate and out-of-order hints are semantic no-ops', () => {
  const state = createCampaignFreshness({
    eventCursor: 42,
    aggregateVersion: 7,
    hasSnapshot: true,
    connected: true
  })
  const duplicate = applyCampaignHint(state, hint())
  const older = applyCampaignHint(state, hint({ event_cursor: 41, aggregate_version: 6 }))

  assert.equal(duplicate.reconcile, false)
  assert.equal(duplicate.state, state)
  assert.equal(older.reconcile, false)
  assert.equal(older.state, state)
})

test('subscription acknowledgement is required before target coverage can become live', () => {
  const initial = createCampaignFreshness({
    eventCursor: 0,
    aggregateVersion: 0,
    hasSnapshot: false,
    connected: true
  })
  const request = beginCampaignReconciliation(initial)
  const coveredBeforeAck = finishCampaignReconciliation(request, {
    generation: request.generation,
    eventCursor: 4,
    aggregateVersion: 2,
    semanticKey: 'snapshot-4-2',
    verifiedAt: '2026-07-16T18:00:00Z'
  })

  assert.notEqual(coveredBeforeAck.freshness, 'live')
  assert.equal(coveredBeforeAck.subscribed, false)

  const acknowledged = acknowledgeCampaignSubscription(coveredBeforeAck, 1)
  const verify = beginCampaignReconciliation(acknowledged)
  const coveredAfterAck = finishCampaignReconciliation(verify, {
    generation: verify.generation,
    eventCursor: 4,
    aggregateVersion: 2,
    semanticKey: 'snapshot-4-2',
    verifiedAt: '2026-07-16T18:00:01Z'
  })
  assert.equal(coveredAfterAck.freshness, 'live')
})

test('targets and applied authority are component-wise monotonic, including crossed hints', () => {
  const subscribed = acknowledgeCampaignSubscription(
    createCampaignFreshness({
      eventCursor: 40,
      aggregateVersion: 7,
      hasSnapshot: true,
      connected: true
    }),
    1
  )
  const cursorAdvance = applyCampaignHint(
    subscribed,
    hint({ event_cursor: 50, aggregate_version: 6 })
  )
  assert.equal(cursorAdvance.reconcile, true)
  assert.equal(cursorAdvance.state.targetCursor, 50)
  assert.equal(cursorAdvance.state.targetVersion, 7)

  const versionAdvance = applyCampaignHint(
    cursorAdvance.state,
    hint({ event_cursor: 39, aggregate_version: 9 })
  )
  assert.equal(versionAdvance.reconcile, true)
  assert.equal(versionAdvance.state.targetCursor, 50)
  assert.equal(versionAdvance.state.targetVersion, 9)

  const request = beginCampaignReconciliation(versionAdvance.state)
  const lowerCursor = finishCampaignReconciliation(request, {
    generation: request.generation,
    eventCursor: 39,
    aggregateVersion: 10,
    semanticKey: 'regressive-cursor',
    verifiedAt: '2026-07-16T18:00:00Z'
  })
  assert.equal(lowerCursor.appliedCursor, 40)
  assert.equal(lowerCursor.appliedVersion, 7)
  assert.notEqual(lowerCursor.freshness, 'live')
})

test('snapshot validator rejects schema and identity mismatches before store application', () => {
  const snapshot = controlRoomSnapshot({
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-a',
    campaign: { ...controlRoomSnapshot().campaign, campaign_id: 'campaign-a' }
  })
  assert.equal(
    validateCampaignSnapshot(snapshot, 'workspace-a', 'campaign-a')?.campaign_id,
    'campaign-a'
  )
  assert.equal(
    validateCampaignSnapshot(
      { ...snapshot, schema_version: 'control_room_snapshot.v2' },
      'workspace-a',
      'campaign-a'
    ),
    null
  )
  assert.equal(
    validateCampaignSnapshot(
      { ...snapshot, workspace_id: 'workspace-b' },
      'workspace-a',
      'campaign-a'
    ),
    null
  )
  assert.equal(
    validateCampaignSnapshot(
      { ...snapshot, campaign_id: 'campaign-b' },
      'workspace-a',
      'campaign-a'
    ),
    null
  )
  assert.equal(
    validateCampaignSnapshot(
      {
        schema_version: 'control_room_snapshot.v1',
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-a',
        aggregate_version: 1,
        latest_event_cursor: 1,
        campaign: { campaign_id: 'campaign-a', aggregate_version: 1 },
        controller: {},
        readiness: {}
      },
      'workspace-a',
      'campaign-a'
    ),
    null
  )
})

test('snapshot validator accepts backend-valid champion references through the 2000-character boundary', () => {
  const championRef = 'c'.repeat(2_000)
  const base = controlRoomSnapshot()
  const snapshot = controlRoomSnapshot({
    campaign: { ...base.campaign, champion_ref: championRef },
    champion: { ...base.champion!, candidate_ref: championRef }
  })

  assert.equal(
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id)?.campaign
      .champion_ref,
    championRef
  )
})

test('snapshot validator accepts canonical budget Identifier units through 160 characters', () => {
  const unit = `u${'n'.repeat(159)}`
  const base = controlRoomSnapshot()
  const snapshot = controlRoomSnapshot({
    budget: {
      ...base.budget,
      resources: [{ ...base.budget.resources[0], unit }]
    }
  })

  assert.equal(
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id)?.budget
      .resources[0]?.unit,
    unit
  )
})

test('snapshot validator rejects budget units outside the canonical Identifier contract', () => {
  const base = controlRoomSnapshot()
  const withUnit = (unit: string) =>
    controlRoomSnapshot({
      budget: {
        ...base.budget,
        resources: [{ ...base.budget.resources[0], unit }]
      }
    })

  assert.equal(
    validateCampaignSnapshot(withUnit('../gpu'), base.workspace_id, base.campaign_id),
    null
  )
  assert.equal(
    validateCampaignSnapshot(withUnit(`u${'n'.repeat(160)}`), base.workspace_id, base.campaign_id),
    null
  )
})

test('snapshot validator does not invent a launch-ready materializability implication', () => {
  const base = controlRoomSnapshot()
  const snapshot = controlRoomSnapshot({
    readiness: { ...base.readiness, materializable: false, launch_ready: true }
  })

  assert.equal(
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id)?.readiness
      .launch_ready,
    true
  )
})

test('snapshot validator accepts strings without renderer-only caps when the v1 contract is unconstrained', () => {
  const snapshot = structuredClone(controlRoomSnapshot())
  const long = 'x'.repeat(4_001)

  snapshot.campaign.title = long
  snapshot.campaign.objective = long
  snapshot.campaign.champion_ref = long
  snapshot.campaign.stop_reason = long
  snapshot.controller.safe_guidance = long
  snapshot.bindings.model!.display_label = long
  snapshot.journey[0]!.primary_blocker = {
    schema_version: 'decision_blocker.v1',
    code: 'campaign_blocked',
    summary: long,
    evidence_ids: [],
    secondary_codes: []
  }
  snapshot.active_work!.hypothesis_summary = long
  snapshot.active_work!.primary_variable_summary = long
  snapshot.active_work!.controlled_variable_summary = [long]
  snapshot.champion!.candidate_ref = long
  snapshot.metrics[0]!.display_name = long
  snapshot.metrics[0]!.unit = long
  snapshot.metrics[0]!.evaluator_revision = long
  snapshot.metrics[0]!.uncertainty_method = long
  snapshot.collections.events.next_cursor = long
  snapshot.collections.events.has_more = true

  assert.equal(
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id),
    snapshot
  )
})

test('snapshot validator rejects every wired Identifier boundary outside the backend grammar', () => {
  const invalidIdentifier = '../outside'
  const mutations: Array<(snapshot: CampaignControlRoomSnapshotV1) => void> = [
    (snapshot) => {
      snapshot.campaign.active_study_id = invalidIdentifier
    },
    (snapshot) => {
      snapshot.controller.controller_instance_id = invalidIdentifier
    },
    (snapshot) => {
      snapshot.readiness.blocking_codes = [invalidIdentifier]
    },
    (snapshot) => {
      snapshot.journey[0]!.next_action_ids = [invalidIdentifier]
    },
    (snapshot) => {
      snapshot.active_work!.attempt_id = invalidIdentifier
    },
    (snapshot) => {
      snapshot.candidate!.source_attempt_ids = [invalidIdentifier]
    },
    (snapshot) => {
      snapshot.candidate!.latest_comparable_evaluation_id = invalidIdentifier
    },
    (snapshot) => {
      snapshot.agents = [
        {
          schema_version: 'attached_agent_summary.v1',
          session_id: 'session-1',
          actor_id: 'actor-1',
          origin_id: 'origin-1',
          bundle_id: 'bundle-1',
          capability_revision: 1,
          expires_at: '2026-07-16T18:00:00Z',
          liveness: 'active',
          last_cursor: 0,
          last_request_id: invalidIdentifier
        }
      ]
    },
    (snapshot) => {
      snapshot.decision_surface.next_actions[0]!.action = invalidIdentifier
    },
    (snapshot) => {
      snapshot.decision_surface.recovery_actions = [invalidIdentifier]
    }
  ]

  for (const mutate of mutations) {
    const snapshot = structuredClone(controlRoomSnapshot())
    mutate(snapshot)
    assert.equal(
      validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id),
      null
    )
  }

  const crossScope = structuredClone(controlRoomSnapshot())
  crossScope.workspace_id = invalidIdentifier
  assert.equal(
    validateCampaignSnapshot(crossScope, invalidIdentifier, crossScope.campaign_id),
    null
  )
})

test('snapshot validator rejects backend-invalid enum and digest values', () => {
  const mutations: Array<(snapshot: CampaignControlRoomSnapshotV1) => void> = [
    (snapshot) => {
      snapshot.campaign.status =
        'future_status' as CampaignControlRoomSnapshotV1['campaign']['status']
    },
    (snapshot) => {
      snapshot.metrics[0]!.comparability_key = 'not-a-digest'
    },
    (snapshot) => {
      snapshot.decision_surface.next_actions[0]!.capability = 'campaign.future'
    }
  ]

  for (const mutate of mutations) {
    const snapshot = structuredClone(controlRoomSnapshot())
    mutate(snapshot)
    assert.equal(
      validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id),
      null
    )
  }
})

test('snapshot validator fails closed instead of throwing on malformed decision actions', () => {
  const snapshot = structuredClone(controlRoomSnapshot())
  snapshot.decision_surface.next_actions = [
    null
  ] as unknown as CampaignControlRoomSnapshotV1['decision_surface']['next_actions']

  assert.doesNotThrow(() =>
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id)
  )
  assert.equal(
    validateCampaignSnapshot(snapshot, snapshot.workspace_id, snapshot.campaign_id),
    null
  )
})

test('canonical identifiers accept colons and reject leading punctuation', () => {
  assert.equal(
    parseCampaignHint(
      hint({
        workspace_id: 'workspace:a',
        campaign_id: 'campaign:a',
        correlation_id: 'correlation:a'
      })
    )?.workspace_id,
    'workspace:a'
  )
  assert.equal(parseCampaignHint(hint({ workspace_id: '-workspace' })), null)
})

test('new hints advance targets and global cursor gaps remain ordinary wakeups', () => {
  const state = createCampaignFreshness({
    eventCursor: 40,
    aggregateVersion: 5,
    hasSnapshot: true,
    connected: true
  })
  const contiguous = applyCampaignHint(state, hint({ event_cursor: 41, aggregate_version: 6 }))
  const gap = applyCampaignHint(state, hint({ event_cursor: 44, aggregate_version: 7 }))

  assert.equal(contiguous.reconcile, true)
  assert.equal(contiguous.reason, 'hint')
  assert.equal(contiguous.state.freshness, 'stale')
  assert.equal(gap.reconcile, true)
  assert.equal(gap.reason, 'hint')
  assert.equal(gap.state.freshness, 'stale')
})

test('expired event detail cursor marks authority stale until compact reconciliation starts', () => {
  const state = createCampaignFreshness({
    eventCursor: 42,
    aggregateVersion: 7,
    hasSnapshot: true,
    connected: true
  })
  const expired = expireCampaignCursor(state)

  assert.equal(expired.freshness, 'stale')
  assert.equal(expired.errorCode, 'campaign_event_cursor_expired')
  const reconciling = beginCampaignReconciliation(expired)
  assert.equal(reconciling.freshness, 'reconciling')
  assert.equal(reconciling.generation, state.generation + 1)
})

test('disconnect, reconnect, and renderer reload preserve cached state without claiming authority', () => {
  const reloaded = createCampaignFreshness({
    eventCursor: 42,
    aggregateVersion: 7,
    hasSnapshot: true,
    connected: true
  })
  assert.equal(reloaded.freshness, 'stale')

  const offline = markCampaignDisconnected(reloaded)
  assert.equal(offline.freshness, 'offline')
  const reconnecting = markCampaignReconnected(offline)
  assert.equal(reconnecting.freshness, 'reconciling')
  assert.equal(reconnecting.generation, offline.generation + 1)
})

test('overlapping refresh completion cannot regress the newest generation', () => {
  const initial = acknowledgeCampaignSubscription(
    createCampaignFreshness({
      eventCursor: 40,
      aggregateVersion: 5,
      hasSnapshot: true,
      connected: true
    }),
    1
  )
  const first = beginCampaignReconciliation(initial)
  const second = beginCampaignReconciliation(first)
  const late = finishCampaignReconciliation(second, {
    generation: first.generation,
    eventCursor: 41,
    aggregateVersion: 6
  })
  const current = finishCampaignReconciliation(late, {
    generation: second.generation,
    eventCursor: 42,
    aggregateVersion: 7
  })

  assert.equal(late, second)
  assert.equal(current.freshness, 'live')
  assert.equal(current.appliedCursor, 42)
  assert.equal(current.appliedVersion, 7)
})

test('failures classify stale, offline, and error with bounded exponential retry', () => {
  const cached = beginCampaignReconciliation(
    createCampaignFreshness({
      eventCursor: 42,
      aggregateVersion: 7,
      hasSnapshot: true,
      connected: true
    })
  )
  assert.equal(
    failCampaignReconciliation(cached, { generation: cached.generation, connected: true })
      .freshness,
    'stale'
  )
  assert.equal(
    failCampaignReconciliation(cached, { generation: cached.generation, connected: false })
      .freshness,
    'offline'
  )

  const empty = beginCampaignReconciliation(
    createCampaignFreshness({
      eventCursor: 0,
      aggregateVersion: 0,
      hasSnapshot: false,
      connected: true
    })
  )
  assert.equal(
    failCampaignReconciliation(empty, {
      generation: empty.generation,
      connected: true,
      exhausted: true
    }).freshness,
    'error'
  )
  assert.deepEqual([0, 1, 2, 10].map(reconciliationRetryDelay), [250, 500, 1_000, 8_000])
})

test('campaign hint parser fails closed on extra or unsafe fields', () => {
  assert.deepEqual(parseCampaignHint(hint()), hint())
  assert.equal(parseCampaignHint({ ...hint(), payload: { private_path: 'C:/private' } }), null)
  assert.equal(parseCampaignHint({ ...hint(), aggregate_version: -1 }), null)
  assert.equal(parseCampaignHint({ ...hint(), event_cursor: 0 }), null)
  assert.equal(parseCampaignHint({ ...hint(), workspace_id: '../private' }), null)
  assert.equal(parseCampaignHint({ ...hint(), correlation_id: 'unsafe/correlation' }), null)
  assert.equal(parseCampaignHint({ ...hint(), schema_version: 'campaign_hint.v2' }), null)
})

test('campaign hint parser accepts only the backend canonical UTC timestamp form', () => {
  assert.equal(
    parseCampaignHint(hint({ emitted_at: '2026-07-16T18:00:00Z' }))?.emitted_at,
    '2026-07-16T18:00:00Z'
  )
  assert.equal(
    parseCampaignHint(hint({ emitted_at: '2026-07-16T18:00:00.123456Z' }))?.emitted_at,
    '2026-07-16T18:00:00.123456Z'
  )
  assert.equal(parseCampaignHint(hint({ emitted_at: '2026-07-16T11:00:00-07:00' })), null)
  assert.equal(parseCampaignHint(hint({ emitted_at: '2026-07-16 18:00:00Z' })), null)
  assert.equal(parseCampaignHint(hint({ emitted_at: '2026-07-16T18:00:00.123Z' })), null)
  assert.equal(parseCampaignHint(hint({ emitted_at: '2026-02-30T18:00:00Z' })), null)
})

test('semantic identity mirrors stable authoritative snapshot fields and ignores observation time', () => {
  const snapshot = controlRoomSnapshot({ aggregate_version: 7, latest_event_cursor: 42 })
  const key = snapshotSemanticIdentity(snapshot)
  assert.equal(
    snapshotSemanticIdentity({
      ...snapshot,
      snapshot_at: '2026-07-16T19:00:00Z',
      controller: { ...snapshot.controller, observed_at: '2026-07-16T19:00:00Z' },
      readiness: { ...snapshot.readiness, checked_at: '2026-07-16T19:00:00Z' }
    }),
    key
  )
  assert.notEqual(
    snapshotSemanticIdentity({
      ...snapshot,
      controller: {
        ...snapshot.controller,
        controller_observation_version: snapshot.controller.controller_observation_version + 1
      }
    }),
    key
  )
  assert.notEqual(
    snapshotSemanticIdentity({
      ...snapshot,
      readiness: {
        ...snapshot.readiness,
        blocking_codes: [...snapshot.readiness.blocking_codes, 'new_blocker']
      }
    }),
    key
  )
})

function campaign(id: string, title = id): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: id,
    workspace_id: 'workspace-a',
    title,
    kind: 'general',
    objective: 'Authoritative objective',
    target_model: {},
    owner_actor_id: 'desktop-user',
    manifest_revision: 1,
    status: 'active',
    version: 1,
    created_at: '2026-07-16T18:00:00Z',
    updated_at: '2026-07-16T18:00:00Z'
  }
}

test('store ignores duplicate hints and reconciles a gap from the compact snapshot', async () => {
  const originals = { ...campaignApi }
  useCampaignStore.setState({ workspaces: {} })
  campaignApi.list = async () => ({
    ok: true,
    data: { campaigns: [campaign('campaign-a')], controller }
  })
  let snapshots = 0
  campaignApi.snapshot = async () => {
    snapshots += 1
    const advanced = snapshots >= 3
    return {
      ok: true,
      data: controlRoomSnapshot({
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-a',
        aggregate_version: advanced ? 8 : 7,
        latest_event_cursor: advanced ? 45 : 42,
        campaign: {
          ...controlRoomSnapshot().campaign,
          campaign_id: 'campaign-a',
          aggregate_version: advanced ? 8 : 7
        }
      })
    }
  }
  try {
    await useCampaignStore.getState().load('workspace-a', 'campaign-a')
    await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
    await useCampaignStore.getState().handleHint(hint())
    assert.equal(snapshots, 2)

    await useCampaignStore.getState().handleHint(hint({ event_cursor: 45, aggregate_version: 8 }))
    const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-a']
    assert.equal(snapshots, 3)
    assert.equal(detail?.snapshot?.latest_event_cursor, 45)
    assert.equal(detail?.freshness, 'live')
  } finally {
    Object.assign(campaignApi, originals)
    useCampaignStore.setState({ workspaces: {} })
  }
})

test('campaign-created hint materializes only after authoritative fleet state contains it', async () => {
  const originals = { ...campaignApi }
  useCampaignStore.setState({ workspaces: {} })
  let includeCreated = false
  campaignApi.list = async () => ({
    ok: true,
    data: {
      campaigns: [
        campaign('campaign-a'),
        ...(includeCreated ? [campaign('campaign-new', 'Authoritative New Campaign')] : [])
      ],
      controller
    }
  })
  campaignApi.snapshot = async (workspaceId, campaignId) => ({
    ok: true,
    data: controlRoomSnapshot({
      workspace_id: workspaceId,
      campaign_id: campaignId,
      latest_event_cursor: campaignId === 'campaign-new' ? 43 : 42,
      campaign: {
        ...controlRoomSnapshot().campaign,
        campaign_id: campaignId,
        title:
          campaignId === 'campaign-new'
            ? 'Authoritative New Campaign'
            : controlRoomSnapshot().campaign.title
      }
    })
  })
  try {
    await useCampaignStore.getState().load('workspace-a', 'campaign-a')
    await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
    includeCreated = true
    await useCampaignStore.getState().handleHint(
      hint({
        campaign_id: 'campaign-new',
        event_cursor: 43,
        aggregate_version: 1,
        event_type: 'campaign:created'
      })
    )

    const campaigns = useCampaignStore.getState().workspaces['workspace-a']?.campaigns || []
    const missing = campaignsMissingPanels(campaigns, [])
    assert.equal(
      missing.find((item) => item.campaign_id === 'campaign-new')?.title,
      'Authoritative New Campaign'
    )
  } finally {
    Object.assign(campaignApi, originals)
    useCampaignStore.setState({ workspaces: {} })
  }
})

test('retry exhaustion keeps an unsatisfied pending target and becomes error', async () => {
  const originals = { ...campaignApi }
  const originalSetTimeout = globalThis.setTimeout
  useCampaignStore.setState({ workspaces: {} })
  campaignApi.list = async () => ({
    ok: true,
    data: { campaigns: [campaign('campaign-a')], controller }
  })
  let snapshots = 0
  campaignApi.snapshot = async () => {
    snapshots += 1
    return {
      ok: true,
      data: controlRoomSnapshot({
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-a',
        aggregate_version: 7,
        latest_event_cursor: 42,
        campaign: { ...controlRoomSnapshot().campaign, campaign_id: 'campaign-a' }
      })
    }
  }
  globalThis.setTimeout = ((callback: (...args: unknown[]) => void) => {
    queueMicrotask(callback)
    return 0 as unknown as ReturnType<typeof setTimeout>
  }) as typeof setTimeout
  try {
    await useCampaignStore.getState().load('workspace-a', 'campaign-a')
    await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
    await useCampaignStore.getState().handleHint(hint({ event_cursor: 50, aggregate_version: 8 }))

    const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-a']
    assert.equal(snapshots, 8)
    assert.equal(detail?.freshness, 'error')
    assert.equal(detail?.reconciliation.targetCursor, 50)
    assert.equal(detail?.reconciliation.targetVersion, 8)
  } finally {
    globalThis.setTimeout = originalSetTimeout
    Object.assign(campaignApi, originals)
    useCampaignStore.setState({ workspaces: {} })
  }
})

test('coalesced hint reconciliation honors the highest pending target', async () => {
  const originals = { ...campaignApi }
  const originalSetTimeout = globalThis.setTimeout
  useCampaignStore.setState({ workspaces: {} })
  campaignApi.list = async () => ({
    ok: true,
    data: { campaigns: [campaign('campaign-a')], controller }
  })
  campaignApi.snapshot = async () => ({
    ok: true,
    data: controlRoomSnapshot({
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-a',
      aggregate_version: 7,
      latest_event_cursor: 42,
      campaign: { ...controlRoomSnapshot().campaign, campaign_id: 'campaign-a' }
    })
  })
  globalThis.setTimeout = ((callback: (...args: unknown[]) => void) => {
    queueMicrotask(callback)
    return 0 as unknown as ReturnType<typeof setTimeout>
  }) as typeof setTimeout
  try {
    await useCampaignStore.getState().load('workspace-a', 'campaign-a')
    await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
    let release!: (cursor: number) => void
    const firstSnapshot = new Promise<number>((resolve) => {
      release = resolve
    })
    let liveCalls = 0
    campaignApi.snapshot = async () => {
      liveCalls += 1
      const cursor = liveCalls === 1 ? await firstSnapshot : 45
      const aggregateVersion = cursor === 43 ? 8 : 9
      return {
        ok: true,
        data: controlRoomSnapshot({
          workspace_id: 'workspace-a',
          campaign_id: 'campaign-a',
          aggregate_version: aggregateVersion,
          latest_event_cursor: cursor,
          campaign: {
            ...controlRoomSnapshot().campaign,
            campaign_id: 'campaign-a',
            aggregate_version: aggregateVersion
          }
        })
      }
    }

    const first = useCampaignStore
      .getState()
      .handleHint(hint({ event_cursor: 43, aggregate_version: 8 }))
    await Promise.resolve()
    const second = useCampaignStore
      .getState()
      .handleHint(hint({ event_cursor: 45, aggregate_version: 9 }))
    release(43)
    await Promise.all([first, second])

    const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-a']
    assert.equal(liveCalls, 2)
    assert.equal(detail?.snapshot?.latest_event_cursor, 45)
    assert.equal(detail?.reconciliation.targetCursor, 45)
    assert.equal(detail?.freshness, 'live')
  } finally {
    globalThis.setTimeout = originalSetTimeout
    Object.assign(campaignApi, originals)
    useCampaignStore.setState({ workspaces: {} })
  }
})
