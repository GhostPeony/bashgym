import assert from 'node:assert/strict'
import test from 'node:test'

import { controlRoomSnapshot } from '../components/autoresearch/controlRoomFixtures'
import { campaignApi } from '../services/api'
import { useActivityStore } from './activityStore'
import {
  retainCampaignSafetyReconcile,
  useCampaignStore,
  type CampaignRecord,
} from './campaignStore'

const originals = { ...campaignApi }

function campaign(workspaceId: string, id = `campaign-${workspaceId}`): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: id,
    workspace_id: workspaceId,
    title: `${id} title`,
    kind: 'general',
    objective: 'Improve quality with bounded evidence.',
    target_model: {},
    owner_actor_id: 'desktop-user',
    manifest_revision: 1,
    status: 'active',
    version: 1,
    created_at: '2026-07-16T00:00:00Z',
    updated_at: '2026-07-16T00:00:01Z',
  }
}

function reset() {
  useCampaignStore.setState({ workspaces: {} })
  useActivityStore.getState().clear()
  Object.assign(campaignApi, originals)
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => { resolve = next })
  return { promise, resolve }
}

function installList(campaignsForWorkspace?: (workspaceId: string) => CampaignRecord[]) {
  campaignApi.list = async (workspaceId) => ({
    ok: true,
    data: {
      campaigns: campaignsForWorkspace?.(workspaceId) || [campaign(workspaceId)],
      controller: {
        schema_version: 'campaign_controller_status.v1',
        online: true,
        state: 'online',
        code: 'controller_online',
        observed_at: '2026-07-16T00:00:01Z',
      },
    },
  })
}

test('load selects preferred, prior, then first and hydrates one snapshot', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-first'), campaign('workspace-a', 'campaign-preferred')])
  const requested: string[] = []
  campaignApi.snapshot = async (workspaceId, campaignId) => {
    requested.push(campaignId)
    return { ok: true, data: controlRoomSnapshot({ workspace_id: workspaceId, campaign_id: campaignId, campaign: { ...controlRoomSnapshot().campaign, campaign_id: campaignId } }) }
  }

  await useCampaignStore.getState().load('workspace-a', 'campaign-preferred')
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.selectedCampaignId, 'campaign-preferred')
  assert.deepEqual(requested, ['campaign-preferred'])

  await useCampaignStore.getState().select('workspace-a', 'campaign-first')
  await useCampaignStore.getState().load('workspace-a')
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.selectedCampaignId, 'campaign-first')
})

test('a fresh Control Room defaults to the most recently updated campaign', async () => {
  reset()
  const oldest = campaign('workspace-a', 'campaign-oldest')
  const newest = {
    ...campaign('workspace-a', 'campaign-newest'),
    updated_at: '2026-07-16T01:00:00Z',
  }
  installList(() => [oldest, newest])
  const requested: string[] = []
  campaignApi.snapshot = async (workspaceId, campaignId) => {
    requested.push(campaignId)
    return {
      ok: true,
      data: controlRoomSnapshot({
        workspace_id: workspaceId,
        campaign_id: campaignId,
        campaign: { ...controlRoomSnapshot().campaign, campaign_id: campaignId },
      }),
    }
  }

  await useCampaignStore.getState().load('workspace-a')

  assert.equal(
    useCampaignStore.getState().workspaces['workspace-a']?.selectedCampaignId,
    'campaign-newest',
  )
  assert.deepEqual(requested, ['campaign-newest'])
})

test('compact refresh calls only the atomic snapshot endpoint', async () => {
  reset()
  installList()
  const forbidden = ['get', 'studies', 'proposals', 'evidence', 'attempts', 'artifacts', 'comparisons', 'ledger', 'events', 'metrics'] as const
  for (const name of forbidden) {
    campaignApi[name] = (async () => { throw new Error(`${name} must not hydrate compact state`) }) as never
  }
  campaignApi.snapshot = async (workspaceId, campaignId) => ({
    ok: true,
    data: controlRoomSnapshot({ workspace_id: workspaceId, campaign_id: campaignId, campaign: { ...controlRoomSnapshot().campaign, campaign_id: campaignId } }),
  })

  await useCampaignStore.getState().load('workspace-a')
  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-workspace-a']
  assert.equal(detail?.freshness, 'stale')
  assert.equal(detail?.lastVerifiedAt, '2026-07-16T18:00:00Z')
  assert.equal(detail?.snapshot?.latest_event_cursor, 42)
})

test('semantic snapshot no-op preserves renderer snapshot and campaign collection references', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  let observedAt = '2026-07-16T18:00:00Z'
  campaignApi.snapshot = async () => ({
    ok: true,
    data: controlRoomSnapshot({
      snapshot_at: observedAt,
      controller: { ...controlRoomSnapshot().controller, observed_at: observedAt },
      readiness: { ...controlRoomSnapshot().readiness, checked_at: observedAt },
    }),
  })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  const before = useCampaignStore.getState().workspaces['workspace-a']!
  const snapshot = before.details['campaign-1']!.snapshot
  const campaigns = before.campaigns
  observedAt = '2026-07-16T18:01:00Z'
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  const after = useCampaignStore.getState().workspaces['workspace-a']!
  assert.equal(after.details['campaign-1']!.snapshot, snapshot)
  assert.equal(after.campaigns, campaigns)
})

test('first failure distinguishes offline from contract error', async () => {
  reset()
  installList()
  campaignApi.snapshot = async () => ({ ok: false, code: 'campaign_desktop_bridge_required', error: 'bridge unavailable' })
  await useCampaignStore.getState().load('workspace-a')
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-workspace-a']?.freshness, 'offline')

  for (const code of [
    'campaign_backend_unavailable',
    'campaign_backend_root_unavailable',
    'campaign_backend_configuration_invalid',
  ]) {
    reset()
    installList()
    campaignApi.snapshot = async () => ({ ok: false, code, error: 'AutoResearch campaign backend is unavailable.' })
    await useCampaignStore.getState().load('workspace-a')
    assert.equal(
      useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-workspace-a']?.freshness,
      'offline',
      code,
    )
  }

  reset()
  installList()
  campaignApi.snapshot = async () => ({ ok: false, code: 'campaign_response_invalid', error: 'invalid snapshot' })
  await useCampaignStore.getState().load('workspace-a')
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-workspace-a']?.freshness, 'error')
})

test('failed refresh retains cached snapshot and marks stale or offline', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  campaignApi.snapshot = async () => ({ ok: false, error: 'timeout' })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  let detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.aggregate_version, 7)
  assert.equal(detail?.freshness, 'stale')

  campaignApi.snapshot = async () => ({ ok: false, code: 'campaign_desktop_bridge_required', error: 'offline' })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.aggregate_version, 7)
  assert.equal(detail?.freshness, 'offline')
})

test('aggregate version and request generation guards prevent regression', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({ aggregate_version: 9, latest_event_cursor: 50, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 9 } }) })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({ aggregate_version: 8, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 8 } }) })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  let detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.aggregate_version, 9)
  assert.equal(detail?.freshness, 'stale')
  assert.match(detail?.error || '', /regressive/i)

  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({ aggregate_version: 9, latest_event_cursor: 40, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 9 } }) })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.latest_event_cursor, 50)
  assert.equal(detail?.freshness, 'stale')
  assert.match(detail?.error || '', /regressive/i)

  campaignApi.snapshot = async () => ({ ok: false, error: 'timeout' })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.freshness, 'stale')
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({ aggregate_version: 8, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 8 } }) })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.freshness, 'stale')
  assert.match(detail?.error || '', /regressive/i)

  let release!: (value: ReturnType<typeof controlRoomSnapshot>) => void
  const slow = new Promise<ReturnType<typeof controlRoomSnapshot>>((resolve) => { release = resolve })
  let calls = 0
  campaignApi.snapshot = async () => {
    calls += 1
    return calls === 1
      ? { ok: true, data: await slow }
      : { ok: true, data: controlRoomSnapshot({ aggregate_version: 11, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 11 } }) }
  }
  const earlier = useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  const later = useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  assert.equal(calls, 1)
  release(controlRoomSnapshot({ aggregate_version: 11, latest_event_cursor: 51, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 11 } }))
  await Promise.all([earlier, later])
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.snapshot?.aggregate_version, 11)
})

test('late legacy drill-down cannot overwrite newer snapshot authority', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  const legacyCampaign = deferred<Awaited<ReturnType<typeof campaignApi.get>>>()
  campaignApi.get = async () => legacyCampaign.promise
  campaignApi.studies = async () => ({ ok: true, data: { studies: [] } })
  campaignApi.proposals = async () => ({ ok: true, data: { proposals: [] } })
  campaignApi.evidence = async () => ({ ok: true, data: {} as never })
  campaignApi.attempts = async () => ({ ok: true, data: { attempts: [] } })
  campaignApi.comparisons = async () => ({ ok: true, data: { comparisons: [] } })
  campaignApi.ledger = async () => ({ ok: true, data: {} as never })

  const legacyLoad = useCampaignStore.getState().loadLegacyDetail('workspace-a', 'campaign-1')
  campaignApi.snapshot = async () => ({
    ok: true,
    data: controlRoomSnapshot({
      aggregate_version: 9,
      campaign: {
        ...controlRoomSnapshot().campaign,
        aggregate_version: 9,
        status: 'paused',
        title: 'Snapshot owns this title',
      },
    }),
  })
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
  legacyCampaign.resolve({
    ok: true,
    data: { ...campaign('workspace-a', 'campaign-1'), version: 2, status: 'active', title: 'Obsolete legacy title' },
  })
  await legacyLoad

  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.campaign.version, 9)
  assert.equal(detail?.campaign.status, 'paused')
  assert.equal(detail?.campaign.title, 'Snapshot owns this title')
})

test('newest fleet load owns workspace selection when responses resolve out of order', async () => {
  reset()
  const first = deferred<Awaited<ReturnType<typeof campaignApi.list>>>()
  const second = deferred<Awaited<ReturnType<typeof campaignApi.list>>>()
  let listCalls = 0
  campaignApi.list = async () => (++listCalls === 1 ? first.promise : second.promise)
  campaignApi.snapshot = async (workspaceId, campaignId) => ({
    ok: true,
    data: controlRoomSnapshot({
      workspace_id: workspaceId,
      campaign_id: campaignId,
      campaign: { ...controlRoomSnapshot().campaign, campaign_id: campaignId },
    }),
  })

  const olderLoad = useCampaignStore.getState().load('workspace-a', 'campaign-old')
  const newerLoad = useCampaignStore.getState().load('workspace-a', 'campaign-new')
  const fleet = {
    campaigns: [campaign('workspace-a', 'campaign-old'), campaign('workspace-a', 'campaign-new')],
    controller: {
      schema_version: 'campaign_controller_status.v1',
      online: true,
      state: 'online' as const,
      code: 'controller_online' as const,
      observed_at: '2026-07-16T00:00:01Z',
    },
  }
  second.resolve({ ok: true, data: fleet })
  await newerLoad
  first.resolve({ ok: true, data: fleet })
  await olderLoad

  const workspace = useCampaignStore.getState().workspaces['workspace-a']
  assert.equal(workspace?.selectedCampaignId, 'campaign-new')
  assert.equal(workspace?.details['campaign-new']?.freshness, 'stale')
  assert.equal(workspace?.details['campaign-old'], undefined)
})

test('fleet failures downgrade cached authority and classify initial bridge absence offline', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  campaignApi.list = async () => ({
    ok: false,
    code: 'campaign_desktop_bridge_required',
    error: 'Desktop campaign bridge is unavailable',
  })
  await useCampaignStore.getState().load('workspace-a')
  let workspace = useCampaignStore.getState().workspaces['workspace-a']
  assert.equal(workspace?.freshness, 'offline')
  assert.equal(workspace?.details['campaign-1']?.freshness, 'offline')
  assert.equal(workspace?.details['campaign-1']?.snapshot?.aggregate_version, 7)

  reset()
  campaignApi.list = async () => ({
    ok: false,
    code: 'campaign_desktop_bridge_required',
    error: 'Desktop campaign bridge is unavailable',
  })
  await useCampaignStore.getState().load('workspace-a')
  workspace = useCampaignStore.getState().workspaces['workspace-a']
  assert.equal(workspace?.freshness, 'offline')
  assert.equal(workspace?.campaigns.length, 0)
})

test('workspace projections stay isolated', async () => {
  reset()
  installList()
  campaignApi.snapshot = async (workspaceId, campaignId) => ({
    ok: true,
    data: controlRoomSnapshot({ workspace_id: workspaceId, campaign_id: campaignId, campaign: { ...controlRoomSnapshot().campaign, campaign_id: campaignId } }),
  })
  await Promise.all([useCampaignStore.getState().load('workspace-a'), useCampaignStore.getState().load('workspace-b')])
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-workspace-a']?.snapshot?.workspace_id, 'workspace-a')
  assert.equal(useCampaignStore.getState().workspaces['workspace-b']?.details['campaign-workspace-b']?.snapshot?.workspace_id, 'workspace-b')
})

test('event and artifact pages append with durable dedupe without changing decisions', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  campaignApi.events = async () => ({
    ok: true,
    data: {
      items: [1, 1, 2].map((cursor) => ({
        cursor,
        event: {
          schema_version: 'public_campaign_event.v1' as const,
          event_id: `event-${cursor}`,
          workspace_id: 'workspace-a',
          campaign_id: 'campaign-1',
          sequence: cursor,
          aggregate_version: 7,
          event_type: 'campaign:started',
          actor_id: 'desktop-user',
          credential_kind: 'access',
          created_at: '2026-07-16T00:00:00Z',
        },
      })),
      next_cursor: 2,
    },
  })
  campaignApi.artifacts = async () => ({
    ok: true,
    data: {
      artifacts: [],
      next_cursor: null,
      has_more: false,
    },
  })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  const before = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.snapshot?.decision_surface
  await useCampaignStore.getState().loadEventsPage('workspace-a', 'campaign-1')
  await useCampaignStore.getState().loadArtifactsPage('workspace-a', 'campaign-1')
  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.deepEqual(detail?.pages.events.map((item) => item.event.event_id), ['event-1', 'event-2'])
  assert.equal(useActivityStore.getState().events.length, 2)
  assert.equal(detail?.snapshot?.decision_surface, before)
})

test('concurrent event page loads share one request and forward each durable event once', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  const page = deferred<Awaited<ReturnType<typeof campaignApi.events>>>()
  let requests = 0
  campaignApi.events = async () => {
    requests += 1
    return page.promise
  }
  const first = useCampaignStore.getState().loadEventsPage('workspace-a', 'campaign-1')
  const second = useCampaignStore.getState().loadEventsPage('workspace-a', 'campaign-1')
  assert.equal(requests, 1)
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.pages.eventsLoading, true)

  const duplicate = {
    cursor: 1,
    event: {
      schema_version: 'public_campaign_event.v1' as const,
      event_id: 'event-1',
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-1',
      sequence: 1,
      aggregate_version: 7,
      event_type: 'campaign:started',
      actor_id: 'desktop-user',
      credential_kind: 'access',
      created_at: '2026-07-16T00:00:00Z',
    },
  }
  page.resolve({ ok: true, data: { items: [duplicate, duplicate], next_cursor: 1 } })
  await Promise.all([first, second])

  const pages = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.pages
  assert.deepEqual(pages?.events.map((item) => item.event.event_id), ['event-1'])
  assert.equal(pages?.eventsLoading, false)
  assert.equal(pages?.eventsLoaded, true)
  assert.equal(useActivityStore.getState().events.length, 1)
})

test('page loaders expose retryable errors and suppress concurrent artifact requests', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  campaignApi.events = async () => ({ ok: false, error: 'events unavailable' })
  await useCampaignStore.getState().loadEventsPage('workspace-a', 'campaign-1')
  let pages = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.pages
  assert.equal(pages?.eventsLoading, false)
  assert.equal(pages?.eventsLoaded, false)
  assert.equal(pages?.eventsError, 'events unavailable')

  const artifactPage = deferred<Awaited<ReturnType<typeof campaignApi.artifacts>>>()
  let requests = 0
  campaignApi.artifacts = async () => {
    requests += 1
    return artifactPage.promise
  }
  const first = useCampaignStore.getState().loadArtifactsPage('workspace-a', 'campaign-1')
  const second = useCampaignStore.getState().loadArtifactsPage('workspace-a', 'campaign-1')
  assert.equal(requests, 1)
  assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.pages.artifactsLoading, true)

  artifactPage.resolve({ ok: true, data: { artifacts: [], next_cursor: null, has_more: false } })
  await Promise.all([first, second])
  pages = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.pages
  assert.equal(pages?.artifactsLoading, false)
  assert.equal(pages?.artifactsLoaded, true)
  assert.equal(pages?.artifactsHasMore, false)
  assert.equal(pages?.artifactsError, null)
})

test('active attempt metrics load incrementally, share one request, and stay bounded', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  const firstPage = deferred<Awaited<ReturnType<typeof campaignApi.metrics>>>()
  const afterSteps: number[] = []
  campaignApi.metrics = async (_workspaceId, _campaignId, _attemptId, _source, _metricName, afterStep) => {
    afterSteps.push(afterStep ?? -1)
    return firstPage.promise
  }
  const first = useCampaignStore.getState().loadAttemptMetrics('workspace-a', 'campaign-1', 'attempt-1')
  const duplicate = useCampaignStore.getState().loadAttemptMetrics('workspace-a', 'campaign-1', 'attempt-1')
  assert.deepEqual(afterSteps, [-1])
  firstPage.resolve({
    ok: true,
    data: {
      metric_name: 'loss',
      source: 'training_metrics.jsonl',
      values: [
        { step: 1, source: 'training_metrics.jsonl', value: 1.03, observed_at: '2026-07-16T18:00:00Z' },
        { step: 42, source: 'training_metrics.jsonl', value: 0.27, observed_at: '2026-07-16T18:01:00Z' },
      ],
      next_after_step: 42,
    },
  })
  await Promise.all([first, duplicate])

  campaignApi.metrics = async (_workspaceId, _campaignId, _attemptId, _source, _metricName, afterStep) => {
    afterSteps.push(afterStep ?? -1)
    return {
      ok: true,
      data: {
        metric_name: 'loss',
        source: 'training_metrics.jsonl',
        values: [
          { step: 42, source: 'training_metrics.jsonl', value: 0.26, observed_at: '2026-07-16T18:01:05Z' },
          ...Array.from({ length: 520 }, (_, index) => ({
            step: 43 + index,
            source: 'training_metrics.jsonl',
            value: 0.25 - index / 10_000,
            observed_at: '2026-07-16T18:02:00Z',
          })),
        ],
        next_after_step: 562,
      },
    }
  }
  await useCampaignStore.getState().loadAttemptMetrics('workspace-a', 'campaign-1', 'attempt-1')

  const values = useCampaignStore.getState().workspaces['workspace-a']!.details['campaign-1']!.lossByAttempt['attempt-1']
  assert.deepEqual(afterSteps, [-1, 42])
  assert.equal(values.length, 500)
  assert.equal(values.at(-1)?.step, 562)
  assert.equal(new Set(values.map((point) => `${point.source}:${point.step}`)).size, 500)
})

test('late metric responses cannot cross an active-attempt change', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')

  const page = deferred<Awaited<ReturnType<typeof campaignApi.metrics>>>()
  campaignApi.metrics = async () => page.promise
  const pending = useCampaignStore.getState().loadAttemptMetrics('workspace-a', 'campaign-1', 'attempt-1')
  useCampaignStore.setState((state) => {
    const workspace = state.workspaces['workspace-a']!
    const detail = workspace.details['campaign-1']!
    return { workspaces: { ...state.workspaces, 'workspace-a': { ...workspace, details: {
      ...workspace.details,
      'campaign-1': { ...detail, snapshot: { ...detail.snapshot!, active_work: { ...detail.snapshot!.active_work!, attempt_id: 'attempt-2' } } },
    } } } }
  })
  page.resolve({
    ok: true,
    data: {
      metric_name: 'loss', source: 'training_metrics.jsonl', next_after_step: 1,
      values: [{ step: 1, source: 'training_metrics.jsonl', value: 1, observed_at: '2026-07-16T18:00:00Z' }],
    },
  })
  await pending

  assert.deepEqual(
    useCampaignStore.getState().workspaces['workspace-a']!.details['campaign-1']!.lossByAttempt['attempt-1'],
    undefined,
  )
})

test('transition acknowledges then refreshes the snapshot', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  let snapshots = 0
  campaignApi.snapshot = async () => {
    snapshots += 1
    return { ok: true, data: controlRoomSnapshot() }
  }
  campaignApi.transition = async () => ({ ok: true, data: { campaign: campaign('workspace-a', 'campaign-1'), event: {}, replayed: false } })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
  assert.equal(await useCampaignStore.getState().transition('workspace-a', 'campaign-1', 'pause', 1), true)
  assert.equal(snapshots, 3)
})

test('a rejected transition reconciles the server-owned gate before allowing another attempt', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  const initial = controlRoomSnapshot()
  campaignApi.snapshot = async () => ({ ok: true, data: initial })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)

  let reconciled = 0
  campaignApi.transition = async () => ({ ok: false, status: 409, code: 'campaign_launch_not_ready', error: 'Campaign request failed with HTTP 409' })
  campaignApi.snapshot = async () => {
    reconciled += 1
    return {
      ok: true,
      data: controlRoomSnapshot({
        readiness: {
          ...initial.readiness,
          launch_ready: false,
          blocking_codes: ['controller_offline'],
        },
      }),
    }
  }

  assert.equal(await useCampaignStore.getState().transition('workspace-a', 'campaign-1', 'start', 7), false)
  assert.equal(reconciled, 1)
  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.readiness.launch_ready, false)
  assert.deepEqual(detail?.snapshot?.readiness.blocking_codes, ['controller_offline'])
})

test('store never exposes live before subscription ack or while a snapshot is behind its hint target', async () => {
  const originalSetTimeout = globalThis.setTimeout
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  assert.notEqual(
    useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.freshness,
    'live',
  )

  await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({
    aggregate_version: 7,
    latest_event_cursor: 42,
  }) })
  const observed: string[] = []
  const unsubscribe = useCampaignStore.subscribe((state) => {
    const value = state.workspaces['workspace-a']?.details['campaign-1']?.freshness
    if (value) observed.push(value)
  })
  globalThis.setTimeout = ((callback: (...args: unknown[]) => void) => {
    queueMicrotask(callback)
    return 0 as unknown as ReturnType<typeof setTimeout>
  }) as typeof setTimeout
  const pending = useCampaignStore.getState().handleHint({
    schema_version: 'campaign_hint.v1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    event_cursor: 50,
    aggregate_version: 8,
    event_type: 'campaign:advanced',
    correlation_id: 'correlation-1',
    emitted_at: '2026-07-16T18:00:00Z',
  })
  try {
    await pending
    assert.equal(observed.includes('live'), false)
    assert.equal(useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.freshness, 'error')
  } finally {
    unsubscribe()
    globalThis.setTimeout = originalSetTimeout
  }
})

test('snapshot runtime validation rejects crossed regressions and identity/schema mismatches', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot({
    aggregate_version: 9,
    latest_event_cursor: 50,
    campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 9 },
  }) })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  const retainedSnapshot = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']?.snapshot
  let mutations = 0
  campaignApi.transition = async () => {
    mutations += 1
    return { ok: true, data: { campaign: campaign('workspace-a', 'campaign-1'), event: {}, replayed: false } }
  }

  for (const invalid of [
    controlRoomSnapshot({ aggregate_version: 10, latest_event_cursor: 49, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 10 } }),
    controlRoomSnapshot({ aggregate_version: 8, latest_event_cursor: 51, campaign: { ...controlRoomSnapshot().campaign, aggregate_version: 8 } }),
    controlRoomSnapshot({ workspace_id: 'workspace-b' }),
    controlRoomSnapshot({ campaign_id: 'campaign-b', campaign: { ...controlRoomSnapshot().campaign, campaign_id: 'campaign-b' } }),
    { ...controlRoomSnapshot(), schema_version: 'control_room_snapshot.v2' },
    {
      schema_version: 'control_room_snapshot.v1', workspace_id: 'workspace-a', campaign_id: 'campaign-1',
      aggregate_version: 9, latest_event_cursor: 50,
      campaign: { campaign_id: 'campaign-1', aggregate_version: 9 }, controller: {}, readiness: {},
    },
  ]) {
    campaignApi.snapshot = async () => ({ ok: true, data: invalid as never })
    await useCampaignStore.getState().refresh('workspace-a', 'campaign-1')
    const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
    assert.equal(detail?.snapshot?.aggregate_version, 9)
    assert.equal(detail?.snapshot, retainedSnapshot)
    assert.notEqual(detail?.freshness, 'live')
    assert.equal(await useCampaignStore.getState().transition('workspace-a', 'campaign-1', 'pause', 9), false)
  }
  assert.equal(mutations, 0)
})

test('campaign-created reloads the fleet then reconciles the exact hinted campaign', async () => {
  reset()
  let created = false
  installList(() => [
    campaign('workspace-a', 'campaign-1'),
    ...(created ? [campaign('workspace-a', 'campaign-new')] : []),
  ])
  const snapshots: string[] = []
  campaignApi.snapshot = async (workspaceId, campaignId) => {
    snapshots.push(campaignId)
    return { ok: true, data: controlRoomSnapshot({
      workspace_id: workspaceId,
      campaign_id: campaignId,
      aggregate_version: campaignId === 'campaign-new' ? 2 : 7,
      latest_event_cursor: campaignId === 'campaign-new' ? 51 : 42,
      campaign: {
        ...controlRoomSnapshot().campaign,
        campaign_id: campaignId,
        aggregate_version: campaignId === 'campaign-new' ? 2 : 7,
      },
    }) }
  }
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)
  created = true
  await useCampaignStore.getState().handleHint({
    schema_version: 'campaign_hint.v1', workspace_id: 'workspace-a', campaign_id: 'campaign-new',
    event_cursor: 51, aggregate_version: 2, event_type: 'campaign:created',
    correlation_id: 'correlation-new', emitted_at: '2026-07-16T18:00:00Z',
  })
  assert.equal(snapshots.includes('campaign-new'), true)
  assert.equal(
    useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-new']?.snapshot?.campaign_id,
    'campaign-new',
  )
})

test('store mutation boundary rejects every non-live authority state', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  campaignApi.snapshot = async () => ({ ok: true, data: controlRoomSnapshot() })
  let mutations = 0
  campaignApi.transition = async () => {
    mutations += 1
    return { ok: true, data: { campaign: campaign('workspace-a', 'campaign-1'), event: {}, replayed: false } }
  }
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  for (const freshness of ['reconciling', 'stale', 'offline', 'error'] as const) {
    useCampaignStore.setState((state) => ({ workspaces: {
      ...state.workspaces,
      'workspace-a': {
        ...state.workspaces['workspace-a']!,
        details: {
          ...state.workspaces['workspace-a']!.details,
          'campaign-1': { ...state.workspaces['workspace-a']!.details['campaign-1']!, freshness },
        },
      },
    } }))
    assert.equal(await useCampaignStore.getState().transition('workspace-a', 'campaign-1', 'pause', 1), false)
  }
  assert.equal(mutations, 0)
})

test('explicit structured 410 resets event detail and reconciles compact authority', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  let snapshots = 0
  campaignApi.snapshot = async () => {
    snapshots += 1
    return { ok: true, data: controlRoomSnapshot() }
  }
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  campaignApi.events = async () => ({
    ok: false,
    status: 410,
    code: 'campaign_event_cursor_expired',
    details: { resume_cursor: 17 },
    error: 'Historical campaign events were compacted.',
  })
  await useCampaignStore.getState().loadEventsPage('workspace-a', 'campaign-1')
  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.pages.eventCursor, 17)
  assert.deepEqual(detail?.pages.events, [])
  assert.equal(snapshots, 2)
  assert.equal(detail?.snapshot?.aggregate_version, 7)
})

test('visible compact safety reconciliation is shared and refcounted across consumers', () => {
  reset()
  const priorSetInterval = globalThis.setInterval
  const priorClearInterval = globalThis.clearInterval
  const scheduled: Array<{ callback: () => void; delay: number }> = []
  const cleared: unknown[] = []
  const reconciled: string[] = []
  const originalReconcile = useCampaignStore.getState().reconcileCampaign
  globalThis.setInterval = ((callback: () => void, delay: number) => {
    scheduled.push({ callback, delay })
    return scheduled.length as unknown as ReturnType<typeof setInterval>
  }) as typeof setInterval
  globalThis.clearInterval = ((timer: unknown) => { cleared.push(timer) }) as typeof clearInterval
  useCampaignStore.setState({
    reconcileCampaign: async (workspaceId, campaignId) => {
      reconciled.push(`${workspaceId}/${campaignId}`)
    },
  })
  try {
    const releaseNode = retainCampaignSafetyReconcile('workspace-a', 'campaign-1')
    const releaseControlRoom = retainCampaignSafetyReconcile('workspace-a', 'campaign-1')
    assert.equal(scheduled.length, 1)
    assert.ok(scheduled[0].delay >= 15_000 && scheduled[0].delay <= 20_000)
    scheduled[0].callback()
    assert.deepEqual(reconciled, ['workspace-a/campaign-1'])
    releaseNode()
    assert.equal(cleared.length, 0)
    releaseControlRoom()
    assert.equal(cleared.length, 1)
  } finally {
    useCampaignStore.setState({ reconcileCampaign: originalReconcile })
    globalThis.setInterval = priorSetInterval
    globalThis.clearInterval = priorClearInterval
  }
})

test('disconnect during retry backoff settles the old generation and a newer hint reconciles', async () => {
  reset()
  installList(() => [campaign('workspace-a', 'campaign-1')])
  const snapshotAt = (cursor: number, version: number) => controlRoomSnapshot({
    latest_event_cursor: cursor,
    aggregate_version: version,
    campaign: { ...controlRoomSnapshot().campaign, aggregate_version: version },
  })
  campaignApi.snapshot = async () => ({ ok: true, data: snapshotAt(42, 7) })
  await useCampaignStore.getState().load('workspace-a', 'campaign-1')
  await useCampaignStore.getState().handleConnection(true, 1)
  await useCampaignStore.getState().handleSubscription('workspace-a', true, 1)

  const first = useCampaignStore.getState().handleHint({
    schema_version: 'campaign_hint.v1', workspace_id: 'workspace-a', campaign_id: 'campaign-1',
    event_cursor: 50, aggregate_version: 8, event_type: 'campaign:advanced',
    correlation_id: 'correlation-50', emitted_at: '2026-07-16T18:00:00Z',
  })
  await new Promise((resolve) => setTimeout(resolve, 100))
  await useCampaignStore.getState().handleConnection(false, 2)
  await Promise.race([
    first,
    new Promise((_, reject) => setTimeout(() => reject(new Error('old retry did not settle')), 500)),
  ])

  campaignApi.snapshot = async () => ({ ok: true, data: snapshotAt(60, 9) })
  await useCampaignStore.getState().handleConnection(true, 3)
  await useCampaignStore.getState().handleSubscription('workspace-a', true, 3)
  campaignApi.snapshot = async () => ({ ok: true, data: snapshotAt(61, 10) })
  await useCampaignStore.getState().handleHint({
    schema_version: 'campaign_hint.v1', workspace_id: 'workspace-a', campaign_id: 'campaign-1',
    event_cursor: 61, aggregate_version: 10, event_type: 'campaign:advanced',
    correlation_id: 'correlation-61', emitted_at: '2026-07-16T18:00:01Z',
  })

  const detail = useCampaignStore.getState().workspaces['workspace-a']?.details['campaign-1']
  assert.equal(detail?.snapshot?.latest_event_cursor, 61)
  assert.equal(detail?.freshness, 'live')
})

test.after(() => reset())
