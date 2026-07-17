import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { CampaignRecord } from '../../stores/campaignStore'
import { buildControlRoomModel } from './controlRoomModel'
import { controlRoomSnapshot } from './controlRoomFixtures'
import { ControlRoomContent } from './AutoResearchControlRoom'
import { resolveControlRoomCampaignSelection, shouldCanonicalizeControlRoomSelection } from './controlRoomSelection'
import { createRecoveryLoadGate } from './recoveryLoadGate'

const controlRoomSource = readFileSync(new URL('./AutoResearchControlRoom.tsx', import.meta.url), 'utf8')

const fleet: CampaignRecord[] = [{
  schema_version: 'campaign.v1',
  campaign_id: 'campaign-1',
  workspace_id: 'workspace-a',
  title: 'BashGym retrieval study',
  kind: 'general',
  objective: 'Improve retrieval quality.',
  target_model: {},
  owner_actor_id: 'desktop-user',
  manifest_revision: 2,
  status: 'active',
  version: 7,
  created_at: '2026-07-16T00:00:00Z',
  updated_at: '2026-07-16T18:00:00Z',
}]

const unloadedPages = {
  eventsLoading: false,
  eventsError: null,
  eventsLoaded: false,
  eventsHasMore: true,
  artifactsLoading: false,
  artifactsError: null,
  artifactsLoaded: false,
  artifactsHasMore: true,
}

function render(model = buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null })) {
  return renderToStaticMarkup(createElement(ControlRoomContent, {
    model,
    campaigns: fleet,
    selectedCampaignId: 'campaign-1',
    events: [],
    artifacts: [],
    pages: unloadedPages,
    onSelect: () => {},
    onRetry: () => {},
    onLoadEvents: () => {},
    onLoadArtifacts: () => {},
    onStart: () => {},
    startPending: false,
  }))
}

test('keeps Start in the compact authority rail and gates it with live server readiness', () => {
  const base = controlRoomSnapshot()
  const ready = controlRoomSnapshot({
    campaign: { ...base.campaign, status: 'ready' },
    active_work: null,
  })
  const live = render(buildControlRoomModel({ snapshot: ready, freshness: 'live', error: null }))
  assert.match(live, /Launch authority/)
  assert.match(live, /Start campaign/)
  assert.doesNotMatch(live, /<button[^>]*disabled=""[^>]*aria-label="Start campaign"/)

  const stale = render(buildControlRoomModel({ snapshot: ready, freshness: 'stale', error: null }))
  assert.match(stale, /<button[^>]*disabled=""[^>]*aria-label="Start campaign"/)

  const blocked = render(buildControlRoomModel({
    snapshot: { ...ready, readiness: { ...ready.readiness, launch_ready: false, blocking_codes: ['controller_offline'] } },
    freshness: 'live',
    error: null,
  }))
  assert.match(blocked, /Controller Offline/)
  assert.match(blocked, /<button[^>]*disabled=""[^>]*aria-label="Start campaign"/)
})

test('pagination follows mutable page state, exposes safe fields, and reports retryable failures', () => {
  const event = {
    cursor: 12,
    event: {
      schema_version: 'public_campaign_event.v1' as const,
      event_id: 'event-safe-12', workspace_id: 'workspace-a', campaign_id: 'campaign-1',
      sequence: 12, aggregate_version: 7, event_type: 'campaign:attempt_completed',
      summary: {
        schema_version: 'public_campaign_event_summary.v1' as const,
        action_id: 'action-safe', attempt_id: 'attempt-safe', stage: 'development_evaluation',
      },
      actor_id: 'controller-safe', credential_kind: 'controller', created_at: '2026-07-16T18:01:00Z',
    },
  }
  const artifact = {
    schema_version: 'public_campaign_artifact.v1' as const,
    workspace_id: 'workspace-a', campaign_id: 'campaign-1', artifact_id: 'artifact-safe-1',
    producer_action_id: 'action-safe', sha256: 'a'.repeat(64), size_bytes: 2048,
    schema_name: 'training_metrics_jsonl.v1', sealed: true, valid: true,
    created_at: '2026-07-16T18:02:00Z',
  }
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [event], artifacts: [artifact],
    pages: {
      eventsLoading: false, eventsError: 'event page failed', eventsLoaded: true, eventsHasMore: false,
      artifactsLoading: true, artifactsError: null, artifactsLoaded: true, artifactsHasMore: true,
    },
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))

  assert.match(html, /event-safe-12/)
  assert.match(html, /Action Id/)
  assert.match(html, /action-safe/)
  assert.match(html, /development_evaluation/)
  assert.match(html, /artifact-safe-1/)
  assert.match(html, new RegExp(`a{64}`))
  assert.match(html, /sealed/)
  assert.match(html, /event page failed/)
  assert.match(html, />Retry event page</)
  assert.doesNotMatch(html, />Load more events</)
  assert.match(html, /Loading artifacts/)
})

test('renders every loaded bounded row and stops at the terminal page', () => {
  const events = Array.from({ length: 20 }, (_, index) => ({
    cursor: index + 1,
    event: {
      schema_version: 'public_campaign_event.v1' as const,
      event_id: `event-visible-${index + 1}`,
      workspace_id: 'workspace-a', campaign_id: 'campaign-1', sequence: index + 1,
      aggregate_version: 7, event_type: 'campaign:observed', summary: null,
      actor_id: 'controller-safe', credential_kind: 'controller', created_at: '2026-07-16T18:01:00Z',
    },
  }))
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [...events].reverse(), artifacts: [],
    pages: {
      eventsLoading: false, eventsError: null, eventsLoaded: true, eventsHasMore: false,
      artifactsLoading: false, artifactsError: null, artifactsLoaded: true, artifactsHasMore: false,
    },
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))
  assert.match(html, /event-visible-1/)
  assert.match(html, /event-visible-20/)
  assert.ok(html.indexOf('event-visible-1') < html.indexOf('event-visible-20'))
  assert.doesNotMatch(html, /Load more events|Load artifact page/)
})

test('renders an unknown additive campaign status with a neutral treatment', () => {
  const base = controlRoomSnapshot()
  const html = render(buildControlRoomModel({
    snapshot: controlRoomSnapshot({
      campaign: { ...base.campaign, status: 'future_remote_status' as never },
    }),
    freshness: 'live',
    error: null,
  }))
  assert.match(html, /Future Remote Status/)
  assert.match(html, /border-border-subtle bg-background-secondary text-text-secondary/)
})

test('renders the complete snapshot-backed control room with semantic structure', () => {
  const html = render()
  for (const content of [
    'Campaign', 'Live', 'Execution owner', 'Journey', 'Active work',
    'Evidence &amp; metrics', 'Budget', 'Collection counts', 'Recent activity',
  ]) assert.match(html, new RegExp(content))
  assert.match(html, /<ol/)
  assert.match(html, /<progress[^>]+value="0.6"/)
  assert.match(html, /60%/)
  assert.match(html, /BashGym/)
  assert.doesNotMatch(html, /snapshot/i)

  const idleHtml = render(buildControlRoomModel({
    snapshot: controlRoomSnapshot({ active_work: null }),
    freshness: 'live',
    error: null,
  }))
  assert.doesNotMatch(idleHtml, /snapshot/i)
})

test('stale cached content announces status and exposes no enabled mutation', () => {
  const html = render(buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'stale', error: 'timeout' }))
  assert.match(html, /role="status"/)
  assert.match(html, /Cached campaign state · stale/)
  assert.doesNotMatch(html, /<button[^>]*>[^<]*(Start|Pause|Resume|Cancel)/)
})

test('puts a human blocker ahead of secondary panels', () => {
  const base = controlRoomSnapshot()
  const blocker = {
    schema_version: 'decision_blocker.v1' as const,
    code: 'human_review_required',
    summary: 'A blinded human review is required.',
    evidence_ids: ['evidence-1'],
    secondary_codes: [],
  }
  const html = render(buildControlRoomModel({
    snapshot: controlRoomSnapshot({
      decision_surface: { ...base.decision_surface, attention_owner: 'human', blocker },
    }),
    freshness: 'live',
    error: null,
  }))
  assert.ok(html.indexOf('A blinded human review is required.') < html.indexOf('Active work'))
})

test('mounts durable human oversight in the primary work column before evidence', () => {
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
    humanOversight: createElement('section', { 'aria-label': 'Durable human oversight' }, 'Human queue projection'),
  }))
  assert.match(html, /Human queue projection/)
  assert.ok(html.indexOf('Human queue projection') < html.indexOf('Evidence &amp; metrics'))
})

test('keeps coding-agent launch and activation out of the primary control room journey', () => {
  assert.doesNotMatch(controlRoomSource, /CampaignAgentControl|campaignAgent|Codex|Hermes/)
  assert.match(controlRoomSource, /GuidedAutoResearchSetup/)
  assert.match(controlRoomSource, /HumanOversightQueue/)
  assert.match(controlRoomSource, /CampaignRecoveryPanel/)
  assert.match(controlRoomSource, /handleStart/)
})

test('keeps recovery visibility in the primary work column without replacing campaign evidence', () => {
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
    campaignRecovery: createElement('section', { 'aria-label': 'Durable campaign recovery' }, 'Recovery authority projection'),
  }))
  assert.match(html, /Recovery authority projection/)
  assert.match(html, /Evidence &amp; metrics/)
  assert.ok(html.indexOf('Recovery authority projection') < html.indexOf('Evidence &amp; metrics'))
})

test('renders durable zero-campaign and initial error explanations', () => {
  const emptyHtml = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: null, freshness: 'live', error: null }),
    campaigns: [], selectedCampaignId: null, events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))
  assert.match(emptyHtml, /No durable AutoResearch campaigns yet/)
  assert.match(emptyHtml, /Guided setup/)
  assert.match(emptyHtml, /Private compute is primary/)
  assert.doesNotMatch(emptyHtml, /prototype|NeMo|hosted compute/i)

  const errorHtml = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: null, freshness: 'error', error: 'The authenticated campaign service could not start.' }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))
  assert.match(errorHtml, /AutoResearch control room unavailable/)
  assert.match(errorHtml, /authenticated campaign service could not start/)
  assert.doesNotMatch(errorHtml, /snapshot unavailable/i)
  assert.match(errorHtml, />Retry</)
  assert.match(errorHtml, /<select/)
  assert.match(errorHtml, /BashGym retrieval study/)
  for (const section of [
    'Campaign overview', 'Controller', 'Journey', 'Active work',
    'Evidence &amp; metrics', 'Budget', 'Collection counts', 'Recent activity',
  ]) assert.match(errorHtml, new RegExp(section))
})

test('initial fleet loading does not flash the zero-campaign explanation', () => {
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: null, freshness: 'reconciling', error: null }),
    campaigns: [], selectedCampaignId: null, events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))
  assert.match(html, /Loading AutoResearch control room/)
  assert.doesNotMatch(html, /snapshot/i)
  assert.doesNotMatch(html, /No durable AutoResearch campaigns yet/)
  assert.match(html, /<select/)
  assert.match(html, /Guided setup/)
  assert.match(html, /Model identity &amp; revision/)
})

test('wires zero-campaign setup to authoritative guided endpoints without fabricated receipts', () => {
  assert.match(controlRoomSource, /campaignApi\.guidedSetupContext/)
  assert.match(controlRoomSource, /campaignApi\.advanceGuidedSetupSession/)
  assert.match(controlRoomSource, /campaignApi\.doctorGuidedSetup/)
  assert.match(controlRoomSource, /campaignApi\.validateGuidedSetup/)
  assert.match(controlRoomSource, /campaignApi\.createGuidedSetup/)
  assert.match(controlRoomSource, /GuidedAutoResearchSetup/)
  assert.doesNotMatch(controlRoomSource, /buildSetupFlowModel\(\{\s*receipts:\s*\[\]/)
})

test('keeps the complete control room shell visible while the service is offline', () => {
  const html = renderToStaticMarkup(createElement(ControlRoomContent, {
    model: buildControlRoomModel({ snapshot: null, freshness: 'offline', error: 'The authenticated campaign service could not reconnect.' }),
    campaigns: fleet, selectedCampaignId: 'campaign-1', events: [], artifacts: [], pages: unloadedPages,
    onSelect: () => {}, onRetry: () => {}, onLoadEvents: () => {}, onLoadArtifacts: () => {},
  }))

  assert.match(html, /AutoResearch service is offline/)
  assert.match(html, /BashGym retrieval study/)
  assert.match(html, /Live authority unavailable/)
  assert.match(html, /Last-known campaign data will remain visible/)
  assert.match(html, /Setup/)
  assert.match(html, /Decision/)
  assert.match(html, /Recent activity/)
})

test('mounts recovery observability for every selected campaign without a hard-coded disabled gate', () => {
  assert.doesNotMatch(controlRoomSource, /requestsEnabled=\{false\}/)
  assert.doesNotMatch(controlRoomSource, /humanSnapshotCursor === null \|\| !recoveryRequested/)
  assert.match(controlRoomSource, /if \(!selectedCampaignId\) return\s+void loadRecovery\(\)/)
  assert.match(controlRoomSource, /selectedCampaignId\s*\?\s*recoverySnapshot/)
  assert.match(controlRoomSource, /latest_execution\?\.status/)
})

test('deduplicates recovery polling per selected scope without letting an old scope clear the new guard', async () => {
  const run = createRecoveryLoadGate()
  const releases: Array<() => void> = []
  let operations = 0
  const pending = () => new Promise<void>((resolve) => {
    operations += 1
    releases.push(resolve)
  })

  const firstA = run('workspace-a\u0000campaign-1', pending)
  const secondA = run('workspace-a\u0000campaign-1', pending)
  assert.equal(firstA, secondA)
  await Promise.resolve()
  assert.equal(operations, 1)

  const firstB = run('workspace-b\u0000campaign-1', pending)
  await Promise.resolve()
  assert.equal(operations, 2)
  releases[0]?.()
  await firstA
  const secondB = run('workspace-b\u0000campaign-1', pending)
  assert.equal(firstB, secondB)
  assert.equal(operations, 2)
  releases[1]?.()
  await firstB
})

test('canonical campaign selection remains authoritative while preferred loading is deferred or fails', () => {
  assert.equal(resolveControlRoomCampaignSelection('campaign-b', 'campaign-a'), 'campaign-b')
  assert.equal(resolveControlRoomCampaignSelection(null, 'campaign-a'), 'campaign-a')

  const campaigns = [
    fleet[0],
    { ...fleet[0], campaign_id: 'campaign-b', title: 'Campaign B' },
  ]
  assert.equal(shouldCanonicalizeControlRoomSelection({
    requestedCampaignId: 'campaign-b',
    selectedCampaignId: 'campaign-a',
    campaigns,
    loading: true,
    error: null,
  }), false)
  assert.equal(shouldCanonicalizeControlRoomSelection({
    requestedCampaignId: 'campaign-b',
    selectedCampaignId: 'campaign-a',
    campaigns,
    loading: false,
    error: 'list failed',
  }), false)
  assert.equal(shouldCanonicalizeControlRoomSelection({
    requestedCampaignId: 'campaign-missing',
    selectedCampaignId: 'campaign-a',
    campaigns,
    loading: false,
    error: null,
  }), true)
})
