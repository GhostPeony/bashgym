import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  ControlRoomContent,
  type ControlRoomContentProps
} from '../autoresearch/AutoResearchControlRoom'
import { buildControlRoomModel } from '../autoresearch/controlRoomModel'
import { controlRoomSnapshot } from '../autoresearch/controlRoomFixtures'
import { ExperimentJournal } from './ExperimentJournal'
import { readStudioView, resolveStudioWorkspace } from './studioNavigation'
import { campaignApi } from '../../services/api'
import { useAuthStore } from '../../stores/authStore'
import { useCampaignStore } from '../../stores/campaignStore'

const pages = {
  eventsLoading: false,
  eventsError: null,
  eventsLoaded: false,
  eventsHasMore: false,
  artifactsLoading: false,
  artifactsError: null,
  artifactsLoaded: false,
  artifactsHasMore: false
}
function render(
  model: ControlRoomContentProps['model'],
  studioView: ControlRoomContentProps['studioView'] = 'home'
) {
  return renderToStaticMarkup(
    createElement(ControlRoomContent, {
      model,
      studioView,
      campaigns: [],
      selectedCampaignId: null,
      events: [],
      artifacts: [],
      pages,
      onSelect() {},
      onRetry() {},
      onLoadEvents() {},
      onLoadArtifacts() {},
      journal: createElement(ExperimentJournal),
      guidedSetup: createElement('section', null, 'Registered setup context')
    })
  )
}

test('studio empty and disconnected views offer distinct safe next steps', () => {
  const empty = render(buildControlRoomModel({ snapshot: null, freshness: 'live', error: null }))
  assert.match(empty, /Prepare an experiment/)
  assert.match(empty, /href="#setup"/)
  assert.doesNotMatch(empty, /Evaluation queued/)
  const offline = render(
    buildControlRoomModel({ snapshot: null, freshness: 'offline', error: 'Connection lost' })
  )
  assert.match(offline, /Connection lost/)
  assert.match(offline, /Retry connection/)
  assert.doesNotMatch(offline, />Start</)
})

test('failed and resumed campaigns render journal with authoritative status and evidence panel', () => {
  for (const status of ['failed', 'active'] as const) {
    const base = controlRoomSnapshot()
    const snapshot = controlRoomSnapshot({ campaign: { ...base.campaign, status } })
    const markup = render(
      buildControlRoomModel({ snapshot, freshness: 'live', error: null }),
      'experiments'
    )
    assert.match(markup, /aria-label="Experiment journal"/)
    assert.match(markup, /Hypothesis/)
    assert.match(markup, /Run history/)
    assert.match(markup, /Comparison/)
    assert.match(markup, /Decision/)
    assert.match(markup, /aria-label="Evidence panel"/)
    assert.match(markup, new RegExp(status === 'active' ? 'Active' : 'Failed'))
  }
})

test('setup mounts the supplied real setup surface even when a campaign exists', () => {
  assert.match(
    render(
      buildControlRoomModel({ snapshot: controlRoomSnapshot(), freshness: 'live', error: null }),
      'setup'
    ),
    /Registered setup context/
  )
})

test('studio navigation round trips all four views and safely falls back', () => {
  for (const view of ['home', 'setup', 'experiments', 'resources'] as const)
    assert.equal(readStudioView(`#${view}`), view)
  assert.equal(readStudioView('#unknown'), 'home')
})

test('fresh pairing chooses the server-assigned workspace and rejects unrelated local or URL IDs', () => {
  assert.equal(resolveStudioWorkspace(['personal-study'], null), 'personal-study')
  assert.equal(
    resolveStudioWorkspace(['personal-study'], 'random-local-workspace'),
    'personal-study'
  )
  assert.equal(resolveStudioWorkspace(['first', 'second'], 'second'), 'second')
  assert.equal(resolveStudioWorkspace([], 'random-local-workspace'), null)
  assert.equal(resolveStudioWorkspace(['invalid/path'], null), null)
})

test('browser campaign requests use session cookies with no stored credentials', async () => {
  const priorWindow = globalThis.window
  const priorFetch = globalThis.fetch
  const calls: Array<{ url: string; options?: RequestInit }> = []
  Object.assign(globalThis, { window: {} })
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options })
    return new Response(JSON.stringify({ campaigns: [] }), { status: 200 })
  }
  try {
    await campaignApi.list('workspace with spaces')
    assert.match(calls[0].url, /campaigns\?workspace_id=workspace\+with\+spaces$/)
    assert.equal(calls[0].options?.credentials, 'include')
    assert.equal(new Headers(calls[0].options?.headers).has('Authorization'), false)
    assert.equal(new Headers(calls[0].options?.headers).has('X-API-Key'), false)
  } finally {
    globalThis.window = priorWindow
    globalThis.fetch = priorFetch
  }
})

test('an acknowledged empty workspace finishes reconciliation and exposes preparation', async () => {
  const previousList = campaignApi.list
  campaignApi.list = async () =>
    ({
      ok: true,
      data: {
        campaigns: [],
        controller: {
          schema_version: 'campaign_controller_status.v1',
          online: true,
          state: 'online',
          code: 'controller_online',
          observed_at: '2026-07-16T00:00:00Z'
        }
      }
    }) as Awaited<ReturnType<typeof campaignApi.list>>
  try {
    useCampaignStore.getState().startWorkspaceLive('studio-empty-test')
    await useCampaignStore.getState().handleSubscription('studio-empty-test', true, 1)
    assert.equal(
      useCampaignStore.getState().workspaces['studio-empty-test'].freshness,
      'reconciling'
    )
    // A selection can increment loadGeneration without having fetched the list.
    await useCampaignStore.getState().select('studio-empty-test', 'not-yet-loaded')
    await useCampaignStore.getState().handleSubscription('studio-empty-test', true, 1)
    assert.equal(
      useCampaignStore.getState().workspaces['studio-empty-test'].freshness,
      'reconciling'
    )
    await useCampaignStore.getState().load('studio-empty-test')
    await useCampaignStore.getState().handleSubscription('studio-empty-test', true, 1)
    assert.equal(useCampaignStore.getState().workspaces['studio-empty-test'].freshness, 'live')
    await useCampaignStore.getState().load('studio-empty-test')
    assert.equal(useCampaignStore.getState().workspaces['studio-empty-test'].freshness, 'live')
  } finally {
    campaignApi.list = previousList
    useCampaignStore.setState({ workspaces: {} })
  }
})

test('proposal submission uses explicit role routes and keeps optimistic version and idempotency', async () => {
  const priorWindow = globalThis.window
  const priorFetch = globalThis.fetch
  const calls: Array<{ url: string; options?: RequestInit }> = []
  Object.assign(globalThis, { window: {} })
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options })
    return new Response(
      JSON.stringify({
        record: { validation: { valid: false, reason_codes: ['recipe_not_registered'] } }
      }),
      { status: 200 }
    )
  }
  try {
    for (const role of ['baseline', 'candidate', 'general'] as const) {
      const result = await campaignApi.submitProposal(
        'campaign/one',
        role,
        { workspace_id: 'workspace-a', expected_version: 8 },
        'idem_test'
      )
      assert.equal(result.data?.record.validation.valid, false)
    }
    assert.match(calls[0].url, /campaign%2Fone\/autoresearch\/baseline$/)
    assert.match(calls[1].url, /campaign%2Fone\/autoresearch\/candidates$/)
    assert.match(calls[2].url, /campaign%2Fone\/proposals$/)
    assert.equal(new Headers(calls[1].options?.headers).get('Idempotency-Key'), 'idem_test')
    assert.equal(JSON.parse(String(calls[1].options?.body)).expected_version, 8)
  } finally {
    globalThis.window = priorWindow
    globalThis.fetch = priorFetch
  }
})

test('pairing sends code only in POST body and verifies the session before authenticating', async () => {
  const priorFetch = globalThis.fetch
  const calls: Array<{ url: string; options?: RequestInit }> = []
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options })
    return new Response(
      JSON.stringify(
        String(url).endsWith('/me') ? { id: 1, username: 'researcher' } : { ok: true }
      ),
      { status: 200 }
    )
  }
  try {
    await useAuthStore.getState().pair('temporary-test-code')
    assert.equal(calls[0].url, '/api/auth/local/pair')
    assert.equal(calls[0].options?.method, 'POST')
    assert.equal(new Headers(calls[0].options?.headers).get('X-Requested-With'), 'XMLHttpRequest')
    assert.equal(calls[0].options?.body, JSON.stringify({ code: 'temporary-test-code' }))
    assert.equal(calls[0].options?.credentials, 'include')
    assert.equal(calls[1].url, '/api/auth/me')
    assert.equal(useAuthStore.getState().isAuthenticated, true)
    globalThis.fetch = async () => new Response('{}', { status: 401 })
    await assert.rejects(useAuthStore.getState().pair('expired'), /not accepted/)
  } finally {
    globalThis.fetch = priorFetch
    useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false })
  }
})
