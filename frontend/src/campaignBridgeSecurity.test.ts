import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildBackendChildEnvironment,
  CampaignBridgeClient,
  MANAGED_BACKEND_STARTUP_TIMEOUT_MS,
  validateCampaignRequest,
} from '../electron/campaignBridge'

test('managed desktop backend allows a measured cold-start indexing window', () => {
  assert.equal(MANAGED_BACKEND_STARTUP_TIMEOUT_MS, 120_000)
  assert.ok(MANAGED_BACKEND_STARTUP_TIMEOUT_MS > 71_000)
})

test('campaign bridge allowlists exact REST routes and rejects renderer escape hatches', () => {
  validateCampaignRequest('GET', '/api/campaigns/templates', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/autoresearch', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/attempts', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/comparisons', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/studies', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/proposals', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/evidence', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('GET', '/api/campaigns/campaign-1/ledger', undefined, {
    workspace_id: 'workspace-a',
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/budget/amend', {
    workspace_id: 'workspace-a',
    expected_version: 4,
    resource: 'gpu_hours',
    delta: 1,
    reason: 'approved',
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/actions/action-1/force-stop', {
    workspace_id: 'workspace-a',
    expected_version: 4,
    expected_remote_process_identity: { run_id: 'run-1' },
    confirmed: true,
    reason: 'approved exact identity',
  })
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/attempts/attempt-1/metrics',
    undefined,
    {
      workspace_id: 'workspace-a',
      metric_name: 'loss',
      source: 'training_metrics.jsonl',
      after_step: 10,
      limit: 200,
    },
  )
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/start', {
    workspace_id: 'workspace-a',
    expected_version: 3,
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/protected-result', {
    workspace_id: 'workspace-a',
    expected_version: 5,
    result: {
      protected_epoch_id: 'protected-1',
      candidate_digest: 'a'.repeat(64),
      passed: true,
      metrics: { recall_at_10: 0.8 },
      artifact_sha256: 'b'.repeat(64),
    },
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/autoresearch/baseline', {
    workspace_id: 'workspace-a',
    expected_version: 4,
    proposal_id: 'baseline-1',
  })

  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaign-auth/exchange', {}),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', 'http://attacker.test/api/campaigns'),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaigns/../settings'),
    /Invalid campaign route/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaigns', undefined, {
      authorization: 'Bearer renderer-controlled',
    }),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaigns', {
      payload: 'x'.repeat(70_000),
    }),
    /too large/,
  )
})

test('backend child environment receives bootstrap without mutating or exposing base env', () => {
  const base = { PATH: 'safe-path', RENDERER_VISIBLE: 'safe' }
  const bootstrap = `bgcb.launch.${'s'.repeat(48)}`
  const child = buildBackendChildEnvironment(base, bootstrap)

  assert.equal(base.PATH, 'safe-path')
  assert.equal('BASHGYM_DESKTOP_BOOTSTRAP_SECRET' in base, false)
  assert.equal(child.BASHGYM_DESKTOP_BOOTSTRAP_SECRET, bootstrap)
  assert.equal(child.BASHGYM_MODE, 'desktop')
})

test('main-only client attaches access token and retries one 401 with stable mutation ids', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = []
  const responses = [
    new Response(JSON.stringify({ raw_token: `bgca.access-1.${'a'.repeat(48)}` }), {
      status: 200,
    }),
    new Response(JSON.stringify({ detail: { code: 'campaign_auth_required' } }), {
      status: 401,
    }),
    new Response(JSON.stringify({ raw_token: `bgca.access-2.${'b'.repeat(48)}` }), {
      status: 200,
    }),
    new Response(JSON.stringify({ campaign: { campaign_id: 'campaign-1' } }), {
      status: 200,
    }),
  ]
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async (input, init) => {
      calls.push({ url: String(input), init })
      const response = responses.shift()
      assert.ok(response)
      return response
    },
    () => 'fixed-request-id',
  )

  const result = await client.request('POST', '/api/campaigns/campaign-1/start', {
    workspace_id: 'workspace-a',
    expected_version: 3,
  })

  assert.equal(result.ok, true)
  assert.equal(calls.length, 4)
  assert.equal(calls[0].url, 'http://127.0.0.1:8003/api/campaign-auth/exchange')
  assert.equal(calls[2].url, 'http://127.0.0.1:8003/api/campaign-auth/exchange')
  const firstHeaders = calls[1].init?.headers as Record<string, string>
  const retryHeaders = calls[3].init?.headers as Record<string, string>
  assert.equal(firstHeaders.Authorization, `Bearer bgca.access-1.${'a'.repeat(48)}`)
  assert.equal(retryHeaders.Authorization, `Bearer bgca.access-2.${'b'.repeat(48)}`)
  assert.equal(firstHeaders['Idempotency-Key'], 'desktop-fixed-request-id')
  assert.equal(retryHeaders['Idempotency-Key'], 'desktop-fixed-request-id')
  assert.equal(firstHeaders['X-Correlation-ID'], 'desktop-fixed-request-id')
  assert.equal(JSON.stringify(result).includes('bgca.'), false)
  assert.equal(JSON.stringify(result).includes('bgcb.'), false)
})
