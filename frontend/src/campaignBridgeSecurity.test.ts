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
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/control-room-snapshot',
    undefined,
    { workspace_id: 'workspace-a' },
  )
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/human-work',
    undefined,
    { workspace_id: 'workspace-a', limit: 50 },
  )
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/agent-attachment',
    undefined,
    { workspace_id: 'workspace-a', after_sequence: 0, limit: 20 },
  )
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/recovery',
    undefined,
    { workspaceId: 'workspace-a' },
  )
  validateCampaignRequest('GET', '/api/campaigns/setup/context', undefined, {
    workspace_id: 'workspace-a', session_id: `setupsess_${'a'.repeat(32)}`,
  })
  validateCampaignRequest('POST', '/api/campaigns/setup/session', {
    workspace_id: 'workspace-a', session_id: `setupsess_${'a'.repeat(32)}`,
    expected_version: 0, step: 'template', selection_id: 'template-modern',
  }, undefined, { idempotencyKey: `idem_${'b'.repeat(32)}` })
  validateCampaignRequest('POST', '/api/campaigns/setup/doctor', {
    workspace_id: 'workspace-a', template_id: 'template-modern',
    installation_id: `ins_${'c'.repeat(32)}`, bindings: {
      model: 'model.registered', data: 'data.registered', compute: 'compute.private', evaluation: 'evaluation.registered',
    },
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/human-work/hw_0123456789abcdef/claim', {
    workspace_id: 'workspace-a', expected_campaign_revision: 1, expected_version: 1,
    expected_state: 'pending',
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/human-work/hw_0123456789abcdef/submit', {
    workspace_id: 'workspace-a', expected_campaign_revision: 1, expected_version: 2,
    expected_rubric_version: 1, decision: 'prefer_left', rationale: 'Rubric evidence.',
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/human-promotion', {
    workspace_id: 'workspace-a', receipt_id: 'hrc_0123456789abcdef',
    work_id: 'hw_0123456789abcdef', expected_campaign_revision: 1,
    expected_item_version: 3, expected_rubric_version: 1,
    expected_promotion_version: 2, expected_promotion_state: 'awaiting_human_decision',
    decision: 'hold',
  })
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/recovery/repair', {
    action: 'repair', idempotency_key: `idem_${'a'.repeat(32)}`,
    workspace_id: 'workspace-a', campaign_id: 'campaign-1',
    eligibility_receipt_id: `rcpt_${'b'.repeat(32)}`,
    doctor_evidence_id: `evd_${'c'.repeat(32)}`,
    expected_campaign_revision: 2, expected_event_cursor: 17,
    expected_aggregate_version: 9, expected_controller_lease_id: null,
    checkpoint_id: `ckpt_${'d'.repeat(32)}`, artifact_id: `art_${'e'.repeat(32)}`,
    human_confirmed: true,
  }, undefined, { idempotencyKey: `idem_${'a'.repeat(32)}` })
  validateCampaignRequest(
    'GET',
    '/api/campaigns/campaign-1/artifacts',
    undefined,
    { workspace_id: 'workspace-a', after_cursor: 'artifact-4', limit: 25 },
  )
  validateCampaignRequest('POST', '/api/campaigns/campaign-1/budget/amend', {
    workspace_id: 'workspace-a',
    expected_version: 4,
    resource: 'gpu_hours',
    delta: 1,
    reason: 'approved',
  })
  validateCampaignRequest('POST', '/api/campaigns/live-ticket', {
    workspace_id: 'workspace-a',
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
  validateCampaignRequest(
    'POST',
    '/api/campaigns/campaign-1/proposals/proposal-1/code-lineage/prepare',
    { workspace_id: 'workspace-a' },
  )

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
    () => validateCampaignRequest(
      'GET',
      '/api/campaigns/campaign-1/control-room-snapshot/private',
      undefined,
      { workspace_id: 'workspace-a' },
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaigns/setup/context', undefined, {
      workspace_id: 'workspace-a', private_path: 'C:/secret',
    }),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaigns/setup/context', undefined, {
      workspace_id: 'workspace-a', session_id: 'not-a-sealed-session',
    }),
    /guided setup/i,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaigns/setup/session', {
      workspace_id: 'workspace-a', session_id: `setupsess_${'a'.repeat(32)}`,
      expected_version: 0, step: 'template', selection_id: 'template-modern',
      private_host: 'must-not-cross',
    }),
    /guided setup/i,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaigns', undefined, {
      authorization: 'Bearer renderer-controlled',
    }),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaign-agent/sessions', {
      scope: { workspaceId: 'workspace-a', campaignId: 'campaign-a' },
    }),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest(
      'POST',
      '/api/campaign-agent/sessions/registration-1/deliveries/claim',
      {},
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaigns/campaign-1/agent-grant', {}),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaigns/campaign-1/agent-attachment', {}),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest(
      'GET',
      '/api/campaigns/campaign-1/agent-attachment',
      undefined,
      { workspace_id: 'workspace-a', after_version: 0 },
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest(
      'GET',
      '/api/campaigns/campaign-1/agent-attachment/audit',
      undefined,
      { workspace_id: 'workspace-a', after_sequence: 0, limit: 20 },
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest(
      'GET',
      '/api/campaigns/campaign-1/recovery',
      undefined,
      { workspace_id: 'workspace-a' },
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest(
      'GET',
      '/api/campaigns/campaign-1/artifacts',
      undefined,
      { workspace_id: 'workspace-a', after_cursor: 'artifact-4', uri: 'private-path' },
    ),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaigns', {
      payload: 'x'.repeat(70_000),
    }),
    /too large/,
  )
})

test('main-only campaign-agent transport uses exact authenticated routes without renderer allowlisting', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = []
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async (input, init) => {
      calls.push({ url: String(input), init })
      if (String(input).endsWith('/api/campaign-auth/exchange')) {
        return new Response(JSON.stringify({ raw_token: `bgca.access-1.${'a'.repeat(48)}` }), { status: 200 })
      }
      return new Response(JSON.stringify({ schemaVersion: 'safe-test-response' }), { status: 200 })
    },
    () => 'internal-mutation-1',
  )
  const body = {
    scope: { workspaceId: 'workspace-a', campaignId: 'campaign-a' },
    agentFamily: 'codex' as const,
    agentOrigin: 'desktop_origin_1',
    agentPrincipalId: 'codex_pty_1',
    sessionId: 'pty_session_1',
    ephemeralPublicKey: 'a'.repeat(43),
    ttlSeconds: 300,
    idempotencyKey: 'hostreg_0123456789abcdef0123456789abcdef',
  }

  await client.registerCampaignAgentHostSession(body)
  await client.issueCampaignAgentGrant('campaign-a', { action: 'grant-test' })
  await client.attachCampaignAgent('campaign-a', { action: 'attach-test' })
  await client.revokeCampaignAgentAttachment('campaign-a', 'attachment-1', { action: 'revoke-test' })
  await client.claimCampaignAgentDelivery('registration-1')
  await client.revokeCampaignAgentHostSession('registration-1')

  assert.deepEqual(calls.slice(1).map((call) => call.url), [
    'http://127.0.0.1:8003/api/campaign-agent/sessions',
    'http://127.0.0.1:8003/api/campaigns/campaign-a/agent-grant',
    'http://127.0.0.1:8003/api/campaigns/campaign-a/agent-attachment',
    'http://127.0.0.1:8003/api/campaigns/campaign-a/agent-attachment/attachment-1/revoke',
    'http://127.0.0.1:8003/api/campaign-agent/sessions/registration-1/deliveries/claim',
    'http://127.0.0.1:8003/api/campaign-agent/sessions/registration-1/revoke',
  ])
  for (const call of calls.slice(1)) {
    const headers = call.init?.headers as Record<string, string>
    assert.match(headers.Authorization, /^Bearer bgca\./)
  }
  assert.equal(calls[5].init?.body, undefined)
  assert.equal(calls[6].init?.body, undefined)
})

test('main-only credential adapters use only fixed routes and keep bgag out of body, query, and output', async () => {
  const rawCredential = `bgag.credential-1.${'z'.repeat(48)}`
  const credential = Buffer.from(rawCredential, 'utf8')
  const calls: Array<{ url: string; init?: RequestInit }> = []
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async (input, init) => {
      calls.push({ url: String(input), init })
      const url = String(input)
      if (url.endsWith('/heartbeat')) {
        return new Response(JSON.stringify({ schemaVersion: 'campaign_agent_attachment.v1' }), { status: 200 })
      }
      if (url.endsWith('/actions/observe')) {
        return new Response(JSON.stringify({ schemaVersion: 'campaign_agent_observation.v1' }), { status: 200 })
      }
      return new Response(JSON.stringify({ schemaVersion: 'campaign_agent_artifact_page.v1', items: [] }), { status: 200 })
    },
  )

  const heartbeat = await client.heartbeatCampaignAgent(credential, {
    scope: { workspaceId: 'workspace-a', campaignId: 'campaign-a' },
    agentFamily: 'codex',
    agentOrigin: 'desktop-origin-1',
    agentPrincipalId: 'codex-pty-1',
    sessionId: 'pty-session-1',
    resumeCursor: 'public-cursor-1',
    resumeSequence: 7,
    expectedResumeCursor: 'public-cursor-0',
  })
  const observation = await client.observeCampaignAsAgent(credential)
  const artifacts = await client.listCampaignArtifactsAsAgent(credential, {
    afterCursor: 'a1.ABCdef_1234',
    limit: 25,
  })

  assert.deepEqual(calls.map((call) => call.url), [
    'http://127.0.0.1:8003/api/campaign-agent/heartbeat',
    'http://127.0.0.1:8003/api/campaign-agent/actions/observe',
    'http://127.0.0.1:8003/api/campaign-agent/actions/artifacts?after_cursor=a1.ABCdef_1234&limit=25',
  ])
  calls.forEach((call) => {
    const headers = call.init?.headers as Record<string, string>
    assert.equal(headers.Authorization, `Bearer ${rawCredential}`)
    assert.equal(call.url.includes('bgag.'), false)
    assert.equal(String(call.init?.body ?? '').includes('bgag.'), false)
  })
  assert.equal(calls[0].init?.method, 'POST')
  assert.equal(calls[1].init?.method, 'GET')
  assert.equal(calls[1].init?.body, undefined)
  assert.equal(calls[2].init?.method, 'GET')
  assert.equal(calls[2].init?.body, undefined)
  assert.equal(JSON.stringify([heartbeat, observation, artifacts]).includes('bgag.'), false)

  assert.throws(
    () => validateCampaignRequest('POST', '/api/campaign-agent/heartbeat', {}),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaign-agent/actions/observe'),
    /not allowlisted/,
  )
  assert.throws(
    () => validateCampaignRequest('GET', '/api/campaign-agent/actions/artifacts'),
    /not allowlisted/,
  )
})

test('main-only credential adapters reject unbounded inputs and sanitize every failure', async () => {
  const rawCredential = `bgag.credential-2.${'y'.repeat(48)}`
  const credential = Buffer.from(rawCredential, 'utf8')
  let fetchCount = 0
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async () => {
      fetchCount += 1
      throw new Error(`transport leaked ${rawCredential}`)
    },
  )

  await assert.rejects(
    () => client.listCampaignArtifactsAsAgent(credential, { afterCursor: 'not-a-cursor', limit: 20 }),
    (error: Error) => error.message === 'Campaign agent credential request failed',
  )
  await assert.rejects(
    () => client.listCampaignArtifactsAsAgent(credential, { limit: 51 }),
    (error: Error) => error.message === 'Campaign agent credential request failed',
  )
  assert.equal(fetchCount, 0)

  await assert.rejects(
    () => client.observeCampaignAsAgent(credential),
    (error: Error) => error.message === 'Campaign agent credential request failed'
      && !error.message.includes(rawCredential),
  )
  assert.equal(fetchCount, 1)

  const echoClient = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async () => new Response(JSON.stringify({ raw_token: rawCredential }), { status: 200 }),
  )
  await assert.rejects(
    () => echoClient.observeCampaignAsAgent(credential),
    (error: Error) => error.message === 'Campaign agent credential request failed'
      && !error.message.includes(rawCredential),
  )

  const escapedEchoClient = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async () => new Response(`{"raw_token":"bgag\\u002ecredential-2.${'y'.repeat(48)}"}`, { status: 200 }),
  )
  await assert.rejects(
    () => escapedEchoClient.observeCampaignAsAgent(credential),
    (error: Error) => error.message === 'Campaign agent credential request failed',
  )

  const oversizedClient = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async () => new Response('é'.repeat(1_048_577), { status: 200 }),
  )
  await assert.rejects(
    () => oversizedClient.observeCampaignAsAgent(credential),
    (error: Error) => error.message === 'Campaign agent credential request failed',
  )
})

test('main-only credential adapters preserve only 401/403 authority status', async () => {
  const credential = Buffer.from(`bgag.credential-3.${'x'.repeat(48)}`, 'utf8')
  for (const status of [401, 403] as const) {
    const client = new CampaignBridgeClient(
      'http://127.0.0.1:8003',
      `bgcb.launch.${'s'.repeat(48)}`,
      async () => new Response(JSON.stringify({ detail: 'private backend detail' }), { status }),
    )
    await assert.rejects(
      () => client.observeCampaignAsAgent(credential),
      (error: Error & { status?: number }) => error.message === 'Campaign agent credential request failed'
        && error.status === status
        && !error.message.includes('private backend detail'),
    )
  }

  const transient = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async () => { throw Object.assign(new Error('network failure'), { status: 503 }) },
  )
  await assert.rejects(
    () => transient.observeCampaignAsAgent(credential),
    (error: Error & { status?: number }) => error.message === 'Campaign agent credential request failed'
      && error.status === undefined,
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

test('main-only client forwards an exact server-issued human-work idempotency key', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = []
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async (input, init) => {
      calls.push({ url: String(input), init })
      return String(input).endsWith('/api/campaign-auth/exchange')
        ? new Response(JSON.stringify({ raw_token: `bgca.access-1.${'a'.repeat(48)}` }), { status: 200 })
        : new Response(JSON.stringify({ queue: {} }), { status: 200 })
    },
    () => 'must-not-replace-server-key',
  )
  const serverKey = 'idem_0123456789abcdef'
  await client.request(
    'POST',
    '/api/campaigns/campaign-1/human-work/hw_0123456789abcdef/claim',
    { workspace_id: 'workspace-a', expected_campaign_revision: 1, expected_version: 1, expected_state: 'pending' },
    undefined,
    { idempotencyKey: serverKey },
  )
  assert.equal((calls[1]?.init?.headers as Record<string, string>)['Idempotency-Key'], serverKey)
  await assert.rejects(
    () => client.request('POST', '/api/campaigns/campaign-1/human-promotion', {}, undefined, {
      idempotencyKey: 'desktop-invented-key',
    }),
    /idempotency/i,
  )
})

test('campaign bridge forwards only code-specific allowlisted error metadata', async () => {
  const client = new CampaignBridgeClient(
    'http://127.0.0.1:8003',
    `bgcb.launch.${'s'.repeat(48)}`,
    async (input) => String(input).endsWith('/api/campaign-auth/exchange')
      ? new Response(JSON.stringify({ raw_token: `bgca.access-1.${'a'.repeat(48)}` }), { status: 200 })
      : new Response(JSON.stringify({
        detail: {
          code: 'campaign_event_cursor_expired',
          message: 'private backend text must stay main-only',
          details: {
            resume_cursor: 17,
            retained_from: 18,
            private_path: 'C:/private/campaign.json',
            token: `bgca.access-1.${'a'.repeat(48)}`,
            source_uri: 'file:///private/campaign.json',
            nested: { secret: 'must-not-cross-the-bridge' },
          },
          private_path: 'C:/private/campaign.json',
        },
      }), { status: 410 }),
  )

  const response = await client.request(
    'GET',
    '/api/campaigns/campaign-1/events',
    undefined,
    { workspace_id: 'workspace-a', after_cursor: 3 },
  )

  assert.deepEqual(response, {
    ok: false,
    status: 410,
    error: 'Campaign request failed with HTTP 410',
    code: 'campaign_event_cursor_expired',
    details: { resume_cursor: 17 },
  })
  assert.equal(JSON.stringify(response).includes('private'), false)
  assert.equal(JSON.stringify(response).includes('bgca.'), false)
  assert.equal(JSON.stringify(response).includes('file:'), false)
  assert.equal(JSON.stringify(response).includes('secret'), false)
})
