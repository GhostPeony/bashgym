import assert from 'node:assert/strict'
import test from 'node:test'

import { campaignApi } from './api'

test('campaign API sends only fully qualified allowlisted bridge routes', async () => {
  const calls: unknown[][] = []
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  globals.window = {
    bashgym: {
      campaignRequest: async (...args: unknown[]) => {
        calls.push(args)
        return { ok: true, status: 200, data: {} }
      }
    }
  }
  try {
    await campaignApi.list('workspace-a')
    await campaignApi.liveTicket('workspace-a')
    await campaignApi.snapshot('workspace-a', 'campaign/one')
    await campaignApi.artifacts('workspace-a', 'campaign-1', 'artifact-4', 25)
    await campaignApi.artifactPreview('workspace-a', 'campaign-1', 'artifact-1')
    await campaignApi.attempts('workspace-a', 'campaign-1')
    await campaignApi.studies('workspace-a', 'campaign-1')
    await campaignApi.proposals('workspace-a', 'campaign-1')
    await campaignApi.evidence('workspace-a', 'campaign-1')
    await campaignApi.comparisons('workspace-a', 'campaign-1')
    await campaignApi.ledger('workspace-a', 'campaign-1')
    await campaignApi.metrics(
      'workspace-a',
      'campaign-1',
      'attempt-1',
      'training_metrics.jsonl',
      'loss'
    )
    await campaignApi.transition(
      'workspace-a',
      'campaign-1',
      'conclude',
      4,
      'Operator closed the campaign after the bounded AutoResearch stop rule was reached.'
    )
    await campaignApi.humanWork('workspace-a', 'campaign-1')
    await campaignApi.claimHumanWork({
      workspaceId: 'workspace-a',
      campaignId: 'campaign-1',
      workId: 'hw_0123456789abcdef',
      expectedCampaignRevision: 1,
      expectedVersion: 1,
      expectedState: 'pending',
      idempotencyKey: 'idem_0123456789abcdef'
    })
    await campaignApi.submitHumanWork(
      {
        workspaceId: 'workspace-a',
        campaignId: 'campaign-1',
        workId: 'hw_0123456789abcdef',
        expectedCampaignRevision: 1,
        expectedVersion: 2,
        expectedRubricVersion: 1,
        expectedState: 'claimed',
        idempotencyKey: 'idem_abcdef0123456789'
      },
      'prefer_left',
      'Rubric evidence.'
    )
    await campaignApi.decideHumanPromotion(
      {
        workspaceId: 'workspace-a',
        campaignId: 'campaign-1',
        receiptId: 'hrc_0123456789abcdef',
        workId: 'hw_0123456789abcdef',
        expectedCampaignRevision: 1,
        expectedItemVersion: 3,
        expectedRubricVersion: 1,
        expectedPromotionVersion: 2,
        expectedPromotionState: 'awaiting_human_decision',
        idempotencyKey: 'idem_1111111111111111'
      },
      'hold'
    )
    await campaignApi.recovery('workspace-a', 'campaign-1')
    await campaignApi.requestRecovery({
      action: 'repair',
      idempotencyKey: `idem_${'a'.repeat(32)}`,
      workspaceId: 'workspace-a',
      campaignId: 'campaign-1',
      eligibilityReceiptId: `rcpt_${'b'.repeat(32)}`,
      doctorEvidenceId: `evd_${'c'.repeat(32)}`,
      expectedCampaignRevision: 2,
      expectedEventCursor: 17,
      expectedAggregateVersion: 9,
      expectedControllerLeaseId: null,
      checkpointId: `ckpt_${'d'.repeat(32)}`,
      artifactId: `art_${'e'.repeat(32)}`,
      humanConfirmed: true
    })
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }

  assert.deepEqual(
    calls.map((call) => call[1]),
    [
      '/api/campaigns',
      '/api/campaigns/live-ticket',
      '/api/campaigns/campaign%2Fone/control-room-snapshot',
      '/api/campaigns/campaign-1/artifacts',
      '/api/campaigns/campaign-1/artifacts/artifact-1/preview',
      '/api/campaigns/campaign-1/attempts',
      '/api/campaigns/campaign-1/studies',
      '/api/campaigns/campaign-1/proposals',
      '/api/campaigns/campaign-1/evidence',
      '/api/campaigns/campaign-1/comparisons',
      '/api/campaigns/campaign-1/ledger',
      '/api/campaigns/campaign-1/attempts/attempt-1/metrics',
      '/api/campaigns/campaign-1/conclude',
      '/api/campaigns/campaign-1/human-work',
      '/api/campaigns/campaign-1/human-work/hw_0123456789abcdef/claim',
      '/api/campaigns/campaign-1/human-work/hw_0123456789abcdef/submit',
      '/api/campaigns/campaign-1/human-promotion',
      '/api/campaigns/campaign-1/recovery',
      '/api/campaigns/campaign-1/recovery/repair'
    ]
  )
  assert.equal(
    calls.some((call) => call.some((value) => String(value).includes('Authorization'))),
    false
  )
  assert.deepEqual(calls[1]?.[2], { workspace_id: 'workspace-a' })
  assert.deepEqual(calls[2]?.[3], { workspace_id: 'workspace-a' })
  assert.deepEqual(calls[3]?.[3], {
    workspace_id: 'workspace-a',
    after_cursor: 'artifact-4',
    limit: 25
  })
  assert.deepEqual(calls[4]?.[3], { workspace_id: 'workspace-a' })
  assert.deepEqual(calls[12]?.[2], {
    workspace_id: 'workspace-a',
    expected_version: 4,
    stop_reason:
      'Operator closed the campaign after the bounded AutoResearch stop rule was reached.'
  })
  assert.deepEqual(calls.at(-5)?.[4], { idempotencyKey: 'idem_0123456789abcdef' })
  assert.deepEqual(calls.at(-4)?.[4], { idempotencyKey: 'idem_abcdef0123456789' })
  assert.deepEqual(calls.at(-3)?.[4], { idempotencyKey: 'idem_1111111111111111' })
  assert.deepEqual(calls.at(-2)?.[3], { workspaceId: 'workspace-a' })
  assert.deepEqual(calls.at(-1)?.[4], { idempotencyKey: `idem_${'a'.repeat(32)}` })
})

test('campaign API requires the authenticated desktop bridge', async () => {
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  let fetchCalls = 0
  const priorFetch = globals.fetch
  globals.window = { bashgym: {} }
  globals.fetch = async () => {
    fetchCalls += 1
    throw new Error('direct fetch must not run')
  }
  try {
    const response = await campaignApi.snapshot('workspace-a', 'campaign-1')
    assert.equal(response.ok, false)
    assert.equal(response.code, 'campaign_desktop_bridge_required')
    assert.equal(fetchCalls, 0)
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
    if (priorFetch === undefined) delete globals.fetch
    else globals.fetch = priorFetch
  }
})

test('campaign agent view uses the authenticated bridge and rejects an invalid public projection', async () => {
  const calls: unknown[][] = []
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  const payload = {
    schema_version: 'campaign_agent_public_view.v1',
    observed_at: '2026-07-16T20:01:00Z',
    scope: { workspace_id: 'workspace-a', campaign_id: 'campaign-1' },
    attachment: null,
    audit_events: []
  }
  globals.window = {
    bashgym: {
      campaignRequest: async (...args: unknown[]) => {
        calls.push(args)
        return { ok: true, status: 200, data: payload }
      }
    }
  }
  try {
    const loaded = await campaignApi.campaignAgentView('workspace-a', 'campaign-1')
    assert.equal(loaded.ok, true)
    assert.equal(loaded.data?.scope.workspaceId, 'workspace-a')
    assert.deepEqual(calls[0], [
      'GET',
      '/api/campaigns/campaign-1/agent-attachment',
      undefined,
      { workspace_id: 'workspace-a', after_sequence: 0, limit: 20 },
      undefined
    ])

    globals.window = {
      bashgym: {
        campaignRequest: async () => ({
          ok: true,
          status: 200,
          data: { ...payload, raw_token: `bgag.${'a'.repeat(40)}` }
        })
      }
    }
    const rejected = await campaignApi.campaignAgentView('workspace-a', 'campaign-1')
    assert.equal(rejected.ok, false)
    assert.equal(rejected.code, 'campaign_agent_projection_invalid')
    assert.equal(rejected.data, undefined)
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }
})

test('campaign recovery rejects an invalid bridge projection before it reaches renderer state', async () => {
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  globals.window = {
    bashgym: {
      campaignRequest: async () => ({
        ok: true,
        status: 200,
        data: {
          schema_version: 'campaign_recovery.v1',
          workspace_id: 'workspace-a',
          campaign_id: 'campaign-1',
          latest_execution: { worker_id: 'private-worker' }
        }
      })
    }
  }
  try {
    const response = await campaignApi.recovery('workspace-a', 'campaign-1')
    assert.equal(response.ok, false)
    assert.equal(response.code, 'campaign_recovery_projection_invalid')
    assert.equal(response.data, undefined)
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }
})

test('campaign API preserves structured status, code, and bounded details from the bridge', async () => {
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  globals.window = {
    bashgym: {
      campaignRequest: async () => ({
        ok: false,
        status: 410,
        code: 'campaign_event_cursor_expired',
        details: { resume_cursor: 17 },
        error: 'Campaign request failed with HTTP 410'
      })
    }
  }
  try {
    const response = await campaignApi.events('workspace-a', 'campaign-1', 42)
    assert.equal(response.status, 410)
    assert.equal(response.code, 'campaign_event_cursor_expired')
    assert.deepEqual(response.details, { resume_cursor: 17 })
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }
})

test('guided setup API uses exact authenticated routes, stable idempotency, and strict context parsing', async () => {
  const calls: unknown[][] = []
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  const required = {
    model: 'model.registered',
    data: 'data.registered',
    compute: 'compute.private',
    evaluation: 'evaluation.registered'
  }
  const context = {
    schema_version: 'guided_setup_context.v1',
    workspace_id: 'workspace-a',
    templates: [
      {
        schema_version: 'guided_setup_template.v1',
        template_id: 'template-modern',
        definition_digest: 'a'.repeat(64),
        quality_claim_eligible: true,
        required_bindings: required
      }
    ],
    installations: [],
    session: null,
    reason_codes: ['setup_session_not_started'],
    truncation: {
      truncated: false,
      reason_codes: [],
      limits: { templates: 32, installations: 32, bindings_per_kind: 32 }
    }
  }
  globals.window = {
    bashgym: {
      campaignRequest: async (...args: unknown[]) => {
        calls.push(args)
        return { ok: true, status: 200, data: context }
      }
    }
  }
  try {
    const loaded = await campaignApi.guidedSetupContext(
      'workspace-a',
      `setupsess_${'a'.repeat(32)}`
    )
    assert.equal(loaded.ok, true)
    assert.equal(loaded.data?.workspace_id, 'workspace-a')
    assert.deepEqual(calls[0], [
      'GET',
      '/api/campaigns/setup/context',
      undefined,
      { workspace_id: 'workspace-a', session_id: `setupsess_${'a'.repeat(32)}` },
      undefined
    ])

    await campaignApi.advanceGuidedSetupSession({
      workspaceId: 'workspace-a',
      sessionId: `setupsess_${'a'.repeat(32)}`,
      expectedVersion: 0,
      step: 'template',
      selectionId: 'template-modern',
      idempotencyKey: `idem_${'b'.repeat(32)}`
    })
    assert.deepEqual(calls[1], [
      'POST',
      '/api/campaigns/setup/session',
      {
        workspace_id: 'workspace-a',
        session_id: `setupsess_${'a'.repeat(32)}`,
        expected_version: 0,
        step: 'template',
        selection_id: 'template-modern'
      },
      undefined,
      { idempotencyKey: `idem_${'b'.repeat(32)}` }
    ])

    globals.window = {
      bashgym: {
        campaignRequest: async () => ({
          ok: true,
          status: 200,
          data: { ...context, private_path: 'C:/secret' }
        })
      }
    }
    const rejected = await campaignApi.guidedSetupContext('workspace-a')
    assert.equal(rejected.ok, false)
    assert.equal(rejected.code, 'campaign_guided_setup_projection_invalid')
    assert.equal(rejected.data, undefined)
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }
})

test('guided setup creation returns only the safe campaign handoff', async () => {
  const calls: unknown[][] = []
  const globals = globalThis as unknown as Record<string, unknown>
  const priorWindow = globals.window
  globals.window = {
    bashgym: {
      campaignRequest: async (...args: unknown[]) => {
        calls.push(args)
        return {
          ok: true,
          status: 200,
          data: {
            campaign: {
              workspace_id: 'workspace-a',
              campaign_id: 'campaign-new',
              title: 'New campaign',
              status: 'ready'
            },
            event: { event_id: 'event-new' },
            replayed: false,
            setup: {
              schema_version: 'guided_setup_creation.v1',
              validation_receipt_id: `setuprcpt_${'a'.repeat(32)}`,
              binding_references: {
                model: 'model.registered',
                data: 'data.registered',
                compute: 'compute.private',
                evaluation: 'evaluation.registered'
              }
            }
          }
        }
      }
    }
  }
  try {
    const response = await campaignApi.createGuidedSetup({
      workspaceId: 'workspace-a',
      campaignId: 'campaign-new',
      title: 'New campaign',
      validationReceiptId: `setuprcpt_${'a'.repeat(32)}`,
      idempotencyKey: `idem_${'b'.repeat(32)}`
    })
    assert.equal(response.ok, true)
    assert.equal(response.data?.campaign_id, 'campaign-new')
    assert.deepEqual(calls[0], [
      'POST',
      '/api/campaigns/setup/create',
      {
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-new',
        title: 'New campaign',
        validation_receipt_id: `setuprcpt_${'a'.repeat(32)}`
      },
      undefined,
      { idempotencyKey: `idem_${'b'.repeat(32)}` }
    ])
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }
})
