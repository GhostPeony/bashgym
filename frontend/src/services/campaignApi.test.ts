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
      },
    },
  }
  try {
    await campaignApi.list('workspace-a')
    await campaignApi.attempts('workspace-a', 'campaign-1')
    await campaignApi.studies('workspace-a', 'campaign-1')
    await campaignApi.proposals('workspace-a', 'campaign-1')
    await campaignApi.evidence('workspace-a', 'campaign-1')
    await campaignApi.comparisons('workspace-a', 'campaign-1')
    await campaignApi.ledger('workspace-a', 'campaign-1')
    await campaignApi.metrics(
      'workspace-a', 'campaign-1', 'attempt-1', 'training_metrics.jsonl', 'loss',
    )
    await campaignApi.transition('workspace-a', 'campaign-1', 'pause', 4, 'operator pause')
  } finally {
    if (priorWindow === undefined) delete globals.window
    else globals.window = priorWindow
  }

  assert.deepEqual(calls.map((call) => call[1]), [
    '/api/campaigns',
    '/api/campaigns/campaign-1/attempts',
    '/api/campaigns/campaign-1/studies',
    '/api/campaigns/campaign-1/proposals',
    '/api/campaigns/campaign-1/evidence',
    '/api/campaigns/campaign-1/comparisons',
    '/api/campaigns/campaign-1/ledger',
    '/api/campaigns/campaign-1/attempts/attempt-1/metrics',
    '/api/campaigns/campaign-1/pause',
  ])
  assert.equal(
    calls.some((call) => call.some((value) => String(value).includes('Authorization'))),
    false,
  )
})
