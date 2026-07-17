import assert from 'node:assert/strict'
import test from 'node:test'

import {
  invokeCampaignAgentHostAction,
  launchCodexCampaignAgent,
  type CampaignAgentHostRendererAPI,
} from './campaignAgentHostClient'

test('invokes only high-level host operations with terminal, scope, capabilities, and idempotency', async () => {
  const calls: Array<[string, unknown]> = []
  const host: CampaignAgentHostRendererAPI = {
    eligible: async () => ({ success: true, sessions: [] }),
    attach: async (request) => { calls.push(['attach', request]); return { success: true, status: {} } },
    authorize: async (request) => { calls.push(['authorize', request]); return { success: true, status: {} } },
    activate: async (request) => { calls.push(['activate', request]); return { success: true, status: {} } },
    revoke: async (request) => { calls.push(['revoke', request]); return { success: true } },
  }
  const binding = {
    terminalId: 'terminal-1', workspaceId: 'workspace-a', campaignId: 'campaign-a',
    requestedCapabilities: ['campaign_observe', 'artifact_read'] as const,
    grantedCapabilities: ['campaign_observe'] as const,
    idempotencyKey: 'idem_0123456789abcdef0123456789abcdef',
  }
  await invokeCampaignAgentHostAction(host, 'register', binding)
  await invokeCampaignAgentHostAction(host, 'approve', binding)
  await invokeCampaignAgentHostAction(host, 'activate', binding)
  await invokeCampaignAgentHostAction(host, 'revoke', binding)

  assert.deepEqual(calls, [
    ['attach', { terminalId: 'terminal-1', workspaceId: 'workspace-a', campaignId: 'campaign-a' }],
    ['authorize', {
      terminalId: 'terminal-1', workspaceId: 'workspace-a', campaignId: 'campaign-a',
      requestedCapabilities: ['campaign_observe', 'artifact_read'],
      grantedCapabilities: ['campaign_observe'],
      idempotencyKey: 'idem_0123456789abcdef0123456789abcdef',
    }],
    ['activate', { terminalId: 'terminal-1', workspaceId: 'workspace-a', campaignId: 'campaign-a' }],
    ['revoke', { terminalId: 'terminal-1', workspaceId: 'workspace-a', campaignId: 'campaign-a' }],
  ])
  assert.doesNotMatch(JSON.stringify(calls), /agentOrigin|agentPrincipal|sessionId|family|bgag/i)
  assert.doesNotMatch(JSON.stringify(calls), /claim/i)
})

test('launches Codex through one high-level scope-only operation and validates the safe result', async () => {
  const calls: unknown[] = []
  const launch = async (request: { workspaceId: string; campaignId: string }) => {
    calls.push(request)
    return { success: true as const, terminalId: 'terminal-codex-2', cwd: 'C:\\workspace' }
  }
  const result = await launchCodexCampaignAgent({ launch }, {
    workspaceId: 'workspace-a', campaignId: 'campaign-a',
  })
  assert.deepEqual(calls, [{ workspaceId: 'workspace-a', campaignId: 'campaign-a' }])
  assert.deepEqual(result, { success: true, terminalId: 'terminal-codex-2', cwd: 'C:\\workspace' })
  assert.doesNotMatch(JSON.stringify(calls), /family|url|header|credential|argv|env|origin|principal|session|generation|bgag/i)
})

test('fails closed when direct launch is unavailable or returns extra authority fields', async () => {
  assert.deepEqual(await launchCodexCampaignAgent({}, {
    workspaceId: 'workspace-a', campaignId: 'campaign-a',
  }), { success: false, error: 'Direct Codex launch is unavailable.' })

  assert.deepEqual(await launchCodexCampaignAgent({
    launch: async () => ({ success: true, terminalId: 'terminal-1', credential: 'must-not-cross' }),
  }, { workspaceId: 'workspace-a', campaignId: 'campaign-a' }), {
    success: false, error: 'The desktop host returned an invalid launch result.',
  })
})

test('does not forward additive renderer fields across the launch boundary', async () => {
  let received: unknown = null
  await launchCodexCampaignAgent({
    launch: async (request) => {
      received = request
      return { success: true, terminalId: 'terminal-1', cwd: 'C:\\workspace' }
    },
  }, {
    workspaceId: 'workspace-a',
    campaignId: 'campaign-a',
    mcpUrl: 'must-not-cross',
    credential: 'must-not-cross',
  } as never)
  assert.deepEqual(received, { workspaceId: 'workspace-a', campaignId: 'campaign-a' })
})

test('turns a hostile launch projection into a bounded failure instead of throwing', async () => {
  const hostile = Object.defineProperty({}, 'success', {
    enumerable: true,
    get: () => { throw new Error('renderer must not receive this') },
  })
  assert.deepEqual(await launchCodexCampaignAgent({
    launch: async () => hostile,
  }, { workspaceId: 'workspace-a', campaignId: 'campaign-a' }), {
    success: false,
    error: 'The desktop host returned an invalid launch result.',
  })
})
