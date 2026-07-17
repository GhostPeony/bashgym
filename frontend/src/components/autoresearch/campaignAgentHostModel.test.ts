import assert from 'node:assert/strict'
import test from 'node:test'

import { parseCampaignAgentEligibleSessions } from './campaignAgentHostModel'

test('parses only bounded safe Codex and Hermes terminal descriptors', () => {
  assert.deepEqual(parseCampaignAgentEligibleSessions([
    { terminalId: 'terminal-codex-1', family: 'codex', state: 'eligible' },
    { terminalId: 'terminal-hermes-2', family: 'hermes', state: 'authorized' },
  ]), [
    { terminalId: 'terminal-codex-1', family: 'codex', state: 'eligible' },
    { terminalId: 'terminal-hermes-2', family: 'hermes', state: 'authorized' },
  ])
})

test('rejects unsupported families, states, duplicates, private values, and extra keys', () => {
  for (const value of [
    [{ terminalId: 'terminal-1', family: 'claude', state: 'eligible' }],
    [{ terminalId: 'terminal-1', family: 'codex', state: 'unknown' }],
    [
      { terminalId: 'terminal-1', family: 'codex', state: 'eligible' },
      { terminalId: 'terminal-1', family: 'codex', state: 'registered' },
    ],
    [{ terminalId: '../private/session', family: 'codex', state: 'eligible' }],
    [{ terminalId: 'terminal-1', family: 'codex', state: 'eligible', agentOrigin: 'must-not-cross' }],
  ]) assert.equal(parseCampaignAgentEligibleSessions(value), null)
})

test('rejects non-arrays and oversized host projections', () => {
  assert.equal(parseCampaignAgentEligibleSessions({}), null)
  assert.equal(parseCampaignAgentEligibleSessions(Array.from({ length: 33 }, (_, index) => ({
    terminalId: `terminal-${index}`, family: 'codex', state: 'eligible',
  }))), null)
})
