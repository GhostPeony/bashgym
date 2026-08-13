import assert from 'node:assert/strict'
import test from 'node:test'
import { composeAgentWorkspaceContext, hermesWorkspaceSessionKey } from './agentWorkspaceContext'

test('scopes the legacy shared memory key to the active workspace', () => {
  assert.equal(hermesWorkspaceSessionKey('bashgym-canvas', 'memexai'), 'bashgym:memexai')
  assert.equal(hermesWorkspaceSessionKey('', 'default'), 'bashgym:default')
  assert.equal(hermesWorkspaceSessionKey('bashgym:{workspace_id}', 'research'), 'bashgym:research')
})

test('preserves a deliberately configured custom Hermes memory key', () => {
  assert.equal(
    hermesWorkspaceSessionKey('shared-research-memory', 'memexai'),
    'shared-research-memory'
  )
})

test('puts the authoritative BashGym projection before endpoint-only details', () => {
  const context = composeAgentWorkspaceContext(
    '# BashGym Workspace Context\n\n## Training Sessions / Campaigns',
    '## Hermes endpoint details\n- endpoint: local-agent-host'
  )
  assert.ok(context.indexOf('BashGym Evidence Rules') < context.indexOf('Training Sessions'))
  assert.ok(context.indexOf('Training Sessions') < context.indexOf('Hermes endpoint'))
  assert.match(context, /live runtime > durable ledger/)
})
