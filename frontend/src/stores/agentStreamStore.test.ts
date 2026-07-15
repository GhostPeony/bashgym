import assert from 'node:assert/strict'
import test from 'node:test'
import {
  registerHermesPromptSender,
  sendHermesStreamPrompt,
  useAgentStreamStore
} from './agentStreamStore'

test('publishes Hermes transcript state for lightweight stream surfaces', () => {
  useAgentStreamStore.getState().publishHermesStream({
    panelId: 'hermes-panel',
    label: 'Hermes',
    endpointId: 'hermes',
    messages: [{ role: 'assistant', content: 'Working on it' }],
    sending: true,
    activity: 'Using workspace context',
    error: null
  })

  const snapshot = useAgentStreamStore.getState().hermesStreams.get('hermes-panel')
  assert.equal(snapshot?.messages[0].content, 'Working on it')
  assert.equal(snapshot?.sending, true)
})

test('routes overlay prompts through the mounted Hermes node sender', () => {
  const prompts: string[] = []
  const unregister = registerHermesPromptSender('hermes-panel', (prompt) => {
    prompts.push(prompt)
  })

  assert.equal(sendHermesStreamPrompt('hermes-panel', 'continue'), true)
  assert.deepEqual(prompts, ['continue'])
  unregister()
  assert.equal(sendHermesStreamPrompt('hermes-panel', 'again'), false)
})
