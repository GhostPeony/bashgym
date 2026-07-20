import assert from 'node:assert/strict'
import test from 'node:test'
import { createAgentChatSurfaceActions } from './agentChatLifecycle'

test('dismisses the chat surface without cancelling an active response', () => {
  let abortCount = 0
  let hideCount = 0
  const actions = createAgentChatSurfaceActions({
    abort: () => {
      abortCount += 1
    },
    hide: () => {
      hideCount += 1
    }
  })

  actions.dismiss()

  assert.equal(hideCount, 1)
  assert.equal(abortCount, 0)
})

test('keeps explicit Stop wired to stream cancellation', () => {
  let abortCount = 0
  let hideCount = 0
  const actions = createAgentChatSurfaceActions({
    abort: () => {
      abortCount += 1
    },
    hide: () => {
      hideCount += 1
    }
  })

  actions.stop()

  assert.equal(abortCount, 1)
  assert.equal(hideCount, 0)
})
