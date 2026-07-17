import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clearGuidedSetupIdempotencyKey,
  getOrCreateGuidedSetupIdempotencyKey,
  getOrCreateGuidedSetupSessionId,
  readGuidedSetupSessionId,
} from './guidedSetupSessionStorage'

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
    removeItem: (key: string) => { values.delete(key) },
  }
}

test('persists a valid setup session only inside its exact workspace scope', () => {
  const storage = memoryStorage()
  assert.equal(readGuidedSetupSessionId(storage, 'workspace-a'), null)
  const first = getOrCreateGuidedSetupSessionId(storage, 'workspace-a', () => 'a'.repeat(32))
  assert.equal(readGuidedSetupSessionId(storage, 'workspace-a'), first)
  const replay = getOrCreateGuidedSetupSessionId(storage, 'workspace-a', () => 'b'.repeat(32))
  const other = getOrCreateGuidedSetupSessionId(storage, 'workspace-b', () => 'c'.repeat(32))
  assert.equal(first, `setupsess_${'a'.repeat(32)}`)
  assert.equal(replay, first)
  assert.equal(other, `setupsess_${'c'.repeat(32)}`)
})

test('replaces malformed persisted state and keeps one mutation key stable until success', () => {
  const storage = memoryStorage()
  storage.setItem('bashgym.autoresearch.setup-session.v1:workspace-a', 'private/path')
  assert.equal(getOrCreateGuidedSetupSessionId(storage, 'workspace-a', () => 'd'.repeat(32)), `setupsess_${'d'.repeat(32)}`)
  const request = {
    workspaceId: 'workspace-a', sessionId: `setupsess_${'d'.repeat(32)}`,
    version: 0, step: 'template' as const, selectionId: 'template-modern',
  }
  const first = getOrCreateGuidedSetupIdempotencyKey(storage, request, () => 'e'.repeat(32))
  const replay = getOrCreateGuidedSetupIdempotencyKey(storage, request, () => 'f'.repeat(32))
  assert.equal(first, `idem_${'e'.repeat(32)}`)
  assert.equal(replay, first)
  clearGuidedSetupIdempotencyKey(storage, request)
  assert.equal(getOrCreateGuidedSetupIdempotencyKey(storage, request, () => '1'.repeat(32)), `idem_${'1'.repeat(32)}`)
})
