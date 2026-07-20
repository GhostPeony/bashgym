import assert from 'node:assert/strict'
import test from 'node:test'

import { createKeyedSessionResource, createSessionResource } from './sessionResource'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

test('first ensureLoaded fetches with loading=true, later mounts serve cache without fetching', async () => {
  let calls = 0
  const gate = deferred<void>()
  const store = createSessionResource<string>(async () => {
    calls += 1
    await gate.promise
    return { ok: true, data: 'hardware' }
  })

  const first = store.getState().ensureLoaded()
  assert.equal(store.getState().loading, true)
  assert.equal(store.getState().refreshing, false)
  gate.resolve()
  await first

  assert.equal(store.getState().data, 'hardware')
  assert.equal(store.getState().loading, false)
  assert.ok(store.getState().loadedAt)

  await store.getState().ensureLoaded()
  await store.getState().ensureLoaded()
  assert.equal(calls, 1)
})

test('concurrent ensureLoaded calls share one in-flight request', async () => {
  let calls = 0
  const gate = deferred<void>()
  const store = createSessionResource<number>(async () => {
    calls += 1
    await gate.promise
    return { ok: true, data: 42 }
  })

  const a = store.getState().ensureLoaded()
  const b = store.getState().ensureLoaded()
  gate.resolve()
  await Promise.all([a, b])
  assert.equal(calls, 1)
  assert.equal(store.getState().data, 42)
})

test('refresh refetches in background: cached data stays visible, refreshing not loading', async () => {
  let value = 'v1'
  const store = createSessionResource<string>(async () => ({ ok: true, data: value }))
  await store.getState().ensureLoaded()

  value = 'v2'
  const gatePromise = store.getState().refresh()
  assert.equal(store.getState().loading, false)
  assert.equal(store.getState().data, 'v1')
  await gatePromise
  assert.equal(store.getState().data, 'v2')
  assert.equal(store.getState().refreshing, false)
})

test('failed background refresh keeps stale data and surfaces error', async () => {
  let fail = false
  const store = createSessionResource<string>(async () =>
    fail ? { ok: false, error: 'backend down' } : { ok: true, data: 'good' }
  )
  await store.getState().ensureLoaded()

  fail = true
  await store.getState().refresh()
  assert.equal(store.getState().data, 'good')
  assert.equal(store.getState().error, 'backend down')

  fail = false
  await store.getState().refresh()
  assert.equal(store.getState().error, null)
})

test('thrown fetcher error is captured, first load stays data-less', async () => {
  const store = createSessionResource<string>(async () => {
    throw new Error('network exploded')
  })
  await store.getState().ensureLoaded()
  assert.equal(store.getState().data, null)
  assert.equal(store.getState().error, 'network exploded')
  assert.equal(store.getState().loading, false)
})

test('failed first load retries on next ensureLoaded', async () => {
  let calls = 0
  const store = createSessionResource<string>(async () => {
    calls += 1
    return calls === 1 ? { ok: false, error: 'boom' } : { ok: true, data: 'recovered' }
  })
  await store.getState().ensureLoaded()
  assert.equal(store.getState().data, null)
  await store.getState().ensureLoaded()
  assert.equal(store.getState().data, 'recovered')
  assert.equal(calls, 2)
})

test('invalidate makes the next ensureLoaded refetch without clearing data', async () => {
  let calls = 0
  const store = createSessionResource<number>(async () => {
    calls += 1
    return { ok: true, data: calls }
  })
  await store.getState().ensureLoaded()
  assert.equal(store.getState().data, 1)

  store.getState().invalidate()
  assert.equal(store.getState().data, 1)
  await store.getState().ensureLoaded()
  assert.equal(store.getState().data, 2)
  await store.getState().ensureLoaded()
  assert.equal(calls, 2)
})

test('setData seeds the cache and suppresses the next fetch', async () => {
  let calls = 0
  const store = createSessionResource<string>(async () => {
    calls += 1
    return { ok: true, data: 'fetched' }
  })
  store.getState().setData('pushed-via-ws')
  await store.getState().ensureLoaded()
  assert.equal(calls, 0)
  assert.equal(store.getState().data, 'pushed-via-ws')
})

test('staleAfterMs triggers a refetch after expiry but serves cache before it', async () => {
  let calls = 0
  const store = createSessionResource<number>(
    async () => {
      calls += 1
      return { ok: true, data: calls }
    },
    { staleAfterMs: 10 }
  )
  await store.getState().ensureLoaded()
  await store.getState().ensureLoaded()
  assert.equal(calls, 1)

  await new Promise((res) => setTimeout(res, 15))
  await store.getState().ensureLoaded()
  assert.equal(calls, 2)
})

test('keyed resource caches per key independently', async () => {
  const calls: string[] = []
  const store = createKeyedSessionResource<string>(async (key) => {
    calls.push(key)
    return { ok: true, data: `data:${key}` }
  })

  await store.getState().ensureLoaded('7d')
  await store.getState().ensureLoaded('30d')
  await store.getState().ensureLoaded('7d')

  assert.deepEqual(calls, ['7d', '30d'])
  assert.equal(store.getState().entries['7d']?.data, 'data:7d')
  assert.equal(store.getState().entries['30d']?.data, 'data:30d')
})

test('keyed concurrent ensureLoaded per key is single-flight', async () => {
  let calls = 0
  const gate = deferred<void>()
  const store = createKeyedSessionResource<number>(async () => {
    calls += 1
    await gate.promise
    return { ok: true, data: 1 }
  })
  const a = store.getState().ensureLoaded('x')
  const b = store.getState().ensureLoaded('x')
  gate.resolve()
  await Promise.all([a, b])
  assert.equal(calls, 1)
})

test('keyed invalidate without a key marks every known key stale', async () => {
  let generation = 0
  const store = createKeyedSessionResource<number>(async () => ({ ok: true, data: ++generation }))
  await store.getState().ensureLoaded('a')
  await store.getState().ensureLoaded('b')

  store.getState().invalidate()
  await store.getState().ensureLoaded('a')
  await store.getState().ensureLoaded('b')
  assert.equal(store.getState().entries['a']?.data, 3)
  assert.equal(store.getState().entries['b']?.data, 4)
})

test("keyed failed refresh keeps that key's stale data", async () => {
  let fail = false
  const store = createKeyedSessionResource<string>(async (key) =>
    fail ? { ok: false, error: 'nope' } : { ok: true, data: `ok:${key}` }
  )
  await store.getState().ensureLoaded('k')
  fail = true
  await store.getState().refresh('k')
  assert.equal(store.getState().entries['k']?.data, 'ok:k')
  assert.equal(store.getState().entries['k']?.error, 'nope')
})
