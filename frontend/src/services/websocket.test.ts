import assert from 'node:assert/strict'
import test from 'node:test'

import { MessageTypes, WebSocketService } from './websocket'

class FakeSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  readyState = FakeSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onclose: ((event: { code: number; reason: string }) => void) | null = null
  onerror: ((error: unknown) => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null

  send(message: string) {
    this.sent.push(message)
  }
  close() {
    this.readyState = 3
    this.onclose?.({ code: 1000, reason: 'closed' })
  }
  open() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }
  message(type: string, payload: unknown) {
    this.onmessage?.({ data: JSON.stringify({ type, payload }) })
  }
}

const flush = () => new Promise((resolve) => setImmediate(resolve))

test('campaign subscriptions are refcounted and reconnect mints a fresh ticket', async () => {
  const sockets: FakeSocket[] = []
  const tickets: string[] = []
  const reconnects: Array<() => void> = []
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => {
      const socket = new FakeSocket()
      sockets.push(socket)
      return socket as unknown as WebSocket
    },
    liveTicket: async (workspaceId) => {
      const ticket = `ticket-${workspaceId}-${tickets.length + 1}`
      tickets.push(ticket)
      return { ok: true, data: { ticket } }
    },
    scheduleReconnect: (callback) => {
      reconnects.push(callback)
      return 1 as never
    },
    handleConnection: async () => {}
  })
  const releaseA = service.retainCampaignWorkspace('workspace-a')
  const releaseB = service.retainCampaignWorkspace('workspace-a')

  service.connect()
  service.connect()
  assert.equal(sockets.length, 1)
  sockets[0].open()
  await flush()
  assert.equal(tickets.length, 1)
  assert.deepEqual(JSON.parse(sockets[0].sent[0]), {
    type: 'campaign:subscribe',
    payload: { ticket: 'ticket-workspace-a-1' }
  })

  releaseA()
  assert.equal(sockets[0].sent.length, 1)
  sockets[0].onclose?.({ code: 1006, reason: 'network' })
  assert.equal(reconnects.length, 1)
  reconnects[0]()
  sockets[1].open()
  await flush()
  assert.equal(tickets.length, 2)

  releaseB()
  assert.deepEqual(JSON.parse(sockets[1].sent.at(-1)!), {
    type: 'campaign:unsubscribe',
    payload: { workspace_id: 'workspace-a' }
  })
})

test('campaign hints validate exactly and never enter Activity', async () => {
  const socket = new FakeSocket()
  const hints: unknown[] = []
  const activity: string[] = []
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => ({ ok: false }),
    handleCampaignHint: async (hint) => {
      hints.push(hint)
    },
    handleConnection: async () => {},
    addActivity: (type) => {
      activity.push(type)
    }
  })
  service.connect()
  socket.open()
  socket.message(MessageTypes.CAMPAIGN_HINT, {
    schema_version: 'campaign_hint.v1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-a',
    event_cursor: 4,
    aggregate_version: 2,
    event_type: 'campaign:validation-started',
    correlation_id: 'correlation-a',
    emitted_at: '2026-07-16T18:00:00Z'
  })
  socket.message(MessageTypes.CAMPAIGN_HINT, {
    schema_version: 'campaign_hint.v1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-a',
    event_cursor: 5,
    aggregate_version: 3,
    event_type: 'campaign:ready',
    correlation_id: 'correlation-b',
    emitted_at: '2026-07-16T18:00:01Z',
    payload: { private_path: 'C:/private' }
  })
  socket.message('campaign:subscribed', { workspace_id: 'workspace-a', accepted_cursor: 3 })
  await flush()

  assert.equal(hints.length, 1)
  assert.deepEqual(activity, [])
  service.disconnect()
})

test('pending subscribe acknowledgement prevents duplicate ticket minting', async () => {
  const socket = new FakeSocket()
  let tickets = 0
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => ({ ok: true, data: { ticket: `ticket-${++tickets}` } }),
    handleConnection: async () => {}
  })
  const releaseA = service.retainCampaignWorkspace('workspace-a')
  service.connect()
  socket.open()
  await flush()
  const releaseB = service.retainCampaignWorkspace('workspace-a')
  await flush()

  assert.equal(tickets, 1)
  releaseA()
  releaseB()
})

test('intentional disconnect does not reconnect', () => {
  const socket = new FakeSocket()
  let reconnects = 0
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => ({ ok: false }),
    scheduleReconnect: () => {
      reconnects += 1
      return 1 as never
    },
    handleConnection: async () => {}
  })
  service.connect()
  socket.open()
  service.disconnect()
  assert.equal(reconnects, 0)
})

test('failed live-ticket mint retries with a fresh bounded attempt and then subscribes', async () => {
  const socket = new FakeSocket()
  const scheduled: Array<{ callback: () => void; delay: number }> = []
  let tickets = 0
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => {
      tickets += 1
      return tickets === 1 ? { ok: false } : { ok: true, data: { ticket: 'ticket-fresh' } }
    },
    scheduleReconnect: (callback, delay) => {
      scheduled.push({ callback, delay })
      return scheduled.length as never
    },
    handleConnection: async () => {}
  })
  service.retainCampaignWorkspace('workspace-a')
  service.connect()
  socket.open()
  await flush()

  assert.equal(tickets, 1)
  assert.equal(scheduled.length, 1)
  assert.ok(scheduled[0].delay <= 8_000)
  scheduled[0].callback()
  await flush()
  assert.equal(tickets, 2)
  assert.deepEqual(JSON.parse(socket.sent.at(-1)!), {
    type: 'campaign:subscribe',
    payload: { ticket: 'ticket-fresh' }
  })
})

test('subscription error reconnects and replays every desired workspace with fresh tickets', async () => {
  const sockets: FakeSocket[] = []
  const reconnects: Array<() => void> = []
  const tickets: string[] = []
  const subscriptionStates: Array<[string, boolean, number]> = []
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => {
      const socket = new FakeSocket()
      sockets.push(socket)
      return socket as unknown as WebSocket
    },
    liveTicket: async (workspaceId) => {
      const ticket = `${workspaceId}-${tickets.length + 1}`
      tickets.push(ticket)
      return { ok: true, data: { ticket } }
    },
    scheduleReconnect: (callback) => {
      reconnects.push(callback)
      return reconnects.length as never
    },
    handleConnection: async () => {},
    handleSubscription: async (workspaceId, subscribed, generation) => {
      subscriptionStates.push([workspaceId, subscribed, generation])
    }
  })
  service.retainCampaignWorkspace('workspace-a')
  service.retainCampaignWorkspace('workspace-b')
  service.connect()
  sockets[0].open()
  await flush()
  assert.equal(tickets.length, 2)

  sockets[0].message('campaign:subscription-error', { code: 'campaign_subscription_denied' })
  assert.equal(reconnects.length, 1)
  assert.deepEqual(
    subscriptionStates.map(([workspaceId, subscribed]) => [workspaceId, subscribed]),
    [
      ['workspace-a', false],
      ['workspace-b', false]
    ]
  )
  reconnects[0]()
  sockets[1].open()
  await flush()
  assert.equal(tickets.length, 4)
})

test('subscription acknowledgement notifies the authority coordinator with socket generation', async () => {
  const socket = new FakeSocket()
  const states: Array<[string, boolean, number]> = []
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => ({ ok: true, data: { ticket: 'ticket-a' } }),
    handleConnection: async () => {},
    handleSubscription: async (workspaceId, subscribed, generation) => {
      states.push([workspaceId, subscribed, generation])
    }
  })
  service.retainCampaignWorkspace('workspace-a')
  service.connect()
  socket.open()
  await flush()
  socket.message('campaign:subscribed', { workspace_id: 'workspace-a', accepted_cursor: 0 })
  assert.deepEqual(states, [['workspace-a', true, 1]])
})

test('malformed campaign control payloads are dropped without poisoning later hints', async () => {
  const socket = new FakeSocket()
  const hints: unknown[] = []
  const service = new WebSocketService({
    url: 'ws://test/ws',
    socketFactory: () => socket as unknown as WebSocket,
    liveTicket: async () => ({ ok: false }),
    handleCampaignHint: async (value) => {
      hints.push(value)
    },
    handleConnection: async () => {}
  })
  service.connect()
  socket.open()
  socket.message('campaign:subscribed', null)
  socket.message('campaign:hint', {
    schema_version: 'campaign_hint.v1',
    workspace_id: 'workspace:a',
    campaign_id: 'campaign:a',
    event_cursor: 4,
    aggregate_version: 2,
    event_type: 'campaign:advanced',
    correlation_id: 'correlation:a',
    emitted_at: '2026-07-16T18:00:00Z'
  })
  await flush()
  assert.equal(hints.length, 1)
})
