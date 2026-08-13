import assert from 'node:assert/strict'
import test from 'node:test'
import { destinationFor, eventKeyFor, titleFor, useActivityStore } from './activityStore'

test('derives actionable destinations, including exact AutoResearch campaign scope', () => {
  assert.deepEqual(
    destinationFor('campaign:action-failed', {
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-1',
      action_id: 'action-1'
    }),
    {
      label: 'Open AutoResearch',
      view: 'autoresearch',
      workspaceId: 'workspace-a',
      campaignId: 'campaign-1'
    }
  )
  assert.deepEqual(destinationFor('training:complete', { run_id: 'run-1' }), {
    label: 'Open Training',
    view: 'training'
  })
  assert.deepEqual(destinationFor('trace:added', { trace_id: 'trace-1' }), {
    label: 'Open Traces',
    view: 'traces'
  })
  assert.equal(destinationFor('unknown:message', {}), undefined)
})

test('deduplicates a REST training acknowledgement and its WebSocket echo', () => {
  const store = useActivityStore.getState()
  store.clear()
  const payload = { run_id: 'run_123', strategy: 'sft' }

  store.addEvent('training:queued', payload)
  useActivityStore.getState().addEvent('training:queued', payload)

  const events = useActivityStore.getState().events
  assert.equal(events.length, 1)
  assert.equal(events[0].key, 'training:queued:run_123')
  assert.equal(events[0].title, 'SFT run queued')
})

test('tracks designer jobs with stable lifecycle keys', () => {
  const payload = { job_id: 'designer_42', pipeline: 'tool_use' }

  assert.equal(eventKeyFor('designer:completed', payload), 'designer:completed:designer_42')
  assert.equal(titleFor('designer:completed', payload), 'Data Designer job complete — tool_use')
})

test('maps Hugging Face context lifecycle into the HF activity category', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('hf-context:discovery-completed', {
    evidence_count: 7,
    idempotency_key: 'hf-context:workspace-a:bundle-1:1'
  })

  const [event] = useActivityStore.getState().events
  assert.equal(event.category, 'hf')
  assert.equal(event.severity, 'success')
  assert.equal(event.title, 'Hugging Face evidence ready — 7 items')
  assert.equal(event.key, 'hf-context:workspace-a:bundle-1:1')
})

test('maps Hugging Face collecting and cancellation lifecycle safely', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('hf-context:discovery-started', {
    bundle_id: 'bundle-2',
    version: 1,
    idempotency_key: 'hf-start:bundle-2:1'
  })
  store.addEvent('hf-context:discovery-cancelled', {
    bundle_id: 'bundle-2',
    version: 1,
    evidence_count: 2,
    idempotency_key: 'hf-cancel:bundle-2:1'
  })
  const events = useActivityStore.getState().events
  assert.equal(events[0].category, 'hf')
  assert.equal(events[0].severity, 'warning')
  assert.equal(events[0].title, 'Hugging Face evidence discovery cancelled — kept 2 items')
  assert.equal(events[1].title, 'Hugging Face evidence discovery started')
  assert.equal(JSON.stringify(events).includes('excerpt'), false)
})

test('removes a stale lifecycle event by its stable key', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('designer:completed', { job_id: 'designer_42' })

  useActivityStore.getState().removeEvent('designer:completed:designer_42')

  assert.equal(useActivityStore.getState().events.length, 0)
})

test('formats and keys skill eval lifecycle events', () => {
  const payload = { run_id: 'skill_run_7', skill_name: 'review', verdict: 'effective' }

  assert.equal(eventKeyFor('skill-eval:completed', payload), 'skill-eval:completed:skill_run_7')
  assert.equal(titleFor('skill-eval:completed', payload), 'Skill eval effective — review')
})

test('compacts interleaved training progress by run', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('training:progress', { run_id: 'run_a', step: 1, loss: 1 })
  store.addEvent('trace:added', { trace_id: 'trace_1' })
  store.addEvent('training:progress', { run_id: 'run_a', step: 2, loss: 0.9 })

  const events = useActivityStore.getState().events
  assert.equal(events.filter((event) => event.type === 'training:progress').length, 1)
  assert.equal(eventKeyFor('training:progress', { run_id: 'run_a' }), 'training:progress:run_a')
})

test('compacts other noisy telemetry streams without incrementing unread', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.setOpen(false)
  store.addEvent('hf:job:metrics', { job_id: 'hf_1', loss: 1 })
  store.addEvent('trace:added', { trace_id: 'trace_1' })
  store.addEvent('hf:job:metrics', { job_id: 'hf_1', loss: 0.8 })
  store.addEvent('router:stats', { total_requests: 1 })
  store.addEvent('router:stats', { total_requests: 2 })

  const state = useActivityStore.getState()
  assert.equal(state.events.filter((event) => event.type === 'hf:job:metrics').length, 1)
  assert.equal(state.events.filter((event) => event.type === 'router:stats').length, 1)
  assert.equal(state.unread, 1)
})

test('dismisses an activity event by its visible id', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('trace:added', { trace_id: 'trace_1' })
  const eventId = useActivityStore.getState().events[0].id

  useActivityStore.getState().dismissEvent(eventId)

  assert.equal(useActivityStore.getState().events.length, 0)
})

test('stores destination metadata with an activity event', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('campaign:action-failed', {
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1',
    action_id: 'action-1'
  })
  assert.equal(useActivityStore.getState().events[0].destination?.label, 'Open AutoResearch')
  assert.equal(useActivityStore.getState().events[0].destination?.campaignId, 'campaign-1')
})

test('keeps distinct durable campaign events while deduplicating the same event id', () => {
  const store = useActivityStore.getState()
  store.clear()
  store.addEvent('campaign:started', {
    event_id: 'event-1',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1'
  })
  store.addEvent('campaign:started', {
    event_id: 'event-2',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1'
  })
  store.addEvent('campaign:started', {
    event_id: 'event-2',
    workspace_id: 'workspace-a',
    campaign_id: 'campaign-1'
  })
  assert.equal(useActivityStore.getState().events.length, 2)
})
