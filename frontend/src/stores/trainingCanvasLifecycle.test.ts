import assert from 'node:assert/strict'
import test from 'node:test'
import {
  createTrainingCorrelationId,
  inferTrainingComputeTarget,
  isTrainingRunActive,
  resolveTrainingOrigin,
  type TrainingOriginState,
  trainingQueuedPayloadFromResponse,
} from './trainingCanvasLifecycle'

test('only reports an inspectable run as live while its status is active', () => {
  for (const status of ['starting', 'running', 'paused']) {
    assert.equal(isTrainingRunActive(status), true, status)
  }

  for (const status of [undefined, null, 'idle', 'completed', 'failed', 'cancelled']) {
    assert.equal(isTrainingRunActive(status), false, String(status))
  }
})

test('uses the active terminal as the training run origin', () => {
  const state: TrainingOriginState = {
    activePanelId: 'terminal-panel',
    activeSessionId: 'term-1',
    panels: [{ id: 'terminal-panel', type: 'terminal', terminalId: 'term-1' }],
    sessions: new Map([['term-1', { id: 'term-1', agentKind: 'codex' }]]),
  }
  const origin = resolveTrainingOrigin(state)

  assert.deepEqual(origin, {
    kind: 'terminal',
    panel_id: 'terminal-panel',
    terminal_id: 'term-1',
    agent: 'codex',
  })
})

test('keeps training launched from a node attached to that node', () => {
  const state: TrainingOriginState = {
    activePanelId: 'training-panel',
    activeSessionId: 'term-1',
    panels: [{ id: 'training-panel', type: 'training' }],
    sessions: new Map(),
  }
  const origin = resolveTrainingOrigin(state)

  assert.deepEqual(origin, { kind: 'panel', panel_id: 'training-panel' })
})

test('mirrors backend compute target precedence and preserves backend metadata', () => {
  assert.equal(
    inferTrainingComputeTarget({
      strategy: 'sft',
      baseModel: 'model',
      datasetPath: 'dataset.jsonl',
      useNemoGym: true,
      useRemoteSSH: true,
      deviceId: 'pony0',
    }),
    'ssh:pony0',
  )
  assert.equal(createTrainingCorrelationId(42, () => 0), 'training-16-0000')

  const payload = trainingQueuedPayloadFromResponse({
    run_id: 'run_123',
    status: 'pending',
    strategy: 'sft',
    origin: { kind: 'terminal', panel_id: 'backend-panel', terminal_id: 'backend-term' },
    correlation_id: 'backend-correlation',
    compute_target: 'cloud',
  }, {
    strategy: 'dpo',
    baseModel: 'fallback-model',
    datasetPath: 'fallback.jsonl',
    origin: { kind: 'workspace' },
    correlationId: 'fallback-correlation',
    computeTarget: 'local',
  })

  assert.deepEqual(payload, {
    run_id: 'run_123',
    status: 'pending',
    strategy: 'sft',
    base_model: 'fallback-model',
    dataset_path: 'fallback.jsonl',
    origin: { kind: 'terminal', panel_id: 'backend-panel', terminal_id: 'backend-term' },
    correlation_id: 'backend-correlation',
    compute_target: 'cloud',
  })
})
