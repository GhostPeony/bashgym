import assert from 'node:assert/strict'
import test from 'node:test'
import { hfApi, type HFContextBundle } from '../services/api'
import { pollHFContextBundle, useHFContextStore } from './hfContextStore'

function bundle(
  lifecycle: HFContextBundle['lifecycle'],
  completionOutcome: HFContextBundle['completion_outcome'] = null
): HFContextBundle {
  return {
    schema_version: '1',
    bundle_id: 'hfctx_test',
    version: 1,
    workspace_id: 'workspace-a',
    lifecycle,
    freshness: 'fresh',
    completion_outcome: completionOutcome,
    intent: 'Compare coding models',
    target: {},
    evidence: [],
    selected_evidence_ids: [],
    warnings: [],
    source_status: [],
    content_hash: 'a'.repeat(64),
    created_at: '2026-07-10T00:00:00Z',
    ready_at: lifecycle === 'ready' ? '2026-07-10T00:00:01Z' : null
  }
}

test('polls a collecting context until its exact version is ready', async () => {
  const responses = [bundle('collecting'), bundle('ready', 'complete')]
  let calls = 0
  const result = await pollHFContextBundle(
    bundle('collecting'),
    async () => ({ ok: true, data: responses[calls++] }),
    { wait: async () => undefined }
  )

  assert.equal(result.lifecycle, 'ready')
  assert.equal(result.completion_outcome, 'complete')
  assert.equal(calls, 2)
})

test('cancelled ready context ends polling without substituting another version', async () => {
  const cancelled = bundle('ready', 'cancelled')
  const result = await pollHFContextBundle(
    bundle('collecting'),
    async () => ({ ok: true, data: cancelled }),
    { wait: async () => undefined }
  )
  assert.equal(result, cancelled)
})

test('projection failures are exposed and never return inventory fallback content', async () => {
  const original = hfApi.contextMarkdown
  useHFContextStore.setState({ workspaces: {} })
  hfApi.contextMarkdown = async () => ({
    ok: false,
    code: 'hf_private_context_blocked',
    error: 'Private context access must be refreshed.'
  })
  try {
    await assert.rejects(
      () => useHFContextStore.getState().projection('workspace-a', bundle('ready', 'complete')),
      /Private context access must be refreshed/
    )
    const workspace = useHFContextStore.getState().workspaces['workspace-a']
    assert.equal(workspace.errorCode, 'hf_private_context_blocked')
    assert.equal(workspace.error, 'Private context access must be refreshed.')
  } finally {
    hfApi.contextMarkdown = original
  }
})

test('workspace history remains isolated by workspace key', async () => {
  const original = hfApi.contextHistory
  useHFContextStore.setState({ workspaces: {} })
  hfApi.contextHistory = async (workspaceId) => ({
    ok: true,
    data: {
      bundles: [{ ...bundle('ready', 'complete'), workspace_id: workspaceId }],
      active: null
    }
  })
  try {
    await useHFContextStore.getState().load('workspace-a')
    await useHFContextStore.getState().load('workspace-b')
    assert.equal(
      useHFContextStore.getState().workspaces['workspace-a'].bundles[0].workspace_id,
      'workspace-a'
    )
    assert.equal(
      useHFContextStore.getState().workspaces['workspace-b'].bundles[0].workspace_id,
      'workspace-b'
    )
  } finally {
    hfApi.contextHistory = original
  }
})
