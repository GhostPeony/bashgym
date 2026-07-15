import assert from 'node:assert/strict'
import test from 'node:test'
import { campaignApi } from '../services/api'
import { useActivityStore } from './activityStore'
import {
  useCampaignStore,
  type CampaignAttempt,
  type CampaignRecord,
} from './campaignStore'

const originals = { ...campaignApi }

function campaign(workspaceId: string, version = 1): CampaignRecord {
  return {
    schema_version: 'campaign.v1',
    campaign_id: `campaign-${workspaceId}`,
    workspace_id: workspaceId,
    title: 'Memex embedding campaign',
    kind: 'embedding_finetune',
    objective: 'Improve held-out retrieval',
    target_model: { base_model: 'Qwen' },
    owner_actor_id: 'codex-agent',
    manifest_revision: 1,
    status: 'active',
    version,
    created_at: '2026-07-13T00:00:00Z',
    updated_at: `2026-07-13T00:00:0${version}Z`,
  }
}

function attempt(workspaceId: string): CampaignAttempt {
  return {
    attempt_id: `attempt-${workspaceId}`,
    workspace_id: workspaceId,
    campaign_id: `campaign-${workspaceId}`,
    study_id: 'study-1',
    action_id: 'action-1',
    attempt_number: 1,
    claim_generation: 1,
    status: 'succeeded',
    input_digest: 'a'.repeat(64),
    candidate_digest: 'b'.repeat(64),
    manifest_revision: 1,
    stage: 'full_training',
    executor: { kind: 'fake' },
    created_at: '2026-07-13T00:00:00Z',
    updated_at: '2026-07-13T00:00:01Z',
  }
}

function installReadMocks() {
  campaignApi.list = async (workspaceId) => ({
    ok: true,
    data: {
      campaigns: [campaign(workspaceId)],
      controller: {
        schema_version: 'campaign_controller_status.v1',
        online: false,
        state: 'offline',
        code: 'controller_offline',
        observed_at: '2026-07-13T00:00:00Z',
        guidance: 'Install or restart the per-user campaign worker.',
      },
    },
  })
  campaignApi.get = async (workspaceId) => ({ ok: true, data: campaign(workspaceId) })
  campaignApi.attempts = async (workspaceId) => ({
    ok: true,
    data: { attempts: [attempt(workspaceId)] },
  })
  campaignApi.studies = async () => ({ ok: true, data: { studies: [] } })
  campaignApi.proposals = async (workspaceId) => ({
    ok: true,
    data: {
      proposals: [{
        proposal: {
          proposal_id: `proposal-${workspaceId}`,
          hypothesis: 'A bounded change improves held-out quality.',
          study_family: 'sft',
          primary_variable: 'learning_rate',
          expected_outcome: 'Quality improves.',
          falsification_criterion: 'Held-out quality does not improve.',
          estimated_cost: 0.5,
          status: 'submitted',
          created_at: '2026-07-13T00:00:00Z',
        },
        validation: { valid: true, reason_codes: [] },
        study_id: null,
        updated_at: '2026-07-13T00:00:00Z',
      }],
    },
  })
  campaignApi.evidence = async (workspaceId) => ({
    ok: true,
    data: {
      campaign_version: 1,
      manifest_revision: 1,
      status: 'active',
      objective: 'Improve held-out retrieval',
      approved_data_scopes: ['memexai-approved-training'],
      compute_profile_id: 'ssh-gpu-lab',
      budget_remaining: { gpu_hours: 11.5 },
      proposal_counts: { submitted: 0, accepted: 1 },
      available_executors: ['fake'],
      snapshot_digest: 'c'.repeat(64),
      created_at: '2026-07-13T00:00:01Z',
      active_study_id: null,
      active_action_id: null,
      champion_ref: null,
      best_development_candidate_ref: null,
      workspace_id: workspaceId,
      campaign_id: `campaign-${workspaceId}`,
    },
  })
  campaignApi.artifacts = async () => ({ ok: true, data: { artifacts: [] } })
  campaignApi.comparisons = async () => ({ ok: true, data: { comparisons: [] } })
  campaignApi.ledger = async (workspaceId, campaignId) => ({
    ok: true,
    data: {
      schema_version: 'bashgym.campaign-ledger-projection.v1',
      workspace_id: workspaceId,
      campaign_id: campaignId,
      generated_at: '2026-07-13T00:00:01Z',
      linked: false,
      projects: [],
    },
  })
}

function reset() {
  useCampaignStore.setState({ workspaces: {} })
  useActivityStore.setState({ events: [], unread: 0 })
  Object.assign(campaignApi, originals)
}

test('campaign projections remain isolated by workspace', async () => {
  reset()
  installReadMocks()
  campaignApi.events = async () => ({ ok: true, data: { items: [], next_cursor: 0 } })
  campaignApi.metrics = async () => ({
    ok: true,
    data: { metric_name: 'loss', source: 'training_metrics.jsonl', values: [], next_after_step: -1 },
  })

  await useCampaignStore.getState().load('workspace-a')
  await useCampaignStore.getState().load('workspace-b')

  assert.equal(useCampaignStore.getState().workspaces['workspace-a'].selectedCampaignId, 'campaign-workspace-a')
  assert.equal(useCampaignStore.getState().workspaces['workspace-b'].selectedCampaignId, 'campaign-workspace-b')
  assert.equal(
    useCampaignStore.getState().workspaces['workspace-a'].details['campaign-workspace-a'].campaign.workspace_id,
    'workspace-a',
  )
  assert.equal(
    useCampaignStore.getState().workspaces['workspace-b'].details['campaign-workspace-b'].proposals[0].proposal.proposal_id,
    'proposal-workspace-b',
  )
  assert.equal(useCampaignStore.getState().workspaces['workspace-a'].controller?.state, 'offline')
})

test('controller refresh is workspace-scoped and deduplicates concurrent canvas polls', async () => {
  reset()
  installReadMocks()
  campaignApi.events = async () => ({ ok: true, data: { items: [], next_cursor: 0 } })
  campaignApi.metrics = async () => ({
    ok: true,
    data: { metric_name: 'loss', source: 'training_metrics.jsonl', values: [], next_after_step: -1 },
  })
  await useCampaignStore.getState().load('workspace-a')
  let requests = 0
  let release!: () => void
  const gate = new Promise<void>((resolve) => { release = resolve })
  campaignApi.list = async (workspaceId) => {
    requests += 1
    await gate
    return {
      ok: true,
      data: {
        campaigns: [campaign(workspaceId)],
        controller: {
          schema_version: 'campaign_controller_status.v1',
          online: true,
          state: 'online' as const,
          code: 'controller_online',
          observed_at: '2026-07-13T00:00:05Z',
          heartbeat_age_seconds: 1,
        },
      },
    }
  }

  const first = useCampaignStore.getState().refreshController('workspace-a')
  const second = useCampaignStore.getState().refreshController('workspace-a')
  release()
  await Promise.all([first, second])

  assert.equal(requests, 1)
  assert.equal(useCampaignStore.getState().workspaces['workspace-a'].controller?.state, 'online')
})

test('durable cursors append events once and loss queries use the exact source', async () => {
  reset()
  installReadMocks()
  const cursors: number[] = []
  const metricSources: string[] = []
  campaignApi.events = async (_workspaceId, _campaignId, afterCursor) => {
    const cursorBase = afterCursor ?? 0
    cursors.push(cursorBase)
    const cursor = cursorBase + 1
    return {
      ok: true,
      data: {
        items: [{
          cursor,
          event: {
            event_id: `event-${cursor}`,
            event_type: 'campaign:training-metrics-appended',
            payload: { attempt_id: 'attempt-workspace-a' },
            idempotency_key: `idem-${cursor}`,
            created_at: '2026-07-13T00:00:01Z',
          },
        }],
        next_cursor: cursor,
      },
    }
  }
  campaignApi.metrics = async (
    _workspaceId, _campaignId, _attemptId, source, metricName,
  ) => {
    metricSources.push(source)
    return {
      ok: true,
      data: {
        metric_name: metricName,
        source,
        values: [{ step: metricSources.length, source, value: 0.5, observed_at: '2026-07-13T00:00:01Z' }],
        next_after_step: metricSources.length,
      },
    }
  }

  await useCampaignStore.getState().load('workspace-a')
  await useCampaignStore.getState().refresh('workspace-a', 'campaign-workspace-a')

  const detail = useCampaignStore.getState().workspaces['workspace-a'].details['campaign-workspace-a']
  assert.deepEqual(cursors, [0, 1])
  assert.deepEqual(metricSources, ['training_metrics.jsonl', 'training_metrics.jsonl'])
  assert.deepEqual(detail.events.map((item) => item.cursor), [1, 2])
  assert.equal(useActivityStore.getState().events.length, 1)
})

test('transition refreshes authoritative campaign state after acknowledgement', async () => {
  reset()
  installReadMocks()
  campaignApi.events = async () => ({ ok: true, data: { items: [], next_cursor: 0 } })
  campaignApi.metrics = async () => ({
    ok: true,
    data: { metric_name: 'loss', source: 'training_metrics.jsonl', values: [], next_after_step: -1 },
  })
  const transitions: Array<[string, number]> = []
  campaignApi.transition = async (_workspaceId, _campaignId, action, expectedVersion) => {
    transitions.push([action, expectedVersion])
    return {
      ok: true,
      data: { campaign: campaign('workspace-a', 2), event: {}, replayed: false },
    }
  }

  await useCampaignStore.getState().load('workspace-a')
  const ok = await useCampaignStore.getState().transition(
    'workspace-a', 'campaign-workspace-a', 'pause', 1, 'operator_pause',
  )

  assert.equal(ok, true)
  assert.deepEqual(transitions, [['pause', 1]])
})

test.after(() => reset())
