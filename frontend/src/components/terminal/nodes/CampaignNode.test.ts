import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { CampaignDetailState } from '../../../stores/campaignStore'

const storage = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, String(value)),
    removeItem: (key: string) => storage.delete(key),
    clear: () => storage.clear(),
    key: (index: number) => Array.from(storage.keys())[index] ?? null,
    get length() { return storage.size },
  },
})

function realisticDetail(): CampaignDetailState {
  const campaign = {
    schema_version: 'campaign.v1',
    campaign_id: 'campaign-memex-candidate-b',
    workspace_id: 'workspace-memex',
    title: 'Memex embedding Candidate B',
    kind: 'embedding_finetune',
    objective: 'Improve held-out Memex retrieval without regressing the base model.',
    target_model: { base_model: 'Qwen/Qwen3-Embedding-0.6B' },
    owner_actor_id: 'codex-agent',
    manifest_revision: 4,
    status: 'active' as const,
    active_study_id: 'study-cached-mnrl',
    active_action_id: 'action-dev-eval',
    champion_ref: null,
    stop_reason: null,
    version: 8,
    created_at: '2026-07-13T20:00:00Z',
    updated_at: '2026-07-13T21:14:34Z',
  }
  const attempt = {
    attempt_id: 'candidate-b-full-b128-realized17-mb4-e2-bf16-r1',
    workspace_id: campaign.workspace_id,
    campaign_id: campaign.campaign_id,
    study_id: 'study-cached-mnrl',
    action_id: 'action-full-training',
    attempt_number: 2,
    claim_generation: 1,
    status: 'succeeded',
    input_digest: 'a'.repeat(64),
    candidate_digest: 'b'.repeat(64),
    manifest_revision: 4,
    stage: 'full_training',
    executor: { kind: 'ssh_remote', compute_profile_id: 'ssh-gpu-lab' },
    created_at: '2026-07-13T21:03:39Z',
    updated_at: '2026-07-13T21:14:34Z',
  }
  return {
    campaign,
    studies: [{
      study_id: 'study-cached-mnrl',
      workspace_id: campaign.workspace_id,
      campaign_id: campaign.campaign_id,
      proposal_id: 'proposal-candidate-b',
      status: 'evaluating',
      stage_plan: {
        items: [
          { stage: 'smoke_training', disposition: 'required', reason: 'Safety gate' },
          { stage: 'full_training', disposition: 'required', reason: 'Candidate run' },
          { stage: 'development_evaluation', disposition: 'required', reason: 'Promotion gate' },
        ],
      },
      current_stage_index: 2,
      candidate_digest: attempt.candidate_digest,
      version: 3,
      created_at: '2026-07-13T20:00:00Z',
      updated_at: '2026-07-13T21:14:34Z',
    }],
    proposals: [{
      proposal: {
        proposal_id: 'proposal-candidate-b',
        hypothesis: 'Cached MNRL negatives improve exact retrieval without a same-video regression.',
        study_family: 'embedding_finetune',
        primary_variable: 'negative_mining_strategy',
        expected_outcome: 'Exact MRR improves while same-video MRR remains stable.',
        falsification_criterion: 'Exact MRR does not improve or same-video MRR regresses.',
        estimated_cost: 1.5,
        status: 'accepted',
        created_at: '2026-07-13T20:00:00Z',
      },
      validation: { valid: true, reason_codes: [] },
      study_id: 'study-cached-mnrl',
      updated_at: '2026-07-13T20:00:00Z',
    }],
    evidence: {
      workspace_id: campaign.workspace_id,
      campaign_id: campaign.campaign_id,
      campaign_version: 8,
      manifest_revision: 4,
      status: 'active',
      objective: campaign.objective,
      champion_ref: null,
      approved_data_scopes: ['memexai-approved-training', 'memexai-heldout-dev'],
      compute_profile_id: 'ssh-gpu-lab',
      budget_remaining: { gpu_hours: 10.32, campaigns: 1 },
      proposal_counts: { accepted: 1 },
      available_executors: ['ssh_remote'],
      active_study_id: campaign.active_study_id,
      active_action_id: campaign.active_action_id,
      snapshot_digest: 'c'.repeat(64),
      created_at: '2026-07-13T21:14:34Z',
    },
    attempts: [{
      ...attempt,
      attempt_id: 'candidate-b-development-eval',
      action_id: 'action-dev-eval',
      stage: 'development_evaluation',
      status: 'succeeded',
      created_at: '2026-07-13T21:15:00Z',
      updated_at: '2026-07-13T21:20:00Z',
    }, attempt],
    artifacts: [{
      artifact_id: 'artifact-training-metrics',
      producer_action_id: 'action-full-training',
      sha256: 'd'.repeat(64),
      size_bytes: 1_572_864,
      schema_name: 'training_metrics.v1',
      sealed: true,
      valid: true,
      metadata: { metric_name: 'loss' },
      created_at: '2026-07-13T21:14:34Z',
    }],
    comparisons: [{
      champion_digest: 'e'.repeat(64),
      candidate_digest: attempt.candidate_digest,
      sample_count: 18,
      video_count: 3,
      metrics: { exact_mrr: 0.308144, same_video_mrr: 0.800926 },
      slice_metrics: { memexai_youtube: { exact_mrr: 0.308144 } },
      verdict: 'insufficient_evidence',
      blocking_reasons: ['minimum_dev_queries_not_met'],
      warnings: ['characterization_only'],
      comparison_digest: 'f'.repeat(64),
      created_at: '2026-07-13T21:20:00Z',
    }],
    ledger: {
      schema_version: 'bashgym.campaign-ledger-projection.v1',
      workspace_id: campaign.workspace_id,
      campaign_id: campaign.campaign_id,
      generated_at: '2026-07-13T21:20:01Z',
      linked: true,
      projects: [{
        project: { project_id: 'memexai', display_name: 'MemexAI retrieval', status: 'active' },
        experiments: [{ experiment_id: 'experiment-positive-aware' }],
        runs: [
          {
            run_id: 'run-base-model-baseline',
            status: 'completed',
            run_kind: 'baseline',
            is_simulation: false,
            model_version_id: 'Qwen/Qwen3-Embedding-0.6B',
            queued_at: '2026-07-13T19:00:00Z',
          },
          { run_id: attempt.attempt_id, status: 'completed', run_kind: 'training', is_simulation: false },
        ],
        evaluations: [{
          evaluation_result_id: 'eval-positive-aware-dev',
          evaluation_suite_id: 'suite-retrieval-dev-v1',
          run_id: attempt.attempt_id,
          status: 'completed',
          metrics: { exact_mrr: 0.308144 },
          created_at: '2026-07-13T21:20:00Z',
        }, {
          evaluation_result_id: 'eval-base-model-dev',
          evaluation_suite_id: 'suite-retrieval-dev-v1',
          run_id: 'run-base-model-baseline',
          status: 'completed',
          metrics: { exact_mrr: 0.271 },
          created_at: '2026-07-13T19:30:00Z',
        }],
        artifacts: [],
        decisions: [{
          decision_id: 'decision-retain-champion',
          experiment_id: 'experiment-positive-aware',
          run_id: attempt.attempt_id,
          decision_type: 'retain',
          outcome: 'Retain the current champion.',
          rationale: 'The development evidence is insufficient.',
          evidence_refs: ['eval-positive-aware-dev'],
          created_at: '2026-07-13T21:20:01Z',
        }],
        evidence: {
          experiment_ids: ['experiment-positive-aware'],
          run_ids: [attempt.attempt_id],
          evaluation_result_ids: ['eval-positive-aware-dev'],
          artifact_ids: [],
          decision_ids: ['decision-retain-champion'],
        },
      }],
    },
    events: [{
      cursor: 41,
      event: {
        event_id: 'event-metrics-appended',
        event_type: 'campaign:training-metrics-appended',
        payload: { attempt_id: attempt.attempt_id },
        idempotency_key: 'metrics-appended-once',
        created_at: '2026-07-13T21:14:34Z',
      },
    }],
    lossByAttempt: {
      [attempt.attempt_id]: [
        { step: 1, source: 'training_metrics.jsonl', value: 1.0309, observed_at: '2026-07-13T21:03:40Z' },
        { step: 42, source: 'training_metrics.jsonl', value: 0.273269, observed_at: '2026-07-13T21:09:00Z' },
        { step: 84, source: 'training_metrics.jsonl', value: 0.1738, observed_at: '2026-07-13T21:14:34Z' },
      ],
    },
    nextCursor: 41,
    loading: false,
    error: null,
  }
}

test('campaign evidence panel renders the durable API/store projection and controls', async () => {
  const { CampaignEvidencePanel } = await import('./CampaignNode')
  const detail = realisticDetail()
  const attempt = detail.attempts.at(-1)!
  const comparison = detail.comparisons.at(-1)!
  const chartData = detail.lossByAttempt[attempt.attempt_id].map((point) => ({
    step: point.step,
    loss: point.value,
  }))
  const markup = renderToStaticMarkup(createElement(CampaignEvidencePanel, {
    campaignId: detail.campaign.campaign_id,
    campaigns: [detail.campaign],
    detail,
    controller: {
      schema_version: 'campaign_controller_status.v1',
      online: false,
      state: 'stale',
      code: 'controller_stale',
      observed_at: '2026-07-13T21:20:05Z',
      heartbeat_age_seconds: 20,
      guidance: 'Restart the resident campaign worker.',
    },
    chartData,
    latestAttempt: attempt,
    latestComparison: comparison,
    mutating: false,
    onSelect: () => undefined,
    onTransition: () => undefined,
    onRefresh: () => undefined,
  }))

  assert.match(markup, /Memex embedding Candidate B/)
  assert.match(markup, />active</)
  assert.match(markup, /aria-label="Campaign research brief"/)
  assert.match(markup, /Baseline · evaluated/)
  assert.match(markup, /exact mrr 0\.271/)
  assert.match(markup, /Cached MNRL negatives improve exact retrieval/)
  assert.match(markup, /Retain the current champion/)
  assert.match(markup, /development evaluation · succeeded/)
  assert.match(markup, /aria-label="Campaign controls"/)
  assert.match(markup, /Pause<\/button>/)
  assert.match(markup, /Cancel<\/button>/)
  assert.match(markup, /Refresh<\/button>/)
  assert.match(markup, /Resident Controller/)
  assert.match(markup, /aria-label="Campaign controller stale"/)
  assert.match(markup, /heartbeat 20s ago/)
  assert.match(markup, /Restart the resident campaign worker/)

  assert.match(markup, /Loss · candidate-b-full-b128-realized17-mb4-e2-bf16-r1/)
  assert.match(markup, /aria-label="Training loss curve, 3 points, step 1 loss 1.0309 to step 84 loss 0.1738"/)
  assert.match(markup, /Latest Development Gate/)
  assert.match(markup, /insufficient evidence/)
  assert.match(markup, /18 queries \/ 3 videos/)
  assert.match(markup, /exact mrr/)
  assert.match(markup, /0\.308144/)
  assert.match(markup, /minimum_dev_queries_not_met/)
  assert.match(markup, /Linked Experiment Ledger/)
  assert.match(markup, /eval-positive-aware-dev/)
  assert.match(markup, /Decisions/)
  assert.match(markup, /Retain the current champion/)

  assert.match(markup, /aria-label="Campaign policy evidence"/)
  assert.match(markup, /ssh-gpu-lab/)
  assert.match(markup, /memexai-heldout-dev/)
  assert.match(markup, /Recent Events \(1\)/)
  assert.match(markup, /aria-label="Recent campaign events"/)
  assert.match(markup, /training-metrics-appended/)
  assert.match(markup, /Sealed Evidence \(1\)/)
  assert.match(markup, /aria-label="Sealed campaign evidence"/)
  assert.match(markup, /training metrics\.v1/)
  assert.match(markup, /1\.5 MB/)
})

test('campaign evidence panel exposes lifecycle controls for ready and paused states', async () => {
  const { CampaignEvidencePanel } = await import('./CampaignNode')
  const detail = realisticDetail()
  const renderStatus = (status: 'ready' | 'paused') => renderToStaticMarkup(createElement(
    CampaignEvidencePanel,
    {
      campaignId: detail.campaign.campaign_id,
      campaigns: [detail.campaign],
      detail: { ...detail, campaign: { ...detail.campaign, status } },
      chartData: [],
      latestAttempt: detail.attempts.at(-1),
      latestComparison: detail.comparisons.at(-1),
      mutating: false,
      onSelect: () => undefined,
      onTransition: () => undefined,
      onRefresh: () => undefined,
    },
  ))

  assert.match(renderStatus('ready'), /Start<\/button>/)
  assert.match(renderStatus('paused'), /Resume<\/button>/)
})
