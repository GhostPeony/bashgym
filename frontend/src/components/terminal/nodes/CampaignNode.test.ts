import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'

import type { CampaignDetailState } from '../../../stores/campaignStore'
import { useUIStore } from '../../../stores/uiStore'

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
    schema_version: 'public_campaign_attempt.v1' as const,
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
    executor_kind: 'ssh_remote',
    created_at: '2026-07-13T21:03:39Z',
    updated_at: '2026-07-13T21:14:34Z',
  }
  return {
    snapshot: null,
    freshness: 'live',
    lastVerifiedAt: null,
    reconciliation: {
      freshness: 'live', generation: 0, connectionGeneration: 1, subscribed: true,
      appliedCursor: 0, appliedVersion: 8, targetCursor: 0, targetVersion: 8,
      semanticKey: null, inFlightGeneration: null, retryCount: 0,
      lastHintAt: null, lastVerifiedAt: null, errorCode: null,
    },
    pages: {
      events: [],
      artifacts: [],
      eventCursor: 0,
      artifactCursor: null,
      eventsHasMore: true,
      artifactsHasMore: true,
      eventsLoading: false,
      eventsLoaded: false,
      eventsError: null,
      artifactsLoading: false,
      artifactsLoaded: false,
      artifactsError: null,
    },
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
      schema_version: 'public_campaign_artifact.v1',
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-1',
      artifact_id: 'artifact-training-metrics',
      producer_action_id: 'action-full-training',
      sha256: 'd'.repeat(64),
      size_bytes: 1_572_864,
      schema_name: 'training_metrics_jsonl.v1',
      sealed: true,
      valid: true,
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
      autoresearch: {
        spec: {
          primary_metric: 'exact_mrr',
          metric_direction: 'maximize',
          stop_rules: { max_attempts: 3 },
        },
        state: {
          campaign_status: 'active',
          next_action: 'propose_candidate',
          reason_code: 'ready_for_controlled_hypothesis',
          ready_for_next_proposal: true,
          baseline_verified: true,
          best_metric: 0.271,
          attempts_used: 2,
          proposals_used: 2,
          budget_used: 1.5,
          budget_remaining: 10.32,
        },
        proposals: [],
        outcomes: [],
        diagnostics: {
          schema_version: 'autoresearch_diagnostics.v1',
          workspace_id: campaign.workspace_id,
          campaign_id: campaign.campaign_id,
          primary_metric: 'exact_mrr',
          metric_direction: 'maximize',
          low_signal: true,
          signals: [{
            code: 'checkpoint_evidence_missing',
            severity: 'warning',
            summary: 'Only the terminal candidate is evaluated; retained checkpoints cannot yet be compared.',
            evidence_references: ['eval-positive-aware-dev'],
          }],
          checkpoint_comparisons: [{
            evaluation_result_id: 'eval-positive-aware-dev',
            run_id: attempt.attempt_id,
            role: 'final',
            step: null,
            metric_name: 'exact_mrr',
            metric_value: 0.308144,
            improvement_from_previous: null,
            improvement_from_baseline: 0.037144,
          }],
          error_slices: [{
            slice_path: 'source.youtube.exact_mrr',
            direction: 'maximize',
            candidate_value: 0.308144,
            baseline_value: 0.315,
            improvement: -0.006856,
            status: 'regressed',
            evidence_references: ['eval-positive-aware-dev', 'eval-base-model-dev'],
          }],
          ranked_hypotheses: [{
            hypothesis_id: 'hypothesis-checkpoint-selection',
            rank: 1,
            action_kind: 'evaluation',
            changed_variable: 'evaluation.checkpoint_selection',
            hypothesis: 'An earlier retained checkpoint may outperform the terminal checkpoint on the same fixed suite.',
            rationale: 'The terminal checkpoint is the only measured checkpoint.',
            expected_outcome: 'Identify the best observed checkpoint.',
            falsification_criterion: 'All retained checkpoints underperform the terminal checkpoint.',
            evidence_references: ['eval-positive-aware-dev'],
            eligible_for_submission: false,
          }],
        },
      },
    },
    events: [{
      cursor: 41,
      event: {
        schema_version: 'public_campaign_event.v1',
        event_id: 'event-metrics-appended',
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-1',
        sequence: 41,
        aggregate_version: 7,
        event_type: 'campaign:training-metrics-appended',
        summary: {
          schema_version: 'public_campaign_event_summary.v1',
          attempt_id: attempt.attempt_id,
        },
        actor_id: 'campaign-controller',
        credential_kind: 'controller',
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
  const comparison = detail.comparisons.at(-1)!
  const markup = renderToStaticMarkup(createElement(CampaignEvidencePanel, {
    campaignId: detail.campaign.campaign_id,
    campaigns: [detail.campaign],
    detail,
    latestComparison: comparison,
    mutating: false,
    onSelect: () => undefined,
    onTransition: () => undefined,
    onRefresh: () => undefined,
    onOpenAutoResearch: () => undefined,
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
  assert.match(markup, /Open in AutoResearch<\/button>/)
  assert.doesNotMatch(markup, /Resident Controller/)

  assert.match(markup, /Baseline vs Candidate/)
  assert.match(markup, /27\.1%/)
  assert.match(markup, /30\.8%/)
  assert.match(markup, /\+3\.7 pp/)
  assert.match(markup, /Same fixed evaluation suite/)
  assert.ok(markup.indexOf('Baseline vs Candidate') < markup.indexOf('Campaign research brief'))

  assert.match(markup, /Training loss/)
  assert.match(markup, /aria-label="Training loss curve, 3 points, minimum 0.1738 at step 84, final 0.1738 at step 84"/)
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
  assert.match(markup, /The development evidence is insufficient/)
  assert.match(markup, /decision-retain-champion/)
  assert.match(markup, /AutoResearch Diagnostics/)
  assert.match(markup, /aria-label="AutoResearch diagnostics"/)
  assert.match(markup, /Low signal detected/)
  assert.match(markup, /checkpoint evidence missing/)
  assert.match(markup, /Checkpoint trajectory/)
  assert.match(markup, /source\.youtube\.exact mrr/)
  assert.match(markup, /Ranked next hypotheses · advisory only/)
  assert.match(markup, /An earlier retained checkpoint may outperform/)

  assert.match(markup, /aria-label="Campaign policy evidence"/)
  assert.match(markup, /ssh-gpu-lab/)
  assert.match(markup, /memexai-heldout-dev/)
  assert.match(markup, /Recent Events \(1\)/)
  assert.match(markup, /aria-label="Recent campaign events"/)
  assert.match(markup, /Training Metrics Appended/)
  assert.match(markup, /<button[^>]+aria-label="Inspect Training Metrics Appended"/)
  assert.match(markup, /Sealed Evidence \(1\)/)
  assert.match(markup, /aria-label="Sealed campaign evidence"/)
  assert.match(markup, /Training metrics captured/)
  assert.match(markup, /<button[^>]+aria-label="Inspect Training metrics captured"/)
  assert.match(markup, /1\.5 MB · sealed and valid/)
  assert.doesNotMatch(markup, /text-\[(?:7|8|9)px\]/)
  assert.doesNotMatch(markup, />revision 4<\/span>/)
  assert.doesNotMatch(markup, />version 8<\/span>/)
})

test('campaign control-room deep link uses exact workspace and campaign IDs', async () => {
  const { openCampaignInAutoResearch } = await import('./campaignNodeActions')
  const prior = useUIStore.getState().openTraining
  const calls: unknown[][] = []
  useUIStore.setState({
    openTraining: (subview, selection) => { calls.push([subview, selection]) },
  })
  try {
    openCampaignInAutoResearch('workspace-exact', 'campaign-exact')
    assert.deepEqual(calls, [[
      'autoresearch',
      { workspaceId: 'workspace-exact', campaignId: 'campaign-exact' },
    ]])
  } finally {
    useUIStore.setState({ openTraining: prior })
  }
})

test('campaign node eagerly loads the bounded outcome projection and renders its compact result at a glance', () => {
  const source = readFileSync(new URL('./CampaignNode.tsx', import.meta.url), 'utf8')
  assert.match(source, /const loadLegacyDetail = useCampaignStore/)
  assert.doesNotMatch(source, /shouldLoadCampaignDrillDown\(configOpen, snapshotVersion\)/)
  assert.match(source, /void loadLegacyDetail\(workspaceId, campaignId\)/)
  assert.match(source, /<CampaignOutcomeSummary model=\{outcome\} density="compact" \/>/)
  assert.ok(source.indexOf('<CampaignOutcomeSummary model={outcome} density="compact" />') < source.indexOf("['studies'"))
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

test('campaign evidence panel hides lifecycle mutations unless authority is exactly live', async () => {
  const { CampaignEvidencePanel } = await import('./CampaignNode')
  const base = realisticDetail()

  for (const freshness of ['reconciling', 'stale', 'offline', 'error'] as const) {
    const detail = { ...base, freshness }
    const markup = renderToStaticMarkup(createElement(CampaignEvidencePanel, {
      campaignId: detail.campaign.campaign_id,
      campaigns: [detail.campaign],
      detail,
      latestComparison: undefined,
      mutating: false,
      onSelect: () => undefined,
      onTransition: () => { throw new Error(`transition rendered for ${freshness}`) },
      onRefresh: () => undefined,
      onOpenAutoResearch: () => undefined,
    }))

    assert.doesNotMatch(markup, />Start<|>Pause<|>Resume<|>Cancel</)
    assert.match(markup, />Refresh</)
    assert.match(markup, />Open in AutoResearch</)
  }
})

test('campaign transition submission never calls the bridge without live authority', async () => {
  const { submitCampaignTransitionIfLive } = await import('./campaignNodeActions')
  const calls: unknown[][] = []
  const transition = async (...args: unknown[]) => { calls.push(args) }

  for (const freshness of ['reconciling', 'stale', 'offline', 'error'] as const) {
    const submitted = await submitCampaignTransitionIfLive({
      detail: { ...realisticDetail(), freshness },
      mutating: false,
      action: 'pause',
      workspaceId: 'workspace-a',
      transition,
    })
    assert.equal(submitted, false)
  }
  assert.deepEqual(calls, [])
})
