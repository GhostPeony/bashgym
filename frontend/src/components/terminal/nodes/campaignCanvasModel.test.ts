import assert from 'node:assert/strict'
import test from 'node:test'

import type { CampaignDetailState, CampaignLedgerProject } from '../../../stores/campaignStore'
import { projectCampaignResearch } from './campaignCanvasModel'

function project(
  projectId: string,
  overrides: Partial<CampaignLedgerProject> = {}
): CampaignLedgerProject {
  return {
    project: { project_id: projectId, display_name: projectId, status: 'active' },
    experiments: [],
    runs: [],
    evaluations: [],
    artifacts: [],
    decisions: [],
    evidence: {},
    ...overrides
  }
}

function detail(): CampaignDetailState {
  return {
    snapshot: null,
    freshness: 'live',
    lastVerifiedAt: null,
    reconciliation: {
      freshness: 'live',
      generation: 0,
      connectionGeneration: 1,
      subscribed: true,
      appliedCursor: 0,
      appliedVersion: 3,
      targetCursor: 0,
      targetVersion: 3,
      semanticKey: null,
      inFlightGeneration: null,
      retryCount: 0,
      lastHintAt: null,
      lastVerifiedAt: null,
      errorCode: null
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
      artifactsError: null
    },
    campaign: {
      schema_version: 'campaign.v1',
      campaign_id: 'campaign-auto-1',
      workspace_id: 'workspace-a',
      title: 'AutoResearch slice',
      kind: 'general',
      objective: 'Improve held-out terminal task pass rate.',
      target_model: { base_model_ref: 'org/base-model' },
      owner_actor_id: 'codex-agent',
      manifest_revision: 1,
      status: 'active',
      active_study_id: 'study-active',
      active_action_id: 'action-active',
      version: 3,
      created_at: '2026-07-14T00:00:00Z',
      updated_at: '2026-07-14T01:00:00Z'
    },
    studies: [
      {
        study_id: 'study-active',
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-auto-1',
        proposal_id: 'proposal-active',
        status: 'full_training',
        stage_plan: {
          items: [
            { stage: 'smoke_training', disposition: 'required', reason: 'Bound the first run.' },
            { stage: 'full_training', disposition: 'required', reason: 'Measure the hypothesis.' }
          ]
        },
        current_stage_index: 1,
        candidate_digest: 'a'.repeat(64),
        version: 2,
        created_at: '2026-07-14T00:10:00Z',
        updated_at: '2026-07-14T00:20:00Z'
      },
      {
        study_id: 'study-newer-but-inactive',
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-auto-1',
        proposal_id: 'proposal-inactive',
        status: 'validated',
        stage_plan: {
          items: [{ stage: 'smoke_training', disposition: 'required', reason: 'Later queue item.' }]
        },
        current_stage_index: 0,
        candidate_digest: 'b'.repeat(64),
        version: 1,
        created_at: '2026-07-14T00:30:00Z',
        updated_at: '2026-07-14T00:40:00Z'
      }
    ],
    proposals: [
      {
        proposal: {
          proposal_id: 'proposal-active',
          hypothesis: 'Hard-case augmentation improves pass@1.',
          study_family: 'data_mix',
          primary_variable: 'hard_case_fraction',
          expected_outcome: 'Pass@1 increases by at least five points.',
          falsification_criterion: 'Pass@1 uplift is below five points.',
          estimated_cost: 1,
          status: 'accepted',
          created_at: '2026-07-14T00:05:00Z'
        },
        validation: { valid: true, reason_codes: [] },
        study_id: 'study-active',
        updated_at: '2026-07-14T00:10:00Z'
      }
    ],
    evidence: {
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-auto-1',
      campaign_version: 3,
      manifest_revision: 1,
      status: 'active',
      objective: 'Improve held-out terminal task pass rate.',
      approved_data_scopes: ['approved-training'],
      compute_profile_id: 'local-gpu',
      budget_remaining: { study_count: 3, gpu_hours: 4.25 },
      proposal_counts: { accepted: 1, submitted: 0 },
      available_executors: ['local'],
      active_study_id: 'study-active',
      active_action_id: 'action-active',
      snapshot_digest: 'c'.repeat(64),
      created_at: '2026-07-14T01:00:00Z'
    },
    attempts: [
      {
        schema_version: 'public_campaign_attempt.v1' as const,
        attempt_id: 'attempt-active',
        workspace_id: 'workspace-a',
        campaign_id: 'campaign-auto-1',
        study_id: 'study-active',
        action_id: 'action-active',
        attempt_number: 1,
        claim_generation: 1,
        status: 'running',
        input_digest: 'd'.repeat(64),
        candidate_digest: 'e'.repeat(64),
        manifest_revision: 1,
        stage: 'full_training',
        executor_kind: 'local',
        created_at: '2026-07-14T00:50:00Z',
        updated_at: '2026-07-14T01:00:00Z'
      }
    ],
    artifacts: [],
    comparisons: [],
    ledger: {
      schema_version: 'bashgym.campaign-ledger-projection.v1',
      workspace_id: 'workspace-a',
      campaign_id: 'campaign-auto-1',
      generated_at: '2026-07-14T01:00:00Z',
      linked: true,
      projects: [
        project('project-a', {
          runs: [
            {
              run_id: 'baseline-real',
              status: 'completed',
              run_kind: 'baseline',
              is_simulation: false,
              queued_at: '2026-07-14T00:00:00Z'
            },
            {
              run_id: 'baseline-simulation-newer',
              status: 'completed',
              run_kind: 'baseline',
              is_simulation: true,
              queued_at: '2026-07-14T00:30:00Z'
            }
          ],
          evaluations: [
            {
              evaluation_result_id: 'evaluation-baseline',
              evaluation_suite_id: 'terminal-dev-v1',
              run_id: 'baseline-real',
              status: 'completed',
              metrics: { pass_at_1: 0.25 },
              created_at: '2026-07-14T00:05:00Z'
            }
          ],
          decisions: [
            {
              decision_id: 'decision-older',
              experiment_id: 'experiment-1',
              decision_type: 'discard',
              outcome: 'Discard the earlier candidate.',
              rationale: 'No uplift.',
              evidence_refs: ['evaluation-old'],
              created_at: '2026-07-14T00:25:00Z'
            }
          ]
        }),
        project('project-b', {
          decisions: [
            {
              decision_id: 'decision-latest',
              experiment_id: 'experiment-2',
              decision_type: 'continue',
              outcome: 'Continue with hard-case augmentation.',
              rationale: 'The remaining failures are off-by-one cases.',
              evidence_refs: ['evaluation-current'],
              created_at: '2026-07-14T00:45:00Z'
            }
          ]
        })
      ]
    },
    events: [],
    lossByAttempt: {},
    nextCursor: 0,
    loading: false,
    error: null
  }
}

test('campaign research projection joins only durable authoritative evidence', () => {
  const research = projectCampaignResearch(detail())

  assert.equal(research.objective, 'Improve held-out terminal task pass rate.')
  assert.deepEqual(research.baseline, {
    status: 'evaluated',
    modelRef: 'org/base-model',
    runId: 'baseline-real',
    evaluationSuiteId: 'terminal-dev-v1',
    metrics: { pass_at_1: 0.25 }
  })
  assert.equal(research.latestStudy?.studyId, 'study-active')
  assert.equal(research.latestStudy?.hypothesis, 'Hard-case augmentation improves pass@1.')
  assert.equal(research.latestStudy?.falsificationCriterion, 'Pass@1 uplift is below five points.')
  assert.deepEqual(
    research.budget.map((item) => item.resource),
    ['gpu_hours', 'study_count']
  )
  assert.equal(research.latestDecision?.decisionId, 'decision-latest')
  assert.deepEqual(research.nextAction, {
    kind: 'current',
    label: 'full_training · running',
    actionId: 'action-active'
  })
})

test('projection never promotes simulation to baseline and distinguishes planned from terminal work', () => {
  const source = detail()
  source.campaign.active_action_id = null
  source.evidence!.active_action_id = null
  source.ledger!.projects[0].runs = source.ledger!.projects[0].runs.filter(
    (run) => run.is_simulation
  )

  const planned = projectCampaignResearch(source)
  assert.equal(planned.baseline.status, 'not_recorded')
  assert.equal(planned.baseline.modelRef, 'org/base-model')
  assert.deepEqual(planned.nextAction, { kind: 'planned', label: 'full_training', actionId: null })

  source.campaign.status = 'completed'
  source.campaign.active_study_id = null
  source.studies = []
  const complete = projectCampaignResearch(source)
  assert.deepEqual(complete.nextAction, {
    kind: 'none',
    label: 'no further action',
    actionId: null
  })
})

test('projection exposes the durable AutoResearch decision and safe next action', () => {
  const source = detail()
  source.campaign.active_action_id = null
  source.campaign.active_study_id = null
  source.studies = []
  source.ledger!.projects = []
  source.ledger!.autoresearch = {
    spec: {
      primary_metric: 'pass_at_1',
      metric_direction: 'maximize',
      stop_rules: { max_attempts: 3 }
    },
    state: {
      campaign_status: 'active',
      next_action: 'submit_baseline',
      reason_code: 'real_baseline_required',
      ready_for_next_proposal: true,
      baseline_verified: false,
      attempts_used: 1,
      proposals_used: 1,
      budget_used: 0.01,
      budget_remaining: 0.24,
      latest_decision: 'ineligible'
    },
    proposals: [],
    outcomes: [
      {
        result: {
          result_id: 'result-control-smoke',
          proposal_id: 'baseline-control-smoke',
          study_id: 'study-control-smoke',
          role: 'baseline',
          provenance: 'simulated',
          outcome: 'completed',
          metric_name: 'pass_at_1',
          metric_value: 1,
          actual_cost: 0.01,
          attempt_ids: ['attempt-control-smoke'],
          evidence_references: ['artifact-control-smoke'],
          recorded_at: '2026-07-14T02:00:00Z'
        },
        decision: {
          proposal_id: 'baseline-control-smoke',
          decision: 'ineligible',
          reason_code: 'simulated_result_not_quality_evidence',
          eligible_for_best: false,
          decided_at: '2026-07-14T02:00:01Z'
        }
      }
    ]
  }

  const research = projectCampaignResearch(source)
  assert.equal(research.baseline.status, 'not_recorded')
  assert.equal(research.latestDecision?.type, 'ineligible')
  assert.equal(research.latestDecision?.outcome, 'simulated_result_not_quality_evidence')
  assert.deepEqual(research.nextAction, {
    kind: 'planned',
    label: 'submit real baseline',
    actionId: null
  })
})

test('projection surfaces the Git lineage gate before a code study stage', () => {
  const source = detail()
  source.campaign.active_action_id = null
  source.evidence!.active_action_id = null
  source.ledger!.autoresearch = {
    spec: {
      primary_metric: 'pass_at_1',
      metric_direction: 'maximize',
      stop_rules: { max_attempts: 3 }
    },
    state: {
      campaign_status: 'active',
      next_action: 'wait_for_result',
      reason_code: 'experiment_result_pending',
      ready_for_next_proposal: false,
      baseline_verified: true,
      attempts_used: 1,
      proposals_used: 2,
      budget_used: 0.5,
      budget_remaining: 1.5
    },
    proposals: [],
    outcomes: [],
    code_lineages: [
      {
        lineage_id: 'lineage-proposal-active',
        campaign_id: 'campaign-auto-1',
        proposal_id: 'proposal-active',
        mutation_kind: 'trainer',
        source_repository_profile_id: 'source-profile-1',
        state: 'prepared',
        base_commit: 'a'.repeat(40),
        branch_name: 'bashgym/autoresearch/proposal-active-deadbeef',
        changed_paths: [],
        created_at: '2026-07-14T00:05:00Z',
        updated_at: '2026-07-14T00:06:00Z'
      }
    ]
  }

  assert.deepEqual(projectCampaignResearch(source).nextAction, {
    kind: 'attention',
    label: 'edit and capture code lineage',
    actionId: null
  })
})
