import assert from 'node:assert/strict'
import test from 'node:test'

import type { CampaignDetailState } from '../../stores/campaignStore'
import { projectCampaignOutcome } from './campaignOutcomeModel'

function outcomeDetail(): CampaignDetailState {
  return {
    campaign: {
      campaign_id: 'star-count-qwen35-0-8b-run-1',
      workspace_id: 'default',
      status: 'active',
      title: 'Star count Qwen LoRA',
      target_model: { base_model: 'Qwen/Qwen3.5-0.8B' },
    },
    attempts: [
      { attempt_id: 'attempt-lora-train', stage: 'full_training', updated_at: '2026-07-18T02:00:00Z' },
      { attempt_id: 'attempt-lora-eval', stage: 'development_evaluation', updated_at: '2026-07-18T03:00:00Z' },
    ],
    ledger: {
      projects: [{
        project: { project_id: 'star-count', display_name: 'Star count', status: 'active' },
        runs: [
          { run_id: 'run-baseline', run_kind: 'baseline', status: 'completed', is_simulation: false },
          { run_id: 'attempt-lora-eval', run_kind: 'evaluation', status: 'completed', is_simulation: false },
        ],
        evaluations: [{
          evaluation_result_id: 'evaluation-baseline-fixed64-qwen35-run1',
          evaluation_suite_id: 'star-count-fixed64',
          run_id: 'run-baseline',
          status: 'completed',
          metrics: { exact_count_accuracy: 0.078125, count_accuracy: 0.359375, format_accuracy: 0.8125, mean_reward: 0.38203125 },
          created_at: '2026-07-18T01:00:00Z',
        }, {
          evaluation_result_id: 'evaluation-candidate-lora-160-qwen35-run1',
          evaluation_suite_id: 'star-count-fixed64',
          run_id: 'attempt-lora-eval',
          status: 'completed',
          metrics: { exact_count_accuracy: 0.65625, count_accuracy: 0.89453125, format_accuracy: 1, mean_reward: 0.8998046875 },
          created_at: '2026-07-18T03:00:00Z',
        }],
        experiments: [], artifacts: [], decisions: [], evidence: {},
      }],
      autoresearch: {
        spec: { primary_metric: 'exact_count_accuracy', metric_direction: 'maximize', stop_rules: { max_attempts: 2 } },
        state: {
          campaign_status: 'active', next_action: 'stop', reason_code: 'attempt_limit_reached',
          ready_for_next_proposal: false, baseline_verified: true,
          best_proposal_id: 'candidate-lora-sft-160-qwen35-run1', best_metric: 0.65625,
          attempts_used: 2, proposals_used: 2, budget_used: 3, budget_remaining: 0,
          latest_decision: 'keep',
        },
        proposals: [{ proposal_id: 'candidate-lora-sft-160-qwen35-run1', training_method: 'lora_sft' }],
        outcomes: [{
          result: {
            result_id: 'baseline-result', proposal_id: 'baseline-fixed64', study_id: 'baseline-study',
            role: 'baseline', provenance: 'real', outcome: 'completed', metric_name: 'exact_count_accuracy',
            metric_value: 0.078125, actual_cost: 0, attempt_ids: ['run-baseline'],
            evidence_references: ['evaluation-baseline-fixed64-qwen35-run1'], recorded_at: '2026-07-18T01:00:00Z',
          },
          decision: { proposal_id: 'baseline-fixed64', decision: 'baseline', reason_code: 'baseline_recorded', eligible_for_best: true, decided_at: '2026-07-18T01:00:00Z' },
        }, {
          result: {
            result_id: 'candidate-result', proposal_id: 'candidate-lora-sft-160-qwen35-run1', study_id: 'candidate-study',
            role: 'candidate', provenance: 'real', outcome: 'completed', metric_name: 'exact_count_accuracy',
            metric_value: 0.65625, actual_cost: 3, attempt_ids: ['attempt-lora-train', 'attempt-lora-eval'],
            evidence_references: ['evaluation-candidate-lora-160-qwen35-run1'], recorded_at: '2026-07-18T03:00:00Z',
          },
          decision: { proposal_id: 'candidate-lora-sft-160-qwen35-run1', decision: 'keep', reason_code: 'metric_improved', eligible_for_best: true, improvement: 0.578125, decided_at: '2026-07-18T03:00:00Z' },
        }],
        diagnostics: {
          schema_version: 'autoresearch_diagnostics.v1', workspace_id: 'default', campaign_id: 'star-count-qwen35-0-8b-run-1',
          primary_metric: 'exact_count_accuracy', metric_direction: 'maximize', low_signal: true,
          signals: [{ code: 'checkpoint_evidence_missing', severity: 'warning', summary: 'Only the terminal checkpoint was evaluated.', evidence_references: ['evaluation-candidate-lora-160-qwen35-run1'] }],
          checkpoint_comparisons: [], error_slices: [], ranked_hypotheses: [],
        },
      },
    },
    lossByAttempt: {
      'attempt-lora-train': [
        { step: 5, source: 'training_metrics.jsonl', value: 0.2595, observed_at: '2026-07-18T02:00:00Z' },
        { step: 135, source: 'training_metrics.jsonl', value: 0.04011, observed_at: '2026-07-18T02:10:00Z' },
        { step: 160, source: 'training_metrics.jsonl', value: 0.08392, observed_at: '2026-07-18T02:12:00Z' },
      ],
    },
  } as unknown as CampaignDetailState
}

test('projects a complete same-suite baseline versus LoRA candidate result', () => {
  const model = projectCampaignOutcome(outcomeDetail())
  assert.ok(model)
  assert.equal(model.verdict, 'success')
  assert.equal(model.verdictLabel, 'Candidate kept')
  assert.equal(model.lifecycleLabel, 'Closeout pending')
  assert.equal(model.candidateLabel, 'LoRA candidate')
  assert.equal(model.sameEvaluationSuite, true)
  assert.equal(model.primaryMetricId, 'exact_count_accuracy')
  assert.deepEqual(model.metrics.map((metric) => [metric.id, metric.baseline, metric.candidate, metric.delta]), [
    ['exact_count_accuracy', 0.078125, 0.65625, 0.578125],
    ['count_accuracy', 0.359375, 0.89453125, 0.53515625],
    ['format_accuracy', 0.8125, 1, 0.1875],
    ['mean_reward', 0.38203125, 0.8998046875, 0.5177734375],
  ])
})

test('selects persisted candidate training loss and reports its minimum and final values', () => {
  const model = projectCampaignOutcome(outcomeDetail())
  assert.ok(model?.loss)
  assert.equal(model.loss.attemptId, 'attempt-lora-train')
  assert.equal(model.loss.points.length, 3)
  assert.deepEqual(model.loss.minimum, { step: 135, value: 0.04011 })
  assert.deepEqual(model.loss.final, { step: 160, value: 0.08392 })
  assert.equal(model.checkpointWarning, 'Only the terminal checkpoint was evaluated.')
})

test('does not call incomparable or missing evaluation evidence a success', () => {
  const detail = outcomeDetail()
  detail.ledger!.projects[0]!.evaluations[1]!.evaluation_suite_id = 'different-suite'
  const model = projectCampaignOutcome(detail)
  assert.ok(model)
  assert.equal(model.sameEvaluationSuite, false)
  assert.equal(model.verdict, 'pending')
  assert.equal(model.verdictLabel, 'Comparison not valid')
})
