import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { CampaignOutcomeViewModel } from './campaignOutcomeModel'
import { CampaignOutcomeSummary } from './CampaignOutcomeSummary'

const model: CampaignOutcomeViewModel = {
  verdict: 'success',
  verdictLabel: 'Candidate kept',
  lifecycleLabel: 'Closeout pending',
  lifecycleReason: 'attempt_limit_reached',
  primaryMetricId: 'exact_count_accuracy',
  baselineLabel: 'Baseline',
  candidateLabel: 'LoRA candidate',
  baselineEvaluationId: 'evaluation-baseline-fixed64-qwen35-run1',
  candidateEvaluationId: 'evaluation-candidate-lora-160-qwen35-run1',
  evaluationSuiteId: 'star-count-fixed64',
  sameEvaluationSuite: true,
  decision: 'keep',
  metrics: [
    {
      id: 'exact_count_accuracy',
      baseline: 0.078125,
      candidate: 0.65625,
      delta: 0.578125,
      direction: 'maximize',
      primary: true
    },
    {
      id: 'count_accuracy',
      baseline: 0.359375,
      candidate: 0.89453125,
      delta: 0.53515625,
      direction: 'unknown',
      primary: false
    },
    {
      id: 'format_accuracy',
      baseline: 0.8125,
      candidate: 1,
      delta: 0.1875,
      direction: 'unknown',
      primary: false
    },
    {
      id: 'mean_reward',
      baseline: 0.38203125,
      candidate: 0.8998046875,
      delta: 0.5177734375,
      direction: 'unknown',
      primary: false
    }
  ],
  loss: {
    attemptId: 'attempt-lora-train',
    points: [
      {
        step: 5,
        source: 'training_metrics.jsonl',
        value: 0.2595,
        observed_at: '2026-07-18T02:00:00Z'
      },
      {
        step: 135,
        source: 'training_metrics.jsonl',
        value: 0.04011,
        observed_at: '2026-07-18T02:10:00Z'
      },
      {
        step: 160,
        source: 'training_metrics.jsonl',
        value: 0.08392,
        observed_at: '2026-07-18T02:12:00Z'
      }
    ],
    first: { step: 5, value: 0.2595 },
    minimum: { step: 135, value: 0.04011 },
    final: { step: 160, value: 0.08392 }
  },
  checkpointWarning: 'Only the terminal checkpoint was evaluated.'
}

test('detailed outcome makes success, every metric delta, and loss evidence visible', () => {
  const html = renderToStaticMarkup(
    createElement(CampaignOutcomeSummary, { model, density: 'detailed' })
  )
  assert.match(html, /aria-label="Baseline versus LoRA candidate outcome"/)
  assert.match(html, /Candidate kept/)
  assert.match(html, /Closeout pending/)
  assert.match(html, /Exact count accuracy/)
  assert.match(html, /7\.8%/)
  assert.match(html, /65\.6%/)
  assert.match(html, /\+57\.8 pp/)
  assert.match(html, /Count accuracy/)
  assert.match(html, /Format accuracy/)
  assert.match(html, /Mean reward/)
  assert.match(html, /Same fixed evaluation suite/)
  assert.match(html, /Training loss/)
  assert.match(html, /Minimum 0\.0401 at step 135/)
  assert.match(html, /Final 0\.0839 at step 160/)
  assert.match(
    html,
    /aria-label="Training loss curve, 3 points, minimum 0.04011 at step 135, final 0.08392 at step 160"/
  )
  assert.match(html, /Only the terminal checkpoint was evaluated\./)
})

test('compact outcome preserves the primary verdict, delta, and loss sparkline for the canvas node', () => {
  const html = renderToStaticMarkup(
    createElement(CampaignOutcomeSummary, { model, density: 'compact' })
  )
  assert.match(html, /aria-label="Baseline versus LoRA candidate outcome"/)
  assert.match(html, /Candidate kept/)
  assert.match(html, /7\.8% → 65\.6%/)
  assert.match(html, /\+57\.8 pp/)
  assert.match(html, /aria-label="Training loss sparkline/)
  assert.doesNotMatch(html, /Count accuracy/)
})
