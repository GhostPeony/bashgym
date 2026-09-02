import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { GuidedAutoResearchSetup } from './GuidedAutoResearchSetup'
import { parseGuidedSetupContext, type GuidedSetupContext } from './guidedSetupModel'

const required = {
  model: 'model.registered',
  data: 'data.registered',
  compute: 'compute.private',
  evaluation: 'evaluation.registered'
}
const experimentContract = {
  primary_metric: 'exact_task_accuracy',
  metric_direction: 'maximize',
  max_attempts_limit: 6,
  budget_limits: { gpu_hours: 10 },
  protected_metrics: [
    { metric_name: 'tool_call_validity', direction: 'maximize', max_regression: 0.02 }
  ]
}
const context = parseGuidedSetupContext({
  schema_version: 'guided_setup_context.v1',
  workspace_id: 'workspace-a',
  templates: [
    {
      schema_version: 'guided_setup_template.v1',
      template_id: 'template-modern',
      definition_digest: 'a'.repeat(64),
      quality_claim_eligible: true,
      required_bindings: required,
      experiment_contract: experimentContract
    }
  ],
  installations: [],
  session: null,
  reason_codes: ['setup_session_not_started'],
  truncation: {
    truncated: true,
    reason_codes: ['installations_truncated'],
    limits: { templates: 32, installations: 32, bindings_per_kind: 32 }
  }
})

test('keeps a useful compact six-step setup visible while authority is offline', () => {
  const html = renderToStaticMarkup(
    createElement(GuidedAutoResearchSetup, {
      context: null,
      connectionState: 'offline',
      pending: false,
      error: 'Campaign service is unavailable.',
      doctor: null,
      validation: null,
      selectedOptionId: '',
      campaignId: '',
      title: '',
      budgetUnit: '',
      budgetLimit: '',
      maxAttempts: '',
      minimumImprovement: '',
      onSelectedOptionChange: () => {},
      onCampaignIdChange: () => {},
      onTitleChange: () => {},
      onBudgetUnitChange: () => {},
      onBudgetLimitChange: () => {},
      onMaxAttemptsChange: () => {},
      onMinimumImprovementChange: () => {},
      onAdvance: () => {},
      onDoctor: () => {},
      onValidate: () => {},
      onCreate: () => {},
      onRetry: () => {}
    })
  )
  assert.match(html, /Guided setup/)
  for (const step of ['Template', 'Installation', 'Model', 'Data', 'Compute', 'Evaluation'])
    assert.match(html, new RegExp(step))
  assert.match(html, /Live authority is offline/)
  assert.match(html, /Registered choices will appear here after reconnection/)
  assert.match(html, /grid-cols-\[minmax\(0,1fr\)_18rem\]/)
  assert.doesNotMatch(html, /Qwen|cloud fallback|private hostname|device model/i)
  assert.match(html, /<button[^>]*disabled=""[^>]*>Save choice<\/button>/)
})

test('renders only authoritative registered choices and explicit truncation', () => {
  assert.ok(context)
  const html = renderToStaticMarkup(
    createElement(GuidedAutoResearchSetup, {
      context,
      connectionState: 'live',
      pending: false,
      error: null,
      doctor: null,
      validation: null,
      selectedOptionId: 'template-modern',
      campaignId: '',
      title: '',
      budgetUnit: '',
      budgetLimit: '',
      maxAttempts: '',
      minimumImprovement: '',
      onSelectedOptionChange: () => {},
      onCampaignIdChange: () => {},
      onTitleChange: () => {},
      onBudgetUnitChange: () => {},
      onBudgetLimitChange: () => {},
      onMaxAttemptsChange: () => {},
      onMinimumImprovementChange: () => {},
      onAdvance: () => {},
      onDoctor: () => {},
      onValidate: () => {},
      onCreate: () => {},
      onRetry: () => {}
    })
  )
  assert.match(html, /template-modern/)
  assert.match(html, /installations truncated/)
  assert.match(html, /0 of 6 choices sealed/)
  assert.doesNotMatch(html, /name="host"|name="user"|name="key"|name="path"/)
  assert.doesNotMatch(html, /disabled=""[^>]*>Save choice<\/button>/)
})

test('requires explicit campaign limits and shows the fixed metric before validation', () => {
  assert.ok(context)
  const selectedContext = {
    ...context,
    session: {
      schema_version: 'guided_setup_session.v1',
      workspace_id: 'workspace-a',
      session_id: `setupsess_${'a'.repeat(32)}`,
      version: 6,
      completed_steps: ['template', 'installation', 'model', 'data', 'compute', 'evaluation'],
      selections: {
        template_id: 'template-modern',
        installation_id: `ins_${'b'.repeat(32)}`,
        bindings: required
      },
      ready_for_validation: true,
      reason_codes: [],
      latest_receipt: {
        schema_version: 'guided_setup_step_receipt.v1',
        receipt_id: `setupstep_${'c'.repeat(32)}`,
        session_id: `setupsess_${'a'.repeat(32)}`,
        version: 6,
        step: 'evaluation',
        selection_id: required.evaluation,
        state_digest: `sha256:${'d'.repeat(64)}`,
        previous_receipt_id: `setupstep_${'e'.repeat(32)}`,
        previous_receipt_digest: `sha256:${'f'.repeat(64)}`,
        created_at: '2026-08-16T12:00:00Z',
        receipt_digest: `sha256:${'1'.repeat(64)}`
      },
      updated_at: '2026-08-16T12:00:00Z'
    }
  } as GuidedSetupContext
  const props = {
    context: selectedContext,
    connectionState: 'live' as const,
    pending: false,
    error: null,
    doctor: null,
    validation: null,
    selectedOptionId: '',
    campaignId: '',
    title: '',
    budgetUnit: 'gpu_hours',
    budgetLimit: '8',
    maxAttempts: '5',
    minimumImprovement: '0.01',
    onSelectedOptionChange: () => {},
    onCampaignIdChange: () => {},
    onTitleChange: () => {},
    onBudgetUnitChange: () => {},
    onBudgetLimitChange: () => {},
    onMaxAttemptsChange: () => {},
    onMinimumImprovementChange: () => {},
    onAdvance: () => {},
    onDoctor: () => {},
    onValidate: () => {},
    onCreate: () => {},
    onRetry: () => {}
  }
  const html = renderToStaticMarkup(createElement(GuidedAutoResearchSetup, props))
  assert.match(html, /Primary metric · exact_task_accuracy \(maximize\)/)
  assert.match(html, /Protected metrics · tool_call_validity \(maximize, max regression 0.02\)/)
  assert.match(html, /Maximum attempts \(baseline \+ candidates\)/)
  assert.match(html, /Campaign budget limit/)
  assert.match(html, /value="5"/)
  assert.match(html, /value="8"/)

  const overBudget = renderToStaticMarkup(
    createElement(GuidedAutoResearchSetup, { ...props, budgetLimit: '11' })
  )
  assert.match(overBudget, /within the approved ceilings/)
  assert.match(overBudget, /<button[^>]*disabled=""[^>]*>Run doctor<\/button>/)
})
