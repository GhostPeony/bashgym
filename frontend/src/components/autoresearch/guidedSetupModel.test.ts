import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildGuidedSetupView,
  parseGuidedSetupContext,
  parseGuidedSetupCreation,
  parseGuidedSetupDoctor,
  parseGuidedSetupMutation,
  parseGuidedSetupValidation,
} from './guidedSetupModel'

const required = {
  model: 'model.registered',
  data: 'data.registered',
  compute: 'compute.private',
  evaluation: 'evaluation.registered',
}

function context(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 'guided_setup_context.v1',
    workspace_id: 'workspace-a',
    templates: [{
      schema_version: 'guided_setup_template.v1',
      template_id: 'template-modern',
      definition_digest: 'a'.repeat(64),
      quality_claim_eligible: true,
      required_bindings: required,
    }],
    installations: [{
      installation_id: `ins_${'b'.repeat(32)}`,
      ready: true,
      reason_codes: [],
      bindings: {
        model: [{ logical_id: required.model, availability: 'reachable', selectable: true, reason_codes: [], display_label: 'Operator model' }],
        data: [{ logical_id: required.data, availability: 'reachable', selectable: true, reason_codes: [] }],
        compute: [{ logical_id: required.compute, availability: 'reachable', selectable: true, reason_codes: [] }],
        evaluation: [{ logical_id: required.evaluation, availability: 'reachable', selectable: true, reason_codes: [] }],
      },
      truncation: { truncated: false, reason_codes: [], limit_per_kind: 32, kinds: [] },
    }],
    session: null,
    reason_codes: ['setup_session_not_started'],
    truncation: {
      truncated: false,
      reason_codes: [],
      limits: { templates: 32, installations: 32, bindings_per_kind: 32 },
    },
    ...overrides,
  }
}

test('strictly parses bounded public discovery without inventing a model or provider', () => {
  const parsed = parseGuidedSetupContext(context())
  assert.ok(parsed)
  const view = buildGuidedSetupView(parsed)
  assert.equal(view.currentStep, 'template')
  assert.deepEqual(view.options.map((option) => option.id), ['template-modern'])
  assert.equal(JSON.stringify(view).includes('Qwen'), false)
  assert.equal(JSON.stringify(view).includes('cloud'), false)

  assert.equal(parseGuidedSetupContext({ ...context(), private_path: 'C:/secret' }), null)
  assert.equal(parseGuidedSetupContext({ ...context(), templates: [{ ...(context().templates as object[])[0], extra: true }] }), null)
})

test('projects the exact next receipt step and selectable registered choices', () => {
  const parsed = parseGuidedSetupContext(context({
    session: {
      schema_version: 'guided_setup_session.v1',
      workspace_id: 'workspace-a',
      session_id: `setupsess_${'c'.repeat(32)}`,
      version: 2,
      completed_steps: ['template', 'installation'],
      selections: { template_id: 'template-modern', installation_id: `ins_${'b'.repeat(32)}`, bindings: { model: null, data: null, compute: null, evaluation: null } },
      ready_for_validation: false,
      reason_codes: ['model_binding_not_selected'],
      latest_receipt: {
        schema_version: 'guided_setup_step_receipt.v1', receipt_id: `setupstep_${'d'.repeat(32)}`,
        session_id: `setupsess_${'c'.repeat(32)}`, version: 2, step: 'installation',
        selection_id: `ins_${'b'.repeat(32)}`, state_digest: 'e'.repeat(64),
        previous_receipt_id: `setupstep_${'f'.repeat(32)}`, previous_receipt_digest: `sha256:${'1'.repeat(64)}`,
        created_at: '2026-07-17T00:00:00Z', receipt_digest: `sha256:${'2'.repeat(64)}`,
      },
      updated_at: '2026-07-17T00:00:00Z',
    },
    reason_codes: ['model_binding_not_selected'],
  }))
  assert.ok(parsed)
  const view = buildGuidedSetupView(parsed)
  assert.equal(view.currentStep, 'model')
  assert.deepEqual(view.options.map((option) => option.id), [required.model])
  assert.equal(view.completedCount, 2)
})

test('exposes explicit discovery truncation and rejects unsafe binding response shapes', () => {
  const parsed = parseGuidedSetupContext(context({
    truncation: {
      truncated: true,
      reason_codes: ['bindings_truncated'],
      limits: { templates: 32, installations: 32, bindings_per_kind: 32 },
    },
  }))
  assert.ok(parsed)
  assert.deepEqual(buildGuidedSetupView(parsed).truncationReasons, ['bindings_truncated'])

  const unsafe = structuredClone(context())
  ;(unsafe.installations[0].bindings.model[0] as Record<string, unknown>).host = 'private-host'
  assert.equal(parseGuidedSetupContext(unsafe), null)

  const privateLabel = structuredClone(context())
  ;(privateLabel.installations[0].bindings.model[0] as Record<string, unknown>).display_label = 'C:\\Users\\operator\\model'
  assert.equal(parseGuidedSetupContext(privateLabel), null)

  const wrongLimits = structuredClone(context())
  wrongLimits.truncation.limits.templates = 64
  assert.equal(parseGuidedSetupContext(wrongLimits), null)
})

test('strictly parses session, doctor, and validation authority responses', () => {
  const mutation = {
    schema_version: 'guided_setup_session_mutation.v1',
    session: parseGuidedSetupContext(context())?.session,
    receipt: {},
  }
  assert.equal(parseGuidedSetupMutation(mutation), null)

  const doctor = {
    schema_version: 'guided_setup_doctor.v1', workspace_id: 'workspace-a', template_id: 'template-modern',
    definition_digest: 'a'.repeat(64), draft_digest: 'b'.repeat(64), ready: true,
    blocking_codes: [], binding_references: Object.entries(required).map(([kind, logical_id]) => ({ kind, logical_id, availability: 'reachable' })),
  }
  assert.ok(parseGuidedSetupDoctor(doctor))
  assert.equal(parseGuidedSetupDoctor({ ...doctor, secret: true }), null)
  assert.equal(parseGuidedSetupDoctor({ ...doctor, binding_references: doctor.binding_references.map((row) => ({ ...row, kind: 'model' })) }), null)
  assert.equal(parseGuidedSetupDoctor({ ...doctor, binding_references: doctor.binding_references.map((row, index) => index === 0 ? { ...row, availability: 'private' } : row) }), null)
  assert.ok(parseGuidedSetupValidation({ ...doctor, receipt_id: `setuprcpt_${'a'.repeat(32)}` }))
  assert.equal(parseGuidedSetupValidation({ ...doctor, receipt_id: 'bad' }), null)
})

test('projects only the safe creation handoff from an exact response envelope', () => {
  const creation = {
    campaign: { campaign_id: 'campaign-new', workspace_id: 'workspace-a', title: 'New campaign', status: 'ready' },
    event: { event_id: 'event-1' }, replayed: false,
    setup: {
      schema_version: 'guided_setup_creation.v1', validation_receipt_id: `setuprcpt_${'a'.repeat(32)}`,
      binding_references: required,
    },
  }
  assert.deepEqual(parseGuidedSetupCreation(creation), {
    workspace_id: 'workspace-a', campaign_id: 'campaign-new', title: 'New campaign',
    status: 'ready', replayed: false, validation_receipt_id: `setuprcpt_${'a'.repeat(32)}`,
    binding_references: required,
  })
  assert.equal(parseGuidedSetupCreation({ ...creation, private_path: 'C:/secret' }), null)
})
