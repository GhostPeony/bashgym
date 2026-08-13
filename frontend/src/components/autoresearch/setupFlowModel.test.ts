import assert from 'node:assert/strict'
import test from 'node:test'

import { buildSetupFlowModel, setupStepOrder, type SetupFlowInput } from './setupFlowModel'

const allReceipts: SetupFlowInput['receipts'] = [
  {
    stepId: 'source',
    status: 'complete',
    receiptId: 'receipt-source',
    summary: 'Source binding saved.'
  },
  {
    stepId: 'model',
    status: 'complete',
    receiptId: 'receipt-model',
    summary: 'Immutable model identity saved.'
  },
  { stepId: 'data', status: 'complete', receiptId: 'receipt-data', summary: 'Data binding saved.' },
  {
    stepId: 'evaluator',
    status: 'complete',
    receiptId: 'receipt-evaluator',
    summary: 'Evaluator binding saved.'
  },
  {
    stepId: 'compute',
    status: 'complete',
    receiptId: 'receipt-compute',
    summary: 'Private SSH compute registered.'
  },
  {
    stepId: 'budget_stops',
    status: 'complete',
    receiptId: 'receipt-budget',
    summary: 'Budget saved.'
  },
  {
    stepId: 'activation_doctor',
    status: 'complete',
    receiptId: 'receipt-doctor',
    summary: 'Doctor completed.'
  },
  {
    stepId: 'campaign_creation',
    status: 'complete',
    receiptId: 'receipt-campaign',
    summary: 'Campaign created.'
  },
  {
    stepId: 'baseline_review',
    status: 'complete',
    receiptId: 'receipt-baseline',
    summary: 'Baseline reviewed.'
  }
]

function input(overrides: Partial<SetupFlowInput> = {}): SetupFlowInput {
  return {
    receipts: allReceipts,
    blockers: [],
    doctor: {
      observedAt: '2026-07-16T19:00:00Z',
      isLatestServerDoctor: true,
      launchReady: true,
      controllerIdentityVerified: true,
      computeIdentityVerified: true,
      state: 'live'
    },
    ...overrides
  }
}

test('keeps the setup contract in the required durable order without a model default', () => {
  const model = buildSetupFlowModel(
    input({
      receipts: [],
      drafts: {
        compute_profile_id: 'private-compute-1',
        ssh_device_id: 'registered-ssh-1'
      }
    })
  )

  assert.deepEqual(setupStepOrder, [
    'source',
    'model',
    'data',
    'evaluator',
    'compute',
    'budget_stops',
    'activation_doctor',
    'campaign_creation',
    'baseline_review',
    'start'
  ])
  assert.equal(model.steps[1]?.label, 'Model identity & revision')
  assert.equal(model.steps[1]?.receipt, null)
  assert.deepEqual(
    model.steps[1]?.editor?.fields.map((field) => field.id),
    ['model_ref', 'model_revision']
  )
  assert.equal(
    model.steps[1]?.editor?.fields.every((field) => field.value === ''),
    true
  )
  assert.equal(
    model.steps.find((step) => step.id === 'compute')?.editor?.fields[0]?.value,
    'private-compute-1'
  )
  assert.equal(
    model.steps.find((step) => step.id === 'activation_doctor')?.editor?.fields[0]?.value,
    'registered-ssh-1'
  )
  assert.equal(
    model.steps.some((step) => /qwen|recommended model/i.test(`${step.label} ${step.guidance}`)),
    false
  )
})

test('keeps later editors locked until the current durable receipt is complete', () => {
  const model = buildSetupFlowModel(input({ receipts: [] }))

  assert.equal(model.nextStepId, 'source')
  assert.equal(model.steps[0]?.sequence, 'current')
  assert.equal(model.steps[0]?.action, 'continue')
  assert.equal(model.steps[1]?.sequence, 'locked')
  assert.equal(model.steps[1]?.action, null)
  assert.match(model.steps[1]?.actionGuidance ?? '', /complete Source first/i)
  assert.equal(model.steps.at(-1)?.sequence, 'locked')
  assert.equal(model.steps.at(-1)?.action, null)
})

test('keeps completed receipts complete and resumes a retryable partial operation without repeating it', () => {
  const model = buildSetupFlowModel(
    input({
      receipts: [
        allReceipts[0]!,
        {
          stepId: 'model',
          status: 'partial',
          receiptId: 'receipt-model',
          summary: 'Revision check can resume.',
          retryable: true
        }
      ],
      blockers: [
        {
          stepId: 'compute',
          code: 'ssh_registration_required',
          summary: 'Register a private SSH compute target.'
        }
      ]
    })
  )

  assert.equal(model.steps[0]?.status, 'complete')
  assert.equal(model.steps[1]?.status, 'partial')
  assert.equal(model.steps[1]?.action, 'resume')
  assert.equal(model.steps[1]?.actionReceiptId, 'receipt-model')
  assert.equal(model.steps[1]?.actionAvailable, false)
  assert.match(model.steps[1]?.actionGuidance ?? '', /setup-autoresearch/)
  assert.equal(model.nextStepId, 'model')
  assert.deepEqual(
    model.blockers.map((blocker) => blocker.code),
    ['ssh_registration_required']
  )
})

test('offers retry only for a retryable failed receipt and never echoes an unsafe receipt identifier', () => {
  const retryable = buildSetupFlowModel(
    input({
      receipts: [
        allReceipts[0]!,
        {
          stepId: 'model',
          status: 'failed',
          receiptId: 'receipt-model-2',
          summary: 'Immutable revision check failed.',
          retryable: true
        }
      ]
    })
  )
  assert.equal(retryable.steps[1]?.action, 'retry')
  assert.equal(retryable.steps[1]?.actionReceiptId, 'receipt-model-2')

  const unsafe = buildSetupFlowModel(
    input({
      receipts: [
        allReceipts[0]!,
        {
          stepId: 'model',
          status: 'partial',
          receiptId: '../../private-key',
          summary: 'Resume.',
          retryable: true
        }
      ]
    })
  )
  assert.equal(unsafe.steps[1]?.action, 'remediate')
  assert.equal(unsafe.steps[1]?.receipt, null)
  assert.equal(unsafe.steps[1]?.actionReceiptId, null)
  assert.equal(JSON.stringify(unsafe), JSON.stringify(unsafe).replace('../../private-key', ''))
})

test('enables only an explicitly mounted existing campaign mutation contract', () => {
  const withoutBridge = buildSetupFlowModel(
    input({
      receipts: allReceipts.filter((receipt) => receipt.stepId !== 'campaign_creation'),
      drafts: { template_id: 'installed-template-1', campaign_id: 'campaign-1' }
    })
  )
  const disabled = withoutBridge.steps.find((step) => step.id === 'campaign_creation')!
  assert.equal(disabled.action, 'continue')
  assert.equal(disabled.mutationContract, 'campaign.create_from_template')
  assert.equal(disabled.actionAvailable, false)

  const withBridge = buildSetupFlowModel(
    input({
      receipts: allReceipts.filter((receipt) => receipt.stepId !== 'campaign_creation'),
      drafts: { template_id: 'installed-template-1', campaign_id: 'campaign-1' },
      availableMutationContracts: ['campaign.create_from_template']
    })
  )
  const enabled = withBridge.steps.find((step) => step.id === 'campaign_creation')!
  assert.equal(enabled.editor?.ready, true)
  assert.equal(enabled.actionAvailable, true)
})

test('treats optional integrations as optional while private SSH compute is required', () => {
  const model = buildSetupFlowModel(
    input({
      optionalIntegrations: [{ id: 'nemo', label: 'NeMo adapter', status: 'unconfigured' }]
    })
  )

  assert.equal(model.steps.find((step) => step.id === 'compute')?.status, 'complete')
  assert.equal(model.optionalIntegrations[0]?.required, false)
  assert.equal(model.optionalIntegrations[0]?.status, 'unconfigured')
  assert.equal(
    model.blockers.some((blocker) => blocker.code === 'nemo'),
    false
  )
})

test('enables Start only from the latest live server doctor with verified execution identities', () => {
  const ready = buildSetupFlowModel(input())
  assert.equal(ready.start.enabled, true)
  assert.equal(ready.start.reason, null)

  for (const doctor of [
    { ...input().doctor!, isLatestServerDoctor: false },
    { ...input().doctor!, launchReady: false },
    { ...input().doctor!, controllerIdentityVerified: false },
    { ...input().doctor!, computeIdentityVerified: false },
    { ...input().doctor!, state: 'offline' as const }
  ]) {
    const model = buildSetupFlowModel(input({ doctor }))
    assert.equal(model.start.enabled, false)
    assert.ok(model.start.reason)
  }
})
