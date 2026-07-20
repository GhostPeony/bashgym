export const setupStepOrder = [
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
] as const

export type SetupStepId = (typeof setupStepOrder)[number]
type DurableSetupStepId = Exclude<SetupStepId, 'start'>
export type SetupStepAction = 'continue' | 'resume' | 'retry' | 'remediate'
export type SetupMutationContract = 'campaign.create_from_template'

export type SetupEditorFieldId =
  | 'source_profile_id'
  | 'model_ref'
  | 'model_revision'
  | 'dataset_version_id'
  | 'evaluation_suite_id'
  | 'primary_metric'
  | 'compute_profile_id'
  | 'budget_unit'
  | 'budget_limit'
  | 'max_attempts'
  | 'minimum_improvement'
  | 'ssh_device_id'
  | 'executor_profile_id'
  | 'template_id'
  | 'campaign_id'

export type SetupEditorDraft = Readonly<Partial<Record<SetupEditorFieldId, string>>>

export interface SetupReceipt {
  stepId: DurableSetupStepId
  status: 'complete' | 'partial' | 'failed'
  receiptId: string
  summary: string
  retryable?: boolean
}

export interface SetupBlocker {
  stepId: SetupStepId
  code: string
  summary: string
}

export interface OptionalIntegration {
  id: string
  label: string
  status: 'unconfigured' | 'configured' | 'unavailable'
}

export interface ServerDoctorReadiness {
  observedAt: string
  isLatestServerDoctor: boolean
  launchReady: boolean
  controllerIdentityVerified: boolean
  computeIdentityVerified: boolean
  state: 'live' | 'reconciling' | 'stale' | 'offline' | 'error'
}

export interface SetupFlowInput {
  receipts: readonly SetupReceipt[]
  blockers: readonly SetupBlocker[]
  doctor: ServerDoctorReadiness | null
  optionalIntegrations?: readonly OptionalIntegration[]
  drafts?: SetupEditorDraft
  availableMutationContracts?: readonly SetupMutationContract[]
}

export interface SetupEditorFieldViewModel {
  id: SetupEditorFieldId
  label: string
  placeholder: string
  inputMode: 'text' | 'decimal' | 'numeric'
  value: string
  valid: boolean
  validationMessage: string | null
}

export interface SetupStepEditorViewModel {
  fields: SetupEditorFieldViewModel[]
  ready: boolean
}

export interface SetupStepViewModel {
  id: SetupStepId
  label: string
  guidance: string
  status: 'not_started' | 'partial' | 'blocked' | 'complete' | 'ready'
  receipt: SetupReceipt | null
  sequence: 'complete' | 'current' | 'locked'
  editor: SetupStepEditorViewModel | null
  action: SetupStepAction | null
  actionReceiptId: string | null
  mutationContract: SetupMutationContract | null
  actionAvailable: boolean
  actionGuidance: string | null
}

export interface SetupStartViewModel {
  enabled: boolean
  reason: string | null
}

export interface SetupFlowViewModel {
  steps: SetupStepViewModel[]
  blockers: SetupBlocker[]
  optionalIntegrations: Array<OptionalIntegration & { required: false }>
  nextStepId: SetupStepId
  start: SetupStartViewModel
}

const stepDetails: Record<SetupStepId, Pick<SetupStepViewModel, 'label' | 'guidance'>> = {
  source: {
    label: 'Source',
    guidance: 'Bind the approved source before creating campaign resources.'
  },
  model: {
    label: 'Model identity & revision',
    guidance:
      'Choose an explicit trainable model identity and immutable revision. No model is selected or downloaded automatically.'
  },
  data: {
    label: 'Data',
    guidance: 'Bind the approved dataset identity and access policy.'
  },
  evaluator: {
    label: 'Evaluator',
    guidance: 'Bind the evaluator and its versioned acceptance criteria.'
  },
  compute: {
    label: 'Private SSH compute',
    guidance:
      'Select a registered local or private SSH compute binding; no alternate compute target is selected implicitly.'
  },
  budget_stops: {
    label: 'Budget & stops',
    guidance: 'Set explicit spend, duration, and stop boundaries.'
  },
  activation_doctor: {
    label: 'Worker, controller & doctor',
    guidance:
      'Bind the registered SSH device and executor profile, bring the resident controller online, then retain the latest server doctor receipt.'
  },
  campaign_creation: {
    label: 'Campaign creation',
    guidance: 'Create the durable campaign once its immutable bindings are ready.'
  },
  baseline_review: {
    label: 'Baseline review',
    guidance: 'Review the sealed baseline evidence before launch.'
  },
  start: {
    label: 'Start',
    guidance: 'Start only from the latest live server doctor result.'
  }
}

const SAFE_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const IMMUTABLE_REVISION = /^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$/

interface EditorFieldDefinition {
  id: SetupEditorFieldId
  label: string
  placeholder: string
  inputMode?: SetupEditorFieldViewModel['inputMode']
  validate: (value: string) => string | null
}

function requiredIdentifier(value: string): string | null {
  if (!value) return 'Required logical identifier.'
  return SAFE_IDENTIFIER.test(value)
    ? null
    : 'Use a safe logical identifier (letters, numbers, dot, colon, underscore, or hyphen).'
}

function containsControlCharacters(value: string): boolean {
  return [...value].some((character) => {
    const code = character.charCodeAt(0)
    return code <= 31 || code === 127
  })
}

function requiredReference(value: string): string | null {
  if (!value) return 'Required operator-selected model reference.'
  return value.length <= 512 && !containsControlCharacters(value)
    ? null
    : 'Use a bounded model reference without control characters.'
}

function positiveNumber(value: string): string | null {
  if (!value) return 'Required bounded value.'
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? null : 'Enter a value greater than zero.'
}

function positiveInteger(value: string): string | null {
  if (!value) return 'Required bounded count.'
  return /^[1-9][0-9]*$/.test(value) ? null : 'Enter a whole number greater than zero.'
}

const editorDefinitions: Partial<Record<DurableSetupStepId, readonly EditorFieldDefinition[]>> = {
  source: [
    {
      id: 'source_profile_id',
      label: 'Approved source profile ID',
      placeholder: 'registered-source-profile-id',
      validate: requiredIdentifier
    }
  ],
  model: [
    {
      id: 'model_ref',
      label: 'Trainable model reference',
      placeholder: 'operator-selected immutable model reference',
      validate: requiredReference
    },
    {
      id: 'model_revision',
      label: 'Immutable content revision',
      placeholder: '40 or 64 character revision',
      validate: (value) =>
        IMMUTABLE_REVISION.test(value)
          ? null
          : 'Enter an exact 40 or 64 character content revision.'
    }
  ],
  data: [
    {
      id: 'dataset_version_id',
      label: 'Dataset version ID',
      placeholder: 'ledger-dataset-version-id',
      validate: requiredIdentifier
    }
  ],
  evaluator: [
    {
      id: 'evaluation_suite_id',
      label: 'Evaluation suite ID',
      placeholder: 'ledger-evaluation-suite-id',
      validate: requiredIdentifier
    },
    {
      id: 'primary_metric',
      label: 'Exact primary metric ID',
      placeholder: 'primary-metric-id',
      validate: requiredIdentifier
    }
  ],
  compute: [
    {
      id: 'compute_profile_id',
      label: 'Registered private compute profile ID',
      placeholder: 'registered-private-compute-profile-id',
      validate: requiredIdentifier
    }
  ],
  budget_stops: [
    {
      id: 'budget_unit',
      label: 'Budget unit',
      placeholder: 'gpu_hours',
      validate: requiredIdentifier
    },
    {
      id: 'budget_limit',
      label: 'Budget limit',
      placeholder: 'bounded limit',
      inputMode: 'decimal',
      validate: positiveNumber
    },
    {
      id: 'max_attempts',
      label: 'Maximum attempts',
      placeholder: 'bounded count',
      inputMode: 'numeric',
      validate: positiveInteger
    },
    {
      id: 'minimum_improvement',
      label: 'Minimum improvement',
      placeholder: 'required delta',
      inputMode: 'decimal',
      validate: positiveNumber
    }
  ],
  activation_doctor: [
    {
      id: 'ssh_device_id',
      label: 'Registered SSH device ID',
      placeholder: 'registered-ssh-device-id',
      validate: requiredIdentifier
    },
    {
      id: 'executor_profile_id',
      label: 'Executor profile ID',
      placeholder: 'installation-executor-profile-id',
      validate: requiredIdentifier
    }
  ],
  campaign_creation: [
    {
      id: 'template_id',
      label: 'Installed template ID',
      placeholder: 'installation-template-id',
      validate: requiredIdentifier
    },
    {
      id: 'campaign_id',
      label: 'Campaign ID',
      placeholder: 'operator-chosen-campaign-id',
      validate: requiredIdentifier
    }
  ]
}

function editorFor(
  stepId: DurableSetupStepId,
  drafts: SetupEditorDraft
): SetupStepEditorViewModel | null {
  const definitions = editorDefinitions[stepId]
  if (!definitions) return null
  const fields = definitions.map((definition): SetupEditorFieldViewModel => {
    const raw = drafts[definition.id]
    const value = typeof raw === 'string' && raw.length <= 512 ? raw : ''
    const validationMessage = definition.validate(value)
    return {
      id: definition.id,
      label: definition.label,
      placeholder: definition.placeholder,
      inputMode: definition.inputMode ?? 'text',
      value,
      valid: validationMessage === null,
      validationMessage
    }
  })
  return { fields, ready: fields.every((field) => field.valid) }
}

function validReceipt(receipt: SetupReceipt): boolean {
  return (
    SAFE_IDENTIFIER.test(receipt.receiptId) &&
    typeof receipt.summary === 'string' &&
    receipt.summary.length > 0 &&
    receipt.summary.length <= 500 &&
    !containsControlCharacters(receipt.summary)
  )
}

function firstReceiptFor(
  stepId: DurableSetupStepId,
  receipts: readonly SetupReceipt[]
): SetupReceipt | null {
  const safe = receipts.filter((receipt) => receipt.stepId === stepId && validReceipt(receipt))
  return safe.find((receipt) => receipt.status === 'complete') ?? safe[0] ?? null
}

function hasInvalidReceiptFor(
  stepId: DurableSetupStepId,
  receipts: readonly SetupReceipt[]
): boolean {
  return receipts.some((receipt) => receipt.stepId === stepId && !validReceipt(receipt))
}

const actionGuidance: Record<DurableSetupStepId, string> = {
  source:
    'Use `bashgym campaign setup-autoresearch` in the workspace terminal. Desktop definition writes do not yet have a durable bridge contract.',
  model:
    'Use `bashgym campaign setup-autoresearch` in the workspace terminal. Desktop definition writes do not yet have a durable bridge contract.',
  data: 'Use `bashgym campaign setup-autoresearch` in the workspace terminal. Desktop definition writes do not yet have a durable bridge contract.',
  evaluator:
    'Use `bashgym campaign setup-autoresearch` in the workspace terminal. Desktop definition writes do not yet have a durable bridge contract.',
  compute:
    'Use `bashgym campaign activate-autoresearch` in the workspace terminal. Private device and compute material stay installation-owned.',
  budget_stops:
    'Use `bashgym campaign setup-autoresearch` in the workspace terminal. Desktop definition writes do not yet have a durable bridge contract.',
  activation_doctor:
    'Use `bashgym campaign activate-autoresearch`, then `bashgym campaign doctor`, in the workspace terminal. No durable desktop activation or doctor contract is mounted.',
  campaign_creation:
    'Mount the authenticated `campaign.create_from_template` contract, or run `bashgym campaign create` in the workspace terminal.',
  baseline_review:
    'Use the authenticated campaign baseline contract from an existing campaign. This pre-campaign setup shell cannot invent baseline state.'
}

function mutationContractFor(stepId: DurableSetupStepId): SetupMutationContract | null {
  return stepId === 'campaign_creation' ? 'campaign.create_from_template' : null
}

function startReason(doctor: ServerDoctorReadiness | null): string | null {
  if (!doctor) return 'Run the latest server doctor before starting.'
  if (!doctor.isLatestServerDoctor) return 'Refresh the server doctor result before starting.'
  if (!doctor.launchReady) return 'The latest server doctor has not declared launch readiness.'
  if (!doctor.controllerIdentityVerified)
    return 'Verify the controller execution identity with the server doctor.'
  if (!doctor.computeIdentityVerified)
    return 'Verify the registered compute identity with the server doctor.'
  if (doctor.state !== 'live') return 'Wait for the server doctor state to be live before starting.'
  return null
}

function launchBlocker(doctor: ServerDoctorReadiness | null): SetupBlocker | null {
  const reason = startReason(doctor)
  if (!reason) return null
  const code = !doctor
    ? 'server_doctor_required'
    : !doctor.isLatestServerDoctor
      ? 'server_doctor_stale'
      : !doctor.launchReady
        ? 'launch_not_ready'
        : !doctor.controllerIdentityVerified
          ? 'controller_identity_unverified'
          : !doctor.computeIdentityVerified
            ? 'compute_identity_unverified'
            : 'server_state_not_live'
  return { stepId: 'start', code, summary: reason }
}

function stepIndex(stepId: SetupStepId): number {
  return setupStepOrder.indexOf(stepId)
}

export function buildSetupFlowModel(input: SetupFlowInput): SetupFlowViewModel {
  const durableSteps = setupStepOrder.filter((step): step is DurableSetupStepId => step !== 'start')
  const hasCompletedDurableSetup = durableSteps.every(
    (step) => firstReceiptFor(step, input.receipts)?.status === 'complete'
  )
  const start = {
    enabled: hasCompletedDurableSetup && startReason(input.doctor) === null,
    reason: startReason(input.doctor)
  }
  const blockers = [...input.blockers]
  const doctorBlocker = hasCompletedDurableSetup ? launchBlocker(input.doctor) : null
  if (doctorBlocker) blockers.push(doctorBlocker)
  blockers.sort((left, right) => stepIndex(left.stepId) - stepIndex(right.stepId))

  const statuses = new Map<SetupStepId, SetupStepViewModel['status']>()
  durableSteps.forEach((id) => {
    const receipt = firstReceiptFor(id, input.receipts)
    if (hasInvalidReceiptFor(id, input.receipts)) statuses.set(id, 'blocked')
    else if (receipt?.status === 'complete') statuses.set(id, 'complete')
    else if (receipt?.status === 'partial') statuses.set(id, 'partial')
    else if (receipt?.status === 'failed' || blockers.some((blocker) => blocker.stepId === id))
      statuses.set(id, 'blocked')
    else statuses.set(id, 'not_started')
  })
  statuses.set('start', start.enabled ? 'ready' : 'blocked')
  const nextStepId = durableSteps.find((step) => statuses.get(step) !== 'complete') ?? 'start'
  const availableContracts = new Set(input.availableMutationContracts ?? [])

  const steps = setupStepOrder.map((id): SetupStepViewModel => {
    const detail = stepDetails[id]
    if (id === 'start') {
      const action = !start.enabled && nextStepId === 'start' ? 'remediate' : null
      return {
        id,
        ...detail,
        status: statuses.get(id)!,
        sequence: nextStepId === 'start' ? 'current' : 'locked',
        editor: null,
        receipt: null,
        action,
        actionReceiptId: null,
        mutationContract: null,
        actionAvailable: action !== null,
        actionGuidance:
          nextStepId === 'start'
            ? start.reason
            : 'Complete the durable setup receipts before reviewing Start readiness.'
      }
    }
    const receipt = firstReceiptFor(id, input.receipts)
    const status = statuses.get(id)!
    const sequence = status === 'complete' ? 'complete' : id === nextStepId ? 'current' : 'locked'
    const editor = editorFor(id, input.drafts ?? {})
    let action: SetupStepAction | null = null
    if (sequence === 'current') {
      if (hasInvalidReceiptFor(id, input.receipts)) action = 'remediate'
      else if (receipt?.status === 'partial') action = receipt.retryable ? 'resume' : 'remediate'
      else if (receipt?.status === 'failed') action = receipt.retryable ? 'retry' : 'remediate'
      else if (status === 'blocked') action = 'remediate'
      else action = 'continue'
    }
    const mutationContract = mutationContractFor(id)
    const isLocalRemediation = action === 'remediate'
    const hasMountedContract = mutationContract !== null && availableContracts.has(mutationContract)
    const actionAvailable =
      action !== null &&
      (isLocalRemediation ||
        (hasMountedContract && (action !== 'continue' || editor?.ready !== false)))
    const lockedBy = durableSteps.find((step) => statuses.get(step) !== 'complete')
    return {
      id,
      ...detail,
      status,
      sequence,
      editor,
      receipt,
      action,
      actionReceiptId:
        action === 'resume' || action === 'retry' ? (receipt?.receiptId ?? null) : null,
      mutationContract,
      actionAvailable,
      actionGuidance:
        sequence === 'locked' && lockedBy
          ? `Complete ${stepDetails[lockedBy].label} first; later durable steps remain read-only.`
          : action !== null && !actionAvailable
            ? actionGuidance[id]
            : null
    }
  })

  return {
    steps,
    blockers,
    optionalIntegrations: (input.optionalIntegrations ?? []).map((integration) => ({
      ...integration,
      required: false
    })),
    nextStepId,
    start
  }
}
