export const guidedSetupSteps = ['template', 'installation', 'model', 'data', 'compute', 'evaluation'] as const
export type GuidedSetupStep = typeof guidedSetupSteps[number]
export type GuidedBindingKind = Exclude<GuidedSetupStep, 'template' | 'installation'>

const bindingKinds = ['model', 'data', 'compute', 'evaluation'] as const
const publicId = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const sessionId = /^setupsess_[0-9a-f]{32}$/
const installationId = /^ins_[0-9a-f]{32}$/
const stepReceiptId = /^setupstep_[0-9a-f]{32}$/
const validationReceiptId = /^setuprcpt_[0-9a-f]{32}$/
const hexDigest = /^[0-9a-f]{64}$/
const sealedDigest = /^sha256:[0-9a-f]{64}$/
const publicDisplayLabel = /^[A-Za-z0-9][A-Za-z0-9 .+_-]{0,95}$/
const privateLabelCanary = /(?:https?:\/\/|ssh:\/\/|file:\/\/|[A-Za-z]:\\|\/(?:home|Users)\/|(?:ghp|sk-proj|sk_live|xox[baprs]|AKIA|AIza)[_-]?[A-Za-z0-9])/i
const canonicalUtc = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{6}))?Z$/
const guidedSetupLimit = 32

type PublicRecord = Record<string, unknown>

export interface GuidedSetupBindings {
  model: string
  data: string
  compute: string
  evaluation: string
}

export interface GuidedSetupBindingChoice {
  logical_id: string
  availability: 'reachable' | 'inaccessible' | 'unknown'
  selectable: boolean
  reason_codes: string[]
  display_label?: string
  integration_label?: string
}

export interface GuidedSetupInstallation {
  installation_id: string
  ready: boolean
  reason_codes: string[]
  bindings: Record<GuidedBindingKind, GuidedSetupBindingChoice[]>
  truncation: { truncated: boolean; reason_codes: string[]; limit_per_kind: number; kinds: GuidedBindingKind[] }
}

export interface GuidedSetupTemplate {
  schema_version: 'guided_setup_template.v1'
  template_id: string
  definition_digest: string
  quality_claim_eligible: boolean
  required_bindings: GuidedSetupBindings
}

export interface GuidedSetupStepReceipt {
  schema_version: 'guided_setup_step_receipt.v1'
  receipt_id: string
  session_id: string
  version: number
  step: GuidedSetupStep
  selection_id: string
  state_digest: string
  previous_receipt_id: string | null
  previous_receipt_digest: string | null
  created_at: string
  receipt_digest: string
}

export interface GuidedSetupSession {
  schema_version: 'guided_setup_session.v1'
  workspace_id: string
  session_id: string
  version: number
  completed_steps: GuidedSetupStep[]
  selections: {
    template_id: string | null
    installation_id: string | null
    bindings: Record<GuidedBindingKind, string | null>
  }
  ready_for_validation: boolean
  reason_codes: string[]
  latest_receipt: GuidedSetupStepReceipt
  updated_at: string
}

export interface GuidedSetupContext {
  schema_version: 'guided_setup_context.v1'
  workspace_id: string
  templates: GuidedSetupTemplate[]
  installations: GuidedSetupInstallation[]
  session: GuidedSetupSession | null
  reason_codes: string[]
  truncation: {
    truncated: boolean
    reason_codes: string[]
    limits: { templates: number; installations: number; bindings_per_kind: number }
  }
}

export interface GuidedSetupDoctor {
  schema_version: 'guided_setup_doctor.v1'
  workspace_id: string
  template_id: string
  definition_digest: string
  draft_digest: string
  ready: boolean
  blocking_codes: string[]
  binding_references: Array<{ kind: GuidedBindingKind; logical_id: string; availability: string }>
}

export interface GuidedSetupValidation extends GuidedSetupDoctor { receipt_id: string }

export interface GuidedSetupMutation {
  schema_version: 'guided_setup_session_mutation.v1'
  session: GuidedSetupSession
  receipt: GuidedSetupStepReceipt
}

export interface GuidedSetupCreation {
  workspace_id: string
  campaign_id: string
  title: string
  status: string
  replayed: boolean
  validation_receipt_id: string
  binding_references: GuidedSetupBindings
}

export interface GuidedSetupOption {
  id: string
  label: string
  detail: string | null
  selectable: boolean
  reasonCodes: string[]
}

export interface GuidedSetupView {
  currentStep: GuidedSetupStep | 'doctor' | 'create'
  completedCount: number
  options: GuidedSetupOption[]
  truncationReasons: string[]
  session: GuidedSetupSession | null
}

function record(value: unknown): PublicRecord | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null ? value as PublicRecord : null
}

function exact(value: unknown, required: readonly string[], optional: readonly string[] = []): PublicRecord | null {
  const item = record(value)
  if (!item) return null
  const keys = Object.keys(item)
  if (required.some((key) => !(key in item))) return null
  const allowed = new Set([...required, ...optional])
  return keys.every((key) => allowed.has(key)) ? item : null
}

function safeString(value: unknown, pattern = publicId, max = 160): value is string {
  return typeof value === 'string' && value.length <= max && pattern.test(value)
}

function safeReasonCodes(value: unknown): value is string[] {
  return Array.isArray(value) && value.length <= 64 && value.every((item) => safeString(item))
}

function boundedInteger(value: unknown, max = 10_000): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= max
}

function isCanonicalUtc(value: unknown): value is string {
  if (typeof value !== 'string') return false
  const match = canonicalUtc.exec(value)
  if (!match) return false
  const [, year, month, day, hour, minute, second, fraction = '000000'] = match
  const date = new Date(Date.UTC(
    Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute),
    Number(second), Number(fraction.slice(0, 3)),
  ))
  return !Number.isNaN(date.getTime())
    && date.getUTCFullYear() === Number(year)
    && date.getUTCMonth() + 1 === Number(month)
    && date.getUTCDate() === Number(day)
    && date.getUTCHours() === Number(hour)
    && date.getUTCMinutes() === Number(minute)
    && date.getUTCSeconds() === Number(second)
}

function parseBindings(value: unknown): GuidedSetupBindings | null {
  const item = exact(value, bindingKinds)
  if (!item || !bindingKinds.every((kind) => safeString(item[kind]))) return null
  return Object.fromEntries(bindingKinds.map((kind) => [kind, item[kind]])) as unknown as GuidedSetupBindings
}

function parseBindingChoice(value: unknown): GuidedSetupBindingChoice | null {
  const item = exact(value, ['logical_id', 'availability', 'selectable', 'reason_codes'], ['display_label', 'integration_label'])
  if (!item || !safeString(item.logical_id) || !['reachable', 'inaccessible', 'unknown'].includes(String(item.availability))
    || typeof item.selectable !== 'boolean' || !safeReasonCodes(item.reason_codes)) return null
  if (item.display_label !== undefined && (
    typeof item.display_label !== 'string'
    || !publicDisplayLabel.test(item.display_label)
    || privateLabelCanary.test(item.display_label)
  )) return null
  if (item.integration_label !== undefined && item.integration_label !== 'NeMo') return null
  if (item.selectable !== (item.availability === 'reachable')) return null
  return item as unknown as GuidedSetupBindingChoice
}

function parseStepReceipt(value: unknown): GuidedSetupStepReceipt | null {
  const item = exact(value, [
    'schema_version', 'receipt_id', 'session_id', 'version', 'step', 'selection_id', 'state_digest',
    'previous_receipt_id', 'previous_receipt_digest', 'created_at', 'receipt_digest',
  ])
  if (!item || item.schema_version !== 'guided_setup_step_receipt.v1' || !safeString(item.receipt_id, stepReceiptId)
    || !safeString(item.session_id, sessionId) || !boundedInteger(item.version, 6) || Number(item.version) < 1
    || item.step !== guidedSetupSteps[Number(item.version) - 1]
    || !safeString(item.selection_id) || !safeString(item.state_digest, hexDigest, 64)
    || !(item.previous_receipt_id === null || safeString(item.previous_receipt_id, stepReceiptId))
    || !(item.previous_receipt_digest === null || safeString(item.previous_receipt_digest, sealedDigest, 71))
    || ((Number(item.version) === 1) !== (item.previous_receipt_id === null && item.previous_receipt_digest === null))
    || ((item.previous_receipt_id === null) !== (item.previous_receipt_digest === null))
    || !isCanonicalUtc(item.created_at) || !safeString(item.receipt_digest, sealedDigest, 71)) return null
  return item as unknown as GuidedSetupStepReceipt
}

function parseSession(value: unknown): GuidedSetupSession | null {
  const item = exact(value, [
    'schema_version', 'workspace_id', 'session_id', 'version', 'completed_steps', 'selections',
    'ready_for_validation', 'reason_codes', 'latest_receipt', 'updated_at',
  ])
  const selections = item ? exact(item.selections, ['template_id', 'installation_id', 'bindings']) : null
  const bindings = selections ? exact(selections.bindings, bindingKinds) : null
  const receipt = item ? parseStepReceipt(item.latest_receipt) : null
  if (!item || item.schema_version !== 'guided_setup_session.v1' || !safeString(item.workspace_id)
    || !safeString(item.session_id, sessionId) || !boundedInteger(item.version, 6) || Number(item.version) < 1
    || !Array.isArray(item.completed_steps) || item.completed_steps.length !== item.version
    || !item.completed_steps.every((step, index) => step === guidedSetupSteps[index])
    || !selections || !(selections.template_id === null || safeString(selections.template_id))
    || !(selections.installation_id === null || safeString(selections.installation_id, installationId)) || !bindings
    || !bindingKinds.every((kind) => bindings[kind] === null || safeString(bindings[kind]))
    || typeof item.ready_for_validation !== 'boolean' || !safeReasonCodes(item.reason_codes) || !receipt
    || receipt.session_id !== item.session_id || receipt.version !== item.version
    || item.ready_for_validation !== (Number(item.version) === guidedSetupSteps.length)
    || !isCanonicalUtc(item.updated_at)) return null
  const selected = [selections.template_id, selections.installation_id, ...bindingKinds.map((kind) => bindings[kind])]
  if (selected.some((value, index) => index < Number(item.version) ? value === null : value !== null)) return null
  if (receipt.selection_id !== selected[Number(item.version) - 1]) return null
  return item as unknown as GuidedSetupSession
}

export function parseGuidedSetupContext(value: unknown): GuidedSetupContext | null {
  const item = exact(value, ['schema_version', 'workspace_id', 'templates', 'installations', 'session', 'reason_codes', 'truncation'])
  if (!item || item.schema_version !== 'guided_setup_context.v1' || !safeString(item.workspace_id)
    || !Array.isArray(item.templates) || item.templates.length > guidedSetupLimit
    || !Array.isArray(item.installations) || item.installations.length > guidedSetupLimit
    || !safeReasonCodes(item.reason_codes)) return null
  const templates = item.templates.map((value): GuidedSetupTemplate | null => {
    const row = exact(value, ['schema_version', 'template_id', 'definition_digest', 'quality_claim_eligible', 'required_bindings'])
    const required = row ? parseBindings(row.required_bindings) : null
    return row && row.schema_version === 'guided_setup_template.v1' && safeString(row.template_id)
      && safeString(row.definition_digest, hexDigest, 64) && typeof row.quality_claim_eligible === 'boolean' && required
      ? { ...row, required_bindings: required } as unknown as GuidedSetupTemplate : null
  })
  if (templates.some((row) => row === null)) return null
  const installations = item.installations.map((value): GuidedSetupInstallation | null => {
    const row = exact(value, ['installation_id', 'ready', 'reason_codes', 'bindings', 'truncation'])
    const bindings = row ? exact(row.bindings, bindingKinds) : null
    const truncation = row ? exact(row.truncation, ['truncated', 'reason_codes', 'limit_per_kind', 'kinds']) : null
    if (!row || !safeString(row.installation_id, installationId) || typeof row.ready !== 'boolean' || !safeReasonCodes(row.reason_codes)
      || !bindings || !truncation || typeof truncation.truncated !== 'boolean' || !safeReasonCodes(truncation.reason_codes)
      || truncation.limit_per_kind !== guidedSetupLimit || !Array.isArray(truncation.kinds)
      || new Set(truncation.kinds).size !== truncation.kinds.length
      || !truncation.kinds.every((kind) => bindingKinds.includes(kind as GuidedBindingKind))) return null
    const parsedBindings = Object.fromEntries(bindingKinds.map((kind) => {
      const rows = bindings[kind]
      if (!Array.isArray(rows) || rows.length > Number(truncation.limit_per_kind)) return [kind, null]
      const parsed = rows.map(parseBindingChoice)
      return [kind, parsed.some((choice) => choice === null) ? null : parsed]
    })) as Record<GuidedBindingKind, GuidedSetupBindingChoice[] | null>
    if (bindingKinds.some((kind) => parsedBindings[kind] === null)) return null
    return { ...row, bindings: parsedBindings, truncation } as unknown as GuidedSetupInstallation
  })
  if (installations.some((row) => row === null)) return null
  const truncation = exact(item.truncation, ['truncated', 'reason_codes', 'limits'])
  const limits = truncation ? exact(truncation.limits, ['templates', 'installations', 'bindings_per_kind']) : null
  if (!truncation || typeof truncation.truncated !== 'boolean' || !safeReasonCodes(truncation.reason_codes) || !limits
    || limits.templates !== guidedSetupLimit || limits.installations !== guidedSetupLimit
    || limits.bindings_per_kind !== guidedSetupLimit) return null
  const session = item.session === null ? null : parseSession(item.session)
  if (item.session !== null && (!session || session.workspace_id !== item.workspace_id)) return null
  if (new Set(templates.map((row) => row!.template_id)).size !== templates.length
    || new Set(installations.map((row) => row!.installation_id)).size !== installations.length) return null
  return { ...item, templates, installations, session, truncation: { ...truncation, limits } } as unknown as GuidedSetupContext
}

function parseDoctorBase(value: unknown, withReceipt: boolean): GuidedSetupDoctor | GuidedSetupValidation | null {
  const required = ['schema_version', 'workspace_id', 'template_id', 'definition_digest', 'draft_digest', 'ready', 'blocking_codes', 'binding_references']
  if (withReceipt) required.push('receipt_id')
  const item = exact(value, required)
  if (!item || item.schema_version !== 'guided_setup_doctor.v1' || !safeString(item.workspace_id) || !safeString(item.template_id)
    || !safeString(item.definition_digest, hexDigest, 64) || !safeString(item.draft_digest, hexDigest, 64)
    || typeof item.ready !== 'boolean' || !safeReasonCodes(item.blocking_codes) || !Array.isArray(item.binding_references)
    || item.binding_references.length !== 4) return null
  const references = item.binding_references.map((value) => exact(value, ['kind', 'logical_id', 'availability']))
  if (references.some((row) => !row || !bindingKinds.includes(row.kind as GuidedBindingKind)
    || !safeString(row.logical_id) || !['reachable', 'inaccessible', 'unknown'].includes(String(row.availability)))) return null
  if (new Set(references.map((row) => row!.kind)).size !== bindingKinds.length) return null
  if (item.ready && (item.blocking_codes.length !== 0 || references.some((row) => row!.availability !== 'reachable'))) return null
  if (withReceipt && !safeString(item.receipt_id, validationReceiptId)) return null
  return item as unknown as GuidedSetupDoctor | GuidedSetupValidation
}

export function parseGuidedSetupDoctor(value: unknown): GuidedSetupDoctor | null {
  return parseDoctorBase(value, false) as GuidedSetupDoctor | null
}

export function parseGuidedSetupValidation(value: unknown): GuidedSetupValidation | null {
  return parseDoctorBase(value, true) as GuidedSetupValidation | null
}

export function parseGuidedSetupMutation(value: unknown): GuidedSetupMutation | null {
  const item = exact(value, ['schema_version', 'session', 'receipt'])
  const session = item ? parseSession(item.session) : null
  const receipt = item ? parseStepReceipt(item.receipt) : null
  return item?.schema_version === 'guided_setup_session_mutation.v1' && session && receipt
    && session.latest_receipt.receipt_id === receipt.receipt_id
    ? { schema_version: item.schema_version, session, receipt } : null
}

export function parseGuidedSetupCreation(value: unknown): GuidedSetupCreation | null {
  const item = exact(value, ['campaign', 'event', 'replayed', 'setup'], ['autoresearch'])
  const campaign = item ? record(item.campaign) : null
  const event = item ? record(item.event) : null
  const setup = item ? exact(item.setup, ['schema_version', 'validation_receipt_id', 'binding_references']) : null
  const bindings = setup ? parseBindings(setup.binding_references) : null
  if (!item || !campaign || !event || typeof item.replayed !== 'boolean' || !setup
    || setup.schema_version !== 'guided_setup_creation.v1' || !safeString(setup.validation_receipt_id, validationReceiptId)
    || !bindings || !safeString(campaign.workspace_id) || !safeString(campaign.campaign_id)
    || typeof campaign.title !== 'string' || campaign.title.length < 1 || campaign.title.length > 240
    || !safeString(campaign.status) || !safeString(event.event_id)) return null
  return {
    workspace_id: campaign.workspace_id,
    campaign_id: campaign.campaign_id,
    title: campaign.title,
    status: campaign.status,
    replayed: item.replayed,
    validation_receipt_id: setup.validation_receipt_id,
    binding_references: bindings,
  }
}

export function buildGuidedSetupView(context: GuidedSetupContext): GuidedSetupView {
  const completedCount = context.session?.version ?? 0
  const currentStep = completedCount >= guidedSetupSteps.length ? 'doctor' : guidedSetupSteps[completedCount]
  const installation = context.installations.find((item) => item.installation_id === context.session?.selections.installation_id)
  const template = context.templates.find((item) => item.template_id === context.session?.selections.template_id)
  let options: GuidedSetupOption[] = []
  if (currentStep === 'template') {
    options = context.templates.map((item) => ({ id: item.template_id, label: item.template_id, detail: item.quality_claim_eligible ? 'Quality-claim eligible' : null, selectable: true, reasonCodes: [] }))
  } else if (currentStep === 'installation') {
    options = context.installations.map((item) => ({ id: item.installation_id, label: item.installation_id, detail: 'Registered local/private installation', selectable: item.ready, reasonCodes: item.reason_codes }))
  } else if (bindingKinds.includes(currentStep as GuidedBindingKind) && installation && template) {
    const kind = currentStep as GuidedBindingKind
    options = installation.bindings[kind]
      .filter((item) => item.logical_id === template.required_bindings[kind])
      .map((item) => ({ id: item.logical_id, label: item.display_label || item.logical_id, detail: item.integration_label || null, selectable: item.selectable, reasonCodes: item.reason_codes }))
  }
  return { currentStep, completedCount, options, truncationReasons: [...context.truncation.reason_codes], session: context.session }
}
