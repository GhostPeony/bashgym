import type { CampaignFreshness } from '../../stores/campaignFreshness'

export const HUMAN_WORK_QUEUE_SCHEMA_VERSION = 'human_work_queue.v1' as const

export const HUMAN_WORK_LIMITS = Object.freeze({
  maxItems: 50,
  maxRubricChoices: 3,
  maxLabel: 80,
  maxInstructions: 2_000,
  maxPrompt: 2_000,
  maxSampleDisplay: 12_000,
  maxRationale: 2_000,
})

export type HumanWorkLifecycle = 'pending' | 'claimed' | 'submitted' | 'expired' | 'replaced'
export type HumanReviewDecision = 'prefer_left' | 'prefer_right' | 'no_material_difference'
export type HumanPromotionState = 'blocked_by_review' | 'awaiting_human_decision' | 'promoted' | 'not_required'
export type HumanWorkMutationAction = 'claim' | 'submit' | 'promotion'

export interface BlindedSamplePublicV1 {
  label: string
  display: string
}

export interface RubricChoicePublicV1 {
  choice_id: 'left' | 'right' | 'tie'
  label: string
}

export interface VersionedRubricPublicV1 {
  rubric_id: string
  version: number
  instructions: string
  choices: RubricChoicePublicV1[]
}

export interface SealedDecisionReceiptPublicV1 {
  receipt_id: string
  workspace_id: string
  campaign_id: string
  work_id: string
  campaign_revision: number
  item_version: number
  rubric_version: number
  decision: HumanReviewDecision
  sealed_at: string
  receipt_digest: string
}

export interface HumanWorkItemPublicV1 {
  work_id: string
  campaign_revision: number
  version: number
  state: HumanWorkLifecycle
  blocking: boolean
  rubric: VersionedRubricPublicV1
  sample: {
    prompt: string
    left: BlindedSamplePublicV1
    right: BlindedSamplePublicV1
  }
  lease_expires_at: string | null
  claimed_by_current_reviewer: boolean
  claim_idempotency_key: string
  submit_idempotency_key: string
  receipt: SealedDecisionReceiptPublicV1 | null
}

export interface HumanWorkQueuePublicV1 {
  schema_version: typeof HUMAN_WORK_QUEUE_SCHEMA_VERSION
  workspace_id: string
  campaign_id: string
  campaign_revision: number
  reviewer: {
    authenticated: boolean
    review_capability: boolean
  }
  items: HumanWorkItemPublicV1[]
  promotion: {
    state: HumanPromotionState
    version: number
    eligible_receipt_id: string | null
    idempotency_key: string
  }
  mutation?: {
    action: HumanWorkMutationAction
    state: 'pending' | 'conflict' | 'error'
    code: string
  }
}

type DeepReadonly<T> = T extends (...args: never[]) => unknown
  ? T
  : T extends readonly (infer U)[]
    ? readonly DeepReadonly<U>[]
    : T extends object
      ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
      : T

const PARSED_QUEUE_BRAND: unique symbol = Symbol('ParsedHumanWorkQueuePublicV1')
const OVERSIGHT_MODEL_BRAND: unique symbol = Symbol('HumanOversightViewModel')

export type ParsedHumanWorkQueuePublicV1 = DeepReadonly<HumanWorkQueuePublicV1> & {
  readonly [PARSED_QUEUE_BRAND]: true
}

interface MutationScope {
  workspaceId: string
  campaignId: string
  expectedCampaignRevision: number
}

export interface HumanWorkClaimRequest extends MutationScope {
  workId: string
  expectedVersion: number
  expectedState: 'pending' | 'expired'
  idempotencyKey: string
}

export interface HumanWorkSubmitRequest extends MutationScope {
  workId: string
  expectedVersion: number
  expectedRubricVersion: number
  expectedState: 'claimed'
  idempotencyKey: string
}

export interface HumanPromotionRequestBinding extends MutationScope {
  receiptId: string
  workId: string
  expectedItemVersion: number
  expectedRubricVersion: number
  expectedPromotionVersion: number
  expectedPromotionState: 'awaiting_human_decision'
  idempotencyKey: string
}

export interface HumanPromotionRequest extends HumanPromotionRequestBinding {
  decision: 'promote' | 'hold'
}

export interface HumanReviewResponseBinding {
  workspaceId: string
  campaignId: string
  campaignRevision: number
  workId: string
  itemVersion: number
  rubricVersion: number
}

export function humanReviewResponseKey(binding: HumanReviewResponseBinding): string {
  return JSON.stringify([
    binding.workspaceId,
    binding.campaignId,
    binding.campaignRevision,
    binding.workId,
    binding.itemVersion,
    binding.rubricVersion,
  ])
}

export interface HumanWorkItemViewModel {
  item: DeepReadonly<HumanWorkItemPublicV1>
  effectiveState: HumanWorkLifecycle
  invalidated: boolean
  offeredDecisions: readonly HumanReviewDecision[]
  receiptCurrent: boolean
  responseBinding: Readonly<HumanReviewResponseBinding>
  claim: Readonly<HumanWorkClaimRequest> | null
  submit: Readonly<HumanWorkSubmitRequest> | null
}

export interface HumanOversightViewModel {
  readonly [OVERSIGHT_MODEL_BRAND]: true
  queue: ParsedHumanWorkQueuePublicV1
  scope: Readonly<{ workspaceId: string; campaignId: string; campaignRevision: number }>
  scopeValid: boolean
  timeValid: boolean
  freshness: CampaignFreshness
  items: readonly HumanWorkItemViewModel[]
  mutationsEnabled: boolean
  remediation: string
  mutationMessage: string | null
  blocksComparison: boolean
  blocksPromotion: boolean
  promotion: Readonly<HumanPromotionRequestBinding> | null
}

const DURABLE_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const WORK_ID = /^hw_[A-Za-z0-9_-]{16,120}$/
const RUBRIC_ID = /^rub_[A-Za-z0-9_-]{8,120}$/
const RECEIPT_ID = /^hrc_[A-Za-z0-9_-]{16,120}$/
const IDEMPOTENCY_KEY = /^idem_[A-Za-z0-9_-]{16,120}$/
const RECEIPT_DIGEST = /^sha256:[a-f0-9]{64}$/
const MUTATION_CODE = /^[a-z][a-z0-9_]{0,79}$/
const CANONICAL_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/

const RUBRIC_DECISIONS: Readonly<Record<RubricChoicePublicV1['choice_id'], HumanReviewDecision>> = Object.freeze({
  left: 'prefer_left',
  right: 'prefer_right',
  tie: 'no_material_difference',
})

export function reviewDecisionForChoice(choiceId: string): HumanReviewDecision | null {
  return choiceId === 'left' || choiceId === 'right' || choiceId === 'tie'
    ? RUBRIC_DECISIONS[choiceId]
    : null
}

function hasExactKeys(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  return actual.length === expected.length && actual.every((key, index) => key === expected[index])
}

function hasRequiredAndOptionalKeys(value: unknown, required: readonly string[], optional: readonly string[]): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const actual = Object.keys(value)
  const allowed = [...required, ...optional]
  return required.every((key) => actual.includes(key)) && actual.every((key) => allowed.includes(key))
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0
}

function isBoundedString(value: unknown, maximum: number): value is string {
  if (typeof value !== 'string') return false
  const codePoints = Array.from(value).length
  return codePoints > 0 && codePoints <= maximum
}

function isCanonicalUtc(value: unknown): value is string {
  if (typeof value !== 'string' || !CANONICAL_UTC.test(value)) return false
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return false
  try {
    return new Date(timestamp).toISOString() === value
  } catch {
    return false
  }
}

function parseSample(value: unknown): BlindedSamplePublicV1 | null {
  if (!hasExactKeys(value, ['label', 'display'])) return null
  if (!isBoundedString(value.label, HUMAN_WORK_LIMITS.maxLabel) || !isBoundedString(value.display, HUMAN_WORK_LIMITS.maxSampleDisplay)) return null
  return { label: value.label, display: value.display }
}

function parseRubricChoice(value: unknown): RubricChoicePublicV1 | null {
  if (!hasExactKeys(value, ['choice_id', 'label']) || typeof value.choice_id !== 'string') return null
  if (!reviewDecisionForChoice(value.choice_id) || !isBoundedString(value.label, HUMAN_WORK_LIMITS.maxLabel)) return null
  return { choice_id: value.choice_id as RubricChoicePublicV1['choice_id'], label: value.label }
}

function parseRubric(value: unknown): VersionedRubricPublicV1 | null {
  if (!hasExactKeys(value, ['rubric_id', 'version', 'instructions', 'choices'])) return null
  if (typeof value.rubric_id !== 'string' || !RUBRIC_ID.test(value.rubric_id) || !isPositiveInteger(value.version)) return null
  if (!isBoundedString(value.instructions, HUMAN_WORK_LIMITS.maxInstructions) || !Array.isArray(value.choices) || value.choices.length < 1 || value.choices.length > HUMAN_WORK_LIMITS.maxRubricChoices) return null
  const choices = value.choices.map(parseRubricChoice)
  if (!choices.every((choice): choice is RubricChoicePublicV1 => choice !== null)) return null
  if (new Set(choices.map((choice) => choice.choice_id)).size !== choices.length) return null
  return { rubric_id: value.rubric_id, version: value.version, instructions: value.instructions, choices }
}

function parseReceipt(value: unknown, context: {
  workspaceId: string
  campaignId: string
  workId: string
  campaignRevision: number
  itemVersion: number
  rubric: VersionedRubricPublicV1
}): SealedDecisionReceiptPublicV1 | null {
  const keys = [
    'receipt_id', 'workspace_id', 'campaign_id', 'work_id', 'campaign_revision', 'item_version',
    'rubric_version', 'decision', 'sealed_at', 'receipt_digest',
  ]
  if (!hasExactKeys(value, keys)) return null
  if (typeof value.receipt_id !== 'string' || !RECEIPT_ID.test(value.receipt_id)) return null
  if (value.workspace_id !== context.workspaceId || value.campaign_id !== context.campaignId || value.work_id !== context.workId) return null
  if (value.campaign_revision !== context.campaignRevision || value.item_version !== context.itemVersion || value.rubric_version !== context.rubric.version) return null
  if (value.decision !== 'prefer_left' && value.decision !== 'prefer_right' && value.decision !== 'no_material_difference') return null
  const offered = context.rubric.choices.map((choice) => reviewDecisionForChoice(choice.choice_id))
  if (!offered.includes(value.decision) || !isCanonicalUtc(value.sealed_at)) return null
  if (typeof value.receipt_digest !== 'string' || !RECEIPT_DIGEST.test(value.receipt_digest)) return null
  return {
    receipt_id: value.receipt_id,
    workspace_id: context.workspaceId,
    campaign_id: context.campaignId,
    work_id: context.workId,
    campaign_revision: context.campaignRevision,
    item_version: context.itemVersion,
    rubric_version: context.rubric.version,
    decision: value.decision,
    sealed_at: value.sealed_at,
    receipt_digest: value.receipt_digest,
  }
}

function lifecycleIsValid(item: HumanWorkItemPublicV1): boolean {
  if (item.state === 'pending') return item.lease_expires_at === null && !item.claimed_by_current_reviewer && item.receipt === null
  if (item.state === 'claimed') return item.lease_expires_at !== null && item.receipt === null
  if (item.state === 'submitted') return item.lease_expires_at === null && !item.claimed_by_current_reviewer && item.receipt !== null
  if (item.state === 'expired') return item.lease_expires_at !== null && !item.claimed_by_current_reviewer && item.receipt === null
  return item.lease_expires_at === null && !item.claimed_by_current_reviewer && item.receipt === null
}

function parseItem(value: unknown, scope: { workspaceId: string; campaignId: string }): HumanWorkItemPublicV1 | null {
  const keys = [
    'work_id', 'campaign_revision', 'version', 'state', 'blocking', 'rubric', 'sample', 'lease_expires_at',
    'claimed_by_current_reviewer', 'claim_idempotency_key', 'submit_idempotency_key', 'receipt',
  ]
  if (!hasExactKeys(value, keys)) return null
  if (typeof value.work_id !== 'string' || !WORK_ID.test(value.work_id) || !isPositiveInteger(value.campaign_revision) || !isPositiveInteger(value.version)) return null
  if (value.state !== 'pending' && value.state !== 'claimed' && value.state !== 'submitted' && value.state !== 'expired' && value.state !== 'replaced') return null
  if (typeof value.blocking !== 'boolean' || typeof value.claimed_by_current_reviewer !== 'boolean') return null
  if (value.lease_expires_at !== null && !isCanonicalUtc(value.lease_expires_at)) return null
  if (typeof value.claim_idempotency_key !== 'string' || !IDEMPOTENCY_KEY.test(value.claim_idempotency_key)) return null
  if (typeof value.submit_idempotency_key !== 'string' || !IDEMPOTENCY_KEY.test(value.submit_idempotency_key)) return null
  const rubric = parseRubric(value.rubric)
  if (!rubric || !hasExactKeys(value.sample, ['prompt', 'left', 'right']) || !isBoundedString(value.sample.prompt, HUMAN_WORK_LIMITS.maxPrompt)) return null
  const left = parseSample(value.sample.left)
  const right = parseSample(value.sample.right)
  if (!left || !right || left.label === right.label) return null
  const receipt = value.receipt === null ? null : parseReceipt(value.receipt, {
    workspaceId: scope.workspaceId,
    campaignId: scope.campaignId,
    workId: value.work_id,
    campaignRevision: value.campaign_revision,
    itemVersion: value.version,
    rubric,
  })
  if (value.receipt !== null && receipt === null) return null
  const result: HumanWorkItemPublicV1 = {
    work_id: value.work_id,
    campaign_revision: value.campaign_revision,
    version: value.version,
    state: value.state,
    blocking: value.blocking,
    rubric,
    sample: { prompt: value.sample.prompt, left, right },
    lease_expires_at: value.lease_expires_at,
    claimed_by_current_reviewer: value.claimed_by_current_reviewer,
    claim_idempotency_key: value.claim_idempotency_key,
    submit_idempotency_key: value.submit_idempotency_key,
    receipt,
  }
  return lifecycleIsValid(result) ? result : null
}

function parseMutation(value: unknown): HumanWorkQueuePublicV1['mutation'] | undefined {
  if (value === undefined) return undefined
  if (!hasExactKeys(value, ['action', 'state', 'code'])) return undefined
  if ((value.action !== 'claim' && value.action !== 'submit' && value.action !== 'promotion') || (value.state !== 'pending' && value.state !== 'conflict' && value.state !== 'error')) return undefined
  if (typeof value.code !== 'string' || !MUTATION_CODE.test(value.code)) return undefined
  return { action: value.action, state: value.state, code: value.code }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const key of Reflect.ownKeys(value)) deepFreeze((value as Record<PropertyKey, unknown>)[key])
    Object.freeze(value)
  }
  return value
}

/** Parses, bounds, brands, and deeply freezes the exact public queue shape. */
export function parseHumanWorkQueue(value: unknown): ParsedHumanWorkQueuePublicV1 | null {
  const required = ['schema_version', 'workspace_id', 'campaign_id', 'campaign_revision', 'reviewer', 'items', 'promotion']
  if (!hasRequiredAndOptionalKeys(value, required, ['mutation'])) return null
  if (value.schema_version !== HUMAN_WORK_QUEUE_SCHEMA_VERSION) return null
  if (typeof value.workspace_id !== 'string' || !DURABLE_ID.test(value.workspace_id) || typeof value.campaign_id !== 'string' || !DURABLE_ID.test(value.campaign_id)) return null
  if (!isPositiveInteger(value.campaign_revision)) return null
  if (!hasExactKeys(value.reviewer, ['authenticated', 'review_capability']) || typeof value.reviewer.authenticated !== 'boolean' || typeof value.reviewer.review_capability !== 'boolean') return null
  if (!Array.isArray(value.items) || value.items.length > HUMAN_WORK_LIMITS.maxItems) return null
  const items = value.items.map((entry) => parseItem(entry, { workspaceId: value.workspace_id as string, campaignId: value.campaign_id as string }))
  if (!items.every((entry): entry is HumanWorkItemPublicV1 => entry !== null)) return null
  if (new Set(items.map((entry) => entry.work_id)).size !== items.length) return null
  const receiptIds = items.flatMap((entry) => entry.receipt ? [entry.receipt.receipt_id] : [])
  if (new Set(receiptIds).size !== receiptIds.length) return null
  if (!hasExactKeys(value.promotion, ['state', 'version', 'eligible_receipt_id', 'idempotency_key'])) return null
  const promotionRecord = value.promotion
  if (promotionRecord.state !== 'blocked_by_review' && promotionRecord.state !== 'awaiting_human_decision' && promotionRecord.state !== 'promoted' && promotionRecord.state !== 'not_required') return null
  if (!isPositiveInteger(promotionRecord.version) || typeof promotionRecord.idempotency_key !== 'string' || !IDEMPOTENCY_KEY.test(promotionRecord.idempotency_key)) return null
  if (promotionRecord.eligible_receipt_id !== null && (typeof promotionRecord.eligible_receipt_id !== 'string' || !RECEIPT_ID.test(promotionRecord.eligible_receipt_id))) return null
  const eligibleReceiptId = promotionRecord.eligible_receipt_id
  const eligibleReceipts = items.filter((entry) => entry.receipt?.receipt_id === eligibleReceiptId)
  if (promotionRecord.state === 'awaiting_human_decision' || promotionRecord.state === 'promoted') {
    if (eligibleReceipts.length !== 1 || eligibleReceipts[0]!.campaign_revision !== value.campaign_revision) return null
  } else if (eligibleReceiptId !== null) return null
  const idempotencyKeys = [
    promotionRecord.idempotency_key,
    ...items.flatMap((entry) => [entry.claim_idempotency_key, entry.submit_idempotency_key]),
  ]
  if (new Set(idempotencyKeys).size !== idempotencyKeys.length) return null
  const mutation = parseMutation(value.mutation)
  if (value.mutation !== undefined && !mutation) return null
  const result: HumanWorkQueuePublicV1 = {
    schema_version: HUMAN_WORK_QUEUE_SCHEMA_VERSION,
    workspace_id: value.workspace_id as string,
    campaign_id: value.campaign_id as string,
    campaign_revision: value.campaign_revision,
    reviewer: { authenticated: value.reviewer.authenticated, review_capability: value.reviewer.review_capability },
    items,
    promotion: {
      state: promotionRecord.state,
      version: promotionRecord.version,
      eligible_receipt_id: eligibleReceiptId,
      idempotency_key: promotionRecord.idempotency_key,
    },
    ...(mutation ? { mutation } : {}),
  }
  Object.defineProperty(result, PARSED_QUEUE_BRAND, { value: true, enumerable: false })
  return deepFreeze(result) as unknown as ParsedHumanWorkQueuePublicV1
}

function effectiveState(item: DeepReadonly<HumanWorkItemPublicV1>, now: number, timeValid: boolean): HumanWorkLifecycle {
  if (timeValid && item.state === 'claimed' && item.lease_expires_at && Date.parse(item.lease_expires_at) <= now) return 'expired'
  return item.state
}

function receiptIsCurrent(item: DeepReadonly<HumanWorkItemPublicV1>, campaignRevision: number, now: number, timeValid: boolean): boolean {
  return timeValid
    && item.state === 'submitted'
    && item.campaign_revision === campaignRevision
    && item.receipt !== null
    && item.receipt.campaign_revision === item.campaign_revision
    && item.receipt.item_version === item.version
    && item.receipt.rubric_version === item.rubric.version
    && Date.parse(item.receipt.sealed_at) <= now
}

function mutationMessage(mutation: DeepReadonly<HumanWorkQueuePublicV1['mutation']>): string | null {
  if (!mutation) return null
  if (mutation.state === 'pending') return `The ${mutation.action} request is pending. Wait for authoritative reconciliation.`
  if (mutation.state === 'conflict') return 'The review version changed before the request completed. Reconcile and use the current item version.'
  return `The ${mutation.action} request could not be completed (${mutation.code}). Reconcile before retrying.`
}

export function buildHumanOversightModel(input: {
  queue: ParsedHumanWorkQueuePublicV1
  authority: { workspaceId: string; campaignId: string }
  freshness: CampaignFreshness
  error: string | null
  now: string
}): HumanOversightViewModel {
  const { queue } = input
  const scopeValid = input.authority.workspaceId === queue.workspace_id && input.authority.campaignId === queue.campaign_id
  const timeValid = isCanonicalUtc(input.now)
  const now = timeValid ? Date.parse(input.now) : Number.NaN
  const revisionInvalid = queue.items.some((entry) => entry.campaign_revision !== queue.campaign_revision)
  const leaseLifecycleInvalid = timeValid && queue.items.some((entry) => (
    entry.state === 'expired' && entry.lease_expires_at !== null && Date.parse(entry.lease_expires_at) > now
  ))
  const futureReceipt = timeValid && queue.items.some((entry) => entry.receipt && Date.parse(entry.receipt.sealed_at) > now)
  const reviewerAllowed = queue.reviewer.authenticated && queue.reviewer.review_capability
  const mutationsEnabled = input.freshness === 'live'
    && scopeValid
    && timeValid
    && !revisionInvalid
    && !leaseLifecycleInvalid
    && !futureReceipt
    && !queue.mutation
    && reviewerAllowed
  const items: HumanWorkItemViewModel[] = queue.items.map((entry) => {
    const state = effectiveState(entry, now, timeValid)
    const receiptCurrent = receiptIsCurrent(entry, queue.campaign_revision, now, timeValid)
    const responseBinding = deepFreeze({
      workspaceId: queue.workspace_id,
      campaignId: queue.campaign_id,
      campaignRevision: queue.campaign_revision,
      workId: entry.work_id,
      itemVersion: entry.version,
      rubricVersion: entry.rubric.version,
    })
    const claim: HumanWorkClaimRequest | null = mutationsEnabled && (state === 'pending' || state === 'expired')
      ? {
          workspaceId: queue.workspace_id,
          campaignId: queue.campaign_id,
          workId: entry.work_id,
          expectedCampaignRevision: queue.campaign_revision,
          expectedVersion: entry.version,
          expectedState: state,
          idempotencyKey: entry.claim_idempotency_key,
        }
      : null
    const submit: HumanWorkSubmitRequest | null = mutationsEnabled && state === 'claimed' && entry.claimed_by_current_reviewer
      ? {
          workspaceId: queue.workspace_id,
          campaignId: queue.campaign_id,
          workId: entry.work_id,
          expectedCampaignRevision: queue.campaign_revision,
          expectedVersion: entry.version,
          expectedRubricVersion: entry.rubric.version,
          expectedState: 'claimed',
          idempotencyKey: entry.submit_idempotency_key,
        }
      : null
    return {
      item: entry,
      effectiveState: state,
      invalidated: entry.campaign_revision !== queue.campaign_revision,
      offeredDecisions: entry.rubric.choices.map((choice) => reviewDecisionForChoice(choice.choice_id)!),
      receiptCurrent,
      responseBinding,
      claim: claim ? deepFreeze(claim) : null,
      submit: submit ? deepFreeze(submit) : null,
    }
  })
  const eligible = items.filter((entry) => entry.item.receipt?.receipt_id === queue.promotion.eligible_receipt_id && receiptIsCurrent(entry.item, queue.campaign_revision, now, timeValid))
  const receipt = eligible.length === 1 ? eligible[0]!.item.receipt : null
  const promotion: HumanPromotionRequestBinding | null = mutationsEnabled && queue.promotion.state === 'awaiting_human_decision' && receipt
    ? {
        workspaceId: queue.workspace_id,
        campaignId: queue.campaign_id,
        receiptId: receipt.receipt_id,
        workId: receipt.work_id,
        expectedCampaignRevision: queue.campaign_revision,
        expectedItemVersion: receipt.item_version,
        expectedRubricVersion: receipt.rubric_version,
        expectedPromotionVersion: queue.promotion.version,
        expectedPromotionState: 'awaiting_human_decision',
        idempotencyKey: queue.promotion.idempotency_key,
      }
    : null
  const conflictMessage = mutationMessage(queue.mutation)
  const freshnessMessage = input.freshness === 'live' ? null : {
    reconciling: 'Campaign state is reconciling. Wait for a live authoritative refresh before changing review work.',
    stale: 'Campaign state is stale. Reconcile the campaign before changing review work.',
    offline: 'Campaign service is offline. Reconnect and reconcile before changing review work.',
    error: 'Campaign state has an error. Resolve it and reconcile before changing review work.',
  }[input.freshness]
  const remediation = !scopeValid
    ? 'The visible queue does not match the authoritative workspace and campaign selection. Reconcile the selected scope.'
    : !timeValid
      ? 'Canonical time authority is unavailable. Reconcile with the authenticated service before changing review work.'
      : freshnessMessage
        ? `${freshnessMessage}${input.error ? ` ${input.error}` : ''}`
        : revisionInvalid
          ? 'The campaign version changed. Reconcile the entire queue and claim replacement work before continuing.'
          : leaseLifecycleInvalid
            ? 'The lease lifecycle conflicts with authoritative time. Reconcile the queue before changing review work.'
            : futureReceipt
              ? 'A sealed receipt is newer than authoritative time. Reconcile the queue before changing review work.'
              : conflictMessage
              || (!queue.reviewer.authenticated ? 'Sign in with the authenticated desktop reviewer before changing review work.' : null)
              || (!queue.reviewer.review_capability ? 'A human reviewer capability is required; agents cannot approve or promote human work.' : null)
              || 'Review blinded samples against the versioned rubric. Promotion always requires an explicit human decision.'
  const blocksComparison = items.some((entry) => entry.item.blocking && !entry.receiptCurrent)
  const blocksPromotion = blocksComparison || queue.promotion.state === 'blocked_by_review' || queue.promotion.state === 'awaiting_human_decision'
  const result = {
    queue,
    scope: { workspaceId: queue.workspace_id, campaignId: queue.campaign_id, campaignRevision: queue.campaign_revision },
    scopeValid,
    timeValid,
    freshness: input.freshness,
    items,
    mutationsEnabled,
    remediation,
    mutationMessage: conflictMessage,
    blocksComparison,
    blocksPromotion,
    promotion: promotion ? deepFreeze(promotion) : null,
  } as Omit<HumanOversightViewModel, typeof OVERSIGHT_MODEL_BRAND> & Partial<Pick<HumanOversightViewModel, typeof OVERSIGHT_MODEL_BRAND>>
  Object.defineProperty(result, OVERSIGHT_MODEL_BRAND, { value: true, enumerable: false })
  return deepFreeze(result) as HumanOversightViewModel
}
