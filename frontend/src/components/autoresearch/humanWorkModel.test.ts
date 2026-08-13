import assert from 'node:assert/strict'
import test from 'node:test'

import {
  HUMAN_WORK_LIMITS,
  buildHumanOversightModel,
  parseHumanWorkQueue,
  type HumanWorkQueuePublicV1,
  type ParsedHumanWorkQueuePublicV1
} from './humanWorkModel'

const NOW = '2026-07-16T20:00:00.000Z'
const WORKSPACE_ID = 'workspace:west'
const CAMPAIGN_ID = 'campaign:alpha'
const WORK_ID = 'hw_0123456789abcdef'
const RUBRIC_ID = 'rub_01234567'
const RECEIPT_ID = 'hrc_0123456789abcdef'
const CLAIM_KEY = 'idem_claim_0123456789abcdef'
const SUBMIT_KEY = 'idem_submit_0123456789abcdef'
const PROMOTION_KEY = 'idem_promote_0123456789abcdef'
const DIGEST = `sha256:${'a'.repeat(64)}`

function rawQueue(): Record<string, unknown> {
  return {
    schema_version: 'human_work_queue.v1',
    workspace_id: WORKSPACE_ID,
    campaign_id: CAMPAIGN_ID,
    campaign_revision: 7,
    reviewer: { authenticated: true, review_capability: true },
    items: [
      {
        work_id: WORK_ID,
        campaign_revision: 7,
        version: 3,
        state: 'pending',
        blocking: true,
        rubric: {
          rubric_id: RUBRIC_ID,
          version: 2,
          instructions: 'Choose the response that better follows the rubric.',
          choices: [
            { choice_id: 'left', label: 'Sample A is stronger' },
            { choice_id: 'right', label: 'Sample B is stronger' },
            { choice_id: 'tie', label: 'No material difference' }
          ]
        },
        sample: {
          prompt: 'Assess the blinded samples against the rubric.',
          left: { label: 'Sample A', display: 'Blinded response A.' },
          right: { label: 'Sample B', display: 'Blinded response B.' }
        },
        lease_expires_at: null,
        claimed_by_current_reviewer: false,
        claim_idempotency_key: CLAIM_KEY,
        submit_idempotency_key: SUBMIT_KEY,
        receipt: null
      }
    ],
    promotion: {
      state: 'blocked_by_review',
      version: 4,
      eligible_receipt_id: null,
      idempotency_key: PROMOTION_KEY
    }
  }
}

function item(queue: Record<string, unknown>): Record<string, unknown> {
  return (queue.items as Record<string, unknown>[])[0]!
}

function claimedQueue(lease = '2026-07-16T20:30:00.000Z'): Record<string, unknown> {
  const raw = rawQueue()
  Object.assign(item(raw), {
    state: 'claimed',
    lease_expires_at: lease,
    claimed_by_current_reviewer: true
  })
  return raw
}

function receipt(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    receipt_id: RECEIPT_ID,
    workspace_id: WORKSPACE_ID,
    campaign_id: CAMPAIGN_ID,
    work_id: WORK_ID,
    campaign_revision: 7,
    item_version: 3,
    rubric_version: 2,
    decision: 'prefer_left',
    sealed_at: '2026-07-16T19:59:00.000Z',
    receipt_digest: DIGEST,
    ...overrides
  }
}

function submittedQueue(receiptOverrides: Record<string, unknown> = {}): Record<string, unknown> {
  const raw = rawQueue()
  Object.assign(item(raw), {
    state: 'submitted',
    lease_expires_at: null,
    claimed_by_current_reviewer: false,
    receipt: receipt(receiptOverrides)
  })
  Object.assign(raw.promotion as Record<string, unknown>, {
    state: 'awaiting_human_decision',
    eligible_receipt_id: RECEIPT_ID
  })
  return raw
}

function parsed(raw: unknown = rawQueue()): ParsedHumanWorkQueuePublicV1 {
  const result = parseHumanWorkQueue(raw)
  assert.ok(result, 'fixture must satisfy the public queue parser')
  return result
}

function model(raw: unknown = rawQueue(), overrides: Record<string, unknown> = {}) {
  return buildHumanOversightModel({
    queue: parsed(raw),
    authority: { workspaceId: WORKSPACE_ID, campaignId: CAMPAIGN_ID },
    freshness: 'live',
    error: null,
    now: NOW,
    ...overrides
  })
}

test('C1 binds a canonical workspace and campaign to every claim request', () => {
  const view = model()
  assert.equal(view.scopeValid, true)
  assert.deepEqual(view.items[0]?.claim, {
    workspaceId: WORKSPACE_ID,
    campaignId: CAMPAIGN_ID,
    workId: WORK_ID,
    expectedCampaignRevision: 7,
    expectedVersion: 3,
    expectedState: 'pending',
    idempotencyKey: CLAIM_KEY
  })
})

test('C1 suppresses cross-workspace and cross-campaign replay against authoritative selection', () => {
  for (const authority of [
    { workspaceId: 'workspace:east', campaignId: CAMPAIGN_ID },
    { workspaceId: WORKSPACE_ID, campaignId: 'campaign:beta' }
  ]) {
    const view = model(rawQueue(), { authority })
    assert.equal(view.scopeValid, false)
    assert.equal(view.mutationsEnabled, false)
    assert.equal(view.items[0]?.claim, null)
  }
})

test('C1 distinguishes identical campaign IDs in different workspaces', () => {
  const west = model()
  const eastRaw = rawQueue()
  eastRaw.workspace_id = 'workspace:east'
  const east = model(eastRaw, {
    authority: { workspaceId: 'workspace:east', campaignId: CAMPAIGN_ID }
  })
  assert.notEqual(west.items[0]?.claim?.workspaceId, east.items[0]?.claim?.workspaceId)
})

test('C1 counts public string bounds by Unicode code point like the backend', () => {
  const atBoundary = rawQueue()
  ;((item(atBoundary).sample as Record<string, unknown>).left as Record<string, unknown>).label =
    '🌸'.repeat(HUMAN_WORK_LIMITS.maxLabel)
  assert.ok(parseHumanWorkQueue(atBoundary))

  const beyondBoundary = rawQueue()
  ;(
    (item(beyondBoundary).sample as Record<string, unknown>).left as Record<string, unknown>
  ).label = '🌸'.repeat(HUMAN_WORK_LIMITS.maxLabel + 1)
  assert.equal(parseHumanWorkQueue(beyondBoundary), null)
})

test('C2 rejects normalized timestamps and impossible lease lifecycle combinations', () => {
  assert.equal(parseHumanWorkQueue(claimedQueue('2026-07-16 20:30:00Z')), null)
  assert.equal(parseHumanWorkQueue(claimedQueue('2026-07-16T20:30:00Z')), null)
  assert.equal(parseHumanWorkQueue(claimedQueue(null as unknown as string)), null)

  const pendingWithLease = rawQueue()
  Object.assign(item(pendingWithLease), { lease_expires_at: '2026-07-16T20:30:00.000Z' })
  assert.equal(parseHumanWorkQueue(pendingWithLease), null)

  const pendingClaimant = rawQueue()
  Object.assign(item(pendingClaimant), { claimed_by_current_reviewer: true })
  assert.equal(parseHumanWorkQueue(pendingClaimant), null)
})

test('C2 invalid authoritative time globally suppresses mutation authority', () => {
  const view = model(rawQueue(), { now: 'not-a-date' })
  assert.equal(view.timeValid, false)
  assert.equal(view.mutationsEnabled, false)
  assert.equal(view.items[0]?.claim, null)
  assert.match(view.remediation, /time authority/i)
})

test('C2 expires a lease exactly at the boundary and permits only a future lease to submit', () => {
  const boundary = model(claimedQueue(NOW))
  assert.equal(boundary.items[0]?.effectiveState, 'expired')
  assert.equal(boundary.items[0]?.submit, null)
  assert.equal(boundary.items[0]?.claim?.expectedState, 'expired')

  const future = model(claimedQueue())
  assert.equal(future.items[0]?.effectiveState, 'claimed')
  assert.equal(future.items[0]?.submit?.expectedState, 'claimed')
})

test('C2 suppresses an explicitly expired lifecycle whose lease is still in the future', () => {
  const futureExpired = rawQueue()
  Object.assign(item(futureExpired), {
    state: 'expired',
    lease_expires_at: '2026-07-16T20:00:00.001Z',
    claimed_by_current_reviewer: false
  })
  const invalid = model(futureExpired)
  assert.equal(invalid.mutationsEnabled, false)
  assert.equal(invalid.items[0]?.claim, null)
  assert.match(invalid.remediation, /lease lifecycle/i)

  const pastExpired = rawQueue()
  Object.assign(item(pastExpired), {
    state: 'expired',
    lease_expires_at: '2026-07-16T19:59:59.999Z',
    claimed_by_current_reviewer: false
  })
  assert.equal(model(pastExpired).items[0]?.claim?.expectedState, 'expired')
})

test('C4 rejects receipts not exactly bound to their submitted parent', () => {
  for (const override of [
    { work_id: 'hw_fedcba9876543210' },
    { workspace_id: 'workspace:east' },
    { campaign_id: 'campaign:beta' },
    { campaign_revision: 6 },
    { item_version: 4 },
    { rubric_version: 3 },
    { decision: 'not_offered' },
    { sealed_at: null }
  ])
    assert.equal(parseHumanWorkQueue(submittedQueue(override)), null)

  const wrongState = rawQueue()
  item(wrongState).receipt = receipt()
  assert.equal(parseHumanWorkQueue(wrongState), null)

  const missing = submittedQueue()
  item(missing).receipt = null
  assert.equal(parseHumanWorkQueue(missing), null)
})

test('C4 rejects duplicate receipt identities even when each receipt matches its parent', () => {
  const raw = submittedQueue()
  const second = structuredClone(item(raw))
  second.work_id = 'hw_fedcba9876543210'
  second.claim_idempotency_key = 'idem_claim_fedcba9876543210'
  second.submit_idempotency_key = 'idem_submit_fedcba9876543210'
  const secondReceipt = second.receipt as Record<string, unknown>
  secondReceipt.work_id = second.work_id
  raw.items = [item(raw), second]
  assert.equal(parseHumanWorkQueue(raw), null)
})

test('C4 keeps comparison and promotion blocked for a future-sealed receipt', () => {
  const view = model(submittedQueue({ sealed_at: '2026-07-16T20:00:00.001Z' }))
  assert.equal(view.items[0]?.receiptCurrent, false)
  assert.equal(view.blocksComparison, true)
  assert.equal(view.blocksPromotion, true)
  assert.equal(view.promotion, null)
  assert.equal(view.mutationsEnabled, false)
})

test('C4 exposes promotion only for one current valid sealed eligible receipt', () => {
  const view = model(submittedQueue())
  assert.equal(view.items[0]?.receiptCurrent, true)
  assert.equal(view.blocksComparison, false)
  assert.deepEqual(view.promotion, {
    workspaceId: WORKSPACE_ID,
    campaignId: CAMPAIGN_ID,
    receiptId: RECEIPT_ID,
    workId: WORK_ID,
    expectedCampaignRevision: 7,
    expectedItemVersion: 3,
    expectedRubricVersion: 2,
    expectedPromotionVersion: 4,
    expectedPromotionState: 'awaiting_human_decision',
    idempotencyKey: PROMOTION_KEY
  })

  const missing = submittedQueue()
  ;(missing.promotion as Record<string, unknown>).eligible_receipt_id = 'hrc_fedcba9876543210'
  assert.equal(parseHumanWorkQueue(missing), null)
})

test('I1 globally suppresses claim, submit, and promotion while a mutation is unreconciled', () => {
  for (const state of ['pending', 'conflict', 'error']) {
    for (const raw of [rawQueue(), claimedQueue(), submittedQueue()]) {
      raw.mutation = {
        action: 'submit',
        state,
        code: state === 'pending' ? 'in_flight' : 'version_conflict'
      }
      const view = model(raw)
      assert.equal(view.mutationsEnabled, false)
      assert.equal(
        view.items.every((entry) => entry.claim === null && entry.submit === null),
        true
      )
      assert.equal(view.promotion, null)
    }
  }
})

test('I1 globally suppresses every action for a mixed-revision queue', () => {
  const raw = claimedQueue()
  item(raw).campaign_revision = 6
  const view = model(raw)
  assert.equal(view.mutationsEnabled, false)
  assert.equal(view.items[0]?.submit, null)
})

test('I1 keeps a blocking submitted receipt blocked after campaign revision invalidation', () => {
  const raw = submittedQueue()
  item(raw).campaign_revision = 6
  ;(item(raw).receipt as Record<string, unknown>).campaign_revision = 6
  Object.assign(raw.promotion as Record<string, unknown>, {
    state: 'blocked_by_review',
    eligible_receipt_id: null
  })
  const view = model(raw)
  assert.equal(view.mutationsEnabled, false)
  assert.equal(view.blocksComparison, true)
  assert.equal(view.blocksPromotion, true)
})

test('I2 enforces collection and public text bounds', () => {
  const tooMany = rawQueue()
  tooMany.items = Array.from({ length: HUMAN_WORK_LIMITS.maxItems + 1 }, (_, index) => ({
    ...structuredClone(item(rawQueue())),
    work_id: `hw_${String(index).padStart(16, '0')}`,
    claim_idempotency_key: `idem_claim_${String(index).padStart(16, '0')}`,
    submit_idempotency_key: `idem_submit_${String(index).padStart(16, '0')}`
  }))
  assert.equal(parseHumanWorkQueue(tooMany), null)

  const longText = rawQueue()
  ;((item(longText).sample as Record<string, unknown>).left as Record<string, unknown>).display =
    'x'.repeat(HUMAN_WORK_LIMITS.maxSampleDisplay + 1)
  assert.equal(parseHumanWorkQueue(longText), null)
})

test('I2 rejects duplicate work and rubric choice IDs', () => {
  const duplicateWork = rawQueue()
  duplicateWork.items = [item(duplicateWork), { ...structuredClone(item(duplicateWork)) }]
  assert.equal(parseHumanWorkQueue(duplicateWork), null)

  const duplicateChoice = rawQueue()
  const rubric = item(duplicateChoice).rubric as Record<string, unknown>
  rubric.choices = [
    { choice_id: 'left', label: 'First' },
    { choice_id: 'left', label: 'Second' }
  ]
  assert.equal(parseHumanWorkQueue(duplicateChoice), null)
})

test('I3 returns a deeply frozen parsed public projection', () => {
  const queue = parsed(submittedQueue({ sealed_at: '2026-07-16T19:59:00.000Z' }))
  const objects: object[] = [
    queue,
    queue.reviewer,
    queue.items,
    queue.items[0]!,
    queue.items[0]!.rubric,
    queue.items[0]!.rubric.choices,
    queue.items[0]!.rubric.choices[0]!,
    queue.items[0]!.sample,
    queue.items[0]!.sample.left,
    queue.items[0]!.receipt!,
    queue.promotion
  ]
  assert.equal(objects.every(Object.isFrozen), true)
  assert.throws(() => {
    ;(queue.items[0] as unknown as { version: number }).version = 99
  })
  assert.throws(() => {
    ;(queue.items[0]!.sample.left as unknown as { display: string }).display = 'changed'
  })
})

test('I3 freezes the optional mutation projection', () => {
  const raw = rawQueue()
  raw.mutation = { action: 'claim', state: 'pending', code: 'in_flight' }
  assert.equal(Object.isFrozen(parsed(raw).mutation!), true)
})

test('I4 accepts durable colon IDs and rejects malformed type-specific references', () => {
  assert.ok(parseHumanWorkQueue(rawQueue()))
  for (const mutate of [
    (raw: Record<string, unknown>) => {
      raw.workspace_id = '_workspace'
    },
    (raw: Record<string, unknown>) => {
      item(raw).work_id = 'campaign:alpha'
    },
    (raw: Record<string, unknown>) => {
      item(raw).claim_idempotency_key = 'short'
    },
    (raw: Record<string, unknown>) => {
      ;(item(raw).rubric as Record<string, unknown>).rubric_id = 'not-rubric'
    }
  ]) {
    const raw = rawQueue()
    mutate(raw)
    assert.equal(parseHumanWorkQueue(raw), null)
  }

  const badDigest = submittedQueue({ receipt_digest: 'a'.repeat(64) })
  assert.equal(parseHumanWorkQueue(badDigest), null)
})

test('I5 binds optimistic state and stable idempotency to all actions', () => {
  const claimA = model().items[0]?.claim
  const claimB = model().items[0]?.claim
  assert.deepEqual(claimA, claimB)

  const submitA = model(claimedQueue()).items[0]?.submit
  const submitB = model(claimedQueue()).items[0]?.submit
  assert.deepEqual(submitA, submitB)
  assert.equal(submitA?.idempotencyKey, SUBMIT_KEY)
  assert.equal(submitA?.expectedVersion, 3)
  assert.equal(submitA?.expectedRubricVersion, 2)

  const promotionA = model(submittedQueue()).promotion
  const promotionB = model(submittedQueue()).promotion
  assert.deepEqual(promotionA, promotionB)
  assert.equal(promotionA?.idempotencyKey, PROMOTION_KEY)
  assert.equal('correlationId' in (promotionA || {}), false)
})

test('I5 rejects conflicting reuse of one server idempotency key', () => {
  const raw = rawQueue()
  item(raw).submit_idempotency_key = CLAIM_KEY
  assert.equal(parseHumanWorkQueue(raw), null)

  const promotionConflict = rawQueue()
  ;(promotionConflict.promotion as Record<string, unknown>).idempotency_key = CLAIM_KEY
  assert.equal(parseHumanWorkQueue(promotionConflict), null)
})

test('preserves replaced work as visible history without mutation authority', () => {
  const raw = rawQueue()
  item(raw).state = 'replaced'
  const view = model(raw)
  assert.equal(view.items[0]?.effectiveState, 'replaced')
  assert.equal(view.items[0]?.claim, null)
  assert.equal(view.items[0]?.submit, null)
})

test('still rejects extra candidate mapping and private path fields', () => {
  assert.equal(parseHumanWorkQueue({ ...rawQueue(), candidate_mapping: {} }), null)
  const raw = rawQueue()
  item(raw).private_path = 'C:\\private\\result.json'
  assert.equal(parseHumanWorkQueue(raw), null)
})

// Compile-time public API check; parsed queues are intentionally readonly and branded.
const _queueShape: HumanWorkQueuePublicV1['schema_version'] = 'human_work_queue.v1'
void _queueShape
