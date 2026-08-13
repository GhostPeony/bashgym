import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement, isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  HumanOversightQueue,
  type HumanOversightQueueProps,
  type HumanReviewResponse
} from './HumanOversightQueue'
import {
  HUMAN_WORK_LIMITS,
  buildHumanOversightModel,
  humanReviewResponseKey,
  parseHumanWorkQueue,
  type HumanOversightViewModel
} from './humanWorkModel'

const NOW = '2026-07-16T20:00:00.000Z'
const WORKSPACE_ID = 'workspace:west'
const CAMPAIGN_ID = 'campaign:alpha'
const WORK_ID = 'hw_0123456789abcdef'
const RECEIPT_ID = 'hrc_0123456789abcdef'

function rawQueue(state: 'pending' | 'claimed' | 'submitted' = 'claimed'): Record<string, unknown> {
  const submitted = state === 'submitted'
  const raw: Record<string, unknown> = {
    schema_version: 'human_work_queue.v1',
    workspace_id: WORKSPACE_ID,
    campaign_id: CAMPAIGN_ID,
    campaign_revision: 3,
    reviewer: { authenticated: true, review_capability: true },
    items: [
      {
        work_id: WORK_ID,
        campaign_revision: 3,
        version: 2,
        state,
        blocking: true,
        rubric: {
          rubric_id: 'rub_01234567',
          version: 1,
          instructions: 'Choose the stronger blinded sample.',
          choices: [
            { choice_id: 'left', label: 'Sample A' },
            { choice_id: 'right', label: 'Sample B' }
          ]
        },
        sample: {
          prompt: 'Assess both samples.',
          left: { label: 'Sample A', display: 'Visible blinded sample A.' },
          right: { label: 'Sample B', display: 'Visible blinded sample B.' }
        },
        lease_expires_at: state === 'claimed' ? '2026-07-16T21:00:00.000Z' : null,
        claimed_by_current_reviewer: state === 'claimed',
        claim_idempotency_key: 'idem_claim_0123456789abcdef',
        submit_idempotency_key: 'idem_submit_0123456789abcdef',
        receipt: submitted
          ? {
              receipt_id: RECEIPT_ID,
              workspace_id: WORKSPACE_ID,
              campaign_id: CAMPAIGN_ID,
              work_id: WORK_ID,
              campaign_revision: 3,
              item_version: 2,
              rubric_version: 1,
              decision: 'prefer_left',
              sealed_at: '2026-07-16T19:59:00.000Z',
              receipt_digest: `sha256:${'a'.repeat(64)}`
            }
          : null
      }
    ],
    promotion: {
      state: submitted ? 'awaiting_human_decision' : 'blocked_by_review',
      version: 5,
      eligible_receipt_id: submitted ? RECEIPT_ID : null,
      idempotency_key: 'idem_promote_0123456789abcdef'
    }
  }
  return raw
}

function model(
  state: 'pending' | 'claimed' | 'submitted' = 'claimed',
  overrides: Record<string, unknown> = {}
): HumanOversightViewModel {
  const queue = parseHumanWorkQueue(overrides.raw || rawQueue(state))
  assert.ok(queue)
  return buildHumanOversightModel({
    queue,
    authority: { workspaceId: WORKSPACE_ID, campaignId: CAMPAIGN_ID },
    freshness: 'live',
    error: null,
    now: NOW,
    ...overrides
  })
}

function response(
  view: HumanOversightViewModel,
  overrides: Partial<HumanReviewResponse> = {}
): HumanReviewResponse {
  return {
    ...view.items[0]!.responseBinding,
    decision: 'prefer_left',
    rationale: 'A follows the requested structure.',
    ...overrides
  }
}

function props(view = model()): HumanOversightQueueProps {
  const answer = response(view)
  return {
    model: view,
    responses: { [humanReviewResponseKey(answer)]: answer },
    onResponseChange: () => {},
    onClaim: () => {},
    onSubmit: () => {},
    onPromotion: () => {}
  }
}

type ElementProps = Record<string, unknown> & { children?: ReactNode }

function expand(
  node: ReactNode,
  found: ReactElement<ElementProps>[] = []
): ReactElement<ElementProps>[] {
  if (Array.isArray(node)) {
    for (const child of node) expand(child, found)
    return found
  }
  if (!isValidElement(node)) return found
  const element = node as ReactElement<ElementProps>
  found.push(element)
  if (typeof element.type === 'function') {
    expand((element.type as (props: ElementProps) => ReactNode)(element.props), found)
  } else {
    expand(element.props.children, found)
  }
  return found
}

function controls(input: HumanOversightQueueProps): ReactElement<ElementProps>[] {
  return expand(HumanOversightQueue(input))
}

function byLabel(
  elements: ReactElement<ElementProps>[],
  label: string
): ReactElement<ElementProps> {
  const element = elements.find((candidate) => candidate.props.children === label)
  assert.ok(element, `expected control ${label}`)
  return element
}

test('renders a compact blinded queue with labels, bounds, and a live status announcement', () => {
  const html = renderToStaticMarkup(createElement(HumanOversightQueue, props()))
  assert.match(html, /Human oversight queue/)
  assert.match(html, /Sample A/)
  assert.match(html, /Sample B/)
  assert.match(html, /for="hw_0123456789abcdef-decision"/)
  assert.match(html, /for="hw_0123456789abcdef-rationale"/)
  assert.match(html, /role="status"/)
  assert.match(html, new RegExp(`maxLength="${HUMAN_WORK_LIMITS.maxRationale}"`, 'i'))
  assert.doesNotMatch(html, /candidate_mapping|private_path|agent-internal/i)
})

test('renders an empty queue as a healthy live state without doctrine or blocker filler', () => {
  const raw = rawQueue('pending')
  raw.items = []
  raw.promotion = {
    state: 'not_required',
    version: 5,
    eligible_receipt_id: null,
    idempotency_key: 'idem_promote_0123456789abcdef'
  }
  const view = model('pending', { raw })
  const html = renderToStaticMarkup(
    createElement(HumanOversightQueue, { ...props(), model: view, responses: {} })
  )

  assert.match(
    html,
    /Nothing needs your review\. Blinded samples will appear here before any promotion\./
  )
  assert.doesNotMatch(
    html,
    /authenticated desktop reviewer|versioned rubric|No current human-review comparison block|Promotion is not blocked/
  )
})

test('C3 keys response state by the complete scope and revision binding', () => {
  const first = model()
  const raw = rawQueue('claimed')
  const entry = (raw.items as Record<string, unknown>[])[0]!
  entry.version = 3
  ;(entry.rubric as Record<string, unknown>).version = 2
  const revised = model('claimed', { raw })
  assert.notEqual(
    humanReviewResponseKey(first.items[0]!.responseBinding),
    humanReviewResponseKey(revised.items[0]!.responseBinding)
  )
})

test('C3 refuses a response retained under the same work ID after item or rubric revision', () => {
  const first = model()
  const old = response(first)
  const raw = rawQueue('claimed')
  const entry = (raw.items as Record<string, unknown>[])[0]!
  entry.version = 3
  ;(entry.rubric as Record<string, unknown>).version = 2
  const revised = model('claimed', { raw })
  let submissions = 0
  const elements = controls({
    ...props(revised),
    responses: { [humanReviewResponseKey(old)]: old },
    onSubmit: () => {
      submissions += 1
    }
  })
  const button = byLabel(elements, 'Submit sealed review')
  assert.equal(button.props.disabled, true)
  ;(button.props.onClick as (() => void) | undefined)?.()
  assert.equal(submissions, 0)
})

test('C3 accepts only decisions offered by the current rubric and a bounded rationale', () => {
  const view = model()
  for (const answer of [
    response(view, { decision: 'no_material_difference' }),
    response(view, { rationale: 'x'.repeat(HUMAN_WORK_LIMITS.maxRationale + 1) })
  ]) {
    const elements = controls({
      ...props(view),
      responses: { [humanReviewResponseKey(answer)]: answer }
    })
    assert.equal(byLabel(elements, 'Submit sealed review').props.disabled, true)
  }
})

test('I6 renders and dispatches from one branded model even when an extra queue-like prop is injected', () => {
  const view = model()
  const injected = { ...props(view), queue: { campaign_revision: 999 } } as HumanOversightQueueProps
  const html = renderToStaticMarkup(createElement(HumanOversightQueue, injected))
  assert.match(html, /Campaign revision 3/)
  assert.doesNotMatch(html, /Campaign revision 999/)
})

test('I7 invokes the claim handler with its fully bound optimistic idempotent request', () => {
  const view = model('pending')
  const captured: unknown[] = []
  const elements = controls({ ...props(view), onClaim: (request) => captured.push(request) })
  const button = byLabel(elements, 'Claim review')
  assert.equal(button.props.disabled, false)
  ;(button.props.onClick as () => void)()
  assert.deepEqual(captured, [view.items[0]!.claim])
})

test('I7 invokes submit with a trimmed revision-bound answer and stable replay key', () => {
  const view = model()
  const answer = response(view, { rationale: '  clear reason  ' })
  const captured: unknown[] = []
  const elements = controls({
    ...props(view),
    responses: { [humanReviewResponseKey(answer)]: answer },
    onSubmit: (request, submitted) => captured.push({ request, submitted })
  })
  const button = byLabel(elements, 'Submit sealed review')
  ;(button.props.onClick as () => void)()
  ;(button.props.onClick as () => void)()
  assert.equal(captured.length, 2)
  assert.deepEqual(captured[0], captured[1])
  assert.equal(
    (captured[0] as { submitted: HumanReviewResponse }).submitted.rationale,
    'clear reason'
  )
})

test('I7 blocks duplicate submit handlers once the controlled mutation is pending', () => {
  const raw = rawQueue('claimed')
  raw.mutation = { action: 'submit', state: 'pending', code: 'in_flight' }
  const view = model('claimed', { raw })
  let calls = 0
  const elements = controls({
    ...props(view),
    onSubmit: () => {
      calls += 1
    }
  })
  const button = byLabel(elements, 'Submit sealed review')
  assert.equal(button.props.disabled, true)
  ;(button.props.onClick as () => void)()
  assert.equal(calls, 0)
})

test('I7 invokes explicit promote and hold handlers with complete bound requests', () => {
  const view = model('submitted')
  const captured: unknown[] = []
  const elements = controls({
    ...props(view),
    responses: {},
    onPromotion: (request) => captured.push(request)
  })
  ;(byLabel(elements, 'Promote with human decision').props.onClick as () => void)()
  ;(byLabel(elements, 'Hold for further review').props.onClick as () => void)()
  assert.deepEqual(captured, [
    { ...view.promotion, decision: 'promote' },
    { ...view.promotion, decision: 'hold' }
  ])
})

test('I7 globally disables handlers while stale, offline, error, or reconciling', () => {
  for (const freshness of ['stale', 'offline', 'error', 'reconciling'] as const) {
    for (const state of ['pending', 'claimed', 'submitted'] as const) {
      const view = model(state, { freshness, error: 'Reconcile first.' })
      let calls = 0
      const elements = controls({
        ...props(view),
        onClaim: () => {
          calls += 1
        },
        onSubmit: () => {
          calls += 1
        },
        onPromotion: () => {
          calls += 1
        }
      })
      for (const label of [
        'Claim review',
        'Submit sealed review',
        'Promote with human decision',
        'Hold for further review'
      ]) {
        const control = elements.find((candidate) => candidate.props.children === label)
        if (!control) continue
        assert.equal(control.props.disabled, true)
        ;(control.props.onClick as (() => void) | undefined)?.()
      }
      assert.equal(calls, 0)
      const html = renderToStaticMarkup(
        createElement(HumanOversightQueue, { ...props(view), model: view })
      )
      assert.match(html, /Visible blinded sample A/)
    }
  }
})

test('I7 suppresses submit after authoritative time advances through lease expiry', () => {
  const expired = model('claimed', { now: '2026-07-16T21:00:00.000Z' })
  const elements = controls(props(expired))
  assert.equal(
    elements.some((candidate) => candidate.props.children === 'Submit sealed review'),
    false
  )
  assert.equal(byLabel(elements, 'Claim review').props.disabled, false)
})

test('renders a non-current sealed receipt as a warning instead of green success', () => {
  const view = model('submitted', { now: '2026-07-16T19:58:00.000Z' })
  const html = renderToStaticMarkup(
    createElement(HumanOversightQueue, { ...props(view), model: view })
  )
  assert.match(html, /Receipt not current/)
  assert.match(html, /text-status-warning/)
  assert.doesNotMatch(html, /Current sealed receipt/)
})

test('I7 exposes current response binding through change handlers', () => {
  const view = model()
  const changes: unknown[] = []
  const elements = controls({
    ...props(view),
    responses: {},
    onResponseChange: (key, answer) => changes.push({ key, answer })
  })
  const select = elements.find((element) => element.type === 'select')
  assert.ok(select)
  ;(select.props.onChange as (event: { target: { value: string } }) => void)({
    target: { value: 'prefer_left' }
  })
  assert.deepEqual(changes, [
    {
      key: humanReviewResponseKey(view.items[0]!.responseBinding),
      answer: { ...view.items[0]!.responseBinding, decision: 'prefer_left', rationale: '' }
    }
  ])
})
