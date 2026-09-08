/* global URL, localStorage, document, window, console, structuredClone */
import assert from 'node:assert/strict'
import { chromium } from 'playwright-core'
import { mkdir, writeFile } from 'node:fs/promises'
import { controlRoomSnapshot } from '../src/components/autoresearch/controlRoomFixtures.ts'
// API/WebSocket interception keeps this test independent of live research services.
const baseUrl = process.env.STUDIO_TEST_URL || 'http://127.0.0.1:4179'
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(baseUrl).hostname))
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } })
const page = await context.newPage()
const errors = []
const requestedUrls = []
async function assertStudioContrast() {
  const failures = await page.evaluate(() => {
    const color = (value) => value.match(/[\d.]+/g)?.map(Number)
    const blend = (front, back) => {
      const alpha = front[3] ?? 1
      return front.slice(0, 3).map((value, index) => value * alpha + back[index] * (1 - alpha))
    }
    const luminance = (rgb) =>
      rgb
        .slice(0, 3)
        .map((value) => value / 255)
        .map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4))
        .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0)
    const failures = []
    for (const element of document.querySelectorAll('.research-studio *')) {
      const text = [...element.childNodes]
        .filter((node) => node.nodeType === 3)
        .map((node) => node.textContent.trim())
        .join(' ')
      if (!text || !element.getClientRects().length || element.closest(':disabled')) continue
      const ancestors = []
      for (let current = element; current; current = current.parentElement) ancestors.push(current)
      const styles = ancestors.map((node) => window.getComputedStyle(node))
      // Gradients/images and translucent ancestors need separate visual inspection.
      if (
        styles.some(
          (style) =>
            style.visibility !== 'visible' ||
            Number(style.opacity) < 1 ||
            style.backgroundImage !== 'none'
        )
      )
        continue
      const background = styles
        .toReversed()
        .reduce((back, style) => blend(color(style.backgroundColor), back), [255, 255, 255])
      const foreground = blend(color(styles[0].color), background)
      const light = [luminance(foreground), luminance(background)].sort((a, b) => a - b)
      const ratio = (light[1] + 0.05) / (light[0] + 0.05)
      const size = Number.parseFloat(styles[0].fontSize)
      const large = size >= 24 || (size >= 18.66 && Number(styles[0].fontWeight) >= 700)
      const required = large ? 3 : 4.5
      if (ratio + 0.005 < required)
        failures.push({ text: text.slice(0, 100), ratio, required, className: element.className })
    }
    return failures
  })
  assert.deepEqual(failures, [], 'Visible studio text must meet its contrast threshold')
}
page.on('request', (request) => requestedUrls.push(request.url()))
page.on('pageerror', (error) => errors.push(error.message))
let paired = false,
  state = 'empty',
  campaignStatus = 'failed',
  pairingCalls = 0
let submittedProposal = null
const proposalRequests = []
let activeVersion = 8
// These are browser-only fixtures, not backend registration or execution evidence.
const fixtureInstallation = `ins_${'b'.repeat(32)}`
const fixtureTemplate = 'fixture-template'
let setupSession = null
let setupUnavailable = false
let designerUnavailable = false
let designerChecks = 0
const setupContextRequests = []
const setupMutations = []
const bindingKinds = ['model', 'data', 'compute', 'evaluation']
const setupFixture = () => ({
  schema_version: 'guided_setup_context.v1',
  workspace_id: 'workspace-a',
  templates: [
    {
      schema_version: 'guided_setup_template.v1',
      template_id: fixtureTemplate,
      definition_digest: 'a'.repeat(64),
      quality_claim_eligible: true,
      required_bindings: Object.fromEntries(
        bindingKinds.map((kind) => [kind, `${kind}.registered`])
      ),
      experiment_contract: {
        primary_metric: 'exact_task_accuracy',
        metric_direction: 'maximize',
        max_attempts_limit: 6,
        budget_limits: { gpu_hours: 10 },
        protected_metrics: []
      }
    }
  ],
  installations: [
    {
      installation_id: fixtureInstallation,
      ready: false,
      reason_codes: bindingKinds.map((kind) => `${kind}_binding_unavailable`),
      bindings: Object.fromEntries(bindingKinds.map((kind) => [kind, []])),
      truncation: { truncated: false, reason_codes: [], limit_per_kind: 32, kinds: [] }
    }
  ],
  session: setupSession,
  reason_codes: setupSession?.reason_codes || ['setup_session_not_started'],
  truncation: {
    truncated: false,
    reason_codes: [],
    limits: { templates: 32, installations: 32, bindings_per_kind: 32 }
  }
})
const campaign = () => ({
  ...controlRoomSnapshot().campaign,
  schema_version: 'campaign.v1',
  workspace_id: 'workspace-a',
  target_model: {},
  owner_actor_id: 'researcher',
  status: campaignStatus,
  version: campaignStatus === 'active' ? activeVersion : 7,
  created_at: '2026-07-16T00:00:00Z',
  updated_at: '2026-07-16T18:00:00Z'
})
await context.routeWebSocket('**/ws', (socket) =>
  socket.onMessage((raw) => {
    const message = JSON.parse(String(raw))
    if (message.type === 'campaign:subscribe')
      socket.send(
        JSON.stringify({
          type: 'campaign:subscribed',
          payload: { workspace_id: message.payload.ticket }
        })
      )
  })
)
await context.route('**/api/**', async (route) => {
  const request = route.request(),
    path = new URL(request.url()).pathname
  assert.equal(new URL(request.url()).origin, new URL(baseUrl).origin)
  let status = 200,
    data = {}
  if (path === '/api/auth/me') {
    status = paired ? 200 : 401
    data = { id: 1, username: 'researcher' }
  } else if (path === '/api/auth/local/pair') {
    pairingCalls++
    assert.equal(request.method(), 'POST')
    assert.equal(request.headers()['x-requested-with'], 'XMLHttpRequest')
    assert.deepEqual(request.postDataJSON(), { code: 'local-test-code' })
    paired = true
  } else if (path === '/api/campaign-auth/capabilities') {
    data = { workspace_ids: ['workspace-a'] }
  } else if (state === 'offline') {
    await route.abort('connectionrefused')
    return
  } else if (path === '/api/campaigns/setup/context') {
    const params = new URL(request.url()).searchParams
    assert.equal(params.get('workspace_id'), 'workspace-a')
    setupContextRequests.push(params.get('session_id'))
    if (setupUnavailable) {
      await route.abort('connectionrefused')
      return
    }
    data = setupFixture()
  } else if (path === '/api/campaigns/setup/session') {
    const body = request.postDataJSON()
    setupMutations.push(body)
    assert.equal(body.workspace_id, 'workspace-a')
    assert.equal(body.expected_version, setupSession?.version || 0)
    assert.match(body.session_id, /^setupsess_[0-9a-f]{32}$/)
    assert.equal(request.headers()['x-requested-with'], 'XMLHttpRequest')
    assert.match(request.headers()['idempotency-key'], /^idem_/)
    const version = body.expected_version + 1
    assert.ok(version <= 2, 'Fixture must never advance incomplete bindings to READY')
    assert.equal(body.step, version === 1 ? 'template' : 'installation')
    assert.equal(body.selection_id, version === 1 ? fixtureTemplate : fixtureInstallation)
    const receipt = {
      schema_version: 'guided_setup_step_receipt.v1',
      receipt_id: `setupstep_${String(version).repeat(32)}`,
      session_id: body.session_id,
      version,
      step: body.step,
      selection_id: body.selection_id,
      state_digest: 'c'.repeat(64),
      previous_receipt_id: setupSession?.latest_receipt.receipt_id || null,
      previous_receipt_digest: setupSession?.latest_receipt.receipt_digest || null,
      created_at: '2026-09-01T12:00:00Z',
      receipt_digest: `sha256:${String(version).repeat(64)}`
    }
    setupSession = {
      schema_version: 'guided_setup_session.v1',
      workspace_id: 'workspace-a',
      session_id: body.session_id,
      version,
      completed_steps: version === 1 ? ['template'] : ['template', 'installation'],
      selections: {
        template_id: fixtureTemplate,
        installation_id: version === 2 ? fixtureInstallation : null,
        bindings: Object.fromEntries(bindingKinds.map((kind) => [kind, null]))
      },
      ready_for_validation: false,
      reason_codes: [version === 1 ? 'installation_not_selected' : 'model_binding_not_selected'],
      latest_receipt: receipt,
      updated_at: '2026-09-01T12:00:00Z'
    }
    data = { schema_version: 'guided_setup_session_mutation.v1', session: setupSession, receipt }
  } else if (path === '/api/factory/designer/pipelines') {
    designerChecks++
    if (designerUnavailable) {
      status = 503
      data = { detail: 'Fixture readiness check unavailable' }
    } else
      data = {
        pipelines: [],
        available: false,
        readiness: {
          checked_at: '2026-09-01T12:00:00+00:00',
          scope: 'backend_process_imports',
          data_designer_importable: false,
          pandas_importable: true,
          pipeline_builders_importable: false,
          browser_provider: 'nvidia',
          credential_configured: false,
          provider_verified: false,
          generation_verified: false,
          recipe_verified: false
        }
      }
  } else if (path === '/api/factory/designer/models') {
    data = { models: [], provider_models: [], available: false }
  } else if (path.endsWith('/live-ticket')) data = { ticket: request.postDataJSON().workspace_id }
  else if (path.endsWith('/autoresearch/candidates') && request.method() === 'POST') {
    submittedProposal = request.postDataJSON()
    assert.match(request.headers()['idempotency-key'], /^idem_/)
    proposalRequests.push({ body: submittedProposal, key: request.headers()['idempotency-key'] })
    if (proposalRequests.length === 1) {
      await route.abort('connectionrefused')
      return
    }
    if (proposalRequests.length === 2) {
      activeVersion = 9
      status = 409
      data = {
        detail: {
          code: 'campaign_version_conflict',
          message: 'Campaign version changed.',
          current: 9,
          expected: 8
        }
      }
    } else data = { record: { validation: { valid: true, reason_codes: [] } } }
  } else if (path === '/api/campaigns') {
    assert.equal(new URL(request.url()).searchParams.get('workspace_id'), 'workspace-a')
    data = { campaigns: state === 'empty' ? [] : [campaign()], controller: null }
  } else if (path.endsWith('/control-room-snapshot')) {
    const base = controlRoomSnapshot()
    const version = campaignStatus === 'active' ? activeVersion : 7
    data = {
      ...base,
      aggregate_version: version,
      campaign: { ...base.campaign, aggregate_version: version, status: campaignStatus },
      active_work: null
    }
  } else if (path === '/api/campaigns/campaign-1') data = campaign()
  else if (path.endsWith('/proposals'))
    data = {
      proposals: [
        {
          proposal: {
            proposal_id: 'proposal-1',
            hypothesis: 'Better hard negatives improve retrieval precision.',
            primary_variable: 'Hard negative sampling',
            expected_outcome: 'Higher precision on the fixed suite.',
            falsification_criterion: 'Precision fails to improve.',
            status: 'validated'
          },
          validation: { valid: true, reason_codes: [] },
          updated_at: '2026-07-16'
        }
      ]
    }
  else if (path.endsWith('/attempts'))
    data = {
      attempts: [
        {
          attempt_id: 'attempt-1',
          attempt_number: 1,
          stage: 'evaluation',
          status: campaignStatus === 'failed' ? 'failed' : 'completed',
          updated_at: '2026-07-16'
        }
      ]
    }
  else if (path.endsWith('/ledger')) data = { projects: [], autoresearch_outcomes: [] }
  else if (path.endsWith('/evidence')) data = {}
  else if (path.endsWith('/events')) data = { items: [], next_cursor: 42 }
  else if (/\/(studies|comparisons|events|artifacts)$/.test(path))
    data = { [path.split('/').at(-1)]: [], next_cursor: null, has_more: false }
  else {
    status = 503
    data = { detail: 'Fixture service unavailable' }
  }
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(data) })
})
try {
  await mkdir('test-results/studio', { recursive: true })
  await page.goto(baseUrl)
  await page.getByRole('heading', { name: 'Connect your studio' }).waitFor()
  assert.ok(!requestedUrls.some((url) => /fonts\.(googleapis|gstatic)\.com/.test(url)))
  assert.ok(
    !requestedUrls.some((url) => /(?:ResearchStudio|websocket|DesktopShell)-.*\.js/.test(url))
  )
  await page.screenshot({ path: 'test-results/studio/pairing.png', fullPage: true })
  await page.getByLabel('Pairing code').fill('local-test-code')
  await page.getByRole('button', { name: 'Open studio' }).click()
  await page.getByRole('link', { name: 'Prepare an experiment' }).waitFor()
  assert.equal(new URL(page.url()).searchParams.get('workspace_id'), 'workspace-a')
  const originalHash = new URL(page.url()).hash
  await page.keyboard.press('Tab')
  await page.locator('.studio-skip').focus()
  assert.equal(
    await page
      .locator('.studio-skip')
      .evaluate((element) => window.getComputedStyle(element).outlineWidth),
    '2px'
  )
  await page.keyboard.press('Enter')
  assert.equal(await page.evaluate(() => document.activeElement?.id), 'studio-main')
  assert.equal(new URL(page.url()).hash, originalHash)
  assert.equal(pairingCalls, 1)
  assert.equal(
    await page.evaluate(() =>
      Object.values(localStorage).some((value) => value.includes('local-test-code'))
    ),
    false
  )
  await page.getByRole('link', { name: 'Setup', exact: true }).click()
  await page.getByRole('heading', { name: 'Guided setup' }).waitFor()
  await page.getByRole('combobox', { name: /^Registered template/ }).selectOption(fixtureTemplate)
  await page.getByRole('button', { name: 'Save choice', exact: true }).click()
  await page
    .getByRole('combobox', { name: /^Registered installation/ })
    .selectOption(fixtureInstallation)
  await page.getByText(/Saving this installation does not verify its recipe inputs/).waitFor()
  assert.equal(
    await page.getByRole('button', { name: 'Save choice', exact: true }).isEnabled(),
    true
  )
  await page.getByRole('button', { name: 'Save choice', exact: true }).click()
  await page.getByText('2 of 6 choices sealed', { exact: true }).waitFor()
  assert.equal(
    await page.getByRole('button', { name: 'Save choice', exact: true }).isDisabled(),
    true
  )
  assert.equal(
    await page.getByRole('button', { name: 'Create and review Start', exact: true }).count(),
    0
  )
  assert.equal(setupSession.ready_for_validation, false)
  const savedDraft = structuredClone(setupSession)
  await Promise.all([
    page.waitForResponse(
      (response) => new URL(response.url()).pathname === '/api/campaigns/setup/context'
    ),
    page.getByRole('button', { name: 'Refresh registrations', exact: true }).click()
  ])
  await page
    .getByRole('button', { name: 'Refresh registrations', exact: true })
    .waitFor({ state: 'visible' })
  await page.getByText('2 of 6 choices sealed', { exact: true }).waitFor()
  assert.equal(setupContextRequests.at(-1), savedDraft.session_id)
  assert.deepEqual(setupSession, savedDraft)
  assert.equal(setupMutations.length, 2)
  await page.screenshot({
    path: 'test-results/studio/setup-incomplete-fixture.png',
    fullPage: true
  })
  await assertStudioContrast()
  setupUnavailable = true
  await page.getByRole('button', { name: 'Refresh registrations', exact: true }).click()
  await page.getByText('Live authority is offline', { exact: true }).waitFor()
  assert.equal(
    await page.getByRole('button', { name: 'Save choice', exact: true }).isDisabled(),
    true
  )
  setupUnavailable = false
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await page.getByText('2 of 6 choices sealed', { exact: true }).waitFor()
  assert.deepEqual(setupSession, savedDraft)
  await page.getByRole('link', { name: 'Resources', exact: true }).click()
  await page.getByRole('button', { name: 'Data Designer', exact: true }).click()
  await page.getByRole('heading', { name: 'Optional Data Designer', exact: true }).waitFor()
  await page.getByText(/Backend process imports: Data Designer unavailable/).waitFor()
  assert.equal(await page.getByText('Data Creator', { exact: true }).count(), 0)
  assert.equal(await page.getByRole('button', { name: 'Generate', exact: true }).count(), 0)
  await page.getByText(/NVIDIA_API_KEY is missing/).waitFor()
  await page
    .getByText(/does not verify provider access, generated data or campaign recipe readiness/)
    .waitFor()
  await page.screenshot({
    path: 'test-results/studio/designer-missing-fixture.png',
    fullPage: true
  })
  await assertStudioContrast()
  designerUnavailable = true
  await page.getByRole('button', { name: 'Recheck dependencies', exact: true }).click()
  await page.getByText(/Previous evidence may be stale/).waitFor()
  await page.getByText(/Backend process imports: Data Designer unavailable/).waitFor()
  designerUnavailable = false
  await page.getByRole('button', { name: 'Recheck dependencies', exact: true }).click()
  await page.getByText(/Previous evidence may be stale/).waitFor({ state: 'hidden' })
  assert.equal(designerChecks, 3)
  await page.getByRole('link', { name: 'Continue setup with existing data', exact: true }).click()
  await page.getByRole('heading', { name: 'Guided setup', exact: true }).waitFor()
  await page.getByText('2 of 6 choices sealed', { exact: true }).waitFor()
  assert.deepEqual(setupSession, savedDraft)
  assert.equal(setupContextRequests.at(-1), savedDraft.session_id)
  assert.equal(setupMutations.length, 2)
  state = 'campaign'
  await page.goto(
    `${baseUrl}/?view=training&tab=autoresearch&workspace_id=workspace-a&campaign_id=campaign-1#experiments`
  )
  await page.getByRole('heading', { name: 'Hypothesis', exact: true }).waitFor()
  await page
    .getByText('Better hard negatives improve retrieval precision.', { exact: true })
    .waitFor()
  await page.screenshot({ path: 'test-results/studio/experiments.png', fullPage: true })
  campaignStatus = 'active'
  await page.reload()
  await page.getByRole('heading', { name: 'Run history', exact: true }).waitFor()
  await page.getByText('Propose an experiment', { exact: true }).click()
  await page.getByLabel('Proposal role', { exact: false }).selectOption('candidate')
  await page.getByLabel('Proposal ID', { exact: true }).fill('candidate-test')
  await page.getByLabel('Study family', { exact: true }).fill('retrieval')
  await page.getByLabel('Estimated cost', { exact: true }).fill('1')
  await page.getByLabel('Parent proposal ID', { exact: true }).fill('proposal-1')
  await page.getByLabel('Hypothesis', { exact: true }).fill('Hard negatives improve precision.')
  await page.getByLabel('One variable to change', { exact: true }).fill('negative_sampling')
  await page.getByLabel('Expected outcome', { exact: true }).fill('Precision increases.')
  await page.getByLabel('Falsification criterion', { exact: true }).fill('Precision decreases.')
  await page.getByLabel('Rationale', { exact: true }).fill('Evaluate a controlled data change.')
  await page.getByLabel('Recipes and stage plan (JSON)', { exact: false }).fill(
    JSON.stringify({
      dataset_recipe: {},
      training_recipe: {},
      evaluation_recipe: {},
      stage_plan: {
        items: [
          {
            stage: 'development_evaluation',
            disposition: 'required',
            reason: 'Compare precision.'
          }
        ]
      }
    })
  )
  await page.getByRole('button', { name: 'Submit proposal for validation' }).click()
  await page.getByText(/The research service is disconnected/).waitFor()
  await page.getByRole('button', { name: 'Submit proposal for validation' }).click()
  await page.getByText(/The campaign changed. Your draft is preserved/).waitFor()
  assert.deepEqual(proposalRequests[1], proposalRequests[0])
  assert.equal(
    await page.getByLabel('Hypothesis', { exact: true }).inputValue(),
    'Hard negatives improve precision.'
  )
  await page.getByRole('button', { name: 'Submit proposal for validation' }).click()
  await page.getByText(/Proposal recorded and validated/).waitFor()
  assert.equal(submittedProposal.expected_version, 9)
  assert.notEqual(proposalRequests[2].key, proposalRequests[1].key)
  assert.equal(submittedProposal.workspace_id, 'workspace-a')
  assert.equal(submittedProposal.parent_proposal_id, 'proposal-1')
  await page.getByRole('link', { name: 'Home', exact: true }).click()
  await page
    .getByRole('heading', {
      name: 'Improve retrieval quality without regressing latency.',
      exact: true
    })
    .waitFor()
  await page.screenshot({ path: 'test-results/studio/home.png', fullPage: true })
  await assertStudioContrast()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('link', { name: 'Experiments', exact: true }).click()
  await page.getByRole('heading', { name: 'Hypothesis', exact: true }).waitFor()
  await page.screenshot({ path: 'test-results/studio/mobile.png', fullPage: true })
  await assertStudioContrast()
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    true
  )
  state = 'offline'
  await page.goto(`${baseUrl}/#home`)
  await page.getByRole('button', { name: 'Retry connection', exact: true }).waitFor()
  assert.deepEqual(errors, [])
  assert.equal(setupMutations.length, 2)
  assert.ok(!requestedUrls.some((url) => /\/setup\/(doctor|validate|create)(?:\?|$)/.test(url)))
  assert.ok(!requestedUrls.some((url) => /\/designer\/(preview|create)(?:\?|$)/.test(url)))
  await writeFile(
    'test-results/studio/browser-fixture-result.json',
    JSON.stringify(
      {
        status: 'passed',
        evidence_scope: 'mocked_api_and_websocket_browser_flows',
        incomplete_installation_saved: true,
        draft_preserved_after_refresh_and_reconnect: true,
        direct_designer_missing_readiness: true,
        readiness_failed_refresh_recovered: true,
        existing_data_link_resumes_setup: true,
        generation_requested: false,
        setup_validation_or_creation_requested: false,
        page_errors: errors
      },
      null,
      2
    )
  )
  console.log(
    'PASS (API/WebSocket browser fixtures): pairing, empty, incomplete installation selection, saved draft refresh/recovery, direct Data Designer missing readiness and failed recheck, existing-data setup return, proposal submission and validation, failed, resumed, navigation, disconnected, mobile; no page errors. No real backend readiness or generation verified.'
  )
} catch (error) {
  await page.screenshot({ path: 'test-results/studio/failure.png', fullPage: true })
  await writeFile(
    'test-results/studio/failure-details.json',
    JSON.stringify(
      {
        fixture: true,
        error: String(error),
        pageErrors: errors,
        setupContextRequests,
        setupMutations,
        designerChecks
      },
      null,
      2
    )
  )
  throw error
} finally {
  await browser.close()
}
