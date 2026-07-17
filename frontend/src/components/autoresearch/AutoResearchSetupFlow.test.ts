import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AutoResearchSetupFlow } from './AutoResearchSetupFlow'
import { buildSetupFlowModel, type SetupFlowInput } from './setupFlowModel'

function model(overrides: Partial<SetupFlowInput> = {}) {
  return buildSetupFlowModel({
    receipts: [{ stepId: 'source', status: 'complete', receiptId: 'receipt-source', summary: 'Source binding saved.' }],
    blockers: [{ stepId: 'compute', code: 'ssh_registration_required', summary: 'Register a private SSH compute target.' }],
    doctor: {
      observedAt: '2026-07-16T19:00:00Z',
      isLatestServerDoctor: true,
      launchReady: false,
      controllerIdentityVerified: true,
      computeIdentityVerified: false,
      state: 'offline',
    },
    ...overrides,
  })
}

test('keeps the operational setup shell visible offline with receipts and remediation', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model(),
    connectionState: 'offline',
    mutationsEnabled: false,
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /Guided setup/)
  assert.match(html, /Offline/)
  assert.match(html, /Source binding saved/)
  assert.match(html, /Register a private SSH compute target/)
  assert.match(html, /Private SSH compute/)
  assert.match(html, /role="status"/)
  assert.match(html, /aria-label="Refresh server doctor"/)
  assert.doesNotMatch(html, /Qwen|automatic fallback/i)
})

test('renders Start disabled until the server-owned readiness gate is satisfied', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model(),
    connectionState: 'offline',
    mutationsEnabled: false,
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /The latest server doctor has not declared launch readiness/)
  assert.match(html, /<button[^>]*disabled=""[^>]*>Start<\/button>/)
})

test('fails closed when a parent claims mutations are enabled without live authority', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model({ receipts: [], blockers: [] }),
    connectionState: 'offline',
    mutationsEnabled: true,
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /Live authority is unavailable/)
  assert.match(html, /<button[^>]*disabled=""[^>]*>Start<\/button>/)
  assert.match(html, /<button[^>]*disabled=""[^>]*aria-label="Continue: Source"/)
})

test('renders a compact controlled logical-binding editor with no model, cloud, or secret defaults', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model({
      receipts: [{ stepId: 'source', status: 'complete', receiptId: 'receipt-source', summary: 'Source binding saved.' }],
      blockers: [],
      drafts: { model_ref: '', model_revision: '' },
    }),
    connectionState: 'live',
    mutationsEnabled: true,
    onDraftChange: () => {},
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /aria-label="Model identity &amp; revision editor"/)
  assert.match(html, /name="model_ref"/)
  assert.match(html, /name="model_revision"/)
  assert.match(html, /No model is selected or downloaded automatically/)
  assert.match(html, /logical binding IDs only/)
  assert.doesNotMatch(html, /Qwen|cloud fallback|name="host"|name="user"|name="key"|name="path"/i)
})

test('shows safe receipt identity and a disabled resume path when no durable setup API is mounted', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model({
      receipts: [
        { stepId: 'source', status: 'complete', receiptId: 'receipt-source', summary: 'Source binding saved.' },
        { stepId: 'model', status: 'partial', receiptId: 'receipt-model-2', summary: 'Revision validation can resume.', retryable: true },
      ],
      blockers: [],
    }),
    connectionState: 'live',
    mutationsEnabled: true,
    onDraftChange: () => {},
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /receipt-model-2/)
  assert.match(html, /aria-label="Resume: Model identity &amp; revision"/)
  assert.match(html, /<button[^>]*disabled=""[^>]*aria-label="Resume: Model identity &amp; revision"/)
  assert.match(html, /bashgym campaign setup-autoresearch/)
})

test('keeps doctor refresh disabled until a typed authoritative refresh contract is mounted', () => {
  const html = renderToStaticMarkup(createElement(AutoResearchSetupFlow, {
    model: model(),
    connectionState: 'live',
    mutationsEnabled: true,
    onStepAction: () => {},
    onStart: () => {},
    onRefreshDoctor: () => {},
  }))

  assert.match(html, /<button[^>]*disabled=""[^>]*aria-label="Refresh server doctor"/)
  assert.match(html, /Run bashgym campaign doctor in the workspace terminal/)
})
