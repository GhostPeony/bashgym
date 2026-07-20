import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { GuidedAutoResearchSetup } from './GuidedAutoResearchSetup'
import { parseGuidedSetupContext } from './guidedSetupModel'

const required = {
  model: 'model.registered',
  data: 'data.registered',
  compute: 'compute.private',
  evaluation: 'evaluation.registered'
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
      required_bindings: required
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
      onSelectedOptionChange: () => {},
      onCampaignIdChange: () => {},
      onTitleChange: () => {},
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
  assert.doesNotMatch(html, /Qwen|cloud fallback|Ponyo|GX10/i)
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
      onSelectedOptionChange: () => {},
      onCampaignIdChange: () => {},
      onTitleChange: () => {},
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
