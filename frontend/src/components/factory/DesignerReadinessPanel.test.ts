import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { DesignerReadinessPanel } from './DesignerReadinessPanel'
import { DataDesignerTab, DesignerDatasetHandoff } from './DataDesignerTab'
import { designerPipelinesResource } from '../../stores/factoryResources'
import type { DesignerPipelinesResponse } from '../../services/api'

const missing: DesignerPipelinesResponse = {
  available: false,
  pipelines: [],
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

test('generated campaign data returns to registration instead of selecting a direct trainer dataset', () => {
  const html = renderToStaticMarkup(
    createElement(DesignerDatasetHandoff, {
      campaignMode: true,
      output: 'generated-export',
      onUseDataset() {}
    })
  )
  assert.match(html, /register the approved dataset before selecting it in setup/)
  assert.match(html, /href="#setup"/)
  assert.doesNotMatch(html, /Use this dataset/)
  const direct = renderToStaticMarkup(
    createElement(DesignerDatasetHandoff, {
      campaignMode: false,
      output: 'generated-export',
      onUseDataset() {}
    })
  )
  assert.match(direct, /Use this dataset/)
  assert.doesNotMatch(direct, /href="#setup"/)
})

test('missing dependencies provide backend-specific install, restart, recheck and setup actions', () => {
  const html = renderToStaticMarkup(
    createElement(DesignerReadinessPanel, {
      data: missing,
      error: null,
      checking: false,
      onRefresh() {}
    })
  )
  assert.match(html, /Data Designer unavailable/)
  assert.match(html, /pandas available/)
  assert.match(html, /pip install &quot;bashgym\[data-designer\]&quot;/)
  assert.match(html, /restart that backend/)
  assert.match(html, /Recheck dependencies/)
  assert.match(html, /href="#setup"/)
  assert.match(html, /NVIDIA_API_KEY is missing/)
  assert.match(html, /2026-09-01T12:00:00\+00:00/)
  assert.match(html, /does not verify provider access, generated data or campaign recipe readiness/)
})

test('failed refresh keeps evidence visibly stale and leaves a retry action', () => {
  const html = renderToStaticMarkup(
    createElement(DesignerReadinessPanel, {
      data: missing,
      error: 'Connection lost',
      checking: false,
      onRefresh() {}
    })
  )
  assert.match(html, /role="alert"/)
  assert.match(html, /Previous evidence may be stale/)
  assert.match(html, /Recheck dependencies/)
})

test('the rendered Data Designer integrates missing-readiness recovery and makes no generation claim', () => {
  const initial = designerPipelinesResource.getInitialState()
  const previous = initial.data
  try {
    initial.data = missing
    const html = renderToStaticMarkup(createElement(DataDesignerTab, { campaignMode: true }))
    assert.match(html, /aria-label="Data Designer readiness"/)
    assert.match(html, /Continue setup with existing data/)
    assert.doesNotMatch(html, />Generate<|>Generate preview</)
  } finally {
    initial.data = previous
  }
})

test('available dependencies require an explicit pipeline choice', () => {
  const initial = designerPipelinesResource.getInitialState()
  const previous = initial.data
  try {
    initial.data = {
      ...missing,
      available: true,
      pipelines: [{ name: 'example', description: 'Example pipeline', columns: [] }]
    }
    const html = renderToStaticMarkup(createElement(DataDesignerTab))
    assert.match(html, /Choose a pipeline/)
    assert.match(html, /Example pipeline/)
    assert.doesNotMatch(html, />Generate<|>Generate preview</)
  } finally {
    initial.data = previous
  }
})
