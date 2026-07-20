import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { DataNodeShell } from './DataNodeShell'
import { NodeSurfaceProvider } from './NodeSurfaceProvider'

test('renders a full-size grid summary without React Flow handles or close control', () => {
  const summary = createElement(DataNodeShell, {
    panelId: 'training-1',
    title: 'Training Run',
    selected: true,
    children: createElement('div', null, 'loss 0.4992')
  })
  const html = renderToStaticMarkup(
    createElement(NodeSurfaceProvider, { surface: 'grid', children: summary })
  )

  assert.match(html, /data-node-surface="grid"/)
  assert.match(html, /h-full w-full min-w-0/)
  assert.match(html, /Training Run/)
  assert.match(html, /loss 0.4992/)
  assert.doesNotMatch(html, /react-flow__handle/)
  assert.doesNotMatch(html, /title="Close"/)
  assert.doesNotMatch(html, /w-\[360px\]/)
})
