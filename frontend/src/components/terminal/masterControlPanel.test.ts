import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const componentPath = fileURLToPath(new URL('./MasterControlPanel.tsx', import.meta.url))
const component = readFileSync(componentPath, 'utf8')

test('master control removes view options and keeps canvas controls in a compact grid', () => {
  assert.doesNotMatch(component, /View Options/)
  assert.doesNotMatch(component, /showMetrics|showToolHistory|showRecentFiles/)
  assert.match(component, /grid grid-cols-3 gap-1\.5/)
  assert.match(component, /Grid[\s\S]*Snap[\s\S]*Minimap/)
  assert.match(component, /Zoom:[\s\S]*onZoomOut[\s\S]*onZoomIn[\s\S]*onFitView/)
})
