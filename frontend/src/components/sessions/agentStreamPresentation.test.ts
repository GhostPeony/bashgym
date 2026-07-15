import assert from 'node:assert/strict'
import test from 'node:test'
import { clampStreamGeometry, normalizeTerminalFeed } from './agentStreamPresentation'

test('removes terminal redraw frames while preserving meaningful indentation', () => {
  const lines = normalizeTerminalFeed([
    '────────────────────────────────────────',
    '│ Building workspace',
    '│ Building workspace',
    '    const answer = 42',
    '██ Claude Code',
    '│ Building workspace',
    '╰───────────────────────────────────────'
  ])

  assert.deepEqual(lines, ['    const answer = 42', 'Claude Code', '  Building workspace'])
})

test('keeps a dragged and resized stream inside the viewport', () => {
  assert.deepEqual(
    clampStreamGeometry(
      { x: 900, y: -20, width: 500, height: 900 },
      { width: 1200, height: 800 }
    ),
    { x: 692, y: 56, width: 500, height: 736 }
  )
})
