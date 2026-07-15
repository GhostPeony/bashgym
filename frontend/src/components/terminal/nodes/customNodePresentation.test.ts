import assert from 'node:assert/strict'
import test from 'node:test'
import type { PanelType } from '../../../stores'
import { CUSTOM_NODE_TYPES, isCustomNodeType } from './customNodeTypes'

const expected: PanelType[] = [
  'context', 'neon', 'vercel', 'activity', 'training', 'campaign', 'evals',
  'designer', 'huggingface', 'agent', 'toolkit', 'skilllab', 'mcp', 'knowledge',
]

test('registers every persisted custom panel type for Canvas and Grid', () => {
  assert.deepEqual([...CUSTOM_NODE_TYPES], expected)
  for (const type of expected) {
    assert.equal(isCustomNodeType(type), true)
  }
  assert.equal(isCustomNodeType('terminal'), false)
  assert.equal(isCustomNodeType('browser'), false)
  assert.equal(isCustomNodeType('files'), false)
})
