import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildKnowledgeContext,
  countVisibleKnowledgeNodes,
  sanitizeKnowledgeConfig,
  type KnowledgeSnapshot
} from './knowledgeBaseModel'

test('knowledge config keeps only bounded presentation and source fields', () => {
  assert.deepEqual(sanitizeKnowledgeConfig({
    provider: 'gbrain',
    root: 'C:/brain',
    label: 'Team Brain',
    selectedSection: 'search',
    token: 'must-not-survive'
  }), {
    provider: 'gbrain',
    root: 'C:/brain',
    label: 'Team Brain',
    selectedPath: undefined,
    selectedSection: 'search'
  })
})

test('knowledge tree count includes nested files and folders', () => {
  assert.equal(countVisibleKnowledgeNodes([
    { name: 'docs', path: 'docs', type: 'folder', children: [
      { name: 'readme.md', path: 'docs/readme.md', type: 'file' }
    ] }
  ]), 2)
})

test('published knowledge context preserves provenance paths', () => {
  const snapshot: KnowledgeSnapshot = {
    provider: 'workspace', root: 'C:/repo', label: 'repo', tree: [],
    counts: { files: 3, folders: 2, knowledge_files: 2 },
    truncated: false, capabilities: ['browse']
  }
  const context = buildKnowledgeContext(
    sanitizeKnowledgeConfig({ label: 'Product Brain' }),
    snapshot,
    {
      provider: 'workspace', root: 'C:/repo', query: 'oauth',
      results: [{ path: 'docs/auth.md', snippet: 'OAuth uses PKCE.' }],
      scanned_files: 2, truncated: false
    }
  )
  assert.match(context, /docs\/auth\.md/)
  assert.match(context, /read-only context/)
})
