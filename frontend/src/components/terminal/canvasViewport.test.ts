import assert from 'node:assert/strict'
import test from 'node:test'
import { loadCanvasViewport, saveCanvasViewport, type ViewportStorage } from './canvasViewport'

function memoryStorage(): ViewportStorage {
  const values = new Map<string, string>()
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
  }
}

test('round-trips a workspace viewport', () => {
  const storage = memoryStorage()
  saveCanvasViewport(storage, 'workspace-a', { x: 12, y: -8, zoom: 0.45 })
  assert.deepEqual(loadCanvasViewport(storage, 'workspace-a'), { x: 12, y: -8, zoom: 0.45 })
  assert.equal(loadCanvasViewport(storage, 'workspace-b'), null)
})

test('rejects malformed and invalid viewport state', () => {
  const storage = memoryStorage()
  storage.setItem('broken', '{')
  storage.setItem('missing', JSON.stringify({ x: 1, zoom: 1 }))
  storage.setItem('zero', JSON.stringify({ x: 1, y: 2, zoom: 0 }))
  assert.equal(loadCanvasViewport(storage, 'broken'), null)
  assert.equal(loadCanvasViewport(storage, 'missing'), null)
  assert.equal(loadCanvasViewport(storage, 'zero'), null)
})
