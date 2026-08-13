import assert from 'node:assert/strict'
import test from 'node:test'
import { findDynamicNodePosition } from './canvasPlacement'

test('places a generated training node beside its active origin', () => {
  assert.deepEqual(findDynamicNodePosition('training', { x: 100, y: 120 }, [{ x: 100, y: 120 }]), {
    x: 520,
    y: 80
  })
})

test('chooses another nearby lane when the preferred position is occupied', () => {
  const occupied = [
    { x: 100, y: 120 },
    { x: 520, y: 120 }
  ]

  const position = findDynamicNodePosition('designer', { x: 100, y: 120 }, occupied)

  assert.deepEqual(position, { x: 520, y: 400 })
})

test('stays near the existing cluster when no explicit origin is known', () => {
  const position = findDynamicNodePosition('designer', undefined, [
    { x: 80, y: 80 },
    { x: 500, y: 80 }
  ])

  assert.deepEqual(position, { x: 920, y: 80 })
})

test('places the singleton Skill Lab in the visible lane beside Toolkit', () => {
  assert.deepEqual(findDynamicNodePosition('skilllab', { x: 240, y: 320 }, [{ x: 240, y: 320 }]), {
    x: -180,
    y: 320
  })
})
