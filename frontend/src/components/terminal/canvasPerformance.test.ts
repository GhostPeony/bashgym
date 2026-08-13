import assert from 'node:assert/strict'
import test from 'node:test'
import type { Node } from '@xyflow/react'
import type { CanvasEdge, Panel } from '../../stores'
import { buildCanvasGraphIndex, reconcileCanvasNodes } from './canvasPerformance'

const panel = (id: string, type: Panel['type'] = 'terminal'): Panel => ({ id, type, title: id })

test('indexes canvas connections and monitor direction in one graph snapshot', () => {
  const panels = [panel('a'), panel('b'), panel('data', 'activity')]
  const edges: CanvasEdge[] = [
    { id: 'monitor', source: 'a', target: 'b' },
    { id: 'data', source: 'b', target: 'data' }
  ]

  const graph = buildCanvasGraphIndex(panels, edges)
  assert.equal(graph.monitorByPanelId.get('a')?.isWatched, true)
  assert.equal(graph.monitorByPanelId.get('b')?.watchingTitle, 'a')
  assert.equal(graph.dataConnectedPanelIds.has('a'), false)
  assert.equal(graph.dataConnectedPanelIds.has('b'), true)
  assert.equal(graph.terminalConnectedPanelIds.has('data'), true)
})

test('preserves all unrelated node objects when one high-volume task changes', () => {
  const previous: Node[] = Array.from({ length: 200 }, (_, index) => ({
    id: `task-${index}`,
    type: 'terminal',
    position: { x: index * 10, y: 0 },
    data: { status: 'running', step: 1 }
  }))
  const candidates = previous.map((node, index) => ({
    id: node.id,
    type: node.type,
    position: { ...node.position },
    data: { status: 'running', step: index === 91 ? 2 : 1 }
  }))

  const next = reconcileCanvasNodes(previous, candidates)
  assert.notEqual(next, previous)
  assert.notEqual(next[91], previous[91])
  assert.equal(next.filter((node, index) => node === previous[index]).length, 199)
})
