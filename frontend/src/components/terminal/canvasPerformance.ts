import type { Node } from '@xyflow/react'
import type { CanvasEdge, Panel } from '../../stores'

export interface CanvasGraphIndex {
  connectedPanelIds: Set<string>
  dataConnectedPanelIds: Set<string>
  terminalConnectedPanelIds: Set<string>
  linkedPanelsById: Map<string, Panel[]>
  monitorByPanelId: Map<string, { isWatched: boolean; watchingTitle?: string }>
}

/** Build all edge-derived node context in one O(panels + edges) pass. */
export function buildCanvasGraphIndex(panels: Panel[], edges: CanvasEdge[]): CanvasGraphIndex {
  const panelById = new Map(panels.map((panel) => [panel.id, panel]))
  const connectedPanelIds = new Set<string>()
  const dataConnectedPanelIds = new Set<string>()
  const terminalConnectedPanelIds = new Set<string>()
  const linkedIdsByPanel = new Map<string, Set<string>>()
  const watchedPanelIds = new Set<string>()
  const watchedTitlesByWatcher = new Map<string, string[]>()

  const link = (panelId: string, linkedId: string) => {
    const linked = linkedIdsByPanel.get(panelId) ?? new Set<string>()
    linked.add(linkedId)
    linkedIdsByPanel.set(panelId, linked)
  }

  for (const edge of edges) {
    const source = panelById.get(edge.source)
    const target = panelById.get(edge.target)
    if (!source || !target) continue

    connectedPanelIds.add(source.id)
    connectedPanelIds.add(target.id)
    link(source.id, target.id)
    link(target.id, source.id)

    const isMonitor =
      source.type === 'terminal' && target.type === 'terminal' && source.id !== target.id
    if (isMonitor) {
      watchedPanelIds.add(source.id)
      const titles = watchedTitlesByWatcher.get(target.id) ?? []
      titles.push(source.title)
      watchedTitlesByWatcher.set(target.id, titles)
      continue
    }

    dataConnectedPanelIds.add(source.id)
    dataConnectedPanelIds.add(target.id)
    if (source.type === 'terminal') terminalConnectedPanelIds.add(target.id)
    if (target.type === 'terminal') terminalConnectedPanelIds.add(source.id)
  }

  const linkedPanelsById = new Map<string, Panel[]>()
  for (const [panelId, linkedIds] of linkedIdsByPanel) {
    linkedPanelsById.set(
      panelId,
      Array.from(linkedIds, (linkedId) => panelById.get(linkedId)).filter((panel): panel is Panel =>
        Boolean(panel)
      )
    )
  }

  const monitorByPanelId = new Map<string, { isWatched: boolean; watchingTitle?: string }>()
  for (const panel of panels) {
    const titles = watchedTitlesByWatcher.get(panel.id) ?? []
    monitorByPanelId.set(panel.id, {
      isWatched: watchedPanelIds.has(panel.id),
      watchingTitle:
        titles.length === 0
          ? undefined
          : titles.length === 1
            ? titles[0]
            : `${titles.length} terminals`
    })
  }

  return {
    connectedPanelIds,
    dataConnectedPanelIds,
    terminalConnectedPanelIds,
    linkedPanelsById,
    monitorByPanelId
  }
}

function shallowValueEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
  return left.every((value, index) => {
    const other = right[index]
    if (Object.is(value, other)) return true
    if (!value || !other || typeof value !== 'object' || typeof other !== 'object') return false
    const leftEntries = Object.entries(value as Record<string, unknown>)
    const rightRecord = other as Record<string, unknown>
    return (
      leftEntries.length === Object.keys(rightRecord).length &&
      leftEntries.every(([key, entry]) => Object.is(entry, rightRecord[key]))
    )
  })
}

function shallowRecordEqual(
  left: Record<string, unknown>,
  right: Record<string, unknown>
): boolean {
  if (left === right) return true
  const leftKeys = Object.keys(left)
  if (leftKeys.length !== Object.keys(right).length) return false
  return leftKeys.every((key) => shallowValueEqual(left[key], right[key]))
}

/** Preserve unchanged node objects so React Flow and memoized nodes can skip work. */
export function reconcileCanvasNodes<T extends Node>(previous: T[], candidates: T[]): T[] {
  const previousById = new Map(previous.map((node) => [node.id, node]))
  let changed = previous.length !== candidates.length
  const next = candidates.map((candidate, index) => {
    const current = previousById.get(candidate.id)
    if (!current) {
      changed = true
      return candidate
    }
    const unchanged =
      current.type === candidate.type &&
      current.position.x === candidate.position.x &&
      current.position.y === candidate.position.y &&
      current.selected === candidate.selected &&
      current.className === candidate.className &&
      shallowRecordEqual(
        current.data as Record<string, unknown>,
        candidate.data as Record<string, unknown>
      )
    if (unchanged) {
      if (previous[index] !== current) changed = true
      return current
    }
    changed = true
    return { ...current, ...candidate }
  })
  return changed ? next : previous
}
