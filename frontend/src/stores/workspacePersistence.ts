/**
 * Workspace persistence — key layout, registry, per-workspace snapshots.
 *
 * Exactly one workspace is materialized in the global stores at a time;
 * every other workspace exists only as the namespaced localStorage state
 * written here (plus its live PTYs in the Electron main process). Pure
 * data module: no store imports (type-only), safe at the bottom of the
 * dependency graph.
 */

import type { CanvasEdge, CanvasNode, Panel, PanelType, TerminalSession } from './terminalStore'
import type { AgentKind } from './terminalStore'

export interface WorkspaceMeta {
  id: string
  name: string
  createdAt: number
  lastActiveAt: number
}

export interface WorkspaceRegistry {
  version: 1
  activeWorkspaceId: string
  workspaces: WorkspaceMeta[]
}

/**
 * Terminal panels persist a session stub (cwd/agentKind) so switching in can
 * rebuild session objects immediately, before the PTY re-attach reports its
 * cwd. `thumbnail` is deliberately never persisted (in-memory only, quota).
 */
export interface WsPersistedPanel {
  id: string
  type: PanelType
  title: string
  terminalId?: string
  cwd?: string
  agentKind?: AgentKind
  url?: string
  filePath?: string
  adapterConfig?: Record<string, unknown>
}

export interface WorkspaceSnapshotV1 {
  panels: WsPersistedPanel[]
  positions: Record<string, CanvasNode>
  edges: CanvasEdge[]
  activePanelId: string | null
}

const REGISTRY_KEY = 'bashgym_workspaces'
export const DEFAULT_WORKSPACE_ID = 'default'

type WsKeySuffix = 'panels' | 'positions' | 'edges' | 'viewport' | 'active_panel'

export const wsKey = (id: string, suffix: WsKeySuffix) => `bashgym_ws_${id}_${suffix}`

let workspaceIdCounter = 0
export const generateWorkspaceId = () => `ws-${Date.now().toString(36)}-${++workspaceIdCounter}`

const readJson = <T>(key: string): T | null => {
  try {
    const stored = localStorage.getItem(key)
    if (stored) return JSON.parse(stored) as T
  } catch {
    // Ignore
  }
  return null
}

const writeJson = (key: string, value: unknown) => {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage errors
  }
}

// Legacy single-workspace keys, migrated once into the default workspace
const LEGACY_KEYS = {
  panels: 'bashgym_saved_panels',
  positions: 'bashgym_canvas_positions',
  edges: 'bashgym_canvas_edges',
  viewport: 'bashgym_canvas_viewport',
  panelOrder: 'bashgym_panel_order'
}

function migrateLegacyKeys(): void {
  for (const [suffix, legacyKey] of [
    ['panels', LEGACY_KEYS.panels],
    ['positions', LEGACY_KEYS.positions],
    ['edges', LEGACY_KEYS.edges],
    ['viewport', LEGACY_KEYS.viewport]
  ] as Array<[WsKeySuffix, string]>) {
    try {
      const value = localStorage.getItem(legacyKey)
      if (value !== null) {
        localStorage.setItem(wsKey(DEFAULT_WORKSPACE_ID, suffix), value)
      }
    } catch {
      // Ignore
    }
  }
  for (const legacyKey of Object.values(LEGACY_KEYS)) {
    try {
      localStorage.removeItem(legacyKey)
    } catch {
      // Ignore
    }
  }
}

let registryCache: WorkspaceRegistry | null = null

export function loadRegistry(): WorkspaceRegistry {
  if (registryCache) return registryCache
  const stored = readJson<WorkspaceRegistry>(REGISTRY_KEY)
  if (stored && stored.version === 1 && Array.isArray(stored.workspaces) && stored.workspaces.length > 0) {
    if (!stored.workspaces.some((w) => w.id === stored.activeWorkspaceId)) {
      stored.activeWorkspaceId = stored.workspaces[0].id
    }
    registryCache = stored
    return stored
  }

  // First run under the workspace system: adopt the existing single-canvas
  // state as the default workspace
  const now = Date.now()
  const registry: WorkspaceRegistry = {
    version: 1,
    activeWorkspaceId: DEFAULT_WORKSPACE_ID,
    workspaces: [{ id: DEFAULT_WORKSPACE_ID, name: 'MAIN', createdAt: now, lastActiveAt: now }]
  }
  migrateLegacyKeys()
  writeJson(REGISTRY_KEY, registry)
  registryCache = registry
  return registry
}

export function saveRegistry(registry: WorkspaceRegistry): void {
  registryCache = registry
  writeJson(REGISTRY_KEY, registry)
}

export function getActiveWorkspaceId(): string {
  return loadRegistry().activeWorkspaceId
}

export function setActiveWorkspaceId(id: string): void {
  const registry = loadRegistry()
  saveRegistry({
    ...registry,
    activeWorkspaceId: id,
    workspaces: registry.workspaces.map((w) => (w.id === id ? { ...w, lastActiveAt: Date.now() } : w))
  })
}

export function toPersistedPanel(panel: Panel, sessions: Map<string, TerminalSession>): WsPersistedPanel {
  const session = panel.terminalId ? sessions.get(panel.terminalId) : undefined
  return {
    id: panel.id,
    type: panel.type,
    title: panel.title,
    terminalId: panel.terminalId,
    cwd: session?.cwd,
    agentKind: session?.agentKind,
    url: panel.url,
    filePath: panel.filePath,
    adapterConfig: panel.adapterConfig
  }
}

export function savePanelsFor(id: string, panels: Panel[], sessions: Map<string, TerminalSession>): void {
  writeJson(wsKey(id, 'panels'), panels.map((p) => toPersistedPanel(p, sessions)))
}

export function savePositionsFor(id: string, nodes: Map<string, CanvasNode>): void {
  writeJson(wsKey(id, 'positions'), Object.fromEntries(nodes))
}

export function saveEdgesFor(id: string, edges: CanvasEdge[]): void {
  writeJson(wsKey(id, 'edges'), edges)
}

export function saveActivePanelFor(id: string, panelId: string | null): void {
  writeJson(wsKey(id, 'active_panel'), panelId)
}

/** Load a workspace's persisted state, dropping references to missing panels */
export function loadWorkspaceSnapshot(id: string): WorkspaceSnapshotV1 {
  const panels = readJson<WsPersistedPanel[]>(wsKey(id, 'panels')) ?? []
  const panelIds = new Set(panels.map((p) => p.id))

  const rawPositions = readJson<Record<string, CanvasNode>>(wsKey(id, 'positions')) ?? {}
  const positions = Object.fromEntries(Object.entries(rawPositions).filter(([key]) => panelIds.has(key)))

  const rawEdges = readJson<CanvasEdge[]>(wsKey(id, 'edges')) ?? []
  const edges = rawEdges.filter((e) => panelIds.has(e.source) && panelIds.has(e.target))

  const rawActive = readJson<string | null>(wsKey(id, 'active_panel'))
  const activePanelId = rawActive && panelIds.has(rawActive) ? rawActive : null

  return { panels, positions, edges, activePanelId }
}

export function deleteWorkspaceKeys(id: string): void {
  for (const suffix of ['panels', 'positions', 'edges', 'viewport', 'active_panel'] as WsKeySuffix[]) {
    try {
      localStorage.removeItem(wsKey(id, suffix))
    } catch {
      // Ignore
    }
  }
}

/** terminalId → owning workspace id, across every workspace's persisted panels */
export function collectClaimedTerminalIds(registry: WorkspaceRegistry): Map<string, string> {
  const claimed = new Map<string, string>()
  for (const ws of registry.workspaces) {
    const panels = readJson<WsPersistedPanel[]>(wsKey(ws.id, 'panels')) ?? []
    for (const panel of panels) {
      if (panel.terminalId) claimed.set(panel.terminalId, ws.id)
    }
  }
  return claimed
}
