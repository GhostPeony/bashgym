/**
 * Workspace registry + switching.
 *
 * Exactly one workspace is materialized in terminalStore's globals at a time.
 * switchWorkspace is fully synchronous: serialize the outgoing workspace to
 * its namespaced keys, flip the active id, load the incoming snapshot into
 * the globals in one setState. PTYs are NEVER killed by a switch — panes
 * unmount (xterm disposes) and re-attach with scrollback replay on return.
 */

import { create } from 'zustand'
import { useTerminalStore } from './terminalStore'
import type { Panel, TerminalSession } from './terminalStore'
import {
  type WorkspaceMeta,
  deleteWorkspaceKeys,
  generateWorkspaceId,
  loadRegistry,
  loadWorkspaceSnapshot,
  saveActivePanelFor,
  saveEdgesFor,
  savePanelsFor,
  savePositionsFor,
  saveRegistry,
  setActiveWorkspaceId,
  wsKey
} from './workspacePersistence'

interface WorkspaceState {
  workspaces: WorkspaceMeta[]
  activeWorkspaceId: string
  /** True for the synchronous duration of a switch — gates external event handlers */
  switching: boolean

  createWorkspace: (name: string, opts?: { activate?: boolean }) => string
  switchWorkspace: (id: string) => void
  renameWorkspace: (id: string, name: string) => void
  /** Refuses the active workspace; confirms before killing a background workspace's live PTYs */
  deleteWorkspace: (id: string) => Promise<boolean>
  moveLivePanelToWorkspace: (panelId: string, targetWsId: string) => void
}

function serializeWorkspaceState(wsId: string): void {
  const state = useTerminalStore.getState()
  savePanelsFor(wsId, state.panels, state.sessions)
  savePositionsFor(wsId, state.canvasNodes)
  saveEdgesFor(wsId, state.canvasEdges)
  saveActivePanelFor(wsId, state.activePanelId)
}

/** Load a workspace's snapshot into the terminalStore globals (one setState) */
function materializeWorkspace(wsId: string): void {
  const snap = loadWorkspaceSnapshot(wsId)

  const sessions = new Map<string, TerminalSession>()
  const panels: Panel[] = []
  for (const stub of snap.panels) {
    panels.push({
      id: stub.id,
      type: stub.type,
      title: stub.title,
      terminalId: stub.terminalId,
      url: stub.url,
      filePath: stub.filePath,
      adapterConfig: stub.adapterConfig
    })
    if (stub.type === 'terminal' && stub.terminalId) {
      sessions.set(stub.terminalId, {
        id: stub.terminalId,
        title: stub.title,
        cwd: stub.cwd ?? '~',
        isActive: false,
        attention: 'none',
        showBanner: false,
        status: 'idle',
        agentKind: stub.agentKind,
        lastActivity: Date.now()
      })
    }
  }

  const activePanelId =
    snap.activePanelId ?? (panels.length > 0 ? panels[panels.length - 1].id : null)
  const activePanel = panels.find((p) => p.id === activePanelId)

  useTerminalStore.setState((state) => ({
    sessions,
    panels,
    canvasNodes: new Map(Object.entries(snap.positions)),
    canvasEdges: snap.edges,
    activePanelId,
    activeSessionId: activePanel?.terminalId ?? null,
    sessionsVersion: state.sessionsVersion + 1
  }))
}

const initialRegistry = loadRegistry()

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  workspaces: initialRegistry.workspaces,
  activeWorkspaceId: initialRegistry.activeWorkspaceId,
  switching: false,

  createWorkspace: (name: string, opts?: { activate?: boolean }) => {
    const id = generateWorkspaceId()
    const now = Date.now()
    const registry = loadRegistry()
    const meta: WorkspaceMeta = {
      id,
      name: name.trim() || `Workspace ${registry.workspaces.length + 1}`,
      createdAt: now,
      lastActiveAt: now
    }
    saveRegistry({ ...registry, workspaces: [...registry.workspaces, meta] })
    set({ workspaces: loadRegistry().workspaces })
    if (opts?.activate) get().switchWorkspace(id)
    return id
  },

  switchWorkspace: (id: string) => {
    const { switching, activeWorkspaceId, workspaces } = get()
    if (switching || id === activeWorkspaceId || !workspaces.some((w) => w.id === id)) return

    set({ switching: true })
    try {
      serializeWorkspaceState(activeWorkspaceId)
      setActiveWorkspaceId(id)
      set({ activeWorkspaceId: id, workspaces: loadRegistry().workspaces })
      materializeWorkspace(id)
    } finally {
      set({ switching: false })
    }
  },

  renameWorkspace: (id: string, name: string) => {
    const trimmed = name.trim()
    if (!trimmed) return
    const registry = loadRegistry()
    saveRegistry({
      ...registry,
      workspaces: registry.workspaces.map((w) => (w.id === id ? { ...w, name: trimmed } : w))
    })
    set({ workspaces: loadRegistry().workspaces })
  },

  deleteWorkspace: async (id: string) => {
    const { activeWorkspaceId } = get()
    if (id === activeWorkspaceId) return false
    const registry = loadRegistry()
    const meta = registry.workspaces.find((w) => w.id === id)
    if (!meta || registry.workspaces.length <= 1) return false

    // Live PTYs owned by this workspace get killed only after explicit confirm
    const snap = loadWorkspaceSnapshot(id)
    const ownedTerminalIds = snap.panels
      .filter((p) => p.type === 'terminal' && p.terminalId)
      .map((p) => p.terminalId as string)
    let liveOwned: string[] = []
    try {
      const infos = (await window.bashgym?.terminal.list()) ?? []
      const live = new Set(infos.filter((i) => !i.exited).map((i) => i.id))
      liveOwned = ownedTerminalIds.filter((tid) => live.has(tid))
    } catch {
      // No PTY bridge
    }
    if (liveOwned.length > 0) {
      const ok = window.confirm(
        `Workspace "${meta.name}" has ${liveOwned.length} running terminal${liveOwned.length !== 1 ? 's' : ''}. ` +
          `Delete the workspace and kill them? (Backend training runs are NOT stopped.)`
      )
      if (!ok) return false
      for (const tid of liveOwned) {
        void window.bashgym?.terminal.kill(tid)
      }
    }

    deleteWorkspaceKeys(id)
    const after = loadRegistry()
    saveRegistry({ ...after, workspaces: after.workspaces.filter((w) => w.id !== id) })
    set({ workspaces: loadRegistry().workspaces })
    return true
  },

  moveLivePanelToWorkspace: (panelId: string, targetWsId: string) => {
    const { workspaces, activeWorkspaceId } = get()
    if (targetWsId === activeWorkspaceId || !workspaces.some((w) => w.id === targetWsId)) return

    const detached = useTerminalStore.getState().detachPanel(panelId)
    if (!detached) return

    // Append the panel (and its position) to the target workspace's keys
    try {
      const rawPanels = localStorage.getItem(wsKey(targetWsId, 'panels'))
      const panels = rawPanels ? JSON.parse(rawPanels) : []
      panels.push(detached.stub)
      localStorage.setItem(wsKey(targetWsId, 'panels'), JSON.stringify(panels))
      if (detached.position) {
        const rawPos = localStorage.getItem(wsKey(targetWsId, 'positions'))
        const positions = rawPos ? JSON.parse(rawPos) : {}
        positions[panelId] = detached.position
        localStorage.setItem(wsKey(targetWsId, 'positions'), JSON.stringify(positions))
      }
    } catch {
      // Storage error: the panel is detached but not re-homed; orphan
      // adoption will recover its PTY on next boot
    }

    get().switchWorkspace(targetWsId)
  }
}))
