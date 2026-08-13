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
  appendPanelToWorkspace,
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
  terminalSessionFromPersistedPanel,
  removePanelFromWorkspaceSnapshot,
  toPersistedPanel
} from './workspacePersistence'

interface WorkspaceState {
  workspaces: WorkspaceMeta[]
  activeWorkspaceId: string
  /** True for the synchronous duration of a switch — gates external event handlers */
  switching: boolean
  sessionIndexVersion: number

  createWorkspace: (name: string, opts?: { activate?: boolean }) => string
  switchWorkspace: (id: string) => void
  renameWorkspace: (id: string, name: string) => void
  /** Refuses the active workspace; confirms before killing a background workspace's live PTYs */
  deleteWorkspace: (id: string) => Promise<boolean>
  moveLivePanelToWorkspace: (
    panelId: string,
    targetWsId: string,
    opts?: { switchAfter?: boolean }
  ) => boolean
  adoptTerminalIntoWorkspace: (
    workspaceId: string,
    terminal: { terminalId: string; title: string; cwd: string }
  ) => boolean
}

let adoptedPanelCounter = 0
const generateAdoptedPanelId = () => `panel-adopted-${++adoptedPanelCounter}-${Date.now()}`

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
      sessions.set(stub.terminalId, terminalSessionFromPersistedPanel(stub))
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
  sessionIndexVersion: 0,

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
    set((state) => ({
      workspaces: loadRegistry().workspaces,
      sessionIndexVersion: state.sessionIndexVersion + 1
    }))
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
      set((state) => ({
        activeWorkspaceId: id,
        workspaces: loadRegistry().workspaces,
        sessionIndexVersion: state.sessionIndexVersion + 1
      }))
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
    set((state) => ({
      workspaces: loadRegistry().workspaces,
      sessionIndexVersion: state.sessionIndexVersion + 1
    }))
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
    set((state) => ({
      workspaces: loadRegistry().workspaces,
      sessionIndexVersion: state.sessionIndexVersion + 1
    }))
    return true
  },

  moveLivePanelToWorkspace: (panelId, targetWsId, opts) => {
    const { workspaces, activeWorkspaceId } = get()
    if (targetWsId === activeWorkspaceId || !workspaces.some((w) => w.id === targetWsId))
      return false

    const terminalState = useTerminalStore.getState()
    const panel = terminalState.panels.find((candidate) => candidate.id === panelId)
    if (!panel) return false
    const stub = toPersistedPanel(panel, terminalState.sessions)
    const position = terminalState.canvasNodes.get(panelId)

    // Commit the destination first. A failed write leaves the source untouched.
    if (!appendPanelToWorkspace(targetWsId, stub, position)) return false
    const detached = terminalState.detachPanel(panelId)
    if (!detached) {
      removePanelFromWorkspaceSnapshot(targetWsId, panelId)
      return false
    }

    set((state) => ({ sessionIndexVersion: state.sessionIndexVersion + 1 }))
    if (opts?.switchAfter !== false) get().switchWorkspace(targetWsId)
    return true
  },

  adoptTerminalIntoWorkspace: (workspaceId, terminal) => {
    const { workspaces, activeWorkspaceId } = get()
    if (!workspaces.some((workspace) => workspace.id === workspaceId)) return false

    if (workspaceId === activeWorkspaceId) {
      const terminalState = useTerminalStore.getState()
      const existingPanel = terminalState.panels.find(
        (panel) => panel.terminalId === terminal.terminalId
      )
      if (existingPanel) {
        terminalState.setActivePanel(existingPanel.id)
        return true
      }

      terminalState.createTerminal(terminal.terminalId, terminal.title, undefined, terminal.cwd)
      set((state) => ({ sessionIndexVersion: state.sessionIndexVersion + 1 }))
      return true
    }

    const snapshot = loadWorkspaceSnapshot(workspaceId)
    if (snapshot.panels.some((panel) => panel.terminalId === terminal.terminalId)) {
      return true
    }

    const appended = appendPanelToWorkspace(workspaceId, {
      id: generateAdoptedPanelId(),
      type: 'terminal',
      title: terminal.title,
      terminalId: terminal.terminalId,
      cwd: terminal.cwd,
      sessionState: {
        status: 'idle',
        attention: 'none',
        lastActivity: Date.now()
      }
    })
    if (!appended) return false

    set((state) => ({ sessionIndexVersion: state.sessionIndexVersion + 1 }))
    return true
  }
}))
