import { create } from 'zustand'
import {
  DEFAULT_WORKSPACE_ID,
  type WsPersistedPanel,
  collectClaimedTerminalIds,
  getActiveWorkspaceId,
  loadRegistry,
  loadWorkspaceSnapshot,
  saveActivePanelFor,
  savePanelsFor,
  savePositionsFor,
  toPersistedPanel
} from './workspacePersistence'

export type PanelType = 'terminal' | 'preview' | 'browser' | 'files' | 'context' | 'neon' | 'vercel' | 'activity' | 'training' | 'evals' | 'designer' | 'huggingface' | 'agent' | 'toolkit'
export type AttentionState = 'none' | 'waiting' | 'success' | 'error'
export type ViewMode = 'grid' | 'single' | 'canvas'
export type AgentStatus = 'running' | 'idle' | 'waiting_input' | 'tool_calling'
export type AgentKind = 'claude' | 'codex'

// Tool history entry for tracking recent tool calls
export interface ToolHistoryItem {
  tool: string
  target?: string
  timestamp: number
}

// Token/cost metrics for a session
export interface SessionMetrics {
  inputTokens: number
  outputTokens: number
  cost: number
}

export interface TerminalSession {
  id: string
  title: string
  cwd: string
  isActive: boolean
  attention: AttentionState
  gitBranch?: string
  model?: string
  showBanner: boolean  // Only first terminal shows the Bash Gym banner
  // Enhanced status fields (OpenUI-inspired)
  status: AgentStatus
  lastActivity: number
  currentTool?: string
  taskSummary?: string
  // Canvas enhancement fields
  toolHistory?: ToolHistoryItem[]
  metrics?: SessionMetrics
  recentFiles?: string[]
  errorMessage?: string
  isExpanded?: boolean
  isPaused?: boolean
  /** Last few lines of terminal output for canvas preview */
  lastOutput?: string[]
  /** Detected agent CLI running in this terminal (undefined = plain shell) */
  agentKind?: AgentKind
  /** Command auto-typed into a freshly created PTY (cleared after use) */
  launchCommand?: string
}

// Canvas node for React Flow view
export interface CanvasNode {
  id: string
  panelId: string
  position: { x: number; y: number }
  data: {
    label: string
    status: AttentionState
    gitBranch?: string
    model?: string
  }
}

/** Auto-snapshot mode for a monitor edge: prefill types the path, send submits it */
export type MonitorAutoMode = 'off' | 'prefill' | 'send'

// Canvas edge — a connection between two panels on the canvas.
// Terminal→terminal edges are monitor edges: source = watched, target = watcher.
export interface CanvasEdge {
  id: string
  source: string  // panelId of source node
  target: string  // panelId of target node
  type?: 'monitor'
  data?: { auto?: MonitorAutoMode }
}

export interface Panel {
  id: string
  type: PanelType
  title: string
  // For terminal panels
  terminalId?: string
  // For preview panels
  filePath?: string
  // For browser panels
  url?: string
  // In-memory screenshot (lives only for panel lifetime, never persisted)
  thumbnail?: string
  adapterConfig?: Record<string, unknown>  // Integration node config
}

interface TerminalState {
  // Sessions
  sessions: Map<string, TerminalSession>
  activeSessionId: string | null
  /** Version counter that increments on every session update - forces React re-renders */
  sessionsVersion: number

  // Panel layout
  panels: Panel[]
  activePanelId: string | null

  // View mode
  viewMode: ViewMode

  // Canvas nodes for React Flow view
  canvasNodes: Map<string, CanvasNode>

  // Canvas edges — connections between panels (browser→terminal routing)
  canvasEdges: CanvasEdge[]

  // Drag state for tab reordering
  draggedPanelId: string | null

  // Broadcast
  broadcastCommand: string

  // Actions
  createTerminal: (id?: string, title?: string, launchCommand?: string, cwd?: string) => string
  closeTerminal: (id: string) => void
  /** Close with confirmation if the agent is busy. Returns true if closed. */
  requestCloseTerminal: (id: string) => boolean
  /**
   * Materialize the active workspace on startup: rebuild panels/sessions from
   * its persisted snapshot, re-adopt live PTYs, and adopt any orphaned PTYs
   * unclaimed by every workspace. Call once on app boot.
   */
  bootActiveWorkspace: () => Promise<void>
  setActiveTerminal: (id: string) => void
  updateSession: (id: string, updates: Partial<TerminalSession>) => void

  addPanel: (panel: Omit<Panel, 'id'>) => string
  removePanel: (id: string) => void
  /**
   * Remove a panel (and its session/edges/position) from the ACTIVE workspace
   * WITHOUT killing its PTY — used to move a live terminal between workspaces.
   */
  detachPanel: (panelId: string) => { stub: WsPersistedPanel; position?: CanvasNode } | null
  setActivePanel: (id: string) => void
  renamePanel: (id: string, title: string) => void
  reorderPanels: (sourceId: string, targetId: string) => void

  setViewMode: (mode: ViewMode) => void
  toggleViewMode: () => void
  cycleViewMode: () => void

  // Canvas actions
  updateCanvasNode: (panelId: string, position: { x: number; y: number }) => void
  setCanvasEdges: (edges: CanvasEdge[]) => void

  // Drag actions
  setDraggedPanel: (id: string | null) => void

  setBroadcastCommand: (command: string) => void
  executeBroadcast: () => void
  setPanelUrl: (id: string, url: string) => void
  setPanelThumbnail: (id: string, thumbnail: string) => void
  updatePanelConfig: (panelId: string, adapterConfig: Record<string, unknown>) => void
}

let panelCounter = 0
const generateId = () => `panel-${++panelCounter}-${Date.now()}`

// Track banner shown state in sessionStorage to persist across page refreshes
const getBannerShown = (): boolean => {
  if (typeof window === 'undefined') return false
  return sessionStorage.getItem('bashgym_banner_shown') === 'true'
}

const setBannerShown = (): void => {
  if (typeof window !== 'undefined') {
    sessionStorage.setItem('bashgym_banner_shown', 'true')
  }
}

// View mode stays global (not per-workspace)
const VIEW_MODE_KEY = 'bashgym_view_mode'

// Workspace-keyed persistence: positions/panels/active-panel are written to
// the ACTIVE workspace's namespaced keys (workspacePersistence.ts)
const loadCanvasPositions = (): Map<string, CanvasNode> =>
  new Map(Object.entries(loadWorkspaceSnapshot(getActiveWorkspaceId()).positions))

const saveCanvasPositions = (nodes: Map<string, CanvasNode>) =>
  savePositionsFor(getActiveWorkspaceId(), nodes)

const persistPanels = (panels: Panel[], sessions: Map<string, TerminalSession>) =>
  savePanelsFor(getActiveWorkspaceId(), panels, sessions)

const loadViewMode = (): ViewMode => {
  try {
    const stored = localStorage.getItem(VIEW_MODE_KEY)
    if (stored && ['grid', 'single', 'canvas'].includes(stored)) {
      return stored as ViewMode
    }
  } catch {
    // Ignore
  }
  return 'grid'
}

const saveViewMode = (mode: ViewMode) => {
  try {
    localStorage.setItem(VIEW_MODE_KEY, mode)
  } catch {
    // Ignore
  }
}

// Session metadata (id + title) persisted so live PTYs re-adopted from the
// main process after a renderer reload keep their names
const SESSION_META_KEY = 'bashgym_session_meta'

interface PersistedSessionMeta {
  id: string
  title: string
}

const loadSessionMeta = (): Map<string, PersistedSessionMeta> => {
  try {
    const stored = localStorage.getItem(SESSION_META_KEY)
    if (stored) {
      const arr: PersistedSessionMeta[] = JSON.parse(stored)
      return new Map(arr.map((m) => [m.id, m]))
    }
  } catch {
    // Ignore
  }
  return new Map()
}

const saveSessionMeta = (sessions: Map<string, TerminalSession>) => {
  try {
    const arr = Array.from(sessions.values()).map((s) => ({ id: s.id, title: s.title }))
    localStorage.setItem(SESSION_META_KEY, JSON.stringify(arr))
  } catch {
    // Ignore storage errors
  }
}

// A running/tool_calling session is never killed without explicit confirmation
const confirmCloseIfBusy = (session: TerminalSession | undefined): boolean => {
  if (session && (session.status === 'running' || session.status === 'tool_calling')) {
    return window.confirm(
      `"${session.title}" still has an agent working. Close it and kill the process?`
    )
  }
  return true
}

export const useTerminalStore = create<TerminalState>((set, get) => ({
  sessions: new Map(),
  activeSessionId: null,
  sessionsVersion: 0,
  panels: [],
  activePanelId: null,
  viewMode: loadViewMode(),
  canvasNodes: loadCanvasPositions(),
  canvasEdges: [],
  draggedPanelId: null,
  broadcastCommand: '',

  createTerminal: (id?: string, title?: string, launchCommand?: string, cwd?: string) => {
    const terminalId = id || generateId()

    // Show banner only if it hasn't been shown yet in this session
    const showBanner = !getBannerShown()
    if (showBanner) {
      setBannerShown()
    }

    const session: TerminalSession = {
      id: terminalId,
      title: title || `Terminal ${get().sessions.size + 1}`,
      cwd: cwd || '~',
      isActive: true,
      attention: 'none',
      showBanner,  // Only first terminal shows the banner
      status: 'idle',
      lastActivity: Date.now(),
      launchCommand
    }

    const panel: Panel = {
      id: generateId(),
      type: 'terminal',
      title: session.title,
      terminalId
    }

    set((state) => {
      const newSessions = new Map(state.sessions)
      newSessions.set(terminalId, session)

      return {
        sessions: newSessions,
        activeSessionId: terminalId,
        panels: [...state.panels, panel],
        activePanelId: panel.id
      }
    })

    saveSessionMeta(get().sessions)
    persistPanels(get().panels, get().sessions)
    return terminalId
  },

  closeTerminal: (id: string) => {
    set((state) => {
      const newSessions = new Map(state.sessions)
      newSessions.delete(id)

      const newPanels = state.panels.filter((p) => p.terminalId !== id)
      const newActiveSession = newSessions.size > 0
        ? Array.from(newSessions.keys())[0]
        : null

      return {
        sessions: newSessions,
        activeSessionId: state.activeSessionId === id ? newActiveSession : state.activeSessionId,
        panels: newPanels,
        activePanelId: newPanels.length > 0 ? newPanels[newPanels.length - 1].id : null
      }
    })

    saveSessionMeta(get().sessions)
    persistPanels(get().panels, get().sessions)

    // Kill the terminal process
    window.bashgym?.terminal.kill(id)
  },

  requestCloseTerminal: (id: string) => {
    if (!confirmCloseIfBusy(get().sessions.get(id))) return false
    get().closeTerminal(id)
    return true
  },

  bootActiveWorkspace: async () => {
    const registry = loadRegistry()
    const snap = loadWorkspaceSnapshot(registry.activeWorkspaceId)
    const meta = loadSessionMeta()

    let infos: Awaited<ReturnType<NonNullable<typeof window.bashgym>['terminal']['list']>> = []
    try {
      infos = (await window.bashgym?.terminal.list()) ?? []
    } catch {
      // No PTY bridge (web mode) — panels still restore
    }
    const live = new Map(infos.filter((i) => !i.exited).map((i) => [i.id, i]))

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
        // Dead PTYs keep their panel: TerminalPane respawns a fresh shell in
        // the stub's cwd (layout restore after full app quit)
        const info = live.get(stub.terminalId)
        sessions.set(stub.terminalId, {
          id: stub.terminalId,
          title: meta.get(stub.terminalId)?.title || stub.title,
          cwd: info?.cwd ?? stub.cwd ?? '~',
          isActive: false,
          attention: 'none',
          showBanner: false,
          status: 'idle',
          agentKind: stub.agentKind,
          lastActivity: Date.now()
        })
      }
    }

    // Orphan adoption: live PTYs unclaimed by ANY workspace land in the active
    // one, so no running process is ever stranded invisible
    const claimed = collectClaimedTerminalIds(registry)
    for (const [id, info] of live) {
      if (claimed.has(id) || sessions.has(id)) continue
      const title = meta.get(id)?.title || `Terminal ${sessions.size + 1}`
      sessions.set(id, {
        id,
        title,
        cwd: info.cwd,
        isActive: false,
        attention: 'none',
        showBanner: false,
        status: 'idle',
        lastActivity: Date.now()
      })
      panels.push({ id: generateId(), type: 'terminal', title, terminalId: id })
    }

    const activePanelId =
      snap.activePanelId && panels.some((p) => p.id === snap.activePanelId)
        ? snap.activePanelId
        : panels.length > 0
          ? panels[panels.length - 1].id
          : null
    const activePanel = panels.find((p) => p.id === activePanelId)

    set((state) => ({
      sessions,
      panels,
      canvasNodes: new Map(Object.entries(snap.positions)),
      canvasEdges: snap.edges,
      activePanelId,
      activeSessionId: activePanel?.terminalId ?? null,
      sessionsVersion: state.sessionsVersion + 1
    }))
    saveSessionMeta(get().sessions)
    persistPanels(get().panels, get().sessions)

    // First run (lone default workspace, nothing restored): open a terminal.
    // User-created empty workspaces stay empty by design.
    if (
      get().panels.length === 0 &&
      registry.workspaces.length === 1 &&
      registry.workspaces[0].id === DEFAULT_WORKSPACE_ID
    ) {
      get().createTerminal()
    }
  },

  setActiveTerminal: (id: string) => {
    set({ activeSessionId: id })
    // Find and set active panel
    const panel = get().panels.find((p) => p.terminalId === id)
    if (panel) {
      set({ activePanelId: panel.id })
    }
  },

  updateSession: (id: string, updates: Partial<TerminalSession>) => {
    set((state) => {
      const newSessions = new Map(state.sessions)
      const session = newSessions.get(id)
      if (!session) return state

      // Check which fields actually changed (significant updates that canvas should reflect)
      const statusChanged = updates.status !== undefined && updates.status !== session.status
      const toolChanged = updates.currentTool !== undefined && updates.currentTool !== session.currentTool
      const outputChanged = false // lastOutput no longer displayed in canvas nodes
      const attentionChanged = updates.attention !== undefined && updates.attention !== session.attention
      const taskSummaryChanged = updates.taskSummary !== undefined && updates.taskSummary !== session.taskSummary
      const agentKindChanged = 'agentKind' in updates && updates.agentKind !== session.agentKind
      const pausedChanged = 'isPaused' in updates && updates.isPaused !== session.isPaused
      const cwdChanged = updates.cwd !== undefined && updates.cwd !== session.cwd
      const errorChanged = 'errorMessage' in updates && updates.errorMessage !== session.errorMessage

      newSessions.set(id, { ...session, ...updates })

      // Increment version for any significant change to force canvas re-renders
      if (statusChanged || toolChanged || outputChanged || attentionChanged || taskSummaryChanged || agentKindChanged || pausedChanged || cwdChanged || errorChanged) {
        return { sessions: newSessions, sessionsVersion: state.sessionsVersion + 1 }
      }
      return { sessions: newSessions }
    })
  },

  addPanel: (panel: Omit<Panel, 'id'>) => {
    const id = generateId()
    const newPanel: Panel = { ...panel, id }

    set((state) => ({
      panels: [...state.panels, newPanel],
      activePanelId: id
    }))

    persistPanels(get().panels, get().sessions)
    return id
  },

  removePanel: (id: string) => {
    // Guard every panel-removal path (tab close, canvas node close, Ctrl+W hotkey)
    const target = get().panels.find((p) => p.id === id)
    if (target?.type === 'terminal' && target.terminalId) {
      if (!confirmCloseIfBusy(get().sessions.get(target.terminalId))) return
    }

    set((state) => {
      const panel = state.panels.find((p) => p.id === id)
      const newPanels = state.panels.filter((p) => p.id !== id)

      // Clean up persisted canvas position for this panel
      const newCanvasNodes = new Map(state.canvasNodes)
      newCanvasNodes.delete(id)
      saveCanvasPositions(newCanvasNodes)

      // If it's a terminal panel, close the terminal too
      if (panel?.type === 'terminal' && panel.terminalId) {
        get().closeTerminal(panel.terminalId)
        return { panels: newPanels, canvasNodes: newCanvasNodes }
      }

      return {
        panels: newPanels,
        canvasNodes: newCanvasNodes,
        activePanelId: state.activePanelId === id
          ? (newPanels.length > 0 ? newPanels[newPanels.length - 1].id : null)
          : state.activePanelId
      }
    })

    persistPanels(get().panels, get().sessions)
  },

  detachPanel: (panelId: string) => {
    const current = get()
    const panel = current.panels.find((p) => p.id === panelId)
    if (!panel) return null

    const stub = toPersistedPanel(panel, current.sessions)
    const position = current.canvasNodes.get(panelId)

    set((state) => {
      const newSessions = new Map(state.sessions)
      if (panel.terminalId) newSessions.delete(panel.terminalId)
      const newNodes = new Map(state.canvasNodes)
      newNodes.delete(panelId)
      const newPanels = state.panels.filter((p) => p.id !== panelId)
      const newEdges = state.canvasEdges.filter((e) => e.source !== panelId && e.target !== panelId)
      return {
        panels: newPanels,
        sessions: newSessions,
        canvasNodes: newNodes,
        canvasEdges: newEdges,
        activePanelId:
          state.activePanelId === panelId
            ? (newPanels.length > 0 ? newPanels[newPanels.length - 1].id : null)
            : state.activePanelId,
        activeSessionId:
          panel.terminalId && state.activeSessionId === panel.terminalId ? null : state.activeSessionId,
        sessionsVersion: state.sessionsVersion + 1
      }
    })

    saveCanvasPositions(get().canvasNodes)
    persistPanels(get().panels, get().sessions)
    return { stub, position }
  },

  setActivePanel: (id: string) => {
    set({ activePanelId: id })
    // If it's a terminal panel, set active session too
    const panel = get().panels.find((p) => p.id === id)
    if (panel?.type === 'terminal' && panel.terminalId) {
      set({ activeSessionId: panel.terminalId })
    }
    saveActivePanelFor(getActiveWorkspaceId(), id)
  },

  renamePanel: (id: string, title: string) => {
    set((state) => {
      const newPanels = state.panels.map((p) =>
        p.id === id ? { ...p, title } : p
      )

      // Also update the session title if it's a terminal
      const panel = state.panels.find((p) => p.id === id)
      if (panel?.type === 'terminal' && panel.terminalId) {
        const newSessions = new Map(state.sessions)
        const session = newSessions.get(panel.terminalId)
        if (session) {
          newSessions.set(panel.terminalId, { ...session, title })
        }
        saveSessionMeta(newSessions)
        return { panels: newPanels, sessions: newSessions }
      }

      return { panels: newPanels }
    })

    persistPanels(get().panels, get().sessions)
  },

  setViewMode: (mode: ViewMode) => {
    set({ viewMode: mode })
    saveViewMode(mode)
  },

  toggleViewMode: () => {
    set((state) => {
      const newMode = state.viewMode === 'grid' ? 'single' : 'grid'
      saveViewMode(newMode)
      return { viewMode: newMode }
    })
  },

  cycleViewMode: () => {
    set((state) => {
      const modes: ViewMode[] = ['grid', 'single', 'canvas']
      const currentIndex = modes.indexOf(state.viewMode)
      const newMode = modes[(currentIndex + 1) % modes.length]
      saveViewMode(newMode)
      return { viewMode: newMode }
    })
  },

  reorderPanels: (sourceId: string, targetId: string) => {
    if (sourceId === targetId) return

    set((state) => {
      const sourceIndex = state.panels.findIndex(p => p.id === sourceId)
      const targetIndex = state.panels.findIndex(p => p.id === targetId)

      if (sourceIndex === -1 || targetIndex === -1) return state

      const newPanels = [...state.panels]
      const [removed] = newPanels.splice(sourceIndex, 1)
      newPanels.splice(targetIndex, 0, removed)

      return { panels: newPanels }
    })
    persistPanels(get().panels, get().sessions)
  },

  updateCanvasNode: (panelId: string, position: { x: number; y: number }) => {
    set((state) => {
      const panel = state.panels.find(p => p.id === panelId)
      if (!panel) return state

      const session = panel.terminalId ? state.sessions.get(panel.terminalId) : undefined

      const newNodes = new Map(state.canvasNodes)
      newNodes.set(panelId, {
        id: panelId,
        panelId,
        position,
        data: {
          label: panel.title,
          status: session?.attention || 'none',
          gitBranch: session?.gitBranch,
          model: session?.model
        }
      })

      saveCanvasPositions(newNodes)
      return { canvasNodes: newNodes }
    })
  },

  setCanvasEdges: (edges: CanvasEdge[]) => {
    set({ canvasEdges: edges })
  },

  setDraggedPanel: (id: string | null) => {
    set({ draggedPanelId: id })
  },

  setBroadcastCommand: (command: string) => {
    set({ broadcastCommand: command })
  },

  executeBroadcast: () => {
    const { broadcastCommand, sessions } = get()
    if (!broadcastCommand.trim()) return

    // Send command to all terminals
    sessions.forEach((_, id) => {
      window.bashgym?.terminal.write(id, broadcastCommand + '\r')
    })

    set({ broadcastCommand: '' })
  },

  setPanelUrl: (id: string, url: string) => {
    set((state) => ({
      panels: state.panels.map(p => p.id === id ? { ...p, url } : p)
    }))
  },

  setPanelThumbnail: (id: string, thumbnail: string) => {
    set((state) => ({
      panels: state.panels.map(p => p.id === id ? { ...p, thumbnail } : p)
    }))
  },

  updatePanelConfig: (panelId: string, adapterConfig: Record<string, unknown>) => {
    set((state) => ({
      panels: state.panels.map(p =>
        p.id === panelId ? { ...p, adapterConfig } : p
      )
    }))
    persistPanels(get().panels, get().sessions)
  }
}))
