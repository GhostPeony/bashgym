import { create } from 'zustand'
// persist and createJSONStorage available if needed for future state persistence
// import { persist, createJSONStorage } from 'zustand/middleware'

export type PanelType = 'terminal' | 'preview' | 'browser' | 'files' | 'context' | 'neon' | 'vercel' | 'activity' | 'training' | 'evals'
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

// Canvas edge — a connection between two panels on the canvas
export interface CanvasEdge {
  id: string
  source: string  // panelId of source node
  target: string  // panelId of target node
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
  createTerminal: (id?: string, title?: string, launchCommand?: string) => string
  closeTerminal: (id: string) => void
  /** Close with confirmation if the agent is busy. Returns true if closed. */
  requestCloseTerminal: (id: string) => boolean
  /** Re-adopt live PTY sessions from the main process (call once on startup). Returns count restored. */
  restoreSessions: () => Promise<number>
  setActiveTerminal: (id: string) => void
  updateSession: (id: string, updates: Partial<TerminalSession>) => void

  addPanel: (panel: Omit<Panel, 'id'>) => string
  removePanel: (id: string) => void
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

// Persist panel order to localStorage
const PANEL_ORDER_KEY = 'bashgym_panel_order'
const CANVAS_POSITIONS_KEY = 'bashgym_canvas_positions'
const VIEW_MODE_KEY = 'bashgym_view_mode'

const savePanelOrder = (panels: Panel[]) => {
  try {
    localStorage.setItem(PANEL_ORDER_KEY, JSON.stringify(panels.map(p => p.id)))
  } catch {
    // Ignore storage errors
  }
}

const loadCanvasPositions = (): Map<string, CanvasNode> => {
  try {
    const stored = localStorage.getItem(CANVAS_POSITIONS_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      return new Map(Object.entries(parsed))
    }
  } catch {
    // Ignore
  }
  return new Map()
}

const saveCanvasPositions = (nodes: Map<string, CanvasNode>) => {
  try {
    const obj = Object.fromEntries(nodes)
    localStorage.setItem(CANVAS_POSITIONS_KEY, JSON.stringify(obj))
  } catch {
    // Ignore storage errors
  }
}

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

  createTerminal: (id?: string, title?: string, launchCommand?: string) => {
    const terminalId = id || generateId()

    // Show banner only if it hasn't been shown yet in this session
    const showBanner = !getBannerShown()
    if (showBanner) {
      setBannerShown()
    }

    const session: TerminalSession = {
      id: terminalId,
      title: title || `Terminal ${get().sessions.size + 1}`,
      cwd: '~',
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

    // Kill the terminal process
    window.bashgym?.terminal.kill(id)
  },

  requestCloseTerminal: (id: string) => {
    if (!confirmCloseIfBusy(get().sessions.get(id))) return false
    get().closeTerminal(id)
    return true
  },

  restoreSessions: async () => {
    const api = window.bashgym?.terminal
    if (!api?.list) return 0
    let infos: Awaited<ReturnType<typeof api.list>> = []
    try {
      infos = await api.list()
    } catch {
      return 0
    }
    const meta = loadSessionMeta()
    const live = infos.filter((i) => !i.exited && !get().sessions.has(i.id))
    if (live.length === 0) return 0

    set((state) => {
      const newSessions = new Map(state.sessions)
      const newPanels = [...state.panels]
      for (const info of live) {
        const title = meta.get(info.id)?.title || `Terminal ${newSessions.size + 1}`
        newSessions.set(info.id, {
          id: info.id,
          title,
          cwd: info.cwd,
          isActive: false,
          attention: 'none',
          showBanner: false,
          status: 'idle',
          lastActivity: Date.now()
        })
        newPanels.push({ id: generateId(), type: 'terminal', title, terminalId: info.id })
      }
      const lastPanel = newPanels[newPanels.length - 1]
      return {
        sessions: newSessions,
        panels: newPanels,
        activeSessionId: state.activeSessionId ?? live[live.length - 1].id,
        activePanelId: state.activePanelId ?? lastPanel.id
      }
    })
    saveSessionMeta(get().sessions)
    return live.length
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

      newSessions.set(id, { ...session, ...updates })

      // Increment version for any significant change to force canvas re-renders
      if (statusChanged || toolChanged || outputChanged || attentionChanged || taskSummaryChanged || agentKindChanged) {
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
  },

  setActivePanel: (id: string) => {
    set({ activePanelId: id })
    // If it's a terminal panel, set active session too
    const panel = get().panels.find((p) => p.id === id)
    if (panel?.type === 'terminal' && panel.terminalId) {
      set({ activeSessionId: panel.terminalId })
    }
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

      savePanelOrder(newPanels)
      return { panels: newPanels }
    })
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
  }
}))
