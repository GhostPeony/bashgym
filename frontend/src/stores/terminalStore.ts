import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export type PanelType = 'terminal' | 'preview' | 'browser' | 'files'
export type AttentionState = 'none' | 'waiting' | 'success' | 'error'
export type ViewMode = 'grid' | 'single' | 'canvas'
export type AgentStatus = 'running' | 'idle' | 'waiting_input' | 'tool_calling'

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

  // Drag state for tab reordering
  draggedPanelId: string | null

  // Broadcast
  broadcastCommand: string

  // Actions
  createTerminal: (id?: string, title?: string) => string
  closeTerminal: (id: string) => void
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

  // Drag actions
  setDraggedPanel: (id: string | null) => void

  setBroadcastCommand: (command: string) => void
  executeBroadcast: () => void
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

const loadPanelOrder = (): string[] => {
  try {
    const stored = localStorage.getItem(PANEL_ORDER_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

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
    if (stored && ['grid', 'single'].includes(stored)) {
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

export const useTerminalStore = create<TerminalState>((set, get) => ({
  sessions: new Map(),
  activeSessionId: null,
  sessionsVersion: 0,
  panels: [],
  activePanelId: null,
  viewMode: loadViewMode(),
  canvasNodes: loadCanvasPositions(),
  draggedPanelId: null,
  broadcastCommand: '',

  createTerminal: (id?: string, title?: string) => {
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
      lastActivity: Date.now()
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

    // Kill the terminal process
    window.bashgym?.terminal.kill(id)
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

      // Check if status or currentTool actually changed (significant updates)
      const statusChanged = updates.status !== undefined && updates.status !== session.status
      const toolChanged = updates.currentTool !== undefined && updates.currentTool !== session.currentTool

      newSessions.set(id, { ...session, ...updates })

      // Only increment version for significant changes (status/tool) to avoid excessive re-renders
      if (statusChanged || toolChanged) {
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
    set((state) => {
      const panel = state.panels.find((p) => p.id === id)
      const newPanels = state.panels.filter((p) => p.id !== id)

      // If it's a terminal panel, close the terminal too
      if (panel?.type === 'terminal' && panel.terminalId) {
        get().closeTerminal(panel.terminalId)
        return { panels: newPanels }
      }

      return {
        panels: newPanels,
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
  }
}))
