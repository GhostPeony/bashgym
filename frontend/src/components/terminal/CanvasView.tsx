import { useCallback, useEffect, useState, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  Connection,
  Node,
  Edge,
  NodeChange,
  BackgroundVariant,
  Panel as FlowPanel,
  Viewport,
  SelectionMode
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useTerminalStore, useCanvasControlStore } from '../../stores'
import type { Panel, TerminalSession, CanvasEdge, MonitorAutoMode } from '../../stores'
import { getActiveWorkspaceId, wsKey } from '../../stores/workspacePersistence'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import { getMonitorInfo, isMonitorEdge, sendMonitorSnapshot } from '../../utils/monitorRouting'
import { applyPreset, type CanvasPreset } from './canvasPresets'
import { TerminalNode, type TerminalNodeData } from './TerminalNode'
import { MonitorEdge, type MonitorEdgeData } from './edges/MonitorEdge'
import { PreviewNode, type PreviewNodeData } from './PreviewNode'
import { BrowserNode, type BrowserNodeData } from './BrowserNode'
import { IntegrationNode } from './nodes/IntegrationNode'
import { ActivityFeedNode } from './nodes/ActivityFeedNode'
import { TrainingRunNode } from './nodes/TrainingRunNode'
import { EvalNode } from './nodes/EvalNode'
import { DataDesignerNode } from './nodes/DataDesignerNode'
import { HFInventoryNode } from './nodes/HFInventoryNode'
import { AgentEndpointNode } from './nodes/AgentEndpointNode'
import { ToolKitNode } from './nodes/ToolKitNode'
import { DATA_NODE_TYPES } from './nodes/dataPanels'
import type { IntegrationNodeData, DataNodeData } from './nodes/types'
import { workspaceApi } from '../../services/api'
// Import adapters to trigger registration side effects
import './nodes/adapters/context'
import './nodes/adapters/neon'
import './nodes/adapters/vercel'
import { MasterControlPanel } from './MasterControlPanel'
import { useCanvasHotkeys } from '../../hooks/useCanvasHotkeys'
import { AlertCircle } from 'lucide-react'

const EDGE_STYLE = { stroke: 'var(--accent)', strokeWidth: 2 }

// Edges persist as {id, source, target} plus type/data for monitor edges;
// style/animation re-applied on load (monitor edges style themselves)
type PersistedEdge = {
  id: string
  source: string
  target: string
  type?: 'monitor'
  data?: { auto?: MonitorAutoMode }
}

const toPersistedEdge = (e: Edge): PersistedEdge => ({
  id: e.id,
  source: e.source,
  target: e.target,
  ...(e.type === 'monitor'
    ? { type: 'monitor' as const, data: { auto: (e.data as MonitorEdgeData | undefined)?.auto ?? 'off' } }
    : {})
})

const toFlowEdge = (e: PersistedEdge): Edge =>
  e.type === 'monitor'
    ? {
        id: e.id,
        source: e.source,
        target: e.target,
        type: 'monitor' as const,
        data: { auto: e.data?.auto ?? 'off' }
      }
    : {
        id: e.id,
        source: e.source,
        target: e.target,
        animated: true,
        style: EDGE_STYLE,
        className: 'canvas-edge-pulse'
      }

// Edges/viewport are stored per workspace. All load/save calls take the
// workspace id captured at mount (wsIdRef) so a component unmounting during a
// workspace switch can only ever write to its OWN workspace's keys.
const loadEdges = (wsId: string): Edge[] => {
  try {
    const stored = localStorage.getItem(wsKey(wsId, 'edges'))
    if (stored) {
      const parsed: PersistedEdge[] = JSON.parse(stored)
      return parsed.map(toFlowEdge)
    }
  } catch {
    // Ignore
  }
  return []
}

const saveEdges = (wsId: string, edges: Edge[]) => {
  try {
    localStorage.setItem(wsKey(wsId, 'edges'), JSON.stringify(edges.map(toPersistedEdge)))
  } catch {
    // Ignore storage errors
  }
}

// Handler bundle injected into monitor edge data at runtime (never persisted)
interface MonitorHandlers {
  onSendSnapshot: (edgeId: string) => Promise<{ sent: boolean; error?: string }>
  onCycleAuto: (edgeId: string) => void
  onSwapDirection: (edgeId: string) => void
}

const cycleAutoMode = (mode?: MonitorAutoMode): MonitorAutoMode =>
  mode === 'prefill' ? 'send' : mode === 'send' ? 'off' : 'prefill'

// Terminal→terminal edges are monitor edges: ensure type + data with live handlers.
// Idempotent — returns the same array when nothing needs upgrading.
function decorateMonitorEdges(edges: Edge[], panels: Panel[], handlers: MonitorHandlers): Edge[] {
  let changed = false
  const next = edges.map((e) => {
    if (!isMonitorEdge(e, panels)) return e
    const data = e.data as MonitorEdgeData | undefined
    if (e.type === 'monitor' && data?.onSendSnapshot === handlers.onSendSnapshot) return e
    changed = true
    return {
      ...e,
      type: 'monitor' as const,
      animated: false,
      style: undefined,
      data: { auto: data?.auto ?? 'off', ...handlers }
    }
  })
  return changed ? next : edges
}

// Union of all node data shapes rendered on the canvas
type CanvasNodeData = TerminalNodeData | PreviewNodeData | BrowserNodeData | IntegrationNodeData | DataNodeData
type CanvasFlowNode = Node<CanvasNodeData>

// Load saved viewport (per workspace)
const loadViewport = (wsId: string): Viewport | null => {
  try {
    const stored = localStorage.getItem(wsKey(wsId, 'viewport'))
    if (stored) {
      return JSON.parse(stored)
    }
  } catch {
    // Ignore
  }
  return null
}

// Save viewport (per workspace)
const saveViewport = (wsId: string, viewport: Viewport) => {
  try {
    localStorage.setItem(wsKey(wsId, 'viewport'), JSON.stringify(viewport))
  } catch {
    // Ignore
  }
}

const SECRET_KEY_RE = /(api[_-]?key|authorization|bearer|password|secret|token)/i

function sanitizeConfig(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitizeConfig)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entry]) => [
        key,
        SECRET_KEY_RE.test(key) ? '[redacted]' : sanitizeConfig(entry)
      ])
    )
  }
  if (typeof value === 'string' && /^(sk-|ghp_|hf_|xox[bp]-)/.test(value)) return '[redacted]'
  return value
}

function buildWorkspaceSnapshot(state: ReturnType<typeof useTerminalStore.getState>) {
  const panels = state.panels.map((panel) => ({
    panel_id: panel.id,
    type: panel.type,
    title: panel.title,
    terminal_id: panel.terminalId,
    adapter_config: sanitizeConfig(panel.adapterConfig || {}) as Record<string, unknown>,
    visible: true
  }))
  const terminals = Array.from(state.sessions.values()).map((session) => {
    const panel = state.panels.find((candidate) => candidate.terminalId === session.id)
    return {
      terminal_id: session.id,
      panel_id: panel?.id,
      title: session.title,
      cwd: session.cwd,
      agent_kind: session.agentKind,
      status: session.status,
      current_tool: session.currentTool
    }
  })
  const wsId = getActiveWorkspaceId()
  const wsName = useWorkspaceStore.getState().workspaces.find((w) => w.id === wsId)?.name
  return {
    schema_version: 'bashgym.workspace.canvas.v1',
    updated_at: new Date().toISOString(),
    workspace_id: wsId,
    workspace_name: wsName,
    panels,
    edges: state.canvasEdges,
    terminals,
    data_summaries: {
      panel_count: panels.length,
      edge_count: state.canvasEdges.length,
      terminal_count: terminals.length
    },
    allowed_actions: [
      'workspace.context.read',
      'workspace.canvas.intent.emit',
      'training.start',
      'terminal.snapshot.send'
    ]
  }
}

// Register all node types
const nodeTypes = {
  terminal: TerminalNode,
  preview: PreviewNode,
  browser: BrowserNode,
  context: IntegrationNode,
  neon: IntegrationNode,
  vercel: IntegrationNode,
  activity: ActivityFeedNode,
  training: TrainingRunNode,
  evals: EvalNode,
  designer: DataDesignerNode,
  huggingface: HFInventoryNode,
  agent: AgentEndpointNode,
  toolkit: ToolKitNode,
}

// Register custom edge types
const edgeTypes = {
  monitor: MonitorEdge,
}

export interface CanvasViewProps {
  onFocusPanel: (panelId: string) => void
  onClosePopup?: () => void
}

// Build node data for a panel — single source of truth, avoids duplication between init and effect
function buildNodeData(
  panel: Panel,
  sessions: Map<string, TerminalSession>,
  onFocus: (id: string) => void,
  onClose: (id: string) => void,
  canvasEdges: CanvasEdge[] = [],
  panels: Panel[] = []
): TerminalNodeData | PreviewNodeData | BrowserNodeData | IntegrationNodeData | DataNodeData {
  const session = panel.terminalId ? sessions.get(panel.terminalId) : undefined
  const hasConnections = canvasEdges.some(e => e.source === panel.id || e.target === panel.id)

  if (panel.type === 'context' || panel.type === 'neon' || panel.type === 'vercel') {
    return {
      panelId: panel.id,
      title: panel.title,
      adapterType: panel.type,
      adapterConfig: { ...panel.adapterConfig, _panelId: panel.id },
      hasConnections,
      onFocus,
      onClose,
    } as IntegrationNodeData
  } else if (DATA_NODE_TYPES.includes(panel.type)) {
    return {
      panelId: panel.id,
      title: panel.title,
      adapterConfig: panel.adapterConfig,
      hasConnections,
      onFocus,
      onClose,
    } as DataNodeData
  } else if (panel.type === 'preview') {
    return {
      panelId: panel.id,
      filePath: panel.filePath || '',
      onFocus,
      onClose,
      onCopyPath: (path: string) => {
        if (window.bashgym?.clipboard?.writeText) {
          window.bashgym.clipboard.writeText(path)
        } else {
          navigator.clipboard.writeText(path)
        }
      }
    } as PreviewNodeData
  } else if (panel.type === 'browser') {
    return {
      panelId: panel.id,
      url: panel.url || 'http://localhost:3000',
      thumbnailUrl: panel.thumbnail,
      hasConnections,
      onFocus,
      onClose,
      onCopyUrl: (url: string) => {
        if (window.bashgym?.clipboard?.writeText) {
          window.bashgym.clipboard.writeText(url)
        } else {
          navigator.clipboard.writeText(url)
        }
      },
      onOpenExternal: (url: string) => {
        window.open(url, '_blank')
      }
    } as BrowserNodeData
  } else {
    const monitor = getMonitorInfo(panel.id, canvasEdges, panels)
    // Link2 chip stays about data-node routing; monitor edges get their own chips
    const dataConnections = canvasEdges.some(
      e => (e.source === panel.id || e.target === panel.id) && !isMonitorEdge(e, panels)
    )
    return {
      panelId: panel.id,
      title: panel.title,
      type: panel.type,
      status: session?.status,
      attention: session?.attention,
      gitBranch: session?.gitBranch,
      model: session?.model,
      agentKind: session?.agentKind,
      currentTool: session?.currentTool,
      cwd: session?.cwd,
      lastActivity: session?.lastActivity,
      taskSummary: session?.taskSummary,
      toolHistory: session?.toolHistory,
      metrics: session?.metrics,
      recentFiles: session?.recentFiles,
      errorMessage: session?.errorMessage,
      isExpanded: session?.isExpanded,
      isPaused: session?.isPaused,
      hasConnections: dataConnections,
      isWatched: monitor.isWatched,
      watchingTitle: monitor.watchingTitle,
      onFocus,
      onClose
    } as TerminalNodeData
  }
}

function CanvasViewInner({ onFocusPanel, onClosePopup }: CanvasViewProps) {
  const {
    panels: _panels,
    sessions: _sessions,
    sessionsVersion: _sessionsVersion,
    activePanelId: _activePanelId,
    canvasNodes,
    updateCanvasNode,
    setActivePanel,
    createTerminal,
    setCanvasEdges
  } = useTerminalStore()

  const {
    gridEnabled,
    snapToGrid,
    showMiniMap
  } = useCanvasControlStore()

  const { fitView, zoomIn, zoomOut, getZoom, getNodes } = useReactFlow()
  const [currentZoom, setCurrentZoom] = useState(1)

  // Stable callbacks — consistent references across renders so TerminalNode memo stays effective
  const handleFocusPanel = useCallback((id: string) => {
    setActivePanel(id)
    onFocusPanel(id)
  }, [setActivePanel, onFocusPanel])

  const handleClosePanel = useCallback((id: string) => {
    // Close the popup overlay if open, then remove the panel from the canvas
    onClosePopup?.()
    useTerminalStore.getState().removePanel(id)
  }, [onClosePopup])

  // Track canvasNodes via ref so the update effect can read current saved positions for new
  // panels without re-running every time a drag updates canvasNodes
  const canvasNodesRef = useRef(canvasNodes)
  canvasNodesRef.current = canvasNodes
  const snapshotTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // Workspace this component instance belongs to — captured at mount. The
  // component remounts (keyed) on workspace switch, so this never goes stale,
  // and unmount-straggler effects can't write into another workspace's keys.
  const wsIdRef = useRef(getActiveWorkspaceId())

  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasFlowNode>([])
  const initialEdgesRef = useRef<Edge[] | null>(null)
  if (initialEdgesRef.current === null) {
    initialEdgesRef.current = loadEdges(wsIdRef.current)
  }
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdgesRef.current)

  // Sync React Flow edges to Zustand store for cross-component routing (e.g. BrowserPane
  // screenshot routing, monitor snapshots) and persist them across reloads
  useEffect(() => {
    setCanvasEdges(edges.map(toPersistedEdge))
    saveEdges(wsIdRef.current, edges)
  }, [edges, setCanvasEdges])

  // Monitor edge actions — stable references so edge data survives memo comparison
  const handleSendSnapshot = useCallback(
    (edgeId: string) => sendMonitorSnapshot(edgeId, false),
    []
  )
  const handleCycleAuto = useCallback((edgeId: string) => {
    setEdges((eds) => eds.map((e) =>
      e.id === edgeId
        ? { ...e, data: { ...e.data, auto: cycleAutoMode((e.data as MonitorEdgeData | undefined)?.auto) } }
        : e
    ))
  }, [setEdges])
  const handleSwapDirection = useCallback((edgeId: string) => {
    setEdges((eds) => eds.map((e) =>
      e.id === edgeId
        ? { ...e, source: e.target, target: e.source, sourceHandle: null, targetHandle: null }
        : e
    ))
  }, [setEdges])
  const monitorHandlersRef = useRef<MonitorHandlers | null>(null)
  if (!monitorHandlersRef.current) {
    monitorHandlersRef.current = {
      onSendSnapshot: handleSendSnapshot,
      onCycleAuto: handleCycleAuto,
      onSwapDirection: handleSwapDirection
    }
  }
  const monitorHandlers = monitorHandlersRef.current

  // Orchestrated graph changes write to terminalStore.canvasEdges. Reconcile
  // those back into React Flow while leaving manual React Flow changes alone.
  useEffect(() => {
    const unsubscribe = useTerminalStore.subscribe((state, prev) => {
      if (state.canvasEdges === prev.canvasEdges) return
      setEdges((current) => {
        const currentShape = JSON.stringify(current.map(toPersistedEdge))
        const nextShape = JSON.stringify(state.canvasEdges)
        if (currentShape === nextShape) return current
        return decorateMonitorEdges(state.canvasEdges.map(toFlowEdge), state.panels, monitorHandlers)
      })
    })
    return unsubscribe
  }, [setEdges, monitorHandlers])

  // Upgrade terminal↔terminal edges to monitor edges (covers edges loaded from
  // storage before panels restored, and legacy saved edges)
  useEffect(() => {
    setEdges((eds) => decorateMonitorEdges(eds, useTerminalStore.getState().panels, monitorHandlers))
  }, [_panels, setEdges, monitorHandlers])

  // Use stable refs for callbacks so the Zustand subscription never needs to be recreated
  const handleFocusPanelRef = useRef(handleFocusPanel)
  const handleClosePanelRef = useRef(handleClosePanel)
  handleFocusPanelRef.current = handleFocusPanel
  handleClosePanelRef.current = handleClosePanel

  // Direct Zustand subscription — bypasses React's dep comparison and batching.
  // Fires immediately whenever any display-relevant store state changes.
  useEffect(() => {
    const rebuildNodes = (state: ReturnType<typeof useTerminalStore.getState>) => {
      setNodes((prevNodes) => {
        const positionMap = new Map(prevNodes.map(n => [n.id, n.position]))

        return state.panels.map((panel, index) => {
          const savedPosition = positionMap.get(panel.id)
          const persistedPosition = canvasNodesRef.current.get(panel.id)?.position
          const defaultPosition = {
            x: 50 + (index % 3) * 380,
            y: 50 + Math.floor(index / 3) * 280
          }
          const integrationTypes = ['context', 'neon', 'vercel'] as const
          const nodeType = panel.type === 'preview' ? 'preview' :
                           panel.type === 'browser' ? 'browser' :
                           integrationTypes.includes(panel.type as any) ? panel.type :
                           DATA_NODE_TYPES.includes(panel.type) ? panel.type : 'terminal'
          return {
            id: panel.id,
            type: nodeType,
            position: savedPosition || persistedPosition || defaultPosition,
            selected: panel.id === state.activePanelId,
            data: buildNodeData(panel, state.sessions, handleFocusPanelRef.current, handleClosePanelRef.current, state.canvasEdges, state.panels)
          }
        })
      })
    }

    // Run immediately on mount with current state
    rebuildNodes(useTerminalStore.getState())

    // Subscribe to all subsequent changes. Deliberately NOT comparing the sessions
    // Map reference — updateSession reallocates it on every 80ms output flush;
    // sessionsVersion only bumps on render-relevant changes (status/tool/attention/
    // taskSummary/agentKind/isPaused/cwd/errorMessage), so keying on it keeps the
    // canvas from rebuilding all nodes while a terminal is merely streaming output.
    const unsubscribe = useTerminalStore.subscribe((state, prev) => {
      if (
        state.panels === prev.panels &&
        state.sessionsVersion === prev.sessionsVersion &&
        state.activePanelId === prev.activePanelId &&
        state.canvasEdges === prev.canvasEdges
      ) return
      rebuildNodes(state)
    })

    return unsubscribe
  }, [setNodes]) // setNodes is stable; callbacks accessed via ref

  // Sync a sanitized live canvas snapshot for terminal/Hermes agents. Throttled
  // and keyed on render-relevant state so terminal output streaming does not
  // hammer the backend.
  useEffect(() => {
    const pushSnapshot = () => {
      if (snapshotTimerRef.current) clearTimeout(snapshotTimerRef.current)
      snapshotTimerRef.current = setTimeout(() => {
        void workspaceApi.putCanvasSnapshot(buildWorkspaceSnapshot(useTerminalStore.getState()))
      }, 800)
    }

    pushSnapshot()
    const unsubscribe = useTerminalStore.subscribe((state, prev) => {
      if (
        state.panels === prev.panels &&
        state.sessionsVersion === prev.sessionsVersion &&
        state.canvasEdges === prev.canvasEdges &&
        state.canvasNodes === prev.canvasNodes
      ) return
      pushSnapshot()
    })
    return () => {
      if (snapshotTimerRef.current) clearTimeout(snapshotTimerRef.current)
      unsubscribe()
    }
  }, [])

  // Handle node position changes
  const handleNodesChange = useCallback((changes: NodeChange<CanvasFlowNode>[]) => {
    onNodesChange(changes)

    // Save position changes
    changes.forEach((change) => {
      if (change.type === 'position' && change.position && !change.dragging) {
        updateCanvasNode(change.id, change.position)
      }
    })
  }, [onNodesChange, updateCanvasNode])

  // Handle connections between nodes. Terminal→terminal = monitor edge
  // (source = watched, target = watcher)
  const onConnect = useCallback((connection: Connection) => {
    const panels = useTerminalStore.getState().panels
    const isTerminal = (pid: string | null) =>
      !!pid && panels.some((p) => p.id === pid && p.type === 'terminal')
    if (isTerminal(connection.source) && isTerminal(connection.target) && connection.source !== connection.target) {
      setEdges((eds) => addEdge({
        ...connection,
        type: 'monitor',
        data: { auto: 'off', ...monitorHandlers }
      }, eds))
      return
    }
    setEdges((eds) => addEdge({
      ...connection,
      animated: true,
      style: { stroke: 'var(--accent)', strokeWidth: 2 },
      className: 'canvas-edge-pulse'
    }, eds))
  }, [setEdges, monitorHandlers])

  // Auto-connect all nodes when shift+drag box-selects 2+ nodes
  const onSelectionEnd = useCallback(() => {
    const selectedNodes = getNodes().filter(n => n.selected)
    if (selectedNodes.length < 2) return

    setEdges((eds) => {
      let updated = eds
      // Create edges between every pair of selected nodes (skip if already connected)
      for (let i = 0; i < selectedNodes.length; i++) {
        for (let j = i + 1; j < selectedNodes.length; j++) {
          const a = selectedNodes[i].id
          const b = selectedNodes[j].id
          const exists = updated.some(
            e => (e.source === a && e.target === b) || (e.source === b && e.target === a)
          )
          if (!exists) {
            updated = addEdge({
              source: a,
              target: b,
              sourceHandle: null,
              targetHandle: null,
              animated: true,
              style: { stroke: 'var(--accent)', strokeWidth: 2 }
            }, updated)
          }
        }
      }
      // Terminal↔terminal pairs become monitor edges
      return decorateMonitorEdges(updated, useTerminalStore.getState().panels, monitorHandlers)
    })
  }, [getNodes, setEdges, monitorHandlers])

  // Handle node selection
  const onNodeClick = useCallback((_: any, node: Node) => {
    setActivePanel(node.id)
  }, [setActivePanel])

  // Load saved viewport or use default
  const savedViewport = useRef(loadViewport(wsIdRef.current))
  const shouldFitView = !savedViewport.current

  // Save viewport on move/zoom
  const onMoveEnd = useCallback((_: any, viewport: Viewport) => {
    saveViewport(wsIdRef.current, viewport)
    setCurrentZoom(viewport.zoom)
  }, [])

  // Control panel handlers
  const handleZoomIn = useCallback(() => {
    zoomIn()
    setCurrentZoom(getZoom())
  }, [zoomIn, getZoom])

  const handleZoomOut = useCallback(() => {
    zoomOut()
    setCurrentZoom(getZoom())
  }, [zoomOut, getZoom])

  const handleFitView = useCallback(() => {
    fitView({ padding: 0.2 })
    setTimeout(() => setCurrentZoom(getZoom()), 100)
  }, [fitView, getZoom])

  // Canvas keyboard shortcuts (1-9 focus, Tab cycle, F fit, G grid, M minimap, Space pause)
  useCanvasHotkeys({ enabled: true, onFitView: handleFitView, onFocusPanel })

  const handleAutoArrange = useCallback(() => {
    // Auto-arrange nodes in a grid
    setNodes((nodes) => {
      return nodes.map((node, index) => ({
        ...node,
        position: {
          x: 50 + (index % 3) * 380,
          y: 50 + Math.floor(index / 3) * 280
        }
      }))
    })
    setTimeout(() => {
      fitView({ padding: 0.2 })
      setCurrentZoom(getZoom())
    }, 100)
  }, [setNodes, fitView, getZoom])

  const handleNewSession = useCallback(() => {
    createTerminal()
  }, [createTerminal])

  const handleLaunchClaude = useCallback(() => {
    createTerminal(undefined, 'Claude Code', 'claude')
  }, [createTerminal])

  const handleLaunchCodex = useCallback(() => {
    createTerminal(undefined, 'Codex', 'codex')
  }, [createTerminal])

  const addIntegrationPanel = useCallback((type: 'context' | 'neon' | 'vercel', title: string) => {
    const { addPanel } = useTerminalStore.getState()
    addPanel({ type, title, adapterConfig: {} })
  }, [])

  const addDataPanel = useCallback((type: Panel['type'], title: string) => {
    const { addPanel } = useTerminalStore.getState()
    addPanel({ type, title })
  }, [])

  // Spawn a saved/built-in workspace preset: panels at their positions plus edges
  const handleApplyPreset = useCallback((preset: CanvasPreset) => {
    const setPosition = (panelId: string, pos: { x: number; y: number }) => {
      updateCanvasNode(panelId, pos)
      setNodes((nds) => nds.map((n) => (n.id === panelId ? { ...n, position: pos } : n)))
    }
    const pairs = applyPreset(preset, setPosition)
    setEdges((eds) => {
      let updated = eds
      for (const [source, target] of pairs) {
        const exists = updated.some(
          (e) => (e.source === source && e.target === target) || (e.source === target && e.target === source)
        )
        if (!exists) {
          updated = addEdge(
            { source, target, sourceHandle: null, targetHandle: null, animated: true, style: EDGE_STYLE },
            updated
          )
        }
      }
      return decorateMonitorEdges(updated, useTerminalStore.getState().panels, monitorHandlers)
    })
    setTimeout(() => {
      fitView({ padding: 0.2 })
      setCurrentZoom(getZoom())
    }, 150)
  }, [setEdges, setNodes, updateCanvasNode, fitView, getZoom, monitorHandlers])

  return (
    <div className={`h-full w-full relative ${currentZoom < 0.55 ? 'canvas-compact' : ''}`}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onSelectionEnd={onSelectionEnd}
        onMoveEnd={onMoveEnd}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        panActivationKeyCode={null}
        fitView={shouldFitView}
        fitViewOptions={{
          padding: 0.2
        }}
        selectionMode={SelectionMode.Partial}
        snapToGrid={snapToGrid}
        snapGrid={[20, 20]}
        minZoom={0.1}
        maxZoom={2}
        defaultViewport={savedViewport.current || { x: 0, y: 0, zoom: 1 }}
        proOptions={{ hideAttribution: true }}
        className="bg-background-primary"
      >
        {gridEnabled && (
          <Background
            variant={BackgroundVariant.Dots}
            gap={20}
            size={1}
            color="var(--color-border-subtle)"
          />
        )}
        <Controls
          className="!bg-background-card !border-brutal !border-border !shadow-brutal-sm [&>button]:!bg-background-card [&>button]:!border-brutal [&>button]:!border-border [&>button]:!text-text-primary [&>button:hover]:!bg-background-secondary [&>button>svg]:!fill-text-primary"
          showZoom
          showFitView
          showInteractive={false}
        />
        {showMiniMap && (
          <MiniMap
            className="!bg-background-card !border-brutal !border-border !shadow-brutal-sm"
            nodeColor={(node) => {
              const data = node.data as TerminalNodeData
              if (data.attention === 'error') return 'var(--status-error)'
              if (data.attention === 'success') return 'var(--status-success)'
              if (data.status === 'running') return 'var(--accent)'
              return 'var(--text-muted)'
            }}
            maskColor="rgba(0, 0, 0, 0.5)"
          />
        )}
        <FlowPanel position="bottom-center" className="text-xs font-mono text-text-muted bg-background-card border-brutal border-border shadow-brutal-sm px-2 py-1 rounded-brutal">
          Drag to reposition &middot; Shift+drag to select &amp; connect
        </FlowPanel>
      </ReactFlow>

      {/* Master Control Panel */}
      <MasterControlPanel
        currentZoom={currentZoom}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onFitView={handleFitView}
        onAutoArrange={handleAutoArrange}
        onNewSession={handleNewSession}
        onLaunchClaude={handleLaunchClaude}
        onLaunchCodex={handleLaunchCodex}
        onAddContext={() => addIntegrationPanel('context', 'Context')}
        onAddNeon={() => addIntegrationPanel('neon', 'Neon DB')}
        onAddVercel={() => addIntegrationPanel('vercel', 'Vercel')}
        onAddDataPanel={addDataPanel}
        onApplyPreset={handleApplyPreset}
      />
    </div>
  )
}

export function CanvasView({ onFocusPanel, onClosePopup }: CanvasViewProps) {
  return (
    <ReactFlowProvider>
      <CanvasViewInner onFocusPanel={onFocusPanel} onClosePopup={onClosePopup} />
    </ReactFlowProvider>
  )
}

// Wrapper component with error boundary
export function CanvasViewWrapper({ onFocusPanel, onClosePopup }: CanvasViewProps) {
  const [hasError, setHasError] = useState(false)

  if (hasError) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-background-primary">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 border-brutal border-status-error rounded-brutal flex items-center justify-center bg-background-secondary">
            <AlertCircle className="w-8 h-8 text-status-error" />
          </div>
          <p className="text-sm text-status-error font-mono">Failed to load canvas view</p>
          <button
            onClick={() => setHasError(false)}
            className="btn-ghost mt-2 !text-xs font-mono text-accent"
          >
            Try again
          </button>
        </div>
      </div>
    )
  }

  return <CanvasView onFocusPanel={onFocusPanel} onClosePopup={onClosePopup} />
}
