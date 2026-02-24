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
  BackgroundVariant,
  Panel as FlowPanel,
  Viewport
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useTerminalStore, useCanvasControlStore } from '../../stores'
import type { Panel, TerminalSession, CanvasEdge } from '../../stores'
import { TerminalNode, type TerminalNodeData } from './TerminalNode'
import { PreviewNode, type PreviewNodeData } from './PreviewNode'
import { BrowserNode, type BrowserNodeData } from './BrowserNode'
import { MasterControlPanel } from './MasterControlPanel'
import { AlertCircle } from 'lucide-react'

const VIEWPORT_KEY = 'bashgym_canvas_viewport'

// Load saved viewport
const loadViewport = (): Viewport | null => {
  try {
    const stored = localStorage.getItem(VIEWPORT_KEY)
    if (stored) {
      return JSON.parse(stored)
    }
  } catch {
    // Ignore
  }
  return null
}

// Save viewport
const saveViewport = (viewport: Viewport) => {
  try {
    localStorage.setItem(VIEWPORT_KEY, JSON.stringify(viewport))
  } catch {
    // Ignore
  }
}

// Register all node types
const nodeTypes = {
  terminal: TerminalNode,
  preview: PreviewNode,
  browser: BrowserNode
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
  canvasEdges: CanvasEdge[] = []
): TerminalNodeData | PreviewNodeData | BrowserNodeData {
  const session = panel.terminalId ? sessions.get(panel.terminalId) : undefined
  const hasConnections = canvasEdges.some(e => e.source === panel.id || e.target === panel.id)

  if (panel.type === 'preview') {
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
    return {
      panelId: panel.id,
      title: panel.title,
      type: panel.type,
      status: session?.status,
      attention: session?.attention,
      gitBranch: session?.gitBranch,
      model: session?.model,
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
      hasConnections,
      onFocus,
      onClose
    } as TerminalNodeData
  }
}

function CanvasViewInner({ onFocusPanel, onClosePopup }: CanvasViewProps) {
  const {
    panels,
    sessions,
    sessionsVersion,
    activePanelId,
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

  const { fitView, zoomIn, zoomOut, getZoom } = useReactFlow()
  const [currentZoom, setCurrentZoom] = useState(1)

  // Stable callbacks — consistent references across renders so TerminalNode memo stays effective
  const handleFocusPanel = useCallback((id: string) => {
    setActivePanel(id)
    onFocusPanel(id)
  }, [setActivePanel, onFocusPanel])

  const handleClosePanel = useCallback((_id: string) => {
    // Close the popup overlay — don't destroy the panel itself
    onClosePopup?.()
  }, [onClosePopup])

  // Track canvasNodes via ref so the update effect can read current saved positions for new
  // panels without re-running every time a drag updates canvasNodes
  const canvasNodesRef = useRef(canvasNodes)
  canvasNodesRef.current = canvasNodes

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  // Sync React Flow edges to Zustand store for cross-component routing (e.g. BrowserPane screenshot routing)
  useEffect(() => {
    setCanvasEdges(edges.map(e => ({ id: e.id, source: e.source, target: e.target })))
  }, [edges, setCanvasEdges])

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
          const nodeType = panel.type === 'preview' ? 'preview' :
                           panel.type === 'browser' ? 'browser' : 'terminal'
          return {
            id: panel.id,
            type: nodeType,
            position: savedPosition || persistedPosition || defaultPosition,
            selected: panel.id === state.activePanelId,
            data: buildNodeData(panel, state.sessions, handleFocusPanelRef.current, handleClosePanelRef.current, state.canvasEdges)
          }
        })
      })
    }

    // Run immediately on mount with current state
    rebuildNodes(useTerminalStore.getState())

    // Subscribe to all subsequent changes
    const unsubscribe = useTerminalStore.subscribe((state, prev) => {
      if (
        state.sessions === prev.sessions &&
        state.panels === prev.panels &&
        state.sessionsVersion === prev.sessionsVersion &&
        state.activePanelId === prev.activePanelId &&
        state.canvasEdges === prev.canvasEdges
      ) return
      rebuildNodes(state)
    })

    return unsubscribe
  }, [setNodes]) // setNodes is stable; callbacks accessed via ref

  // Handle node position changes
  const handleNodesChange = useCallback((changes: any) => {
    onNodesChange(changes)

    // Save position changes
    changes.forEach((change: any) => {
      if (change.type === 'position' && change.position && !change.dragging) {
        updateCanvasNode(change.id, change.position)
      }
    })
  }, [onNodesChange, updateCanvasNode])

  // Handle connections between nodes
  const onConnect = useCallback((connection: Connection) => {
    setEdges((eds) => addEdge({
      ...connection,
      animated: true,
      style: { stroke: 'var(--accent)', strokeWidth: 2 }
    }, eds))
  }, [setEdges])

  // Handle node selection
  const onNodeClick = useCallback((_: any, node: Node) => {
    setActivePanel(node.id)
  }, [setActivePanel])

  // Load saved viewport or use default
  const savedViewport = useRef(loadViewport())
  const shouldFitView = !savedViewport.current

  // Save viewport on move/zoom
  const onMoveEnd = useCallback((_: any, viewport: Viewport) => {
    saveViewport(viewport)
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

  return (
    <div className="h-full w-full relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onMoveEnd={onMoveEnd}
        nodeTypes={nodeTypes}
        fitView={shouldFitView}
        fitViewOptions={{
          padding: 0.2
        }}
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
          Drag nodes to reposition
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
