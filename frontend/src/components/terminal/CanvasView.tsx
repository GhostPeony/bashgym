import { useCallback, useMemo, useEffect, useState, useRef } from 'react'
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
}

function CanvasViewInner({ onFocusPanel }: CanvasViewProps) {
  const {
    panels,
    sessions,
    sessionsVersion,
    activePanelId,
    canvasNodes,
    updateCanvasNode,
    removePanel,
    setActivePanel,
    createTerminal
  } = useTerminalStore()

  const {
    gridEnabled,
    snapToGrid,
    showMiniMap
  } = useCanvasControlStore()

  const { fitView, zoomIn, zoomOut, getZoom } = useReactFlow()
  const [currentZoom, setCurrentZoom] = useState(1)

  // Convert panels to React Flow nodes
  const initialNodes = useMemo(() => {
    return panels.map((panel, index) => {
      const existingNode = canvasNodes.get(panel.id)
      const session = panel.terminalId ? sessions.get(panel.terminalId) : undefined

      // Default position: grid layout
      const defaultPosition = {
        x: 50 + (index % 3) * 380,
        y: 50 + Math.floor(index / 3) * 280
      }

      // Determine node type based on panel type
      const nodeType = panel.type === 'preview' ? 'preview' :
                       panel.type === 'browser' ? 'browser' : 'terminal'

      // Build data based on node type
      let nodeData: TerminalNodeData | PreviewNodeData | BrowserNodeData

      if (panel.type === 'preview') {
        nodeData = {
          panelId: panel.id,
          filePath: panel.filePath || '',
          onFocus: (id: string) => {
            setActivePanel(id)
            onFocusPanel(id)
          },
          onClose: (id: string) => {
            removePanel(id)
          },
          onCopyPath: (path: string) => {
            navigator.clipboard.writeText(path)
          }
        } as PreviewNodeData
      } else if (panel.type === 'browser') {
        nodeData = {
          panelId: panel.id,
          url: panel.url || '',
          onFocus: (id: string) => {
            setActivePanel(id)
            onFocusPanel(id)
          },
          onClose: (id: string) => {
            removePanel(id)
          },
          onCopyUrl: (url: string) => {
            navigator.clipboard.writeText(url)
          },
          onOpenExternal: (url: string) => {
            window.open(url, '_blank')
          }
        } as BrowserNodeData
      } else {
        nodeData = {
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
          onFocus: (id: string) => {
            setActivePanel(id)
            onFocusPanel(id)
          },
          onClose: (id: string) => {
            removePanel(id)
          }
        } as TerminalNodeData
      }

      return {
        id: panel.id,
        type: nodeType,
        position: existingNode?.position || defaultPosition,
        selected: panel.id === activePanelId,
        data: nodeData
      }
    })
  }, [panels, sessions, sessionsVersion, activePanelId, canvasNodes, setActivePanel, onFocusPanel, removePanel])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  // Update nodes when panels/sessions change
  useEffect(() => {
    setNodes((prevNodes) => {
      // Create map of existing positions
      const positionMap = new Map(prevNodes.map(n => [n.id, n.position]))

      return panels.map((panel, index) => {
        const existingNode = canvasNodes.get(panel.id)
        const savedPosition = positionMap.get(panel.id)
        const session = panel.terminalId ? sessions.get(panel.terminalId) : undefined

        // Default position: grid layout
        const defaultPosition = {
          x: 50 + (index % 3) * 380,
          y: 50 + Math.floor(index / 3) * 280
        }

        const nodeType = panel.type === 'preview' ? 'preview' :
                         panel.type === 'browser' ? 'browser' : 'terminal'

        let nodeData: TerminalNodeData | PreviewNodeData | BrowserNodeData

        if (panel.type === 'preview') {
          nodeData = {
            panelId: panel.id,
            filePath: panel.filePath || '',
            onFocus: (id: string) => {
              setActivePanel(id)
              onFocusPanel(id)
            },
            onClose: (id: string) => {
              removePanel(id)
            },
            onCopyPath: (path: string) => {
              navigator.clipboard.writeText(path)
            }
          } as PreviewNodeData
        } else if (panel.type === 'browser') {
          nodeData = {
            panelId: panel.id,
            url: panel.url || '',
            onFocus: (id: string) => {
              setActivePanel(id)
              onFocusPanel(id)
            },
            onClose: (id: string) => {
              removePanel(id)
            },
            onCopyUrl: (url: string) => {
              navigator.clipboard.writeText(url)
            },
            onOpenExternal: (url: string) => {
              window.open(url, '_blank')
            }
          } as BrowserNodeData
        } else {
          nodeData = {
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
            onFocus: (id: string) => {
              setActivePanel(id)
              onFocusPanel(id)
            },
            onClose: (id: string) => {
              removePanel(id)
            }
          } as TerminalNodeData
        }

        return {
          id: panel.id,
          type: nodeType,
          position: savedPosition || existingNode?.position || defaultPosition,
          selected: panel.id === activePanelId,
          data: nodeData
        }
      })
    })
  }, [panels, sessions, sessionsVersion, activePanelId, canvasNodes, setActivePanel, onFocusPanel, removePanel, setNodes])

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

export function CanvasView({ onFocusPanel }: CanvasViewProps) {
  return (
    <ReactFlowProvider>
      <CanvasViewInner onFocusPanel={onFocusPanel} />
    </ReactFlowProvider>
  )
}

// Wrapper component with error boundary
export function CanvasViewWrapper({ onFocusPanel }: CanvasViewProps) {
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

  return <CanvasView onFocusPanel={onFocusPanel} />
}
