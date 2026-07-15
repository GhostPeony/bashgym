import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { Plus, Terminal, X, Globe, FileText, Layers, FolderTree, GripVertical, LayoutGrid, Maximize2, MessageSquare } from 'lucide-react'
import { useTerminalStore, useUIStore, useWorkspaceStore } from '../../stores'
import { TerminalPane } from './TerminalPane'
import { PreviewPane } from './PreviewPane'
import { BrowserPane } from './BrowserPane'
import { FileBrowser } from '../files/FileBrowser'
import { CanvasViewWrapper } from './CanvasView'
import { DATA_PANEL_DEFS, DATA_NODE_TYPES } from './nodes/dataPanels'
import { CustomNodeSummary } from './nodes/CustomNodeSummary'
import { isCustomNodeType } from './nodes/customNodeTypes'
import { buildCanvasGraphIndex } from './canvasPerformance'
import { getGridColumnSpan, getGridLayout, getGridTemplateRows } from './gridLayout'
import { clsx } from 'clsx'
import { GhostPeonyIcon, type GhostPeonyIconName } from '../common'

export function TerminalGrid() {
  const panels = useTerminalStore((state) => state.panels)
  const createTerminal = useTerminalStore((state) => state.createTerminal)
  const activePanelId = useTerminalStore((state) => state.activePanelId)
  const setActivePanel = useTerminalStore((state) => state.setActivePanel)
  const viewMode = useTerminalStore((state) => state.viewMode)
  const setViewMode = useTerminalStore((state) => state.setViewMode)
  const removePanel = useTerminalStore((state) => state.removePanel)
  const renamePanel = useTerminalStore((state) => state.renamePanel)
  const reorderPanels = useTerminalStore((state) => state.reorderPanels)
  const draggedPanelId = useTerminalStore((state) => state.draggedPanelId)
  const setDraggedPanel = useTerminalStore((state) => state.setDraggedPanel)
  const addPanel = useTerminalStore((state) => state.addPanel)
  const canvasEdges = useTerminalStore((state) => state.canvasEdges)
  const canvasGraph = useMemo(
    () => buildCanvasGraphIndex(panels, canvasEdges),
    [canvasEdges, panels],
  )

  // Drag state for visual feedback
  const [dragOverId, setDragOverId] = useState<string | null>(null)

  // Editing state for renaming tabs
  const [editingPanelId, setEditingPanelId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  // Prevent React Strict Mode from creating multiple terminals
  const initializedRef = useRef(false)

  // On first mount: materialize the active workspace (restores its panels,
  // re-adopts live PTYs, adopts orphans, opens a terminal on true first run)
  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true
      void useTerminalStore.getState().bootActiveWorkspace()
    }
  }, [])

  // State for canvas popup - must be before any early returns
  const [canvasPopupPanelId, setCanvasPopupPanelId] = useState<string | null>(null)

  // Workspace switching: remount the canvas per workspace and drop any open popup
  const activeWorkspaceId = useWorkspaceStore((s) => s.activeWorkspaceId)
  const panelPresentationRequest = useUIStore((state) => state.panelPresentationRequest)
  const clearPanelPresentationRequest = useUIStore((state) => state.clearPanelPresentationRequest)
  const isAgentStreamOpen = useUIStore((state) => state.isAgentStreamOpen)
  const setAgentStreamOpen = useUIStore((state) => state.setAgentStreamOpen)
  useEffect(() => {
    setCanvasPopupPanelId(null)
  }, [activeWorkspaceId])

  // Shared presentation requests let session/history surfaces focus a panel
  // without reaching into this component's local popup implementation.
  useEffect(() => {
    if (!panelPresentationRequest) return
    if (panelPresentationRequest.workspaceId !== activeWorkspaceId) {
      useWorkspaceStore.getState().switchWorkspace(panelPresentationRequest.workspaceId)
      return
    }

    const panel = useTerminalStore.getState().panels.find(
      (candidate) => candidate.id === panelPresentationRequest.panelId
    )
    if (!panel) {
      clearPanelPresentationRequest(panelPresentationRequest.requestedAt)
      return
    }
    setActivePanel(panel.id)
    if (viewMode === 'canvas' && !isCustomNodeType(panel.type)) {
      setCanvasPopupPanelId(panel.id)
    }
    clearPanelPresentationRequest(panelPresentationRequest.requestedAt)
  }, [
    activeWorkspaceId,
    clearPanelPresentationRequest,
    panelPresentationRequest,
    setActivePanel,
    viewMode
  ])

  // Handle focus panel from canvas view. Data/config nodes stay on-canvas and own their modal settings.
  const handleFocusPanel = useCallback((panelId: string) => {
    setActivePanel(panelId)
    const panel = useTerminalStore.getState().panels.find(p => p.id === panelId)
    if (panel && isCustomNodeType(panel.type)) return
    setCanvasPopupPanelId(panelId)
  }, [setActivePanel])

  // Close canvas popup
  const closeCanvasPopup = useCallback(() => {
    setCanvasPopupPanelId(null)
  }, [])

  // Close popup on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && canvasPopupPanelId) {
        closeCanvasPopup()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [canvasPopupPanelId, closeCanvasPopup])

  // Get icon for panel type
  const getPanelIcon = (type: string) => {
    switch (type) {
      case 'terminal':
        return <Terminal className="w-3.5 h-3.5 flex-shrink-0" />
      case 'browser':
        return <Globe className="w-3.5 h-3.5 flex-shrink-0" />
      case 'preview':
        return <FileText className="w-3.5 h-3.5 flex-shrink-0" />
      case 'files':
        return <FolderTree className="w-3.5 h-3.5 flex-shrink-0" />
      default: {
        const def = DATA_PANEL_DEFS.find((d) => d.type === type)
        if (def) {
          return (
            <GhostPeonyIcon
              name={def.type as GhostPeonyIconName}
              size="xs"
              tone="node"
              hue={def.hue}
              className="flex-shrink-0"
            />
          )
        }
        return <Terminal className="w-3.5 h-3.5 flex-shrink-0" />
      }
    }
  }

  // Drag handlers for tab reordering
  const handleDragStart = useCallback((e: React.DragEvent, panelId: string) => {
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', panelId)
    setDraggedPanel(panelId)
  }, [setDraggedPanel])

  const handleDragOver = useCallback((e: React.DragEvent, panelId: string) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    if (draggedPanelId && draggedPanelId !== panelId) {
      setDragOverId(panelId)
    }
  }, [draggedPanelId])

  const handleDragLeave = useCallback(() => {
    setDragOverId(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent, targetPanelId: string) => {
    e.preventDefault()
    const sourcePanelId = e.dataTransfer.getData('text/plain')
    if (sourcePanelId && sourcePanelId !== targetPanelId) {
      reorderPanels(sourcePanelId, targetPanelId)
    }
    setDraggedPanel(null)
    setDragOverId(null)
  }, [reorderPanels, setDraggedPanel])

  const handleDragEnd = useCallback(() => {
    setDraggedPanel(null)
    setDragOverId(null)
  }, [setDraggedPanel])

  // Start editing a panel title
  const startEditing = useCallback((panelId: string, currentTitle: string) => {
    setEditingPanelId(panelId)
    setEditingTitle(currentTitle)
    // Focus input after render
    setTimeout(() => editInputRef.current?.focus(), 0)
  }, [])

  // Save the edited title
  const saveEditing = useCallback(() => {
    if (editingPanelId && editingTitle.trim()) {
      renamePanel(editingPanelId, editingTitle.trim())
    }
    setEditingPanelId(null)
    setEditingTitle('')
  }, [editingPanelId, editingTitle, renamePanel])

  // Cancel editing
  const cancelEditing = useCallback(() => {
    setEditingPanelId(null)
    setEditingTitle('')
  }, [])

  // Open file browser panel
  const openFileBrowser = useCallback(() => {
    addPanel({
      type: 'files',
      title: 'File Browser'
    })
  }, [addPanel])

  // Open browser panel
  const openBrowser = useCallback(() => {
    addPanel({
      type: 'browser',
      title: 'Browser'
    })
  }, [addPanel])

  // Render panel content
  const renderPanel = (panel: typeof panels[0], isActive: boolean, onPopupClose?: () => void) => {
    if (panel.type === 'terminal' && panel.terminalId) {
      return (
        <TerminalPane
          id={panel.terminalId}
          title={panel.title}
          isActive={isActive}
          onPopupClose={onPopupClose}
        />
      )
    }
    if (panel.type === 'preview') {
      return (
        <PreviewPane
          id={panel.id}
          title={panel.title}
          filePath={panel.filePath}
          isActive={isActive}
        />
      )
    }
    if (panel.type === 'browser') {
      return (
        <BrowserPane
          id={panel.id}
          title={panel.title}
          url={panel.url}
          isActive={isActive}
        />
      )
    }
    if (panel.type === 'files') {
      return (
        <FileBrowser
          id={panel.id}
          title={panel.title}
          isActive={isActive}
        />
      )
    }
    if (isCustomNodeType(panel.type) && viewMode === 'grid') {
      return (
        <CustomNodeSummary
          panel={panel}
          graph={canvasGraph}
          selected={isActive}
          onFocus={setActivePanel}
          onClose={removePanel}
        />
      )
    }
    if (DATA_NODE_TYPES.includes(panel.type)) {
      return (
        <div className="h-full flex items-center justify-center bg-background-secondary">
          <span className="text-xs font-mono text-text-muted px-4 text-center">
            "{panel.title}" is a canvas node — switch to Canvas view
          </span>
        </div>
      )
    }
    return null
  }

  // Empty state — this workspace has no panels yet
  if (panels.length === 0) {
    const wsName = useWorkspaceStore.getState().workspaces.find((w) => w.id === activeWorkspaceId)?.name
    return (
      <div className="h-full flex items-center justify-center bg-background-primary">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 border-brutal border-border rounded-brutal bg-background-secondary flex items-center justify-center shadow-brutal-sm">
            <Terminal className="w-8 h-8 text-text-muted" />
          </div>
          <h3 className="text-lg font-brand font-semibold text-text-primary mb-2">
            {wsName ? `${wsName} is empty` : 'No terminals open'}
          </h3>
          <p className="text-sm text-text-muted mb-4 font-mono">
            Launch a session to start working in this workspace
          </p>
          <div className="flex items-center justify-center gap-2">
            <button
              onClick={() => createTerminal(undefined, 'Claude Code', 'claude')}
              className="node-btn node-btn-wide node-btn-accent"
            >
              NEW CLAUDE
            </button>
            <button
              onClick={() => createTerminal(undefined, 'Codex', 'codex')}
              className="node-btn node-btn-wide"
            >
              NEW CODEX
            </button>
            <button
              onClick={() => createTerminal()}
              className="node-btn node-btn-wide"
            >
              <span className="flex items-center gap-1"><Plus className="w-2.5 h-2.5" /> SHELL</span>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Toolbar component (shared between views)
  const renderToolbar = () => (
    <div className="flex items-center gap-1 px-2 py-1 bg-background-secondary border-b border-brutal border-border overflow-x-auto">
      {panels.map((panel) => {
        const isActive = panel.id === activePanelId
        const isDragging = draggedPanelId === panel.id
        const isDragOver = dragOverId === panel.id
        const isEditing = editingPanelId === panel.id
        return (
          <div
            key={panel.id}
            draggable={!isEditing}
            onDragStart={(e) => !isEditing && handleDragStart(e, panel.id)}
            onDragOver={(e) => handleDragOver(e, panel.id)}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, panel.id)}
            onDragEnd={handleDragEnd}
            className={clsx(
              'flex items-center gap-1.5 px-2 py-1.5 cursor-pointer group min-w-0 font-mono text-xs',
              'border-brutal rounded-brutal transition-press',
              isActive
                ? 'bg-background-card text-text-primary border-border shadow-brutal-sm border-b-accent'
                : 'text-text-secondary border-transparent hover:text-accent hover:bg-background-tertiary hover:border-border-subtle',
              isDragging && 'scale-95',
              isDragOver && 'border-accent shadow-brutal-sm'
            )}
            onClick={() => {
              if (isEditing) return
              setActivePanel(panel.id)
              // In canvas mode with popup open, switch the popup to show this panel
              if (viewMode === 'canvas' && canvasPopupPanelId) {
                setCanvasPopupPanelId(panel.id)
              }
            }}
          >
            <GripVertical className="w-3 h-3 text-text-muted hidden group-hover:block cursor-grab active:cursor-grabbing flex-shrink-0" />
            {getPanelIcon(panel.type)}
            {isEditing ? (
              <input
                ref={editInputRef}
                type="text"
                value={editingTitle}
                onChange={(e) => setEditingTitle(e.target.value)}
                onBlur={saveEditing}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    saveEditing()
                  } else if (e.key === 'Escape') {
                    cancelEditing()
                  }
                  e.stopPropagation()
                }}
                onClick={(e) => e.stopPropagation()}
                className="input !text-xs !py-0.5 !px-1 w-24 font-mono"
                autoFocus
              />
            ) : (
              <span
                className="text-sm truncate max-w-[120px] font-mono"
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  startEditing(panel.id, panel.title)
                }}
                title="Double-click to rename"
              >
                {panel.title}
              </span>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation()
                removePanel(panel.id)
              }}
              className="p-0.5 hidden group-hover:block hover:text-status-error"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )
      })}

      {/* Add new terminal button */}
      <button
        onClick={() => createTerminal()}
        className="btn-icon !w-7 !h-7 text-text-muted hover:text-accent"
        title="New Terminal"
      >
        <Plus className="w-4 h-4" />
      </button>

      {/* Add file browser button */}
      <button
        onClick={openFileBrowser}
        className="btn-icon !w-7 !h-7 text-text-muted hover:text-accent"
        title="Open File Browser"
      >
        <FolderTree className="w-4 h-4" />
      </button>

      {/* Add browser button */}
      <button
        onClick={openBrowser}
        className="btn-icon !w-7 !h-7 text-text-muted hover:text-accent"
        title="Open Browser"
      >
        <Globe className="w-4 h-4" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      <button
        type="button"
        onClick={() => setAgentStreamOpen(!isAgentStreamOpen)}
        className={clsx(
          'flex h-7 flex-shrink-0 items-center gap-1.5 rounded-brutal px-2 font-mono text-[10px] font-semibold uppercase tracking-[0.08em] transition-colors',
          isAgentStreamOpen
            ? 'bg-accent/[0.12] text-accent'
            : 'text-text-muted hover:bg-background-tertiary hover:text-text-primary'
        )}
        title={isAgentStreamOpen ? 'Hide live agent feed' : 'Open live agent feed'}
        aria-label={isAgentStreamOpen ? 'Hide live agent feed' : 'Open live agent feed'}
        aria-pressed={isAgentStreamOpen}
      >
        <MessageSquare className="h-3.5 w-3.5" />
        <span>Agent feed</span>
      </button>

      {/* View mode buttons */}
      <div className="flex items-center gap-0.5 p-0.5 border-brutal border-border rounded-brutal bg-background-tertiary">
        <button
          onClick={() => setViewMode('grid')}
          className={clsx(
            'p-1.5 rounded-brutal transition-press',
            viewMode === 'grid'
              ? 'bg-background-card text-accent shadow-brutal-sm border-brutal border-border'
              : 'text-text-muted hover:text-text-secondary border border-transparent'
          )}
          title="Grid view"
        >
          <LayoutGrid className="w-4 h-4" />
        </button>
        <button
          onClick={() => setViewMode('single')}
          className={clsx(
            'p-1.5 rounded-brutal transition-press',
            viewMode === 'single'
              ? 'bg-background-card text-accent shadow-brutal-sm border-brutal border-border'
              : 'text-text-muted hover:text-text-secondary border border-transparent'
          )}
          title="Single view"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
        <button
          onClick={() => setViewMode('canvas')}
          className={clsx(
            'p-1.5 rounded-brutal transition-press',
            viewMode === 'canvas'
              ? 'bg-background-card text-accent shadow-brutal-sm border-brutal border-border'
              : 'text-text-muted hover:text-text-secondary border border-transparent'
          )}
          title="Canvas view"
        >
          <Layers className="w-4 h-4" />
        </button>
      </div>
    </div>
  )

  const gridLayout = getGridLayout(panels.length)

  return (
    <div className="h-full flex flex-col bg-background-primary">
      {/* Toolbar - always shown */}
      {renderToolbar()}

      {/* Main content area */}
      {/*
        IMPORTANT: All panels are ALWAYS mounted EXACTLY ONCE to prevent terminal reload.
        We use CSS visibility/positioning to show/hide them based on view mode.
        This ensures xterm.js instances are never disposed and recreated.

        In canvas mode with popup: the terminal stays in its container but the container
        is repositioned to appear as a popup (not re-rendered).
      */}
      <div
        className={clsx(
          'flex-1 min-h-0',
          viewMode === 'grid'
            ? 'grid gap-1 overflow-auto p-1'
            : 'relative overflow-hidden',
        )}
        style={viewMode === 'grid'
          ? {
              gridTemplateColumns: `repeat(${gridLayout.tracks}, minmax(0, 1fr))`,
              gridTemplateRows: getGridTemplateRows(panels.length),
            }
          : undefined
        }
      >
        {/* Canvas view background - rendered first so terminals appear on top when in popup */}
        {viewMode === 'canvas' && (
          <div className="absolute inset-0 z-10">
            <CanvasViewWrapper key={activeWorkspaceId} onFocusPanel={handleFocusPanel} onClosePopup={closeCanvasPopup} />
          </div>
        )}

        {/* Canvas popup backdrop - only when popup is open */}
        {viewMode === 'canvas' && canvasPopupPanelId && (
          <div
            className="absolute inset-0 z-50 canvas-popup-backdrop cursor-pointer"
            onClick={closeCanvasPopup}
          />
        )}

        {/* Always render all panels exactly once - use CSS to control visibility and position */}
        {panels.map((panel, index) => {
          const isActive = panel.id === activePanelId
          const isInPopup = viewMode === 'canvas' && panel.id === canvasPopupPanelId
          // Canvas panels remain mounted while hidden, but they must become
          // inactive so reopening the popup creates a real false -> true
          // transition and TerminalPane restores xterm keyboard focus.
          const isPresented = viewMode === 'canvas' ? isInPopup : isActive

          // Determine container style based on view mode
          let containerStyle: React.CSSProperties = {}
          let containerClass = ''

          if (viewMode === 'grid') {
            const span = getGridColumnSpan(index, panels.length)
            containerStyle = { gridColumn: `span ${span} / span ${span}` }
            containerClass = clsx(
              'min-h-0 min-w-0 overflow-hidden rounded-brutal border-brutal transition-press',
              isActive
                ? 'border-accent shadow-brutal-sm'
                : 'border-border-subtle hover:border-border'
            )
          } else if (viewMode === 'single') {
            // Single view - stack all, show only active
            containerStyle = { position: 'absolute', inset: 0 }
            containerClass = clsx(
              isActive ? 'visible z-10' : 'invisible z-0'
            )
          } else if (isInPopup) {
            // Canvas view - THIS terminal is in the popup
            // Use flexbox centering (not transforms) to avoid sub-pixel blurriness
            containerStyle = {
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 55
            }
            containerClass = 'rounded-brutal shadow-brutal border-brutal border-border overflow-hidden bg-background-primary'
          } else {
            // Canvas view - hide terminals not in popup
            containerStyle = { position: 'absolute', inset: 0 }
            containerClass = 'invisible pointer-events-none'
          }

          // For popup mode, we need a sized inner container inside the flex centering container
          // But we must keep the same component structure to prevent remounting
          const needsPopupWrapper = isInPopup

          return (
            <div
              key={panel.id}
              className={needsPopupWrapper ? '' : containerClass}
              style={needsPopupWrapper ? {
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 55,
                pointerEvents: 'none'
              } : containerStyle}
              onClick={() => viewMode === 'grid' && setActivePanel(panel.id)}
            >
              <div
                className={needsPopupWrapper
                  ? clsx(
                      'canvas-popup-panel pointer-events-auto',
                      panel.type === 'terminal' && 'canvas-popup-panel-terminal'
                    )
                  : 'h-full w-full'
                }
                style={needsPopupWrapper ? { width: '90%', maxWidth: '1280px', height: '85%' } : undefined}
              >
                {renderPanel(panel, isPresented, isInPopup ? closeCanvasPopup : undefined)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
