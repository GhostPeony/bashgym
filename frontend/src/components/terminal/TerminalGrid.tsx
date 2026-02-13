import { useEffect, useRef, useState, useCallback } from 'react'
import { Plus, Terminal, Grid, X, Globe, FileText, Layers, FolderTree, GripVertical, LayoutGrid, Maximize2 } from 'lucide-react'
import { useTerminalStore, type Panel } from '../../stores'
import { TerminalPane } from './TerminalPane'
import { PreviewPane } from './PreviewPane'
import { BrowserPane } from './BrowserPane'
import { FileBrowser } from '../files/FileBrowser'
import { CanvasViewWrapper } from './CanvasView'
import { clsx } from 'clsx'

export function TerminalGrid() {
  const {
    panels,
    createTerminal,
    activePanelId,
    setActivePanel,
    viewMode,
    setViewMode,
    cycleViewMode,
    removePanel,
    renamePanel,
    reorderPanels,
    draggedPanelId,
    setDraggedPanel,
    addPanel
  } = useTerminalStore()

  // Drag state for visual feedback
  const [dragOverId, setDragOverId] = useState<string | null>(null)

  // Editing state for renaming tabs
  const [editingPanelId, setEditingPanelId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  // Prevent React Strict Mode from creating multiple terminals
  const initializedRef = useRef(false)

  // Create initial terminal on mount if none exist
  useEffect(() => {
    if (panels.length === 0 && !initializedRef.current) {
      initializedRef.current = true
      createTerminal()
    }
  }, [])

  // State for canvas popup - must be before any early returns
  const [canvasPopupPanelId, setCanvasPopupPanelId] = useState<string | null>(null)

  // Handle focus panel from canvas view - show popup
  const handleFocusPanel = useCallback((panelId: string) => {
    setActivePanel(panelId)
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

  // Calculate grid layout based on panel count
  const getGridClass = () => {
    const count = panels.length
    if (count === 0) return ''
    if (count === 1) return 'grid-cols-1'
    if (count === 2) return 'grid-cols-2'
    if (count === 3) return 'grid-cols-2 grid-rows-2'
    if (count === 4) return 'grid-cols-2 grid-rows-2'
    if (count <= 6) return 'grid-cols-3 grid-rows-2'
    return 'grid-cols-3 grid-rows-3'
  }

  // Get span class for 3-panel layout
  const getSpanClass = (index: number, total: number) => {
    if (total === 3 && index === 2) {
      return 'col-span-2'
    }
    return ''
  }

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
      default:
        return <Terminal className="w-3.5 h-3.5 flex-shrink-0" />
    }
  }

  // Get view mode icon
  const getViewModeIcon = () => {
    switch (viewMode) {
      case 'grid':
        return <LayoutGrid className="w-4 h-4" />
      case 'single':
        return <Maximize2 className="w-4 h-4" />
      case 'canvas':
        return <Layers className="w-4 h-4" />
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
    return null
  }

  // Empty state
  if (panels.length === 0) {
    return (
      <div className="h-full flex items-center justify-center bg-background-primary">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 border-brutal border-border rounded-brutal bg-background-secondary flex items-center justify-center shadow-brutal-sm">
            <Terminal className="w-8 h-8 text-text-muted" />
          </div>
          <h3 className="text-lg font-brand font-semibold text-text-primary mb-2">No terminals open</h3>
          <p className="text-sm text-text-muted mb-4 font-mono">
            Create a terminal to start working
          </p>
          <button
            onClick={() => createTerminal()}
            className="btn-primary inline-flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Terminal
          </button>
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
                ? 'bg-background-card text-text-primary border-border shadow-brutal-sm'
                : 'text-text-secondary border-transparent hover:text-text-primary hover:bg-background-tertiary hover:border-border-subtle',
              isDragging && 'scale-95',
              isDragOver && 'border-accent shadow-brutal-sm'
            )}
            onClick={() => !isEditing && setActivePanel(panel.id)}
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
        className="btn-icon !w-7 !h-7 text-text-muted hover:text-text-primary"
        title="New Terminal"
      >
        <Plus className="w-4 h-4" />
      </button>

      {/* Add file browser button */}
      <button
        onClick={openFileBrowser}
        className="btn-icon !w-7 !h-7 text-text-muted hover:text-text-primary"
        title="Open File Browser"
      >
        <FolderTree className="w-4 h-4" />
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* View mode buttons */}
      <div className="flex items-center gap-0.5 p-0.5 border-brutal border-border rounded-brutal bg-background-tertiary">
        <button
          onClick={() => setViewMode('grid')}
          className={clsx(
            'p-1.5 rounded-brutal transition-press',
            viewMode === 'grid'
              ? 'bg-background-card text-text-primary shadow-brutal-sm border-brutal border-border'
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
              ? 'bg-background-card text-text-primary shadow-brutal-sm border-brutal border-border'
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
              ? 'bg-background-card text-text-primary shadow-brutal-sm border-brutal border-border'
              : 'text-text-muted hover:text-text-secondary border border-transparent'
          )}
          title="Canvas view"
        >
          <Layers className="w-4 h-4" />
        </button>
      </div>
    </div>
  )

  // Calculate grid position for a panel
  const getGridStyle = (index: number, total: number): React.CSSProperties => {
    const cols = total <= 2 ? total : total <= 4 ? 2 : 3
    const row = Math.floor(index / cols)
    const col = index % cols
    const rows = Math.ceil(total / cols)

    // Handle 3-panel layout where last panel spans 2 columns
    if (total === 3 && index === 2) {
      return {
        position: 'absolute',
        left: '0%',
        top: `${(row / rows) * 100}%`,
        width: '100%',
        height: `${(1 / rows) * 100}%`,
        padding: '2px'
      }
    }

    return {
      position: 'absolute',
      left: `${(col / cols) * 100}%`,
      top: `${(row / rows) * 100}%`,
      width: `${(1 / cols) * 100}%`,
      height: `${(1 / rows) * 100}%`,
      padding: '2px'
    }
  }

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
      <div className="flex-1 overflow-hidden relative" style={{ minHeight: 0 }}>
        {/* Canvas view background - rendered first so terminals appear on top when in popup */}
        {viewMode === 'canvas' && (
          <div className="absolute inset-0 z-10">
            <CanvasViewWrapper onFocusPanel={handleFocusPanel} />
          </div>
        )}

        {/* Canvas popup backdrop - only when popup is open */}
        {viewMode === 'canvas' && canvasPopupPanelId && (
          <div
            className="absolute inset-0 z-50 bg-black/60 cursor-pointer"
            onClick={closeCanvasPopup}
          />
        )}

        {/* Always render all panels exactly once - use CSS to control visibility and position */}
        {panels.map((panel, index) => {
          const isActive = panel.id === activePanelId
          const isInPopup = viewMode === 'canvas' && panel.id === canvasPopupPanelId
          const spanClass = getSpanClass(index, panels.length)

          // Determine container style based on view mode
          let containerStyle: React.CSSProperties = {}
          let containerClass = ''

          if (viewMode === 'grid') {
            // Grid view - positioned in grid
            const gridStyle = getGridStyle(index, panels.length)
            containerStyle = gridStyle
            containerClass = clsx(
              'rounded-brutal overflow-hidden border-brutal transition-press',
              spanClass,
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
                  ? 'rounded-brutal shadow-brutal border-brutal border-border overflow-hidden bg-background-primary pointer-events-auto'
                  : 'h-full w-full'
                }
                style={needsPopupWrapper ? { width: '90%', maxWidth: '1280px', height: '85%' } : undefined}
              >
                {renderPanel(panel, isActive || isInPopup, isInPopup ? closeCanvasPopup : undefined)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
