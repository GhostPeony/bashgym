import { memo, useCallback } from 'react'
import {
  Play,
  Pause,
  Plus,
  LayoutGrid,
  Eye,
  EyeOff,
  Grid3X3,
  Magnet,
  Map,
  Minus,
  Maximize,
  ChevronUp,
  ChevronDown,
  Settings2,
  Keyboard,
  X
} from 'lucide-react'
import { clsx } from 'clsx'
import { useCanvasControlStore, useTerminalStore } from '../../stores'

export interface MasterControlPanelProps {
  onZoomIn?: () => void
  onZoomOut?: () => void
  onFitView?: () => void
  onAutoArrange?: () => void
  onNewSession?: () => void
  currentZoom?: number
}

export const MasterControlPanel = memo(function MasterControlPanel({
  onZoomIn,
  onZoomOut,
  onFitView,
  onAutoArrange,
  onNewSession,
  currentZoom = 1
}: MasterControlPanelProps) {
  const {
    globalPaused,
    showMetrics,
    showToolHistory,
    showRecentFiles,
    gridEnabled,
    snapToGrid,
    showMiniMap,
    isPanelCollapsed,
    setGlobalPaused,
    setShowMetrics,
    setShowToolHistory,
    setShowRecentFiles,
    setGridEnabled,
    setSnapToGrid,
    setShowMiniMap,
    togglePanelCollapsed
  } = useCanvasControlStore()

  const { sessions } = useTerminalStore()

  // Count active and waiting sessions
  const sessionStats = Array.from(sessions.values()).reduce(
    (acc, session) => {
      if (session.status === 'running' || session.status === 'tool_calling') {
        acc.active++
      } else if (session.status === 'waiting_input') {
        acc.waiting++
      } else {
        acc.idle++
      }
      return acc
    },
    { active: 0, waiting: 0, idle: 0 }
  )

  const handlePauseAll = useCallback(() => {
    setGlobalPaused(true)
  }, [setGlobalPaused])

  const handleResumeAll = useCallback(() => {
    setGlobalPaused(false)
  }, [setGlobalPaused])

  if (isPanelCollapsed) {
    return (
      <div className="absolute top-3 right-3 z-10">
        <button
          onClick={togglePanelCollapsed}
          className="flex items-center gap-2 px-3 py-2 bg-background-secondary border border-border-subtle rounded-lg shadow-lg hover:bg-background-tertiary transition-colors"
          title="Expand control panel"
        >
          <Settings2 className="w-4 h-4 text-text-muted" />
          <span className="text-xs text-text-secondary">Controls</span>
          <ChevronDown className="w-3 h-3 text-text-muted" />
        </button>
      </div>
    )
  }

  return (
    <div className="absolute top-3 right-3 z-10 w-[240px] bg-background-secondary border border-border-subtle rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-background-tertiary/50 border-b border-border-subtle">
        <span className="text-xs font-semibold text-text-primary uppercase tracking-wide">
          Master Control
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={togglePanelCollapsed}
            className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
            title="Collapse"
          >
            <ChevronUp className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Session controls */}
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={handleResumeAll}
            disabled={!globalPaused}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium transition-colors',
              globalPaused
                ? 'bg-status-success/20 text-status-success hover:bg-status-success/30'
                : 'bg-background-tertiary text-text-muted cursor-not-allowed'
            )}
          >
            <Play className="w-3 h-3" />
            Resume All
          </button>
          <button
            onClick={handlePauseAll}
            disabled={globalPaused}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium transition-colors',
              !globalPaused
                ? 'bg-status-warning/20 text-status-warning hover:bg-status-warning/30'
                : 'bg-background-tertiary text-text-muted cursor-not-allowed'
            )}
          >
            <Pause className="w-3 h-3" />
            Pause All
          </button>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-text-muted">
          <span>
            <span className="text-status-success font-medium">{sessionStats.active}</span> active
          </span>
          <span>
            <span className="text-status-warning font-medium">{sessionStats.waiting}</span> waiting
          </span>
          <span>
            <span className="text-text-secondary font-medium">{sessionStats.idle}</span> idle
          </span>
        </div>
      </div>

      {/* View options */}
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="text-[10px] text-text-muted uppercase tracking-wide mb-2">View Options</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showMetrics}
              onChange={(e) => setShowMetrics(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Metrics</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showToolHistory}
              onChange={(e) => setShowToolHistory(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Tool History</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showRecentFiles}
              onChange={(e) => setShowRecentFiles(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Recent Files</span>
          </label>
        </div>
      </div>

      {/* Canvas settings */}
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="text-[10px] text-text-muted uppercase tracking-wide mb-2">Canvas</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={gridEnabled}
              onChange={(e) => setGridEnabled(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <Grid3X3 className="w-3 h-3 text-text-muted" />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Grid</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={snapToGrid}
              onChange={(e) => setSnapToGrid(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <Magnet className="w-3 h-3 text-text-muted" />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Snap</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showMiniMap}
              onChange={(e) => setShowMiniMap(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border-subtle bg-background-tertiary checked:bg-primary checked:border-primary"
            />
            <Map className="w-3 h-3 text-text-muted" />
            <span className="text-xs text-text-secondary group-hover:text-text-primary">Minimap</span>
          </label>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-2 mt-3">
          <span className="text-[10px] text-text-muted">Zoom:</span>
          <div className="flex items-center gap-1 flex-1">
            <button
              onClick={onZoomOut}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
              title="Zoom out"
            >
              <Minus className="w-3 h-3" />
            </button>
            <span className="text-xs text-text-secondary min-w-[40px] text-center">
              {Math.round(currentZoom * 100)}%
            </span>
            <button
              onClick={onZoomIn}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
              title="Zoom in"
            >
              <Plus className="w-3 h-3" />
            </button>
            <button
              onClick={onFitView}
              className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary ml-1"
              title="Fit view"
            >
              <Maximize className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="px-3 py-2 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          {onNewSession && (
            <button
              onClick={onNewSession}
              className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
            >
              <Plus className="w-3 h-3" />
              New Session
            </button>
          )}
          {onAutoArrange && (
            <button
              onClick={onAutoArrange}
              className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded text-xs font-medium bg-background-tertiary text-text-secondary hover:bg-background-primary transition-colors"
            >
              <LayoutGrid className="w-3 h-3" />
              Auto-Arrange
            </button>
          )}
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="px-3 py-2 bg-background-tertiary/30">
        <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
          <Keyboard className="w-3 h-3" />
          <span>
            <kbd className="px-1 py-0.5 rounded bg-background-tertiary border border-border-subtle text-[9px]">1-9</kbd>
            =focus
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-background-tertiary border border-border-subtle text-[9px]">Tab</kbd>
            =cycle
          </span>
          <span>
            <kbd className="px-1 py-0.5 rounded bg-background-tertiary border border-border-subtle text-[9px]">F</kbd>
            =fit
          </span>
        </div>
      </div>
    </div>
  )
})
