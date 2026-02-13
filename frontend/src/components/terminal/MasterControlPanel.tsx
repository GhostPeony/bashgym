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
          className="btn-secondary !px-3 !py-2 flex items-center gap-2 !text-xs"
          title="Expand control panel"
        >
          <Settings2 className="w-4 h-4 text-text-muted" />
          <span className="font-mono text-text-secondary uppercase tracking-wider">Controls</span>
          <ChevronDown className="w-3 h-3 text-text-muted" />
        </button>
      </div>
    )
  }

  return (
    <div className="absolute top-3 right-3 z-10 w-[240px] card overflow-hidden">
      {/* Header */}
      <div className="terminal-header !py-2 !px-3">
        <div className="flex items-center gap-1.5">
          <span className="terminal-dot terminal-dot-red" />
          <span className="terminal-dot terminal-dot-yellow" />
          <span className="terminal-dot terminal-dot-green" />
        </div>
        <span className="text-[10px] font-mono font-semibold text-text-primary uppercase tracking-wider ml-2 flex-1">
          Master Control
        </span>
        <button
          onClick={togglePanelCollapsed}
          className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
          title="Collapse"
        >
          <ChevronUp className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Session controls */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={handleResumeAll}
            disabled={!globalPaused}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 text-xs font-mono font-semibold uppercase tracking-wider',
              'border-brutal rounded-brutal transition-press',
              globalPaused
                ? 'bg-status-success text-white border-status-success hover-press shadow-brutal-sm'
                : 'bg-background-secondary text-text-muted border-border-subtle cursor-not-allowed'
            )}
          >
            <Play className="w-3 h-3" />
            Resume
          </button>
          <button
            onClick={handlePauseAll}
            disabled={globalPaused}
            className={clsx(
              'flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 text-xs font-mono font-semibold uppercase tracking-wider',
              'border-brutal rounded-brutal transition-press',
              !globalPaused
                ? 'bg-status-warning text-white border-status-warning hover-press shadow-brutal-sm'
                : 'bg-background-secondary text-text-muted border-border-subtle cursor-not-allowed'
            )}
          >
            <Pause className="w-3 h-3" />
            Pause
          </button>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-text-muted">
          <span>
            <span className="text-status-success font-semibold">{sessionStats.active}</span> active
          </span>
          <span>
            <span className="text-status-warning font-semibold">{sessionStats.waiting}</span> waiting
          </span>
          <span>
            <span className="text-text-secondary font-semibold">{sessionStats.idle}</span> idle
          </span>
        </div>
      </div>

      {/* View options */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-mono font-semibold">View Options</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showMetrics}
              onChange={(e) => setShowMetrics(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Metrics</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showToolHistory}
              onChange={(e) => setShowToolHistory(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Tool History</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showRecentFiles}
              onChange={(e) => setShowRecentFiles(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Recent Files</span>
          </label>
        </div>
      </div>

      {/* Canvas settings */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-mono font-semibold">Canvas</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={gridEnabled}
              onChange={(e) => setGridEnabled(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <Grid3X3 className="w-3 h-3 text-text-muted" />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Grid</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={snapToGrid}
              onChange={(e) => setSnapToGrid(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <Magnet className="w-3 h-3 text-text-muted" />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Snap</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showMiniMap}
              onChange={(e) => setShowMiniMap(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <Map className="w-3 h-3 text-text-muted" />
            <span className="text-xs font-mono text-text-secondary group-hover:text-text-primary">Minimap</span>
          </label>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-2 mt-3">
          <span className="text-[10px] text-text-muted font-mono">Zoom:</span>
          <div className="flex items-center gap-1 flex-1">
            <button
              onClick={onZoomOut}
              className="btn-icon !w-5 !h-5 !border-1"
              title="Zoom out"
            >
              <Minus className="w-3 h-3" />
            </button>
            <span className="text-xs font-mono text-text-secondary min-w-[40px] text-center">
              {Math.round(currentZoom * 100)}%
            </span>
            <button
              onClick={onZoomIn}
              className="btn-icon !w-5 !h-5 !border-1"
              title="Zoom in"
            >
              <Plus className="w-3 h-3" />
            </button>
            <button
              onClick={onFitView}
              className="btn-icon !w-5 !h-5 !border-1 ml-1"
              title="Fit view"
            >
              <Maximize className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="flex items-center gap-2">
          {onNewSession && (
            <button
              onClick={onNewSession}
              className="btn-primary !py-1.5 !px-3 !text-xs flex-1"
            >
              <Plus className="w-3 h-3" />
              New Session
            </button>
          )}
          {onAutoArrange && (
            <button
              onClick={onAutoArrange}
              className="btn-secondary !py-1.5 !px-3 !text-xs flex-1"
            >
              <LayoutGrid className="w-3 h-3" />
              Arrange
            </button>
          )}
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="px-3 py-2 bg-background-secondary">
        <div className="flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
          <Keyboard className="w-3 h-3" />
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">1-9</kbd>
            =focus
          </span>
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">Tab</kbd>
            =cycle
          </span>
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">F</kbd>
            =fit
          </span>
        </div>
      </div>
    </div>
  )
})
