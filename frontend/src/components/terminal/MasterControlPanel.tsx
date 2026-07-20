import { memo, useCallback, useMemo, useState } from 'react'
import {
  Play,
  Pause,
  Plus,
  Bot,
  Code2,
  FolderOpen,
  Save,
  LayoutGrid,
  Minus,
  Maximize,
  ChevronUp,
  ChevronDown,
  Settings2,
  Keyboard,
  StickyNote,
  Database,
  Triangle,
  Edit3
} from 'lucide-react'
import { clsx } from 'clsx'
import { useCanvasControlStore, useTerminalStore, useWorkspaceStore } from '../../stores'
import { DATA_PANEL_DEFS } from './nodes/dataPanels'
import { GhostPeonyIcon, type GhostPeonyIconName } from '../common'
import {
  BUILTIN_PRESETS,
  captureCurrentAsPreset,
  loadCustomPresets,
  saveCustomPreset,
  type CanvasPreset
} from './canvasPresets'
import type { PanelType } from '../../stores/terminalStore'

export interface MasterControlPanelProps {
  onZoomIn?: () => void
  onZoomOut?: () => void
  onFitView?: () => void
  onAutoArrange?: () => void
  onNewSession?: () => void
  onLaunchClaude?: () => void
  onLaunchCodex?: () => void
  onAddContext?: () => void
  onAddNeon?: () => void
  onAddVercel?: () => void
  onAddDataPanel?: (type: PanelType, title: string) => void
  onApplyPreset?: (preset: CanvasPreset) => void
  currentZoom?: number
}

export const MasterControlPanel = memo(function MasterControlPanel({
  onZoomIn,
  onZoomOut,
  onFitView,
  onAutoArrange,
  onNewSession,
  onLaunchClaude,
  onLaunchCodex,
  onAddContext,
  onAddNeon,
  onAddVercel,
  onAddDataPanel,
  onApplyPreset,
  currentZoom = 1
}: MasterControlPanelProps) {
  const [customPresets, setCustomPresets] = useState<CanvasPreset[]>(() => loadCustomPresets())
  const [selectedPresetId, setSelectedPresetId] = useState<string>(BUILTIN_PRESETS[0].id)
  const { workspaces, activeWorkspaceId, switchWorkspace, createWorkspace, renameWorkspace } =
    useWorkspaceStore()

  const allPresets = useMemo(() => [...BUILTIN_PRESETS, ...customPresets], [customPresets])

  const handleOpenPreset = useCallback(() => {
    const preset = [...BUILTIN_PRESETS, ...loadCustomPresets()].find(
      (p) => p.id === selectedPresetId
    )
    if (preset && onApplyPreset) onApplyPreset(preset)
  }, [selectedPresetId, onApplyPreset])

  const handleSavePreset = useCallback(() => {
    const name = window.prompt('Preset name?')
    if (!name?.trim()) return
    const preset = captureCurrentAsPreset(name.trim())
    saveCustomPreset(preset)
    setCustomPresets(loadCustomPresets())
    setSelectedPresetId(preset.id)
  }, [])
  const {
    globalPaused,
    gridEnabled,
    snapToGrid,
    showMiniMap,
    isPanelCollapsed,
    setGlobalPaused,
    setGridEnabled,
    setSnapToGrid,
    setShowMiniMap,
    togglePanelCollapsed
  } = useCanvasControlStore()

  const { sessions } = useTerminalStore()

  // Count active and waiting sessions
  const sessionStats = useMemo(
    () =>
      Array.from(sessions.values()).reduce(
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
      ),
    [sessions]
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
        <span className="text-[10px] font-mono font-semibold text-text-primary uppercase tracking-wider flex-1">
          Master Control
        </span>
        <button onClick={togglePanelCollapsed} className="node-btn" title="Collapse">
          <ChevronUp className="w-3 h-3" />
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
            <span className="text-status-warning font-semibold">{sessionStats.waiting}</span>{' '}
            waiting
          </span>
          <span>
            <span className="text-text-secondary font-semibold">{sessionStats.idle}</span> idle
          </span>
        </div>
      </div>

      {/* Canvas settings */}
      <div className="px-3 py-1.5 border-b border-brutal border-border">
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <div className="text-[10px] text-text-muted uppercase tracking-wider font-mono font-semibold">
            Canvas
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-text-muted font-mono">Zoom:</span>
            <button onClick={onZoomOut} className="node-btn" title="Zoom out">
              <Minus className="w-3 h-3" />
            </button>
            <span className="text-[10px] font-mono text-text-secondary min-w-[34px] text-center">
              {Math.round(currentZoom * 100)}%
            </span>
            <button onClick={onZoomIn} className="node-btn" title="Zoom in">
              <Plus className="w-3 h-3" />
            </button>
            <button onClick={onFitView} className="node-btn node-btn-accent" title="Fit view">
              <Maximize className="w-3 h-3" />
            </button>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          <label className="flex min-w-0 items-center justify-center gap-1 cursor-pointer group">
            <input
              type="checkbox"
              checked={gridEnabled}
              onChange={(e) => setGridEnabled(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-[10px] font-mono text-text-secondary group-hover:text-text-primary">
              Grid
            </span>
          </label>
          <label className="flex min-w-0 items-center justify-center gap-1 cursor-pointer group">
            <input
              type="checkbox"
              checked={snapToGrid}
              onChange={(e) => setSnapToGrid(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-[10px] font-mono text-text-secondary group-hover:text-text-primary">
              Snap
            </span>
          </label>
          <label className="flex min-w-0 items-center justify-center gap-1 cursor-pointer group">
            <input
              type="checkbox"
              checked={showMiniMap}
              onChange={(e) => setShowMiniMap(e.target.checked)}
              className="w-3.5 h-3.5 border-brutal border-border bg-background-card accent-accent"
            />
            <span className="text-[10px] font-mono text-text-secondary group-hover:text-text-primary">
              Minimap
            </span>
          </label>
        </div>
      </div>

      {/* Actions */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="flex items-center gap-2">
          {onNewSession && (
            <button onClick={onNewSession} className="btn-primary btn-compact flex-1">
              <Plus className="w-3 h-3" />
              New Session
            </button>
          )}
          {onAutoArrange && (
            <button onClick={onAutoArrange} className="btn-secondary btn-compact flex-1">
              <LayoutGrid className="w-3 h-3" />
              Arrange
            </button>
          )}
        </div>
        {(onLaunchClaude || onLaunchCodex) && (
          <div className="flex items-center gap-2 mt-2">
            {onLaunchClaude && (
              <button onClick={onLaunchClaude} className="btn-secondary btn-compact flex-1">
                <Bot className="w-3 h-3" />
                Claude
              </button>
            )}
            {onLaunchCodex && (
              <button onClick={onLaunchCodex} className="btn-secondary btn-compact flex-1">
                <Code2 className="w-3 h-3" />
                Codex
              </button>
            )}
          </div>
        )}
        {(onAddContext || onAddNeon || onAddVercel) && (
          <div className="flex items-center gap-2 mt-2">
            {onAddContext && (
              <button onClick={onAddContext} className="btn-secondary btn-compact flex-1">
                <StickyNote className="w-3 h-3" />
                Context
              </button>
            )}
            {onAddNeon && (
              <button onClick={onAddNeon} className="btn-secondary btn-compact flex-1">
                <Database className="w-3 h-3" />
                Neon
              </button>
            )}
            {onAddVercel && (
              <button onClick={onAddVercel} className="btn-secondary btn-compact flex-1">
                <Triangle className="w-3 h-3" />
                Vercel
              </button>
            )}
          </div>
        )}
        {onAddDataPanel && (
          <div className="master-control-action-grid mt-2">
            {DATA_PANEL_DEFS.map((def) => {
              return (
                <button
                  key={def.type}
                  onClick={() => onAddDataPanel(def.type, def.title)}
                  className="btn-secondary btn-compact master-control-action"
                  style={{ color: `hsl(${def.hue}, 45%, 48%)` }}
                  title={`Add ${def.title} node`}
                >
                  <GhostPeonyIcon
                    name={def.type as GhostPeonyIconName}
                    size="xs"
                    tone="node"
                    hue={def.hue}
                  />
                  {def.title}
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Workspace instances */}
      <div className="px-3 py-2 border-b border-brutal border-border">
        <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-mono font-semibold">
          Workspace
        </div>
        <div className="flex items-center gap-1.5">
          <select
            value={activeWorkspaceId}
            onChange={(e) => switchWorkspace(e.target.value)}
            className="flex-1 min-w-0 text-xs font-mono bg-background-card border-brutal border-border rounded-brutal px-1.5 py-1 focus:border-accent focus:outline-none"
            title="Switch workspace — background workspaces keep their terminals running"
          >
            {workspaces.map((ws) => (
              <option key={ws.id} value={ws.id}>
                {ws.name}
              </option>
            ))}
          </select>
          <button
            onClick={() => {
              const name = window.prompt('Workspace name?')
              if (name?.trim()) createWorkspace(name, { activate: true })
            }}
            className="node-btn node-btn-accent"
            title="New workspace"
          >
            <Plus className="w-3 h-3" />
          </button>
          <button
            onClick={() => {
              const current = workspaces.find((w) => w.id === activeWorkspaceId)
              const name = window.prompt('Rename workspace', current?.name)
              if (name?.trim()) renameWorkspace(activeWorkspaceId, name)
            }}
            className="node-btn"
            title="Rename this workspace"
          >
            <Edit3 className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Workspace layouts */}
      {onApplyPreset && (
        <div className="px-3 py-2 border-b border-brutal border-border">
          <div className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-mono font-semibold">
            Layouts
          </div>
          <div className="flex items-center gap-1.5">
            <select
              value={selectedPresetId}
              onChange={(e) => setSelectedPresetId(e.target.value)}
              className="flex-1 min-w-0 text-xs font-mono bg-background-card border-brutal border-border rounded-brutal px-1.5 py-1 focus:border-accent focus:outline-none"
            >
              {allPresets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <button
              onClick={handleOpenPreset}
              className="node-btn node-btn-accent"
              title="Open this layout"
            >
              <FolderOpen className="w-3 h-3" />
            </button>
            <button
              onClick={handleSavePreset}
              className="node-btn"
              title="Save current canvas as a layout"
            >
              <Save className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts hint */}
      <div className="px-3 py-2 bg-background-secondary">
        <div className="flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
          <Keyboard className="w-3 h-3" />
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">
              1-9
            </kbd>
            =focus
          </span>
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">
              Tab
            </kbd>
            =cycle
          </span>
          <span>
            <kbd className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-card text-[9px] font-mono">
              F
            </kbd>
            =fit
          </span>
        </div>
      </div>
    </div>
  )
})
