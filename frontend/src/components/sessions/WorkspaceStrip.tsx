import { useState } from 'react'
import { Plus, X, LayoutGrid } from 'lucide-react'
import { clsx } from 'clsx'
import { useTerminalStore, useWorkspaceStore } from '../../stores'

/**
 * Workspace switcher: one chip per named canvas workspace. Click switches,
 * double-click renames inline, hover × deletes (background workspaces only).
 * Background workspaces keep their PTYs alive but render nothing.
 */
export function WorkspaceStrip() {
  const { workspaces, activeWorkspaceId, switchWorkspace, createWorkspace, renameWorkspace, deleteWorkspace } =
    useWorkspaceStore()
  useTerminalStore((s) => s.sessionsVersion)
  const sessions = useTerminalStore.getState().sessions
  const hasRunning = Array.from(sessions.values()).some(
    (s) => s.status === 'running' || s.status === 'tool_calling'
  )

  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const handleCreate = () => {
    const name = window.prompt('Workspace name?')
    if (name?.trim()) createWorkspace(name, { activate: true })
  }

  const commitRename = () => {
    if (renamingId && renameValue.trim()) renameWorkspace(renamingId, renameValue)
    setRenamingId(null)
  }

  return (
    <div className="flex items-center gap-1 flex-wrap">
      <LayoutGrid className="w-3 h-3 text-text-muted flex-shrink-0" />
      {workspaces.map((ws) => {
        const isActive = ws.id === activeWorkspaceId
        if (renamingId === ws.id) {
          return (
            <input
              key={ws.id}
              value={renameValue}
              autoFocus
              onChange={(e) => setRenameValue(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename()
                if (e.key === 'Escape') setRenamingId(null)
              }}
              className="input !py-0.5 !px-1.5 !text-[10px] font-mono w-24"
            />
          )
        }
        return (
          <div
            key={ws.id}
            onClick={() => switchWorkspace(ws.id)}
            onDoubleClick={() => {
              setRenamingId(ws.id)
              setRenameValue(ws.name)
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter') switchWorkspace(ws.id)
            }}
            className={clsx(
              'group node-btn node-btn-wide cursor-pointer select-none !inline-flex items-center gap-1',
              isActive && 'node-btn-accent'
            )}
            title={isActive ? `${ws.name} (active) — double-click to rename` : `Switch to ${ws.name}`}
          >
            {isActive && hasRunning && <span className="status-dot status-success" />}
            <span className="truncate max-w-[90px]">{ws.name}</span>
            {!isActive && workspaces.length > 1 && (
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  void deleteWorkspace(ws.id)
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-text-muted hover:text-status-error"
                title={`Delete workspace "${ws.name}"`}
              >
                <X className="w-2.5 h-2.5" />
              </span>
            )}
          </div>
        )
      })}
      <button onClick={handleCreate} className="node-btn" title="New workspace">
        <Plus className="w-3 h-3" />
      </button>
    </div>
  )
}
