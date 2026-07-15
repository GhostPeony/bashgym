import { useState } from 'react'
import { Check, LayoutGrid, Pencil, Plus, X } from 'lucide-react'
import { clsx } from 'clsx'
import { useWorkspaceStore } from '../../stores'
import type { WorkspaceSessionGroup } from '../../stores'

interface WorkspaceStripProps {
  groups: WorkspaceSessionGroup[]
}

/** Workspace ownership and health stay visible above the live-session feed. */
export function WorkspaceStrip({ groups }: WorkspaceStripProps) {
  const { workspaces, activeWorkspaceId, switchWorkspace, createWorkspace, renameWorkspace, deleteWorkspace } =
    useWorkspaceStore()
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [creating, setCreating] = useState(false)

  const commitName = () => {
    const name = draftName.trim()
    if (name && editingId) renameWorkspace(editingId, name)
    if (name && creating) createWorkspace(name, { activate: true })
    setEditingId(null)
    setCreating(false)
    setDraftName('')
  }

  const cancelEdit = () => {
    setEditingId(null)
    setCreating(false)
    setDraftName('')
  }

  return (
    <section aria-labelledby="workspace-switcher-title" className="space-y-1.5">
      <div className="flex items-center gap-2">
        <LayoutGrid className="w-3.5 h-3.5 text-accent" />
        <h2 id="workspace-switcher-title" className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-primary flex-1">
          Workspaces
        </h2>
        <button
          type="button"
          onClick={() => {
            setCreating(true)
            setEditingId(null)
            setDraftName('')
          }}
          className="node-btn"
          title="Create workspace"
          aria-label="Create workspace"
        >
          <Plus className="w-3 h-3" />
        </button>
      </div>

      <div className="space-y-1">
        {workspaces.map((workspace) => {
          const isActive = workspace.id === activeWorkspaceId
          const summary = groups.find((group) => group.workspace.id === workspace.id)
          const total = summary?.sessions.length ?? 0
          const live = summary?.liveCount ?? 0
          const waiting = summary?.waitingCount ?? 0

          if (editingId === workspace.id) {
            return (
              <div key={workspace.id} className="flex items-center gap-1">
                <input
                  value={draftName}
                  autoFocus
                  onChange={(event) => setDraftName(event.target.value)}
                  onBlur={commitName}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') commitName()
                    if (event.key === 'Escape') cancelEdit()
                  }}
                  className="input !py-1 !px-2 !text-[10px] font-mono flex-1 min-w-0"
                  aria-label={`Rename ${workspace.name}`}
                />
                <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={commitName} className="node-btn" aria-label="Save workspace name">
                  <Check className="w-3 h-3" />
                </button>
              </div>
            )
          }

          return (
            <div key={workspace.id} className="group flex items-center gap-1 min-w-0">
              <button
                type="button"
                onClick={() => switchWorkspace(workspace.id)}
                className={clsx(
                  'node-btn node-btn-wide flex-1 !justify-start min-w-0',
                  isActive ? 'node-btn-accent' : null
                )}
                aria-current={isActive ? 'page' : undefined}
                title={isActive ? `${workspace.name} is active` : `Switch to ${workspace.name}`}
              >
                <span className={clsx('status-dot flex-shrink-0', live > 0 ? 'status-success' : waiting > 0 ? 'status-warning' : '')} />
                <span className="truncate flex-1 text-left">{workspace.name}</span>
                <span className="text-[9px] opacity-70 flex-shrink-0">
                  {live > 0 ? `${live} live` : `${total} saved`}
                </span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setEditingId(workspace.id)
                  setCreating(false)
                  setDraftName(workspace.name)
                }}
                className="node-btn"
                title={`Rename ${workspace.name}`}
                aria-label={`Rename ${workspace.name}`}
              >
                <Pencil className="w-3 h-3" />
              </button>
              {!isActive && workspaces.length > 1 ? (
                <button
                  type="button"
                  onClick={() => void deleteWorkspace(workspace.id)}
                  className="node-btn hover:!text-status-error"
                  title={`Delete ${workspace.name}`}
                  aria-label={`Delete ${workspace.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              ) : null}
            </div>
          )
        })}

        {creating ? (
          <div className="flex items-center gap-1">
            <input
              value={draftName}
              autoFocus
              onChange={(event) => setDraftName(event.target.value)}
              onBlur={commitName}
              onKeyDown={(event) => {
                if (event.key === 'Enter') commitName()
                if (event.key === 'Escape') cancelEdit()
              }}
              className="input !py-1 !px-2 !text-[10px] font-mono flex-1 min-w-0"
              placeholder="Workspace name"
              aria-label="New workspace name"
            />
            <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={commitName} className="node-btn" aria-label="Create workspace">
              <Check className="w-3 h-3" />
            </button>
          </div>
        ) : null}
      </div>
    </section>
  )
}
