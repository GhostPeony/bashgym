import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Pencil, Plus, X } from 'lucide-react'
import { clsx } from 'clsx'
import { useWorkspaceStore } from '../../stores'
import type { WorkspaceSessionGroup } from '../../stores'

interface WorkspaceSwitcherProps {
  groups: WorkspaceSessionGroup[]
}

/**
 * Compact workspace control: one row shows the active workspace and its live
 * count; clicking opens an on-demand dropdown with the full list plus rename /
 * delete / create. The list never expands inline, so it can't push the session
 * feed down or bury the header no matter how many workspaces exist.
 */
export function WorkspaceSwitcher({ groups }: WorkspaceSwitcherProps) {
  const {
    workspaces,
    activeWorkspaceId,
    switchWorkspace,
    createWorkspace,
    renameWorkspace,
    deleteWorkspace
  } = useWorkspaceStore()
  const [open, setOpen] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [creating, setCreating] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId)
  const activeSummary = groups.find((group) => group.workspace.id === activeWorkspaceId)
  const activeLive = activeSummary?.liveCount ?? 0
  const activeWaiting = activeSummary?.waitingCount ?? 0
  const activeTotal = activeSummary?.sessions.length ?? 0

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  const resetDrafts = () => {
    setEditingId(null)
    setCreating(false)
    setDraftName('')
  }

  const commitName = () => {
    const name = draftName.trim()
    if (name && editingId) renameWorkspace(editingId, name)
    if (name && creating) createWorkspace(name, { activate: true })
    resetDrafts()
    if (creating) setOpen(false)
  }

  const startCreate = () => {
    setCreating(true)
    setEditingId(null)
    setDraftName('')
  }

  const handleSwitch = (workspaceId: string) => {
    switchWorkspace(workspaceId)
    resetDrafts()
    setOpen(false)
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={clsx(
          'node-btn node-btn-wide w-full !justify-start min-w-0',
          open ? 'node-btn-accent' : null
        )}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Switch or manage workspaces"
      >
        <span
          className={clsx(
            'status-dot flex-shrink-0',
            activeLive > 0 ? 'status-success' : activeWaiting > 0 ? 'status-warning' : ''
          )}
        />
        <span className="truncate flex-1 text-left">{activeWorkspace?.name ?? 'Workspace'}</span>
        <span className="text-[9px] opacity-70 flex-shrink-0">
          {activeLive > 0 ? `${activeLive} live` : `${activeTotal} saved`}
        </span>
        <ChevronDown
          className={clsx('w-3 h-3 flex-shrink-0 transition-transform', open ? 'rotate-180' : null)}
        />
      </button>

      {open ? (
        <div className="absolute left-0 right-0 top-full z-40 mt-1 bg-background-card border-brutal border-border shadow-brutal rounded-brutal overflow-hidden">
          <div className="flex items-center gap-2 px-2.5 py-1.5 bg-background-secondary border-b border-brutal border-border">
            <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-text-muted flex-1">
              Workspaces
            </span>
            <button
              type="button"
              onClick={startCreate}
              className="node-btn"
              title="New workspace"
              aria-label="New workspace"
            >
              <Plus className="w-3 h-3" />
            </button>
          </div>

          <div className="max-h-[45vh] overflow-y-auto p-1 space-y-0.5">
            {workspaces.map((workspace) => {
              const isActive = workspace.id === activeWorkspaceId
              const summary = groups.find((group) => group.workspace.id === workspace.id)
              const live = summary?.liveCount ?? 0
              const waiting = summary?.waitingCount ?? 0
              const total = summary?.sessions.length ?? 0

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
                        if (event.key === 'Escape') resetDrafts()
                      }}
                      className="input !py-1 !px-2 !text-[10px] font-mono flex-1 min-w-0"
                      aria-label={`Rename ${workspace.name}`}
                    />
                    <button
                      type="button"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={commitName}
                      className="node-btn"
                      aria-label="Save workspace name"
                    >
                      <Check className="w-3 h-3" />
                    </button>
                  </div>
                )
              }

              return (
                <div key={workspace.id} className="group flex items-center gap-1 min-w-0">
                  <button
                    type="button"
                    onClick={() => handleSwitch(workspace.id)}
                    className={clsx(
                      'node-btn node-btn-wide flex-1 !justify-start min-w-0',
                      isActive ? 'node-btn-accent' : null
                    )}
                    aria-current={isActive ? 'page' : undefined}
                    title={isActive ? `${workspace.name} is active` : `Switch to ${workspace.name}`}
                  >
                    <span
                      className={clsx(
                        'status-dot flex-shrink-0',
                        live > 0 ? 'status-success' : waiting > 0 ? 'status-warning' : ''
                      )}
                    />
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
                    className="node-btn opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                    title={`Rename ${workspace.name}`}
                    aria-label={`Rename ${workspace.name}`}
                  >
                    <Pencil className="w-3 h-3" />
                  </button>
                  {!isActive && workspaces.length > 1 ? (
                    <button
                      type="button"
                      onClick={() => void deleteWorkspace(workspace.id)}
                      className="node-btn node-btn-danger opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
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
                    if (event.key === 'Escape') resetDrafts()
                  }}
                  className="input !py-1 !px-2 !text-[10px] font-mono flex-1 min-w-0"
                  placeholder="Workspace name"
                  aria-label="New workspace name"
                />
                <button
                  type="button"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={commitName}
                  className="node-btn"
                  aria-label="Create workspace"
                >
                  <Check className="w-3 h-3" />
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
