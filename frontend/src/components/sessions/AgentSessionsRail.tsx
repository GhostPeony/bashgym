import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Clock3,
  FolderGit2,
  Plus,
  RefreshCw
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  buildWorkspaceSessionIndex,
  useTerminalStore,
  useUIStore,
  useWorkspaceStore
} from '../../stores'
import type { WorkspaceSessionRecord } from '../../stores'
import { useAgentSessionsStore } from '../../stores/agentSessionsStore'
import { WorkspaceSwitcher } from './WorkspaceSwitcher'
import { candidatesForTerminal, normPath } from '../../services/agentSessions/matching'
import type { AgentSessionSnapshot, AgentSessionKind } from '../../services/agentSessions/types'
import { SessionCard } from './SessionCard'
import { JournalSessionRow } from './JournalSessionRow'
import { SessionDetailPopover, type SessionDetailTarget } from './SessionDetailPopover'
import { folderNameFromPath } from './format'

type KindFilter = 'all' | AgentSessionKind

type DetailRef =
  | { type: 'live'; workspaceId: string; panelId: string; x: number; y: number }
  | { type: 'journal'; filePath: string; x: number; y: number }

interface ProjectEntry {
  key: string
  path: string
  name: string
  branch?: string
  journals: AgentSessionSnapshot[]
  lastActivity: number
}

const JOURNALS_COLLAPSED_COUNT = 4
const JOURNALS_EXPANDED_LIMIT = 50

function projectKeyOf(cwd: string): string {
  return normPath(cwd).toLowerCase()
}

interface NewSessionLauncherProps {
  cwd?: string
  onLaunch: (kind: 'claude' | 'codex' | 'shell', cwd?: string) => void
}

function NewSessionLauncher({ cwd, onLaunch }: NewSessionLauncherProps) {
  return (
    <div className="grid grid-cols-3 gap-1 py-1">
      <button
        type="button"
        onClick={() => onLaunch('claude', cwd)}
        className="node-btn node-btn-wide node-btn-accent"
      >
        CLAUDE
      </button>
      <button
        type="button"
        onClick={() => onLaunch('codex', cwd)}
        className="node-btn node-btn-wide"
      >
        CODEX
      </button>
      <button
        type="button"
        onClick={() => onLaunch('shell', cwd)}
        className="node-btn node-btn-wide"
      >
        SHELL
      </button>
    </div>
  )
}

/**
 * Workspace-first companion rail: live processes keep their canvas ownership,
 * while historical Claude/Codex journals live in a separate project archive.
 */
export function AgentSessionsRail() {
  const setSidebarMode = useUIStore((state) => state.setSidebarMode)
  const closeOverlay = useUIStore((state) => state.closeOverlay)
  const presentWorkspacePanel = useUIStore((state) => state.presentWorkspacePanel)
  const panels = useTerminalStore((state) => state.panels)
  const sessionsVersion = useTerminalStore((state) => state.sessionsVersion)
  const canvasEdges = useTerminalStore((state) => state.canvasEdges)
  const workspaces = useWorkspaceStore((state) => state.workspaces)
  const activeWorkspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const sessionIndexVersion = useWorkspaceStore((state) => state.sessionIndexVersion)
  const {
    snapshots,
    matches,
    liveTerminalIds,
    hasLiveTerminalSnapshot,
    lastScanAt,
    error,
    startPolling,
    stopPolling,
    pinSession,
    pollOnce
  } = useAgentSessionsStore()

  const [collapsedWorkspaces, setCollapsedWorkspaces] = useState<Set<string>>(new Set())
  const [collapsedProjects, setCollapsedProjects] = useState<Set<string>>(new Set())
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set())
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [detail, setDetail] = useState<DetailRef | null>(null)
  const [launcherFor, setLauncherFor] = useState<string | null>(null)

  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  const sessionIndexInvalidation = `${sessionIndexVersion}:${sessionsVersion}`
  const workspaceGroups = useMemo(() => {
    void sessionIndexInvalidation
    return buildWorkspaceSessionIndex({
      workspaces,
      activeWorkspaceId,
      activePanels: panels,
      activeSessions: useTerminalStore.getState().sessions,
      liveTerminalIds: hasLiveTerminalSnapshot ? liveTerminalIds : undefined
    })
  }, [
    activeWorkspaceId,
    hasLiveTerminalSnapshot,
    liveTerminalIds,
    panels,
    sessionIndexInvalidation,
    workspaces
  ])

  const projects = useMemo<ProjectEntry[]>(() => {
    const matchedPaths = new Set(Array.from(matches.values()).map((match) => match.filePath))
    const byKey = new Map<string, ProjectEntry>()

    for (const snapshot of snapshots.values()) {
      if (!snapshot.cwd || matchedPaths.has(snapshot.filePath)) continue
      const key = projectKeyOf(snapshot.cwd)
      let project = byKey.get(key)
      if (!project) {
        project = {
          key,
          path: snapshot.cwd,
          name: folderNameFromPath(snapshot.cwd),
          branch: snapshot.gitBranch,
          journals: [],
          lastActivity: 0
        }
        byKey.set(key, project)
      }
      if (!project.branch && snapshot.gitBranch) project.branch = snapshot.gitBranch
      project.journals.push(snapshot)
      project.lastActivity = Math.max(
        project.lastActivity,
        snapshot.lastEventAt ?? snapshot.fileMtime
      )
    }

    for (const project of byKey.values()) {
      project.journals.sort(
        (a, b) => (b.lastEventAt ?? b.fileMtime) - (a.lastEventAt ?? a.fileMtime)
      )
    }
    return Array.from(byKey.values()).sort((a, b) => b.lastActivity - a.lastActivity)
  }, [matches, snapshots])

  const toggleSet = useCallback(
    (current: Set<string>, key: string, apply: (next: Set<string>) => void) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      apply(next)
    },
    []
  )

  const handleFocus = useCallback(
    (panelId: string, workspaceId: string) => {
      closeOverlay()
      presentWorkspacePanel(workspaceId, panelId, 'peek')
    },
    [closeOverlay, presentWorkspacePanel]
  )

  const handleResume = useCallback(
    (snapshot: AgentSessionSnapshot) => {
      if (!snapshot.sessionId) return
      const command =
        snapshot.kind === 'claude'
          ? `claude --resume ${snapshot.sessionId}`
          : `codex resume ${snapshot.sessionId}`
      const title = snapshot.title ?? folderNameFromPath(snapshot.cwd, snapshot.kind)
      useTerminalStore.getState().createTerminal(undefined, title, command, snapshot.cwd)
      closeOverlay()
    },
    [closeOverlay]
  )

  const handleLaunch = useCallback(
    (kind: 'claude' | 'codex' | 'shell', cwd?: string) => {
      const title = kind === 'claude' ? 'Claude Code' : kind === 'codex' ? 'Codex' : 'Terminal'
      useTerminalStore
        .getState()
        .createTerminal(undefined, title, kind === 'shell' ? undefined : kind, cwd)
      setLauncherFor(null)
      closeOverlay()
    },
    [closeOverlay]
  )

  const handleOpenInNewWorkspace = useCallback(
    (snapshot: AgentSessionSnapshot) => {
      if (!snapshot.sessionId) return
      const workspaceStore = useWorkspaceStore.getState()
      const workspaceId = workspaceStore.createWorkspace(
        folderNameFromPath(snapshot.cwd, snapshot.kind)
      )
      workspaceStore.switchWorkspace(workspaceId)
      const command =
        snapshot.kind === 'claude'
          ? `claude --resume ${snapshot.sessionId}`
          : `codex resume ${snapshot.sessionId}`
      const title = snapshot.title ?? folderNameFromPath(snapshot.cwd, snapshot.kind)
      useTerminalStore.getState().createTerminal(undefined, title, command, snapshot.cwd)
      closeOverlay()
    },
    [closeOverlay]
  )

  const handleMoveToWorkspace = useCallback((panelId: string, workspaceId: string) => {
    useWorkspaceStore
      .getState()
      .moveLivePanelToWorkspace(panelId, workspaceId, { switchAfter: false })
  }, [])

  const handleMoveToNewWorkspace = useCallback(
    (panelId: string) => {
      const record = workspaceGroups
        .flatMap((group) => group.sessions)
        .find((candidate) => candidate.panel.id === panelId)
      if (!record || !record.isActiveWorkspace) return
      const name = folderNameFromPath(record.session.cwd, record.panel.title)
      const workspaceStore = useWorkspaceStore.getState()
      const workspaceId = workspaceStore.createWorkspace(name)
      workspaceStore.moveLivePanelToWorkspace(panelId, workspaceId, { switchAfter: false })
    },
    [workspaceGroups]
  )

  const liveRecords = useMemo(
    () => workspaceGroups.flatMap((group) => group.sessions),
    [workspaceGroups]
  )
  let detailTarget: SessionDetailTarget | null = null
  let detailPinCandidates: AgentSessionSnapshot[] = []
  if (detail?.type === 'live') {
    const record = liveRecords.find(
      (candidate) =>
        candidate.workspaceId === detail.workspaceId && candidate.panel.id === detail.panelId
    )
    if (record) {
      const match = matches.get(record.session.id)
      const snapshot = match ? snapshots.get(match.filePath) : undefined
      detailTarget = {
        type: 'live',
        session: record.session,
        panel: record.panel,
        workspaceId: record.workspaceId,
        workspaceName: record.workspaceName,
        snapshot,
        match
      }
      if (!snapshot && record.isActiveWorkspace) {
        detailPinCandidates = candidatesForTerminal(
          {
            terminalId: record.session.id,
            panelId: record.panel.id,
            cwd: record.session.cwd,
            agentKind: record.session.agentKind,
            lastActivity: record.session.lastActivity
          },
          snapshots
        )
      }
    }
  } else if (detail?.type === 'journal') {
    const snapshot = snapshots.get(detail.filePath)
    if (snapshot) detailTarget = { type: 'journal', snapshot }
  }

  const renderLiveSession = (record: WorkspaceSessionRecord) => {
    const match = matches.get(record.session.id)
    const snapshot = match ? snapshots.get(match.filePath) : undefined
    return (
      <SessionCard
        key={`${record.workspaceId}:${record.panel.id}`}
        session={record.session}
        snapshot={snapshot}
        runtimeState={record.runtimeState}
        onOpenDetail={(event) =>
          setDetail({
            type: 'live',
            workspaceId: record.workspaceId,
            panelId: record.panel.id,
            x: event.clientX,
            y: event.clientY
          })
        }
      />
    )
  }

  return (
    <div className="p-3 space-y-4 min-h-full">
      {/* Sticky top bar — MENU stays reachable and is never buried under the
          workspace list, no matter how far the feed is scrolled. */}
      <div className="sticky top-0 z-30 -mx-3 -mt-3 px-3 pt-3 pb-2 bg-background-card border-b border-border-subtle flex min-w-0 items-center gap-1.5">
        <button
          type="button"
          onClick={() => setSidebarMode('nav')}
          className="node-btn node-btn-wide flex-shrink-0"
          title="Back to menu"
          aria-label="Back to menu"
        >
          <span className="flex items-center gap-1">
            <ArrowLeft className="w-3 h-3" /> MENU
          </span>
        </button>
        <h1 className="min-w-0 flex-1 truncate font-mono text-[11px] font-bold uppercase tracking-wider text-text-primary">
          Agent Sessions
        </h1>
      </div>

      <header className="space-y-2">
        <WorkspaceSwitcher groups={workspaceGroups} />
        <div className="flex min-w-0 items-center gap-1">
          <div className="flex min-w-0 flex-1 items-center gap-1">
            {(['all', 'claude', 'codex'] as const).map((filter) => (
              <button
                type="button"
                key={filter}
                onClick={() => setKindFilter(filter)}
                className={clsx(
                  'px-2 py-0.5 border-brutal rounded-brutal font-mono text-[9px] font-bold uppercase tracking-wider transition-colors',
                  kindFilter === filter
                    ? 'border-accent bg-accent/10 text-accent shadow-brutal-sm'
                    : 'border-border-subtle text-text-muted hover:text-text-primary'
                )}
                aria-pressed={kindFilter === filter}
              >
                {filter}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setLauncherFor(launcherFor === '' ? null : '')}
            className={clsx(
              'node-btn flex-shrink-0',
              launcherFor === '' ? 'node-btn-accent' : null
            )}
            title="Start a new session"
            aria-expanded={launcherFor === ''}
          >
            <Plus className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={() => void pollOnce()}
            className="node-btn flex-shrink-0"
            title={
              lastScanAt
                ? `Last refreshed ${new Date(lastScanAt).toLocaleTimeString()}`
                : 'Refresh session index'
            }
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
        {launcherFor === '' ? <NewSessionLauncher onLaunch={handleLaunch} /> : null}
      </header>

      {error ? (
        <div className="card p-2 border-l-2 border-l-status-warning">
          <p className="font-mono text-[10px] text-text-secondary">{error}</p>
        </div>
      ) : null}

      <section aria-labelledby="live-session-title" className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="status-dot status-success" />
          <h2
            id="live-session-title"
            className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-primary flex-1"
          >
            Live and saved terminals
          </h2>
          <span className="font-mono text-[9px] text-text-muted">{liveRecords.length}</span>
        </div>

        {workspaceGroups.map((group) => {
          const filtered = group.sessions.filter(
            ({ session }) => kindFilter === 'all' || session.agentKind === kindFilter
          )
          if (filtered.length === 0) return null
          const isCollapsed = collapsedWorkspaces.has(group.workspace.id)
          return (
            <div
              key={group.workspace.id}
              className="border-t border-border-subtle pt-1.5 space-y-1.5"
            >
              <button
                type="button"
                onClick={() =>
                  toggleSet(collapsedWorkspaces, group.workspace.id, setCollapsedWorkspaces)
                }
                className="w-full flex items-center gap-2 px-1 py-1 text-left hover:bg-accent/[0.06] rounded-brutal min-w-0"
                aria-expanded={!isCollapsed}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-3 h-3 text-text-muted" />
                ) : (
                  <ChevronDown className="w-3 h-3 text-text-muted" />
                )}
                <span className="font-mono text-[11px] font-bold uppercase tracking-wider text-text-primary truncate flex-1">
                  {group.workspace.name}
                </span>
                {group.isActive ? (
                  <span className="font-mono text-[8px] uppercase tracking-wider text-accent">
                    active
                  </span>
                ) : null}
                {group.waitingCount > 0 ? (
                  <span className="font-mono text-[9px] text-status-warning">
                    {group.waitingCount} waiting
                  </span>
                ) : null}
                <span className="font-mono text-[9px] text-text-muted">{filtered.length}</span>
              </button>
              {!isCollapsed ? (
                <div className="space-y-1">{filtered.map(renderLiveSession)}</div>
              ) : null}
            </div>
          )
        })}

        {liveRecords.length === 0 ? (
          <div className="card p-3 font-mono text-[10px] text-text-muted">
            No terminal sessions are saved in these workspaces.
          </div>
        ) : null}
      </section>

      <section
        aria-labelledby="session-history-title"
        className="space-y-2 pt-2 border-t border-border-subtle"
      >
        <div className="flex items-center gap-2">
          <Clock3 className="w-3.5 h-3.5 text-text-muted" />
          <h2
            id="session-history-title"
            className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-primary flex-1"
          >
            Conversation history
          </h2>
          <span className="font-mono text-[9px] text-text-muted">by project</span>
        </div>

        {projects.map((project) => {
          const journals = project.journals.filter(
            (snapshot) => kindFilter === 'all' || snapshot.kind === kindFilter
          )
          if (journals.length === 0) return null
          const isCollapsed = collapsedProjects.has(project.key)
          const isExpanded = expandedProjects.has(project.key)
          const shown = isExpanded
            ? journals.slice(0, JOURNALS_EXPANDED_LIMIT)
            : journals.slice(0, JOURNALS_COLLAPSED_COUNT)
          return (
            <div key={project.key} className="border-t border-border-subtle pt-1.5 space-y-1.5">
              <div className="group flex items-center gap-1 min-w-0">
                <button
                  type="button"
                  onClick={() => toggleSet(collapsedProjects, project.key, setCollapsedProjects)}
                  className="flex-1 flex items-center gap-2 px-1 py-1 text-left hover:bg-accent/[0.06] rounded-brutal min-w-0"
                  title={`Open ${project.name} sessions`}
                  aria-expanded={!isCollapsed}
                >
                  {isCollapsed ? (
                    <ChevronRight className="w-3 h-3 text-text-muted" />
                  ) : (
                    <ChevronDown className="w-3 h-3 text-text-muted" />
                  )}
                  <FolderGit2 className="w-3.5 h-3.5 text-accent" />
                  <span className="flex-1 min-w-0">
                    <span className="block font-mono text-[11px] font-bold uppercase tracking-wider text-text-primary truncate">
                      {project.name}
                    </span>
                    {project.branch ? (
                      <span className="block truncate font-mono text-[9px] text-accent">
                        ⎇ {project.branch}
                      </span>
                    ) : null}
                  </span>
                  <span className="font-mono text-[9px] text-text-muted">{journals.length}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setLauncherFor(launcherFor === project.key ? null : project.key)}
                  className={clsx(
                    'node-btn transition-opacity focus:opacity-100',
                    launcherFor === project.key
                      ? 'node-btn-accent opacity-100'
                      : 'opacity-0 group-hover:opacity-100'
                  )}
                  title={`Start a session in ${project.name}`}
                  aria-expanded={launcherFor === project.key}
                >
                  <Plus className="w-3 h-3" />
                </button>
              </div>
              {launcherFor === project.key ? (
                <NewSessionLauncher cwd={project.path} onLaunch={handleLaunch} />
              ) : null}
              {!isCollapsed ? (
                <div className="divide-y divide-border-subtle border-y border-border-subtle">
                  {shown.map((snapshot) => (
                    <JournalSessionRow
                      key={snapshot.filePath}
                      snapshot={snapshot}
                      onOpenDetail={(event) =>
                        setDetail({
                          type: 'journal',
                          filePath: snapshot.filePath,
                          x: event.clientX,
                          y: event.clientY
                        })
                      }
                    />
                  ))}
                </div>
              ) : null}
              {!isCollapsed && journals.length > JOURNALS_COLLAPSED_COUNT ? (
                <button
                  type="button"
                  onClick={() => toggleSet(expandedProjects, project.key, setExpandedProjects)}
                  className="font-mono text-[10px] text-text-muted hover:text-text-primary px-2"
                >
                  {isExpanded
                    ? 'show fewer'
                    : `show ${Math.min(journals.length, JOURNALS_EXPANDED_LIMIT) - JOURNALS_COLLAPSED_COUNT} more`}
                </button>
              ) : null}
            </div>
          )
        })}

        {!lastScanAt && projects.length === 0 ? (
          <div className="space-y-2" aria-label="Loading session history">
            {[0, 1, 2].map((item) => (
              <div
                key={item}
                className="h-9 border border-border-subtle bg-background-secondary animate-pulse rounded-brutal"
              />
            ))}
          </div>
        ) : null}
      </section>

      <SessionDetailPopover
        target={detailTarget}
        anchor={detail ? { x: detail.x, y: detail.y } : null}
        onClose={() => setDetail(null)}
        onFocus={handleFocus}
        onResume={handleResume}
        onPin={pinSession}
        onOpenInNewWorkspace={handleOpenInNewWorkspace}
        onMoveToWorkspace={handleMoveToWorkspace}
        onMoveToNewWorkspace={handleMoveToNewWorkspace}
        pinCandidates={detailPinCandidates}
        panels={panels}
        canvasEdges={canvasEdges}
      />
    </div>
  )
}

export default AgentSessionsRail
