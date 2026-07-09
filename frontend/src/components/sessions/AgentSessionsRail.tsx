import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, ListTree, RefreshCw, FolderGit2, ChevronDown, ChevronRight, Plus } from 'lucide-react'
import { clsx } from 'clsx'
import { useTerminalStore, useUIStore, useWorkspaceStore } from '../../stores'
import { useAgentSessionsStore } from '../../stores/agentSessionsStore'
import { WorkspaceStrip } from './WorkspaceStrip'
import { candidatesForTerminal, normPath } from '../../services/agentSessions/matching'
import type { AgentSessionSnapshot, AgentSessionKind } from '../../services/agentSessions/types'
import type { Panel, TerminalSession } from '../../stores'
import { SessionCard } from './SessionCard'
import { JournalSessionRow } from './JournalSessionRow'
import { SessionDetailPopover, type SessionDetailTarget } from './SessionDetailPopover'

type KindFilter = 'all' | AgentSessionKind

/** Popover target held as ids so the open popover live-updates from the stores */
type DetailRef =
  | { type: 'live'; panelId: string; x: number; y: number }
  | { type: 'journal'; filePath: string; x: number; y: number }

const JOURNALS_COLLAPSED_COUNT = 4

interface ProjectEntry {
  key: string
  /** Display path — best-cased source we saw */
  path: string
  name: string
  branch?: string
  live: Array<{ session: TerminalSession; panel: Panel }>
  journals: AgentSessionSnapshot[]
  lastActivity: number
}

function projectKeyOf(cwd: string): string {
  return normPath(cwd).toLowerCase()
}

function basenameOf(p: string): string {
  const norm = p.replace(/\\/g, '/').replace(/\/+$/, '')
  return norm.split('/').pop() || norm
}

function shortenPath(p: string): string {
  return p.replace(/^(C:\\Users\\[^\\]+|\/home\/[^/]+|\/Users\/[^/]+)/i, '~')
}

/**
 * Master feed of agent work, organized like a project browser: each project is
 * a folder on this machine (discovered from live terminals and the CLIs' own
 * session journals), with live terminals and resumable past sessions beneath.
 * Mounting starts journal polling; unmounting stops it.
 */
export function AgentSessionsRail() {
  const { setSidebarMode, closeOverlay } = useUIStore()
  const sessionsVersion = useTerminalStore((s) => s.sessionsVersion)
  const panels = useTerminalStore((s) => s.panels)
  const canvasEdges = useTerminalStore((s) => s.canvasEdges)
  const { snapshots, matches, lastScanAt, error, startPolling, stopPolling, pinSession, pollOnce } =
    useAgentSessionsStore()

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [showAllJournals, setShowAllJournals] = useState<Set<string>>(new Set())
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [detail, setDetail] = useState<DetailRef | null>(null)
  /** Project key showing the new-session launcher ('' = the header-level one) */
  const [launcherFor, setLauncherFor] = useState<string | null>(null)

  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  const projects = useMemo<ProjectEntry[]>(() => {
    const terminalSessions = useTerminalStore.getState().sessions
    const matchedPaths = new Set(Array.from(matches.values()).map((m) => m.filePath))
    const byKey = new Map<string, ProjectEntry>()

    const ensure = (cwd: string): ProjectEntry | null => {
      if (!cwd || cwd === '~') return null
      const key = projectKeyOf(cwd)
      let entry = byKey.get(key)
      if (!entry) {
        entry = { key, path: cwd, name: basenameOf(cwd), live: [], journals: [], lastActivity: 0 }
        byKey.set(key, entry)
      }
      return entry
    }

    // Live terminals anchor their projects (and carry real path casing)
    for (const panel of panels) {
      if (panel.type !== 'terminal' || !panel.terminalId) continue
      const session = terminalSessions.get(panel.terminalId)
      if (!session) continue
      const entry = ensure(session.cwd)
      if (!entry) continue
      entry.path = session.cwd
      entry.name = basenameOf(session.cwd)
      entry.live.push({ session, panel })
      entry.lastActivity = Math.max(entry.lastActivity, session.lastActivity)
    }

    // Past sessions from the journals (unmatched ones only — matched journals
    // render inside their live terminal's card)
    for (const snap of snapshots.values()) {
      if (!snap.cwd) continue
      const entry = ensure(snap.cwd)
      if (!entry) continue
      if (!entry.branch && snap.gitBranch) entry.branch = snap.gitBranch
      if (matchedPaths.has(snap.filePath)) continue
      entry.journals.push(snap)
      entry.lastActivity = Math.max(entry.lastActivity, snap.lastEventAt ?? snap.fileMtime)
    }

    for (const entry of byKey.values()) {
      entry.live.sort((a, b) => b.session.lastActivity - a.session.lastActivity)
      entry.journals.sort((a, b) => (b.lastEventAt ?? b.fileMtime) - (a.lastEventAt ?? a.fileMtime))
    }
    return Array.from(byKey.values()).sort((a, b) => b.lastActivity - a.lastActivity)
    // sessionsVersion drives recompute when live session state changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panels, matches, snapshots, sessionsVersion])

  const handleFocus = (terminalId: string) => {
    useTerminalStore.getState().setActiveTerminal(terminalId)
    closeOverlay()
  }

  const handleResume = (snap: AgentSessionSnapshot) => {
    if (!snap.sessionId) return
    const cmd = snap.kind === 'claude' ? `claude --resume ${snap.sessionId}` : `codex resume ${snap.sessionId}`
    const title = snap.title ?? (snap.cwd ? basenameOf(snap.cwd) : snap.kind)
    useTerminalStore.getState().createTerminal(undefined, title, cmd, snap.cwd)
    closeOverlay()
  }

  const handleLaunch = (kind: 'claude' | 'codex' | 'shell', cwd?: string) => {
    const title = kind === 'claude' ? 'Claude Code' : kind === 'codex' ? 'Codex' : 'Terminal'
    useTerminalStore.getState().createTerminal(undefined, title, kind === 'shell' ? undefined : kind, cwd)
    setLauncherFor(null)
    closeOverlay()
  }

  const handleNewWorkspace = () => {
    const name = window.prompt('Workspace name?')
    if (name?.trim()) {
      useWorkspaceStore.getState().createWorkspace(name, { activate: true })
      setLauncherFor(null)
      closeOverlay()
    }
  }

  const handleOpenInNewWorkspace = (snap: AgentSessionSnapshot) => {
    if (!snap.sessionId) return
    const ws = useWorkspaceStore.getState()
    const wsId = ws.createWorkspace(snap.cwd ? basenameOf(snap.cwd) : snap.kind)
    ws.switchWorkspace(wsId)
    const cmd = snap.kind === 'claude' ? `claude --resume ${snap.sessionId}` : `codex resume ${snap.sessionId}`
    const title = snap.title ?? (snap.cwd ? basenameOf(snap.cwd) : snap.kind)
    useTerminalStore.getState().createTerminal(undefined, title, cmd, snap.cwd)
    closeOverlay()
  }

  const handleMoveToNewWorkspace = (panelId: string) => {
    const state = useTerminalStore.getState()
    const panel = state.panels.find((p) => p.id === panelId)
    const session = panel?.terminalId ? state.sessions.get(panel.terminalId) : undefined
    const name = session?.cwd && session.cwd !== '~' ? basenameOf(session.cwd) : panel?.title ?? 'workspace'
    const ws = useWorkspaceStore.getState()
    const wsId = ws.createWorkspace(name)
    ws.moveLivePanelToWorkspace(panelId, wsId)
    closeOverlay()
  }

  const NewSessionLauncher = ({ cwd }: { cwd?: string }) => (
    <div className="flex items-center gap-1 py-1 flex-wrap">
      <button onClick={() => handleLaunch('claude', cwd)} className="node-btn node-btn-wide node-btn-accent">
        NEW CLAUDE
      </button>
      <button onClick={() => handleLaunch('codex', cwd)} className="node-btn node-btn-wide">
        NEW CODEX
      </button>
      <button onClick={() => handleLaunch('shell', cwd)} className="node-btn node-btn-wide">
        SHELL
      </button>
      <button
        onClick={handleNewWorkspace}
        className="node-btn node-btn-wide"
        title="Create a new named canvas workspace and switch to it"
      >
        NEW WORKSPACE
      </button>
    </div>
  )

  const toggle = (set: Set<string>, key: string, apply: (next: Set<string>) => void) => {
    const next = new Set(set)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    apply(next)
  }

  // Resolve the modal target fresh each render so it live-updates
  let detailTarget: SessionDetailTarget | null = null
  let detailPinCandidates: AgentSessionSnapshot[] = []
  if (detail?.type === 'live') {
    const panel = panels.find((p) => p.id === detail.panelId)
    const session = panel?.terminalId
      ? useTerminalStore.getState().sessions.get(panel.terminalId)
      : undefined
    if (panel && session) {
      const match = matches.get(session.id)
      const snapshot = match ? snapshots.get(match.filePath) : undefined
      detailTarget = { type: 'live', session, panel, snapshot, match }
      if (!snapshot) {
        detailPinCandidates = candidatesForTerminal(
          {
            terminalId: session.id,
            panelId: panel.id,
            cwd: session.cwd,
            agentKind: session.agentKind,
            lastActivity: session.lastActivity
          },
          snapshots
        )
      }
    }
  } else if (detail?.type === 'journal') {
    const snapshot = snapshots.get(detail.filePath)
    if (snapshot) detailTarget = { type: 'journal', snapshot }
  }

  const totalLive = projects.reduce((n, p) => n + p.live.length, 0)
  const totalJournals = projects.reduce((n, p) => n + p.journals.length, 0)

  return (
    <div className="p-3 space-y-3">
      {/* Workspace switcher */}
      <WorkspaceStrip />

      {/* Header */}
      <div className="space-y-2.5 pb-1">
        <div className="flex items-center gap-2">
          <button onClick={() => setSidebarMode('nav')} className="node-btn" title="Back to navigation">
            <ArrowLeft className="w-3 h-3" />
          </button>
          <ListTree className="w-3.5 h-3.5 text-accent" />
          <span className="font-mono text-xs font-bold uppercase tracking-widest text-text-primary flex-1">
            Agent Sessions
          </span>
          <button
            onClick={() => setLauncherFor(launcherFor === '' ? null : '')}
            className={clsx('node-btn', launcherFor === '' && 'node-btn-accent')}
            title="Start a new session"
          >
            <Plus className="w-3 h-3" />
          </button>
          <button
            onClick={() => void pollOnce()}
            className="node-btn"
            title={lastScanAt ? `Last scan ${new Date(lastScanAt).toLocaleTimeString()}` : 'Scan now'}
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
        {launcherFor === '' && <NewSessionLauncher />}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            {(['all', 'claude', 'codex'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setKindFilter(f)}
                className={clsx(
                  'px-2 py-0.5 border-brutal rounded-brutal font-mono text-[9px] font-bold uppercase tracking-wider transition-colors',
                  kindFilter === f
                    ? 'border-accent bg-accent/10 text-accent shadow-brutal-sm'
                    : 'border-border-subtle text-text-muted hover:text-text-primary'
                )}
              >
                {f}
              </button>
            ))}
          </div>
          <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">
            {projects.length} proj · {totalLive} live · {totalJournals} past
          </span>
        </div>
      </div>

      {error && (
        <div className="card p-2 border-l-2 border-l-status-warning">
          <p className="font-mono text-[10px] text-text-secondary">{error}</p>
        </div>
      )}

      {projects.length === 0 && !error && (
        <div className="card p-3">
          <p className="font-mono text-[10px] text-text-muted leading-relaxed">
            {lastScanAt
              ? 'No projects yet — sessions appear here as agents work in folders on this machine.'
              : 'Scanning local agent sessions…'}
          </p>
        </div>
      )}

      {projects.map((project) => {
        const isCollapsed = collapsed.has(project.key)
        const live = project.live.filter(
          ({ session }) => kindFilter === 'all' || (session.agentKind ?? 'shell') === kindFilter
        )
        const journals = project.journals.filter((s) => kindFilter === 'all' || s.kind === kindFilter)
        if (live.length === 0 && journals.length === 0) return null
        const liveCount = live.filter(
          ({ session }) => session.status === 'running' || session.status === 'tool_calling'
        ).length
        const journalsShown = showAllJournals.has(project.key)
          ? journals
          : journals.slice(0, JOURNALS_COLLAPSED_COUNT)

        return (
          <div key={project.key} className="pt-2 border-t border-border-subtle space-y-1.5">
            {/* Project header — a folder on this machine */}
            <div className="flex items-center gap-1 min-w-0">
              <button
                onClick={() => toggle(collapsed, project.key, setCollapsed)}
                className="flex-1 flex items-center gap-2 px-1 py-1 text-left hover:bg-accent/[0.06] rounded-brutal transition-colors min-w-0"
                title={project.path}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-3 h-3 text-text-muted flex-shrink-0" />
                ) : (
                  <ChevronDown className="w-3 h-3 text-text-muted flex-shrink-0" />
                )}
                <FolderGit2 className="w-3.5 h-3.5 text-accent flex-shrink-0" />
                <span className="flex-1 min-w-0">
                  <span className="block font-mono text-xs font-bold uppercase tracking-wider text-text-primary truncate">
                    {project.name}
                  </span>
                  <span className="block font-mono text-[9px] text-text-muted truncate">
                    {shortenPath(project.path)}
                    {project.branch && <span className="text-accent"> · ⎇ {project.branch}</span>}
                  </span>
                </span>
                {liveCount > 0 && (
                  <span className="flex items-center gap-1 flex-shrink-0">
                    <span className="status-dot status-success" />
                    <span className="font-mono text-[10px] text-status-success">{liveCount}</span>
                  </span>
                )}
                <span className="font-mono text-[10px] text-text-muted flex-shrink-0">
                  {live.length + journals.length}
                </span>
              </button>
              <button
                onClick={() => setLauncherFor(launcherFor === project.key ? null : project.key)}
                className={clsx('node-btn flex-shrink-0', launcherFor === project.key && 'node-btn-accent')}
                title={`Start a new session in ${project.name}`}
              >
                <Plus className="w-3 h-3" />
              </button>
            </div>
            {launcherFor === project.key && <NewSessionLauncher cwd={project.path} />}

            {!isCollapsed && (
              <div className="space-y-1 pb-1">
                {live.map(({ session, panel }) => {
                  const match = matches.get(session.id)
                  const snapshot = match ? snapshots.get(match.filePath) : undefined
                  return (
                    <SessionCard
                      key={panel.id}
                      session={session}
                      panel={panel}
                      snapshot={snapshot}
                      match={match}
                      onOpenDetail={(e) => setDetail({ type: 'live', panelId: panel.id, x: e.clientX, y: e.clientY })}
                    />
                  )
                })}

                {journalsShown.length > 0 && (
                  <div className="divide-y divide-border-subtle border-t border-b border-border-subtle">
                    {journalsShown.map((snap) => (
                      <JournalSessionRow
                        key={snap.filePath}
                        snapshot={snap}
                        onOpenDetail={(e) => setDetail({ type: 'journal', filePath: snap.filePath, x: e.clientX, y: e.clientY })}
                      />
                    ))}
                  </div>
                )}
                {journals.length > JOURNALS_COLLAPSED_COUNT && (
                  <button
                    onClick={() => toggle(showAllJournals, project.key, setShowAllJournals)}
                    className="font-mono text-[10px] text-text-muted hover:text-text-primary transition-press px-2"
                  >
                    {showAllJournals.has(project.key)
                      ? 'show fewer'
                      : `+${journals.length - JOURNALS_COLLAPSED_COUNT} more session${journals.length - JOURNALS_COLLAPSED_COUNT !== 1 ? 's' : ''}`}
                  </button>
                )}
              </div>
            )}
          </div>
        )
      })}

      <SessionDetailPopover
        target={detailTarget}
        anchor={detail ? { x: detail.x, y: detail.y } : null}
        onClose={() => setDetail(null)}
        onFocus={handleFocus}
        onResume={handleResume}
        onPin={pinSession}
        onOpenInNewWorkspace={handleOpenInNewWorkspace}
        onMoveToNewWorkspace={handleMoveToNewWorkspace}
        pinCandidates={detailPinCandidates}
        panels={panels}
        canvasEdges={canvasEdges}
      />
    </div>
  )
}

export default AgentSessionsRail
