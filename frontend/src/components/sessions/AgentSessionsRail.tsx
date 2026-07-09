import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, ListTree, RefreshCw, FolderGit2, ChevronDown, ChevronRight } from 'lucide-react'
import { useTerminalStore, useUIStore } from '../../stores'
import { useAgentSessionsStore } from '../../stores/agentSessionsStore'
import { candidatesForTerminal, normPath } from '../../services/agentSessions/matching'
import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import type { Panel, TerminalSession } from '../../stores'
import { SessionCard } from './SessionCard'
import { JournalSessionRow } from './JournalSessionRow'
import { AccountChip } from './AccountChip'

const JOURNALS_COLLAPSED_COUNT = 4

interface ProjectEntry {
  key: string
  /** Display path — best-cased source we saw */
  path: string
  name: string
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
      if (matchedPaths.has(snap.filePath) || !snap.cwd) continue
      const entry = ensure(snap.cwd)
      if (!entry) continue
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

  const toggle = (set: Set<string>, key: string, apply: (next: Set<string>) => void) => {
    const next = new Set(set)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    apply(next)
  }

  return (
    <div className="p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2">
        <button onClick={() => setSidebarMode('nav')} className="node-btn" title="Back to navigation">
          <ArrowLeft className="w-3 h-3" />
        </button>
        <ListTree className="w-4 h-4 text-accent" />
        <h2 className="font-brand font-semibold text-sm flex-1">Agent Sessions</h2>
        <button
          onClick={() => void pollOnce()}
          className="node-btn"
          title={lastScanAt ? `Last scan ${new Date(lastScanAt).toLocaleTimeString()}` : 'Scan now'}
        >
          <RefreshCw className="w-3 h-3" />
        </button>
      </div>

      {error && <p className="font-mono text-[10px] text-status-error break-all">{error}</p>}

      {projects.length === 0 && (
        <p className="font-mono text-xs text-text-muted">
          {lastScanAt
            ? 'No projects found yet — sessions appear here as agents work in folders on this machine.'
            : 'Scanning local agent sessions…'}
        </p>
      )}

      {projects.map((project) => {
        const isCollapsed = collapsed.has(project.key)
        const liveCount = project.live.filter(
          ({ session }) => session.status === 'running' || session.status === 'tool_calling'
        ).length
        const journalsShown = showAllJournals.has(project.key)
          ? project.journals
          : project.journals.slice(0, JOURNALS_COLLAPSED_COUNT)

        return (
          <div key={project.key} className="space-y-1.5">
            {/* Project header — a folder on this machine */}
            <button
              onClick={() => toggle(collapsed, project.key, setCollapsed)}
              className="w-full flex items-center gap-2 px-1 py-1 text-left hover:bg-accent/[0.06] rounded-brutal transition-colors min-w-0"
              title={project.path}
            >
              {isCollapsed ? (
                <ChevronRight className="w-3 h-3 text-text-muted flex-shrink-0" />
              ) : (
                <ChevronDown className="w-3 h-3 text-text-muted flex-shrink-0" />
              )}
              <FolderGit2 className="w-3.5 h-3.5 text-accent flex-shrink-0" />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-text-primary truncate">
                {project.name}
              </span>
              <span className="font-mono text-[10px] text-text-muted truncate flex-1">
                {shortenPath(project.path)}
              </span>
              {liveCount > 0 && (
                <span className="flex items-center gap-1 flex-shrink-0">
                  <span className="status-dot status-success" />
                  <span className="font-mono text-[10px] text-status-success">{liveCount}</span>
                </span>
              )}
              <span className="font-mono text-[10px] text-text-muted flex-shrink-0">
                {project.live.length + project.journals.length}
              </span>
            </button>

            {!isCollapsed && (
              <div className="space-y-1.5 pl-2 border-l border-brutal border-border-subtle ml-1.5">
                {project.live.map(({ session, panel }) => {
                  const match = matches.get(session.id)
                  const snapshot = match ? snapshots.get(match.filePath) : undefined
                  return (
                    <SessionCard
                      key={panel.id}
                      session={session}
                      panel={panel}
                      snapshot={snapshot}
                      match={match}
                      pinCandidates={
                        snapshot
                          ? []
                          : candidatesForTerminal(
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
                      onPin={pinSession}
                      onFocus={handleFocus}
                      panels={panels}
                      canvasEdges={canvasEdges}
                    />
                  )
                })}

                {journalsShown.map((snap) => (
                  <JournalSessionRow key={snap.filePath} snapshot={snap} onResume={handleResume} />
                ))}
                {project.journals.length > JOURNALS_COLLAPSED_COUNT && (
                  <button
                    onClick={() => toggle(showAllJournals, project.key, setShowAllJournals)}
                    className="font-mono text-[10px] text-text-muted hover:text-text-primary transition-press px-2"
                  >
                    {showAllJournals.has(project.key)
                      ? 'show fewer'
                      : `+${project.journals.length - JOURNALS_COLLAPSED_COUNT} more session${project.journals.length - JOURNALS_COLLAPSED_COUNT !== 1 ? 's' : ''}`}
                  </button>
                )}
              </div>
            )}
          </div>
        )
      })}

      <AccountChip />
    </div>
  )
}

export default AgentSessionsRail
