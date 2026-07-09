import { useEffect, useMemo } from 'react'
import { ArrowLeft, ListTree, RefreshCw } from 'lucide-react'
import { useTerminalStore, useUIStore } from '../../stores'
import { useAgentSessionsStore } from '../../stores/agentSessionsStore'
import { candidatesForTerminal } from '../../services/agentSessions/matching'
import type { Panel, TerminalSession } from '../../stores'
import { SessionCard } from './SessionCard'
import { AccountChip } from './AccountChip'

interface RepoEntry {
  key: string
  label: string
  branch?: string
  items: Array<{ session: TerminalSession; panel: Panel }>
  lastActivity: number
}

function repoKeyOf(cwd: string): { key: string; label: string } {
  if (!cwd || cwd === '~') return { key: '~', label: 'home' }
  const norm = cwd.replace(/\\/g, '/').replace(/\/+$/, '')
  const label = norm.split('/').pop() || norm
  return { key: norm.toLowerCase(), label }
}

/**
 * Master feed of live agent terminals: repo-grouped cards with session intel
 * from local CLI journals. Mounting starts the journal polling; unmounting
 * stops it — the feed costs nothing while closed.
 */
export function AgentSessionsRail() {
  const { setSidebarMode, closeOverlay } = useUIStore()
  const sessionsVersion = useTerminalStore((s) => s.sessionsVersion)
  const panels = useTerminalStore((s) => s.panels)
  const canvasEdges = useTerminalStore((s) => s.canvasEdges)
  const { snapshots, matches, lastScanAt, error, startPolling, stopPolling, pinSession, pollOnce } =
    useAgentSessionsStore()

  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [startPolling, stopPolling])

  const groups = useMemo<RepoEntry[]>(() => {
    const terminalSessions = useTerminalStore.getState().sessions
    const byRepo = new Map<string, RepoEntry>()
    for (const panel of panels) {
      if (panel.type !== 'terminal' || !panel.terminalId) continue
      const session = terminalSessions.get(panel.terminalId)
      if (!session) continue
      const { key, label } = repoKeyOf(session.cwd)
      let entry = byRepo.get(key)
      if (!entry) {
        entry = { key, label, items: [], lastActivity: 0 }
        byRepo.set(key, entry)
      }
      entry.items.push({ session, panel })
      entry.lastActivity = Math.max(entry.lastActivity, session.lastActivity)
      if (!entry.branch) {
        const match = matches.get(session.id)
        const snap = match ? snapshots.get(match.filePath) : undefined
        if (snap?.gitBranch) entry.branch = snap.gitBranch
      }
    }
    const out = Array.from(byRepo.values())
    out.forEach((g) => g.items.sort((a, b) => b.session.lastActivity - a.session.lastActivity))
    return out.sort((a, b) => b.lastActivity - a.lastActivity)
    // sessionsVersion drives recompute when session state changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panels, matches, snapshots, sessionsVersion])

  const handleFocus = (terminalId: string) => {
    useTerminalStore.getState().setActiveTerminal(terminalId)
    closeOverlay()
  }

  const totalTerminals = groups.reduce((n, g) => n + g.items.length, 0)

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setSidebarMode('nav')}
          className="node-btn"
          title="Back to navigation"
        >
          <ArrowLeft className="w-3 h-3" />
        </button>
        <ListTree className="w-4 h-4 text-accent" />
        <h2 className="font-brand font-semibold text-base flex-1">Agent Sessions</h2>
        <button
          onClick={() => void pollOnce()}
          className="node-btn"
          title={lastScanAt ? `Last scan ${new Date(lastScanAt).toLocaleTimeString()}` : 'Scan now'}
        >
          <RefreshCw className="w-3 h-3" />
        </button>
      </div>

      {error && (
        <p className="font-mono text-[10px] text-status-error break-all">{error}</p>
      )}

      {totalTerminals === 0 && (
        <p className="font-mono text-xs text-text-muted">
          No terminals yet — open the Workspace and launch an agent.
        </p>
      )}

      {groups.map((group) => (
        <div key={group.key} className="space-y-2">
          <div className="flex items-center gap-2 px-1">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
              {group.label}
            </span>
            <span className="font-mono text-[10px] text-text-muted">
              {group.items.length}
            </span>
            {group.branch && (
              <span className="font-mono text-[10px] text-accent truncate">⎇ {group.branch}</span>
            )}
          </div>
          {group.items.map(({ session, panel }) => {
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
        </div>
      ))}

      <AccountChip />
    </div>
  )
}

export default AgentSessionsRail
