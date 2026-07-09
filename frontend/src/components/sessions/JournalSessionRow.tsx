import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import { AgentBadge } from './AgentBadge'

interface JournalSessionRowProps {
  snapshot: AgentSessionSnapshot
  onOpenDetail: (e: React.MouseEvent) => void
}

function timeAgo(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

/**
 * Past session, one quiet line. Click opens the detail modal where resume
 * and (soon) workspace grouping live — no heavy actions on the row itself.
 */
export function JournalSessionRow({ snapshot, onOpenDetail }: JournalSessionRowProps) {
  const when = snapshot.lastEventAt ?? snapshot.fileMtime

  return (
    <div
      className="flex items-center gap-2 px-1.5 py-1 rounded-brutal hover:bg-accent/[0.06] transition-colors min-w-0 cursor-pointer"
      onClick={onOpenDetail}
      title={snapshot.topic ?? snapshot.filePath}
    >
      <AgentBadge kind={snapshot.kind} />
      <span className="font-mono text-[11px] text-text-secondary truncate flex-1">
        {snapshot.topic ?? snapshot.title ?? snapshot.sessionId?.slice(0, 8) ?? '—'}
      </span>
      <span className="font-mono text-[10px] text-text-muted flex-shrink-0">{timeAgo(when)}</span>
    </div>
  )
}
