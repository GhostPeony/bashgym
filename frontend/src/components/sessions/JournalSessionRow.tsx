import { Play } from 'lucide-react'
import { clsx } from 'clsx'
import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import { KIND_CHIP_BASE, kindChipClass } from './kindStyles'

interface JournalSessionRowProps {
  snapshot: AgentSessionSnapshot
  onResume: (snapshot: AgentSessionSnapshot) => void
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

/** Compact row for a past session (journal exists, no live terminal attached) */
export function JournalSessionRow({ snapshot, onResume }: JournalSessionRowProps) {
  const when = snapshot.lastEventAt ?? snapshot.fileMtime

  return (
    <div
      className="group flex items-center gap-2 px-1.5 py-1 rounded-brutal hover:bg-accent/[0.06] transition-colors min-w-0"
      title={snapshot.topic ?? snapshot.filePath}
    >
      <span className={clsx(KIND_CHIP_BASE, kindChipClass(snapshot.kind))}>{snapshot.kind}</span>
      <span className="font-mono text-[11px] text-text-secondary truncate flex-1">
        {snapshot.topic ?? snapshot.title ?? snapshot.sessionId?.slice(0, 8) ?? '—'}
      </span>
      <span className="font-mono text-[10px] text-text-muted flex-shrink-0">{timeAgo(when)}</span>
      {snapshot.sessionId && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onResume(snapshot)
          }}
          className="node-btn node-btn-accent flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
          title={`Resume this session in a new terminal (${snapshot.kind === 'claude' ? 'claude --resume' : 'codex resume'})`}
        >
          <Play className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}
