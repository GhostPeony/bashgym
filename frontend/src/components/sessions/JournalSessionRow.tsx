import { Play } from 'lucide-react'
import { clsx } from 'clsx'
import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import { formatTokens } from './format'

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
  const tokens = snapshot.totals.input + snapshot.totals.output

  return (
    <div
      className="flex items-center gap-2 px-2 py-1.5 border-brutal border-border-subtle rounded-brutal bg-background-secondary/50 min-w-0"
      title={snapshot.filePath}
    >
      <span
        className={clsx(
          'flex-shrink-0 px-1 py-px border-brutal rounded-brutal text-[8px] font-bold uppercase tracking-wider font-mono',
          snapshot.kind === 'claude'
            ? 'border-accent/60 bg-accent/10 text-accent'
            : 'border-status-warning/60 bg-status-warning/10 text-status-warning'
        )}
      >
        {snapshot.kind}
      </span>
      <span className="font-mono text-[11px] text-text-secondary truncate flex-1">
        {snapshot.title ?? snapshot.sessionId?.slice(0, 8) ?? '—'}
      </span>
      <span className="font-mono text-[10px] text-text-muted flex-shrink-0">
        {tokens > 0 && `${snapshot.totalsApprox ? '≈' : ''}${formatTokens(tokens)} tok · `}
        {snapshot.estCostUsd !== undefined && snapshot.estCostUsd > 0.005 && `$${snapshot.estCostUsd.toFixed(2)} · `}
        {timeAgo(when)}
      </span>
      {snapshot.sessionId && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onResume(snapshot)
          }}
          className="node-btn node-btn-accent flex-shrink-0"
          title={`Resume this session in a new terminal (${snapshot.kind === 'claude' ? 'claude --resume' : 'codex resume'})`}
        >
          <Play className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}
