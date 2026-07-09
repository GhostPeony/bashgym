import { useState } from 'react'
import { Play, ChevronDown, ChevronRight } from 'lucide-react'
import { clsx } from 'clsx'
import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import { formatTokens } from './format'
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

/**
 * Past session (journal exists, no live terminal attached). Clicking selects
 * the row and reveals details plus explicit actions — nothing heavy happens
 * on a bare click.
 */
export function JournalSessionRow({ snapshot, onResume }: JournalSessionRowProps) {
  const [selected, setSelected] = useState(false)
  const when = snapshot.lastEventAt ?? snapshot.fileMtime
  const tokens = snapshot.totals.input + snapshot.totals.output

  return (
    <div
      className={clsx(
        'rounded-brutal transition-colors min-w-0 cursor-pointer',
        selected ? 'bg-accent/[0.08]' : 'hover:bg-accent/[0.05]'
      )}
      onClick={() => setSelected((v) => !v)}
    >
      <div className="flex items-center gap-2 px-1.5 py-1 min-w-0">
        {selected ? (
          <ChevronDown className="w-3 h-3 text-text-muted flex-shrink-0" />
        ) : (
          <ChevronRight className="w-3 h-3 text-text-muted flex-shrink-0" />
        )}
        <span className={clsx(KIND_CHIP_BASE, kindChipClass(snapshot.kind))}>{snapshot.kind}</span>
        <span className="font-mono text-[11px] text-text-secondary truncate flex-1">
          {snapshot.topic ?? snapshot.title ?? snapshot.sessionId?.slice(0, 8) ?? '—'}
        </span>
        <span className="font-mono text-[10px] text-text-muted flex-shrink-0">{timeAgo(when)}</span>
      </div>

      {selected && (
        <div className="px-1.5 pb-1.5 pl-6 space-y-1.5" onClick={(e) => e.stopPropagation()}>
          {snapshot.topic && (
            <p className="font-mono text-[10px] text-text-secondary leading-relaxed">{snapshot.topic}</p>
          )}
          <div className="flex items-center gap-1.5 font-mono text-[10px] text-text-muted flex-wrap">
            {snapshot.gitBranch && <span className="text-accent">⎇ {snapshot.gitBranch}</span>}
            {snapshot.model && <span className="truncate max-w-[110px]">{snapshot.model}</span>}
            {tokens > 0 && <span>{snapshot.totalsApprox ? '≈' : ''}{formatTokens(tokens)} tok</span>}
            {snapshot.estCostUsd !== undefined && snapshot.estCostUsd > 0.005 && (
              <span>${snapshot.estCostUsd.toFixed(2)}</span>
            )}
            {snapshot.sessionId && <span title={snapshot.filePath}>{snapshot.sessionId.slice(0, 8)}</span>}
          </div>
          {snapshot.sessionId && (
            <button
              onClick={() => onResume(snapshot)}
              className="node-btn node-btn-wide node-btn-accent"
              title={`Opens a new terminal in this project folder and resumes this conversation (${snapshot.kind === 'claude' ? 'claude --resume' : 'codex resume'})`}
            >
              <span className="flex items-center gap-1"><Play className="w-2.5 h-2.5" /> RESUME IN TERMINAL</span>
            </button>
          )}
        </div>
      )}
    </div>
  )
}
