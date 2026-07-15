import { useState } from 'react'
import { clsx } from 'clsx'
import type { TerminalSession } from '../../stores'
import type { SessionRuntimeState } from '../../stores'
import type { AgentSessionSnapshot } from '../../services/agentSessions/types'
import { QuickPrompt } from './QuickPrompt'
import { AgentBadge } from './AgentBadge'
import { folderNameFromPath } from './format'
import { MessageSquare } from 'lucide-react'

interface SessionCardProps {
  session: TerminalSession
  snapshot?: AgentSessionSnapshot
  onOpenDetail: (e: React.MouseEvent) => void
  runtimeState?: SessionRuntimeState
}

function timeAgo(timestamp: number): string {
  if (timestamp <= 0) return 'saved'
  const seconds = Math.floor((Date.now() - timestamp) / 1000)
  if (seconds < 60) return 'now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

const STATUS_DOT: Record<string, string> = {
  running: 'status-success',
  tool_calling: 'status-success',
  waiting_input: 'status-warning',
  idle: ''
}

/**
 * One live terminal in the feed: a slim row (status, topic, agent, age) with
 * a quiet context bar. Click opens the detail modal; right-click summons the
 * quick prompt in place. Nothing heavier happens from the row itself.
 */
export function SessionCard({
  session,
  snapshot,
  onOpenDetail,
  runtimeState = 'unknown'
}: SessionCardProps) {
  const [showPrompt, setShowPrompt] = useState(false)

  const kind = session.agentKind ?? snapshot?.kind
  const topic = snapshot?.topic ?? session.taskSummary ?? snapshot?.title ?? session.title
  const projectName = folderNameFromPath(snapshot?.cwd ?? session.cwd, session.title)
  const statusLabel = runtimeState === 'saved'
    ? 'saved'
    : session.isPaused
      ? 'paused'
      : session.status === 'tool_calling'
        ? 'working'
        : session.status.replace('_', ' ')

  return (
    <article
      className="w-full min-w-0 overflow-hidden rounded-brutal bg-background-secondary px-2.5 py-2 space-y-2 transition-colors hover:bg-accent/[0.06]"
    >
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-2 min-w-0">
        <AgentBadge kind={kind} />
        <div className="min-w-0 space-y-1">
          <button
            type="button"
            onClick={onOpenDetail}
            className="block w-full min-w-0 truncate text-left text-xs font-semibold text-text-primary hover:text-accent transition-colors"
            title={`Open details for ${topic}`}
          >
            {topic}
          </button>
          <div className="flex min-w-0 items-center gap-1.5 font-mono text-[9px] text-text-muted">
            <span className={clsx('status-dot flex-shrink-0', STATUS_DOT[session.status])} />
            <span className="min-w-0 truncate text-text-secondary">{projectName}</span>
            <span aria-hidden="true" className="flex-shrink-0 text-border">·</span>
            <span className="flex-shrink-0">{statusLabel}</span>
            <span aria-hidden="true" className="flex-shrink-0 text-border">·</span>
            <span className="flex-shrink-0">{timeAgo(session.lastActivity)}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowPrompt((visible) => !visible)}
          className={clsx('node-btn flex-shrink-0', showPrompt ? 'node-btn-accent' : null)}
          title={`Prompt ${session.title}`}
          aria-label={`Prompt ${session.title}`}
          aria-expanded={showPrompt}
        >
          <MessageSquare className="w-3 h-3" />
        </button>
      </div>

      {showPrompt ? (
        <QuickPrompt
          terminalId={session.id}
          status={session.status}
          autoFocus
          onClose={() => setShowPrompt(false)}
        />
      ) : null}
    </article>
  )
}
