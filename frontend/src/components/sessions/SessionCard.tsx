import { useState } from 'react'
import { clsx } from 'clsx'
import type { Panel, TerminalSession } from '../../stores'
import type { AgentSessionSnapshot, SessionMatch } from '../../services/agentSessions/types'
import { ContextMeter } from './ContextMeter'
import { QuickPrompt } from './QuickPrompt'
import { AgentBadge } from './AgentBadge'

interface SessionCardProps {
  session: TerminalSession
  panel: Panel
  snapshot?: AgentSessionSnapshot
  match?: SessionMatch
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
export function SessionCard({ session, panel: _panel, snapshot, match: _match, onOpenDetail }: SessionCardProps) {
  const [showPrompt, setShowPrompt] = useState(false)

  const kind = session.agentKind ?? snapshot?.kind
  const topic = snapshot?.topic ?? session.taskSummary ?? snapshot?.title ?? session.title

  return (
    <div
      className={clsx(
        'card p-2 space-y-1.5 cursor-pointer',
        !session.isPaused && session.status === 'running' && 'terminal-status-running',
        !session.isPaused && session.status === 'tool_calling' && 'terminal-status-tool-calling',
        !session.isPaused && session.status === 'waiting_input' && 'terminal-status-waiting-input'
      )}
      onClick={onOpenDetail}
      onContextMenu={(e) => {
        e.preventDefault()
        setShowPrompt((v) => !v)
      }}
      title={`${session.title} — click for details · right-click to prompt`}
    >
      <div className="flex items-center gap-1.5 min-w-0">
        <span className={clsx('status-dot flex-shrink-0', STATUS_DOT[session.status])} />
        <span className="font-mono text-[11px] font-semibold text-text-primary truncate flex-1" title={topic}>
          {topic}
        </span>
        <AgentBadge kind={kind} />
        <span className="font-mono text-[10px] text-text-muted flex-shrink-0">{timeAgo(session.lastActivity)}</span>
      </div>

      <ContextMeter
        contextTokens={snapshot?.contextTokens}
        contextWindow={snapshot?.contextWindow}
        approx={snapshot?.contextWindowApprox ?? true}
      />

      {showPrompt && (
        <QuickPrompt
          terminalId={session.id}
          status={session.status}
          autoFocus
          onClose={() => setShowPrompt(false)}
        />
      )}
    </div>
  )
}
