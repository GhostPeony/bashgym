import { useState } from 'react'
import { clsx } from 'clsx'
import { Pin, ChevronDown, ChevronRight, Maximize2, MessageSquare } from 'lucide-react'
import type { Panel, TerminalSession, CanvasEdge } from '../../stores'
import type { AgentSessionSnapshot, SessionMatch } from '../../services/agentSessions/types'
import { ContextMeter } from './ContextMeter'
import { QuickPrompt } from './QuickPrompt'
import { ConnectionsTree } from './ConnectionsTree'
import { formatTokens } from './format'
import { KIND_CHIP_BASE, kindChipClass } from './kindStyles'

interface SessionCardProps {
  session: TerminalSession
  panel: Panel
  snapshot?: AgentSessionSnapshot
  match?: SessionMatch
  pinCandidates: AgentSessionSnapshot[]
  onPin: (panelId: string, filePath: string | null) => void
  onFocus: (terminalId: string) => void
  panels: Panel[]
  canvasEdges: CanvasEdge[]
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

function formatCost(cost: number): string {
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

/** Best-effort % tags from the Codex rate_limits payload */
function rateLimitTags(rl?: Record<string, unknown>): string[] {
  if (!rl) return []
  const tags: string[] = []
  for (const key of ['primary', 'secondary'] as const) {
    const entry = rl[key] as Record<string, unknown> | undefined
    const pct = Number(entry?.used_percent)
    if (!entry || !Number.isFinite(pct)) continue
    const mins = Number(entry.window_minutes)
    const label = Number.isFinite(mins) && mins > 0
      ? mins >= 10_080 ? `${Math.round(mins / 1_440)}d` : mins >= 60 ? `${Math.round(mins / 60)}h` : `${mins}m`
      : key
    tags.push(`${label} ${Math.round(pct)}%`)
  }
  return tags
}

const STATUS_DOT: Record<string, string> = {
  running: 'status-success',
  tool_calling: 'status-success',
  waiting_input: 'status-warning',
  idle: ''
}

export function SessionCard({
  session,
  panel,
  snapshot,
  match,
  pinCandidates,
  onPin,
  onFocus,
  panels,
  canvasEdges
}: SessionCardProps) {
  const [showPrompt, setShowPrompt] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const kind = session.agentKind ?? snapshot?.kind
  const limits = rateLimitTags(snapshot?.rateLimits)
  const showPinPicker = !snapshot && pinCandidates.length > 0

  // What the session is ABOUT — journal topic beats the generic terminal name
  const topic = snapshot?.topic ?? session.taskSummary ?? snapshot?.title ?? session.title

  return (
    <div
      className={clsx(
        'card p-2 space-y-1.5 cursor-pointer',
        !session.isPaused && session.status === 'running' && 'terminal-status-running',
        !session.isPaused && session.status === 'tool_calling' && 'terminal-status-tool-calling',
        !session.isPaused && session.status === 'waiting_input' && 'terminal-status-waiting-input'
      )}
      onClick={() => setExpanded((v) => !v)}
      onContextMenu={(e) => {
        e.preventDefault()
        setShowPrompt((v) => !v)
      }}
      title={`${session.title} — click for details · right-click to prompt`}
    >
      {/* Topic line */}
      <div className="flex items-center gap-1.5 min-w-0">
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-text-muted flex-shrink-0" />
        ) : (
          <ChevronRight className="w-3 h-3 text-text-muted flex-shrink-0" />
        )}
        <span className={clsx('status-dot flex-shrink-0', STATUS_DOT[session.status])} />
        <span className="font-mono text-[11px] font-semibold text-text-primary truncate flex-1" title={topic}>
          {topic}
        </span>
        <span className={clsx(KIND_CHIP_BASE, kindChipClass(kind))}>{kind ?? 'shell'}</span>
        <span className="font-mono text-[10px] text-text-muted flex-shrink-0">{timeAgo(session.lastActivity)}</span>
      </div>

      {/* Quiet context bar (collapsed) */}
      {!expanded && (
        <ContextMeter
          contextTokens={snapshot?.contextTokens}
          contextWindow={snapshot?.contextWindow}
          approx={snapshot?.contextWindowApprox ?? true}
        />
      )}

      {expanded && (
        <div className="space-y-1.5" onClick={(e) => e.stopPropagation()}>
          {/* Explicit actions — opening the terminal swaps the workspace view,
              so it's a deliberate button, never a side effect of selecting */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => onFocus(session.id)}
              className="node-btn node-btn-wide node-btn-accent"
              title="Open this terminal in the Workspace"
            >
              <span className="flex items-center gap-1"><Maximize2 className="w-2.5 h-2.5" /> OPEN TERMINAL</span>
            </button>
            <button
              onClick={() => setShowPrompt((v) => !v)}
              className="node-btn node-btn-wide"
              title="Send a prompt into this agent without opening it (also: right-click the card)"
            >
              <span className="flex items-center gap-1"><MessageSquare className="w-2.5 h-2.5" /> PROMPT</span>
            </button>
          </div>
          <ContextMeter
            contextTokens={snapshot?.contextTokens}
            contextWindow={snapshot?.contextWindow}
            approx={snapshot?.contextWindowApprox ?? true}
            detailed
          />
          {snapshot && (
            <div className="flex items-center gap-1.5 font-mono text-[10px] text-text-muted min-w-0 flex-wrap">
              {match?.confidence === 'manual' && (
                <span title="Manually pinned to this session file"><Pin className="w-2.5 h-2.5 text-accent" /></span>
              )}
              {match?.confidence === 'probable' && <span title="Best-guess match — pin to lock it">~</span>}
              {snapshot.gitBranch && <span className="text-accent">⎇ {snapshot.gitBranch}</span>}
              {snapshot.model && <span className="truncate max-w-[110px]">{snapshot.model}</span>}
              {snapshot.totals.input + snapshot.totals.output > 0 && (
                <span>
                  {snapshot.totalsApprox ? '≈' : ''}in {formatTokens(snapshot.totals.input)} · out {formatTokens(snapshot.totals.output)}
                  {snapshot.totals.cacheRead > 0 && ` · cache ${formatTokens(snapshot.totals.cacheRead)}`}
                </span>
              )}
              {snapshot.estCostUsd !== undefined && snapshot.estCostUsd > 0 && (
                <span className="text-text-primary font-bold" title="Estimated API-equivalent cost for this session">
                  {formatCost(snapshot.estCostUsd)}
                </span>
              )}
              {limits.map((tag) => (
                <span key={tag} className="px-1 border-brutal border-border-subtle rounded-brutal" title="Provider rate limit usage">
                  {tag}
                </span>
              ))}
            </div>
          )}

          {showPinPicker && (
            <select
              className="input w-full !py-1 !px-2 !text-[10px] font-mono"
              value=""
              onChange={(e) => {
                if (e.target.value) onPin(panel.id, e.target.value)
              }}
            >
              <option value="">no session file matched — pin one…</option>
              {pinCandidates.map((c) => (
                <option key={c.filePath} value={c.filePath}>
                  {c.kind} · {c.topic ?? c.title ?? c.sessionId?.slice(0, 8) ?? c.filePath} · {timeAgo(c.fileMtime)}
                </option>
              ))}
            </select>
          )}
          {match?.confidence === 'manual' && (
            <button
              onClick={() => onPin(panel.id, null)}
              className="font-mono text-[10px] text-text-muted hover:text-status-error transition-press"
            >
              unpin session file
            </button>
          )}

          <ConnectionsTree
            panelId={panel.id}
            panels={panels}
            canvasEdges={canvasEdges}
            recentFiles={snapshot?.recentFiles ?? []}
          />
        </div>
      )}

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
