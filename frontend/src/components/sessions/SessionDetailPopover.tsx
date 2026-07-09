import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { clsx } from 'clsx'
import { Play, Maximize2, Pin, FolderGit2, X } from 'lucide-react'
import type { Panel, TerminalSession, CanvasEdge } from '../../stores'
import type { AgentSessionSnapshot, SessionMatch } from '../../services/agentSessions/types'
import { ContextMeter } from './ContextMeter'
import { QuickPrompt } from './QuickPrompt'
import { ConnectionsTree } from './ConnectionsTree'
import { formatTokens } from './format'
import { KIND_CHIP_BASE, kindChipClass } from './kindStyles'

export type SessionDetailTarget =
  | { type: 'live'; session: TerminalSession; panel: Panel; snapshot?: AgentSessionSnapshot; match?: SessionMatch }
  | { type: 'journal'; snapshot: AgentSessionSnapshot }

interface SessionDetailPopoverProps {
  target: SessionDetailTarget | null
  /** Viewport coordinates of the opening click */
  anchor: { x: number; y: number } | null
  onClose: () => void
  onFocus: (terminalId: string) => void
  onResume: (snapshot: AgentSessionSnapshot) => void
  onPin: (panelId: string, filePath: string | null) => void
  pinCandidates: AgentSessionSnapshot[]
  panels: Panel[]
  canvasEdges: CanvasEdge[]
}

const POPOVER_WIDTH = 340
const POPOVER_EST_HEIGHT = 380

function formatCost(cost: number): string {
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <span className="font-mono text-[9px] uppercase tracking-widest text-text-muted w-14 flex-shrink-0 pt-0.5">
        {label}
      </span>
      <div className="font-mono text-[11px] text-text-secondary min-w-0 flex-1">{children}</div>
    </div>
  )
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

/**
 * Session detail as a light popover anchored at the click location — no
 * backdrop, no scroll lock; the app behind stays visible and live. Click
 * outside or Esc dismisses. Heavy actions (open terminal, resume) are
 * explicit buttons. The Project row is where workspace-canvas grouping
 * slots in once named canvases land.
 */
export function SessionDetailPopover({
  target,
  anchor,
  onClose,
  onFocus,
  onResume,
  onPin,
  pinCandidates,
  panels,
  canvasEdges
}: SessionDetailPopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!target) return
    const handlePointerDown = (e: PointerEvent) => {
      if (!popoverRef.current?.contains(e.target as Node)) onClose()
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [target, onClose])

  if (!target || !anchor) return null

  const snapshot = target.snapshot
  const session = target.type === 'live' ? target.session : undefined
  const kind = session?.agentKind ?? snapshot?.kind
  const topic = snapshot?.topic ?? session?.taskSummary ?? snapshot?.title ?? session?.title ?? 'Session'
  const cwd = snapshot?.cwd ?? session?.cwd
  const limits = rateLimitTags(snapshot?.rateLimits)
  const tokens = snapshot ? snapshot.totals.input + snapshot.totals.output : 0

  const left = Math.max(8, Math.min(anchor.x, window.innerWidth - POPOVER_WIDTH - 8))
  const top = Math.max(8, Math.min(anchor.y, window.innerHeight - POPOVER_EST_HEIGHT - 8))

  return createPortal(
    <div
      ref={popoverRef}
      className="fixed z-50 bg-background-card border-brutal border-border shadow-brutal rounded-brutal overflow-hidden"
      style={{ left, top, width: POPOVER_WIDTH }}
    >
      {/* Title strip */}
      <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border min-w-0">
        <span className={clsx(KIND_CHIP_BASE, kindChipClass(kind))}>{kind ?? 'shell'}</span>
        <span className="font-mono text-[11px] font-semibold text-text-primary truncate flex-1" title={topic}>
          {topic}
        </span>
        <button onClick={onClose} className="node-btn flex-shrink-0" title="Close">
          <X className="w-3 h-3" />
        </button>
      </div>

      <div className="p-3 space-y-2 max-h-[50vh] overflow-y-auto">
        {session && (
          <Row label="Status">{session.status.replace('_', ' ')}</Row>
        )}
        {snapshot?.model && <Row label="Model">{snapshot.model}</Row>}

        {cwd && (
          <Row label="Project">
            <span className="flex items-center gap-1.5 min-w-0">
              <FolderGit2 className="w-3 h-3 text-accent flex-shrink-0" />
              <span className="truncate" title={cwd}>{cwd}</span>
              {snapshot?.gitBranch && <span className="text-accent flex-shrink-0">⎇ {snapshot.gitBranch}</span>}
            </span>
          </Row>
        )}

        {snapshot && (
          <Row label="Context">
            <ContextMeter
              contextTokens={snapshot.contextTokens}
              contextWindow={snapshot.contextWindow}
              approx={snapshot.contextWindowApprox}
              detailed
            />
          </Row>
        )}

        {snapshot && tokens > 0 && (
          <Row label="Usage">
            <span className="flex items-center gap-1.5 flex-wrap">
              <span>
                {snapshot.totalsApprox ? '≈' : ''}in {formatTokens(snapshot.totals.input)} · out {formatTokens(snapshot.totals.output)}
                {snapshot.totals.cacheRead > 0 && ` · cache ${formatTokens(snapshot.totals.cacheRead)}`}
              </span>
              {snapshot.estCostUsd !== undefined && snapshot.estCostUsd > 0 && (
                <span className="text-text-primary font-bold" title="Estimated API-equivalent cost">
                  {formatCost(snapshot.estCostUsd)}
                </span>
              )}
              {limits.map((tag) => (
                <span key={tag} className="px-1 border-brutal border-border-subtle rounded-brutal text-[10px]" title="Provider rate limit usage">
                  {tag}
                </span>
              ))}
            </span>
          </Row>
        )}

        {snapshot?.sessionId && (
          <Row label="Session">
            <span className="truncate block" title={snapshot.filePath}>{snapshot.sessionId}</span>
          </Row>
        )}

        {target.type === 'live' && (
          <>
            {!snapshot && pinCandidates.length > 0 && (
              <Row label="Journal">
                <select
                  className="input w-full !py-1 !px-2 !text-[10px] font-mono"
                  value=""
                  onChange={(e) => {
                    if (e.target.value) onPin(target.panel.id, e.target.value)
                  }}
                >
                  <option value="">no session file matched — pin one…</option>
                  {pinCandidates.map((c) => (
                    <option key={c.filePath} value={c.filePath}>
                      {c.kind} · {c.topic ?? c.title ?? c.sessionId?.slice(0, 8) ?? c.filePath}
                    </option>
                  ))}
                </select>
              </Row>
            )}
            {target.match?.confidence === 'manual' && (
              <Row label="Journal">
                <button
                  onClick={() => onPin(target.panel.id, null)}
                  className="flex items-center gap-1 text-text-muted hover:text-status-error transition-press"
                >
                  <Pin className="w-2.5 h-2.5" /> unpin session file
                </button>
              </Row>
            )}
            {(canvasEdges.some((e) => e.source === target.panel.id || e.target === target.panel.id) ||
              (snapshot?.recentFiles.length ?? 0) > 0) && (
              <Row label="Links">
                <ConnectionsTree
                  panelId={target.panel.id}
                  panels={panels}
                  canvasEdges={canvasEdges}
                  recentFiles={snapshot?.recentFiles ?? []}
                />
              </Row>
            )}
            <QuickPrompt terminalId={target.session.id} status={target.session.status} />
          </>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 px-3 py-2 border-t border-brutal border-border bg-background-secondary">
        {target.type === 'live' ? (
          <button
            onClick={() => { onFocus(target.session.id); onClose() }}
            className="node-btn node-btn-wide node-btn-accent"
            title="Open this terminal in the Workspace"
          >
            <span className="flex items-center gap-1"><Maximize2 className="w-2.5 h-2.5" /> OPEN TERMINAL</span>
          </button>
        ) : (
          target.snapshot.sessionId && (
            <button
              onClick={() => { onResume(target.snapshot); onClose() }}
              className="node-btn node-btn-wide node-btn-accent"
              title={`Opens a new terminal in this project folder and resumes this conversation (${target.snapshot.kind === 'claude' ? 'claude --resume' : 'codex resume'})`}
            >
              <span className="flex items-center gap-1"><Play className="w-2.5 h-2.5" /> RESUME IN TERMINAL</span>
            </button>
          )
        )}
      </div>
    </div>,
    document.body
  )
}
