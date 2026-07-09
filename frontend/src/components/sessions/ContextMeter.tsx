import { clsx } from 'clsx'
import { formatTokens } from './format'

interface ContextMeterProps {
  contextTokens?: number
  contextWindow?: number
  approx: boolean
}

/** Context-window occupancy bar for a session card */
export function ContextMeter({ contextTokens, contextWindow, approx }: ContextMeterProps) {
  if (contextTokens === undefined || !contextWindow) {
    return (
      <div className="font-mono text-[10px] text-text-muted">context: —</div>
    )
  }

  const pct = Math.min((contextTokens / contextWindow) * 100, 100)
  const level = pct >= 95 ? 'error' : pct >= 80 ? 'warning' : 'ok'

  return (
    <div title={`${contextTokens.toLocaleString()} tokens in context${approx ? ' (window size assumed)' : ''}`}>
      <div className="flex items-center justify-between font-mono text-[10px] text-text-muted mb-0.5">
        <span>CONTEXT</span>
        <span className={clsx(
          level === 'error' && 'text-status-error',
          level === 'warning' && 'text-status-warning'
        )}>
          {approx ? '~' : ''}{Math.round(pct)}% of {formatTokens(contextWindow)}
        </span>
      </div>
      <div className="progress-bar !h-1.5">
        <div
          className={clsx(
            'progress-fill',
            level === 'error' && '!bg-status-error',
            level === 'warning' && '!bg-status-warning'
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
