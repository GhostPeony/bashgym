import { clsx } from 'clsx'
import { formatTokens } from './format'

interface ContextMeterProps {
  contextTokens?: number
  contextWindow?: number
  approx: boolean
  /** Labeled meter with percentages; default is a quiet thin bar */
  detailed?: boolean
}

/** Context-window occupancy for a session card */
export function ContextMeter({
  contextTokens,
  contextWindow,
  approx,
  detailed
}: ContextMeterProps) {
  if (contextTokens === undefined || !contextWindow) {
    return detailed ? <div className="font-mono text-[10px] text-text-muted">context: —</div> : null
  }

  const pct = Math.min((contextTokens / contextWindow) * 100, 100)
  const level = pct >= 95 ? 'error' : pct >= 80 ? 'warning' : 'ok'
  const label = `${approx ? '~' : ''}${Math.round(pct)}% of ${formatTokens(contextWindow)} context`
  const tooltip = `${contextTokens.toLocaleString()} tokens in context — ${label}${approx ? ' (window size assumed)' : ''}`

  if (!detailed) {
    return (
      <div className="progress-bar !h-1 opacity-70" title={tooltip}>
        <div
          className={clsx(
            'progress-fill',
            level === 'error' && '!bg-status-error',
            level === 'warning' && '!bg-status-warning'
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    )
  }

  return (
    <div title={tooltip}>
      <div className="flex items-center justify-between font-mono text-[10px] text-text-muted mb-0.5">
        <span>CONTEXT</span>
        <span
          className={clsx(
            level === 'error' && 'text-status-error',
            level === 'warning' && 'text-status-warning'
          )}
        >
          {label}
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
