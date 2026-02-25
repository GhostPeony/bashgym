import { clsx } from 'clsx'
import { TraceSpan } from '../../services/api'

interface SpanTimelineProps {
  spans: TraceSpan[]
  totalDurationMs: number
}

export function SpanTimeline({ spans, totalDurationMs }: SpanTimelineProps) {
  if (spans.length === 0) return null

  const maxDuration = totalDurationMs || Math.max(...spans.map(s => s.duration_ms), 1)

  return (
    <div className="space-y-1.5">
      {spans.map(span => {
        const widthPct = Math.max((span.duration_ms / maxDuration) * 100, 2)
        const isLlm = span.kind === 'llm_call'
        const isTool = span.kind === 'tool_call'
        const isError = span.status === 'error'

        return (
          <div key={span.span_id} className="flex items-center gap-3">
            <div className="w-28 flex-shrink-0 truncate font-mono text-xs text-text-muted" title={span.name}>
              {span.name}
            </div>
            <div className="flex-1 h-5 bg-background-secondary border border-border-subtle rounded-sm overflow-hidden relative">
              <div
                className={clsx(
                  'h-full rounded-sm transition-all duration-300',
                  isError ? 'bg-status-error' :
                  isLlm ? 'bg-accent' :
                  isTool ? 'bg-status-success' :
                  'bg-text-muted'
                )}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <div className="w-20 flex-shrink-0 text-right font-mono text-xs text-text-muted">
              {span.duration_ms.toFixed(0)}ms
            </div>
          </div>
        )
      })}
      {/* Legend */}
      <div className="flex items-center gap-4 pt-2 font-mono text-xs text-text-muted">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-accent" />
          LLM
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-status-success" />
          Tool
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-status-error" />
          Error
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-text-muted" />
          Other
        </span>
      </div>
    </div>
  )
}
