import { useState } from 'react'
import { ChevronLeft, ChevronRight, FileSearch, RefreshCw, TriangleAlert } from 'lucide-react'
import { datasetInspectKey, datasetInspectResource } from '../../stores/opsResources'
import { useKeyedSessionResource } from '../../stores/sessionResource'
import { clsx } from 'clsx'

const PAGE_SIZE = 5

const ROLE_COLORS: Record<string, string> = {
  system: 'text-status-info',
  user: 'text-accent',
  assistant: 'text-text-primary',
  tool: 'text-status-warning'
}

export function DatasetInspector() {
  const [offset, setOffset] = useState(0)
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)
  const {
    data: report,
    loading,
    refreshing,
    error,
    refresh
  } = useKeyedSessionResource(datasetInspectResource, datasetInspectKey(offset, PAGE_SIZE))
  const isFetching = loading || refreshing

  const total = report?.total ?? 0
  const canPrev = offset > 0
  const canNext = offset + PAGE_SIZE < total

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="flex items-center gap-2 text-sm font-mono font-semibold text-text-primary">
          <FileSearch className="w-4 h-4 text-accent" />
          Dataset Inspector
        </h3>
        <div className="flex items-center gap-2">
          {report && (
            <span className="text-xs font-mono text-text-muted">
              {total} examples
              {report.with_warnings_in_slice > 0 && (
                <span className="text-status-warning">
                  {' '}
                  · {report.with_warnings_in_slice} flagged on this page
                </span>
              )}
            </span>
          )}
          <button
            onClick={() => void refresh()}
            className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
            title="Refresh"
          >
            <RefreshCw className={clsx('w-3.5 h-3.5', isFetching && 'animate-spin')} />
          </button>
        </div>
      </div>

      {error && (
        <div className="text-xs font-mono text-text-muted py-2">
          {error.includes('404') || error.toLowerCase().includes('not found')
            ? 'No exported dataset found — export training examples first.'
            : error}
        </div>
      )}

      {report?.examples.map((example) => (
        <div
          key={example.index}
          className="border border-border-subtle mb-2 cursor-pointer"
          onClick={() => setExpandedIndex(expandedIndex === example.index ? null : example.index)}
        >
          <div className="flex items-center gap-2 px-3 py-1.5 bg-background-tertiary">
            <span className="text-xs font-mono text-text-muted">#{example.index}</span>
            <span className="text-xs font-mono text-text-secondary">
              {example.messages.length} messages
            </span>
            {example.warnings.length > 0 && (
              <span className="flex items-center gap-1 text-xs font-mono text-status-warning">
                <TriangleAlert className="w-3 h-3" />
                {example.warnings.length} warning{example.warnings.length > 1 ? 's' : ''}
              </span>
            )}
          </div>

          {example.warnings.length > 0 && (
            <div className="px-3 py-1.5 border-t border-border-subtle">
              {example.warnings.map((w, i) => (
                <div key={i} className="text-xs font-mono text-status-error">
                  {w}
                </div>
              ))}
            </div>
          )}

          {expandedIndex === example.index && (
            <div className="border-t border-border-subtle">
              {example.messages.map((msg, i) => (
                <div key={i} className="px-3 py-2 border-b border-border-subtle last:border-b-0">
                  <span
                    className={clsx(
                      'text-[11px] font-mono font-semibold uppercase',
                      ROLE_COLORS[msg.role ?? ''] || 'text-status-error'
                    )}
                  >
                    {msg.role ?? 'missing role'}
                  </span>
                  {typeof msg.content === 'string' && msg.content.length > 0 && (
                    <pre className="text-xs font-mono text-text-secondary whitespace-pre-wrap break-words mt-1 max-h-48 overflow-y-auto">
                      {msg.content}
                      {msg.truncated && <span className="text-text-muted"> …[truncated]</span>}
                    </pre>
                  )}
                  {msg.tool_calls != null && (
                    <pre className="text-xs font-mono text-status-info whitespace-pre-wrap break-words mt-1 max-h-48 overflow-y-auto">
                      {JSON.stringify(msg.tool_calls, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between mt-2">
          <button
            onClick={(e) => {
              e.stopPropagation()
              if (canPrev) setOffset(Math.max(0, offset - PAGE_SIZE))
            }}
            disabled={!canPrev}
            className="flex items-center gap-1 text-xs font-mono text-text-muted hover:text-text-primary disabled:opacity-40 transition-press"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            Prev
          </button>
          <span className="text-xs font-mono text-text-muted">
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation()
              if (canNext) setOffset(offset + PAGE_SIZE)
            }}
            disabled={!canNext}
            className="flex items-center gap-1 text-xs font-mono text-text-muted hover:text-text-primary disabled:opacity-40 transition-press"
          >
            Next
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
