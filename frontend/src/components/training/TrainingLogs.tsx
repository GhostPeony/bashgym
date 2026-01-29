import { useEffect, useRef, useState } from 'react'
import { Terminal, ChevronDown, ChevronUp, Trash2, Download } from 'lucide-react'
import { useTrainingStore, TrainingLog } from '../../stores'
import { clsx } from 'clsx'

interface TrainingLogsProps {
  maxHeight?: number
  defaultExpanded?: boolean
}

export function TrainingLogs({ maxHeight = 300, defaultExpanded = true }: TrainingLogsProps) {
  const { logs, clearLogs } = useTrainingStore()
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [autoScroll, setAutoScroll] = useState(true)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  // Detect manual scroll to disable auto-scroll
  const handleScroll = () => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
    setAutoScroll(isAtBottom)
  }

  const handleDownload = () => {
    const content = logs.map(l =>
      `[${new Date(l.timestamp).toISOString()}] [${l.level.toUpperCase()}] ${l.message}`
    ).join('\n')

    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training-logs-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const getLevelColor = (level: TrainingLog['level']) => {
    switch (level) {
      case 'error':
        return 'text-status-error'
      case 'warning':
        return 'text-status-warning'
      default:
        return 'text-text-secondary'
    }
  }

  return (
    <div className="card-elevated">
      {/* Header */}
      <div
        className="flex items-center justify-between p-3 cursor-pointer hover:bg-background-tertiary transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-text-muted" />
          <h3 className="text-sm font-medium text-text-primary">Training Logs</h3>
          <span className="px-2 py-0.5 text-xs bg-background-tertiary text-text-muted rounded-full">
            {logs.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {expanded && logs.length > 0 && (
            <>
              <button
                onClick={(e) => { e.stopPropagation(); handleDownload() }}
                className="p-1.5 hover:bg-background-elevated rounded transition-colors"
                title="Download logs"
              >
                <Download className="w-4 h-4 text-text-muted" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); clearLogs() }}
                className="p-1.5 hover:bg-background-elevated rounded transition-colors"
                title="Clear logs"
              >
                <Trash2 className="w-4 h-4 text-text-muted" />
              </button>
            </>
          )}
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-text-muted" />
          ) : (
            <ChevronDown className="w-4 h-4 text-text-muted" />
          )}
        </div>
      </div>

      {/* Logs Container */}
      {expanded && (
        <div
          ref={containerRef}
          onScroll={handleScroll}
          className="bg-background-primary border-t border-border overflow-auto font-mono text-xs"
          style={{ maxHeight }}
        >
          {logs.length === 0 ? (
            <div className="p-4 text-center text-text-muted">
              No training logs yet. Logs will appear here when training starts.
            </div>
          ) : (
            <div className="p-2 space-y-0.5">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={clsx(
                    'py-0.5 px-2 rounded hover:bg-background-tertiary',
                    getLevelColor(log.level)
                  )}
                >
                  <span className="text-text-muted mr-2">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="whitespace-pre-wrap break-all">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      )}

      {/* Auto-scroll indicator */}
      {expanded && logs.length > 0 && !autoScroll && (
        <div
          className="absolute bottom-2 right-4 px-2 py-1 bg-primary text-white text-xs rounded cursor-pointer"
          onClick={() => {
            setAutoScroll(true)
            logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
          }}
        >
          Scroll to bottom
        </div>
      )}
    </div>
  )
}
