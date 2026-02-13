import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronUp, Trash2, Download } from 'lucide-react'
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
    <div className="terminal-chrome relative">
      {/* Terminal Header */}
      <div className="terminal-header">
        <div className="flex items-center gap-2 flex-1 cursor-pointer" onClick={() => setExpanded(!expanded)}>
          <div className="terminal-dot terminal-dot-red" />
          <div className="terminal-dot terminal-dot-yellow" />
          <div className="terminal-dot terminal-dot-green" />
          <span className="font-mono text-xs uppercase tracking-widest text-text-muted ml-2">Training Logs</span>
          <span className="tag ml-2"><span>{logs.length}</span></span>
        </div>
        <div className="flex items-center gap-1">
          {expanded && logs.length > 0 && (
            <>
              <button
                onClick={(e) => { e.stopPropagation(); handleDownload() }}
                className="btn-ghost p-1.5"
                title="Download logs"
              >
                <Download className="w-4 h-4 text-text-muted" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); clearLogs() }}
                className="btn-ghost p-1.5"
                title="Clear logs"
              >
                <Trash2 className="w-4 h-4 text-text-muted" />
              </button>
            </>
          )}
          <button onClick={() => setExpanded(!expanded)} className="btn-ghost p-1.5">
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-text-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-text-muted" />
            )}
          </button>
        </div>
      </div>

      {/* Logs Container */}
      {expanded && (
        <div
          ref={containerRef}
          onScroll={handleScroll}
          className="bg-background-terminal overflow-auto font-mono text-xs"
          style={{ maxHeight }}
        >
          {logs.length === 0 ? (
            <div className="p-4 text-center text-text-muted">
              <span className="terminal-prompt">$</span> No training logs yet. Logs will appear here when training starts.
            </div>
          ) : (
            <div className="p-2 space-y-0.5">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={clsx(
                    'py-0.5 px-2',
                    getLevelColor(log.level)
                  )}
                >
                  <span className="text-text-muted mr-2">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="terminal-prompt mr-1">{'>'}</span>
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
          className="absolute bottom-2 right-4 tag cursor-pointer"
          onClick={() => {
            setAutoScroll(true)
            logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
          }}
        >
          <span>Scroll to bottom</span>
        </div>
      )}
    </div>
  )
}
