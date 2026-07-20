import { useEffect, useMemo, useState } from 'react'
import { RefreshCw, Download, Search } from 'lucide-react'
import { clsx } from 'clsx'
import { API_BASE } from '../../services/api'
import {
  trainingLogKey,
  trainingLogResource,
  trainingRunOptionsResource,
} from '../../stores/opsResources'
import { useSessionResource } from '../../stores/sessionResource'

function classifyLine(line: string): 'error' | 'warning' | 'info' {
  const head = line.slice(0, 40).toUpperCase()
  if (head.includes('ERROR') || head.includes('TRACEBACK') || head.includes('FATAL')) return 'error'
  if (head.includes('WARN')) return 'warning'
  return 'info'
}

export function TrainingLogViewer() {
  const { data: runsData } = useSessionResource(trainingRunOptionsResource)
  const runs = useMemo(() => runsData ?? [], [runsData])
  const [selectedRun, setSelectedRun] = useState<string>('')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [tail, setTail] = useState(500)

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 200)
    return () => clearTimeout(t)
  }, [search])

  // Default to the most recent run once the run list is available
  useEffect(() => {
    if (!selectedRun && runs.length > 0) setSelectedRun(runs[0].id)
  }, [runs, selectedRun])

  const logKey = selectedRun ? trainingLogKey(selectedRun, tail) : null
  const entry = trainingLogResource((s) => (logKey ? s.entries[logKey] : undefined))
  useEffect(() => {
    if (logKey) void trainingLogResource.getState().ensureLoaded(logKey)
  }, [logKey])

  const log = entry?.data ?? null
  const loading = (entry?.loading ?? false) || (entry?.refreshing ?? false)
  const error = entry?.error ?? null

  const fetchLog = () => {
    if (logKey) void trainingLogResource.getState().refresh(logKey)
  }

  const filtered = useMemo(() => {
    if (!log) return []
    if (!debouncedSearch) return log.lines
    const needle = debouncedSearch.toLowerCase()
    return log.lines.filter((l) => l.toLowerCase().includes(needle))
  }, [log, debouncedSearch])

  const downloadHref = useMemo(() => {
    if (!selectedRun) return '#'
    return `${API_BASE}/training/${encodeURIComponent(selectedRun)}/log?tail=0`
  }, [selectedRun])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="font-brand text-xl text-text-primary">Training Logs</h2>
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="input font-mono text-xs max-w-xs"
          >
            {runs.length === 0 && <option value="">No runs available</option>}
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id}
                {r.strategy ? ` · ${r.strategy}` : ''}
                {r.status ? ` · ${r.status}` : ''}
              </option>
            ))}
          </select>
          <select
            value={tail}
            onChange={(e) => setTail(Number(e.target.value))}
            className="input font-mono text-xs"
          >
            <option value={100}>Last 100</option>
            <option value={500}>Last 500</option>
            <option value={2000}>Last 2000</option>
            <option value={5000}>Last 5000</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchLog}
            disabled={loading || !selectedRun}
            className="btn-icon flex items-center justify-center"
            title="Refresh"
          >
            <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
          </button>
          <a
            href={downloadHref}
            download
            className={clsx(
              'btn-secondary flex items-center gap-2',
              !selectedRun && 'opacity-50 pointer-events-none'
            )}
          >
            <Download className="w-4 h-4" />
            <span className="font-mono text-xs">Full log</span>
          </a>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <Search className="w-4 h-4 text-text-muted" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter lines (substring)"
          className="input font-mono text-xs flex-1"
        />
        {log && (
          <span className="font-mono text-xs text-text-muted whitespace-nowrap">
            {filtered.length} / {log.lines.length} lines
            {log.truncated && ` (of ${log.total_lines})`}
          </span>
        )}
      </div>

      {error && (
        <div className="card p-3 border-l-4 border-l-status-error bg-status-error/10 mb-3">
          <p className="font-mono text-xs text-status-error">{error}</p>
        </div>
      )}

      <pre className="card p-3 bg-background-secondary overflow-auto max-h-[60vh] font-mono text-xs leading-relaxed">
        {log === null && !error && (
          <span className="text-text-muted">
            {loading ? 'Loading...' : selectedRun ? 'Select a run to view logs' : 'No run selected'}
          </span>
        )}
        {log && filtered.length === 0 && (
          <span className="text-text-muted">No lines match filter</span>
        )}
        {log &&
          filtered.map((line, idx) => {
            const severity = classifyLine(line)
            return (
              <div
                key={idx}
                className={clsx(
                  'whitespace-pre-wrap break-words',
                  severity === 'error' && 'text-status-error',
                  severity === 'warning' && 'text-status-warning'
                )}
              >
                <span className="text-text-muted select-none mr-3">
                  {String(idx + 1).padStart(4, ' ')}
                </span>
                {line}
              </div>
            )
          })}
      </pre>
    </div>
  )
}
