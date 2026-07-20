import { useState } from 'react'
import {
  Search,
  RefreshCw,
  Loader2,
  FileText,
  BarChart3,
  Database,
  Play,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { API_BASE } from '../../services/api'
import { hfResearchResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'

const API_URL = API_BASE

async function fetchJSON(path: string, opts?: RequestInit) {
  try {
    const res = await fetch(`${API_URL}${path}`, opts)
    const text = await res.text()
    try { return { ok: res.ok, data: JSON.parse(text), error: undefined } }
    catch { return { ok: false, data: undefined, error: text } }
  } catch (e: any) {
    return { ok: false, data: undefined, error: e.message }
  }
}

export function ResearchTab() {
  const { data, loading, refreshing, refresh } = useSessionResource(hfResearchResource)
  const report = data?.report ?? null
  const empirical = data?.empirical ?? null
  const cacheStats = data?.cacheStats ?? null
  const [scanning, setScanning] = useState(false)
  const [runningEmpirical, setRunningEmpirical] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<'report' | 'empirical'>('report')

  const handleEmpirical = async () => {
    setRunningEmpirical(true)
    setError(null)
    const result = await fetchJSON('/research/empirical', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ top_n: 5, mode: 'simulate' }),
    })
    if (result.ok && result.data) {
      const { data: cached, setData } = hfResearchResource.getState()
      setData({
        report: cached?.report ?? null,
        cacheStats: cached?.cacheStats ?? null,
        empirical: result.data.content,
      })
      setView('empirical')
    } else {
      setError(result.error || 'Empirical ranking failed')
    }
    setRunningEmpirical(false)
  }

  const handleScan = async () => {
    setScanning(true)
    setError(null)
    const result = await fetchJSON('/research/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_candidates: 200 }),
    })
    if (result.ok) {
      await refresh()
    } else {
      setError(result.error || 'Scan failed')
    }
    setScanning(false)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-brand text-text-primary">Dataset Research</h2>
          <p className="text-sm text-text-secondary mt-1">
            HuggingFace Hub scanner — find and rank datasets for training code/bash agents.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {cacheStats && (
            <span className="text-xs text-text-muted font-mono flex items-center gap-1">
              <Database className="w-3 h-3" />
              {cacheStats.cached_datasets} cached
            </span>
          )}
          <button onClick={() => refresh()} className="btn-icon" title="Refresh reports">
            <RefreshCw className={`w-4 h-4${refreshing ? ' animate-spin' : ''}`} />
          </button>
          <button
            onClick={handleEmpirical}
            disabled={runningEmpirical || !cacheStats?.cached_datasets}
            className="btn-ghost flex items-center gap-2 text-sm"
            title={cacheStats?.cached_datasets ? 'Run empirical ranking on top SFT datasets' : 'Run a scan first'}
          >
            {runningEmpirical ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Ranking...
              </>
            ) : (
              <>
                <BarChart3 className="w-4 h-4" />
                Empirical Rank
              </>
            )}
          </button>
          <button
            onClick={handleScan}
            disabled={scanning}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            {scanning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Scanning...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Scan Now
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 border-2 border-status-error rounded-brutal text-sm text-status-error">
          {error}
        </div>
      )}

      {/* Sub-tabs */}
      <div className="flex gap-2">
        <button
          onClick={() => setView('report')}
          className={`flex items-center gap-2 px-3 py-1.5 text-sm font-mono border-2 rounded-brutal ${
            view === 'report'
              ? 'bg-accent text-white border-accent-dark'
              : 'text-text-secondary border-border hover:bg-background-secondary'
          }`}
        >
          <FileText className="w-4 h-4" />
          Scanner Report
        </button>
        <button
          onClick={() => setView('empirical')}
          className={`flex items-center gap-2 px-3 py-1.5 text-sm font-mono border-2 rounded-brutal ${
            view === 'empirical'
              ? 'bg-accent text-white border-accent-dark'
              : 'text-text-secondary border-border hover:bg-background-secondary'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          Empirical Ranking
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-text-secondary" />
        </div>
      ) : view === 'report' ? (
        report ? (
          <div className="border-2 border-border rounded-brutal p-6 bg-background-card overflow-auto max-h-[70vh] prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
          </div>
        ) : (
          <div className="text-center py-12 text-text-secondary">
            <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No scanner report yet. Click "Scan Now" to search HuggingFace Hub.</p>
          </div>
        )
      ) : (
        empirical ? (
          <div className="border-2 border-border rounded-brutal p-6 bg-background-card overflow-auto max-h-[70vh] prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{empirical}</ReactMarkdown>
          </div>
        ) : (
          <div className="text-center py-12 text-text-secondary">
            <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No empirical ranking yet. Run the dataset research runner CLI first.</p>
            <code className="text-xs mt-2 block text-text-muted">
              python -m bashgym.research.dataset_research_runner --top-n 5 --mode simulate
            </code>
          </div>
        )
      )}
    </div>
  )
}
