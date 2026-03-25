import { useState, useEffect, useCallback } from 'react'
import {
  Activity,
  Clock,
  Zap,
  BarChart3,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  Play,
  Terminal,
  Code,
  FileText,
  MessageSquare,
  Download,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import { observabilityApi, TraceSummary, TraceDetail, ToolStat, ObservabilityMetrics } from '../../services/api'
import { SpanTimeline } from './SpanTimeline'

interface ProfileMetric {
  name: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  change: number
  accentVar: string
}

interface Bottleneck {
  type: string
  severity: 'low' | 'medium' | 'high'
  description: string
  recommendation: string
  trace_id?: string
}

export function ProfilerDashboard() {
  const [isLive, setIsLive] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'traces' | 'tools' | 'bottlenecks'>('overview')
  const [selectedTrace, setSelectedTrace] = useState<TraceDetail | null>(null)
  const [isLoadingDetail, setIsLoadingDetail] = useState(false)

  // API data
  const [metrics, setMetrics] = useState<ObservabilityMetrics | null>(null)
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [toolStats, setToolStats] = useState<ToolStat[]>([])
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)
  const [isLoadingTraces, setIsLoadingTraces] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Computed metrics for display
  const [displayMetrics, setDisplayMetrics] = useState<ProfileMetric[]>([])

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    setIsLoadingMetrics(true)
    setError(null)
    const result = await observabilityApi.getMetrics()
    if (result.ok && result.data) {
      setMetrics(result.data)

      const profilerData = result.data.profiler
      const newMetrics: ProfileMetric[] = []
      const accents = ['--chart-1', '--chart-2', '--chart-3', '--chart-4']

      if (profilerData.total_traces !== undefined) {
        newMetrics.push({
          name: 'Total Traces', value: profilerData.total_traces,
          unit: '', trend: 'stable', change: 0, accentVar: accents[0],
        })
      }
      if (profilerData.avg_duration_ms !== undefined) {
        newMetrics.push({
          name: 'Avg Duration', value: Math.round(profilerData.avg_duration_ms),
          unit: 'ms', trend: 'stable', change: 0, accentVar: accents[1],
        })
      }
      if (profilerData.total_tokens !== undefined) {
        newMetrics.push({
          name: 'Total Tokens', value: profilerData.total_tokens,
          unit: '', trend: 'stable', change: 0, accentVar: accents[2],
        })
      }
      if (profilerData.avg_tokens_per_trace !== undefined) {
        newMetrics.push({
          name: 'Avg Tokens/Trace', value: Math.round(profilerData.avg_tokens_per_trace),
          unit: '', trend: 'stable', change: 0, accentVar: accents[3],
        })
      }

      setDisplayMetrics(newMetrics)
    } else if (!result.ok) {
      setError(result.error || 'Failed to fetch metrics')
    }
    setIsLoadingMetrics(false)
  }, [])

  // Fetch traces
  const fetchTraces = useCallback(async () => {
    setIsLoadingTraces(true)
    const result = await observabilityApi.listTraces(50, 0)
    if (result.ok && result.data) {
      setTraces(result.data.traces)
    } else if (!result.ok) {
      setError(prev => prev || (result.error || 'Failed to fetch traces'))
    }
    setIsLoadingTraces(false)
  }, [])

  // Fetch tool stats
  const fetchToolStats = useCallback(async () => {
    const result = await observabilityApi.getToolStats()
    if (result.ok && result.data) {
      setToolStats(result.data)
    }
  }, [])

  // Fetch full trace detail
  const selectTrace = useCallback(async (trace: TraceSummary) => {
    setIsLoadingDetail(true)
    const result = await observabilityApi.getTrace(trace.trace_id)
    if (result.ok && result.data) {
      setSelectedTrace(result.data)
    } else {
      // Fallback: show summary data without spans
      setSelectedTrace({ ...trace, spans: [] } as TraceDetail)
    }
    setIsLoadingDetail(false)
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchMetrics()
    fetchTraces()
    fetchToolStats()
  }, [fetchMetrics, fetchTraces, fetchToolStats])

  // Live polling
  useEffect(() => {
    if (!isLive) return
    const interval = setInterval(() => {
      fetchMetrics()
      fetchTraces()
      fetchToolStats()
    }, 5000)
    return () => clearInterval(interval)
  }, [isLive, fetchMetrics, fetchTraces, fetchToolStats])

  const getToolIcon = (tool: string) => {
    switch (tool) {
      case 'Bash': return Terminal
      case 'Read': return FileText
      case 'Edit': case 'Write': return Code
      case 'Grep': case 'Glob': return FileText
      default: return Zap
    }
  }

  // Export handler
  const handleExport = () => {
    const exportData = {
      metrics,
      traces,
      toolStats,
      exportedAt: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `profiler-export-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Compute bottlenecks from traces (now populated by backend)
  const bottlenecks: Bottleneck[] = traces
    .filter(t => t.bottlenecks && t.bottlenecks.length > 0)
    .flatMap(t => (t.bottlenecks || []).map((b: any) => ({
      type: b.type || 'slow_tool',
      severity: b.severity || 'medium',
      description: b.suggestion || b.description || `Bottleneck in ${t.name}`,
      recommendation: b.recommendation || b.suggestion || 'Review trace details',
      trace_id: b.trace_id || t.trace_id,
    })))
    .slice(0, 10)

  // Navigate to trace from bottleneck
  const viewBottleneckTrace = (traceId: string) => {
    const trace = traces.find(t => t.trace_id === traceId)
    if (trace) {
      setActiveTab('traces')
      selectTrace(trace)
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b-brutal border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="tag"><span>Observability</span></span>
            <div>
              <h1 className="font-brand text-3xl text-text-primary">Agent Profiler</h1>
              <p className="text-sm text-text-secondary mt-1">
                Real-time performance monitoring and bottleneck detection
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsLive(!isLive)}
              className={clsx(
                'flex items-center gap-2',
                isLive ? 'btn-primary' : 'btn-secondary'
              )}
            >
              {isLive ? (
                <>
                  <span className="status-dot status-success animate-pulse" />
                  Live
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Live
                </>
              )}
            </button>
            <button
              onClick={() => { fetchMetrics(); fetchTraces(); fetchToolStats(); }}
              disabled={isLoadingMetrics || isLoadingTraces}
              className="btn-secondary flex items-center gap-2"
            >
              <RefreshCw className={clsx('w-4 h-4', (isLoadingMetrics || isLoadingTraces) && 'animate-spin')} />
              Refresh
            </button>
            <button onClick={handleExport} className="btn-secondary flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mt-6">
          {[
            { id: 'overview' as const, label: 'Overview', icon: BarChart3 },
            { id: 'traces' as const, label: 'Traces', icon: Activity, badge: traces.length },
            { id: 'tools' as const, label: 'Tool Analytics', icon: Zap, badge: toolStats.length },
            { id: 'bottlenecks' as const, label: 'Bottlenecks', icon: AlertTriangle, badge: bottlenecks.length },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-widest border-brutal rounded-brutal transition-press',
                activeTab === tab.id
                  ? 'bg-accent text-white border-border shadow-brutal-sm'
                  : 'bg-background-card text-text-secondary border-border-subtle hover:border-border hover-press'
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.badge !== undefined && tab.badge > 0 && (
                <span className="tag ml-1"><span>{tab.badge}</span></span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-6 mt-4 p-3 border-l-[4px] border-l-status-error bg-status-error/10 border-brutal flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-status-error flex-shrink-0" />
          <span className="text-sm text-text-primary flex-1">{error}</span>
          <button onClick={() => setError(null)} className="btn-ghost p-1">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-auto p-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Metrics Grid */}
            {displayMetrics.length > 0 ? (
              <div className="grid grid-cols-4 gap-4">
                {displayMetrics.map(metric => (
                  <div
                    key={metric.name}
                    className="card p-4"
                    style={{ borderLeftWidth: '4px', borderLeftColor: `var(${metric.accentVar})` }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs uppercase tracking-widest text-text-muted">{metric.name}</span>
                      {metric.trend === 'up' && <TrendingUp className="w-4 h-4 text-status-success" />}
                      {metric.trend === 'down' && <TrendingDown className="w-4 h-4 text-status-error" />}
                      {metric.trend === 'stable' && <Activity className="w-4 h-4 text-text-muted" />}
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="font-brand text-3xl text-text-primary">
                        {metric.value.toLocaleString()}
                      </span>
                      {metric.unit && <span className="font-mono text-xs text-text-muted">{metric.unit}</span>}
                    </div>
                    {metric.change !== 0 && (
                      <span className={clsx(
                        'font-mono text-xs',
                        metric.change > 0 ? 'text-status-success' : 'text-status-error'
                      )}>
                        {metric.change > 0 ? '+' : ''}{metric.change}% from last hour
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="card p-8 text-center">
                <Activity className="w-12 h-12 text-text-muted mx-auto mb-3" />
                <h3 className="font-brand text-xl text-text-primary mb-2">
                  {isLoadingMetrics ? 'Loading metrics...' : 'No profiler data yet'}
                </h3>
                <p className="text-sm text-text-muted">
                  {isLoadingMetrics ? 'Please wait' : 'Run tasks to collect profiler metrics'}
                </p>
              </div>
            )}

            <div className="section-divider" />

            {/* Guardrail Stats Summary */}
            {metrics?.guardrails && (
              <div className="card p-4">
                <h3 className="font-brand text-xl text-text-primary mb-4">Guardrails Summary</h3>
                <div className="section-divider mb-4" />
                <div className="grid grid-cols-4 gap-4">
                  <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                    <div className="font-mono text-xs uppercase tracking-widest text-text-muted">Total Events</div>
                    <div className="font-brand text-2xl text-text-primary">
                      {metrics.guardrails.total_events}
                    </div>
                  </div>
                  <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                    <div className="font-mono text-xs uppercase tracking-widest text-text-muted">Block Rate</div>
                    <div className="font-brand text-2xl text-status-error">
                      {(metrics.guardrails.block_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                    <div className="font-mono text-xs uppercase tracking-widest text-text-muted">Blocked</div>
                    <div className="font-brand text-2xl text-status-error">
                      {metrics.guardrails.by_action?.block || 0}
                    </div>
                  </div>
                  <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                    <div className="font-mono text-xs uppercase tracking-widest text-text-muted">Warnings</div>
                    <div className="font-brand text-2xl text-status-warning">
                      {metrics.guardrails.by_action?.warn || 0}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="section-divider" />

            {/* Recent Traces */}
            {traces.length > 0 && (
              <div className="card">
                <div className="p-4 border-b-brutal border-border">
                  <h3 className="font-brand text-xl text-text-primary">Recent Traces</h3>
                </div>
                <div className="divide-y divide-border">
                  {traces.slice(0, 5).map(trace => (
                    <button
                      key={trace.trace_id}
                      onClick={() => { setActiveTab('traces'); selectTrace(trace); }}
                      className={clsx(
                        'w-full p-4 text-left hover:bg-background-secondary transition-press',
                        selectedTrace?.trace_id === trace.trace_id && 'bg-accent-light'
                      )}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className={clsx(
                            'status-dot',
                            trace.status === 'error' ? 'status-error' :
                            trace.status === 'in_progress' ? 'status-warning' :
                            'status-success'
                          )} />
                          <span className="font-medium text-text-primary">{trace.name}</span>
                        </div>
                        <span className="font-mono text-xs text-text-muted">
                          {trace.trace_id.slice(0, 8)}...
                        </span>
                      </div>
                      <div className="flex items-center gap-4 font-mono text-xs text-text-muted">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {trace.duration_ms.toFixed(0)}ms
                        </span>
                        <span className="flex items-center gap-1">
                          <Zap className="w-3 h-3" />
                          {trace.total_spans} spans
                        </span>
                        {trace.llm_calls?.total_tokens && (
                          <span className="flex items-center gap-1">
                            <MessageSquare className="w-3 h-3" />
                            {(trace.llm_calls.total_tokens / 1000).toFixed(1)}k tokens
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'traces' && (
          <div className={clsx('flex gap-4', selectedTrace && 'h-full')}>
            {/* Trace list — shrinks when detail is open */}
            <div className={clsx(
              'space-y-4 overflow-auto',
              selectedTrace ? 'w-1/2 flex-shrink-0' : 'w-full'
            )}>
              {traces.length === 0 ? (
                <div className="card p-12 text-center">
                  <Terminal className="w-12 h-12 text-text-muted mx-auto mb-3" />
                  <h3 className="font-brand text-xl text-text-primary mb-2">No traces recorded</h3>
                  <p className="text-sm text-text-muted">Run tasks to capture execution traces</p>
                </div>
              ) : (
                <div className="card">
                  <div className="p-4 border-b-brutal border-border flex items-center justify-between">
                    <h3 className="font-brand text-xl text-text-primary">Execution Traces</h3>
                    <span className="tag"><span>{traces.length} traces</span></span>
                  </div>
                  <div className="divide-y divide-border max-h-[600px] overflow-y-auto">
                    {traces.map(trace => (
                      <div
                        key={trace.trace_id}
                        className={clsx(
                          'p-4 hover:bg-background-secondary transition-press cursor-pointer',
                          selectedTrace?.trace_id === trace.trace_id && 'bg-accent-light'
                        )}
                        onClick={() => selectTrace(trace)}
                      >
                        <div className="flex items-center gap-3">
                          <span className={clsx(
                            'status-dot flex-shrink-0',
                            trace.status === 'error' ? 'status-error' :
                            trace.status === 'in_progress' ? 'status-warning' :
                            'status-success'
                          )} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium text-text-primary truncate">{trace.name}</span>
                              <span className="tag flex-shrink-0">
                                <span>{trace.total_spans} spans</span>
                              </span>
                            </div>
                            <div className="flex items-center gap-4 font-mono text-xs text-text-muted">
                              <span>ID: {trace.trace_id.slice(0, 12)}...</span>
                              <span>{trace.duration_ms.toFixed(0)}ms</span>
                              {trace.llm_calls?.count > 0 && (
                                <span>LLM: {trace.llm_calls.count}</span>
                              )}
                              {trace.tool_calls?.count > 0 && (
                                <span>Tools: {trace.tool_calls.count}</span>
                              )}
                            </div>
                          </div>
                        </div>

                        {trace.bottlenecks && trace.bottlenecks.length > 0 && (
                          <div className="mt-2 flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-status-warning" />
                            <span className="font-mono text-xs text-status-warning">
                              {trace.bottlenecks.length} bottleneck(s) detected
                            </span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Trace detail — slides in from right */}
            {selectedTrace && (
              <div className="w-1/2 flex-shrink-0 animate-slide-in-right overflow-auto">
                <div className="card card-accent p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-brand text-xl text-text-primary truncate">
                      {isLoadingDetail ? 'Loading...' : selectedTrace.name}
                    </h3>
                    <button
                      onClick={() => setSelectedTrace(null)}
                      className="btn-ghost p-1"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="section-divider mb-4" />
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Trace ID</div>
                      <div className="text-sm text-text-primary font-mono truncate">{selectedTrace.trace_id}</div>
                    </div>
                    <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Duration</div>
                      <div className="font-brand text-xl text-text-primary">{selectedTrace.duration_ms.toFixed(0)}ms</div>
                    </div>
                    <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Total Spans</div>
                      <div className="font-brand text-xl text-text-primary">{selectedTrace.total_spans}</div>
                    </div>
                    <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">LLM Calls</div>
                      <div className="font-brand text-xl text-text-primary">{selectedTrace.llm_calls?.count || 0}</div>
                    </div>
                    {selectedTrace.llm_calls?.total_tokens > 0 && (
                      <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                        <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Total Tokens</div>
                        <div className="font-brand text-xl text-text-primary">{selectedTrace.llm_calls.total_tokens.toLocaleString()}</div>
                      </div>
                    )}
                    {selectedTrace.tool_calls?.count > 0 && (
                      <div className="p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                        <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Tool Calls</div>
                        <div className="font-brand text-xl text-text-primary">{selectedTrace.tool_calls.count}</div>
                      </div>
                    )}
                  </div>

                  {/* Span timeline waterfall */}
                  {selectedTrace.spans && selectedTrace.spans.length > 0 && (
                    <>
                      <div className="section-divider mb-4" />
                      <h4 className="font-brand text-lg text-text-primary mb-3">Span Timeline</h4>
                      <SpanTimeline
                        spans={selectedTrace.spans}
                        totalDurationMs={selectedTrace.duration_ms}
                      />
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="space-y-6">
            {toolStats.length === 0 ? (
              <div className="card p-12 text-center">
                <Zap className="w-12 h-12 text-text-muted mx-auto mb-3" />
                <h3 className="font-brand text-xl text-text-primary mb-2">No tool analytics yet</h3>
                <p className="text-sm text-text-muted">Run tasks to collect tool performance data</p>
              </div>
            ) : (
              <div className="card">
                <div className="p-4 border-b-brutal border-border">
                  <h3 className="font-brand text-xl text-text-primary">Tool Performance</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-background-secondary border-b-brutal border-border">
                      <tr>
                        <th className="px-4 py-3 text-left font-mono text-xs uppercase tracking-widest text-text-muted">Tool</th>
                        <th className="px-4 py-3 text-right font-mono text-xs uppercase tracking-widest text-text-muted">Calls</th>
                        <th className="px-4 py-3 text-right font-mono text-xs uppercase tracking-widest text-text-muted">Avg Duration</th>
                        <th className="px-4 py-3 text-right font-mono text-xs uppercase tracking-widest text-text-muted">Success Rate</th>
                        <th className="px-4 py-3 text-left font-mono text-xs uppercase tracking-widest text-text-muted">Performance</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {toolStats.map(stat => {
                        const Icon = getToolIcon(stat.tool)
                        const barColor = stat.success_rate >= 95
                          ? 'var(--status-success)'
                          : stat.success_rate >= 85
                          ? 'var(--status-warning)'
                          : 'var(--status-error)'
                        return (
                          <tr key={stat.tool} className="hover:bg-background-secondary transition-press">
                            <td className="px-4 py-3">
                              <div className="flex items-center gap-2">
                                <Icon className="w-4 h-4 text-accent" />
                                <span className="font-medium text-text-primary">{stat.tool}</span>
                              </div>
                            </td>
                            <td className="px-4 py-3 text-right font-brand text-lg text-text-primary">{stat.calls}</td>
                            <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">{stat.avg_duration_ms}ms</td>
                            <td className="px-4 py-3 text-right">
                              <span className={clsx(
                                'font-mono text-sm',
                                stat.success_rate >= 95 ? 'text-status-success' :
                                stat.success_rate >= 85 ? 'text-status-warning' :
                                'text-status-error'
                              )}>
                                {stat.success_rate}%
                              </span>
                            </td>
                            <td className="px-4 py-3">
                              <div className="progress-bar w-24">
                                <div
                                  className="progress-fill"
                                  style={{
                                    width: `${stat.success_rate}%`,
                                    background: barColor,
                                  }}
                                />
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'bottlenecks' && (
          <div className="space-y-4">
            {bottlenecks.length === 0 ? (
              <div className="card p-12 text-center">
                <AlertTriangle className="w-12 h-12 text-text-muted mx-auto mb-3" />
                <h3 className="font-brand text-xl text-text-primary mb-2">No bottlenecks detected</h3>
                <p className="text-sm text-text-muted">Your agent tasks are running efficiently</p>
              </div>
            ) : (
              bottlenecks.map((bottleneck, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'card p-4 border-l-[4px]',
                    bottleneck.severity === 'high' ? 'border-l-status-error' :
                    bottleneck.severity === 'medium' ? 'border-l-status-warning' :
                    'border-l-status-info'
                  )}
                >
                  <div className="flex items-start gap-4">
                    <AlertTriangle className={clsx(
                      'w-5 h-5 mt-0.5',
                      bottleneck.severity === 'high' ? 'text-status-error' :
                      bottleneck.severity === 'medium' ? 'text-status-warning' :
                      'text-status-info'
                    )} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-text-primary">{bottleneck.description}</span>
                        <span className={clsx(
                          'font-mono text-xs uppercase tracking-widest px-2 py-0.5 border-brutal rounded-brutal',
                          bottleneck.severity === 'high' ? 'bg-status-error/20 text-status-error border-status-error' :
                          bottleneck.severity === 'medium' ? 'bg-status-warning/20 text-status-warning border-status-warning' :
                          'bg-status-info/20 text-status-info border-status-info'
                        )}>
                          {bottleneck.severity}
                        </span>
                      </div>
                      <p className="text-sm text-text-muted mb-2">{bottleneck.recommendation}</p>
                      {bottleneck.trace_id && (
                        <button
                          onClick={() => viewBottleneckTrace(bottleneck.trace_id!)}
                          className="font-mono text-xs uppercase tracking-widest text-accent hover:text-accent-dark"
                        >
                          View details
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}

            {/* Optimization Tips */}
            <div className="card p-4 border-l-[4px] border-l-status-info">
              <h3 className="font-brand text-xl text-text-primary mb-3">Optimization Tips</h3>
              <div className="section-divider mb-3" />
              <ul className="space-y-2 text-sm text-text-secondary">
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 flex-shrink-0" />
                  Use file line limits when reading large files to reduce token usage
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 flex-shrink-0" />
                  Batch related file operations together when possible
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 flex-shrink-0" />
                  Set appropriate timeouts for long-running bash commands
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 flex-shrink-0" />
                  Use Grep with file type filters instead of globbing everything
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
