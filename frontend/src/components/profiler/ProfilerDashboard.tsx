import { useState, useEffect, useCallback } from 'react'
import {
  Activity,
  Clock,
  Cpu,
  Zap,
  BarChart3,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  RefreshCw,
  Play,
  Pause,
  ChevronDown,
  ChevronRight,
  Terminal,
  Code,
  FileText,
  MessageSquare,
  Settings,
  Download,
  Loader2
} from 'lucide-react'
import { clsx } from 'clsx'
import { observabilityApi, TraceSummary, ObservabilityMetrics } from '../../services/api'

interface ProfileMetric {
  name: string
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  change: number
}

interface ToolCall {
  id: string
  tool: string
  timestamp: string
  duration: number
  tokens_in: number
  tokens_out: number
  status: 'success' | 'error'
  input_preview: string
  output_preview: string
}

interface AgentSession {
  id: string
  started_at: string
  status: 'running' | 'completed' | 'failed'
  total_duration: number
  tool_calls: number
  tokens_used: number
  model: string
  task: string
}

interface Bottleneck {
  type: 'slow_tool' | 'high_tokens' | 'repeated_errors' | 'long_wait'
  severity: 'low' | 'medium' | 'high'
  description: string
  recommendation: string
}

export function ProfilerDashboard() {
  const [isLive, setIsLive] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'traces' | 'tools' | 'bottlenecks'>('overview')
  const [selectedTrace, setSelectedTrace] = useState<TraceSummary | null>(null)
  const [expandedTool, setExpandedTool] = useState<string | null>(null)

  // API data
  const [metrics, setMetrics] = useState<ObservabilityMetrics | null>(null)
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)
  const [isLoadingTraces, setIsLoadingTraces] = useState(false)

  // Computed metrics for display
  const [displayMetrics, setDisplayMetrics] = useState<ProfileMetric[]>([])

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    setIsLoadingMetrics(true)
    const result = await observabilityApi.getMetrics()
    if (result.ok && result.data) {
      setMetrics(result.data)

      // Convert to display format
      const profilerData = result.data.profiler
      const newMetrics: ProfileMetric[] = []

      if (profilerData.total_traces !== undefined) {
        newMetrics.push({
          name: 'Total Traces',
          value: profilerData.total_traces,
          unit: '',
          trend: 'stable',
          change: 0
        })
      }
      if (profilerData.avg_duration_ms !== undefined) {
        newMetrics.push({
          name: 'Avg Duration',
          value: Math.round(profilerData.avg_duration_ms),
          unit: 'ms',
          trend: 'stable',
          change: 0
        })
      }
      if (profilerData.total_tokens !== undefined) {
        newMetrics.push({
          name: 'Total Tokens',
          value: profilerData.total_tokens,
          unit: '',
          trend: 'stable',
          change: 0
        })
      }
      if (profilerData.avg_tokens_per_trace !== undefined) {
        newMetrics.push({
          name: 'Avg Tokens/Trace',
          value: Math.round(profilerData.avg_tokens_per_trace),
          unit: '',
          trend: 'stable',
          change: 0
        })
      }

      setDisplayMetrics(newMetrics)
    }
    setIsLoadingMetrics(false)
  }, [])

  // Fetch traces
  const fetchTraces = useCallback(async () => {
    setIsLoadingTraces(true)
    const result = await observabilityApi.listTraces(50, 0)
    if (result.ok && result.data) {
      setTraces(result.data.traces)
    }
    setIsLoadingTraces(false)
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchMetrics()
    fetchTraces()
  }, [fetchMetrics, fetchTraces])

  // Live polling
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      fetchMetrics()
      fetchTraces()
    }, 5000)

    return () => clearInterval(interval)
  }, [isLive, fetchMetrics, fetchTraces])

  const getToolIcon = (tool: string) => {
    switch (tool) {
      case 'Bash': return Terminal
      case 'Read': return FileText
      case 'Edit': case 'Write': return Code
      case 'Grep': case 'Glob': return FileText
      default: return Zap
    }
  }

  // Compute tool stats from traces
  const toolStats = traces.reduce((acc, trace) => {
    const toolCalls = trace.tool_calls || {}
    const count = toolCalls.count || 0
    if (count > 0) {
      const tool = 'Various' // We don't have per-tool breakdown in current API
      if (!acc[tool]) {
        acc[tool] = { tool, calls: 0, avg_duration: 0, success_rate: 100, tokens: 0 }
      }
      acc[tool].calls += count
    }
    return acc
  }, {} as Record<string, { tool: string; calls: number; avg_duration: number; success_rate: number; tokens: number }>)

  // Compute bottlenecks from traces
  const bottlenecks: Bottleneck[] = traces
    .filter(t => t.bottlenecks && t.bottlenecks.length > 0)
    .flatMap(t => (t.bottlenecks || []).map((b: any) => ({
      type: b.type || 'slow_tool',
      severity: b.severity || 'medium',
      description: b.description || `Bottleneck in ${t.name}`,
      recommendation: b.recommendation || 'Review trace details'
    })))
    .slice(0, 10)

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-text-primary">Agent Profiler</h1>
            <p className="text-sm text-text-secondary mt-1">
              Real-time performance monitoring and bottleneck detection
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsLive(!isLive)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                isLive
                  ? 'bg-status-success text-white'
                  : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
              )}
            >
              {isLive ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
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
              onClick={() => { fetchMetrics(); fetchTraces(); }}
              disabled={isLoadingMetrics || isLoadingTraces}
              className="btn-secondary flex items-center gap-2"
            >
              <RefreshCw className={clsx('w-4 h-4', (isLoadingMetrics || isLoadingTraces) && 'animate-spin')} />
              Refresh
            </button>
            <button className="btn-secondary flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-6">
          {[
            { id: 'overview' as const, label: 'Overview', icon: BarChart3 },
            { id: 'traces' as const, label: 'Traces', icon: Activity, badge: traces.length },
            { id: 'tools' as const, label: 'Tool Analytics', icon: Zap },
            { id: 'bottlenecks' as const, label: 'Bottlenecks', icon: AlertTriangle, badge: bottlenecks.length },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                activeTab === tab.id
                  ? 'bg-primary text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.badge !== undefined && tab.badge > 0 && (
                <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-white/20">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Metrics Grid */}
            {displayMetrics.length > 0 ? (
              <div className="grid grid-cols-4 gap-4">
                {displayMetrics.map(metric => (
                  <div key={metric.name} className="card-elevated p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-text-muted">{metric.name}</span>
                      {metric.trend === 'up' && <TrendingUp className="w-4 h-4 text-status-success" />}
                      {metric.trend === 'down' && <TrendingDown className="w-4 h-4 text-status-error" />}
                      {metric.trend === 'stable' && <Activity className="w-4 h-4 text-text-muted" />}
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-2xl font-semibold text-text-primary">
                        {metric.value.toLocaleString()}
                      </span>
                      {metric.unit && <span className="text-sm text-text-muted">{metric.unit}</span>}
                    </div>
                    {metric.change !== 0 && (
                      <span className={clsx(
                        'text-xs',
                        metric.change > 0 ? 'text-status-success' : 'text-status-error'
                      )}>
                        {metric.change > 0 ? '+' : ''}{metric.change}% from last hour
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="card-elevated p-8 text-center">
                <Activity className="w-12 h-12 text-text-muted mx-auto mb-3 opacity-30" />
                <h3 className="text-lg font-medium text-text-primary mb-2">
                  {isLoadingMetrics ? 'Loading metrics...' : 'No profiler data yet'}
                </h3>
                <p className="text-sm text-text-muted">
                  {isLoadingMetrics ? 'Please wait' : 'Run tasks to collect profiler metrics'}
                </p>
              </div>
            )}

            {/* Guardrail Stats Summary */}
            {metrics?.guardrails && (
              <div className="card-elevated p-4">
                <h3 className="font-medium text-text-primary mb-4">Guardrails Summary</h3>
                <div className="grid grid-cols-4 gap-4">
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-sm text-text-muted">Total Events</div>
                    <div className="text-xl font-semibold text-text-primary">
                      {metrics.guardrails.total_events}
                    </div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-sm text-text-muted">Block Rate</div>
                    <div className="text-xl font-semibold text-status-error">
                      {(metrics.guardrails.block_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-sm text-text-muted">Blocked</div>
                    <div className="text-xl font-semibold text-status-error">
                      {metrics.guardrails.by_action?.block || 0}
                    </div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-sm text-text-muted">Warnings</div>
                    <div className="text-xl font-semibold text-status-warning">
                      {metrics.guardrails.by_action?.warn || 0}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Recent Traces */}
            {traces.length > 0 && (
              <div className="card-elevated">
                <div className="p-4 border-b border-border-subtle">
                  <h3 className="font-medium text-text-primary">Recent Traces</h3>
                </div>
                <div className="divide-y divide-border-subtle">
                  {traces.slice(0, 5).map(trace => (
                    <button
                      key={trace.trace_id}
                      onClick={() => setSelectedTrace(trace)}
                      className={clsx(
                        'w-full p-4 text-left hover:bg-background-tertiary transition-colors',
                        selectedTrace?.trace_id === trace.trace_id && 'bg-primary/5'
                      )}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-status-info" />
                          <span className="font-medium text-text-primary">{trace.name}</span>
                        </div>
                        <span className="text-xs text-text-muted">
                          {trace.trace_id.slice(0, 8)}...
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-text-muted">
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
          <div className="space-y-4">
            {traces.length === 0 ? (
              <div className="card-elevated p-12 text-center">
                <Terminal className="w-12 h-12 text-text-muted mx-auto mb-4 opacity-30" />
                <h3 className="text-lg font-medium text-text-primary mb-2">No traces recorded</h3>
                <p className="text-text-muted">Run tasks to capture execution traces</p>
              </div>
            ) : (
              <div className="card-elevated">
                <div className="p-4 border-b border-border-subtle flex items-center justify-between">
                  <h3 className="font-medium text-text-primary">Execution Traces</h3>
                  <span className="text-sm text-text-muted">{traces.length} traces</span>
                </div>
                <div className="divide-y divide-border-subtle max-h-[600px] overflow-y-auto">
                  {traces.map(trace => (
                    <div key={trace.trace_id} className="p-4 hover:bg-background-tertiary">
                      <div className="flex items-center gap-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium text-text-primary">{trace.name}</span>
                            <span className="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                              {trace.total_spans} spans
                            </span>
                          </div>
                          <div className="flex items-center gap-4 text-xs text-text-muted">
                            <span>ID: {trace.trace_id.slice(0, 12)}...</span>
                            <span>Duration: {trace.duration_ms.toFixed(0)}ms</span>
                            {trace.llm_calls?.count && (
                              <span>LLM calls: {trace.llm_calls.count}</span>
                            )}
                            {trace.tool_calls?.count && (
                              <span>Tool calls: {trace.tool_calls.count}</span>
                            )}
                          </div>
                        </div>
                        <button
                          onClick={() => setSelectedTrace(trace)}
                          className="btn-secondary text-sm"
                        >
                          View Details
                        </button>
                      </div>

                      {/* Bottlenecks indicator */}
                      {trace.bottlenecks && trace.bottlenecks.length > 0 && (
                        <div className="mt-2 flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-status-warning" />
                          <span className="text-sm text-status-warning">
                            {trace.bottlenecks.length} bottleneck(s) detected
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Selected Trace Details */}
            {selectedTrace && (
              <div className="card-elevated p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-text-primary">Trace Details: {selectedTrace.name}</h3>
                  <button
                    onClick={() => setSelectedTrace(null)}
                    className="text-text-muted hover:text-text-primary"
                  >
                    Close
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-xs text-text-muted mb-1">Trace ID</div>
                    <div className="text-sm text-text-primary font-mono">{selectedTrace.trace_id}</div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-xs text-text-muted mb-1">Duration</div>
                    <div className="text-sm text-text-primary">{selectedTrace.duration_ms.toFixed(2)}ms</div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-xs text-text-muted mb-1">Total Spans</div>
                    <div className="text-sm text-text-primary">{selectedTrace.total_spans}</div>
                  </div>
                  <div className="p-3 bg-background-tertiary rounded-lg">
                    <div className="text-xs text-text-muted mb-1">LLM Calls</div>
                    <div className="text-sm text-text-primary">{selectedTrace.llm_calls?.count || 0}</div>
                  </div>
                  {selectedTrace.llm_calls?.total_tokens && (
                    <div className="p-3 bg-background-tertiary rounded-lg">
                      <div className="text-xs text-text-muted mb-1">Total Tokens</div>
                      <div className="text-sm text-text-primary">{selectedTrace.llm_calls.total_tokens.toLocaleString()}</div>
                    </div>
                  )}
                  {selectedTrace.tool_calls?.count && (
                    <div className="p-3 bg-background-tertiary rounded-lg">
                      <div className="text-xs text-text-muted mb-1">Tool Calls</div>
                      <div className="text-sm text-text-primary">{selectedTrace.tool_calls.count}</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="space-y-6">
            {Object.keys(toolStats).length === 0 ? (
              <div className="card-elevated p-12 text-center">
                <Zap className="w-12 h-12 text-text-muted mx-auto mb-4 opacity-30" />
                <h3 className="text-lg font-medium text-text-primary mb-2">No tool analytics yet</h3>
                <p className="text-text-muted">Run tasks to collect tool performance data</p>
              </div>
            ) : (
              <>
                <div className="card-elevated">
                  <div className="p-4 border-b border-border-subtle">
                    <h3 className="font-medium text-text-primary">Tool Performance</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-background-tertiary">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Tool</th>
                          <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Calls</th>
                          <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Avg Duration</th>
                          <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Success Rate</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Performance</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border-subtle">
                        {Object.values(toolStats).map(stat => {
                          const Icon = getToolIcon(stat.tool)
                          return (
                            <tr key={stat.tool} className="hover:bg-background-tertiary">
                              <td className="px-4 py-3">
                                <div className="flex items-center gap-2">
                                  <Icon className="w-4 h-4 text-primary" />
                                  <span className="font-medium text-text-primary">{stat.tool}</span>
                                </div>
                              </td>
                              <td className="px-4 py-3 text-right text-text-primary">{stat.calls}</td>
                              <td className="px-4 py-3 text-right text-text-primary">{stat.avg_duration}ms</td>
                              <td className="px-4 py-3 text-right">
                                <span className={clsx(
                                  stat.success_rate >= 95 ? 'text-status-success' :
                                  stat.success_rate >= 85 ? 'text-status-warning' :
                                  'text-status-error'
                                )}>
                                  {stat.success_rate}%
                                </span>
                              </td>
                              <td className="px-4 py-3">
                                <div className="w-24 h-2 bg-background-tertiary rounded-full overflow-hidden">
                                  <div
                                    className={clsx(
                                      'h-full rounded-full',
                                      stat.success_rate >= 95 ? 'bg-status-success' :
                                      stat.success_rate >= 85 ? 'bg-status-warning' :
                                      'bg-status-error'
                                    )}
                                    style={{ width: `${stat.success_rate}%` }}
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
              </>
            )}
          </div>
        )}

        {activeTab === 'bottlenecks' && (
          <div className="space-y-4">
            {bottlenecks.length === 0 ? (
              <div className="card-elevated p-12 text-center">
                <AlertTriangle className="w-12 h-12 text-text-muted mx-auto mb-4 opacity-30" />
                <h3 className="text-lg font-medium text-text-primary mb-2">No bottlenecks detected</h3>
                <p className="text-text-muted">Great! Your agent tasks are running efficiently</p>
              </div>
            ) : (
              bottlenecks.map((bottleneck, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'card-elevated p-4 border-l-4',
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
                          'text-xs px-2 py-0.5 rounded-full',
                          bottleneck.severity === 'high' ? 'bg-status-error/20 text-status-error' :
                          bottleneck.severity === 'medium' ? 'bg-status-warning/20 text-status-warning' :
                          'bg-status-info/20 text-status-info'
                        )}>
                          {bottleneck.severity}
                        </span>
                      </div>
                      <p className="text-sm text-text-muted mb-2">{bottleneck.recommendation}</p>
                      <button className="text-sm text-primary hover:underline">
                        View details
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}

            {/* Optimization Tips */}
            <div className="card-elevated p-4 bg-status-info/5 border-status-info/20">
              <h3 className="font-medium text-text-primary mb-3">Optimization Tips</h3>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li className="flex items-start gap-2">
                  <span className="text-status-info">•</span>
                  Use file line limits when reading large files to reduce token usage
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-status-info">•</span>
                  Batch related file operations together when possible
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-status-info">•</span>
                  Set appropriate timeouts for long-running bash commands
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-status-info">•</span>
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
