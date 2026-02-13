import { useState, useEffect, useCallback } from 'react'
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Clock,
  Loader2,
  RefreshCw,
  Calendar
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, TrendDataPoint } from '../../services/api'

interface ModelTrendsProps {
  onBack: () => void
  onSelectModel: (modelId: string) => void
}

interface MetricOption {
  value: string
  label: string
  description: string
  higherIsBetter: boolean
}

const METRIC_OPTIONS: MetricOption[] = [
  { value: 'benchmark_avg_score', label: 'Benchmark Score', description: 'Average across all benchmarks', higherIsBetter: true },
  { value: 'custom_eval_pass_rate', label: 'Custom Eval', description: 'Custom evaluation pass rate', higherIsBetter: true },
  { value: 'final_loss', label: 'Training Loss', description: 'Final training loss', higherIsBetter: false },
  { value: 'model_size_bytes', label: 'Model Size', description: 'Size of model artifacts', higherIsBetter: false },
]

const TIME_RANGE_OPTIONS = [
  { value: 7, label: '7 days' },
  { value: 14, label: '14 days' },
  { value: 30, label: '30 days' },
  { value: 90, label: '90 days' },
]

function formatMetricValue(value: number, metric: string): string {
  switch (metric) {
    case 'benchmark_avg_score':
    case 'custom_eval_pass_rate':
      return `${value.toFixed(1)}%`
    case 'final_loss':
      return value.toFixed(4)
    case 'model_size_bytes':
      const gb = value / (1024 * 1024 * 1024)
      if (gb >= 1) return `${gb.toFixed(2)} GB`
      return `${(value / (1024 * 1024)).toFixed(0)} MB`
    default:
      return value.toFixed(2)
  }
}

function SimpleLineChart({ data, metric, onSelectModel }: {
  data: TrendDataPoint[]
  metric: MetricOption
  onSelectModel: (modelId: string) => void
}) {
  if (data.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-text-muted font-mono text-xs">
        No trend data available
      </div>
    )
  }

  // Group by date for visualization
  const dateGroups = data.reduce((acc, point) => {
    const date = point.timestamp.split('T')[0]
    if (!acc[date]) acc[date] = []
    acc[date].push(point)
    return acc
  }, {} as Record<string, TrendDataPoint[]>)

  const dates = Object.keys(dateGroups).sort()
  const values = data.map(d => d.value)
  const minVal = Math.min(...values)
  const maxVal = Math.max(...values)
  const range = maxVal - minVal || 1

  // Get unique models with their latest values
  const modelLatest = data.reduce((acc, point) => {
    if (!acc[point.model_id] || new Date(point.timestamp) > new Date(acc[point.model_id].timestamp)) {
      acc[point.model_id] = point
    }
    return acc
  }, {} as Record<string, TrendDataPoint>)

  const sortedModels = Object.values(modelLatest).sort((a, b) => {
    return metric.higherIsBetter ? b.value - a.value : a.value - b.value
  })

  // Colors for different models
  const modelColors = [
    'bg-accent',
    'bg-status-success',
    'bg-status-warning',
    'bg-status-info',
    'bg-accent-dark',
    'bg-status-error',
    'bg-accent-light',
    'bg-text-secondary',
  ]

  const getModelColor = (index: number) => modelColors[index % modelColors.length]

  return (
    <div className="space-y-6">
      {/* Simple bar representation per date */}
      <div className="h-64 flex items-end gap-1 p-4 border-brutal border-border rounded-brutal bg-background-secondary">
        {dates.map((date, i) => {
          const dayPoints = dateGroups[date]
          const avgValue = dayPoints.reduce((sum, p) => sum + p.value, 0) / dayPoints.length
          const height = ((avgValue - minVal) / range) * 100 || 5

          return (
            <div
              key={date}
              className="flex-1 flex flex-col items-center justify-end"
            >
              <div
                className="w-full bg-accent hover:bg-accent-dark transition-colors cursor-pointer"
                style={{ height: `${height}%` }}
                title={`${date}: ${formatMetricValue(avgValue, metric.value)} (${dayPoints.length} model${dayPoints.length > 1 ? 's' : ''})`}
              />
              {i === 0 || i === dates.length - 1 || i === Math.floor(dates.length / 2) ? (
                <span className="font-mono text-xs text-text-muted mt-2 rotate-45 origin-left">
                  {new Date(date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                </span>
              ) : null}
            </div>
          )
        })}
      </div>

      {/* Model Legend with values */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {sortedModels.slice(0, 8).map((model, i) => (
          <button
            key={model.model_id}
            onClick={() => onSelectModel(model.model_id)}
            className="card flex items-center gap-3 p-3 text-left"
          >
            <div className={clsx('w-3 h-3 border-brutal border-border', getModelColor(i))} />
            <div className="flex-1 min-w-0">
              <div className="font-brand text-text-primary truncate">{model.display_name}</div>
              <div className="font-mono text-xs text-text-muted">{formatMetricValue(model.value, metric.value)}</div>
            </div>
            {i === 0 && (
              <div className="flex items-center gap-1 text-status-success font-mono text-xs uppercase tracking-widest">
                <TrendingUp className="w-3 h-3" />
                Best
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Stats summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Latest Best</div>
          <div className="font-brand text-2xl text-status-success">
            {sortedModels[0] ? formatMetricValue(sortedModels[0].value, metric.value) : '\u2014'}
          </div>
          <div className="font-mono text-xs text-text-muted truncate">{sortedModels[0]?.display_name}</div>
        </div>
        <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Average</div>
          <div className="font-brand text-2xl text-text-primary">
            {formatMetricValue(values.reduce((a, b) => a + b, 0) / values.length, metric.value)}
          </div>
        </div>
        <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Data Points</div>
          <div className="font-brand text-2xl text-text-primary">{data.length}</div>
        </div>
        <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Unique Models</div>
          <div className="font-brand text-2xl text-text-primary">{Object.keys(modelLatest).length}</div>
        </div>
      </div>
    </div>
  )
}

export function ModelTrends({ onBack, onSelectModel }: ModelTrendsProps) {
  const [selectedMetric, setSelectedMetric] = useState<MetricOption>(METRIC_OPTIONS[0])
  const [timeRange, setTimeRange] = useState(30)
  const [trendData, setTrendData] = useState<TrendDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTrends = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    const result = await modelsApi.trends(selectedMetric.value, timeRange)
    if (result.ok && result.data) {
      setTrendData(result.data.data)
    } else {
      setError('Failed to load trend data')
    }
    setIsLoading(false)
  }, [selectedMetric.value, timeRange])

  useEffect(() => {
    fetchTrends()
  }, [fetchTrends])

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <button onClick={onBack} className="btn-icon flex items-center justify-center">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="font-brand text-3xl text-text-primary">Model Trends</h1>
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted mt-1">
                Track performance over time
              </p>
            </div>
          </div>

          <button
            onClick={fetchTrends}
            disabled={isLoading}
            className="btn-secondary"
          >
            <RefreshCw className={clsx('w-4 h-4 mr-2', isLoading && 'animate-spin')} />
            Refresh
          </button>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4 flex-wrap">
          {/* Metric Selection */}
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-text-muted" />
            <select
              value={selectedMetric.value}
              onChange={(e) => setSelectedMetric(METRIC_OPTIONS.find(m => m.value === e.target.value) || METRIC_OPTIONS[0])}
              className="input font-mono text-sm"
            >
              {METRIC_OPTIONS.map(metric => (
                <option key={metric.value} value={metric.value}>{metric.label}</option>
              ))}
            </select>
          </div>

          {/* Time Range */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-text-muted" />
            <div className="flex items-center border-brutal border-border rounded-brutal overflow-hidden">
              {TIME_RANGE_OPTIONS.map(option => (
                <button
                  key={option.value}
                  onClick={() => setTimeRange(option.value)}
                  className={clsx(
                    'px-3 py-1 font-mono text-xs uppercase tracking-widest transition-colors border-r border-border last:border-r-0',
                    timeRange === option.value
                      ? 'bg-accent text-white'
                      : 'bg-background-card text-text-muted hover:text-text-primary'
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Metric Description */}
          <div className="font-mono text-xs text-text-muted">
            {selectedMetric.description}
            {selectedMetric.higherIsBetter ? ' (higher is better)' : ' (lower is better)'}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 text-accent animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-64">
            <BarChart3 className="w-12 h-12 text-text-muted mb-4" />
            <p className="text-text-muted">{error}</p>
          </div>
        ) : trendData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64">
            <TrendingUp className="w-16 h-16 text-text-muted mb-4" />
            <h2 className="font-brand text-xl text-text-primary mb-2">No Trend Data</h2>
            <p className="text-text-muted">
              Train some models and run evaluations to see trends here
            </p>
          </div>
        ) : (
          <SimpleLineChart
            data={trendData}
            metric={selectedMetric}
            onSelectModel={onSelectModel}
          />
        )}
      </div>
    </div>
  )
}
