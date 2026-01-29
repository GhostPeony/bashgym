import { useState, useEffect, useCallback } from 'react'
import {
  ArrowLeft,
  Plus,
  X,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Loader2,
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  Activity
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, ModelProfile as ModelProfileData } from '../../services/api'

interface ModelComparisonProps {
  modelIds: string[]
  onBack: () => void
  onAddModel: () => void
  onRemoveModel: (modelId: string) => void
}

interface ComparisonMetric {
  key: string
  label: string
  category: 'performance' | 'training' | 'operational'
  format: 'percent' | 'number' | 'time' | 'size'
  higherIsBetter: boolean
}

const COMPARISON_METRICS: ComparisonMetric[] = [
  // Performance metrics
  { key: 'custom_eval_pass_rate', label: 'Custom Eval', category: 'performance', format: 'percent', higherIsBetter: true },
  { key: 'benchmark_avg_score', label: 'Benchmark Avg', category: 'performance', format: 'percent', higherIsBetter: true },
  // Training metrics
  { key: 'final_loss', label: 'Final Loss', category: 'training', format: 'number', higherIsBetter: false },
  { key: 'training_examples', label: 'Training Examples', category: 'training', format: 'number', higherIsBetter: true },
  { key: 'num_epochs', label: 'Epochs', category: 'training', format: 'number', higherIsBetter: false },
  { key: 'duration_seconds', label: 'Training Time', category: 'training', format: 'time', higherIsBetter: false },
  // Operational metrics
  { key: 'model_size_bytes', label: 'Model Size', category: 'operational', format: 'size', higherIsBetter: false },
  { key: 'inference_latency_ms', label: 'Latency', category: 'operational', format: 'number', higherIsBetter: false },
]

function formatValue(value: number | null | undefined, format: string): string {
  if (value === null || value === undefined) return '—'
  switch (format) {
    case 'percent':
      return `${value.toFixed(1)}%`
    case 'number':
      return value < 0.01 ? value.toExponential(2) : value.toFixed(2)
    case 'time':
      if (value < 60) return `${value.toFixed(0)}s`
      if (value < 3600) return `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`
      return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`
    case 'size':
      const gb = value / (1024 * 1024 * 1024)
      if (gb >= 1) return `${gb.toFixed(2)} GB`
      const mb = value / (1024 * 1024)
      return `${mb.toFixed(0)} MB`
    default:
      return String(value)
  }
}

function getMetricValue(profile: ModelProfileData, metric: ComparisonMetric): number | null {
  switch (metric.key) {
    case 'custom_eval_pass_rate':
      return profile.custom_eval_pass_rate
    case 'benchmark_avg_score':
      return profile.benchmark_avg_score
    case 'final_loss':
      return profile.final_metrics.final_loss ?? null
    case 'training_examples':
      return profile.final_metrics.num_train_samples ?? null
    case 'num_epochs':
      return profile.config.num_epochs ?? null
    case 'duration_seconds':
      return profile.duration_seconds
    case 'model_size_bytes':
      return profile.model_size_bytes
    case 'inference_latency_ms':
      return profile.inference_latency_ms
    default:
      return null
  }
}

function ComparisonIndicator({ value, bestValue, metric }: {
  value: number | null
  bestValue: number | null
  metric: ComparisonMetric
}) {
  if (value === null || bestValue === null) return null

  const isBest = value === bestValue
  const diff = value - bestValue
  const diffPercent = bestValue !== 0 ? Math.abs(diff / bestValue * 100) : 0

  if (isBest) {
    return (
      <div className="flex items-center gap-1 text-status-success text-xs">
        <CheckCircle className="w-3 h-3" />
        Best
      </div>
    )
  }

  const isWorse = metric.higherIsBetter ? diff < 0 : diff > 0
  const Icon = isWorse ? TrendingDown : TrendingUp
  const color = isWorse ? 'text-status-error' : 'text-status-warning'

  return (
    <div className={clsx('flex items-center gap-1 text-xs', color)}>
      <Icon className="w-3 h-3" />
      {diffPercent > 0.1 ? `${diffPercent.toFixed(1)}%` : '<0.1%'}
    </div>
  )
}

export function ModelComparison({ modelIds, onBack, onAddModel, onRemoveModel }: ModelComparisonProps) {
  const [profiles, setProfiles] = useState<Record<string, ModelProfileData>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeCategory, setActiveCategory] = useState<'all' | 'performance' | 'training' | 'operational'>('all')

  // Fetch all profiles
  const fetchProfiles = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    const newProfiles: Record<string, ModelProfileData> = {}

    for (const modelId of modelIds) {
      const result = await modelsApi.get(modelId)
      if (result.ok && result.data) {
        newProfiles[modelId] = result.data
      }
    }

    if (Object.keys(newProfiles).length === 0 && modelIds.length > 0) {
      setError('Failed to load any models')
    }

    setProfiles(newProfiles)
    setIsLoading(false)
  }, [modelIds])

  useEffect(() => {
    if (modelIds.length > 0) {
      fetchProfiles()
    } else {
      setProfiles({})
      setIsLoading(false)
    }
  }, [modelIds, fetchProfiles])

  // Calculate best values for each metric
  const bestValues = COMPARISON_METRICS.reduce((acc, metric) => {
    const values = Object.values(profiles)
      .map(p => getMetricValue(p, metric))
      .filter((v): v is number => v !== null)

    if (values.length > 0) {
      acc[metric.key] = metric.higherIsBetter
        ? Math.max(...values)
        : Math.min(...values)
    }
    return acc
  }, {} as Record<string, number>)

  const filteredMetrics = activeCategory === 'all'
    ? COMPARISON_METRICS
    : COMPARISON_METRICS.filter(m => m.category === activeCategory)

  if (modelIds.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center">
        <BarChart3 className="w-16 h-16 text-text-muted mb-4 opacity-30" />
        <h2 className="text-xl font-medium text-text-primary mb-2">No Models Selected</h2>
        <p className="text-text-muted mb-6">Select 2-3 models to compare</p>
        <button onClick={onBack} className="btn-primary">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Models
        </button>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <button onClick={onBack} className="p-2 rounded-lg hover:bg-background-tertiary text-text-muted">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-2xl font-semibold text-text-primary">Model Comparison</h1>
              <p className="text-sm text-text-muted mt-1">
                Comparing {modelIds.length} model{modelIds.length !== 1 ? 's' : ''}
              </p>
            </div>
          </div>

          {modelIds.length < 3 && (
            <button onClick={onAddModel} className="btn-secondary">
              <Plus className="w-4 h-4 mr-2" />
              Add Model
            </button>
          )}
        </div>

        {/* Category Tabs */}
        <div className="flex items-center gap-2">
          {[
            { value: 'all', label: 'All Metrics', icon: BarChart3 },
            { value: 'performance', label: 'Performance', icon: Activity },
            { value: 'training', label: 'Training', icon: TrendingUp },
            { value: 'operational', label: 'Operational', icon: Cpu },
          ].map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setActiveCategory(value as typeof activeCategory)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors',
                activeCategory === value
                  ? 'bg-primary text-white'
                  : 'bg-background-tertiary text-text-muted hover:text-text-primary'
              )}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-64">
            <AlertCircle className="w-12 h-12 text-status-error mb-4" />
            <p className="text-text-muted">{error}</p>
          </div>
        ) : (
          <div className="card-elevated overflow-hidden">
            <table className="w-full">
              <thead className="bg-background-tertiary">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted w-48">Metric</th>
                  {modelIds.map(modelId => {
                    const profile = profiles[modelId]
                    return (
                      <th key={modelId} className="px-4 py-3 text-center relative">
                        {profile && (
                          <button
                            onClick={() => onRemoveModel(modelId)}
                            className="absolute top-2 right-2 p-1 rounded hover:bg-background-secondary text-text-muted hover:text-text-primary"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        )}
                        <div className="pr-6">
                          <div className="font-medium text-text-primary">
                            {profile?.display_name || modelId}
                          </div>
                          {profile && (
                            <div className="text-xs text-text-muted mt-1">
                              {profile.training_strategy.toUpperCase()} • {profile.base_model.split('/').pop()}
                            </div>
                          )}
                        </div>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {filteredMetrics.map(metric => (
                  <tr key={metric.key} className="hover:bg-background-tertiary/50">
                    <td className="px-4 py-3">
                      <div className="font-medium text-text-primary">{metric.label}</div>
                      <div className="text-xs text-text-muted capitalize">{metric.category}</div>
                    </td>
                    {modelIds.map(modelId => {
                      const profile = profiles[modelId]
                      const value = profile ? getMetricValue(profile, metric) : null
                      const isBest = value === bestValues[metric.key]

                      return (
                        <td key={modelId} className="px-4 py-3 text-center">
                          <div className={clsx(
                            'text-lg font-semibold',
                            isBest ? 'text-status-success' : 'text-text-primary'
                          )}>
                            {formatValue(value, metric.format)}
                          </div>
                          <ComparisonIndicator
                            value={value}
                            bestValue={bestValues[metric.key]}
                            metric={metric}
                          />
                        </td>
                      )
                    })}
                  </tr>
                ))}

                {/* Benchmark Results Section */}
                <tr className="bg-background-tertiary">
                  <td colSpan={modelIds.length + 1} className="px-4 py-2 text-sm font-semibold text-text-primary">
                    Benchmark Breakdown
                  </td>
                </tr>
                {(() => {
                  // Collect all unique benchmark names
                  const benchmarkNames = new Set<string>()
                  Object.values(profiles).forEach(p => {
                    Object.keys(p.benchmarks).forEach(name => benchmarkNames.add(name))
                  })

                  return Array.from(benchmarkNames).map(benchName => (
                    <tr key={benchName} className="hover:bg-background-tertiary/50">
                      <td className="px-4 py-3">
                        <div className="font-medium text-text-primary">{benchName}</div>
                        <div className="text-xs text-text-muted">Benchmark</div>
                      </td>
                      {modelIds.map(modelId => {
                        const profile = profiles[modelId]
                        const bench = profile?.benchmarks[benchName]
                        const bestBenchScore = Math.max(
                          ...Object.values(profiles)
                            .map(p => p.benchmarks[benchName]?.score)
                            .filter((v): v is number => v !== undefined)
                        )
                        const isBest = bench?.score === bestBenchScore

                        return (
                          <td key={modelId} className="px-4 py-3 text-center">
                            {bench ? (
                              <>
                                <div className={clsx(
                                  'text-lg font-semibold',
                                  isBest ? 'text-status-success' : 'text-text-primary'
                                )}>
                                  {bench.score.toFixed(1)}%
                                </div>
                                <div className="text-xs text-text-muted">
                                  {bench.passed}/{bench.total}
                                </div>
                              </>
                            ) : (
                              <span className="text-text-muted">—</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))
                })()}

                {/* Custom Eval Results Section */}
                <tr className="bg-background-tertiary">
                  <td colSpan={modelIds.length + 1} className="px-4 py-2 text-sm font-semibold text-text-primary">
                    Custom Evaluations
                  </td>
                </tr>
                {(() => {
                  // Collect all unique eval set IDs
                  const evalSetIds = new Set<string>()
                  Object.values(profiles).forEach(p => {
                    Object.keys(p.custom_evals).forEach(id => evalSetIds.add(id))
                  })

                  if (evalSetIds.size === 0) {
                    return (
                      <tr>
                        <td colSpan={modelIds.length + 1} className="px-4 py-6 text-center text-text-muted">
                          No custom evaluations available
                        </td>
                      </tr>
                    )
                  }

                  return Array.from(evalSetIds).map(evalId => (
                    <tr key={evalId} className="hover:bg-background-tertiary/50">
                      <td className="px-4 py-3">
                        <div className="font-medium text-text-primary">{evalId}</div>
                        <div className="text-xs text-text-muted">Custom Eval</div>
                      </td>
                      {modelIds.map(modelId => {
                        const profile = profiles[modelId]
                        const eval_ = profile?.custom_evals[evalId]
                        const bestEvalScore = Math.max(
                          ...Object.values(profiles)
                            .map(p => p.custom_evals[evalId]?.pass_rate)
                            .filter((v): v is number => v !== undefined)
                        )
                        const isBest = eval_?.pass_rate === bestEvalScore

                        return (
                          <td key={modelId} className="px-4 py-3 text-center">
                            {eval_ ? (
                              <>
                                <div className={clsx(
                                  'text-lg font-semibold',
                                  isBest ? 'text-status-success' : 'text-text-primary'
                                )}>
                                  {eval_.pass_rate.toFixed(1)}%
                                </div>
                                <div className="text-xs text-text-muted">
                                  {eval_.passed}/{eval_.total}
                                </div>
                              </>
                            ) : (
                              <span className="text-text-muted">—</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))
                })()}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
