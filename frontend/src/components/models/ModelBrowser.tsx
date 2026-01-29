import { useState, useEffect, useCallback } from 'react'
import {
  Search,
  Filter,
  Grid3X3,
  List,
  Trophy,
  Plus,
  RefreshCw,
  BarChart3,
  Star,
  ChevronDown,
  X,
  Loader2,
  TrendingUp
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, ModelSummary, LeaderboardEntry } from '../../services/api'
import { useTutorialComplete } from '../../hooks'
import { ModelCard } from './ModelCard'

interface ModelBrowserProps {
  onSelectModel: (modelId: string) => void
  onTrainNew?: () => void
  onCompare?: (modelIds: string[]) => void
  onViewTrends?: () => void
}

type ViewMode = 'grid' | 'list' | 'leaderboard'

const SORT_OPTIONS = [
  { value: 'created_at', label: 'Date Created' },
  { value: 'custom_eval', label: 'Custom Eval Score' },
  { value: 'benchmark_avg', label: 'Benchmark Score' },
  { value: 'display_name', label: 'Name' },
  { value: 'model_size', label: 'Model Size' },
]

const STRATEGY_OPTIONS = [
  { value: '', label: 'All Strategies' },
  { value: 'sft', label: 'SFT' },
  { value: 'dpo', label: 'DPO' },
  { value: 'grpo', label: 'GRPO' },
  { value: 'distillation', label: 'Distillation' },
]

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'ready', label: 'Ready' },
  { value: 'needs_eval', label: 'Needs Eval' },
  { value: 'training', label: 'Training' },
  { value: 'archived', label: 'Archived' },
]

export function ModelBrowser({ onSelectModel, onTrainNew, onCompare, onViewTrends }: ModelBrowserProps) {
  const { complete: completeTutorialStep } = useTutorialComplete()
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [models, setModels] = useState<ModelSummary[]>([])
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [total, setTotal] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  // Filters
  const [search, setSearch] = useState('')
  const [strategy, setStrategy] = useState('')
  const [status, setStatus] = useState('')
  const [starredOnly, setStarredOnly] = useState(false)
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

  // Compare mode
  const [compareMode, setCompareMode] = useState(false)
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([])

  // Fetch models
  const fetchModels = useCallback(async () => {
    setIsLoading(true)
    const result = await modelsApi.list({
      strategy: strategy || undefined,
      status: status || undefined,
      starred: starredOnly || undefined,
      sort_by: sortBy,
      sort_order: sortOrder,
      limit: 50,
    })
    if (result.ok && result.data) {
      let filtered = result.data.models || []
      // Client-side search filter
      if (search && filtered.length > 0) {
        const searchLower = search.toLowerCase()
        filtered = filtered.filter(m =>
          m.display_name?.toLowerCase().includes(searchLower) ||
          m.base_model?.toLowerCase().includes(searchLower) ||
          m.tags?.some(t => t.toLowerCase().includes(searchLower))
        )
      }
      setModels(filtered)
      setTotal(result.data.total || 0)
    } else {
      setModels([])
      setTotal(0)
    }
    setIsLoading(false)
  }, [strategy, status, starredOnly, sortBy, sortOrder, search])

  // Fetch leaderboard
  const fetchLeaderboard = useCallback(async () => {
    const result = await modelsApi.leaderboard('custom_eval_pass_rate', 20)
    if (result.ok && result.data) {
      setLeaderboard(result.data.entries)
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchModels()
    fetchLeaderboard()
  }, [fetchModels, fetchLeaderboard])

  const handleSelectModel = (modelId: string) => {
    onSelectModel(modelId)
    completeTutorialStep('view_model')
  }

  const handleCompare = (modelId: string) => {
    if (!compareMode) {
      setCompareMode(true)
      setSelectedForCompare([modelId])
    } else if (selectedForCompare.includes(modelId)) {
      setSelectedForCompare(prev => prev.filter(id => id !== modelId))
    } else if (selectedForCompare.length < 3) {
      setSelectedForCompare(prev => [...prev, modelId])
    }
  }

  const exitCompareMode = () => {
    setCompareMode(false)
    setSelectedForCompare([])
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-semibold text-text-primary">Models</h1>
            <p className="text-sm text-text-secondary mt-1">
              {total} trained model{total !== 1 ? 's' : ''}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {compareMode ? (
              <>
                <span className="text-sm text-text-muted mr-2">
                  {selectedForCompare.length}/3 selected
                </span>
                <button
                  onClick={() => {
                    if (onCompare && selectedForCompare.length >= 2) {
                      onCompare(selectedForCompare)
                      exitCompareMode()
                    }
                  }}
                  disabled={selectedForCompare.length < 2}
                  className="btn-primary"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Compare
                </button>
                <button onClick={exitCompareMode} className="btn-secondary">
                  <X className="w-4 h-4 mr-2" />
                  Cancel
                </button>
              </>
            ) : (
              <>
                {onViewTrends && (
                  <button onClick={onViewTrends} className="btn-secondary">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Trends
                  </button>
                )}
                <button
                  onClick={() => setCompareMode(true)}
                  className="btn-secondary"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Compare
                </button>
                {onTrainNew && (
                  <button onClick={onTrainNew} className="btn-primary">
                    <Plus className="w-4 h-4 mr-2" />
                    Train New
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Filters Row */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input pl-9 w-full"
            />
          </div>

          {/* Strategy Filter */}
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="input"
          >
            {STRATEGY_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            className="input"
          >
            {STATUS_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>

          {/* Starred Only */}
          <button
            onClick={() => setStarredOnly(!starredOnly)}
            className={clsx(
              'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors',
              starredOnly
                ? 'bg-yellow-500/20 text-yellow-500'
                : 'bg-background-tertiary text-text-muted hover:text-text-primary'
            )}
          >
            <Star className={clsx('w-4 h-4', starredOnly && 'fill-current')} />
            Starred
          </button>

          {/* Sort */}
          <div className="flex items-center gap-1">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="input"
            >
              {SORT_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-2 rounded-lg bg-background-tertiary text-text-muted hover:text-text-primary"
            >
              <ChevronDown className={clsx('w-4 h-4 transition-transform', sortOrder === 'asc' && 'rotate-180')} />
            </button>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-background-tertiary rounded-lg p-1">
            {[
              { mode: 'grid' as ViewMode, icon: Grid3X3, label: 'Grid' },
              { mode: 'list' as ViewMode, icon: List, label: 'List' },
              { mode: 'leaderboard' as ViewMode, icon: Trophy, label: 'Leaderboard' },
            ].map(({ mode, icon: Icon, label }) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={clsx(
                  'p-2 rounded transition-colors',
                  viewMode === mode
                    ? 'bg-primary text-white'
                    : 'text-text-muted hover:text-text-primary'
                )}
                title={label}
              >
                <Icon className="w-4 h-4" />
              </button>
            ))}
          </div>

          {/* Refresh */}
          <button
            onClick={() => { fetchModels(); fetchLeaderboard() }}
            disabled={isLoading}
            className="p-2 rounded-lg bg-background-tertiary text-text-muted hover:text-text-primary"
          >
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {isLoading && models.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
          </div>
        ) : models.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="w-16 h-16 rounded-full bg-background-tertiary flex items-center justify-center mb-4">
              <Grid3X3 className="w-8 h-8 text-text-muted" />
            </div>
            <h3 className="text-lg font-medium text-text-primary mb-2">No models found</h3>
            <p className="text-text-muted mb-4">
              {search || strategy || status
                ? 'Try adjusting your filters'
                : 'Train your first model to get started'}
            </p>
            {onTrainNew && !search && !strategy && !status && (
              <button onClick={onTrainNew} className="btn-primary">
                <Plus className="w-4 h-4 mr-2" />
                Train New Model
              </button>
            )}
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {models.map(model => (
              <div
                key={model.model_id}
                className={clsx(
                  'relative',
                  compareMode && selectedForCompare.includes(model.model_id) && 'ring-2 ring-primary rounded-lg'
                )}
              >
                {compareMode && (
                  <div
                    className="absolute top-2 left-2 z-10 w-6 h-6 rounded-full border-2 border-primary bg-background-secondary flex items-center justify-center cursor-pointer"
                    onClick={(e) => { e.stopPropagation(); handleCompare(model.model_id) }}
                  >
                    {selectedForCompare.includes(model.model_id) && (
                      <div className="w-3 h-3 rounded-full bg-primary" />
                    )}
                  </div>
                )}
                <ModelCard
                  model={model}
                  onSelect={handleSelectModel}
                  onCompare={handleCompare}
                  onRefresh={fetchModels}
                />
              </div>
            ))}
          </div>
        ) : viewMode === 'list' ? (
          <div className="card-elevated">
            <table className="w-full">
              <thead className="bg-background-tertiary">
                <tr>
                  {compareMode && <th className="w-12 px-4 py-3"></th>}
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Model</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Base</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Strategy</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Custom Eval</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Benchmark</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Size</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {models.map(model => (
                  <tr
                    key={model.model_id}
                    onClick={() => !compareMode && handleSelectModel(model.model_id)}
                    className="hover:bg-background-tertiary cursor-pointer"
                  >
                    {compareMode && (
                      <td className="px-4 py-3">
                        <div
                          className="w-5 h-5 rounded-full border-2 border-primary flex items-center justify-center cursor-pointer"
                          onClick={(e) => { e.stopPropagation(); handleCompare(model.model_id) }}
                        >
                          {selectedForCompare.includes(model.model_id) && (
                            <div className="w-2.5 h-2.5 rounded-full bg-primary" />
                          )}
                        </div>
                      </td>
                    )}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {model.starred && <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />}
                        <span className="font-medium text-text-primary">{model.display_name}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-text-secondary text-sm">
                      {model.base_model.split('/').pop()}
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded bg-primary/20 text-primary text-xs">
                        {model.training_strategy.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-text-primary">
                      {model.custom_eval_pass_rate !== null ? `${model.custom_eval_pass_rate.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-text-primary">
                      {model.benchmark_avg_score !== null ? `${model.benchmark_avg_score.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-text-secondary text-sm">
                      {model.model_size_display}
                    </td>
                    <td className="px-4 py-3">
                      <span className={clsx(
                        'px-2 py-0.5 rounded text-xs',
                        model.status === 'ready' ? 'bg-status-success/20 text-status-success' :
                        model.status === 'needs_eval' ? 'bg-status-warning/20 text-status-warning' :
                        'bg-text-muted/20 text-text-muted'
                      )}>
                        {model.status.replace('_', ' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          /* Leaderboard View */
          <div className="card-elevated">
            <div className="p-4 border-b border-border-subtle">
              <div className="flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-500" />
                <h3 className="font-medium text-text-primary">Leaderboard</h3>
                <span className="text-sm text-text-muted">by Custom Eval Pass Rate</span>
              </div>
            </div>
            <table className="w-full">
              <thead className="bg-background-tertiary">
                <tr>
                  <th className="w-16 px-4 py-3 text-center text-sm font-medium text-text-muted">Rank</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Model</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Base</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-text-muted">Strategy</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-text-muted">Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle">
                {leaderboard.map(entry => (
                  <tr
                    key={entry.model_id}
                    onClick={() => handleSelectModel(entry.model_id)}
                    className="hover:bg-background-tertiary cursor-pointer"
                  >
                    <td className="px-4 py-3 text-center">
                      {entry.rank <= 3 ? (
                        <span className={clsx(
                          'inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold',
                          entry.rank === 1 ? 'bg-yellow-500/20 text-yellow-500' :
                          entry.rank === 2 ? 'bg-gray-400/20 text-gray-400' :
                          'bg-amber-700/20 text-amber-700'
                        )}>
                          {entry.rank}
                        </span>
                      ) : (
                        <span className="text-text-muted">{entry.rank}</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-medium text-text-primary">{entry.display_name}</span>
                    </td>
                    <td className="px-4 py-3 text-text-secondary text-sm">
                      {entry.base_model.split('/').pop()}
                    </td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded bg-primary/20 text-primary text-xs">
                        {entry.strategy.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-lg font-semibold text-text-primary">
                        {entry.value.toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
