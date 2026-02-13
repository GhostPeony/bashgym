import { useState, useEffect } from 'react'
import {
  Sparkles,
  Loader2,
  Zap,
  Target,
  Database,
  Layers,
  ChevronDown,
  Settings
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  syntheticApi,
  SyntheticGenerateRequest,
  SyntheticJobStatus,
  SyntheticPreset,
  tracesApi,
  RepoInfo
} from '../../services/api'
import { useTutorialComplete } from '../../hooks'

type Strategy = 'trace_seeded' | 'augmented' | 'schema_driven'
type Provider = 'nim' | 'anthropic'
type Preset = 'quick_test' | 'balanced' | 'production' | 'custom'
type RepoFilter = 'single' | 'all' | 'selected'

const STRATEGY_INFO: Record<Strategy, { label: string; description: string; icon: typeof Sparkles }> = {
  trace_seeded: {
    label: 'Trace-Seeded',
    description: 'From gold trace patterns',
    icon: Target
  },
  augmented: {
    label: 'Augmented',
    description: 'Variations of existing prompts',
    icon: Layers
  },
  schema_driven: {
    label: 'Schema-Driven',
    description: 'From repository structure',
    icon: Database
  }
}

export interface SyntheticGeneratorState {
  isGenerating: boolean
  canGenerate: boolean
  targetExamples: number
  traceCount: number
  onGenerate: () => void
}

interface SyntheticGeneratorProps {
  onStateChange?: (state: SyntheticGeneratorState) => void
}

export function SyntheticGenerator({ onStateChange }: SyntheticGeneratorProps) {
  const { complete: completeTutorialStep } = useTutorialComplete()

  // Form state
  const [strategy, setStrategy] = useState<Strategy>('trace_seeded')
  const [provider, setProvider] = useState<Provider>('nim')
  const [preset, setPreset] = useState<Preset>('balanced')
  const [customTarget, setCustomTarget] = useState<number>(500)
  const [repoFilter, setRepoFilter] = useState<RepoFilter>('all')
  const [selectedRepos, setSelectedRepos] = useState<string[]>([])
  const [repoDropdownOpen, setRepoDropdownOpen] = useState(false)

  // Data state
  const [presets, setPresets] = useState<Record<string, SyntheticPreset>>({})
  const [repos, setRepos] = useState<RepoInfo[]>([])
  const [traceCount, setTraceCount] = useState<number>(0)
  const [jobs, setJobs] = useState<SyntheticJobStatus[]>([])

  // UI state
  const [isLoading, setIsLoading] = useState(true)
  const [isGenerating, setIsGenerating] = useState(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  // Fetch presets and repos on mount
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [presetsResult, reposResult, jobsResult, tracesResult] = await Promise.all([
          syntheticApi.getPresets(),
          tracesApi.listRepos(),
          syntheticApi.listJobs(),
          tracesApi.list({ status: 'gold' })
        ])

        if (presetsResult.ok && presetsResult.data) {
          setPresets(presetsResult.data)
        }
        if (reposResult.ok && reposResult.data) {
          setRepos(reposResult.data)
        }
        if (jobsResult.ok && jobsResult.data) {
          setJobs(jobsResult.data)
        }
        if (tracesResult.ok && tracesResult.data) {
          setTraceCount(tracesResult.data.length)
        }
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [])

  // Poll active job
  useEffect(() => {
    if (!activeJobId) return

    const interval = setInterval(async () => {
      const result = await syntheticApi.getJobStatus(activeJobId)
      if (result.ok && result.data) {
        setJobs(prev => prev.map(j =>
          j.job_id === activeJobId ? result.data! : j
        ))

        if (result.data.status === 'completed' || result.data.status === 'failed') {
          setActiveJobId(null)
        }
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [activeJobId])

  const handleGenerate = async () => {
    setIsGenerating(true)
    try {
      const request: SyntheticGenerateRequest = {
        strategy,
        repo_filter: repoFilter === 'all' ? 'all' : repoFilter === 'single' ? 'single' : 'multi',
        selected_repos: repoFilter === 'all' ? [] : selectedRepos,
        preset,
        target_examples: preset === 'custom' ? customTarget : undefined,
        provider,
        merge_mode: 'mixed'
      }

      const result = await syntheticApi.generate(request)
      if (result.ok && result.data) {
        const newJob: SyntheticJobStatus = {
          job_id: result.data.job_id,
          status: 'queued',
          progress: { current: 0, total: 0 }
        }
        setJobs(prev => [newJob, ...prev])
        setActiveJobId(result.data.job_id)
        completeTutorialStep('generate_examples')
      }
    } finally {
      setIsGenerating(false)
    }
  }

  const toggleRepo = (repoName: string) => {
    setSelectedRepos(prev =>
      prev.includes(repoName)
        ? prev.filter(r => r !== repoName)
        : [...prev, repoName]
    )
  }

  const getTargetExamples = () => {
    if (preset === 'custom') return customTarget
    return presets[preset]?.target_examples || 500
  }

  const getMultiplier = () => {
    const target = getTargetExamples()
    if (traceCount === 0) return 0
    return Math.ceil(target / traceCount)
  }

  const canGenerate = !isGenerating && (repoFilter === 'all' || selectedRepos.length > 0)

  // Notify parent of state changes
  useEffect(() => {
    if (onStateChange && !isLoading) {
      onStateChange({
        isGenerating,
        canGenerate,
        targetExamples: getTargetExamples(),
        traceCount,
        onGenerate: handleGenerate
      })
    }
  }, [isGenerating, canGenerate, preset, customTarget, traceCount, repoFilter, selectedRepos, isLoading])

  const getRepoLabel = () => {
    if (repoFilter === 'all') return 'All repositories'
    if (selectedRepos.length === 0) return 'Select repositories...'
    if (selectedRepos.length === 1) return selectedRepos[0]
    return `${selectedRepos.length} repositories`
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-accent" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
        {/* Repository Filter - Dropdown at top */}
        <div className="flex items-center gap-4">
          <label className="font-mono text-xs uppercase tracking-widest text-text-secondary whitespace-nowrap">Source:</label>
          <div className="relative flex-1">
            <button
              onClick={() => setRepoDropdownOpen(!repoDropdownOpen)}
              className="w-full flex items-center justify-between px-3 py-2 bg-background-secondary border-brutal border-border rounded-brutal text-sm text-text-primary hover:shadow-brutal-sm transition-shadow"
            >
              <span className="font-mono">{getRepoLabel()}</span>
              <ChevronDown className={clsx('w-4 h-4 text-text-muted transition-transform', repoDropdownOpen && 'rotate-180')} />
            </button>
            {repoDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-background-secondary border-brutal border-border rounded-brutal shadow-brutal z-10 max-h-64 overflow-auto">
                <button
                  onClick={() => {
                    setRepoFilter('all')
                    setSelectedRepos([])
                    setRepoDropdownOpen(false)
                  }}
                  className={clsx(
                    'w-full px-3 py-2 text-left text-sm font-mono hover:bg-accent-light transition-colors',
                    repoFilter === 'all' ? 'text-accent-dark font-semibold' : 'text-text-primary'
                  )}
                >
                  All repositories ({traceCount} traces)
                </button>
                <div className="border-t-2 border-border" />
                {repos.map(repo => (
                  <button
                    key={repo.name}
                    onClick={() => {
                      setRepoFilter('selected')
                      toggleRepo(repo.name)
                    }}
                    className={clsx(
                      'w-full px-3 py-2 text-left text-sm font-mono hover:bg-accent-light transition-colors flex items-center justify-between',
                      selectedRepos.includes(repo.name) ? 'text-accent-dark font-semibold' : 'text-text-primary'
                    )}
                  >
                    <span>{repo.name}</span>
                    {repo.trace_count && <span className="text-xs text-text-muted">{repo.trace_count}</span>}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Strategy Selection - Compact toggle with description */}
        <div className="flex items-center gap-4">
          <label className="font-mono text-xs uppercase tracking-widest text-text-secondary whitespace-nowrap">Strategy:</label>
          <div className="flex gap-1 p-1 border-brutal border-border rounded-brutal bg-background-secondary">
            {(Object.entries(STRATEGY_INFO) as [Strategy, typeof STRATEGY_INFO['trace_seeded']][]).map(([key, info]) => (
              <button
                key={key}
                onClick={() => setStrategy(key)}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-brutal text-sm font-mono transition-all',
                  strategy === key
                    ? 'bg-accent-light text-accent-dark border-brutal border-border shadow-brutal-sm'
                    : 'text-text-secondary hover:text-text-primary border border-transparent'
                )}
              >
                <info.icon className="w-4 h-4" />
                {info.label}
              </button>
            ))}
          </div>
          <span className="text-xs text-text-muted font-mono">
            {STRATEGY_INFO[strategy].description}
          </span>
        </div>

        {/* Target Size - Compact preset buttons + custom slider */}
        <div className="flex items-center gap-4">
          <label className="font-mono text-xs uppercase tracking-widest text-text-secondary whitespace-nowrap">Target:</label>
          <div className="flex items-center gap-2 flex-1">
            <div className="flex gap-1 p-1 border-brutal border-border rounded-brutal bg-background-secondary">
              {Object.entries(presets).map(([key, presetData]) => (
                <button
                  key={key}
                  onClick={() => setPreset(key as Preset)}
                  className={clsx(
                    'px-3 py-1.5 rounded-brutal text-sm font-mono transition-all',
                    key === preset
                      ? 'bg-accent-light text-accent-dark border-brutal border-border shadow-brutal-sm'
                      : 'text-text-secondary hover:text-text-primary border border-transparent'
                  )}
                  title={presetData.label}
                >
                  {presetData.target_examples ? presetData.target_examples.toLocaleString() : 'Custom'}
                </button>
              ))}
            </div>
            {preset === 'custom' ? (
              <div className="flex items-center gap-2 flex-1">
                <input
                  type="range"
                  min="50"
                  max="5000"
                  step="50"
                  value={customTarget}
                  onChange={(e) => setCustomTarget(parseInt(e.target.value))}
                  className="flex-1 h-1.5 bg-background-secondary rounded-brutal appearance-none cursor-pointer accent-accent"
                />
                <input
                  type="number"
                  min="50"
                  max="10000"
                  value={customTarget}
                  onChange={(e) => setCustomTarget(parseInt(e.target.value))}
                  className="input w-20 text-sm text-center font-mono"
                />
              </div>
            ) : (
              <span className="text-xs text-text-muted font-mono">
                {presets[preset]?.label || ''}
              </span>
            )}
          </div>
        </div>

        {/* Provider Selection - Compact toggle */}
        <div className="flex items-center gap-4">
          <label className="font-mono text-xs uppercase tracking-widest text-text-secondary whitespace-nowrap">Provider:</label>
          <div className="flex gap-1 p-1 border-brutal border-border rounded-brutal bg-background-secondary">
            <button
              onClick={() => setProvider('nim')}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-brutal text-sm font-mono transition-all',
                provider === 'nim'
                  ? 'bg-accent-light text-accent-dark border-brutal border-border shadow-brutal-sm'
                  : 'text-text-secondary hover:text-text-primary border border-transparent'
              )}
            >
              <Zap className="w-4 h-4" />
              NVIDIA NIM
            </button>
            <button
              onClick={() => setProvider('anthropic')}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-brutal text-sm font-mono transition-all',
                provider === 'anthropic'
                  ? 'bg-accent-light text-accent-dark border-brutal border-border shadow-brutal-sm'
                  : 'text-text-secondary hover:text-text-primary border border-transparent'
              )}
            >
              <Sparkles className="w-4 h-4" />
              Claude
            </button>
          </div>
          <span className="text-xs text-text-muted font-mono">
            {provider === 'nim' ? 'Fast, cost-effective' : 'Higher quality'}
          </span>
        </div>

        {/* Summary Stats */}
        <div className="flex items-center gap-6 pt-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Seeds:</span>
            <span className="font-mono text-sm font-semibold text-text-primary">{traceCount}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Target:</span>
            <span className="font-mono text-sm font-semibold text-text-primary">{getTargetExamples().toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Multiplier:</span>
            <span className="font-mono text-sm font-semibold text-accent">{getMultiplier()}x</span>
          </div>
        </div>

      {/* Info text */}
      <p className="text-xs text-text-muted font-mono pt-2">
        Generates NeMo-compatible JSONL saved to data/synthetic/. Recommended: 500-2000 examples for LoRA fine-tuning.
      </p>
    </div>
  )
}
