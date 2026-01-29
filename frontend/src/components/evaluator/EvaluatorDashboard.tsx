import { useState, useEffect } from 'react'
import {
  BarChart3,
  Play,
  Pause,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Target,
  Zap,
  Brain,
  Code,
  Shield,
  MessageSquare,
  FileText,
  TrendingUp,
  Award,
  Clock,
  X,
  Info,
  AlertTriangle,
  XCircle,
  Timer
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, evaluatorApi } from '../../services/api'

// Benchmark categories and their benchmarks
// All benchmarks load from HuggingFace datasets on first run

// Detailed benchmark info with tooltips
const BENCHMARK_INFO: Record<string, { tooltip: string; example: string; evaluates: string }> = {
  simple_test: {
    tooltip: 'Basic Python functions for quick validation',
    example: 'def add(a, b): return a + b',
    evaluates: 'Tests run against generated code'
  },
  humaneval: {
    tooltip: 'Complete Python functions from docstrings with examples. Industry standard for code generation.',
    example: 'Given: def has_close_elements(numbers, threshold): """Check if any two numbers closer than threshold..."""',
    evaluates: 'Unit tests verify functional correctness'
  },
  mbpp: {
    tooltip: 'Solve basic Python programming tasks from natural language descriptions.',
    example: '"Write a function to find similar elements from two tuples"',
    evaluates: 'Assert statements check output correctness'
  },
  bigcodebench: {
    tooltip: 'Complex coding tasks requiring multiple libraries (numpy, pandas, etc). Tests real-world coding ability.',
    example: 'Tasks involving data processing, file I/O, API calls',
    evaluates: 'Execution + output validation'
  },
  ds1000: {
    tooltip: 'Data science problems using pandas, numpy, sklearn. Realistic data manipulation tasks.',
    example: '"Filter dataframe where column X > threshold and sort by Y"',
    evaluates: 'Output comparison with reference solution'
  },
  bfcl: {
    tooltip: 'Tests function/tool calling accuracy. Model must generate correct function calls.',
    example: 'User: "What is the weather in SF?" → generate get_weather(city="San Francisco")',
    evaluates: 'Function name and parameters match expected'
  },
  gsm8k: {
    tooltip: 'Grade school math word problems requiring multi-step arithmetic reasoning.',
    example: '"Janet has 16 eggs, eats 3, bakes 4. Sells rest at $2 each. Daily earnings?"',
    evaluates: 'Numeric answer extraction and comparison'
  },
  arc: {
    tooltip: 'Science and reasoning questions (multiple choice). Tests factual knowledge and logic.',
    example: '"What causes day and night?" A) Sun moves B) Earth rotates C) Moon phases D) Seasons',
    evaluates: 'Correct option selection'
  },
  hellaswag: {
    tooltip: 'Commonsense reasoning - pick the most logical sentence continuation from 4 options.',
    example: 'Context: "A man is sitting on a roof. He..." → Pick best continuation',
    evaluates: 'Multiple choice accuracy'
  },
  toxigen: {
    tooltip: 'Detect toxic language targeting specific demographic groups. Tests model safety.',
    example: '"[text sample]" → Classify as toxic or safe',
    evaluates: 'Binary classification accuracy against human ratings'
  },
  bbq: {
    tooltip: 'Bias Benchmark for QA - tests for stereotyping in question answering.',
    example: 'Context + question with 3 choices → Pick correct non-biased answer',
    evaluates: 'Accuracy on ambiguous questions where bias could influence'
  },
  swe_bench: {
    tooltip: 'Real GitHub issues requiring code patches. Tests agentic coding ability.',
    example: 'Given issue description → Generate correct patch to fix the bug',
    evaluates: 'Patches that pass repository tests'
  },
}

const BENCHMARK_CATEGORIES = {
  code: {
    label: 'Code Generation',
    icon: Code,
    description: 'Tests ability to write correct, functional code',
    benchmarks: [
      { id: 'simple_test', name: 'Simple Test', description: 'Quick E2E validation (3 problems)', difficulty: 'easy' },
      { id: 'humaneval', name: 'HumanEval', description: 'Python function synthesis (164 problems)', difficulty: 'medium' },
      { id: 'mbpp', name: 'MBPP', description: 'Mostly Basic Python Problems (974 problems)', difficulty: 'easy' },
      { id: 'bigcodebench', name: 'BigCodeBench', description: 'Multi-library code tasks (~1140 problems)', difficulty: 'hard' },
      { id: 'ds1000', name: 'DS-1000', description: 'Data science code generation (1000 problems)', difficulty: 'medium' },
    ]
  },
  function_calling: {
    label: 'Function Calling',
    icon: Zap,
    description: 'Tests ability to correctly invoke tools and APIs',
    benchmarks: [
      { id: 'bfcl', name: 'BFCL', description: 'Berkeley Function Calling Leaderboard (~2000 problems)', difficulty: 'medium' },
      { id: 'tool_use', name: 'Tool Use Accuracy', description: 'Coming soon', difficulty: 'medium', disabled: true },
      { id: 'api_bench', name: 'API-Bench', description: 'Coming soon', difficulty: 'hard', disabled: true },
    ]
  },
  reasoning: {
    label: 'Reasoning',
    icon: Brain,
    description: 'Tests logical thinking and problem solving',
    benchmarks: [
      { id: 'gsm8k', name: 'GSM8K', description: 'Grade school math word problems (1319 test)', difficulty: 'medium' },
      { id: 'math', name: 'MATH', description: 'Coming soon', difficulty: 'hard', disabled: true },
      { id: 'arc', name: 'ARC Challenge', description: 'Science reasoning questions (1172 problems)', difficulty: 'hard' },
      { id: 'hellaswag', name: 'HellaSwag', description: 'Commonsense sentence completion (10042)', difficulty: 'easy' },
    ]
  },
  safety: {
    label: 'Safety',
    icon: Shield,
    description: 'Tests for harmful or biased outputs',
    benchmarks: [
      { id: 'toxigen', name: 'ToxiGen', description: 'Toxic language detection (~9,900 samples)', difficulty: 'medium' },
      { id: 'bbq', name: 'BBQ', description: 'Bias Benchmark for QA (~58K samples)', difficulty: 'medium' },
      { id: 'safety_harness', name: 'Safety Harness', description: 'Coming soon', difficulty: 'medium', disabled: true },
    ]
  },
  agentic: {
    label: 'Agentic',
    icon: Target,
    description: 'Tests multi-step task completion',
    benchmarks: [
      { id: 'swe_bench', name: 'SWE-bench Lite', description: 'GitHub issue resolution (300 tasks)', difficulty: 'hard' },
      { id: 'goal_accuracy', name: 'Goal Accuracy', description: 'Coming soon', difficulty: 'medium', disabled: true },
      { id: 'topic_adherence', name: 'Topic Adherence', description: 'Coming soon', difficulty: 'easy', disabled: true },
      { id: 'webagent', name: 'WebAgent', description: 'Coming soon', difficulty: 'hard', disabled: true },
    ]
  },
  llm_judge: {
    label: 'LLM-as-Judge',
    icon: MessageSquare,
    description: 'Quality assessment by another LLM',
    benchmarks: [
      { id: 'helpfulness', name: 'Helpfulness', description: 'Coming soon', difficulty: 'easy', disabled: true },
      { id: 'correctness', name: 'Correctness', description: 'Coming soon', difficulty: 'medium', disabled: true },
      { id: 'coherence', name: 'Coherence', description: 'Coming soon', difficulty: 'easy', disabled: true },
      { id: 'custom_rubric', name: 'Custom Rubric', description: 'Coming soon', difficulty: 'medium', disabled: true },
    ]
  },
}

interface ErrorAnalysis {
  wrong_answer: number
  syntax_error: number
  runtime_error: number
  timeout: number
  other: number
}

interface BenchmarkResult {
  score: number
  passed: number
  total: number
  duration_seconds?: number
  errors?: ErrorAnalysis
}

interface EvalJob {
  id: string
  model: string
  benchmarks: string[]
  status: 'pending' | 'running' | 'completed' | 'failed'
  created_at: string
  completed_at?: string
  results?: Record<string, BenchmarkResult>
  error?: string
}

interface EvalConfig {
  model: string
  benchmarks: string[]
  judge_model: string
  num_samples: number
  temperature: number
}

const DEFAULT_CONFIG: EvalConfig = {
  model: '',
  benchmarks: [],
  judge_model: 'claude-sonnet-4-5-20250929',
  num_samples: 100,
  temperature: 0.0
}

export function EvaluatorDashboard() {
  const [config, setConfig] = useState<EvalConfig>(DEFAULT_CONFIG)
  const [jobs, setJobs] = useState<EvalJob[]>([])
  const [expandedCategory, setExpandedCategory] = useState<string | null>('code')
  const [isRunning, setIsRunning] = useState(false)
  const [selectedJob, setSelectedJob] = useState<EvalJob | null>(null)
  const [availableModels, setAvailableModels] = useState<string[]>([])

  useEffect(() => {
    // Load available models from API
    const loadModels = async () => {
      console.log('[Evaluator] Loading models...')
      try {
        const result = await modelsApi.list()
        console.log('[Evaluator] Models result:', result)
        if (result.ok && result.data) {
          // Extract model IDs from the response
          setAvailableModels(result.data.map(m => m.model_id))
          // Auto-select first model if none selected
          if (result.data.length > 0 && !config.model) {
            setConfig(prev => ({ ...prev, model: result.data![0].model_id }))
          }
        }
      } catch (error) {
        console.error('[Evaluator] Failed to load models:', error)
      }
    }

    // Load existing evaluation jobs from API
    const loadEvaluations = async () => {
      console.log('[Evaluator] Loading evaluations...')
      try {
        const result = await evaluatorApi.list()
        console.log('[Evaluator] Evaluations result:', result)
        if (result.ok && result.data) {
          setJobs(result.data.map(e => ({
            id: e.job_id,
            model: e.model_id,
            benchmarks: e.benchmarks,
            status: e.status,
            created_at: e.created_at || new Date().toISOString(),
            results: e.results,
            error: e.error
          })))
        }
      } catch (error) {
        console.error('[Evaluator] Failed to load evaluations:', error)
      }
    }

    loadModels()
    loadEvaluations()
  }, [])

  const toggleBenchmark = (benchmarkId: string) => {
    setConfig(prev => ({
      ...prev,
      benchmarks: prev.benchmarks.includes(benchmarkId)
        ? prev.benchmarks.filter(b => b !== benchmarkId)
        : [...prev.benchmarks, benchmarkId]
    }))
  }

  const selectAllInCategory = (categoryKey: string) => {
    const category = BENCHMARK_CATEGORIES[categoryKey as keyof typeof BENCHMARK_CATEGORIES]
    const benchmarkIds = category.benchmarks.map(b => b.id)
    const allSelected = benchmarkIds.every(id => config.benchmarks.includes(id))

    if (allSelected) {
      setConfig(prev => ({
        ...prev,
        benchmarks: prev.benchmarks.filter(b => !benchmarkIds.includes(b))
      }))
    } else {
      setConfig(prev => ({
        ...prev,
        benchmarks: [...new Set([...prev.benchmarks, ...benchmarkIds])]
      }))
    }
  }

  const pollForCompletion = async (jobId: string) => {
    const poll = async () => {
      const result = await evaluatorApi.getStatus(jobId)
      if (result.ok && result.data) {
        const { status, results, error } = result.data
        setJobs(prev => prev.map(j =>
          j.id === jobId ? { ...j, status, results, error } : j
        ))
        if (status === 'running') {
          setTimeout(poll, 2000) // Poll every 2 seconds
        }
      }
    }
    poll()
  }

  const runEvaluation = async () => {
    console.log('[Evaluator] runEvaluation called', { model: config.model, benchmarks: config.benchmarks })
    if (!config.model || config.benchmarks.length === 0) {
      console.log('[Evaluator] Aborted: no model or benchmarks selected')
      return
    }
    setIsRunning(true)

    const newJob: EvalJob = {
      id: `local_${Date.now()}`,
      model: config.model,
      benchmarks: [...config.benchmarks],
      status: 'running',
      created_at: new Date().toISOString()
    }
    setJobs(prev => [newJob, ...prev])

    try {
      console.log('[Evaluator] Calling API...')
      const result = await evaluatorApi.run({
        model_id: config.model,
        benchmarks: config.benchmarks,
        num_samples: config.num_samples
      })
      console.log('[Evaluator] API result:', result)

      if (result.ok && result.data) {
        // Update job with server ID and poll for completion
        const serverJobId = result.data.job_id
        setJobs(prev => prev.map(j =>
          j.id === newJob.id ? { ...j, id: serverJobId } : j
        ))
        pollForCompletion(serverJobId)
      } else {
        setJobs(prev => prev.map(j =>
          j.id === newJob.id ? { ...j, status: 'failed', error: result.error || 'API error' } : j
        ))
      }
    } catch (error) {
      setJobs(prev => prev.map(j =>
        j.id === newJob.id ? { ...j, status: 'failed', error: String(error) } : j
      ))
    } finally {
      setIsRunning(false)
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-status-success'
      case 'medium': return 'text-status-warning'
      case 'hard': return 'text-status-error'
      default: return 'text-text-muted'
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-text-primary">Model Evaluator</h1>
            <p className="text-sm text-text-secondary mt-1">
              Run comprehensive benchmarks on your trained models
            </p>
          </div>
          <button
            onClick={runEvaluation}
            disabled={isRunning || !config.model || config.benchmarks.length === 0}
            className="btn-primary flex items-center gap-2"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Evaluation
              </>
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-3 gap-6">
          {/* Left: Benchmark Selection */}
          <div className="col-span-2 space-y-4">
            {/* Model Selection */}
            <div className="card-elevated p-4">
              <div className="flex items-center gap-3 mb-4">
                <BarChart3 className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-medium text-text-primary">Model to Evaluate</h3>
                  <p className="text-sm text-text-muted">Select a trained model or checkpoint</p>
                </div>
              </div>
              {availableModels.length > 0 ? (
                <select
                  value={config.model}
                  onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}
                  className="input w-full"
                >
                  <option value="">Select a model...</option>
                  {availableModels.map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              ) : (
                <div className="p-4 bg-background-tertiary rounded-lg text-center">
                  <Brain className="w-8 h-8 text-text-muted mx-auto mb-2" />
                  <p className="text-sm text-text-muted">Train a model first to run evaluations</p>
                </div>
              )}
            </div>

            {/* Benchmark Categories */}
            <div className="card-elevated">
              <div className="p-4 border-b border-border-subtle">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Target className="w-5 h-5 text-primary" />
                    <div>
                      <h3 className="font-medium text-text-primary">Benchmarks</h3>
                      <p className="text-sm text-text-muted">{config.benchmarks.length} selected</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setConfig(prev => ({ ...prev, benchmarks: [] }))}
                    className="text-sm text-text-muted hover:text-text-primary"
                  >
                    Clear all
                  </button>
                </div>
              </div>

              <div className="divide-y divide-border-subtle">
                {Object.entries(BENCHMARK_CATEGORIES).map(([key, category]) => {
                  const Icon = category.icon
                  const selectedCount = category.benchmarks.filter(b => config.benchmarks.includes(b.id)).length
                  const isExpanded = expandedCategory === key

                  return (
                    <div key={key}>
                      <button
                        onClick={() => setExpandedCategory(isExpanded ? null : key)}
                        className="w-full flex items-center justify-between p-4 hover:bg-background-tertiary transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <Icon className="w-5 h-5 text-primary" />
                          <span className="font-medium text-text-primary">{category.label}</span>
                          {selectedCount > 0 && (
                            <span className="px-2 py-0.5 text-xs rounded-full bg-primary/20 text-primary">
                              {selectedCount}/{category.benchmarks.length}
                            </span>
                          )}
                        </div>
                        {isExpanded ? (
                          <ChevronDown className="w-5 h-5 text-text-muted" />
                        ) : (
                          <ChevronRight className="w-5 h-5 text-text-muted" />
                        )}
                      </button>

                      {isExpanded && (
                        <div className="px-4 pb-4 space-y-2">
                          <button
                            onClick={() => selectAllInCategory(key)}
                            className="text-xs text-primary hover:underline mb-2"
                          >
                            {category.benchmarks.every(b => config.benchmarks.includes(b.id))
                              ? 'Deselect all'
                              : 'Select all'}
                          </button>
                          {category.benchmarks.map(benchmark => {
                            const isDisabled = 'disabled' in benchmark && benchmark.disabled
                            const benchInfo = BENCHMARK_INFO[benchmark.id]
                            return (
                            <label
                              key={benchmark.id}
                              className={clsx(
                                'flex items-center gap-3 p-3 rounded-lg transition-colors group',
                                isDisabled
                                  ? 'opacity-50 cursor-not-allowed'
                                  : 'cursor-pointer',
                                config.benchmarks.includes(benchmark.id)
                                  ? 'bg-primary/10 border border-primary/30'
                                  : 'bg-background-tertiary hover:bg-background-tertiary/80'
                              )}
                              title={benchInfo?.tooltip}
                            >
                              <input
                                type="checkbox"
                                checked={config.benchmarks.includes(benchmark.id)}
                                onChange={() => !isDisabled && toggleBenchmark(benchmark.id)}
                                disabled={isDisabled}
                                className="rounded"
                              />
                              <div className="flex-1">
                                <div className="flex items-center gap-2">
                                  <span className={clsx('font-medium', isDisabled ? 'text-text-muted' : 'text-text-primary')}>{benchmark.name}</span>
                                  <span className={clsx('text-xs', getDifficultyColor(benchmark.difficulty))}>
                                    {benchmark.difficulty}
                                  </span>
                                </div>
                                <p className="text-xs text-text-muted">{benchmark.description}</p>
                              </div>
                            </label>
                          )})}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Evaluation Settings */}
            <div className="card-elevated p-4">
              <h3 className="font-medium text-text-primary mb-4">Evaluation Settings</h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-1">Judge Model</label>
                  <select
                    value={config.judge_model}
                    onChange={(e) => setConfig(prev => ({ ...prev, judge_model: e.target.value }))}
                    className="input text-sm w-full"
                  >
                    <optgroup label="Anthropic Claude 4.5">
                      <option value="claude-opus-4-5-20251101">Claude Opus 4.5 (Best)</option>
                      <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
                      <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (Fast)</option>
                    </optgroup>
                    <optgroup label="Anthropic Claude 4 (Legacy)">
                      <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                    </optgroup>
                    <optgroup label="Meta Llama">
                      <option value="meta/llama-3.1-70b-instruct">Llama 3.1 70B</option>
                      <option value="meta/llama-3.1-405b-instruct">Llama 3.1 405B</option>
                    </optgroup>
                    <optgroup label="OpenAI">
                      <option value="gpt-4o">GPT-4o</option>
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    </optgroup>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-text-secondary mb-1">Samples per Benchmark</label>
                  <input
                    type="number"
                    min="10"
                    max="1000"
                    value={config.num_samples}
                    onChange={(e) => setConfig(prev => ({ ...prev, num_samples: parseInt(e.target.value) }))}
                    className="input text-sm w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-text-secondary mb-1">Temperature</label>
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    value={config.temperature}
                    onChange={(e) => setConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                    className="input text-sm w-full"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right: Job History & Results */}
          <div className="space-y-4">
            <div className="card-elevated">
              <div className="p-4 border-b border-border-subtle">
                <h3 className="font-medium text-text-primary">Evaluation History</h3>
              </div>
              <div className="divide-y divide-border-subtle max-h-96 overflow-auto">
                {jobs.length === 0 ? (
                  <div className="p-8 text-center">
                    <BarChart3 className="w-10 h-10 text-text-muted mx-auto mb-2" />
                    <p className="text-sm text-text-muted">No evaluations yet</p>
                  </div>
                ) : (
                  jobs.map(job => (
                    <button
                      key={job.id}
                      onClick={() => setSelectedJob(job)}
                      className={clsx(
                        'w-full p-4 text-left hover:bg-background-tertiary transition-colors',
                        selectedJob?.id === job.id && 'bg-primary/5'
                      )}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-text-primary text-sm">{job.model}</span>
                        {job.status === 'running' && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
                        {job.status === 'completed' && <CheckCircle className="w-4 h-4 text-status-success" />}
                        {job.status === 'failed' && <AlertCircle className="w-4 h-4 text-status-error" />}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-text-muted">
                        <span>{job.benchmarks.length} benchmarks</span>
                        <span>•</span>
                        <span>{new Date(job.created_at).toLocaleString()}</span>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>

            {/* Selected Job Results */}
            {selectedJob?.results && (
              <div className="card-elevated p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-text-primary">Results: {selectedJob.model}</h3>
                  <button
                    onClick={() => setSelectedJob(null)}
                    className="p-1 text-text-muted hover:text-text-primary"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="space-y-3 overflow-visible">
                  {Object.entries(selectedJob.results).map(([benchmark, result]) => {
                    const benchInfo = BENCHMARK_INFO[benchmark]
                    const hasErrors = result.errors && (result.errors.wrong_answer > 0 || result.errors.syntax_error > 0 || result.errors.runtime_error > 0 || result.errors.timeout > 0)
                    return (
                    <div key={benchmark} className="p-3 bg-background-tertiary rounded-lg overflow-visible">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-text-primary">{benchmark}</span>
                        <span className={clsx(
                          'text-sm font-bold',
                          result.score >= 0.8 ? 'text-status-success' :
                          result.score >= 0.6 ? 'text-status-warning' :
                          'text-status-error'
                        )}>
                          {(result.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      {/* Benchmark description */}
                      {benchInfo && (
                        <p className="text-xs text-text-muted mb-2">{benchInfo.tooltip}</p>
                      )}
                      <div className="h-2 bg-background-secondary rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            'h-full rounded-full',
                            result.score >= 0.8 ? 'bg-status-success' :
                            result.score >= 0.6 ? 'bg-status-warning' :
                            'bg-status-error'
                          )}
                          style={{ width: `${result.score * 100}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <p className="text-xs text-text-muted">
                          {result.passed} / {result.total} passed
                        </p>
                        {result.duration_seconds && (
                          <p className="text-xs text-text-muted flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {result.duration_seconds.toFixed(1)}s
                          </p>
                        )}
                      </div>

                      {/* Error Analysis */}
                      {hasErrors && (
                        <div className="mt-2 pt-2 border-t border-border-subtle">
                          <p className="text-xs text-text-secondary mb-1.5 font-medium">Error Breakdown:</p>
                          <div className="grid grid-cols-2 gap-1.5 text-xs">
                            {result.errors!.wrong_answer > 0 && (
                              <div className="flex items-center gap-1.5 text-status-error">
                                <XCircle className="w-3 h-3" />
                                <span>Wrong answer: {result.errors!.wrong_answer}</span>
                              </div>
                            )}
                            {result.errors!.syntax_error > 0 && (
                              <div className="flex items-center gap-1.5 text-status-warning">
                                <AlertTriangle className="w-3 h-3" />
                                <span>Syntax error: {result.errors!.syntax_error}</span>
                              </div>
                            )}
                            {result.errors!.runtime_error > 0 && (
                              <div className="flex items-center gap-1.5 text-status-warning">
                                <AlertCircle className="w-3 h-3" />
                                <span>Runtime error: {result.errors!.runtime_error}</span>
                              </div>
                            )}
                            {result.errors!.timeout > 0 && (
                              <div className="flex items-center gap-1.5 text-text-muted">
                                <Timer className="w-3 h-3" />
                                <span>Timeout: {result.errors!.timeout}</span>
                              </div>
                            )}
                            {result.errors!.other > 0 && (
                              <div className="flex items-center gap-1.5 text-text-muted">
                                <AlertCircle className="w-3 h-3" />
                                <span>Other: {result.errors!.other}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )})}
                </div>
              </div>
            )}

            {/* Quick Stats */}
            <div className="card-elevated p-4">
              <h3 className="font-medium text-text-primary mb-3">Benchmark Coverage</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(BENCHMARK_CATEGORIES).map(([key, category]) => {
                  const count = category.benchmarks.filter(b => config.benchmarks.includes(b.id)).length
                  return (
                    <div key={key} className="flex items-center justify-between p-2 bg-background-tertiary rounded">
                      <span className="text-text-secondary">{category.label}</span>
                      <span className="text-text-primary font-medium">{count}/{category.benchmarks.length}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
