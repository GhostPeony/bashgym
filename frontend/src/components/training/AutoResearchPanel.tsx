import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  Zap,
  Play,
  Pause,
  Square,
  FlaskConical,
  TrendingDown,
  Check,
  X,
  Copy,
} from 'lucide-react'
import { useAutoResearchStore, AutoResearchStartConfig, TraceExperimentResult } from '../../stores/autoresearchStore'
import { clsx } from 'clsx'

// Trace research data pipeline parameters
const TRACE_SEARCH_PARAMS = [
  { key: 'min_quality_score', label: 'Min Quality Score', defaultChecked: true },
  { key: 'min_success_rate', label: 'Min Success Rate', defaultChecked: true },
  { key: 'silver_inclusion_ratio', label: 'Silver Inclusion', defaultChecked: true },
  { key: 'include_cognitive', label: 'Cognitive Tags', defaultChecked: false },
  { key: 'include_failed_as_dpo', label: 'Failed as DPO', defaultChecked: false },
  { key: 'time_gap_threshold_minutes', label: 'Time Gap Threshold', defaultChecked: false },
  { key: 'dedup_similarity_threshold', label: 'Dedup Threshold', defaultChecked: false },
  { key: 'max_examples_per_repo', label: 'Max per Repo', defaultChecked: false },
  { key: 'min_steps_per_example', label: 'Min Steps', defaultChecked: false },
  { key: 'max_steps_per_example', label: 'Max Steps', defaultChecked: false },
] as const

// Searchable hyperparameter definitions
const SEARCH_PARAMS = [
  { key: 'learning_rate', label: 'Learning Rate', defaultChecked: true },
  { key: 'lora_r', label: 'LoRA Rank', defaultChecked: true },
  { key: 'lora_alpha', label: 'LoRA Alpha', defaultChecked: true },
  { key: 'lora_dropout', label: 'LoRA Dropout', defaultChecked: false },
  { key: 'warmup_ratio', label: 'Warmup Ratio', defaultChecked: false },
  { key: 'gradient_accumulation_steps', label: 'Grad Accum Steps', defaultChecked: false },
  { key: 'batch_size', label: 'Batch Size', defaultChecked: false },
  { key: 'max_seq_length', label: 'Max Seq Length', defaultChecked: false },
  { key: 'load_in_4bit', label: '4-bit Quantization', defaultChecked: false },
] as const

const STATUS_COLORS: Record<string, string> = {
  idle: 'bg-background-tertiary text-text-muted',
  running: 'bg-status-success text-white',
  paused: 'bg-status-warning text-white',
  completed: 'bg-accent text-white',
  failed: 'bg-status-error text-white',
}

// --- Mini SVG Chart ---

interface MiniChartProps {
  experiments: Array<{
    experimentId: number
    metricValue: number
    bestMetric: number
    improved: boolean
  }>
}

function MiniChart({ experiments }: MiniChartProps) {
  if (experiments.length === 0) return null

  const width = 500
  const height = 160
  const padding = { top: 16, right: 16, bottom: 28, left: 48 }
  const chartW = width - padding.left - padding.right
  const chartH = height - padding.top - padding.bottom

  // Auto-scale Y axis
  const allValues = experiments.map((e) => e.metricValue)
  const minVal = Math.min(...allValues) * 0.95
  const maxVal = Math.max(...allValues) * 1.05
  const yRange = maxVal - minVal || 1

  const xScale = (id: number) =>
    padding.left + ((id - 1) / Math.max(experiments.length - 1, 1)) * chartW
  const yScale = (val: number) =>
    padding.top + (1 - (val - minVal) / yRange) * chartH

  // Build best-metric step line
  const bestLinePts = experiments.map((e) => `${xScale(e.experimentId)},${yScale(e.bestMetric)}`)
  const bestLinePath =
    bestLinePts.length > 1
      ? bestLinePts.reduce((acc, pt, i) => {
          if (i === 0) return `M${pt}`
          const prevX = xScale(experiments[i - 1].experimentId)
          const currX = xScale(experiments[i].experimentId)
          const prevY = yScale(experiments[i - 1].bestMetric)
          // Step function: horizontal then vertical
          return `${acc} L${currX},${prevY} L${pt}`
        }, '')
      : null

  // Y-axis ticks
  const tickCount = 4
  const yTicks = Array.from({ length: tickCount + 1 }, (_, i) =>
    minVal + (yRange * i) / tickCount
  )

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
      {/* Grid lines */}
      {yTicks.map((tick, i) => (
        <g key={i}>
          <line
            x1={padding.left}
            y1={yScale(tick)}
            x2={width - padding.right}
            y2={yScale(tick)}
            stroke="var(--border-subtle)"
            strokeWidth="1"
            strokeDasharray="4,4"
          />
          <text
            x={padding.left - 6}
            y={yScale(tick) + 4}
            textAnchor="end"
            fill="var(--text-muted)"
            fontSize="10"
            fontFamily="JetBrains Mono, monospace"
          >
            {tick.toFixed(3)}
          </text>
        </g>
      ))}

      {/* Best metric step line */}
      {bestLinePath && (
        <path
          d={bestLinePath}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          opacity="0.6"
        />
      )}

      {/* Experiment dots */}
      {experiments.map((e) => (
        <circle
          key={e.experimentId}
          cx={xScale(e.experimentId)}
          cy={yScale(e.metricValue)}
          r="4"
          fill={e.improved ? 'var(--accent)' : 'var(--text-muted)'}
          stroke={e.improved ? 'var(--accent-dark)' : 'var(--border-subtle)'}
          strokeWidth="1.5"
        />
      ))}

      {/* X-axis label */}
      <text
        x={width / 2}
        y={height - 4}
        textAnchor="middle"
        fill="var(--text-muted)"
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
      >
        EXPERIMENT
      </text>
    </svg>
  )
}

// --- Main Panel ---

export function AutoResearchPanel() {
  const {
    status,
    experiments,
    bestMetric,
    bestConfig,
    totalExperiments,
    currentExperiment,
    start,
    stop,
    pause,
    resume,
    reset,
    // Trace research
    activeMode,
    setActiveMode,
    traceStatus,
    traceExperiments,
    traceBestMetric,
    traceBestPipeline,
    traceTotalExperiments,
    traceCurrentExperiment,
    startTraceResearch,
    stopTraceResearch,
    pauseTraceResearch,
    resumeTraceResearch,
    resetTrace,
  } = useAutoResearchStore()

  // Config form state
  const [selectedParams, setSelectedParams] = useState<string[]>(
    SEARCH_PARAMS.filter((p) => p.defaultChecked).map((p) => p.key)
  )
  const [maxExperiments, setMaxExperiments] = useState(50)
  const [mutationRate, setMutationRate] = useState(0.3)
  const [mutationScale, setMutationScale] = useState(0.2)
  const [trainSteps, setTrainSteps] = useState(100)

  // Trace research config state
  const [traceSelectedParams, setTraceSelectedParams] = useState<string[]>(
    TRACE_SEARCH_PARAMS.filter((p) => p.defaultChecked).map((p) => p.key)
  )
  const [traceMaxExperiments, setTraceMaxExperiments] = useState(30)
  const [traceMutationRate, setTraceMutationRate] = useState(0.3)
  const [traceMutationScale, setTraceMutationScale] = useState(0.2)

  const toggleTraceParam = useCallback((key: string) => {
    setTraceSelectedParams((prev) =>
      prev.includes(key) ? prev.filter((p) => p !== key) : [...prev, key]
    )
  }, [])

  const handleStartTrace = useCallback(() => {
    startTraceResearch({
      searchParams: traceSelectedParams,
      maxExperiments: traceMaxExperiments,
      mutationRate: traceMutationRate,
      mutationScale: traceMutationScale,
    })
  }, [traceSelectedParams, traceMaxExperiments, traceMutationRate, traceMutationScale, startTraceResearch])

  // Experiment log auto-scroll
  const logEndRef = useRef<HTMLDivElement>(null)
  const logContainerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [experiments, autoScroll])

  const handleLogScroll = useCallback(() => {
    if (!logContainerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 50)
  }, [])

  const toggleParam = useCallback((key: string) => {
    setSelectedParams((prev) =>
      prev.includes(key) ? prev.filter((p) => p !== key) : [...prev, key]
    )
  }, [])

  const handleStart = useCallback(() => {
    const config: AutoResearchStartConfig = {
      searchParams: selectedParams,
      maxExperiments,
      trainSteps,
      mutationRate,
      mutationScale,
      // Base training defaults
      baseModel: 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
      learningRate: 2e-4,
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.05,
      warmupRatio: 0.1,
      gradientAccumulationSteps: 4,
      batchSize: 2,
      maxSeqLength: 2048,
      load4Bit: true,
    }
    start(config)
  }, [selectedParams, maxExperiments, trainSteps, mutationRate, mutationScale, start])

  // Mode-aware status helpers
  const isHyperparam = activeMode === 'hyperparam'
  const activeStatus = isHyperparam ? status : traceStatus
  const isIdle = activeStatus === 'idle'
  const isRunning = activeStatus === 'running'
  const isPaused = activeStatus === 'paused'
  const hasResults = isHyperparam ? experiments.length > 0 : traceExperiments.length > 0
  const activeCurrent = isHyperparam ? currentExperiment : traceCurrentExperiment
  const activeTotal = isHyperparam ? totalExperiments : traceTotalExperiments
  const progressPct =
    activeTotal > 0 ? (activeCurrent / activeTotal) * 100 : 0
  const activeBestMetric = isHyperparam ? bestMetric : traceBestMetric

  // Compute starting config for comparison table
  const startingConfig = useMemo<Record<string, number | boolean>>(() => ({
    learning_rate: 2e-4,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    warmup_ratio: 0.1,
    gradient_accumulation_steps: 4,
    batch_size: 2,
    max_seq_length: 2048,
    load_in_4bit: true,
  }), [])

  return (
    <div className="card p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <FlaskConical className="w-5 h-5 text-accent" />
          <h2 className="font-brand text-xl text-text-primary">AutoResearch</h2>
          <span
            className={clsx(
              'inline-flex items-center font-mono text-[0.65rem] font-semibold uppercase tracking-[0.12em] px-2 py-0.5 border-brutal rounded-brutal',
              STATUS_COLORS[activeStatus] || STATUS_COLORS.idle
            )}
          >
            {activeStatus}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {isIdle && (
            <button
              onClick={isHyperparam ? handleStart : handleStartTrace}
              disabled={isHyperparam ? selectedParams.length === 0 : traceSelectedParams.length === 0}
              className={clsx(
                'btn-primary flex items-center gap-2',
                (isHyperparam ? selectedParams.length === 0 : traceSelectedParams.length === 0) && 'opacity-50 cursor-not-allowed'
              )}
            >
              <Zap className="w-4 h-4" />
              Start Search
            </button>
          )}
          {isRunning && (
            <>
              <button
                onClick={isHyperparam ? pause : pauseTraceResearch}
                className="btn-icon flex items-center justify-center"
                title="Pause"
              >
                <Pause className="w-4 h-4" />
              </button>
              <button
                onClick={isHyperparam ? stop : stopTraceResearch}
                className="btn-icon flex items-center justify-center text-status-error"
                title="Stop"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
          {isPaused && (
            <>
              <button
                onClick={isHyperparam ? resume : resumeTraceResearch}
                className="btn-icon flex items-center justify-center"
                title="Resume"
              >
                <Play className="w-4 h-4" />
              </button>
              <button
                onClick={isHyperparam ? stop : stopTraceResearch}
                className="btn-icon flex items-center justify-center text-status-error"
                title="Stop"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
          {(activeStatus === 'completed' || activeStatus === 'failed') && (
            <button onClick={isHyperparam ? reset : resetTrace} className="btn-secondary flex items-center gap-2">
              <FlaskConical className="w-4 h-4" />
              New Search
            </button>
          )}
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="flex items-center gap-1 mb-4 p-1 bg-background-secondary border-brutal border-border rounded-brutal w-fit">
        <button
          onClick={() => setActiveMode('hyperparam')}
          className={clsx(
            'px-4 py-1.5 font-mono text-xs uppercase tracking-widest transition-colors rounded-brutal',
            isHyperparam
              ? 'bg-accent text-white border-brutal border-accent-dark'
              : 'text-text-muted hover:text-text-primary'
          )}
        >
          Hyperparams
        </button>
        <button
          onClick={() => setActiveMode('trace')}
          className={clsx(
            'px-4 py-1.5 font-mono text-xs uppercase tracking-widest transition-colors rounded-brutal',
            !isHyperparam
              ? 'bg-accent text-white border-brutal border-accent-dark'
              : 'text-text-muted hover:text-text-primary'
          )}
        >
          Trace Mining
        </button>
      </div>

      <div className="section-divider mb-4" />

      {/* Config Section - shown when idle */}
      {isIdle && !hasResults && isHyperparam && (
        <div className="space-y-4">
          {/* Search params checkboxes */}
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-2">
              Parameters to Search
            </span>
            <div className="grid grid-cols-3 gap-2">
              {SEARCH_PARAMS.map((param) => (
                <label
                  key={param.key}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-2 cursor-pointer border rounded-brutal transition-colors',
                    selectedParams.includes(param.key)
                      ? 'border-accent bg-accent-light/30'
                      : 'border-border-subtle bg-background-secondary'
                  )}
                >
                  <input
                    type="checkbox"
                    checked={selectedParams.includes(param.key)}
                    onChange={() => toggleParam(param.key)}
                    className="accent-accent"
                  />
                  <span className="text-sm text-text-primary">{param.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Numeric controls */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
                Max Experiments
              </label>
              <input
                type="number"
                value={maxExperiments}
                onChange={(e) => setMaxExperiments(Number(e.target.value))}
                min={5}
                max={500}
                className="input w-full"
              />
            </div>
            <div>
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
                Train Steps
              </label>
              <input
                type="number"
                value={trainSteps}
                onChange={(e) => setTrainSteps(Number(e.target.value))}
                min={10}
                max={5000}
                className="input w-full"
              />
            </div>
            <div>
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
                Mutation Scale
              </label>
              <input
                type="number"
                value={mutationScale}
                onChange={(e) => setMutationScale(Number(e.target.value))}
                min={0.05}
                max={1.0}
                step={0.05}
                className="input w-full"
              />
            </div>
          </div>

          {/* Mutation rate slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Mutation Rate
              </label>
              <span className="font-mono text-xs text-text-primary">
                {mutationRate.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              value={mutationRate}
              onChange={(e) => setMutationRate(Number(e.target.value))}
              min={0.1}
              max={0.5}
              step={0.05}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-[0.6rem] font-mono text-text-muted mt-0.5">
              <span>0.1 (Conservative)</span>
              <span>0.5 (Aggressive)</span>
            </div>
          </div>
        </div>
      )}

      {/* Trace Research Config - shown when idle in trace mode */}
      {isIdle && !hasResults && !isHyperparam && (
        <div className="space-y-4">
          <p className="text-sm text-text-secondary">
            Optimize your data curation pipeline by experimenting with quality thresholds,
            segmentation parameters, and example selection criteria.
          </p>
          {/* Data pipeline params checkboxes */}
          <div>
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-2">
              Pipeline Parameters to Search
            </span>
            <div className="grid grid-cols-2 gap-2">
              {TRACE_SEARCH_PARAMS.map((param) => (
                <label
                  key={param.key}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-2 cursor-pointer border rounded-brutal transition-colors',
                    traceSelectedParams.includes(param.key)
                      ? 'border-accent bg-accent-light/30'
                      : 'border-border-subtle bg-background-secondary'
                  )}
                >
                  <input
                    type="checkbox"
                    checked={traceSelectedParams.includes(param.key)}
                    onChange={() => toggleTraceParam(param.key)}
                    className="accent-accent"
                  />
                  <span className="text-sm text-text-primary">{param.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Numeric controls */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
                Max Experiments
              </label>
              <input
                type="number"
                value={traceMaxExperiments}
                onChange={(e) => setTraceMaxExperiments(Number(e.target.value))}
                min={5}
                max={200}
                className="input w-full"
              />
            </div>
            <div>
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
                Mutation Scale
              </label>
              <input
                type="number"
                value={traceMutationScale}
                onChange={(e) => setTraceMutationScale(Number(e.target.value))}
                min={0.05}
                max={1.0}
                step={0.05}
                className="input w-full"
              />
            </div>
          </div>

          {/* Mutation rate slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Mutation Rate
              </label>
              <span className="font-mono text-xs text-text-primary">
                {traceMutationRate.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              value={traceMutationRate}
              onChange={(e) => setTraceMutationRate(Number(e.target.value))}
              min={0.1}
              max={0.5}
              step={0.05}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-[0.6rem] font-mono text-text-muted mt-0.5">
              <span>0.1 (Conservative)</span>
              <span>0.5 (Aggressive)</span>
            </div>
          </div>
        </div>
      )}

      {/* Live Results Section */}
      {(isRunning || isPaused || activeStatus === 'completed' || activeStatus === 'failed') && (
        <div className="space-y-4">
          {/* Progress + Best metric row */}
          <div className="grid grid-cols-12 gap-4">
            {/* Progress */}
            <div className="col-span-8">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
                  Progress
                </span>
                <span className="font-mono text-xs text-text-primary">
                  {activeCurrent} / {activeTotal}
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progressPct}%` }} />
              </div>
            </div>

            {/* Best metric */}
            <div className="col-span-4 card p-3 flex flex-col items-center justify-center">
              <span className="font-mono text-[0.6rem] uppercase tracking-widest text-text-muted">
                {isHyperparam ? 'Best Loss' : 'Best Quality'}
              </span>
              <div className="flex items-center gap-2 mt-1">
                <TrendingDown className="w-4 h-4 text-accent" />
                <span className="font-brand text-2xl text-text-primary">
                  {activeBestMetric != null ? activeBestMetric.toFixed(4) : '\u2014'}
                </span>
              </div>
            </div>
          </div>

          {/* Trace research extra stats */}
          {!isHyperparam && traceExperiments.length > 0 && (
            <div className="grid grid-cols-3 gap-3">
              <div className="card p-3 text-center">
                <span className="font-mono text-[0.6rem] uppercase tracking-widest text-text-muted block">Examples</span>
                <span className="font-brand text-xl text-text-primary">
                  {traceExperiments[traceExperiments.length - 1].examplesGenerated}
                </span>
              </div>
              <div className="card p-3 text-center">
                <span className="font-mono text-[0.6rem] uppercase tracking-widest text-text-muted block">Repos</span>
                <span className="font-brand text-xl text-text-primary">
                  {traceExperiments[traceExperiments.length - 1].uniqueRepos}
                </span>
              </div>
              <div className="card p-3 text-center">
                <span className="font-mono text-[0.6rem] uppercase tracking-widest text-text-muted block">Avg Length</span>
                <span className="font-brand text-xl text-text-primary">
                  {traceExperiments[traceExperiments.length - 1].avgExampleLength.toFixed(0)}
                </span>
              </div>
            </div>
          )}

          {/* Mini loss chart */}
          {hasResults && (
            <div className="card p-3">
              <span className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-2">
                {isHyperparam ? 'Experiment Loss' : 'Pipeline Quality Score'}
              </span>
              <div className="h-40">
                <MiniChart experiments={isHyperparam ? experiments : traceExperiments} />
              </div>
            </div>
          )}

          {/* Experiment log */}
          {hasResults && (
            <div className="terminal-chrome relative">
              <div className="terminal-header">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
                  Experiment Log
                </span>
                <span className="tag ml-2">
                  <span>{isHyperparam ? experiments.length : traceExperiments.length}</span>
                </span>
              </div>
              <div
                ref={logContainerRef}
                onScroll={handleLogScroll}
                className="bg-background-terminal overflow-auto font-mono text-xs"
                style={{ maxHeight: 200 }}
              >
                <div className="p-2 space-y-0.5">
                  {(isHyperparam ? experiments : traceExperiments).map((exp) => (
                    <div
                      key={exp.experimentId}
                      className="flex items-center gap-3 py-1 px-2"
                    >
                      <span className="text-text-muted w-8 text-right shrink-0">
                        #{exp.experimentId}
                      </span>
                      {exp.improved ? (
                        <Check className="w-3.5 h-3.5 text-status-success shrink-0" />
                      ) : (
                        <X className="w-3.5 h-3.5 text-text-muted shrink-0" />
                      )}
                      <span
                        className={clsx(
                          'w-16 text-right shrink-0',
                          exp.improved ? 'text-accent' : 'text-text-secondary'
                        )}
                      >
                        {exp.metricValue.toFixed(4)}
                      </span>
                      {!isHyperparam && 'examplesGenerated' in exp && (
                        <span className="text-text-muted shrink-0 w-12 text-right">
                          {(exp as TraceExperimentResult).examplesGenerated}ex
                        </span>
                      )}
                      <span className="text-text-muted truncate flex-1">
                        {formatConfigDiff(exp.configSnapshot)}
                      </span>
                      <span className="text-text-muted shrink-0">
                        {exp.durationSeconds}s
                      </span>
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </div>
              {!autoScroll && (isHyperparam ? experiments : traceExperiments).length > 6 && (
                <div
                  className="absolute bottom-2 right-4 tag cursor-pointer"
                  onClick={() => {
                    setAutoScroll(true)
                    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
                  }}
                >
                  <span>Scroll to bottom</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Best config comparison table */}
      {((isHyperparam && bestConfig) || (!isHyperparam && traceBestPipeline)) && hasResults && (
        <>
          <div className="section-divider my-4" />
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Best Configuration Found
              </span>
              <button
                onClick={() => {
                  const config = isHyperparam ? bestConfig : traceBestPipeline
                  navigator.clipboard.writeText(JSON.stringify(config, null, 2))
                }}
                className="btn-ghost p-1.5 flex items-center gap-1"
                title="Copy best config"
              >
                <Copy className="w-3.5 h-3.5 text-text-muted" />
                <span className="text-[0.6rem] text-text-muted uppercase tracking-widest">Copy</span>
              </button>
            </div>
            <div className="overflow-hidden border-brutal border-border rounded-brutal">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-background-secondary">
                    <th className="text-left px-3 py-2 font-mono text-xs uppercase tracking-widest text-text-muted border-b border-border">
                      Parameter
                    </th>
                    <th className="text-right px-3 py-2 font-mono text-xs uppercase tracking-widest text-text-muted border-b border-border">
                      Starting
                    </th>
                    <th className="text-right px-3 py-2 font-mono text-xs uppercase tracking-widest text-text-muted border-b border-border">
                      Best
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(isHyperparam ? (bestConfig || {}) : (traceBestPipeline || {})).map((key) => {
                    const activeConfig = isHyperparam ? bestConfig : traceBestPipeline
                    const startVal = startingConfig[key]
                    const bestVal = activeConfig?.[key]
                    const changed = startVal !== bestVal
                    return (
                      <tr
                        key={key}
                        className={clsx(
                          'border-b border-border-subtle',
                          changed && 'bg-accent-light/20'
                        )}
                      >
                        <td className="px-3 py-1.5 font-mono text-xs text-text-secondary">
                          {key}
                        </td>
                        <td className="px-3 py-1.5 text-right font-mono text-xs text-text-muted">
                          {formatValue(startVal)}
                        </td>
                        <td
                          className={clsx(
                            'px-3 py-1.5 text-right font-mono text-xs',
                            changed ? 'text-accent font-semibold' : 'text-text-primary'
                          )}
                        >
                          {formatValue(bestVal)}
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
  )
}

// Format a config snapshot to show only non-default changes
function formatConfigDiff(config: Record<string, number | boolean | string>): string {
  const parts: string[] = []
  for (const [key, val] of Object.entries(config)) {
    const shortKey = key.replace('gradient_accumulation_steps', 'grad_accum')
      .replace('learning_rate', 'lr')
      .replace('max_seq_length', 'seq_len')
      .replace('load_in_4bit', '4bit')
      .replace('warmup_ratio', 'warmup')
      .replace('lora_dropout', 'lora_do')
      .replace('lora_alpha', 'lora_a')
      .replace('lora_r', 'lora_r')
      .replace('batch_size', 'bs')
      .replace('min_quality_score', 'qual')
      .replace('min_success_rate', 'succ')
      .replace('silver_inclusion_ratio', 'silver')
      .replace('include_cognitive', 'cog')
      .replace('include_failed_as_dpo', 'dpo_fail')
      .replace('time_gap_threshold_minutes', 'gap')
      .replace('dedup_similarity_threshold', 'dedup')
      .replace('max_examples_per_repo', 'max_repo')
      .replace('min_steps_per_example', 'min_steps')
      .replace('max_steps_per_example', 'max_steps')
    parts.push(`${shortKey}=${formatValue(val)}`)
  }
  return parts.join(' ')
}

function formatValue(val: number | boolean | string | undefined): string {
  if (val === undefined) return '-'
  if (typeof val === 'boolean') return val ? 'true' : 'false'
  if (typeof val === 'string') return val
  if (val < 0.001) return val.toExponential(1)
  if (Number.isInteger(val)) return String(val)
  return val.toFixed(4)
}
