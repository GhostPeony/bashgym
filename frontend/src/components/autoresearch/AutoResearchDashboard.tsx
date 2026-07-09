import { useState } from 'react'
import { Database, FileJson, Loader2, RefreshCw, Save, Target, Zap } from 'lucide-react'
import { AutoResearchPanel } from '../training/AutoResearchPanel'
import { ResearchNewsPanel } from './ResearchNewsPanel'
import { useAutoResearchStore } from '../../stores/autoresearchStore'
import {
  autoresearchApi,
  DataRecipeProposalResponse,
  EnvironmentRecipeProposalResponse,
} from '../../services/api'

function formatPercent(value: number | undefined | null) {
  if (value == null || Number.isNaN(value)) return '-'
  return `${Math.round(value * 100)}%`
}

function formatMetric(value: number | undefined | null) {
  if (value == null || Number.isNaN(value)) return '-'
  return value.toFixed(3)
}

function DataRecipeProposalPanel() {
  const [goal, setGoal] = useState('dpo')
  const [sourceText, setSourceText] = useState('ultrafeedback_binarized, helpsteer2')
  const [domain, setDomain] = useState('')
  const [outputPath, setOutputPath] = useState('data/data_recipe_proposals/latest.json')
  const [maxExperiments, setMaxExperiments] = useState(8)
  const [sampleSize, setSampleSize] = useState(1000)
  const [qualityThreshold, setQualityThreshold] = useState(0.7)
  const [includeEvalOnly, setIncludeEvalOnly] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [result, setResult] = useState<DataRecipeProposalResponse | null>(null)

  const sourceIds = sourceText
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean)

  const proposal = result?.proposal
  const sourcePreview = proposal?.sources.slice(0, 6) || []

  const handlePropose = async () => {
    setIsRunning(true)
    setError(null)
    setStatusMessage(null)
    try {
      const response = await autoresearchApi.proposeDataRecipe({
        goal,
        sourceIds,
        domain: domain.trim() || undefined,
        includeEvalOnly,
        maxExperiments,
        sampleSize,
        qualityThreshold,
        outputPath: outputPath.trim() || undefined,
      })
      if (!response.ok || !response.data) {
        setError(response.error || 'Proposal failed')
        setResult(null)
        return
      }
      setResult(response.data)
      setStatusMessage(`Completed ${response.data.experiments.length} experiments`)
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setIsRunning(false)
    }
  }

  const handleRefreshStatus = async () => {
    setError(null)
    const response = await autoresearchApi.getDataRecipeStatus()
    if (!response.ok || !response.data) {
      setError(response.error || 'Could not fetch data-recipe status')
      return
    }
    setStatusMessage(`${response.data.status}: ${response.data.completed_experiments}/${response.data.total_experiments}`)
    if (response.data.proposal) {
      setResult({
        status: response.data.status,
        goal: response.data.goal || response.data.proposal.goal,
        source_count: response.data.source_count || response.data.proposal.sources.length,
        excluded_sources: response.data.excluded_sources || [],
        best_metric: response.data.best_metric,
        proposal: response.data.proposal,
        experiments: response.data.experiments,
        output_path: response.data.output_path || null,
      })
    }
  }

  const handleExportLatest = async () => {
    setIsExporting(true)
    setError(null)
    try {
      const response = await autoresearchApi.exportDataRecipe(outputPath.trim())
      if (!response.ok || !response.data) {
        setError(response.error || 'Export failed')
        return
      }
      setStatusMessage(`Exported ${response.data.output_path}`)
      setResult((prev) =>
        prev
          ? { ...prev, proposal: response.data!.proposal, output_path: response.data!.output_path }
          : prev
      )
    } catch (err) {
      setError(String(err))
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="card p-5 mt-6">
      <div className="flex items-center justify-between gap-4 mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <Database className="w-5 h-5 text-accent shrink-0" />
          <h2 className="font-brand text-xl text-text-primary truncate">
            Data Recipe Proposals
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleRefreshStatus}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Status
          </button>
          <button
            onClick={handlePropose}
            disabled={isRunning}
            className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileJson className="w-4 h-4" />}
            Propose
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Goal
          </label>
          <select
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            className="input w-full"
          >
            <option value="sft">SFT</option>
            <option value="dpo">DPO</option>
            <option value="reward_model">Reward Model</option>
            <option value="process_reward">Process Reward</option>
            <option value="terminal_rl">Terminal RL</option>
            <option value="evaluation">Evaluation</option>
          </select>
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Source IDs
          </label>
          <input
            value={sourceText}
            onChange={(event) => setSourceText(event.target.value)}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Domain Filter
          </label>
          <input
            value={domain}
            onChange={(event) => setDomain(event.target.value)}
            placeholder="optional"
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Output JSON
          </label>
          <input
            value={outputPath}
            onChange={(event) => setOutputPath(event.target.value)}
            className="input w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Experiments
          </label>
          <input
            type="number"
            value={maxExperiments}
            min={1}
            max={200}
            onChange={(event) => setMaxExperiments(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Sample Size
          </label>
          <input
            type="number"
            value={sampleSize}
            min={1}
            max={1000000}
            onChange={(event) => setSampleSize(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Quality Floor
          </label>
          <input
            type="number"
            value={qualityThreshold}
            min={0}
            max={1}
            step={0.05}
            onChange={(event) => setQualityThreshold(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <label className="flex items-center gap-2 border-brutal border-border bg-background-secondary px-3 py-2 mt-5 cursor-pointer">
          <input
            type="checkbox"
            checked={includeEvalOnly}
            onChange={(event) => setIncludeEvalOnly(event.target.checked)}
            className="accent-accent"
          />
          <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">
            Include Eval-Only
          </span>
        </label>
      </div>

      {(error || statusMessage) && (
        <div
          className={`mt-4 border-brutal px-3 py-2 text-sm ${
            error
              ? 'border-status-error bg-status-error/10 text-status-error'
              : 'border-border bg-background-secondary text-text-secondary'
          }`}
        >
          {error || statusMessage}
        </div>
      )}

      {proposal && (
        <div className="mt-5 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Metric
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {formatMetric(result?.best_metric)}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Sources
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {proposal.sources.length}/{result?.source_count ?? proposal.sources.length}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Sample Size
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {proposal.data_designer.sample_size}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Excluded
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {result?.excluded_sources.length ?? 0}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="border-brutal border-border bg-background-secondary p-3 min-w-0">
              <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                Source Weights
              </div>
              <div className="space-y-1">
                {sourcePreview.map((source) => (
                  <div key={source.id} className="font-mono text-xs text-text-secondary flex justify-between gap-3">
                    <span className="truncate">{source.id}</span>
                    <span>{source.weight.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3 min-w-0">
              <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                Export
              </div>
              <div className="font-mono text-xs text-text-secondary break-all mb-3">
                {result?.output_path || '-'}
              </div>
              <button
                onClick={handleExportLatest}
                disabled={isExporting || !outputPath.trim()}
                className="btn-secondary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                Export Latest
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function EnvironmentRecipeProposalPanel() {
  const [environmentPath, setEnvironmentPath] = useState('')
  const [outputPath, setOutputPath] = useState('data/environment_recipe_proposals/latest.json')
  const [maxExperiments, setMaxExperiments] = useState(8)
  const [sampleSize, setSampleSize] = useState(64)
  const [passAt1Target, setPassAt1Target] = useState(0.35)
  const [seed, setSeed] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EnvironmentRecipeProposalResponse | null>(null)

  const proposal = result?.proposal
  const selectedPreview = proposal?.selected_environment_ids.slice(0, 10) || []

  const handlePropose = async () => {
    const trimmedPath = environmentPath.trim()
    if (!trimmedPath) {
      setError('Environment source required')
      return
    }

    setIsRunning(true)
    setError(null)
    try {
      const response = await autoresearchApi.proposeEnvironmentRecipe({
        environmentPath: trimmedPath,
        outputPath: outputPath.trim() || undefined,
        maxExperiments,
        sampleSize,
        passAt1Target,
        seed,
      })
      if (!response.ok || !response.data) {
        setError(response.error || 'Proposal failed')
        setResult(null)
        return
      }
      setResult(response.data)
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="card p-5 mt-6">
      <div className="flex items-center justify-between gap-4 mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <Target className="w-5 h-5 text-accent shrink-0" />
          <h2 className="font-brand text-xl text-text-primary truncate">
            Environment Recipe Proposals
          </h2>
        </div>
        <button
          onClick={handlePropose}
          disabled={isRunning || !environmentPath.trim()}
          className="btn-primary flex items-center gap-2 shrink-0 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileJson className="w-4 h-4" />}
          Propose
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Source JSON/JSONL
          </label>
          <input
            value={environmentPath}
            onChange={(event) => setEnvironmentPath(event.target.value)}
            placeholder="data/environments/tmax.jsonl"
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Output JSON
          </label>
          <input
            value={outputPath}
            onChange={(event) => setOutputPath(event.target.value)}
            className="input w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Experiments
          </label>
          <input
            type="number"
            value={maxExperiments}
            min={1}
            max={200}
            onChange={(event) => setMaxExperiments(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Sample Size
          </label>
          <input
            type="number"
            value={sampleSize}
            min={1}
            max={10000}
            onChange={(event) => setSampleSize(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Pass@1 Target
          </label>
          <input
            type="number"
            value={passAt1Target}
            min={0}
            max={1}
            step={0.05}
            onChange={(event) => setPassAt1Target(Number(event.target.value))}
            className="input w-full"
          />
        </div>
        <div>
          <label className="font-mono text-xs uppercase tracking-widest text-text-muted block mb-1">
            Seed
          </label>
          <input
            type="number"
            value={seed}
            min={0}
            onChange={(event) => setSeed(Number(event.target.value))}
            className="input w-full"
          />
        </div>
      </div>

      {error && (
        <div className="mt-4 border-brutal border-status-error bg-status-error/10 text-status-error px-3 py-2 text-sm">
          {error}
        </div>
      )}

      {proposal && (
        <div className="mt-5 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Metric
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {proposal.metric.toFixed(3)}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Selected
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {proposal.selected_count}/{proposal.source_count}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Domain Balance
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {formatPercent(proposal.mix_report.axis_balance.domain)}
              </span>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3">
              <span className="font-mono text-[0.65rem] uppercase tracking-widest text-text-muted block">
                Mean Pass@1
              </span>
              <span className="font-brand text-2xl text-text-primary">
                {formatPercent(proposal.mix_report.mean_pass_rates?.['pass@1'])}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="border-brutal border-border bg-background-secondary p-3 min-w-0">
              <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                Selected IDs
              </div>
              <pre className="font-mono text-xs text-text-secondary whitespace-pre-wrap break-all">
                {selectedPreview.join('\n')}
                {proposal.selected_environment_ids.length > selectedPreview.length
                  ? `\n+${proposal.selected_environment_ids.length - selectedPreview.length} more`
                  : ''}
              </pre>
            </div>
            <div className="border-brutal border-border bg-background-secondary p-3 min-w-0">
              <div className="font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                Export
              </div>
              <div className="font-mono text-xs text-text-secondary break-all">
                {result.output_path || '-'}
              </div>
              {result.validation_errors.length > 0 && (
                <div className="mt-2 text-xs text-status-warning font-mono">
                  {result.validation_errors.length} validation warnings
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export function AutoResearchDashboard() {
  const { status, traceStatus, activeMode } = useAutoResearchStore()

  const activeStatus = activeMode === 'hyperparam' ? status : traceStatus

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-[1200px] mx-auto px-6 py-8">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-accent" />
              <h1 className="font-serif text-2xl">AutoResearch</h1>
              {activeStatus !== 'idle' && (
                <span className={`px-2 py-0.5 text-xs font-mono uppercase tracking-wider rounded-sm ${
                  activeStatus === 'running' ? 'bg-status-success text-white' :
                  activeStatus === 'paused' ? 'bg-status-warning text-white' :
                  activeStatus === 'completed' ? 'bg-accent text-white' :
                  activeStatus === 'failed' ? 'bg-status-error text-white' :
                  'bg-background-tertiary text-text-muted'
                }`}>
                  {activeStatus}
                </span>
              )}
            </div>
            <p className="text-text-muted text-sm mt-1">
              Automated hyperparameter search and trace data optimization
            </p>
          </div>
        </div>

        {/* Main Content */}
        <AutoResearchPanel />

        <DataRecipeProposalPanel />

        <EnvironmentRecipeProposalPanel />

        {/* Research-grounded news feed + advice */}
        <ResearchNewsPanel />
      </div>
    </div>
  )
}
