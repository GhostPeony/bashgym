import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Shield,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  GitCompare,
  Terminal,
  Copy,
  TrendingDown,
  AlertCircle,
  FileJson
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  evalAdvancedApi,
  type ExternalBenchmarkIngestResponse,
  type HeldoutEnvironmentEvidence,
  type HeldoutJobResponse,
  type HeldoutReport,
  type NormalizedBenchmarkResult
} from '../../services/api'
import { useSessionResource } from '../../stores/sessionResource'
import { registeredModelsResource } from '../../stores/factoryResources'
import { ModelSelect } from '../common/ModelSelect'

const METRICS = [
  { id: 'exact_match', name: 'Exact match', hint: 'Tool name + every argument identical' },
  { id: 'name_match', name: 'Tool name', hint: 'Right tool, arguments ignored' },
  { id: 'arg_f1', name: 'Argument F1', hint: 'Per-argument overlap' },
  { id: 'soft', name: 'Soft (SERA)', hint: 'Graded partial credit for near-misses' }
]

interface EndpointForm {
  provider: string
  base_url: string
  model: string
}

const EMPTY_ENDPOINT: EndpointForm = { provider: '', base_url: '', model: '' }

type EvidenceFieldId =
  | 'passk'
  | 'holdout_gate'
  | 'holdout_comparison'
  | 'spurious_reward_control'
  | 'external_benchmarks'
  | 'world_model_quality'
  | 'learned_reward_evidence'

const EVIDENCE_FIELDS: Array<{ id: EvidenceFieldId; label: string; placeholder: string }> = [
  {
    id: 'holdout_gate',
    label: 'Holdout gate',
    placeholder: '{"result":{"gate":{"ship":true,"reasons":[]}}}'
  },
  {
    id: 'holdout_comparison',
    label: 'Holdout comparison',
    placeholder: '{"result":{"gate":{"ship":true,"reasons":[]}}}'
  },
  {
    id: 'spurious_reward_control',
    label: 'Spurious reward control',
    placeholder: '{"result":{"gate":{"ship":true,"reasons":[]}}}'
  },
  {
    id: 'passk',
    label: 'Pass@k report',
    placeholder: '{"pass_at_k":{"pass@1":0.5}}'
  },
  {
    id: 'external_benchmarks',
    label: 'External benchmarks',
    placeholder: '{"report":{"scores":{"harbor_terminal_bench":0.67},"failures":[]}}'
  },
  {
    id: 'world_model_quality',
    label: 'World-model quality',
    placeholder: '{"metrics":{"echo_loss":{"first":1.2,"last":0.8},"rwml_pass_rate":0.72}}'
  },
  {
    id: 'learned_reward_evidence',
    label: 'Learned reward',
    placeholder: '{"metrics":{"heldout_pair_accuracy":0.82,"calibration_error":0.08},"findings":[]}'
  }
]

const EMPTY_EVIDENCE_JSON: Record<EvidenceFieldId, string> = {
  passk: '',
  holdout_gate: '',
  holdout_comparison: '',
  spurious_reward_control: '',
  external_benchmarks: '',
  world_model_quality: '',
  learned_reward_evidence: ''
}

interface ParsedEvidence {
  label: string
  value?: Record<string, unknown>
  error?: string
}

function pct(x: number | undefined): string {
  return x === undefined ? '—' : `${(x * 100).toFixed(1)}%`
}

const BENCHMARK_METRIC_SKIP = new Set(['score', 'accuracy', 'overall_accuracy', 'resolution_rate'])

function metricValue(value: number): string {
  if (!Number.isFinite(value)) return 'n/a'
  if (Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(2)
}

function benchmarkMetricPreview(result: NormalizedBenchmarkResult): string[] {
  return Object.entries(result.metrics || {})
    .filter(([key]) => !BENCHMARK_METRIC_SKIP.has(key))
    .slice(0, 8)
    .map(([key, value]) => `${key} ${metricValue(value)}`)
}

function parseEvidenceObject(label: string, text: string): ParsedEvidence {
  const trimmed = text.trim()
  if (trimmed === '') return { label }
  try {
    const parsed = JSON.parse(trimmed) as unknown
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { label, error: `${label} must be a JSON object` }
    }
    return { label, value: parsed as Record<string, unknown> }
  } catch (e) {
    return { label, error: `${label}: ${e instanceof Error ? e.message : String(e)}` }
  }
}

function EndpointFields({
  label,
  value,
  onChange
}: {
  label: string
  value: EndpointForm
  onChange: (v: EndpointForm) => void
}) {
  return (
    <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary space-y-2">
      <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">{label}</p>
      <input
        className="input text-sm w-full"
        placeholder="Connected provider (e.g. vllm, together) — optional"
        value={value.provider}
        onChange={(e) => onChange({ ...value, provider: e.target.value })}
      />
      <input
        className="input text-sm w-full"
        placeholder="Base URL (e.g. http://localhost:8000/v1)"
        value={value.base_url}
        onChange={(e) => onChange({ ...value, base_url: e.target.value })}
      />
      <ModelSelect
        className="input text-sm w-full"
        placeholder="Select endpoint model..."
        value={value.model}
        onChange={(model) => onChange({ ...value, model })}
      />
    </div>
  )
}

function EnvironmentEvidencePanel({
  enabled,
  required,
  values,
  parsed,
  sectionCount,
  onEnabledChange,
  onRequiredChange,
  onValueChange
}: {
  enabled: boolean
  required: boolean
  values: Record<EvidenceFieldId, string>
  parsed: Record<EvidenceFieldId, ParsedEvidence>
  sectionCount: number
  onEnabledChange: (enabled: boolean) => void
  onRequiredChange: (required: boolean) => void
  onValueChange: (field: EvidenceFieldId, value: string) => void
}) {
  const errors = useMemo(
    () =>
      EVIDENCE_FIELDS.map((field) => parsed[field.id].error).filter(
        (error): error is string => error !== undefined
      ),
    [parsed]
  )
  const missingRequired = enabled && required && sectionCount === 0

  return (
    <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <FileJson className="w-4 h-4 text-accent" />
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">
              Release evidence
            </p>
            <p className="text-xs text-text-muted">
              {sectionCount} JSON object{sectionCount === 1 ? '' : 's'} attached
            </p>
          </div>
        </div>
        <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onEnabledChange(e.target.checked)}
          />
          Include
        </label>
      </div>

      {enabled ? (
        <div className="space-y-3">
          <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-text-secondary">
            <input
              type="checkbox"
              checked={required}
              onChange={(e) => onRequiredChange(e.target.checked)}
            />
            Required for ship
          </label>

          <div className="grid grid-cols-2 gap-3">
            {EVIDENCE_FIELDS.map((field) => (
              <div key={field.id} className="space-y-1">
                <label className="block font-mono text-xs uppercase tracking-widest text-text-muted">
                  {field.label}
                </label>
                <textarea
                  className={clsx(
                    'input text-xs w-full min-h-[88px] font-mono',
                    parsed[field.id].error ? 'border-status-error' : ''
                  )}
                  spellCheck={false}
                  autoCapitalize="off"
                  autoCorrect="off"
                  placeholder={field.placeholder}
                  value={values[field.id]}
                  onChange={(e) => onValueChange(field.id, e.target.value)}
                />
              </div>
            ))}
          </div>

          {missingRequired ? (
            <p className="font-mono text-xs text-status-error flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5" /> Attach at least one release evidence object.
            </p>
          ) : null}
          {errors.length > 0 ? (
            <div className="space-y-1">
              {errors.map((error) => (
                <p key={error} className="font-mono text-xs text-status-error flex gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {error}
                </p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function VerdictBanner({ report }: { report: HeldoutReport }) {
  const ship = report.ship
  return (
    <div
      className={clsx(
        'p-4 border-brutal rounded-brutal flex items-center gap-3',
        ship
          ? 'border-status-success bg-accent-light'
          : 'border-status-error bg-background-secondary'
      )}
    >
      {ship ? (
        <CheckCircle className="w-8 h-8 text-status-success shrink-0" />
      ) : (
        <XCircle className="w-8 h-8 text-status-error shrink-0" />
      )}
      <div>
        <p
          className={clsx(
            'font-brand text-2xl',
            ship ? 'text-status-success' : 'text-status-error'
          )}
        >
          {ship ? 'SHIP' : 'NO-SHIP'}
        </p>
        <p className="font-mono text-xs text-text-muted">
          {ship
            ? 'Candidate beats base on the held-out set with a reliable margin'
            : 'Deploy gate blocked — see reasons below'}
        </p>
      </div>
    </div>
  )
}

function ReleaseGateSummary({ report }: { report: HeldoutReport }) {
  const gate = report.release_gate
  const sections = useMemo(() => {
    if (!gate) return 'none'
    const allSections = [
      ...gate.environment_sections,
      ...(gate.external_benchmark_sections || []),
      ...(gate.world_model_quality_sections || []),
      ...(gate.learned_reward_evidence_sections || [])
    ]
    return allSections.length > 0 ? allSections.join(', ') : 'none'
  }, [gate])
  if (!gate) return null
  const worldModelQualityStatus = gate.world_model_quality_present
    ? gate.world_model_quality_signal === 'needs_attention'
      ? 'WATCH'
      : 'DIAG'
    : 'NONE'
  const learnedRewardStatus = gate.learned_reward_evidence_present
    ? gate.learned_reward_evidence_signal === 'needs_attention'
      ? 'WATCH'
      : 'DIAG'
    : 'NONE'

  return (
    <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary space-y-2">
      <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">
        Combined release gate
      </p>
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-2">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Trace</p>
          <p
            className={clsx(
              'font-mono text-sm',
              gate.trace_ship ? 'text-status-success' : 'text-status-error'
            )}
          >
            {gate.trace_ship ? 'PASS' : 'HOLD'}
          </p>
        </div>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            Environment
          </p>
          <p
            className={clsx(
              'font-mono text-sm',
              gate.environment_ship ? 'text-status-success' : 'text-status-error'
            )}
          >
            {gate.environment_ship ? 'PASS' : 'HOLD'}
          </p>
        </div>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            External
          </p>
          <p
            className={clsx(
              'font-mono text-sm',
              (gate.external_benchmark_ship ?? true) ? 'text-status-success' : 'text-status-error'
            )}
          >
            {(gate.external_benchmark_ship ?? true) ? 'PASS' : 'HOLD'}
          </p>
        </div>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            World Model
          </p>
          <p
            className={clsx(
              'font-mono text-sm',
              gate.world_model_quality_present
                ? gate.world_model_quality_signal === 'needs_attention'
                  ? 'text-status-warning'
                  : 'text-status-success'
                : 'text-text-muted'
            )}
          >
            {worldModelQualityStatus}
          </p>
        </div>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Reward</p>
          <p
            className={clsx(
              'font-mono text-sm',
              gate.learned_reward_evidence_present
                ? gate.learned_reward_evidence_signal === 'needs_attention'
                  ? 'text-status-warning'
                  : 'text-status-success'
                : 'text-text-muted'
            )}
          >
            {learnedRewardStatus}
          </p>
        </div>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
            Evidence
          </p>
          <p className="font-mono text-sm text-text-primary break-words">{sections}</p>
        </div>
      </div>
      {gate.environment_reasons.length > 0 ? (
        <ul className="space-y-0.5 list-disc list-inside">
          {gate.environment_reasons.map((reason) => (
            <li key={reason} className="font-mono text-xs text-text-primary">
              {reason}
            </li>
          ))}
        </ul>
      ) : null}
      {(gate.external_benchmark_reasons || []).length > 0 ? (
        <ul className="space-y-0.5 list-disc list-inside">
          {(gate.external_benchmark_reasons || []).map((reason) => (
            <li key={reason} className="font-mono text-xs text-text-primary">
              {reason}
            </li>
          ))}
        </ul>
      ) : null}
      {(gate.world_model_quality_findings || []).length > 0 ? (
        <ul className="space-y-0.5 list-disc list-inside">
          {(gate.world_model_quality_findings || []).map((reason) => (
            <li key={reason} className="font-mono text-xs text-text-primary">
              {reason}
            </li>
          ))}
        </ul>
      ) : null}
      {(gate.learned_reward_evidence_findings || []).length > 0 ? (
        <ul className="space-y-0.5 list-disc list-inside">
          {(gate.learned_reward_evidence_findings || []).map((reason) => (
            <li key={reason} className="font-mono text-xs text-text-primary">
              {reason}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function ReportView({ report }: { report: HeldoutReport }) {
  const drops = useMemo(
    () => Object.entries(report.forgetting_drops || {}).filter(([, v]) => v > 0),
    [report.forgetting_drops]
  )
  return (
    <div className="space-y-3">
      <VerdictBanner report={report} />
      <ReleaseGateSummary report={report} />

      <div className="grid grid-cols-3 gap-2">
        <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Base</p>
          <p className="font-brand text-xl text-text-primary">{pct(report.base_pass_rate)}</p>
        </div>
        <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Candidate</p>
          <p className="font-brand text-xl text-text-primary">{pct(report.candidate_pass_rate)}</p>
        </div>
        <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Δ Delta</p>
          <p
            className={clsx(
              'font-brand text-xl',
              report.trace_delta > 0 ? 'text-status-success' : 'text-status-error'
            )}
          >
            {report.trace_delta >= 0 ? '+' : ''}
            {pct(report.trace_delta)}
          </p>
        </div>
      </div>

      <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
        <div className="flex items-center justify-between">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">
            95% CI (paired bootstrap, clustered by session)
          </p>
          <span
            className={clsx(
              'tag',
              report.bootstrap.significant ? 'text-status-success' : 'text-text-muted'
            )}
          >
            <span>{report.bootstrap.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT'}</span>
          </span>
        </div>
        <p className="font-mono text-sm text-text-primary mt-1">
          [{pct(report.bootstrap.ci_low)}, {pct(report.bootstrap.ci_high)}] · {report.n} examples ·{' '}
          {report.n_clusters} sessions · metric: {report.metric}
        </p>
      </div>

      {drops.length > 0 && (
        <div className="p-3 border-brutal border-status-error rounded-brutal bg-background-secondary">
          <p className="font-mono text-xs uppercase tracking-widest text-status-error flex items-center gap-1.5">
            <TrendingDown className="w-3.5 h-3.5" /> Forgetting (regressed benchmarks)
          </p>
          <div className="mt-1 space-y-0.5">
            {drops.map(([task, drop]) => (
              <p key={task} className="font-mono text-sm text-text-primary">
                {task}: −{(drop * 100).toFixed(1)}%
              </p>
            ))}
          </div>
        </div>
      )}

      {report.reasons.length > 0 && (
        <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> Why it didn't ship
          </p>
          <ul className="mt-1 space-y-0.5 list-disc list-inside">
            {report.reasons.map((r, i) => (
              <li key={i} className="font-mono text-sm text-text-primary">
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function BenchmarkCommands({ endpoint, modelId }: { endpoint: EndpointForm; modelId: string }) {
  const [commands, setCommands] = useState<Record<string, string[]> | null>(null)
  const [loading, setLoading] = useState(false)
  const [benchmarkName, setBenchmarkName] = useState('harbor_terminal_bench')
  const [externalJson, setExternalJson] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [ingestError, setIngestError] = useState('')
  const [ingestResult, setIngestResult] = useState<ExternalBenchmarkIngestResponse | null>(null)

  const load = async () => {
    if (!endpoint.base_url || !endpoint.model) return
    setLoading(true)
    const r = await evalAdvancedApi.benchmarkCommands({
      base_url: endpoint.base_url,
      model: endpoint.model
    })
    if (r.ok && r.data) setCommands(r.data.commands)
    setLoading(false)
  }

  const ingestExternal = async () => {
    setIngestError('')
    setIngestResult(null)
    let parsed: unknown
    try {
      parsed = JSON.parse(externalJson)
    } catch (e) {
      setIngestError(e instanceof Error ? e.message : String(e))
      return
    }
    setIngesting(true)
    const r = await evalAdvancedApi.ingestExternalBenchmarks({
      model_id: modelId,
      benchmark_name: benchmarkName || undefined,
      results: parsed
    })
    setIngesting(false)
    if (r.ok && r.data) {
      setIngestResult(r.data)
    } else {
      setIngestError(r.error || 'External benchmark ingest failed')
    }
  }

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-accent" />
          <h3 className="font-brand text-lg text-text-primary">External benchmark suite</h3>
        </div>
        <button
          onClick={load}
          disabled={loading || !endpoint.base_url || !endpoint.model}
          className="btn-ghost font-mono text-xs uppercase tracking-widest"
        >
          {loading ? 'Loading…' : 'Show commands'}
        </button>
      </div>
      <p className="text-xs text-text-muted">
        lm-eval (forgetting), Terminal-Bench, Harbor Terminal-Bench, BFCL, and SWE-bench run in the
        serving venv on the host. Run these there, then record public harness scores here.
      </p>
      {commands &&
        Object.entries(commands).map(([name, argv]) => (
          <div
            key={name}
            className="p-2 border-brutal border-border-subtle rounded-brutal bg-background-secondary"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">
                {name}
              </span>
              <button
                onClick={() => navigator.clipboard?.writeText(argv.join(' '))}
                className="btn-icon flex items-center justify-center w-6 h-6"
                title="Copy"
              >
                <Copy className="w-3.5 h-3.5" />
              </button>
            </div>
            <code className="block font-mono text-xs text-text-primary break-all whitespace-pre-wrap">
              {argv.join(' ')}
            </code>
          </div>
        ))}
      <div className="p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Benchmark name
            </label>
            <input
              className="input text-sm w-full"
              value={benchmarkName}
              onChange={(e) => setBenchmarkName(e.target.value)}
              placeholder="harbor_terminal_bench"
            />
          </div>
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Registry model
            </label>
            <input className="input text-sm w-full" value={modelId} readOnly />
          </div>
        </div>
        <textarea
          className="input text-xs w-full min-h-[104px] font-mono"
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          placeholder='{"trials":[{"result":{"reward":1.0}}]}'
          value={externalJson}
          onChange={(e) => setExternalJson(e.target.value)}
        />
        <button
          onClick={ingestExternal}
          disabled={ingesting || !modelId || !externalJson.trim()}
          className="btn-ghost font-mono text-xs uppercase tracking-widest inline-flex items-center gap-2"
        >
          <FileJson className="w-3.5 h-3.5" />
          {ingesting ? 'Recording…' : 'Record result'}
        </button>
        {ingestError ? (
          <p className="font-mono text-xs text-status-error flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> {ingestError}
          </p>
        ) : null}
        {ingestResult ? (
          <div className="space-y-1">
            <p className="font-mono text-xs uppercase tracking-widest text-status-success flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5" /> Recorded {ingestResult.recorded.length}
            </p>
            {ingestResult.report.results.map((result) => (
              <div key={result.name} className="space-y-0.5">
                <p className="font-mono text-xs text-text-primary">
                  {result.name}: {(result.score * 100).toFixed(1)}%
                  {result.total > 0 ? ` (${result.passed}/${result.total})` : ''}
                </p>
                {benchmarkMetricPreview(result).length > 0 ? (
                  <p className="font-mono text-[10px] text-text-muted break-words">
                    {benchmarkMetricPreview(result).join(' · ')}
                  </p>
                ) : null}
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export function HeldoutGatePanel() {
  const { data: modelsData } = useSessionResource(registeredModelsResource)
  const models = useMemo(() => modelsData?.models.map((m) => m.model_id) ?? [], [modelsData])
  const [modelId, setModelId] = useState('')
  const [datasetPath, setDatasetPath] = useState('data/gold_traces/val.jsonl')
  const [metric, setMetric] = useState('exact_match')
  const [candidate, setCandidate] = useState<EndpointForm>({ ...EMPTY_ENDPOINT })
  const [base, setBase] = useState<EndpointForm>({ ...EMPTY_ENDPOINT })
  const [limit, setLimit] = useState<number | ''>(200)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<HeldoutJobResponse | null>(null)
  const [error, setError] = useState('')
  const [environmentEvidenceEnabled, setEnvironmentEvidenceEnabled] = useState(false)
  const [environmentEvidenceRequired, setEnvironmentEvidenceRequired] = useState(true)
  const [evidenceJson, setEvidenceJson] =
    useState<Record<EvidenceFieldId, string>>(EMPTY_EVIDENCE_JSON)

  useEffect(() => {
    if (models.length > 0) setModelId((cur) => cur || models[0])
  }, [models])

  const endpointSpec = (f: EndpointForm) => ({
    provider: f.provider || undefined,
    base_url: f.base_url || undefined,
    model: f.model || undefined
  })

  const parsedPasskEvidence = useMemo(
    () => parseEvidenceObject('Pass@k report', evidenceJson.passk),
    [evidenceJson.passk]
  )
  const parsedHoldoutGateEvidence = useMemo(
    () => parseEvidenceObject('Holdout gate', evidenceJson.holdout_gate),
    [evidenceJson.holdout_gate]
  )
  const parsedHoldoutComparisonEvidence = useMemo(
    () => parseEvidenceObject('Holdout comparison', evidenceJson.holdout_comparison),
    [evidenceJson.holdout_comparison]
  )
  const parsedSpuriousRewardEvidence = useMemo(
    () => parseEvidenceObject('Spurious reward control', evidenceJson.spurious_reward_control),
    [evidenceJson.spurious_reward_control]
  )
  const parsedExternalBenchmarkEvidence = useMemo(
    () => parseEvidenceObject('External benchmarks', evidenceJson.external_benchmarks),
    [evidenceJson.external_benchmarks]
  )
  const parsedWorldModelQualityEvidence = useMemo(
    () => parseEvidenceObject('World-model quality', evidenceJson.world_model_quality),
    [evidenceJson.world_model_quality]
  )
  const parsedLearnedRewardEvidence = useMemo(
    () => parseEvidenceObject('Learned reward', evidenceJson.learned_reward_evidence),
    [evidenceJson.learned_reward_evidence]
  )
  const parsedEvidence = useMemo(
    () => ({
      passk: parsedPasskEvidence,
      holdout_gate: parsedHoldoutGateEvidence,
      holdout_comparison: parsedHoldoutComparisonEvidence,
      spurious_reward_control: parsedSpuriousRewardEvidence,
      external_benchmarks: parsedExternalBenchmarkEvidence,
      world_model_quality: parsedWorldModelQualityEvidence,
      learned_reward_evidence: parsedLearnedRewardEvidence
    }),
    [
      parsedExternalBenchmarkEvidence,
      parsedHoldoutComparisonEvidence,
      parsedHoldoutGateEvidence,
      parsedLearnedRewardEvidence,
      parsedPasskEvidence,
      parsedSpuriousRewardEvidence,
      parsedWorldModelQualityEvidence
    ]
  )
  const evidenceErrors = useMemo(
    () =>
      Object.values(parsedEvidence)
        .map((evidence) => evidence.error)
        .filter((evidenceError): evidenceError is string => evidenceError !== undefined),
    [parsedEvidence]
  )
  const evidenceSectionCount = useMemo(
    () => Object.values(parsedEvidence).filter((evidence) => evidence.value !== undefined).length,
    [parsedEvidence]
  )
  const evidenceReady =
    !environmentEvidenceEnabled ||
    (evidenceErrors.length === 0 && (!environmentEvidenceRequired || evidenceSectionCount > 0))
  const releaseEvidence = useMemo<HeldoutEnvironmentEvidence | undefined>(() => {
    if (!environmentEvidenceEnabled || !evidenceReady) return undefined

    const payload: HeldoutEnvironmentEvidence = { required: environmentEvidenceRequired }
    if (parsedPasskEvidence.value) payload.passk = parsedPasskEvidence.value
    if (parsedHoldoutGateEvidence.value) payload.holdout_gate = parsedHoldoutGateEvidence.value
    if (parsedHoldoutComparisonEvidence.value) {
      payload.holdout_comparison = parsedHoldoutComparisonEvidence.value
    }
    if (parsedSpuriousRewardEvidence.value) {
      payload.spurious_reward_control = parsedSpuriousRewardEvidence.value
    }
    if (parsedExternalBenchmarkEvidence.value) {
      payload.external_benchmarks = parsedExternalBenchmarkEvidence.value
    }
    if (parsedWorldModelQualityEvidence.value) {
      payload.world_model_quality = parsedWorldModelQualityEvidence.value
    }
    if (parsedLearnedRewardEvidence.value) {
      payload.learned_reward_evidence = parsedLearnedRewardEvidence.value
    }
    return payload
  }, [
    environmentEvidenceEnabled,
    environmentEvidenceRequired,
    evidenceReady,
    parsedExternalBenchmarkEvidence,
    parsedHoldoutComparisonEvidence,
    parsedHoldoutGateEvidence,
    parsedLearnedRewardEvidence,
    parsedPasskEvidence,
    parsedSpuriousRewardEvidence,
    parsedWorldModelQualityEvidence
  ])
  const setEvidenceField = useCallback((field: EvidenceFieldId, value: string) => {
    setEvidenceJson((current) => ({ ...current, [field]: value }))
  }, [])

  const canRun =
    !running &&
    !!modelId &&
    !!datasetPath &&
    (!!candidate.provider || !!candidate.base_url) &&
    (!!base.provider || !!base.base_url) &&
    evidenceReady

  const pollJob = async (jobId: string) => {
    const poll = async () => {
      const r = await evalAdvancedApi.heldoutStatus(jobId)
      if (r.ok && r.data) {
        setJob(r.data)
        if (r.data.status === 'running') {
          setTimeout(poll, 2000)
        } else {
          setRunning(false)
          if (r.data.status === 'failed') setError(r.data.error || 'Eval failed')
        }
      } else {
        setRunning(false)
        setError(r.error || 'Polling failed')
      }
    }
    poll()
  }

  const run = async () => {
    setError('')
    setJob(null)
    if (!evidenceReady) {
      setError('Release evidence JSON is incomplete')
      return
    }
    setRunning(true)
    const r = await evalAdvancedApi.runHeldout({
      model_id: modelId,
      dataset_path: datasetPath,
      candidate: endpointSpec(candidate),
      base: endpointSpec(base),
      metric,
      limit: limit === '' ? undefined : limit,
      environment_evidence: releaseEvidence
    })
    if (r.ok && r.data) {
      setJob(r.data)
      pollJob(r.data.job_id)
    } else {
      setRunning(false)
      setError(r.error || 'Failed to start eval')
    }
  }

  return (
    <div className="grid grid-cols-3 gap-6">
      {/* Left: configuration */}
      <div className="col-span-2 space-y-4">
        <div className="card p-4 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 border-brutal border-border rounded-brutal flex items-center justify-center">
              <GitCompare className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h3 className="font-brand text-lg text-text-primary">Held-out trace gate</h3>
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Is the fine-tune actually better than the base?
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                Record verdict against
              </label>
              {models.length > 0 ? (
                <select
                  className="input text-sm w-full"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="input text-sm w-full"
                  placeholder="Model ID"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                />
              )}
            </div>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                Metric
              </label>
              <select
                className="input text-sm w-full"
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                title={METRICS.find((m) => m.id === metric)?.hint}
              >
                {METRICS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                Held-out dataset (.jsonl)
              </label>
              <input
                className="input text-sm w-full"
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
              />
            </div>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                Example limit
              </label>
              <input
                type="number"
                min="1"
                className="input text-sm w-full"
                value={limit}
                onChange={(e) => setLimit(e.target.value === '' ? '' : parseInt(e.target.value))}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <EndpointFields
              label="Candidate (fine-tune)"
              value={candidate}
              onChange={setCandidate}
            />
            <EndpointFields label="Base (comparison)" value={base} onChange={setBase} />
          </div>

          <EnvironmentEvidencePanel
            enabled={environmentEvidenceEnabled}
            required={environmentEvidenceRequired}
            values={evidenceJson}
            parsed={parsedEvidence}
            sectionCount={evidenceSectionCount}
            onEnabledChange={setEnvironmentEvidenceEnabled}
            onRequiredChange={setEnvironmentEvidenceRequired}
            onValueChange={setEvidenceField}
          />

          <button onClick={run} disabled={!canRun} className="btn-primary flex items-center gap-2">
            {running ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Running gate…
              </>
            ) : (
              <>
                <Play className="w-4 h-4" /> Run held-out gate
              </>
            )}
          </button>
          {error && (
            <p className="font-mono text-xs text-status-error flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5" /> {error}
            </p>
          )}
        </div>

        <BenchmarkCommands endpoint={candidate} modelId={modelId} />
      </div>

      {/* Right: result */}
      <div className="space-y-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-5 h-5 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Verdict</h3>
          </div>
          {job?.report ? (
            <ReportView report={job.report} />
          ) : running ? (
            <div className="p-8 text-center">
              <Loader2 className="w-10 h-10 text-accent mx-auto mb-2 animate-spin" />
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Sampling held-out predictions…
              </p>
            </div>
          ) : (
            <div className="p-8 text-center">
              <Shield className="w-10 h-10 text-text-muted mx-auto mb-2" />
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Configure endpoints and run the gate
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
