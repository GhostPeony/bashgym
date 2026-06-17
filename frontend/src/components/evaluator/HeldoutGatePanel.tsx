import { useState, useEffect } from 'react'
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
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  modelsApi,
  evalAdvancedApi,
  type HeldoutJobResponse,
  type HeldoutReport,
} from '../../services/api'

const METRICS = [
  { id: 'exact_match', name: 'Exact match', hint: 'Tool name + every argument identical' },
  { id: 'name_match', name: 'Tool name', hint: 'Right tool, arguments ignored' },
  { id: 'arg_f1', name: 'Argument F1', hint: 'Per-argument overlap' },
  { id: 'soft', name: 'Soft (SERA)', hint: 'Graded partial credit for near-misses' },
]

interface EndpointForm {
  provider: string
  base_url: string
  model: string
}

const EMPTY_ENDPOINT: EndpointForm = { provider: '', base_url: '', model: '' }

function pct(x: number | undefined): string {
  return x === undefined ? '—' : `${(x * 100).toFixed(1)}%`
}

function EndpointFields({
  label,
  value,
  onChange,
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
      <input
        className="input text-sm w-full"
        placeholder="Model name to request"
        value={value.model}
        onChange={(e) => onChange({ ...value, model: e.target.value })}
      />
    </div>
  )
}

function VerdictBanner({ report }: { report: HeldoutReport }) {
  const ship = report.ship
  return (
    <div
      className={clsx(
        'p-4 border-brutal rounded-brutal flex items-center gap-3',
        ship ? 'border-status-success bg-accent-light' : 'border-status-error bg-background-secondary'
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

function ReportView({ report }: { report: HeldoutReport }) {
  const drops = Object.entries(report.forgetting_drops || {}).filter(([, v]) => v > 0)
  return (
    <div className="space-y-3">
      <VerdictBanner report={report} />

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

function BenchmarkCommands({ endpoint }: { endpoint: EndpointForm }) {
  const [commands, setCommands] = useState<Record<string, string[]> | null>(null)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    if (!endpoint.base_url || !endpoint.model) return
    setLoading(true)
    const r = await evalAdvancedApi.benchmarkCommands({
      base_url: endpoint.base_url,
      model: endpoint.model,
    })
    if (r.ok && r.data) setCommands(r.data.commands)
    setLoading(false)
  }

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-accent" />
          <h3 className="font-brand text-lg text-text-primary">External benchmark suite</h3>
        </div>
        <button onClick={load} disabled={loading || !endpoint.base_url || !endpoint.model} className="btn-ghost font-mono text-xs uppercase tracking-widest">
          {loading ? 'Loading…' : 'Show commands'}
        </button>
      </div>
      <p className="text-xs text-text-muted">
        lm-eval (forgetting), Terminal-Bench, BFCL, and SWE-bench run in the serving venv on the
        host. Run these there, then ingest results via the API to fold forgetting into the gate.
      </p>
      {commands &&
        Object.entries(commands).map(([name, argv]) => (
          <div key={name} className="p-2 border-brutal border-border-subtle rounded-brutal bg-background-secondary">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">{name}</span>
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
    </div>
  )
}

export function HeldoutGatePanel() {
  const [models, setModels] = useState<string[]>([])
  const [modelId, setModelId] = useState('')
  const [datasetPath, setDatasetPath] = useState('data/gold_traces/val.jsonl')
  const [metric, setMetric] = useState('exact_match')
  const [candidate, setCandidate] = useState<EndpointForm>({ ...EMPTY_ENDPOINT })
  const [base, setBase] = useState<EndpointForm>({ ...EMPTY_ENDPOINT })
  const [limit, setLimit] = useState<number | ''>(200)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<HeldoutJobResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    modelsApi.list().then((r) => {
      if (r.ok && r.data) {
        const ids = r.data.models.map((m) => m.model_id)
        setModels(ids)
        if (ids.length > 0) setModelId((cur) => cur || ids[0])
      }
    })
  }, [])

  const endpointSpec = (f: EndpointForm) => ({
    provider: f.provider || undefined,
    base_url: f.base_url || undefined,
    model: f.model || undefined,
  })

  const canRun =
    !running &&
    !!modelId &&
    !!datasetPath &&
    (!!candidate.provider || !!candidate.base_url) &&
    (!!base.provider || !!base.base_url)

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
    setRunning(true)
    const r = await evalAdvancedApi.runHeldout({
      model_id: modelId,
      dataset_path: datasetPath,
      candidate: endpointSpec(candidate),
      base: endpointSpec(base),
      metric,
      limit: limit === '' ? undefined : limit,
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
            <EndpointFields label="Candidate (fine-tune)" value={candidate} onChange={setCandidate} />
            <EndpointFields label="Base (comparison)" value={base} onChange={setBase} />
          </div>

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

        <BenchmarkCommands endpoint={candidate} />
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
