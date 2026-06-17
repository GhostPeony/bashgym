import { useState, useEffect } from 'react'
import {
  GitBranch,
  Play,
  Loader2,
  CheckCircle,
  AlertCircle,
  ShieldCheck,
  Filter,
} from 'lucide-react'
import {
  dataQualityApi,
  type DataQualityDefaults,
  type DecisionDpoJob,
} from '../../services/api'

interface QualityForm {
  gold_dir: string
  failed_dir: string
  generate_decision_dpo: boolean
  require_successful_verification: boolean
  min_trace_steps: number
  max_trace_steps: number
}

const DEFAULT_FORM: QualityForm = {
  gold_dir: 'data/gold_traces',
  failed_dir: '',
  generate_decision_dpo: true,
  require_successful_verification: true,
  min_trace_steps: 2,
  max_trace_steps: 50,
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
  icon: Icon,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (v: boolean) => void
  icon: typeof ShieldCheck
}) {
  return (
    <label className="flex items-start gap-3 p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary cursor-pointer hover:border-border transition-colors">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="rounded-brutal mt-0.5"
      />
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-accent" />
          <span className="font-mono text-sm font-semibold text-text-primary">{label}</span>
        </div>
        <p className="text-xs text-text-muted mt-0.5">{hint}</p>
      </div>
    </label>
  )
}

export function DecisionDpoPanel() {
  const [form, setForm] = useState<QualityForm>(DEFAULT_FORM)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<DecisionDpoJob | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    dataQualityApi.defaults().then((r) => {
      if (r.ok && r.data) {
        const d = r.data as DataQualityDefaults
        setForm((f) => ({
          ...f,
          generate_decision_dpo: d.generate_decision_dpo,
          require_successful_verification: d.require_successful_verification,
          min_trace_steps: d.min_trace_steps,
          max_trace_steps: d.max_trace_steps,
        }))
      }
    })
  }, [])

  const pollJob = async (jobId: string) => {
    const poll = async () => {
      const r = await dataQualityApi.decisionDpoStatus(jobId)
      if (r.ok && r.data) {
        setJob(r.data)
        if (r.data.status === 'running' || r.data.status === 'queued') {
          setTimeout(poll, 1500)
        } else {
          setRunning(false)
          if (r.data.status === 'failed') setError(r.data.error || 'Mining failed')
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
    const r = await dataQualityApi.mineDecisionDpo({
      gold_dir: form.gold_dir || undefined,
      failed_dir: form.failed_dir || undefined,
      generate_decision_dpo: form.generate_decision_dpo,
      require_successful_verification: form.require_successful_verification,
      min_trace_steps: form.min_trace_steps,
      max_trace_steps: form.max_trace_steps,
    })
    if (r.ok && r.data) {
      setJob(r.data)
      pollJob(r.data.job_id)
    } else {
      setRunning(false)
      setError(r.error || 'Failed to start mining')
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="card p-6 space-y-5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 border-brutal border-border rounded-brutal flex items-center justify-center bg-accent-light">
            <GitBranch className="w-5 h-5 text-accent-dark" />
          </div>
          <div>
            <h3 className="font-brand text-lg text-text-primary">Decision-DPO mining</h3>
            <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
              Mine FAILURE → SUCCESS preference pairs from gold traces
            </p>
          </div>
        </div>

        <p className="text-sm text-text-secondary">
          Builds step-level DPO pairs from within each gold trace — where the agent failed a step
          then recovered — gated by the trace-quality toggles below. Runs fully local (no LLM
          augmentation); pairs export to a NeMo-compatible DPO batch.
        </p>

        {/* Quality toggles */}
        <div className="grid grid-cols-2 gap-3">
          <Toggle
            label="Generate decision-DPO"
            hint="Mine step-level preference pairs from gold traces"
            checked={form.generate_decision_dpo}
            onChange={(v) => setForm({ ...form, generate_decision_dpo: v })}
            icon={GitBranch}
          />
          <Toggle
            label="Require verified traces"
            hint="Only mine traces where verification_passed=true"
            checked={form.require_successful_verification}
            onChange={(v) => setForm({ ...form, require_successful_verification: v })}
            icon={ShieldCheck}
          />
        </div>

        {/* Step bounds + dirs */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              <Filter className="w-3 h-3 inline mr-1" /> Min trace steps
            </label>
            <input
              type="number"
              min="1"
              value={form.min_trace_steps}
              onChange={(e) => setForm({ ...form, min_trace_steps: parseInt(e.target.value) || 1 })}
              className="input text-sm w-full"
            />
          </div>
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Max trace steps
            </label>
            <input
              type="number"
              min="1"
              value={form.max_trace_steps}
              onChange={(e) => setForm({ ...form, max_trace_steps: parseInt(e.target.value) || 1 })}
              className="input text-sm w-full"
            />
          </div>
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Gold traces directory
            </label>
            <input
              type="text"
              value={form.gold_dir}
              onChange={(e) => setForm({ ...form, gold_dir: e.target.value })}
              placeholder="data/gold_traces"
              className="input text-sm w-full font-mono"
            />
          </div>
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Failed traces directory (optional)
            </label>
            <input
              type="text"
              value={form.failed_dir}
              onChange={(e) => setForm({ ...form, failed_dir: e.target.value })}
              placeholder="data/failed_traces"
              className="input text-sm w-full font-mono"
            />
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={run}
            disabled={running || form.min_trace_steps > form.max_trace_steps}
            className="btn-primary flex items-center gap-2"
          >
            {running ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Mining…
              </>
            ) : (
              <>
                <Play className="w-4 h-4" /> Mine DPO pairs
              </>
            )}
          </button>
          {form.min_trace_steps > form.max_trace_steps && (
            <p className="font-mono text-xs text-status-error">min must be ≤ max</p>
          )}
        </div>
        {error && (
          <p className="font-mono text-xs text-status-error flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> {error}
          </p>
        )}
      </div>

      {/* Result */}
      {job?.status === 'completed' && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-status-success" />
            <h3 className="font-brand text-lg text-text-primary">Mining complete</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary text-center">
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
                DPO pairs
              </p>
              <p className="font-brand text-3xl text-text-primary">{job.n_dpo_pairs ?? 0}</p>
            </div>
            <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary text-center">
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">
                Gold SFT examples
              </p>
              <p className="font-brand text-3xl text-text-primary">
                {job.n_training_examples ?? 0}
              </p>
            </div>
          </div>
          {job.dpo_output_path && (
            <div className="mt-3 pt-3 border-t-2 border-border">
              <p className="text-xs text-text-muted font-mono break-all">
                Output: {job.dpo_output_path}
              </p>
            </div>
          )}
          {job.n_dpo_pairs === 0 && (
            <p className="text-xs text-text-muted mt-3">
              No FAILURE → SUCCESS sequences were found in the selected traces. Loosen the quality
              filters or point at a directory with recovered-error traces.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
