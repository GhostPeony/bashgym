import { useState, useEffect } from 'react'
import { Send, Plus, X, ChevronDown } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

export function SpecForm() {
  const { submitSpec, fetchProviders, providers } = useOrchestratorStore()

  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [constraints, setConstraints] = useState<string[]>([])
  const [acceptanceCriteria, setAcceptanceCriteria] = useState<string[]>([])
  const [constraintInput, setConstraintInput] = useState('')
  const [criteriaInput, setCriteriaInput] = useState('')
  const [provider, setProvider] = useState('anthropic')
  const [maxBudget, setMaxBudget] = useState(10)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetchProviders()
  }, [fetchProviders])

  const handleSubmit = async () => {
    if (!title.trim() || !description.trim()) return
    setSubmitting(true)
    try {
      await submitSpec({
        title: title.trim(),
        description: description.trim(),
        constraints: constraints.length > 0 ? constraints : undefined,
        acceptance_criteria: acceptanceCriteria.length > 0 ? acceptanceCriteria : undefined,
        max_budget_usd: maxBudget,
        llm_config: { provider },
      })
    } finally {
      setSubmitting(false)
    }
  }

  const addConstraint = () => {
    if (constraintInput.trim()) {
      setConstraints([...constraints, constraintInput.trim()])
      setConstraintInput('')
    }
  }

  const addCriteria = () => {
    if (criteriaInput.trim()) {
      setAcceptanceCriteria([...acceptanceCriteria, criteriaInput.trim()])
      setCriteriaInput('')
    }
  }

  return (
    <div className="card p-6">
      {/* Card Header */}
      <div className="flex items-center gap-3 mb-1">
        <div className="w-8 h-8 flex items-center justify-center border-brutal border-border rounded-brutal bg-accent-light">
          <Send className="w-4 h-4 text-accent-dark" />
        </div>
        <h2 className="font-brand text-lg text-text-primary">Orchestration Spec</h2>
      </div>
      <div className="section-divider my-4" />

      <div className="space-y-5">
        {/* Title */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., Add user authentication system"
            className="input w-full text-sm"
          />
        </div>

        {/* Description */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe the feature or change in detail. The orchestrator will decompose this into parallel tasks for your terminal agents..."
            rows={4}
            className="input w-full text-sm resize-y"
          />
        </div>

        {/* Constraints */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            Constraints
          </label>
          <div className="flex gap-2 mb-2">
            <input
              type="text"
              value={constraintInput}
              onChange={(e) => setConstraintInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addConstraint())}
              placeholder="e.g., Must use existing auth middleware"
              className="input flex-1 text-sm"
            />
            <button onClick={addConstraint} className="btn-secondary px-3 py-2">
              <Plus className="w-4 h-4" />
            </button>
          </div>
          {constraints.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {constraints.map((c, i) => (
                <span key={i} className="inline-flex items-center gap-1.5 tag">
                  <span>{c}</span>
                  <button onClick={() => setConstraints(constraints.filter((_, j) => j !== i))}>
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Acceptance Criteria */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            Acceptance Criteria
          </label>
          <div className="flex gap-2 mb-2">
            <input
              type="text"
              value={criteriaInput}
              onChange={(e) => setCriteriaInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addCriteria())}
              placeholder="e.g., All tests pass, Login endpoint returns JWT"
              className="input flex-1 text-sm"
            />
            <button onClick={addCriteria} className="btn-secondary px-3 py-2">
              <Plus className="w-4 h-4" />
            </button>
          </div>
          {acceptanceCriteria.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {acceptanceCriteria.map((c, i) => (
                <span key={i} className="inline-flex items-center gap-1.5 tag">
                  <span>{c}</span>
                  <button onClick={() => setAcceptanceCriteria(acceptanceCriteria.filter((_, j) => j !== i))}>
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Decomposition Config Fieldset */}
        <fieldset className="border-brutal border-border rounded-brutal p-4 bg-background-card">
          <legend className="flex items-center gap-2 px-2">
            <span className="font-brand text-lg text-text-primary">Decomposition Config</span>
          </legend>
          <div className="grid grid-cols-2 gap-4">
            {/* Provider */}
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                LLM Provider
              </label>
              <div className="relative">
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="input w-full text-sm appearance-none cursor-pointer"
                >
                  {providers.length > 0 ? (
                    providers.map(p => (
                      <option key={p.provider} value={p.provider}>
                        {p.provider} ({p.default_model})
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="anthropic">anthropic</option>
                      <option value="openai">openai</option>
                      <option value="gemini">gemini</option>
                      <option value="ollama">ollama</option>
                    </>
                  )}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
              </div>
            </div>

            {/* Budget */}
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                Max Budget
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm">$</span>
                <input
                  type="number"
                  value={maxBudget}
                  onChange={(e) => setMaxBudget(Number(e.target.value))}
                  min={1}
                  max={100}
                  step={1}
                  className="input w-full text-sm font-mono pl-7"
                />
              </div>
            </div>
          </div>
        </fieldset>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={!title.trim() || !description.trim() || submitting}
          className={clsx(
            'btn-primary w-full font-mono text-sm py-3 flex items-center justify-center gap-2',
            (!title.trim() || !description.trim() || submitting) && 'opacity-50 cursor-not-allowed'
          )}
        >
          <Send className="w-4 h-4" />
          {submitting ? 'Decomposing...' : 'Submit Spec'}
        </button>
      </div>
    </div>
  )
}
