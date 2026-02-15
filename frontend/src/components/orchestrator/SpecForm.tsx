import { useState, useEffect } from 'react'
import { Send, Plus, X, ChevronDown } from 'lucide-react'
import { useOrchestratorStore } from '../../stores/orchestratorStore'
import { clsx } from 'clsx'

export function SpecForm() {
  const { submitSpec, fetchProviders, providers } = useOrchestratorStore()

  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [repository, setRepository] = useState('')
  const [baseBranch, setBaseBranch] = useState('main')
  const [constraints, setConstraints] = useState<string[]>([])
  const [acceptanceCriteria, setAcceptanceCriteria] = useState<string[]>([])
  const [constraintInput, setConstraintInput] = useState('')
  const [criteriaInput, setCriteriaInput] = useState('')
  const [provider, setProvider] = useState('anthropic')
  const [maxBudget, setMaxBudget] = useState(10)
  const [maxWorkers, setMaxWorkers] = useState(5)
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
        repository: repository.trim() || undefined,
        base_branch: baseBranch.trim() || undefined,
        constraints: constraints.length > 0 ? constraints : undefined,
        acceptance_criteria: acceptanceCriteria.length > 0 ? acceptanceCriteria : undefined,
        max_budget_usd: maxBudget,
        max_workers: maxWorkers,
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
    <div className="max-w-2xl mx-auto space-y-6">
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
          className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors"
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
          placeholder="Describe the feature or change in detail. The orchestrator will decompose this into parallel tasks..."
          rows={6}
          className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors resize-y"
        />
      </div>

      {/* Repository */}
      <div>
        <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
          Repository
        </label>
        <input
          type="text"
          value={repository}
          onChange={(e) => setRepository(e.target.value)}
          placeholder="e.g., /path/to/repo"
          className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors"
        />
      </div>

      {/* Base Branch */}
      <div>
        <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
          Base Branch
        </label>
        <input
          type="text"
          value={baseBranch}
          onChange={(e) => setBaseBranch(e.target.value)}
          placeholder="e.g., main"
          className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors"
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
            className="flex-1 bg-background-primary border-brutal border-border rounded-brutal px-3 py-2 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors"
          />
          <button onClick={addConstraint} className="btn-secondary px-3 py-2">
            <Plus className="w-4 h-4" />
          </button>
        </div>
        {constraints.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {constraints.map((c, i) => (
              <span key={i} className="inline-flex items-center gap-1.5 tag">
                {c}
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
            className="flex-1 bg-background-primary border-brutal border-border rounded-brutal px-3 py-2 text-sm font-sans text-text-primary focus:outline-none focus:border-accent transition-colors"
          />
          <button onClick={addCriteria} className="btn-secondary px-3 py-2">
            <Plus className="w-4 h-4" />
          </button>
        </div>
        {acceptanceCriteria.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {acceptanceCriteria.map((c, i) => (
              <span key={i} className="inline-flex items-center gap-1.5 tag">
                {c}
                <button onClick={() => setAcceptanceCriteria(acceptanceCriteria.filter((_, j) => j !== i))}>
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Configuration Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Provider */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            LLM Provider
          </label>
          <div className="relative">
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-sans text-text-primary focus:outline-none focus:border-accent appearance-none cursor-pointer"
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
              className="w-full bg-background-primary border-brutal border-border rounded-brutal pl-7 pr-3 py-2.5 text-sm font-mono text-text-primary focus:outline-none focus:border-accent"
            />
          </div>
        </div>

        {/* Workers */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
            Max Workers
          </label>
          <input
            type="number"
            value={maxWorkers}
            onChange={(e) => setMaxWorkers(Number(e.target.value))}
            min={1}
            max={10}
            className="w-full bg-background-primary border-brutal border-border rounded-brutal px-3 py-2.5 text-sm font-mono text-text-primary focus:outline-none focus:border-accent"
          />
        </div>
      </div>

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
  )
}
