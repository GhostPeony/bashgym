import { useRef, useState, type FormEvent } from 'react'
import { campaignApi } from '../../services/api'
import type { CampaignProposalRecord } from '../../stores/campaignStore'

type ProposalRole = 'general' | 'baseline' | 'candidate'
const emptyRecipes = {
  dataset_recipe: {},
  training_recipe: {},
  evaluation_recipe: {},
  required_capabilities: [],
  controlled_variables: [],
  evidence_references: [],
  stage_plan: { schema_version: 'campaign_stage_plan.v1', items: [] }
}

export function ProposalEditor({
  workspaceId,
  campaignId,
  version,
  authoritative,
  latest,
  onSubmitted
}: {
  workspaceId: string
  campaignId: string
  version: number
  authoritative: boolean
  latest?: CampaignProposalRecord
  onSubmitted: () => Promise<void>
}) {
  const [role, setRole] = useState<ProposalRole | ''>('')
  const [proposalId, setProposalId] = useState('')
  const [hypothesis, setHypothesis] = useState('')
  const [variable, setVariable] = useState('')
  const [family, setFamily] = useState('')
  const [expected, setExpected] = useState('')
  const [falsification, setFalsification] = useState('')
  const [rationale, setRationale] = useState('')
  const [cost, setCost] = useState('')
  const [parentId, setParentId] = useState('')
  const [recipes, setRecipes] = useState(JSON.stringify(emptyRecipes, null, 2))
  const [pending, setPending] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const submission = useRef<{
    fingerprint: string
    body: Record<string, unknown>
    key: string
  } | null>(null)
  const parsed = (() => {
    try {
      return JSON.parse(recipes)
    } catch {
      return null
    }
  })()
  const stages: Array<{ stage: string; disposition: string; reason: string }> = Array.isArray(
    parsed?.stage_plan?.items
  )
    ? parsed.stage_plan.items.filter((item: unknown) => item && typeof item === 'object')
    : []
  const validRecipes =
    parsed &&
    typeof parsed === 'object' &&
    !Array.isArray(parsed) &&
    ['dataset_recipe', 'training_recipe', 'evaluation_recipe'].every(
      (key) => parsed[key] && typeof parsed[key] === 'object' && !Array.isArray(parsed[key])
    ) &&
    stages.length > 0
  function cloneLatest() {
    if (!latest) return
    const proposal = latest.proposal as unknown as Record<string, unknown>
    setHypothesis(latest.proposal.hypothesis)
    setVariable(latest.proposal.primary_variable)
    setFamily(latest.proposal.study_family)
    setExpected(latest.proposal.expected_outcome)
    setFalsification(latest.proposal.falsification_criterion)
    setCost(String(latest.proposal.estimated_cost))
    setRationale(typeof proposal.rationale === 'string' ? proposal.rationale : '')
    setParentId(latest.proposal.proposal_id)
    setRecipes(
      JSON.stringify(
        Object.fromEntries(
          Object.entries(emptyRecipes).map(([key, value]) => [key, proposal[key] ?? value])
        ),
        null,
        2
      )
    )
    setMessage(
      'Previous proposal copied. Choose a new proposal ID and review every field before submitting.'
    )
  }
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!authoritative || pending || !role || !validRecipes) return
    setPending(true)
    setMessage(null)
    const fields = {
      ...parsed,
      proposal_id: proposalId,
      hypothesis,
      study_family: family,
      primary_variable: variable,
      expected_outcome: expected,
      falsification_criterion: falsification,
      rationale,
      estimated_cost: Number(cost),
      ...(role === 'candidate'
        ? { parent_proposal_id: parentId, changed_variables: [variable] }
        : {})
    }
    const fingerprint = JSON.stringify({ role, fields })
    if (submission.current?.fingerprint !== fingerprint)
      submission.current = {
        fingerprint,
        body: { ...fields, workspace_id: workspaceId, expected_version: version },
        key: `idem_${crypto.randomUUID().replaceAll('-', '')}`
      }
    try {
      const response = await campaignApi.submitProposal(
        campaignId,
        role,
        submission.current.body,
        submission.current.key
      )
      if (!response.ok) {
        if (response.code === 'campaign_version_conflict') {
          // A version conflict is a definitive rejection, so a reviewed retry
          // can use fresh authority. Ambiguous failures retain their original key.
          submission.current = null
          await onSubmitted()
          setMessage(
            'The campaign changed. Your draft is preserved. Review the latest state and submit again.'
          )
          return
        }
        setMessage(response.error || 'Proposal could not be submitted. Your draft is preserved.')
        return
      }
      const validation = response.data?.record?.validation
      if (typeof validation?.valid !== 'boolean' || !Array.isArray(validation.reason_codes)) {
        setMessage(
          'The service did not return a validation result. Refresh the journal before retrying; your draft is preserved.'
        )
        return
      }
      setMessage(
        validation?.valid
          ? 'Proposal recorded and validated. Review the journal for its next stage.'
          : `Proposal recorded with validation blockers: ${validation?.reason_codes?.join(', ') || 'inspect the journal for details'}.`
      )
      await onSubmitted()
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : 'The service could not complete this request. Your draft is preserved.'
      )
    } finally {
      setPending(false)
    }
  }
  return (
    <details className="studio-paper studio-proposal-editor">
      <summary>Propose an experiment</summary>
      <p>
        Record a hypothesis, recipes, and a stage plan. The campaign validator checks the proposal
        against its approved contract.
      </p>
      {!authoritative && (
        <p className="studio-notice" role="status">
          Reconnect and refresh the campaign before submitting. Your draft remains editable.
        </p>
      )}
      {latest && (
        <button type="button" className="studio-text-link" onClick={cloneLatest} disabled={pending}>
          Copy latest proposal into draft
        </button>
      )}
      <form onSubmit={submit}>
        <div className="studio-editor-grid">
          <label>
            Proposal role
            <select
              aria-label="Proposal role"
              required
              value={role}
              onChange={(event) => setRole(event.target.value as ProposalRole)}
            >
              <option value="">Choose a role</option>
              <option value="baseline">AutoResearch baseline</option>
              <option value="candidate">AutoResearch candidate</option>
              <option value="general">General study</option>
            </select>
          </label>
          <label>
            Proposal ID
            <input
              required
              maxLength={160}
              pattern="[A-Za-z0-9][A-Za-z0-9_.:-]*"
              value={proposalId}
              onChange={(event) => setProposalId(event.target.value)}
            />
          </label>
          <label>
            Study family
            <input
              required
              maxLength={160}
              value={family}
              onChange={(event) => setFamily(event.target.value)}
            />
          </label>
          <label>
            Estimated cost
            <input
              required
              type="number"
              min="0"
              step="any"
              value={cost}
              onChange={(event) => setCost(event.target.value)}
            />
          </label>
        </div>
        {role === 'candidate' && (
          <label>
            Parent proposal ID
            <input
              required
              value={parentId}
              onChange={(event) => setParentId(event.target.value)}
            />
          </label>
        )}
        <label>
          Hypothesis
          <textarea
            aria-label="Hypothesis"
            required
            maxLength={4000}
            value={hypothesis}
            onChange={(event) => setHypothesis(event.target.value)}
          />
        </label>
        <label>
          One variable to change
          <input
            required
            maxLength={1000}
            value={variable}
            onChange={(event) => setVariable(event.target.value)}
          />
        </label>
        <label>
          Expected outcome
          <textarea
            aria-label="Expected outcome"
            required
            maxLength={2000}
            value={expected}
            onChange={(event) => setExpected(event.target.value)}
          />
        </label>
        <label>
          Falsification criterion
          <textarea
            aria-label="Falsification criterion"
            required
            maxLength={2000}
            value={falsification}
            onChange={(event) => setFalsification(event.target.value)}
          />
        </label>
        <label>
          Rationale
          <textarea
            aria-label="Rationale"
            required
            maxLength={4000}
            value={rationale}
            onChange={(event) => setRationale(event.target.value)}
          />
        </label>
        <label>
          Recipes and stage plan (JSON)
          <textarea
            className="studio-recipe-json"
            aria-label="Recipes and stage plan (JSON)"
            spellCheck={false}
            value={recipes}
            onChange={(event) => setRecipes(event.target.value)}
            aria-describedby="recipe-help"
          />
        </label>
        <p id="recipe-help" className="studio-caption">
          Use registered recipe references. Each stage item needs stage, disposition (required or
          not_applicable), and reason. Supported stages: data_build, contract_evaluation,
          smoke_training, full_training, development_evaluation, comparison, recipe_lock,
          protected_evaluation, promotion.
        </p>
        {stages.length > 0 && (
          <ol className="studio-stage-list" aria-label="Draft stage graph">
            {stages.map((stage, index) => (
              <li key={index}>
                <span>{String(stage.stage || 'Unnamed stage')}</span>
                <small>
                  {String(stage.disposition || 'Disposition needed')} ·{' '}
                  {String(stage.reason || 'Reason needed')}
                </small>
              </li>
            ))}
          </ol>
        )}
        {!validRecipes && (
          <p className="studio-caption">
            Add valid recipe objects and at least one stage before submitting.
          </p>
        )}
        {message && (
          <p className="studio-notice" role="status">
            {message}
          </p>
        )}
        <button
          type="submit"
          className="studio-primary"
          disabled={pending || !authoritative || !validRecipes}
        >
          {pending ? 'Submitting…' : 'Submit proposal for validation'}
        </button>
      </form>
    </details>
  )
}
