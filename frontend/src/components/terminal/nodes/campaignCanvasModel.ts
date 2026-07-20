import type {
  CampaignAttempt,
  CampaignAutoResearchOutcome,
  CampaignDetailState,
  CampaignLedgerDecision,
  CampaignLedgerEvaluation,
  CampaignLedgerProject,
  CampaignProposalRecord,
  CampaignStudy
} from '../../../stores/campaignStore'

export interface CampaignBaselineSummary {
  status: 'evaluated' | 'recorded' | 'not_recorded'
  modelRef: string | null
  runId: string | null
  evaluationSuiteId: string | null
  metrics: Record<string, number>
}

export interface CampaignStudySummary {
  studyId: string
  proposalId: string
  status: string
  hypothesis: string | null
  primaryVariable: string | null
  falsificationCriterion: string | null
  plannedStage: string | null
}

export interface CampaignBudgetSummary {
  resource: string
  remaining: number
}

export interface CampaignDecisionSummary {
  decisionId: string
  type: string
  outcome: string
  rationale: string
  createdAt: string
}

export interface CampaignNextActionSummary {
  kind: 'current' | 'planned' | 'attention' | 'none'
  label: string
  actionId: string | null
}

export interface CampaignCanvasResearch {
  objective: string
  baseline: CampaignBaselineSummary
  latestStudy: CampaignStudySummary | null
  budget: CampaignBudgetSummary[]
  latestDecision: CampaignDecisionSummary | null
  nextAction: CampaignNextActionSummary
}

type CampaignLedgerRun = CampaignLedgerProject['runs'][number]

function stringField(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  return typeof value === 'string' && value.trim() ? value : null
}

function newestBy<T>(
  items: readonly T[],
  timestamp: (item: T) => string,
  tie: (item: T) => string
): T | undefined {
  return [...items]
    .sort((left, right) => {
      const byTime = timestamp(left).localeCompare(timestamp(right))
      return byTime || tie(left).localeCompare(tie(right))
    })
    .at(-1)
}

function latestStudy(detail: CampaignDetailState): CampaignStudy | undefined {
  const active = detail.campaign.active_study_id
    ? detail.studies.find((study) => study.study_id === detail.campaign.active_study_id)
    : undefined
  return (
    active ||
    newestBy(
      detail.studies,
      (study) => study.updated_at,
      (study) => study.study_id
    )
  )
}

function proposalFor(
  study: CampaignStudy | undefined,
  proposals: readonly CampaignProposalRecord[]
): CampaignProposalRecord | undefined {
  if (!study) return undefined
  return proposals.find((record) => record.proposal.proposal_id === study.proposal_id)
}

function projectBaseline(detail: CampaignDetailState): CampaignBaselineSummary {
  const baseModel =
    stringField(detail.campaign.target_model, 'base_model_ref') ||
    stringField(detail.campaign.target_model, 'base_model')
  const candidates: Array<{ project: CampaignLedgerProject; run: CampaignLedgerRun }> = []
  for (const project of detail.ledger?.projects || []) {
    for (const run of project.runs) {
      if (run.run_kind === 'baseline' && run.is_simulation !== true)
        candidates.push({ project, run })
    }
  }
  const selected = newestBy(
    candidates,
    ({ run }) => run.queued_at || '',
    ({ project, run }) => `${project.project.project_id}:${run.run_id}`
  )
  const autoresearchBaseline = newestBy(
    (detail.ledger?.autoresearch?.outcomes || []).filter(
      (item) =>
        item.result.role === 'baseline' &&
        item.result.provenance === 'real' &&
        item.result.outcome === 'completed' &&
        item.decision.decision === 'baseline' &&
        item.decision.eligible_for_best &&
        typeof item.result.metric_value === 'number'
    ),
    (item: CampaignAutoResearchOutcome) => item.result.recorded_at,
    (item: CampaignAutoResearchOutcome) => item.result.result_id
  )
  if (autoresearchBaseline) {
    return {
      status: 'evaluated',
      modelRef: baseModel,
      runId: null,
      evaluationSuiteId: null,
      metrics: {
        [autoresearchBaseline.result.metric_name]: autoresearchBaseline.result.metric_value!
      }
    }
  }
  if (!selected) {
    return {
      status: 'not_recorded',
      modelRef: baseModel,
      runId: null,
      evaluationSuiteId: null,
      metrics: {}
    }
  }
  const evaluation = newestBy(
    selected.project.evaluations.filter((item) => item.run_id === selected.run.run_id),
    (item: CampaignLedgerEvaluation) => item.created_at,
    (item: CampaignLedgerEvaluation) => item.evaluation_result_id
  )
  return {
    status: evaluation ? 'evaluated' : 'recorded',
    modelRef: baseModel || selected.run.model_version_id || null,
    runId: selected.run.run_id,
    evaluationSuiteId: evaluation?.evaluation_suite_id || null,
    metrics: evaluation?.metrics || {}
  }
}

function projectDecision(detail: CampaignDetailState): CampaignDecisionSummary | null {
  const ledgerDecisions: CampaignDecisionSummary[] = (detail.ledger?.projects || [])
    .flatMap((project) => project.decisions)
    .map((decision: CampaignLedgerDecision) => ({
      decisionId: decision.decision_id,
      type: decision.decision_type,
      outcome: decision.outcome,
      rationale: decision.rationale,
      createdAt: decision.created_at
    }))
  const autoresearchDecisions: CampaignDecisionSummary[] = (
    detail.ledger?.autoresearch?.outcomes || []
  ).map((item) => ({
    decisionId: `autoresearch:${item.result.result_id}`,
    type: item.decision.decision,
    outcome: item.decision.reason_code,
    rationale:
      item.decision.improvement == null
        ? `${item.result.provenance} ${item.result.role} result`
        : `Primary metric improvement ${item.decision.improvement}`,
    createdAt: item.decision.decided_at
  }))
  return (
    newestBy(
      [...ledgerDecisions, ...autoresearchDecisions],
      (item) => item.createdAt,
      (item) => item.decisionId
    ) || null
  )
}

function projectNextAction(
  detail: CampaignDetailState,
  study: CampaignStudy | undefined
): CampaignNextActionSummary {
  if (detail.campaign.active_action_id) {
    const actionId = detail.campaign.active_action_id
    const attempt = newestBy(
      detail.attempts.filter((item) => item.action_id === actionId),
      (item: CampaignAttempt) => item.updated_at,
      (item: CampaignAttempt) => item.attempt_id
    )
    return {
      kind: 'current',
      label: attempt ? `${attempt.stage} · ${attempt.status}` : actionId,
      actionId
    }
  }
  const codeLineage = study
    ? (detail.ledger?.autoresearch?.code_lineages || []).find(
        (item) => item.proposal_id === study.proposal_id
      )
    : undefined
  if (codeLineage?.state === 'required') {
    return { kind: 'attention', label: 'prepare code lineage', actionId: null }
  }
  if (codeLineage?.state === 'prepared') {
    return { kind: 'attention', label: 'edit and capture code lineage', actionId: null }
  }
  const stage = study?.stage_plan.items[study.current_stage_index]?.stage
  if (stage) return { kind: 'planned', label: stage, actionId: null }

  const autoresearch = detail.ledger?.autoresearch?.state
  if (autoresearch) {
    switch (autoresearch.next_action) {
      case 'prepare_campaign':
        return { kind: 'planned', label: 'validate AutoResearch campaign', actionId: null }
      case 'start_campaign':
        return { kind: 'attention', label: 'authorize campaign start', actionId: null }
      case 'submit_baseline':
        return { kind: 'planned', label: 'submit real baseline', actionId: null }
      case 'wait_for_result':
        return { kind: 'planned', label: 'ingest authoritative experiment result', actionId: null }
      case 'propose_candidate':
        return { kind: 'planned', label: 'propose one controlled candidate', actionId: null }
      case 'stop':
        return {
          kind: 'attention',
          label: `enforce stop · ${autoresearch.reason_code}`,
          actionId: null
        }
      case 'blocked':
        return { kind: 'attention', label: autoresearch.reason_code, actionId: null }
    }
  }

  switch (detail.campaign.status) {
    case 'draft':
    case 'validating':
      return { kind: 'planned', label: 'validate campaign manifest', actionId: null }
    case 'ready':
      return { kind: 'attention', label: 'start campaign', actionId: null }
    case 'paused':
      return { kind: 'attention', label: 'resume campaign', actionId: null }
    case 'awaiting_authority':
      return { kind: 'attention', label: 'satisfy required authority', actionId: null }
    case 'cancelling':
      return { kind: 'current', label: 'settle cancellation', actionId: null }
    case 'active':
      return Number(detail.evidence?.proposal_counts.submitted || 0) > 0
        ? { kind: 'planned', label: 'select submitted proposal', actionId: null }
        : { kind: 'planned', label: 'await scientist proposal', actionId: null }
    default:
      return { kind: 'none', label: 'no further action', actionId: null }
  }
}

/**
 * Build a read-only canvas projection from durable campaign evidence.
 * This is deliberately not a store: every field remains reproducible from CampaignDetailState.
 */
export function projectCampaignResearch(detail: CampaignDetailState): CampaignCanvasResearch {
  const study = latestStudy(detail)
  const proposal = proposalFor(study, detail.proposals)
  return {
    objective: detail.campaign.objective,
    baseline: projectBaseline(detail),
    latestStudy: study
      ? {
          studyId: study.study_id,
          proposalId: study.proposal_id,
          status: study.status,
          hypothesis: proposal?.proposal.hypothesis || null,
          primaryVariable: proposal?.proposal.primary_variable || null,
          falsificationCriterion: proposal?.proposal.falsification_criterion || null,
          plannedStage: study.stage_plan.items[study.current_stage_index]?.stage || null
        }
      : null,
    budget: Object.entries(detail.evidence?.budget_remaining || {})
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([resource, remaining]) => ({ resource, remaining })),
    latestDecision: projectDecision(detail),
    nextAction: projectNextAction(detail, study)
  }
}
