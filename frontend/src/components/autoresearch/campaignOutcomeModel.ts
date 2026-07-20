import type {
  CampaignAutoResearchOutcome,
  CampaignDetailState,
  CampaignLedgerEvaluation,
  CampaignMetricValue,
} from '../../stores/campaignStore'

export interface CampaignOutcomeMetric {
  id: string
  baseline: number | null
  candidate: number | null
  delta: number | null
  direction: 'maximize' | 'minimize' | 'unknown'
  primary: boolean
}

export interface CampaignLossSummary {
  attemptId: string
  points: CampaignMetricValue[]
  first: { step: number; value: number }
  minimum: { step: number; value: number }
  final: { step: number; value: number }
}

export interface CampaignOutcomeViewModel {
  verdict: 'success' | 'not_improved' | 'pending' | 'unavailable'
  verdictLabel: string
  lifecycleLabel: string
  lifecycleReason: string | null
  primaryMetricId: string | null
  baselineLabel: string
  candidateLabel: string
  baselineEvaluationId: string | null
  candidateEvaluationId: string | null
  evaluationSuiteId: string | null
  sameEvaluationSuite: boolean | null
  metrics: CampaignOutcomeMetric[]
  decision: string | null
  loss: CampaignLossSummary | null
  checkpointWarning: string | null
}

function newestBy<T>(items: readonly T[], timestamp: (item: T) => string, tie: (item: T) => string): T | undefined {
  return [...items].sort((left, right) => (
    timestamp(left).localeCompare(timestamp(right)) || tie(left).localeCompare(tie(right))
  )).at(-1)
}

function completedOutcome(
  outcomes: readonly CampaignAutoResearchOutcome[],
  role: 'baseline' | 'candidate',
  bestProposalId?: string | null,
): CampaignAutoResearchOutcome | undefined {
  const eligible = outcomes.filter((item) => (
    item.result.role === role
    && item.result.provenance === 'real'
    && item.result.outcome === 'completed'
    && typeof item.result.metric_value === 'number'
  ))
  const preferred = role === 'candidate' && bestProposalId
    ? eligible.filter((item) => item.result.proposal_id === bestProposalId)
    : eligible
  return newestBy(preferred.length ? preferred : eligible, (item) => item.result.recorded_at, (item) => item.result.result_id)
}

function allEvaluations(detail: CampaignDetailState): CampaignLedgerEvaluation[] {
  return (detail.ledger?.projects || []).flatMap((project) => project.evaluations)
}

function findEvaluation(
  evaluations: readonly CampaignLedgerEvaluation[],
  outcome: CampaignAutoResearchOutcome | undefined,
  explicitId?: string | null,
): CampaignLedgerEvaluation | undefined {
  if (explicitId) {
    const explicit = evaluations.find((evaluation) => evaluation.evaluation_result_id === explicitId)
    if (explicit) return explicit
  }
  if (!outcome) return undefined
  const byEvidence = evaluations.filter((evaluation) => outcome.result.evidence_references.includes(evaluation.evaluation_result_id))
  if (byEvidence.length) return newestBy(byEvidence, (item) => item.created_at, (item) => item.evaluation_result_id)
  const byRun = evaluations.filter((evaluation) => outcome.result.attempt_ids.includes(evaluation.run_id))
  return newestBy(byRun, (item) => item.created_at, (item) => item.evaluation_result_id)
}

function baselineLedgerEvaluation(detail: CampaignDetailState): CampaignLedgerEvaluation | undefined {
  const candidates: CampaignLedgerEvaluation[] = []
  for (const project of detail.ledger?.projects || []) {
    const baselineRunIds = new Set(project.runs.filter((run) => run.run_kind === 'baseline' && run.is_simulation !== true).map((run) => run.run_id))
    candidates.push(...project.evaluations.filter((evaluation) => baselineRunIds.has(evaluation.run_id)))
  }
  return newestBy(candidates, (item) => item.created_at, (item) => item.evaluation_result_id)
}

function candidateLedgerEvaluation(detail: CampaignDetailState): CampaignLedgerEvaluation | undefined {
  const candidates: CampaignLedgerEvaluation[] = []
  for (const project of detail.ledger?.projects || []) {
    const baselineRunIds = new Set(project.runs.filter((run) => run.run_kind === 'baseline' && run.is_simulation !== true).map((run) => run.run_id))
    candidates.push(...project.evaluations.filter((evaluation) => !baselineRunIds.has(evaluation.run_id)))
  }
  return newestBy(candidates, (item) => item.created_at, (item) => item.evaluation_result_id)
}

function outcomeMetrics(
  evaluation: CampaignLedgerEvaluation | undefined,
  outcome: CampaignAutoResearchOutcome | undefined,
): Record<string, number> {
  if (evaluation) return evaluation.metrics
  if (outcome && typeof outcome.result.metric_value === 'number') {
    return { [outcome.result.metric_name]: outcome.result.metric_value }
  }
  return {}
}

function candidateLooksLikeLora(detail: CampaignDetailState, candidate: CampaignAutoResearchOutcome | undefined): boolean {
  const identifiers = [candidate?.result.proposal_id, detail.snapshot?.candidate?.candidate_ref]
    .filter((value): value is string => Boolean(value))
    .join(' ')
  if (/lora/i.test(identifiers)) return true
  const proposal = (detail.ledger?.autoresearch?.proposals || []).find((item) => (
    typeof item.proposal_id === 'string' && item.proposal_id === candidate?.result.proposal_id
  ))
  return proposal ? /lora/i.test(JSON.stringify(proposal)) : false
}

function projectLoss(detail: CampaignDetailState, candidate: CampaignAutoResearchOutcome | undefined): CampaignLossSummary | null {
  const preferredAttemptIds = candidate?.result.attempt_ids || []
  const trainingAttemptIds = [...detail.attempts]
    .filter((attempt) => ['full_training', 'smoke_training'].includes(attempt.stage))
    .sort((left, right) => left.updated_at.localeCompare(right.updated_at))
    .reverse()
    .map((attempt) => attempt.attempt_id)
  const attemptId = [...new Set([...preferredAttemptIds, ...trainingAttemptIds, ...Object.keys(detail.lossByAttempt)])]
    .find((id) => (detail.lossByAttempt[id] || []).length > 0)
  if (!attemptId) return null
  const points = [...detail.lossByAttempt[attemptId]!].sort((left, right) => left.step - right.step)
  const minimumPoint = points.reduce((minimum, point) => point.value < minimum.value ? point : minimum)
  const first = points[0]!
  const final = points.at(-1)!
  return {
    attemptId,
    points,
    first: { step: first.step, value: first.value },
    minimum: { step: minimumPoint.step, value: minimumPoint.value },
    final: { step: final.step, value: final.value },
  }
}

function comparisonValidity(
  baseline: CampaignLedgerEvaluation | undefined,
  candidate: CampaignLedgerEvaluation | undefined,
  baselineOutcome: CampaignAutoResearchOutcome | undefined,
  candidateOutcome: CampaignAutoResearchOutcome | undefined,
): boolean | null {
  if (baseline && candidate) return baseline.evaluation_suite_id === candidate.evaluation_suite_id
  if (!baseline && !candidate && baselineOutcome && candidateOutcome) return true
  return null
}

function lifecycleLabel(detail: CampaignDetailState): string {
  const autoresearch = detail.ledger?.autoresearch?.state
  if (autoresearch?.next_action === 'stop' && !['completed', 'exhausted', 'failed', 'cancelled'].includes(detail.campaign.status)) {
    return 'Closeout pending'
  }
  return detail.campaign.status.replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function projectCampaignOutcome(detail: CampaignDetailState): CampaignOutcomeViewModel | null {
  const autoresearch = detail.ledger?.autoresearch
  const outcomes = autoresearch?.outcomes || []
  const baselineOutcome = completedOutcome(outcomes, 'baseline')
  const candidateOutcome = completedOutcome(outcomes, 'candidate', autoresearch?.state.best_proposal_id)
  const evaluations = allEvaluations(detail)
  const baselineEvaluation = findEvaluation(evaluations, baselineOutcome) || baselineLedgerEvaluation(detail)
  const candidateEvaluation = findEvaluation(evaluations, candidateOutcome, detail.snapshot?.candidate?.latest_comparable_evaluation_id)
    || candidateLedgerEvaluation(detail)
  const baselineMetrics = outcomeMetrics(baselineEvaluation, baselineOutcome)
  const candidateMetrics = outcomeMetrics(candidateEvaluation, candidateOutcome)
  const loss = projectLoss(detail, candidateOutcome)
  if (!Object.keys(baselineMetrics).length && !Object.keys(candidateMetrics).length && !loss) return null

  const primaryMetricId = autoresearch?.spec.primary_metric
    || baselineOutcome?.result.metric_name
    || candidateOutcome?.result.metric_name
    || Object.keys(candidateMetrics)[0]
    || Object.keys(baselineMetrics)[0]
    || null
  const metricIds = [...new Set([
    ...(primaryMetricId ? [primaryMetricId] : []),
    ...Object.keys(baselineMetrics),
    ...Object.keys(candidateMetrics),
  ])]
  const direction = autoresearch?.spec.metric_direction || 'unknown'
  const metrics = metricIds.map((id): CampaignOutcomeMetric => {
    const baseline = baselineMetrics[id] ?? null
    const candidate = candidateMetrics[id] ?? null
    return {
      id,
      baseline,
      candidate,
      delta: baseline == null || candidate == null ? null : candidate - baseline,
      direction: id === primaryMetricId ? direction : 'unknown',
      primary: id === primaryMetricId,
    }
  })
  const sameEvaluationSuite = comparisonValidity(baselineEvaluation, candidateEvaluation, baselineOutcome, candidateOutcome)
  const decision = candidateOutcome?.decision.decision || autoresearch?.state.latest_decision || null
  const verdict = sameEvaluationSuite === false
    ? 'pending'
    : decision === 'keep'
      ? 'success'
      : decision === 'discard' || decision === 'crash' || decision === 'ineligible'
        ? 'not_improved'
        : metrics.some((metric) => metric.baseline != null && metric.candidate != null)
          ? 'pending'
          : 'unavailable'
  const verdictLabel = sameEvaluationSuite === false
    ? 'Comparison not valid'
    : verdict === 'success'
      ? 'Candidate kept'
      : verdict === 'not_improved'
        ? 'Baseline retained'
        : verdict === 'pending'
          ? 'Decision pending'
          : 'Outcome unavailable'
  const checkpointWarning = autoresearch?.diagnostics?.signals.find((signal) => (
    signal.code === 'checkpoint_evidence_missing' || signal.severity === 'critical'
  ))?.summary || null

  return {
    verdict,
    verdictLabel,
    lifecycleLabel: lifecycleLabel(detail),
    lifecycleReason: autoresearch?.state.reason_code || detail.campaign.stop_reason || null,
    primaryMetricId,
    baselineLabel: 'Baseline',
    candidateLabel: candidateLooksLikeLora(detail, candidateOutcome) ? 'LoRA candidate' : 'Candidate',
    baselineEvaluationId: baselineEvaluation?.evaluation_result_id || null,
    candidateEvaluationId: candidateEvaluation?.evaluation_result_id || null,
    evaluationSuiteId: sameEvaluationSuite === true ? baselineEvaluation?.evaluation_suite_id || candidateEvaluation?.evaluation_suite_id || null : null,
    sameEvaluationSuite,
    metrics,
    decision,
    loss,
    checkpointWarning,
  }
}

export function selectCampaignOutcomeLoss(detail: CampaignDetailState): CampaignMetricValue[] {
  return projectCampaignOutcome(detail)?.loss?.points || []
}
