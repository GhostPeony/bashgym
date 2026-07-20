import type { CampaignArtifact, CampaignPublicEvent } from '../campaignVisibility'

export type CampaignMeaningTone = 'success' | 'warning' | 'error' | 'info' | 'neutral'

export interface CampaignMeaningPresentation {
  summary: string
  detail: string
  forensic: string
  tone: CampaignMeaningTone
}

export interface CampaignDecisionMeaningInput {
  decision_id: string
  decision_type: string
  outcome: string
  rationale?: string | null
  evidence_refs?: string[]
}

// Remote supervisors seal the numeric exit code with a trailing newline.
const SUCCESSFUL_REMOTE_EXIT_SHA256 = '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa'

const ARTIFACT_SUMMARIES: Record<string, string> = {
  'campaign_development_comparison.v1': 'Development comparison recorded',
  'campaign_fake_summary.v1': 'Simulation summary recorded',
  'campaign_remote_launch_manifest.v2': 'Remote run launch details sealed',
  'campaign_remote_output.v1': 'Remote output sealed for review',
  'campaign_retrieval_evaluation.v1': 'Retrieval evaluation recorded',
  'campaign_scored_development_rows.v1': 'Development scores recorded',
  'campaign_training_log.v1': 'Training log sealed for review',
  'campaign_unlaunched_cancellation.v1': 'Cancelled run recorded',
  'campaign_validated_dev_dataset.v1': 'Development dataset validated',
  'embedding_training_manifest.v1': 'Embedding training plan recorded',
  'huggingface_model_file.v1': 'Trained model file sealed',
  'memexai_query_format_ablation_manifest.v1': 'Query-format ablation plan recorded',
  'nemo_gym_campaign_evidence.v1': 'NeMo Gym evidence recorded',
  'query_format_ablation_manifest.v2': 'Query-format ablation plan recorded',
  'training_manifest.v1': 'Training plan recorded',
  'training_metrics_jsonl.v1': 'Training metrics captured',
  'unclassified_artifact.v1': 'Unclassified artifact recorded',
}

const EVENT_SUMMARIES: Record<string, string> = {
  'campaign:action_blocked': 'Campaign action blocked',
  'campaign:attempt_claimed': 'Training attempt claimed',
  'campaign:attempt_completed': 'Training attempt completed',
  'campaign:attempt_failed': 'Training attempt failed',
  'campaign:attempt_started': 'Training attempt started',
  'campaign:campaign_started': 'Campaign started',
  'campaign:campaign_paused': 'Campaign paused',
  'campaign:campaign_completed': 'Campaign completed',
  'campaign:stage_advanced': 'Campaign advanced to the next stage',
  'campaign:study_completed': 'Study completed',
  'campaign:study_created': 'Study created',
}

function readable(value: string): string {
  return value
    .replace(/^campaign:/, '')
    .replace(/[._:-]+/g, ' ')
    .replace(/\bv(\d+)\b/gi, 'V$1')
    .replace(/\bssh\b/gi, 'SSH')
    .replace(/\bjsonl\b/gi, 'JSONL')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${Number((value / 1024).toFixed(1))} KB`
  return `${Number((value / (1024 * 1024)).toFixed(1))} MB`
}

function artifactState(artifact: CampaignArtifact): { label: string; tone: CampaignMeaningTone } {
  if (!artifact.valid) return { label: 'integrity check failed', tone: 'error' }
  if (!artifact.sealed) return { label: 'not yet sealed', tone: 'warning' }
  return { label: 'sealed and valid', tone: 'success' }
}

export function describeCampaignArtifact(artifact: CampaignArtifact): CampaignMeaningPresentation {
  const state = artifactState(artifact)
  const successfulRemoteExit = artifact.schema_name === 'campaign_remote_exit_code.v1'
    && artifact.sha256 === SUCCESSFUL_REMOTE_EXIT_SHA256
  const knownSummary = ARTIFACT_SUMMARIES[artifact.schema_name]
  const summary = artifact.schema_name === 'campaign_remote_exit_code.v1'
    ? successfulRemoteExit ? 'Remote run finished successfully' : 'Remote run exit status sealed'
    : knownSummary || 'Artifact sealed for inspection'
  const schemaDetail = knownSummary || artifact.schema_name === 'campaign_remote_exit_code.v1'
    ? ''
    : `${readable(artifact.schema_name)} · `

  return {
    summary,
    detail: `${schemaDetail}${formatBytes(artifact.size_bytes)} · ${state.label}`,
    forensic: [
      `Artifact ${artifact.artifact_id}`,
      `schema ${artifact.schema_name}`,
      `SHA-256 ${artifact.sha256}`,
      `producer ${artifact.producer_action_id || 'none'}`,
      `created ${artifact.created_at}`,
    ].join(' · '),
    tone: successfulRemoteExit
      ? 'success'
      : artifact.schema_name === 'campaign_remote_exit_code.v1'
        ? 'neutral'
        : state.tone,
  }
}

function stageCompletionSummary(event: CampaignPublicEvent): string | null {
  const stage = event.summary?.stage
  if (!stage || event.event_type !== 'campaign:attempt_completed') return null
  const stageLabel = readable(stage)
  return `${stageLabel.charAt(0)}${stageLabel.slice(1).toLowerCase()} completed`
}

export function describeCampaignEvent(event: CampaignPublicEvent): CampaignMeaningPresentation {
  const summary = stageCompletionSummary(event)
    || EVENT_SUMMARIES[event.event_type]
    || readable(event.event_type)
  const details: string[] = []
  if (event.summary?.attempt_id) details.push(`Attempt ${event.summary.attempt_id}`)
  if (event.summary?.study_id) details.push(`Study ${event.summary.study_id}`)
  if (event.summary?.stage && !stageCompletionSummary(event)) details.push(`Stage ${readable(event.summary.stage)}`)
  if (event.summary?.code) details.push(readable(event.summary.code))
  if (event.summary?.alert_count !== undefined) details.push(`${event.summary.alert_count} alerts`)

  const failed = /failed|blocked/i.test(event.event_type)
  const completed = /completed|advanced|started/i.test(event.event_type)
  return {
    summary,
    detail: details.join(' · ') || `Recorded by ${readable(event.actor_id)}`,
    forensic: `${event.event_id} · seq ${event.sequence} · v${event.aggregate_version}`,
    tone: failed ? 'error' : completed ? 'success' : 'info',
  }
}

export function describeCampaignExecutor(executor: string | null | undefined): string {
  if (executor === 'ssh_remote') return 'Registered SSH compute'
  if (executor === 'development_evaluation') return 'Local evaluation'
  if (executor === 'fake') return 'Simulated executor'
  return executor ? readable(executor) : 'Executor not assigned'
}

export function describeCampaignDecision(decision: CampaignDecisionMeaningInput): CampaignMeaningPresentation {
  const rawOutcome = decision.outcome.trim()
  const outcomeLabel = /[._:-]/.test(rawOutcome) && !/\s/.test(rawOutcome)
    ? readable(rawOutcome)
    : rawOutcome
  const summary = outcomeLabel
    ? `${outcomeLabel.charAt(0)}${/[._:-]/.test(rawOutcome) ? outcomeLabel.slice(1).toLowerCase() : outcomeLabel.slice(1)}`
    : `${readable(decision.decision_type)} decision recorded`
  const evidence = decision.evidence_refs?.length ? decision.evidence_refs.join(', ') : 'none'

  return {
    summary,
    detail: decision.rationale?.trim() || `${readable(decision.decision_type)} decision recorded`,
    forensic: `Decision ${decision.decision_id} · type ${decision.decision_type} · evidence ${evidence}`,
    tone: /promote|accept|advance/i.test(decision.outcome) ? 'success' : 'info',
  }
}
