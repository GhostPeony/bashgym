export interface CampaignPublicEventSummary {
  schema_version: 'public_campaign_event_summary.v1'
  action_id?: string
  attempt_id?: string
  study_id?: string
  proposal_id?: string
  entry_id?: string
  stage?: string
  code?: string
  manifest_revision?: number
  stage_index?: number
  next_stage_index?: number
  claim_generation?: number
  cursor_end?: number
  alert_count?: number
  study_completed?: boolean
}

export interface CampaignPublicEvent {
  schema_version: 'public_campaign_event.v1'
  event_id: string
  workspace_id: string
  campaign_id: string
  sequence: number
  aggregate_version: number
  event_type: string
  summary?: CampaignPublicEventSummary | null
  actor_id: string
  credential_kind: string
  created_at: string
}

export interface CampaignEventItem {
  cursor: number
  event: CampaignPublicEvent
}

export interface CampaignArtifact {
  schema_version: 'public_campaign_artifact.v1'
  workspace_id: string
  campaign_id: string
  artifact_id: string
  producer_action_id?: string | null
  sha256: string
  size_bytes: number
  schema_name: string
  sealed: boolean
  valid: boolean
  created_at: string
}

const EVENT_FIELDS = new Set([
  'schema_version',
  'event_id',
  'workspace_id',
  'campaign_id',
  'sequence',
  'aggregate_version',
  'event_type',
  'summary',
  'actor_id',
  'credential_kind',
  'created_at',
])
const SUMMARY_FIELDS = new Set([
  'schema_version',
  'action_id',
  'attempt_id',
  'study_id',
  'proposal_id',
  'entry_id',
  'stage',
  'code',
  'manifest_revision',
  'stage_index',
  'next_stage_index',
  'claim_generation',
  'cursor_end',
  'alert_count',
  'study_completed',
])
const ARTIFACT_FIELDS = new Set([
  'schema_version',
  'workspace_id',
  'campaign_id',
  'artifact_id',
  'producer_action_id',
  'sha256',
  'size_bytes',
  'schema_name',
  'sealed',
  'valid',
  'created_at',
])
const ID_FIELDS = new Set([
  'action_id',
  'attempt_id',
  'study_id',
  'proposal_id',
  'entry_id',
])
const INTEGER_FIELDS = new Set([
  'manifest_revision',
  'stage_index',
  'next_stage_index',
  'claim_generation',
  'cursor_end',
  'alert_count',
])
const STAGES = new Set([
  'data_build',
  'contract_evaluation',
  'smoke_training',
  'full_training',
  'development_evaluation',
  'comparison',
  'recipe_lock',
  'protected_evaluation',
  'promotion',
])
const BLOCKER_CODES = new Set([
  'campaign_controller_action_blocked',
  'campaign_not_found',
  'campaign_stage_cursor_exhausted',
  'campaign_code_lineage_not_captured',
  'campaign_code_lineage_not_registered',
  'campaign_code_lineage_mutation_kind_mismatch',
  'campaign_code_lineage_execution_binding_required',
  'campaign_code_lineage_execution_binding_mismatch',
  'campaign_recipe_runtime_invalid',
  'campaign_remote_stage_not_allowed',
  'campaign_remote_profile_unavailable',
  'campaign_remote_target_model_mismatch',
  'campaign_remote_profile_material_invalid',
  'campaign_executor_kind_not_registered',
  'campaign_budget_unit_not_approved',
])
const ARTIFACT_SCHEMA_NAMES = new Set([
  'campaign_development_comparison.v1',
  'campaign_fake_summary.v1',
  'campaign_remote_exit_code.v1',
  'campaign_remote_launch_manifest.v2',
  'campaign_remote_output.v1',
  'campaign_retrieval_evaluation.v1',
  'campaign_scored_development_rows.v1',
  'campaign_training_log.v1',
  'campaign_unlaunched_cancellation.v1',
  'campaign_validated_dev_dataset.v1',
  'embedding_training_manifest.v1',
  'huggingface_model_file.v1',
  'memexai_query_format_ablation_manifest.v1',
  'nemo_gym_campaign_evidence.v1',
  'training_manifest.v1',
  'training_metrics_jsonl.v1',
  'unclassified_artifact.v1',
])
const CREDENTIAL_KINDS = new Set(['desktop_bootstrap', 'refresh', 'access', 'controller'])
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const SHA256 = /^[0-9a-f]{64}$/

function recordOf(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: Set<string>): boolean {
  return Object.keys(value).every((key) => allowed.has(key))
}

function identifier(value: unknown): value is string {
  return typeof value === 'string' && IDENTIFIER.test(value)
}

function finiteInteger(value: unknown, minimum: number): value is number {
  return Number.isSafeInteger(value) && (value as number) >= minimum
}

function isoTimestamp(value: unknown): value is string {
  return typeof value === 'string' && Number.isFinite(Date.parse(value))
}

function publicSummary(value: unknown): CampaignPublicEventSummary | null {
  const summary = recordOf(value)
  if (!summary || !hasOnlyKeys(summary, SUMMARY_FIELDS)) return null
  if (summary.schema_version !== 'public_campaign_event_summary.v1') return null

  const safe: Record<string, unknown> = {
    schema_version: 'public_campaign_event_summary.v1',
  }
  for (const [key, field] of Object.entries(summary)) {
    if (key === 'schema_version') continue
    if (ID_FIELDS.has(key)) {
      if (!identifier(field)) return null
      safe[key] = field
    } else if (INTEGER_FIELDS.has(key)) {
      const minimum = key === 'manifest_revision' ? 1 : 0
      if (!finiteInteger(field, minimum)) return null
      safe[key] = field
    } else if (key === 'study_completed') {
      if (typeof field !== 'boolean') return null
      safe[key] = field
    } else {
      const allowed = key === 'stage' ? STAGES : key === 'code' ? BLOCKER_CODES : null
      if (!allowed || typeof field !== 'string' || !allowed.has(field)) return null
      safe[key] = field
    }
  }
  return safe as unknown as CampaignPublicEventSummary
}

export function toCampaignActivityFields(value: unknown): Record<string, unknown> | null {
  const event = recordOf(value)
  if (!event || !hasOnlyKeys(event, EVENT_FIELDS)) return null
  if (event.schema_version !== 'public_campaign_event.v1') return null
  if (!identifier(event.event_id)
    || !identifier(event.workspace_id)
    || !identifier(event.campaign_id)
    || !finiteInteger(event.sequence, 1)
    || !finiteInteger(event.aggregate_version, 1)
    || !identifier(event.event_type)
    || !identifier(event.actor_id)
    || typeof event.credential_kind !== 'string'
    || !CREDENTIAL_KINDS.has(event.credential_kind)
    || !isoTimestamp(event.created_at)) return null

  const fields: Record<string, unknown> = {
    event_id: event.event_id,
    aggregate_version: event.aggregate_version,
  }
  if (event.summary !== undefined && event.summary !== null) {
    const summary = publicSummary(event.summary)
    if (!summary) return null
    if (summary.action_id !== undefined) fields.action_id = summary.action_id
    if (summary.attempt_id !== undefined) fields.attempt_id = summary.attempt_id
    if (summary.study_id !== undefined) fields.study_id = summary.study_id
    if (summary.proposal_id !== undefined) fields.proposal_id = summary.proposal_id
    if (summary.entry_id !== undefined) fields.entry_id = summary.entry_id
    if (summary.stage !== undefined) fields.stage = summary.stage
    if (summary.code !== undefined) fields.code = summary.code
    if (summary.manifest_revision !== undefined) fields.manifest_revision = summary.manifest_revision
    if (summary.stage_index !== undefined) fields.stage_index = summary.stage_index
    if (summary.next_stage_index !== undefined) fields.next_stage_index = summary.next_stage_index
    if (summary.claim_generation !== undefined) fields.claim_generation = summary.claim_generation
    if (summary.cursor_end !== undefined) fields.cursor_end = summary.cursor_end
    if (summary.alert_count !== undefined) fields.alert_count = summary.alert_count
    if (summary.study_completed !== undefined) fields.study_completed = summary.study_completed
  }
  return fields
}

export function toCampaignPublicArtifact(value: unknown): CampaignArtifact | null {
  const artifact = recordOf(value)
  if (!artifact || !hasOnlyKeys(artifact, ARTIFACT_FIELDS)) return null
  if (artifact.schema_version !== 'public_campaign_artifact.v1'
    || !identifier(artifact.workspace_id)
    || !identifier(artifact.campaign_id)
    || !identifier(artifact.artifact_id)
    || (artifact.producer_action_id !== undefined
      && artifact.producer_action_id !== null
      && !identifier(artifact.producer_action_id))
    || typeof artifact.sha256 !== 'string'
    || !SHA256.test(artifact.sha256)
    || !finiteInteger(artifact.size_bytes, 0)
    || typeof artifact.schema_name !== 'string'
    || !ARTIFACT_SCHEMA_NAMES.has(artifact.schema_name)
    || typeof artifact.sealed !== 'boolean'
    || typeof artifact.valid !== 'boolean'
    || !isoTimestamp(artifact.created_at)) return null

  return {
    schema_version: 'public_campaign_artifact.v1',
    workspace_id: artifact.workspace_id,
    campaign_id: artifact.campaign_id,
    artifact_id: artifact.artifact_id,
    producer_action_id: artifact.producer_action_id as string | null | undefined,
    sha256: artifact.sha256,
    size_bytes: artifact.size_bytes,
    schema_name: artifact.schema_name,
    sealed: artifact.sealed,
    valid: artifact.valid,
    created_at: artifact.created_at,
  }
}
