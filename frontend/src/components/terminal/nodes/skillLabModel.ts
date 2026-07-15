import type {
  SkillLabCase,
  SkillLabContract,
  SkillLabKpis,
  SkillLabThresholds,
  ToolkitSkill,
} from '../../../services/api'

export const DEFAULT_SKILL_THRESHOLDS: SkillLabThresholds = {
  min_uplift: 0.1,
  min_forced_pass_rate: 0.8,
  min_routing_precision: 0.9,
  min_routing_recall: 0.85,
  max_false_activation_rate: 0.05,
}

export type SkillLabDepth = 'quick' | 'thorough'

const SOURCE_LABELS: Record<string, string> = {
  agents: 'Agent library',
  claude: 'Claude',
  codex: 'Codex',
  'codex-system': 'Codex system',
  hermes: 'Hermes',
  peony: 'BashGym',
  workspace: 'Workspace',
}

export function skillLabPlanScope(depth: SkillLabDepth) {
  const cases = depth === 'quick' ? 4 : 8
  return {
    cases,
    evaluationCalls: cases * 3,
    totalCalls: cases * 3 + 1,
  }
}

export function skillIdFor(skill: ToolkitSkill): string {
  if (skill.skill_id) return skill.skill_id
  return [skill.source, skill.path || skill.name, skill.revision || 'unversioned'].join(':')
}

export function isCatalogSkillActive(skill: ToolkitSkill): boolean {
  return !skill.catalog_status || skill.catalog_status === 'active'
}

export function skillSourceLabel(skill: ToolkitSkill): string {
  const sources = Array.from(new Set([skill.source, ...(skill.available_sources || [])]))
  return sources.map((source) => SOURCE_LABELS[source] || source).join(' + ')
}

export function buildSkillLabTerminalPrompt({
  skill,
  workspaceId,
  goal,
  depth,
}: {
  skill: ToolkitSkill
  workspaceId: string
  goal?: string
  depth: SkillLabDepth
}): string {
  const focus = goal?.trim() || 'Judge whether the skill activates appropriately and improves task quality.'
  return [
    `Help me test the BashGym skill "${skill.name}" in workspace "${workspaceId}".`,
    `Use the BashGym Skill Lab MCP tools to inspect skill id "${skillIdFor(skill)}" before testing it.`,
    `Prepare a ${depth} evaluation focused on: ${focus}`,
    'Exercise representative target and non-target prompts in this session, explain observed failures, and propose concrete skill changes.',
    'If a healthy model endpoint is available, preview the model-call count and ask me before launching the recorded paired Skill Lab run.',
  ].join(' ')
}

export function emptySkillCase(index = 1): SkillLabCase {
  return {
    case_id: `case-${Date.now()}-${index}`,
    name: `Case ${index}`,
    prompt: '',
    should_invoke: true,
    expected_patterns: [],
    forbidden_patterns: [],
  }
}

export function defaultSkillContract(
  workspaceId: string,
  skillId: string,
  endpointId = '',
): SkillLabContract {
  return {
    workspace_id: workspaceId,
    skill_id: skillId,
    endpoint_id: endpointId,
    cases: [emptySkillCase(1)],
    thresholds: { ...DEFAULT_SKILL_THRESHOLDS },
  }
}

export function normalizePatternList(value: string): string[] {
  return value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean)
}

export function validateSkillContract(contract: SkillLabContract): string[] {
  const errors: string[] = []
  if (!contract.skill_id) errors.push('Choose a skill')
  if (!contract.endpoint_id) errors.push('Choose an agent endpoint')
  if (!contract.cases.length) errors.push('Add at least one eval case')
  if (contract.cases.length && !contract.cases.some((item) => item.should_invoke)) {
    errors.push('Add at least one target case')
  }
  if (contract.cases.length && !contract.cases.some((item) => !item.should_invoke)) {
    errors.push('Add at least one negative routing case')
  }
  for (const [index, item] of contract.cases.entries()) {
    if (!item.name.trim()) errors.push(`Case ${index + 1} needs a name`)
    if (!item.prompt.trim()) errors.push(`Case ${index + 1} needs a prompt`)
    if (!item.expected_patterns.length && !item.forbidden_patterns.length) {
      errors.push(`Case ${index + 1} needs an expected or forbidden pattern`)
    }
  }
  return errors
}

export function formatSkillPercent(value?: number | null, signed = false): string {
  if (value == null || !Number.isFinite(value)) return '-'
  const formatted = `${Math.round(value * 100)}%`
  return signed && value > 0 ? `+${formatted}` : formatted
}

export function kpiTone(kpis?: SkillLabKpis | null): 'success' | 'warning' | 'error' | 'neutral' {
  if (!kpis || kpis.verdict === 'untested') return 'neutral'
  if (kpis.verdict === 'effective') return 'success'
  if (kpis.verdict === 'watch') return 'warning'
  return 'error'
}
