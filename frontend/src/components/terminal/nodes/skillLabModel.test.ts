import assert from 'node:assert/strict'
import test from 'node:test'
import {
  defaultSkillContract,
  buildSkillLabTerminalPrompt,
  formatSkillPercent,
  isCatalogSkillActive,
  normalizePatternList,
  skillLabPlanScope,
  skillIdFor,
  skillSourceLabel,
  validateSkillContract
} from './skillLabModel'

test('uses backend skill identity and a stable fallback', () => {
  const base = {
    name: 'review',
    description: '',
    source: 'workspace',
    path: 'C:/skills/review/SKILL.md',
    resource_counts: { scripts: 0, references: 0, assets: 0 },
    tool_count: 0
  }
  assert.equal(skillIdFor({ ...base, skill_id: 'skill_123' }), 'skill_123')
  assert.equal(skillIdFor(base), 'workspace:C:/skills/review/SKILL.md:unversioned')
})

test('collapses shared runtime ownership into one readable source label', () => {
  const skill = {
    name: 'review',
    description: 'Review code',
    source: 'agents',
    available_sources: ['agents', 'hermes', 'codex'],
    resource_counts: { scripts: 0, references: 0, assets: 0 },
    tool_count: 0
  }
  assert.equal(skillSourceLabel(skill), 'Agent library + Hermes + Codex')
})

test('builds a terminal handoff that identifies the exact skill and asks before model calls', () => {
  const skill = {
    skill_id: 'skill_123',
    name: 'review',
    description: 'Review code',
    source: 'hermes',
    resource_counts: { scripts: 0, references: 0, assets: 0 },
    tool_count: 0
  }
  const prompt = buildSkillLabTerminalPrompt({
    skill,
    workspaceId: 'main',
    goal: 'Catch behavioral regressions',
    depth: 'quick'
  })
  assert.match(prompt, /skill_123/)
  assert.match(prompt, /Catch behavioral regressions/)
  assert.match(prompt, /ask me before launching/)
})

test('shows only canonical active catalog skills by default', () => {
  const base = {
    name: 'review',
    description: 'Review code',
    source: 'agents',
    resource_counts: { scripts: 0, references: 0, assets: 0 },
    tool_count: 0
  }
  assert.equal(isCatalogSkillActive(base), true)
  assert.equal(isCatalogSkillActive({ ...base, catalog_status: 'active' }), true)
  assert.equal(isCatalogSkillActive({ ...base, catalog_status: 'alternate' }), false)
  assert.equal(isCatalogSkillActive({ ...base, catalog_status: 'deprecated' }), false)
  assert.equal(isCatalogSkillActive({ ...base, catalog_status: 'invalid' }), false)
})

test('presents novice evaluation depth as examples and total calls', () => {
  assert.deepEqual(skillLabPlanScope('quick'), {
    cases: 4,
    evaluationCalls: 12,
    totalCalls: 13
  })
  assert.deepEqual(skillLabPlanScope('thorough'), {
    cases: 8,
    evaluationCalls: 24,
    totalCalls: 25
  })
})

test('normalizes comma and newline pattern input', () => {
  assert.deepEqual(normalizePatternList('tests pass, no regression\nverified'), [
    'tests pass',
    'no regression',
    'verified'
  ])
})

test('requires executable case criteria before launch', () => {
  const contract = defaultSkillContract('main', 'skill_123', 'hermes')
  assert.deepEqual(validateSkillContract(contract), [
    'Add at least one negative routing case',
    'Case 1 needs a prompt',
    'Case 1 needs an expected or forbidden pattern'
  ])
  contract.cases[0] = {
    ...contract.cases[0],
    prompt: 'Review this patch',
    expected_patterns: ['finding']
  }
  contract.cases.push({
    ...contract.cases[0],
    case_id: 'negative',
    name: 'Negative routing',
    should_invoke: false
  })
  assert.deepEqual(validateSkillContract(contract), [])
  assert.equal(formatSkillPercent(0.125, true), '+13%')
})
