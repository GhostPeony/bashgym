import assert from 'node:assert/strict'
import test from 'node:test'
import {
  recipeFromTrainingQueued,
  recipeFromWorkspaceIntent,
  shouldReuseTrainingOrigin,
} from './canvasRecipes'

test('turns an agent Skill Lab intent into a linked singleton recipe', () => {
  const recipe = recipeFromWorkspaceIntent({
    type: 'skill_lab.prepared',
    workspace_id: 'main',
    source: { panel_id: 'agent-panel', terminal_id: 'terminal-1', agent: 'codex' },
    entity: { kind: 'skill_lab', skill_id: 'skill-1' },
    suggested_nodes: [{
      recipe: 'skill_lab',
      config: { selectedSkillId: 'skill-1', selectedSkillName: 'factory' },
    }],
  })

  assert.equal(recipe?.type, 'skilllab')
  assert.equal(recipe?.key, 'skilllab:main')
  assert.equal(recipe?.originPanelId, 'agent-panel')
  assert.equal(recipe?.config.selectedSkillId, 'skill-1')
  assert.equal(recipe?.edgeType, 'context')
})

test('reuses an unbound training origin for the run it launches', () => {
  const recipe = recipeFromTrainingQueued({
    run_id: 'run_123',
    strategy: 'sft',
    origin: { panel_id: 'training-panel' },
  })

  assert.equal(
    shouldReuseTrainingOrigin(recipe, { type: 'training', adapterConfig: { strategy: 'sft' } }),
    true,
  )
})

test('does not overwrite a training origin already bound to a run', () => {
  const recipe = recipeFromTrainingQueued({ run_id: 'run_new', strategy: 'dpo' })

  assert.equal(
    shouldReuseTrainingOrigin(recipe, {
      type: 'training',
      adapterConfig: { runId: 'run_existing' },
    }),
    false,
  )
})
