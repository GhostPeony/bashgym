import assert from 'node:assert/strict'
import test from 'node:test'
import type { SkillLabRun } from '../services/api'
import { skillRunTransitions } from './skillLabStore'

function run(status: SkillLabRun['status']): SkillLabRun {
  return {
    run_id: 'run-1',
    workspace_id: 'main',
    skill_id: 'skill-1',
    skill_name: 'Review',
    endpoint_id: 'hermes',
    status,
    created_at: '2026-07-10T00:00:00Z',
  }
}

test('announces new and terminal skill eval states once', () => {
  assert.deepEqual(skillRunTransitions([], [run('running')]).map((item) => item.type), [
    'skill-eval:started',
  ])
  assert.deepEqual(
    skillRunTransitions([run('running')], [run('completed')]).map((item) => item.type),
    ['skill-eval:completed'],
  )
  assert.deepEqual(skillRunTransitions([run('completed')], [run('completed')]), [])
})
