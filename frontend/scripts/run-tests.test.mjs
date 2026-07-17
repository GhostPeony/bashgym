import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import test from 'node:test'

import { findTests, selectTests } from './run-tests.mjs'

function run(...filters) {
  return spawnSync(process.execPath, ['scripts/run-tests.mjs', ...filters], {
    cwd: process.cwd(),
    encoding: 'utf8',
  })
}

test('an exact frontend test filter runs only the selected file', () => {
  const tests = [
    'C:\\repo\\src\\campaignApi.test.ts',
    'C:\\repo\\src\\components\\TrainingRunNodeDefaults.test.ts',
  ]
  const selected = selectTests(tests, ['TrainingRunNodeDefaults.test.ts'], 'C:\\repo')

  assert.deepEqual(selected, [tests[1]])
})

test('an unknown frontend test filter fails instead of silently running everything', () => {
  const result = run('missing-focused-test.test.ts')

  assert.notEqual(result.status, 0)
  assert.match(`${result.stdout}\n${result.stderr}`, /No frontend tests matched/)
})

test('the default discovery includes test-runner regressions', () => {
  const scripts = findTests(join(process.cwd(), 'scripts'))

  assert.ok(scripts.some((path) => path.endsWith('run-tests.test.mjs')))
})
