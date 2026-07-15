import assert from 'node:assert/strict'
import test from 'node:test'
import { BASE_MODEL_GROUPS } from './baseModels'

test('Gemma 4 suggestions separate the 12B training base from NVFP4 deployment', () => {
  const gemma = BASE_MODEL_GROUPS.find((group) => group.label === 'Gemma 4')
  assert.ok(gemma)

  assert.ok(gemma.models.some((model) => model.value === 'unsloth/gemma-4-12b-it'))
  assert.ok(gemma.models.every((model) => !model.value.toLowerCase().includes('nvfp4')))
})
