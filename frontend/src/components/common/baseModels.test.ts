import assert from 'node:assert/strict'
import test from 'node:test'
import { BASE_MODEL_GROUPS, DEFAULT_TRAINING_BASE_MODEL, isExplicitBaseModel } from './baseModels'

test('Gemma 4 suggestions separate the 12B training base from NVFP4 deployment', () => {
  const gemma = BASE_MODEL_GROUPS.find((group) => group.label === 'Gemma 4')
  assert.ok(gemma)

  assert.ok(gemma.models.some((model) => model.value === 'unsloth/gemma-4-12b-it'))
  assert.ok(gemma.models.every((model) => !model.value.toLowerCase().includes('nvfp4')))
})

test('fresh installs are model-neutral and do not suggest the retired Qwen 2.5 family', () => {
  assert.equal(DEFAULT_TRAINING_BASE_MODEL, '')
  assert.equal(isExplicitBaseModel(''), false)
  assert.equal(isExplicitBaseModel('   '), false)
  assert.equal(isExplicitBaseModel('operator/model@revision'), true)

  const suggestions = BASE_MODEL_GROUPS.flatMap((group) => group.models)
  assert.ok(suggestions.length > 0)
  assert.ok(suggestions.every((model) => !/qwen2\.5/i.test(`${model.value} ${model.label}`)))
  assert.ok(suggestions.every((model) => !/default/i.test(model.label)))
})
