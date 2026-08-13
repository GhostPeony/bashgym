import assert from 'node:assert/strict'
import test from 'node:test'

import { FALLBACK_MODEL_OPTIONS } from './modelOptions'

test('inference fallbacks use current model families and never reintroduce Qwen 2.5', () => {
  assert.ok(
    FALLBACK_MODEL_OPTIONS.some((model) =>
      /qwen3\.5|gemma-4/i.test(`${model.value} ${model.label}`)
    )
  )
  assert.ok(
    FALLBACK_MODEL_OPTIONS.every((model) => !/qwen2\.5/i.test(`${model.value} ${model.label}`))
  )
})
