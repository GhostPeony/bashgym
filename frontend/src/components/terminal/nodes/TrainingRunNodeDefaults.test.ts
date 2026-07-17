import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

test('a fresh training canvas node never invents a base model', async () => {
  const source = await readFile(new URL('./TrainingRunNode.tsx', import.meta.url), 'utf8')

  assert.doesNotMatch(source, /Qwen\/Qwen2\.5|Qwen2\.5-Coder/i)
  assert.match(
    source,
    /baseModel:\s*config\?\.baseModel\s*\|\|\s*node\.baseModel\s*\|\|\s*DEFAULT_TRAINING_BASE_MODEL/,
  )
})
