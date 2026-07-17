import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'

const source = (path: string) => readFileSync(resolve(process.cwd(), path), 'utf8')

test('active settings and environment surfaces require an explicit or catalog-provided model', () => {
  const activeSurfaces = [
    'src/components/factory/EnvironmentLab.tsx',
    'src/components/settings/ModelsSection.tsx',
    'src/components/settings/HooksSection.tsx',
  ]

  for (const surface of activeSurfaces) {
    assert.equal(
      /qwen/i.test(source(surface)),
      false,
      `${surface} must not ship a repository-owned Qwen choice`
    )
  }

  assert.match(
    source('src/components/factory/EnvironmentLab.tsx'),
    /dppoSmokeBaseModel, setDppoSmokeBaseModel\] = useState\(''\)/
  )
  assert.match(source('src/components/settings/ModelsSection.tsx'), /custom model/i)
  assert.match(source('src/components/settings/HooksSection.tsx'), /Choose a model available in your Ollama catalog/i)
})
