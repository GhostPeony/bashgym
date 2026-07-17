import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { BaseModelSelect } from '../common/BaseModelSelect'
import { BASE_MODEL_GROUPS } from '../common/baseModels'
import { ModelSelect } from '../common/ModelSelect'
import { mergeModelOptions, selectBaseModelGroups } from '../common/modelSelectOptions'
import { FALLBACK_MODEL_OPTIONS } from '../common/modelOptions'

const source = (path: string) => readFileSync(resolve(process.cwd(), path), 'utf8')
const exactPattern = (value: string) =>
  new RegExp(value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))

test('catalog-only inference options keep mocked catalog entries and suppress every static fallback', () => {
  const catalog = [{ value: 'registered/example', label: 'Registered example' }]
  assert.deepEqual(mergeModelOptions(catalog, true), catalog)
  assert.deepEqual(
    mergeModelOptions(catalog, false),
    [...FALLBACK_MODEL_OPTIONS, ...catalog]
  )
})

test('catalog-only base-model groups keep a mocked catalog and suppress every static group', () => {
  const catalog = [
    {
      label: 'Registered catalog',
      models: [{ value: 'registered/base', label: 'Registered base' }],
    },
  ]
  assert.deepEqual(selectBaseModelGroups(catalog, true), catalog)
  assert.deepEqual(selectBaseModelGroups(catalog, false), [...BASE_MODEL_GROUPS, ...catalog])
})

test('rendered model selectors toggle static fallbacks while retaining custom selection', () => {
  const renderInference = (catalogOnly: boolean) =>
    renderToStaticMarkup(
      createElement(ModelSelect, { value: '', onChange() {}, catalogOnly })
    )
  const renderBase = (catalogOnly: boolean) =>
    renderToStaticMarkup(
      createElement(BaseModelSelect, { value: '', onChange() {}, catalogOnly })
    )

  const inferenceWithFallbacks = renderInference(false)
  const inferenceCatalogOnly = renderInference(true)
  for (const fallback of FALLBACK_MODEL_OPTIONS) {
    assert.match(inferenceWithFallbacks, exactPattern(fallback.value))
    assert.doesNotMatch(inferenceCatalogOnly, exactPattern(fallback.value))
  }
  assert.match(inferenceCatalogOnly, /Custom model/)

  const baseWithFallbacks = renderBase(false)
  const baseCatalogOnly = renderBase(true)
  for (const fallback of BASE_MODEL_GROUPS.flatMap((group) => group.models)) {
    assert.match(baseWithFallbacks, exactPattern(fallback.value))
    assert.doesNotMatch(baseCatalogOnly, exactPattern(fallback.value))
  }
  assert.match(baseCatalogOnly, /Custom model/)
})

test('active settings and environment surfaces bind selectors to catalog-only mode', () => {
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
  assert.equal(
    source('src/components/factory/EnvironmentLab.tsx').match(/\bcatalogOnly\b/g)?.length,
    2
  )
  assert.equal(
    source('src/components/settings/ModelsSection.tsx').match(/\bcatalogOnly\b/g)?.length,
    1
  )
  assert.match(
    source('src/components/settings/HooksSection.tsx'),
    /Choose a model available in your Ollama catalog/i
  )
})
