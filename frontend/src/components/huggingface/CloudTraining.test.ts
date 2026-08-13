import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { hfApi, type HFJob } from '../../services/api'
import { CloudTrainingView } from './CloudTraining'

function renderView(patch: Partial<Parameters<typeof CloudTrainingView>[0]> = {}): string {
  return renderToStaticMarkup(
    createElement(CloudTrainingView, {
      jobs: [],
      error: null,
      loading: false,
      refreshing: false,
      onRefresh() {},
      ...patch
    })
  )
}

function job(overrides: Partial<HFJob> = {}): HFJob {
  return {
    job_id: 'provider-job-123',
    status: 'running',
    hardware: 'a10g-small',
    created_at: '2026-08-11T00:00:00+00:00',
    ...overrides
  }
}

test('presents Hugging Face Jobs as an observation-only provider surface', () => {
  const html = renderView()

  assert.match(html, /Observe jobs launched through Hugging Face Jobs/)
  assert.match(html, /Launch and cancellation stay with the workflow that owns the provider job/)
  assert.doesNotMatch(html, /New Job|Submit Job|Cancel Job|Estimated cost|\$[0-9]|VRAM/)
})

test('renders provider read failures instead of an empty-jobs success state', () => {
  const html = renderView({ error: 'HTTP 503: provider unavailable' })

  assert.match(html, /HTTP 503: provider unavailable/)
  assert.doesNotMatch(html, /No jobs returned/)
})

test('renders provider namespace and identity without synthetic hardware details', () => {
  const html = renderView({
    jobs: [
      job({
        namespace: 'research-org',
        logs_url: 'https://huggingface.co/jobs/research-org/provider-job-123'
      })
    ]
  })

  assert.match(html, /provider-job-123/)
  assert.match(html, /research-org/)
  assert.match(html, /a10g-small/)
  assert.doesNotMatch(html, /24GB|\$1\.05/)
})

test('job observation clients explicitly confirm access and preserve namespaces', async () => {
  const originalFetch = globalThis.fetch
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
  const originalLocalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  const urls: string[] = []
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {} })
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: () => null }
  })
  globalThis.fetch = (async (input: string | URL | Request) => {
    urls.push(String(input))
    const body = urls.length === 1 ? '[]' : urls.length === 3 ? '{"logs":"ok"}' : '{}'
    return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch

  try {
    await hfApi.listJobs('research org')
    await hfApi.getJob('provider-job-123', 'research org')
    await hfApi.getJobLogs('provider-job-123', 'research org')
  } finally {
    globalThis.fetch = originalFetch
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow)
    else Reflect.deleteProperty(globalThis, 'window')
    if (originalLocalStorage)
      Object.defineProperty(globalThis, 'localStorage', originalLocalStorage)
    else Reflect.deleteProperty(globalThis, 'localStorage')
  }

  assert.deepEqual(
    urls.map((url) => new URL(url, 'http://localhost').searchParams.toString()),
    [
      'jobs_access_confirmed=true&namespace=research+org',
      'jobs_access_confirmed=true&namespace=research+org',
      'jobs_access_confirmed=true&namespace=research+org'
    ]
  )
})
