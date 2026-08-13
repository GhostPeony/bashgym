import assert from 'node:assert/strict'
import test from 'node:test'
import { runtimeApi, type RuntimeJob } from '../services/api'
import { useActivityStore } from './activityStore'
import { useCanvasOrchestratorStore } from './canvasOrchestratorStore'
import { useRuntimeStore } from './runtimeStore'

const observedJob: RuntimeJob = {
  job_id: 'runtime_designer_42',
  kind: 'designer',
  status: 'running',
  pid: 42,
  title: 'Generate dataset',
  script: 'generate.py',
  cwd: 'C:\\workspace',
  started_at: '2026-07-09T20:00:00Z',
  pipeline: 'generate_dataset',
  artifacts: [],
  options: {},
  source: 'process_observer'
}

test('confirms completion across three missing polls and retracts it on rediscovery', async () => {
  const originalListJobs = runtimeApi.listJobs
  let activeJobs: RuntimeJob[] = []
  runtimeApi.listJobs = async () => ({
    ok: true,
    data: { jobs: activeJobs, polled_at: '2026-07-09T20:01:00Z' }
  })

  useActivityStore.getState().clear()
  useRuntimeStore.getState().clear()
  useRuntimeStore.setState({ jobs: [observedJob] })

  try {
    await useRuntimeStore.getState().poll()
    assert.equal(useRuntimeStore.getState().jobs[0].status, 'running')
    assert.equal(useActivityStore.getState().events.length, 0)

    await useRuntimeStore.getState().poll()
    assert.equal(useRuntimeStore.getState().jobs[0].status, 'running')
    assert.equal(useActivityStore.getState().events.length, 0)

    await useRuntimeStore.getState().poll()
    assert.equal(useRuntimeStore.getState().jobs[0].status, 'completed')
    assert.equal(
      useActivityStore.getState().events[0].key,
      'designer:completed:runtime_designer_42'
    )

    activeJobs = [observedJob]
    await useRuntimeStore.getState().poll()
    assert.equal(useRuntimeStore.getState().jobs[0].status, 'running')
    assert.equal(useActivityStore.getState().events.length, 0)
  } finally {
    runtimeApi.listJobs = originalListJobs
    useRuntimeStore.getState().clear()
    useActivityStore.getState().clear()
  }
})

test('does not republish or replace identical active runtime jobs', async () => {
  const originalListJobs = runtimeApi.listJobs
  const originalHandleRuntimeJob = useCanvasOrchestratorStore.getState().handleRuntimeJob
  let handled = 0
  let pollNumber = 0
  runtimeApi.listJobs = async () => ({
    ok: true,
    data: {
      jobs: [{ ...observedJob }],
      polled_at: `2026-07-09T20:0${pollNumber++}:00Z`
    }
  })
  useCanvasOrchestratorStore.setState({
    handleRuntimeJob: () => {
      handled += 1
    }
  })
  useActivityStore.getState().clear()
  useRuntimeStore.getState().clear()

  try {
    await useRuntimeStore.getState().poll()
    const firstJobs = useRuntimeStore.getState().jobs
    assert.equal(handled, 1)

    await useRuntimeStore.getState().poll()
    assert.equal(handled, 1)
    assert.equal(useRuntimeStore.getState().jobs, firstJobs)
  } finally {
    runtimeApi.listJobs = originalListJobs
    useCanvasOrchestratorStore.setState({ handleRuntimeJob: originalHandleRuntimeJob })
    useRuntimeStore.getState().clear()
    useActivityStore.getState().clear()
  }
})

test('publishes and replaces runtime state when progress actually changes', async () => {
  const originalListJobs = runtimeApi.listJobs
  const originalHandleRuntimeJob = useCanvasOrchestratorStore.getState().handleRuntimeJob
  let handled = 0
  let current = 10
  runtimeApi.listJobs = async () => ({
    ok: true,
    data: {
      jobs: [{ ...observedJob, progress: { current, total: 100, unit: 'seeds' } }],
      polled_at: '2026-07-09T20:01:00Z'
    }
  })
  useCanvasOrchestratorStore.setState({
    handleRuntimeJob: () => {
      handled += 1
    }
  })
  useRuntimeStore.getState().clear()

  try {
    await useRuntimeStore.getState().poll()
    const firstJobs = useRuntimeStore.getState().jobs
    current = 20
    await useRuntimeStore.getState().poll()

    assert.equal(handled, 2)
    assert.notEqual(useRuntimeStore.getState().jobs, firstJobs)
    assert.equal(useRuntimeStore.getState().jobs[0].progress?.current, 20)
  } finally {
    runtimeApi.listJobs = originalListJobs
    useCanvasOrchestratorStore.setState({ handleRuntimeJob: originalHandleRuntimeJob })
    useRuntimeStore.getState().clear()
  }
})

test('replaces stale process progress with a finalized manifest from the same output', async () => {
  const originalListJobs = runtimeApi.listJobs
  const completedJob: RuntimeJob = {
    ...observedJob,
    job_id: 'runtime_designer_manifest_done',
    status: 'completed',
    pid: 0,
    output_dir: 'C:/workspace/output/real_chunks',
    progress: { current: 200, total: 200, unit: 'seeds' }
  }
  runtimeApi.listJobs = async () => ({
    ok: true,
    data: { jobs: [completedJob], polled_at: '2026-07-09T23:00:00Z' }
  })
  useActivityStore.getState().clear()
  useRuntimeStore.getState().clear()
  useRuntimeStore.setState({
    jobs: [
      {
        ...observedJob,
        output_dir: 'C:\\workspace\\output\\real_chunks',
        progress: { current: 192, total: 200, unit: 'seeds' }
      }
    ]
  })

  try {
    await useRuntimeStore.getState().poll()

    const jobs = useRuntimeStore.getState().jobs
    assert.equal(jobs.length, 1)
    assert.equal(jobs[0].job_id, completedJob.job_id)
    assert.equal(jobs[0].status, 'completed')
    assert.equal(jobs[0].progress?.current, 200)
    assert.equal(useActivityStore.getState().events[0].type, 'designer:completed')
  } finally {
    runtimeApi.listJobs = originalListJobs
    useRuntimeStore.getState().clear()
    useActivityStore.getState().clear()
  }
})
