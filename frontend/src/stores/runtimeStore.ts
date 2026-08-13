import { create } from 'zustand'
import { runtimeApi, type RuntimeJob } from '../services/api'
import { useActivityStore } from './activityStore'
import { useCanvasOrchestratorStore } from './canvasOrchestratorStore'

export interface ObservedRuntimeJob extends RuntimeJob {
  observed_completed_at?: string
}

interface RuntimeState {
  jobs: ObservedRuntimeJob[]
  error: string | null
  lastPolledAt: string | null
  poll: () => Promise<void>
  clear: () => void
}

let pollInFlight = false
const missingPolls = new Map<string, number>()
const COMPLETION_CONFIRMATION_POLLS = 3

function sameRuntimeJob(left: ObservedRuntimeJob, right: ObservedRuntimeJob): boolean {
  return left === right || JSON.stringify(left) === JSON.stringify(right)
}

function sameRuntimeJobs(left: ObservedRuntimeJob[], right: ObservedRuntimeJob[]): boolean {
  return (
    left === right ||
    (left.length === right.length && left.every((job, index) => sameRuntimeJob(job, right[index])))
  )
}

function normalizedOutputDir(job: ObservedRuntimeJob): string | undefined {
  if (!job.output_dir) return undefined
  return job.output_dir.replaceAll('\\', '/').replace(/\/$/, '').toLowerCase()
}

function activityPayload(job: ObservedRuntimeJob): Record<string, unknown> {
  return {
    ...(job.kind === 'training' ? { run_id: job.job_id } : { job_id: job.job_id }),
    pid: job.pid,
    strategy: job.strategy,
    pipeline: job.pipeline,
    progress: job.progress,
    output_dir: job.output_dir,
    source: job.source
  }
}

function notifyDiscovered(job: ObservedRuntimeJob): void {
  const type = job.kind === 'training' ? 'training:queued' : 'designer:queued'
  useActivityStore.getState().addEvent(type, activityPayload(job))
}

function notifyCompleted(job: ObservedRuntimeJob): void {
  const type = job.kind === 'training' ? 'training:complete' : 'designer:completed'
  useActivityStore.getState().addEvent(type, activityPayload(job))
}

function retractCompletion(job: ObservedRuntimeJob): void {
  const type = job.kind === 'training' ? 'training:complete' : 'designer:completed'
  useActivityStore.getState().removeEvent(`${type}:${job.job_id}`)
}

export const useRuntimeStore = create<RuntimeState>((set, get) => ({
  jobs: [],
  error: null,
  lastPolledAt: null,

  poll: async () => {
    if (pollInFlight) return
    pollInFlight = true
    try {
      const response = await runtimeApi.listJobs()
      if (!response.ok || !response.data) {
        set({ error: response.error || 'Runtime observer unavailable' })
        return
      }

      const previous = get().jobs
      const previousById = new Map(previous.map((job) => [job.job_id, job]))
      const responseIds = new Set(response.data.jobs.map((job) => job.job_id))
      const activeIds = new Set(
        response.data.jobs.filter((job) => job.status === 'running').map((job) => job.job_id)
      )
      const completedOutputDirs = new Set(
        response.data.jobs
          .filter((job) => job.status === 'completed')
          .map(normalizedOutputDir)
          .filter((value): value is string => Boolean(value))
      )
      const pendingMissing: ObservedRuntimeJob[] = []
      const completedNow: ObservedRuntimeJob[] = []

      for (const job of previous) {
        if (job.status !== 'running' || activeIds.has(job.job_id)) continue
        const outputDir = normalizedOutputDir(job)
        if (outputDir && completedOutputDirs.has(outputDir)) {
          missingPolls.delete(job.job_id)
          continue
        }
        const misses = (missingPolls.get(job.job_id) ?? 0) + 1
        if (misses < COMPLETION_CONFIRMATION_POLLS) {
          missingPolls.set(job.job_id, misses)
          pendingMissing.push(job)
          continue
        }
        missingPolls.delete(job.job_id)
        completedNow.push({
          ...job,
          status: 'completed',
          observed_completed_at: new Date().toISOString()
        })
      }

      for (const job of response.data.jobs) {
        missingPolls.delete(job.job_id)
        const previousJob = previousById.get(job.job_id)
        if (job.status === 'running') {
          if (previousJob?.status === 'completed') retractCompletion(job)
          if (!previousJob) notifyDiscovered(job)
        } else if (!previousJob || previousJob.status !== 'completed') {
          notifyCompleted(job)
        }
        if (!previousJob || !sameRuntimeJob(previousJob, job)) {
          useCanvasOrchestratorStore.getState().handleRuntimeJob(job)
        }
      }
      for (const job of completedNow) {
        notifyCompleted(job)
        useCanvasOrchestratorStore.getState().handleRuntimeJob(job)
      }

      const retainedCompleted = previous.filter(
        (job) => job.status === 'completed' && !responseIds.has(job.job_id)
      )
      const nextById = new Map<string, ObservedRuntimeJob>()
      for (const job of [
        ...response.data.jobs,
        ...pendingMissing,
        ...completedNow,
        ...retainedCompleted
      ]) {
        if (!nextById.has(job.job_id)) nextById.set(job.job_id, job)
      }
      const nextJobs = Array.from(nextById.values()).slice(0, 30)
      set({
        jobs: sameRuntimeJobs(previous, nextJobs) ? previous : nextJobs,
        error: null,
        lastPolledAt: response.data.polled_at
      })
    } finally {
      pollInFlight = false
    }
  },

  clear: () => {
    missingPolls.clear()
    set({ jobs: [], error: null, lastPolledAt: null })
  }
}))
