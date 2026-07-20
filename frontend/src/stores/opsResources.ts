import {
  hooksApi,
  integrationApi,
  observabilityApi,
  pipelineApi,
  providersApi,
  routerApi,
  settingsApi,
  systemApi,
  trainingApi,
  type DatasetInspectReport,
  type EnvKeyStatus,
  type HooksStatus,
  type IntegrationSettings,
  type IntegrationStatus,
  type ModelVersion,
  type ObservabilityMetrics,
  type PendingTrace,
  type PipelineConfig,
  type PipelineStatus,
  type ProvidersResponse,
  type RouterConfigResponse,
  type RouterStats,
  type RunMetricPoint,
  type SystemStats,
  type ToolStat,
  type TraceSummary,
  type TrainingRunSummary,
} from '../services/api'
import { createKeyedSessionResource, createSessionResource } from './sessionResource'

/**
 * Ops-page session resources: router, pipeline, integration, profiler,
 * settings, home, and training detail views. Cached per app session so these
 * pages render instantly on remount. See sessionResource.ts for the contract.
 */

/** Teacher/Student router request stats. */
export const routerStatsResource = createSessionResource<RouterStats>(() => routerApi.getStats())

/** Router config including the active Student model. */
export const routerConfigResource = createSessionResource<RouterConfigResponse>(() =>
  routerApi.getConfig()
)

export interface PipelineOverview {
  status: PipelineStatus
  config: PipelineConfig
}

/** Auto-import pipeline status + config, fetched together. */
export const pipelineOverviewResource = createSessionResource<PipelineOverview>(async () => {
  const [statusRes, configRes] = await Promise.all([
    pipelineApi.getStatus(),
    pipelineApi.getConfig(),
  ])
  if (statusRes.ok && statusRes.data && configRes.ok && configRes.data) {
    return { ok: true, data: { status: statusRes.data, config: configRes.data } }
  }
  return { ok: false, error: statusRes.error || configRes.error || 'Failed to load pipeline' }
})

export interface IntegrationOverview {
  status: IntegrationStatus | null
  settings: IntegrationSettings | null
  modelVersions: ModelVersion[]
  pendingTraces: PendingTrace[]
}

/** Bashbros integration status, settings, model versions, and pending traces. */
export const integrationOverviewResource = createSessionResource<IntegrationOverview>(async () => {
  const [statusRes, settingsRes, modelsRes, tracesRes] = await Promise.all([
    integrationApi.getStatus(),
    integrationApi.getSettings(),
    integrationApi.listModelVersions(),
    integrationApi.listPendingTraces(),
  ])
  if (!statusRes.ok && !settingsRes.ok && !modelsRes.ok && !tracesRes.ok) {
    return { ok: false, error: statusRes.error || 'Failed to load integration data' }
  }
  return {
    ok: true,
    data: {
      status: statusRes.ok ? statusRes.data ?? null : null,
      settings: settingsRes.ok ? settingsRes.data ?? null : null,
      modelVersions: modelsRes.ok ? modelsRes.data ?? [] : [],
      pendingTraces: tracesRes.ok ? tracesRes.data ?? [] : [],
    },
  }
})

/** Aggregated profiler + guardrail metrics. */
export const profilerMetricsResource = createSessionResource<ObservabilityMetrics>(() =>
  observabilityApi.getMetrics()
)

/** Most recent execution traces (profiler). */
export const profilerTracesResource = createSessionResource<TraceSummary[]>(async () => {
  const result = await observabilityApi.listTraces(50, 0)
  if (result.ok && result.data) {
    return { ok: true, data: result.data.traces }
  }
  return { ok: false, error: result.error || 'Failed to fetch traces' }
})

/** Per-tool call performance stats. */
export const profilerToolStatsResource = createSessionResource<ToolStat[]>(() =>
  observabilityApi.getToolStats()
)

/** Provider API key statuses (masked values only — never raw keys). */
export const envKeysResource = createSessionResource<EnvKeyStatus[]>(async () => {
  const result = await settingsApi.getEnvKeys()
  if (result.ok && result.data) {
    return { ok: true, data: result.data.keys }
  }
  return { ok: false, error: result.error || 'Failed to fetch API keys' }
})

/** Trace-capture hook install status per AI coding tool. */
export const hooksStatusResource = createSessionResource<HooksStatus>(() => hooksApi.getStatus())

/** Inference provider availability summary. */
export const providersResource = createSessionResource<ProvidersResponse>(() =>
  providersApi.getProviders()
)

/** System-wide trace/model counts for the home screen. */
export const systemStatsResource = createSessionResource<SystemStats>(() => systemApi.stats())

export interface CheckpointInfo {
  id: string
  run_id: string
  kind: 'final' | 'merged' | 'intermediate' | 'gguf'
  path: string
  size_mb: number
  created_at: string
  base_model: string | null
}

/** Saved training checkpoints on disk. */
export const checkpointsResource = createSessionResource<CheckpointInfo[]>(() =>
  trainingApi.listCheckpoints()
)

export interface TrainingRunOption {
  id: string
  status?: string
  strategy?: string
}

/** Recent training runs for the log viewer's run selector. */
export const trainingRunOptionsResource = createSessionResource<TrainingRunOption[]>(async () => {
  const result = await trainingApi.list(undefined, 50)
  if (result.ok && Array.isArray(result.data)) {
    const list = (result.data as unknown as Array<Record<string, unknown>>)
      .map((r) => ({
        id: String(r.run_id ?? r.id ?? ''),
        status: r.status as string | undefined,
        strategy: r.strategy as string | undefined,
      }))
      .filter((r) => r.id)
    return { ok: true, data: list }
  }
  return { ok: false, error: result.error || 'Failed to load runs' }
})

export interface TrainingRunLog {
  run_id: string
  path: string
  total_lines: number
  truncated: boolean
  lines: string[]
}

export function trainingLogKey(runId: string, tail: number): string {
  return JSON.stringify([runId, tail])
}

/** Tail of a run's training.log, keyed by [runId, tail]. */
export const trainingLogResource = createKeyedSessionResource<TrainingRunLog>((key) => {
  const [runId, tail] = JSON.parse(key) as [string, number]
  return trainingApi.getLog(runId, { tail })
})

export function datasetInspectKey(offset: number, limit: number): string {
  return JSON.stringify([offset, limit])
}

/** Exported-dataset inspection pages, keyed by [offset, limit]. */
export const datasetInspectResource = createKeyedSessionResource<DatasetInspectReport>((key) => {
  const [offset, limit] = JSON.parse(key) as [number, number]
  return trainingApi.inspectDataset(offset, limit)
})

/** Persisted runs that recorded metrics.jsonl (for run comparison). */
export const metricRunsResource = createSessionResource<TrainingRunSummary[]>(async () => {
  const result = await trainingApi.listRuns()
  if (result.ok && result.data) {
    return { ok: true, data: result.data.runs.filter((r) => r.has_metrics) }
  }
  return { ok: false, error: result.error || 'Failed to load runs' }
})

/** Loss-curve points per run id (points without a numeric loss/step are dropped). */
export const runMetricsResource = createKeyedSessionResource<RunMetricPoint[]>(async (runId) => {
  const result = await trainingApi.getRunMetrics(runId)
  if (result.ok && result.data) {
    return {
      ok: true,
      data: result.data.metrics.filter(
        (m) => typeof m.loss === 'number' && typeof m.step === 'number'
      ),
    }
  }
  return { ok: false, error: result.error || 'Failed to load metrics' }
})
