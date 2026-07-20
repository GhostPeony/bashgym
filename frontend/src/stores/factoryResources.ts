import {
  dataQualityApi,
  designerApi,
  environmentApi,
  evaluatorApi,
  factoryApi,
  modelsApi,
  observabilityApi,
  sourcesApi,
  syntheticApi,
  tracesApi,
  type DataQualityDefaults,
  type DesignerModel,
  type DesignerPipelineInfo,
  type EnvironmentPipelinesResponse,
  type EvaluationResponse,
  type FactoryConfig,
  type GuardrailEvent,
  type GuardrailStats,
  type ModelSummary,
  type ObservabilitySettings,
  type SourceCatalogResponse,
  type SourceRecommendation,
  type SourceUse,
  type SynthesisJob,
  type SyntheticJobStatus,
  type SyntheticPreset,
  type TraceAnalytics
} from '../services/api'
import { createKeyedSessionResource, createSessionResource } from './sessionResource'

/**
 * Session resources for the Factory, Evaluator, Guardrails, and Traces
 * analytics pages. See sessionResource.ts for the caching contract.
 */

/** Data factory schema/seed config. Kept in sync via setData after a successful save. */
export const factoryConfigResource = createSessionResource<FactoryConfig>(() =>
  factoryApi.getConfig()
)

/** Schema-based synthesis jobs. */
export const factoryJobsResource = createSessionResource<SynthesisJob[]>(() =>
  factoryApi.listJobs()
)

/** Models available to the factory default-model picker. */
export const factoryModelsResource = createSessionResource<
  { id: string; name: string; provider: string }[]
>(() => factoryApi.listModels())

/** AI-powered synthetic generation jobs (shared by the Create and Jobs tabs). */
export const syntheticJobsResource = createSessionResource<SyntheticJobStatus[]>(() =>
  syntheticApi.listJobs()
)

/** Synthetic generation target-size presets. */
export const syntheticPresetsResource = createSessionResource<Record<string, SyntheticPreset>>(() =>
  syntheticApi.getPresets()
)

/** Data Designer pipeline catalog + install availability. */
export const designerPipelinesResource = createSessionResource<{
  pipelines: DesignerPipelineInfo[]
  available: boolean
}>(() => designerApi.listPipelines())

/** Live teacher-model catalog for Data Designer. */
export const designerModelsResource = createSessionResource<{
  models: DesignerModel[]
  provider_models: string[]
  available: boolean
}>(() => designerApi.listModels())

/** Environment Lab pipeline availability + external dataset sources. */
export const environmentPipelinesResource = createSessionResource<EnvironmentPipelinesResponse>(
  () => environmentApi.pipelines()
)

/** Public source-card catalog. */
export const sourceCatalogResource = createSessionResource<SourceCatalogResponse>(() =>
  sourcesApi.list()
)

/** Source recommendations keyed by JSON.stringify([domain, goal, includeEvalOnly]). */
export const sourceRecommendationsResource = createKeyedSessionResource<SourceRecommendation[]>(
  async (key) => {
    const [domain, goal, includeEvalOnly] = JSON.parse(key) as [string, SourceUse, boolean]
    const result = await sourcesApi.recommend({
      domain: domain || undefined,
      goal,
      include_eval_only: includeEvalOnly
    })
    return result.ok && result.data
      ? { ok: true, data: result.data.recommendations }
      : { ok: false, error: result.error || 'Recommendation request failed' }
  }
)

/** Trace-quality defaults for decision-DPO mining. */
export const dataQualityDefaultsResource = createSessionResource<DataQualityDefaults>(() =>
  dataQualityApi.defaults()
)

/** Registered trained models (Evaluator and Held-out gate model pickers). */
export const registeredModelsResource = createSessionResource<{
  models: ModelSummary[]
  total: number
}>(() => modelsApi.list())

/** Benchmark evaluation job history. */
export const evaluationsResource = createSessionResource<EvaluationResponse[]>(() =>
  evaluatorApi.list()
)

/** Observability settings (guardrail toggles). */
export const observabilitySettingsResource = createSessionResource<ObservabilitySettings>(() =>
  observabilityApi.getSettings()
)

/** Guardrail event statistics. */
export const guardrailStatsResource = createSessionResource<GuardrailStats>(() =>
  observabilityApi.getGuardrailStats()
)

/** Guardrail activity log keyed by JSON.stringify([action, checkType]). */
export const guardrailEventsResource = createKeyedSessionResource<GuardrailEvent[]>(async (key) => {
  const [action, checkType] = JSON.parse(key) as [string | null, string | null]
  const result = await observabilityApi.listGuardrailEvents({
    action: action ?? undefined,
    check_type: checkType ?? undefined,
    limit: 100
  })
  return result.ok && result.data
    ? { ok: true, data: result.data.events }
    : { ok: false, error: result.error || 'Failed to fetch guardrail events' }
})

/** Aggregated trace analytics (Traces page analytics view). */
export const traceAnalyticsResource = createSessionResource<TraceAnalytics>(() =>
  tracesApi.getAnalytics()
)
