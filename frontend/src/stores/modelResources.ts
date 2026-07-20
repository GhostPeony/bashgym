import {
  modelsApi,
  type LeaderboardEntry,
  type ModelProfile,
  type ModelSummary,
  type TrendDataPoint,
} from '../services/api'
import { createKeyedSessionResource, createSessionResource } from './sessionResource'

/**
 * Models-domain session resources: cached per app session so the Models pages
 * render instantly on remount. See sessionResource.ts for the caching contract.
 */

export interface ModelListFilters {
  strategy: string
  status: string
  starred: boolean
  sortBy: string
  sortOrder: 'asc' | 'desc'
}

export function modelListKey(filters: ModelListFilters): string {
  return JSON.stringify([
    filters.strategy,
    filters.status,
    filters.starred,
    filters.sortBy,
    filters.sortOrder,
  ])
}

/** Model list keyed by server-side filter signature (search is client-side). */
export const modelListResource = createKeyedSessionResource<{
  models: ModelSummary[]
  total: number
}>((key) => {
  const [strategy, status, starred, sortBy, sortOrder] = JSON.parse(key) as [
    string,
    string,
    boolean,
    string,
    'asc' | 'desc',
  ]
  return modelsApi.list({
    strategy: strategy || undefined,
    status: status || undefined,
    starred: starred || undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: 50,
  })
})

/** Leaderboard by custom eval pass rate (fixed metric, top 20). */
export const modelLeaderboardResource = createSessionResource<LeaderboardEntry[]>(async () => {
  const result = await modelsApi.leaderboard('custom_eval_pass_rate', 20)
  if (result.ok && result.data) {
    return { ok: true, data: result.data.entries }
  }
  return { ok: false, error: result.error || 'Failed to load leaderboard' }
})

export function modelTrendsKey(metric: string, days: number): string {
  return JSON.stringify([metric, days])
}

/** Trend data points keyed by [metric, days]. */
export const modelTrendsResource = createKeyedSessionResource<TrendDataPoint[]>(async (key) => {
  const [metric, days] = JSON.parse(key) as [string, number]
  const result = await modelsApi.trends(metric, days)
  if (result.ok && result.data) {
    return { ok: true, data: result.data.data }
  }
  return { ok: false, error: result.error || 'Failed to load trend data' }
})

/** Full model profile keyed by model id. */
export const modelProfileResource = createKeyedSessionResource<ModelProfile>((modelId) =>
  modelsApi.get(modelId)
)

export function modelComparisonKey(modelIds: string[]): string {
  return JSON.stringify(modelIds)
}

/** Profiles for a comparison set, keyed by the JSON list of model ids. */
export const modelComparisonResource = createKeyedSessionResource<Record<string, ModelProfile>>(
  async (key) => {
    const modelIds = JSON.parse(key) as string[]
    const profiles: Record<string, ModelProfile> = {}
    for (const modelId of modelIds) {
      const result = await modelsApi.get(modelId)
      if (result.ok && result.data) {
        profiles[modelId] = result.data
      }
    }
    if (modelIds.length > 0 && Object.keys(profiles).length === 0) {
      return { ok: false, error: 'Failed to load any models' }
    }
    return { ok: true, data: profiles }
  }
)

/** All models for the lineage tree (larger page size than the browser list). */
export const modelLineageResource = createSessionResource<ModelSummary[]>(async () => {
  const result = await modelsApi.list({ limit: 100, sort_by: 'created_at', sort_order: 'desc' })
  if (result.ok && result.data) {
    return { ok: true, data: result.data.models }
  }
  return { ok: false, error: result.error || 'Failed to load models' }
})
