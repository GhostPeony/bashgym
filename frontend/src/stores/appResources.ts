import {
  systemInfoApi,
  providersApi,
  securityApi,
  sshApi,
  tracesApi,
  type SecurityDatasetInfo,
  type SystemInfo,
  type ModelRecommendations,
  type OllamaModelsResponse,
  type RepoInfo
} from '../services/api'
import { createKeyedSessionResource, createSessionResource } from './sessionResource'

/**
 * App-wide session resources: fetched once per app session, served from cache
 * on every later page mount, refreshed explicitly or via WebSocket events.
 * See sessionResource.ts for the caching contract.
 */

let forceHardwareRefresh = false

/** Hardware detection — detected once per app session. */
export const systemInfoResource = createSessionResource<SystemInfo>(() => {
  const force = forceHardwareRefresh
  forceHardwareRefresh = false
  return systemInfoApi.getInfo(force)
})

/** Re-run hardware detection on the backend (bypasses its 30s cache) and update the session cache. */
export function refreshSystemInfo(): Promise<void> {
  forceHardwareRefresh = true
  return systemInfoResource.getState().refresh()
}

/** Model recommendations keyed by compute device id ('' = local machine). */
export const modelRecommendationsResource = createKeyedSessionResource<ModelRecommendations>(
  (deviceId) => systemInfoApi.getRecommendations(deviceId || undefined)
)

/** Ollama server availability + installed models. */
export const ollamaStatusResource = createSessionResource<OllamaModelsResponse>(() =>
  providersApi.getOllamaModels()
)

export interface SshPreflightStatus {
  ok: boolean
  python_version?: string
  disk_free_gb?: number
  error?: string
  host?: string
  username?: string
}

/** Registered SSH training target preflight. */
export const sshPreflightResource = createSessionResource<SshPreflightStatus>(() =>
  sshApi.preflight()
)

/** Repos discovered across imported traces (shared by Traces and Training pages). */
export const traceReposResource = createSessionResource<RepoInfo[]>(() => tracesApi.listRepos())

/** Security fine-tuning datasets available to Training Config. */
export const securityDatasetsResource = createSessionResource<SecurityDatasetInfo[]>(() =>
  securityApi.listDatasets()
)

export interface TraceStatsData {
  timeline: { time: string; gold: number; failed: number; pending: number }[]
  totals: { gold: number; failed: number; pending: number; total: number }
}

/** Trace timeline stats keyed by time range ('24h' | '7d' | '30d' | 'all'). */
export const traceStatsResource = createKeyedSessionResource<TraceStatsData>((range) =>
  tracesApi.stats({ range })
)

export interface TraceCounts {
  gold: number
  silver: number
  bronze: number
  failed: number
  pending: number
}

/** Server-side trace status counts. Kept in sync by the traces page when it fetches. */
export const traceCountsResource = createSessionResource<TraceCounts>(async () => {
  const result = await tracesApi.list({ limit: 1, offset: 0 })
  if (result.ok && result.data && !Array.isArray(result.data) && result.data.counts) {
    return { ok: true, data: result.data.counts }
  }
  return { ok: false, error: result.error || 'Trace counts unavailable' }
})
