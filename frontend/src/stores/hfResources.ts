import {
  API_BASE,
  hfApi,
  type HFJob,
  type HFMyModel,
  type HFSpace,
  type HFStatus,
} from '../services/api'
import {
  createKeyedSessionResource,
  createSessionResource,
  type ResourceResult,
  type SessionResourceStore,
} from './sessionResource'

/**
 * HuggingFace session resources: fetched once per app session, served from
 * cache on every later mount of the HF dashboard tabs and status indicator.
 * See sessionResource.ts for the caching contract.
 */

/** Account/token status shared by HFDashboard and the HFStatus nav indicator. */
export const hfStatusResource = createSessionResource<HFStatus>(() => hfApi.getStatus())

/** Cloud training jobs. */
export const hfJobsResource = createSessionResource<HFJob[]>(() => hfApi.listJobs())

/** ZeroGPU Spaces. */
export const hfSpacesResource = createSessionResource<HFSpace[]>(() => hfApi.listSpaces())

/** The user's models on the Hub. */
export const hfMyModelsResource = createSessionResource<HFMyModel[]>(() => hfApi.listMyModels())

/** Hub dataset ids keyed by name-prefix filter. */
export const hfDatasetsResource = createKeyedSessionResource<string[]>((prefix) =>
  hfApi.listDatasets(prefix || undefined)
)

export interface HFBucket {
  id: string
  private: boolean
  created_at: string
  updated_at: string
}

export interface HFBucketItem {
  name: string
  type: string
  size?: number
  last_modified?: string
}

/** Storage buckets. */
export const hfBucketsResource = createSessionResource<HFBucket[]>(() => hfApi.listBuckets())

/** Bucket file tree keyed by bucket id ('' = no bucket selected). */
export const hfBucketTreeResource = createKeyedSessionResource<HFBucketItem[]>((bucketId) =>
  bucketId ? hfApi.listBucketTree(bucketId) : Promise.resolve({ ok: true, data: [] })
)

export interface HFTraceDataset {
  id: string
  private: boolean | null
  downloads: number
  last_modified: string
}

/** Trace datasets on the Hub. */
export const hfTraceDatasetsResource = createSessionResource<HFTraceDataset[]>(() =>
  hfApi.listTraceDatasets()
)

async function researchGet<T>(path: string): Promise<ResourceResult<T>> {
  try {
    const res = await fetch(`${API_BASE}${path}`)
    const text = await res.text()
    try {
      return { ok: res.ok, data: JSON.parse(text) as T }
    } catch {
      return { ok: false, error: text }
    }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}

export interface ResearchOverview {
  report: string | null
  empirical: string | null
  cacheStats: { cached_datasets: number } | null
}

/**
 * Dataset research reports + cache stats. Parts that fail keep their previous
 * value (the page has always rendered missing reports as empty states).
 */
export const hfResearchResource: SessionResourceStore<ResearchOverview> =
  createSessionResource<ResearchOverview>(async () => {
    const [reportRes, empiricalRes, cacheRes] = await Promise.all([
      researchGet<{ content: string }>('/research/report'),
      researchGet<{ content: string }>('/research/empirical'),
      researchGet<{ cached_datasets: number }>('/research/cache/stats'),
    ])
    const prev = hfResearchResource.getState().data
    return {
      ok: true,
      data: {
        report: reportRes.ok && reportRes.data ? reportRes.data.content : prev?.report ?? null,
        empirical:
          empiricalRes.ok && empiricalRes.data ? empiricalRes.data.content : prev?.empirical ?? null,
        cacheStats: cacheRes.ok && cacheRes.data ? cacheRes.data : prev?.cacheStats ?? null,
      },
    }
  })
