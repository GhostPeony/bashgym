import { useCallback, useEffect } from 'react'
import { create, type StoreApi, type UseBoundStore } from 'zustand'

/**
 * Session-scoped data cache for page-level fetches.
 *
 * Pages unmount on navigation (MainLayout renders overlays conditionally), so
 * any fetch lifecycle held in component state re-runs on every tab switch.
 * A session resource moves that lifecycle into a zustand store that survives
 * navigation: the first mount fetches, later mounts render the cached data
 * immediately, and refreshes happen in the background without blanking the UI.
 *
 * Contract:
 * - `ensureLoaded()` fetches only when there is no usable cached data
 *   (never loaded, or explicitly invalidated, or older than `staleAfterMs`).
 *   Concurrent calls share one in-flight request.
 * - `loading` is true only while fetching with no cached data — the only time
 *   a page may show a full spinner.
 * - `refresh()` refetches while keeping cached data on screen (`refreshing`).
 * - `invalidate()` marks the cache stale; the next `ensureLoaded()` refetches
 *   in the background if data exists, or as a first load if not.
 * - A failed background refresh keeps the stale data and surfaces `error`.
 */

export interface ResourceResult<T> {
  ok: boolean
  data?: T
  error?: string
}

export type ResourceFetcher<T> = () => Promise<ResourceResult<T>>

export interface SessionResourceOptions {
  /** Revalidate in the background when cached data is older than this. Default: never. */
  staleAfterMs?: number
}

export interface SessionResourceState<T> {
  data: T | null
  error: string | null
  loading: boolean
  refreshing: boolean
  loadedAt: number | null
  ensureLoaded: () => Promise<void>
  refresh: () => Promise<void>
  invalidate: () => void
  setData: (data: T) => void
}

export type SessionResourceStore<T> = UseBoundStore<StoreApi<SessionResourceState<T>>>

export function createSessionResource<T>(
  fetcher: ResourceFetcher<T>,
  options: SessionResourceOptions = {}
): SessionResourceStore<T> {
  let inFlight: Promise<void> | null = null
  let invalidated = false

  return create<SessionResourceState<T>>((set, get) => {
    const doFetch = (): Promise<void> => {
      if (inFlight) return inFlight
      const hasData = get().data !== null
      set({ loading: !hasData, refreshing: hasData })
      inFlight = (async () => {
        try {
          const result = await fetcher()
          if (result.ok && result.data !== undefined) {
            invalidated = false
            set({ data: result.data, error: null, loadedAt: Date.now() })
          } else {
            set({ error: result.error || 'Request failed' })
          }
        } catch (err) {
          set({ error: err instanceof Error ? err.message : String(err) })
        } finally {
          inFlight = null
          set({ loading: false, refreshing: false })
        }
      })()
      return inFlight
    }

    return {
      data: null,
      error: null,
      loading: false,
      refreshing: false,
      loadedAt: null,

      ensureLoaded: () => {
        if (inFlight) return inFlight
        const { data, loadedAt } = get()
        const stale =
          options.staleAfterMs !== undefined &&
          loadedAt !== null &&
          Date.now() - loadedAt >= options.staleAfterMs
        if (data !== null && !invalidated && !stale) return Promise.resolve()
        return doFetch()
      },

      refresh: () => doFetch(),

      invalidate: () => {
        invalidated = true
      },

      setData: (data) => {
        invalidated = false
        set({ data, error: null, loadedAt: Date.now() })
      }
    }
  })
}

/**
 * Subscribe to a session resource and ensure it is loaded on mount.
 * Remounts with cached data render immediately (`loading` stays false).
 */
export function useSessionResource<T>(store: SessionResourceStore<T>): SessionResourceState<T> {
  const state = store()
  const ensureLoaded = store.getState().ensureLoaded
  useEffect(() => {
    void ensureLoaded()
  }, [ensureLoaded])
  return state
}

/**
 * Keyed variant for fetches parameterized by a string key (a time range, a
 * device id, a filter signature). Each key caches independently.
 */
export interface KeyedEntry<T> {
  data: T | null
  error: string | null
  loading: boolean
  refreshing: boolean
  loadedAt: number | null
}

export interface KeyedSessionResourceState<T> {
  entries: Record<string, KeyedEntry<T>>
  ensureLoaded: (key: string) => Promise<void>
  refresh: (key: string) => Promise<void>
  invalidate: (key?: string) => void
  setData: (key: string, data: T) => void
}

export type KeyedSessionResourceStore<T> = UseBoundStore<StoreApi<KeyedSessionResourceState<T>>>

export type KeyedResourceFetcher<T> = (key: string) => Promise<ResourceResult<T>>

const EMPTY_ENTRY = {
  data: null,
  error: null,
  loading: false,
  refreshing: false,
  loadedAt: null
}

export function createKeyedSessionResource<T>(
  fetcher: KeyedResourceFetcher<T>,
  options: SessionResourceOptions = {}
): KeyedSessionResourceStore<T> {
  const inFlight = new Map<string, Promise<void>>()
  const invalidatedKeys = new Set<string>()

  return create<KeyedSessionResourceState<T>>((set, get) => {
    const patchEntry = (key: string, patch: Partial<KeyedEntry<T>>) => {
      set((state) => ({
        entries: {
          ...state.entries,
          [key]: { ...(state.entries[key] || EMPTY_ENTRY), ...patch }
        }
      }))
    }

    const doFetch = (key: string): Promise<void> => {
      const existing = inFlight.get(key)
      if (existing) return existing
      const hasData = (get().entries[key]?.data ?? null) !== null
      patchEntry(key, { loading: !hasData, refreshing: hasData })
      const promise = (async () => {
        try {
          const result = await fetcher(key)
          if (result.ok && result.data !== undefined) {
            invalidatedKeys.delete(key)
            patchEntry(key, { data: result.data, error: null, loadedAt: Date.now() })
          } else {
            patchEntry(key, { error: result.error || 'Request failed' })
          }
        } catch (err) {
          patchEntry(key, { error: err instanceof Error ? err.message : String(err) })
        } finally {
          inFlight.delete(key)
          patchEntry(key, { loading: false, refreshing: false })
        }
      })()
      inFlight.set(key, promise)
      return promise
    }

    return {
      entries: {},

      ensureLoaded: (key) => {
        const existing = inFlight.get(key)
        if (existing) return existing
        const entry = get().entries[key]
        const stale =
          options.staleAfterMs !== undefined &&
          entry?.loadedAt != null &&
          Date.now() - entry.loadedAt >= options.staleAfterMs
        if (entry && entry.data !== null && !invalidatedKeys.has(key) && !stale) {
          return Promise.resolve()
        }
        return doFetch(key)
      },

      refresh: (key) => doFetch(key),

      invalidate: (key) => {
        if (key === undefined) {
          for (const k of Object.keys(get().entries)) invalidatedKeys.add(k)
        } else {
          invalidatedKeys.add(key)
        }
      },

      setData: (key, data) => {
        invalidatedKeys.delete(key)
        patchEntry(key, { data, error: null, loadedAt: Date.now() })
      }
    }
  })
}

/**
 * Subscribe to one key of a keyed session resource and ensure it is loaded.
 */
export function useKeyedSessionResource<T>(
  store: KeyedSessionResourceStore<T>,
  key: string
): KeyedEntry<T> & { refresh: () => Promise<void> } {
  const entry = store((state) => state.entries[key]) || (EMPTY_ENTRY as KeyedEntry<T>)
  const ensureLoaded = store.getState().ensureLoaded
  const refreshByKey = store.getState().refresh
  const refresh = useCallback(() => refreshByKey(key), [refreshByKey, key])
  useEffect(() => {
    void ensureLoaded(key)
  }, [ensureLoaded, key])
  return { ...entry, refresh }
}
