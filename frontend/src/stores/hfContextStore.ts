import { create } from 'zustand'
import {
  hfApi,
  type HFContextBundle,
  type HFContextEvalPreview,
} from '../services/api'

interface WorkspaceContextState {
  bundles: HFContextBundle[]
  active: { bundle_id: string; version: number } | null
  collecting: HFContextBundle | null
  loading: boolean
  error: string | null
  errorCode: string | null
}

interface HFContextState {
  workspaces: Record<string, WorkspaceContextState>
  load: (workspaceId: string) => Promise<void>
  discover: (workspaceId: string, intent: string, task?: string, origin?: Record<string, string>) => Promise<HFContextBundle | null>
  pin: (workspaceId: string, bundle: HFContextBundle, selected: string[]) => Promise<HFContextBundle | null>
  refresh: (workspaceId: string, bundle: HFContextBundle) => Promise<HFContextBundle | null>
  cancel: (workspaceId: string, bundle: HFContextBundle) => Promise<HFContextBundle | null>
  activate: (workspaceId: string, bundle: HFContextBundle) => Promise<boolean>
  deactivate: (workspaceId: string) => Promise<boolean>
  projection: (workspaceId: string, bundle: HFContextBundle) => Promise<string>
  prepareEval: (workspaceId: string, bundle: HFContextBundle) => Promise<HFContextEvalPreview | null>
}

const emptyWorkspace = (): WorkspaceContextState => ({
  bundles: [],
  active: null,
  collecting: null,
  loading: false,
  error: null,
  errorCode: null,
})

export async function pollHFContextBundle(
  initial: HFContextBundle,
  fetchVersion: () => Promise<{ ok: boolean; data?: HFContextBundle; error?: string; code?: string }>,
  options: { attempts?: number; wait?: (milliseconds: number) => Promise<void> } = {},
): Promise<HFContextBundle> {
  if (initial.lifecycle === 'ready') return initial
  const attempts = options.attempts ?? 60
  const wait = options.wait ?? ((milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)))
  let current = initial
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    await wait(500)
    const response = await fetchVersion()
    if (!response.ok || !response.data) {
      const error = new Error(response.error || 'Unable to reconcile Hugging Face discovery') as Error & { code?: string }
      error.code = response.code
      throw error
    }
    current = response.data
    if (current.lifecycle === 'ready') return current
  }
  throw new Error('Hugging Face discovery is still collecting. You can close this view and return later.')
}

export const useHFContextStore = create<HFContextState>((set, get) => ({
  workspaces: {},

  load: async (workspaceId) => {
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), loading: true, error: null, errorCode: null },
      },
    }))
    const response = await hfApi.contextHistory(workspaceId)
    set((state) => ({
      workspaces: {
        ...state.workspaces,
        [workspaceId]: response.ok && response.data
          ? {
              bundles: response.data.bundles,
              active: response.data.active
                ? { bundle_id: response.data.active.bundle_id, version: response.data.active.version }
                : null,
              collecting: state.workspaces[workspaceId]?.collecting || null,
              loading: false,
              error: null,
              errorCode: null,
            }
          : { ...(state.workspaces[workspaceId] || emptyWorkspace()), loading: false, error: response.error || 'Unable to load Hugging Face context', errorCode: response.code || null },
      },
    }))
  },

  discover: async (workspaceId, intent, task, origin) => {
    const response = await hfApi.discoverContext({ workspace_id: workspaceId, intent, task, origin })
    if (!response.ok || !response.data) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Discovery failed', errorCode: response.code || null } } }))
      return null
    }
    set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), collecting: response.data!, error: null, errorCode: null } } }))
    try {
      const ready = await pollHFContextBundle(
        response.data,
        () => hfApi.contextVersion(workspaceId, response.data!.bundle_id, response.data!.version),
      )
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), collecting: null } } }))
      await get().load(workspaceId)
      return ready
    } catch (caught) {
      const error = caught as Error & { code?: string }
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: error.message, errorCode: error.code || null } } }))
      return null
    }
  },

  pin: async (workspaceId, bundle, selected) => {
    const response = await hfApi.pinContext(bundle.bundle_id, bundle.version, {
      workspace_id: workspaceId,
      expected_version: bundle.version,
      selected_evidence_ids: selected,
    })
    if (!response.ok || !response.data) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to pin context', errorCode: response.code || null } } }))
      return null
    }
    await get().load(workspaceId)
    return response.data
  },

  refresh: async (workspaceId, bundle) => {
    const response = await hfApi.refreshContext(workspaceId, bundle.bundle_id, bundle.version)
    if (!response.ok || !response.data) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to refresh context', errorCode: response.code || null } } }))
      return null
    }
    set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), collecting: response.data!, error: null, errorCode: null } } }))
    try {
      const ready = await pollHFContextBundle(
        response.data,
        () => hfApi.contextVersion(workspaceId, response.data!.bundle_id, response.data!.version),
      )
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), collecting: null } } }))
      await get().load(workspaceId)
      return ready
    } catch (caught) {
      const error = caught as Error & { code?: string }
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: error.message, errorCode: error.code || null } } }))
      return null
    }
  },

  cancel: async (workspaceId, bundle) => {
    const response = await hfApi.cancelContext(workspaceId, bundle.bundle_id, bundle.version)
    if (!response.ok || !response.data) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to cancel discovery', errorCode: response.code || null } } }))
      return null
    }
    set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), collecting: null, error: null, errorCode: null } } }))
    await get().load(workspaceId)
    return response.data
  },

  activate: async (workspaceId, bundle) => {
    const response = await hfApi.activateContext(workspaceId, bundle.bundle_id, bundle.version)
    if (!response.ok) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to activate context', errorCode: response.code || null } } }))
      return false
    }
    await get().load(workspaceId)
    return true
  },

  deactivate: async (workspaceId) => {
    const response = await hfApi.deactivateContext(workspaceId)
    if (!response.ok) {
      set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to deactivate context', errorCode: response.code || null } } }))
      return false
    }
    await get().load(workspaceId)
    return true
  },

  projection: async (workspaceId, bundle) => {
    const response = await hfApi.contextMarkdown(workspaceId, bundle.bundle_id, bundle.version)
    if (response.ok && response.data) return response.data.markdown
    set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to project active Hugging Face context', errorCode: response.code || null } } }))
    const error = new Error(response.error || 'Unable to project active Hugging Face context') as Error & { code?: string }
    error.code = response.code
    throw error
  },

  prepareEval: async (workspaceId, bundle) => {
    const response = await hfApi.prepareContextEval(workspaceId, bundle.bundle_id, bundle.version)
    if (response.ok && response.data) return response.data
    set((state) => ({ workspaces: { ...state.workspaces, [workspaceId]: { ...(state.workspaces[workspaceId] || emptyWorkspace()), error: response.error || 'Unable to prepare Eval preview', errorCode: response.code || null } } }))
    return null
  },
}))
