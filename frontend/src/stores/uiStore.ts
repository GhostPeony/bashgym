import { create } from 'zustand'

export type ViewMode = 'home' | 'workspace' | 'training' | 'autoresearch' | 'router' | 'traces' | 'factory' | 'evaluator' | 'guardrails' | 'profiler' | 'models' | 'huggingface' | 'integration' | 'achievements' | 'orchestrator' | 'pipeline' | 'download'
export type TrainingSubview = 'runs' | 'autoresearch'

export interface TrainingSelection {
  workspaceId: string | null
  campaignId: string | null
}

/** What the left sidebar renders: the nav menu or the Agent Sessions feed */
export type SidebarMode = 'nav' | 'sessions'

export interface PanelPresentationRequest {
  workspaceId: string
  panelId: string
  presentation: 'focus' | 'peek'
  requestedAt: number
}

interface UIState {
  // Navigation
  currentView: ViewMode
  isSidebarOpen: boolean
  sidebarMode: SidebarMode
  isSettingsOpen: boolean
  isCommandPaletteOpen: boolean
  isOnboardingOpen: boolean
  isKeyboardShortcutsOpen: boolean
  isAgentChatOpen: boolean
  isAgentStreamOpen: boolean
  panelPresentationRequest: PanelPresentationRequest | null

  // Overlays - terminals persist behind these
  overlayView: ViewMode | null
  trainingSubview: TrainingSubview
  trainingSelection: TrainingSelection

  // Actions
  setView: (view: ViewMode) => void
  openOverlay: (view: ViewMode) => void
  openTraining: (
    subview?: TrainingSubview,
    selection?: Partial<TrainingSelection>,
    historyMode?: 'push' | 'replace',
  ) => void
  hydrateNavigationFromUrl: (search?: string) => void
  closeOverlay: () => void
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setSidebarMode: (mode: SidebarMode) => void
  setSettingsOpen: (open: boolean) => void
  setCommandPaletteOpen: (open: boolean) => void
  setOnboardingOpen: (open: boolean) => void
  setKeyboardShortcutsOpen: (open: boolean) => void
  setAgentChatOpen: (open: boolean) => void
  toggleAgentChat: () => void
  setAgentStreamOpen: (open: boolean) => void
  toggleAgentStream: () => void
  presentWorkspacePanel: (
    workspaceId: string,
    panelId: string,
    presentation?: PanelPresentationRequest['presentation']
  ) => void
  clearPanelPresentationRequest: (requestedAt: number) => void
}

function canonicalTrainingUrl(
  subview: TrainingSubview,
  selection: TrainingSelection,
): string | null {
  if (typeof window === 'undefined') return null
  const params = new URLSearchParams()
  params.set('view', 'training')
  params.set('tab', subview)
  if (subview === 'autoresearch') {
    if (selection.workspaceId) params.set('workspace_id', selection.workspaceId)
    if (selection.campaignId) params.set('campaign_id', selection.campaignId)
  }
  return `${window.location.pathname}?${params.toString()}${window.location.hash || ''}`
}

function canonicalViewUrl(view: ViewMode | 'workspace'): string | null {
  if (typeof window === 'undefined') return null
  const params = new URLSearchParams()
  params.set('view', view)
  return `${window.location.pathname}?${params.toString()}${window.location.hash || ''}`
}

const CANONICAL_OVERLAY_VIEWS = new Set<ViewMode>([
  'home',
  'router',
  'traces',
  'factory',
  'evaluator',
  'guardrails',
  'profiler',
  'models',
  'huggingface',
  'integration',
  'achievements',
  'orchestrator',
  'pipeline',
  'download',
])

function normalizedId(value: string | null): string | null {
  const normalized = value?.trim() || ''
  return /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(normalized) ? normalized : null
}

export const useUIStore = create<UIState>((set, get) => ({
  currentView: 'home',
  isSidebarOpen: true,
  sidebarMode: 'nav',
  isSettingsOpen: false,
  isCommandPaletteOpen: false,
  isOnboardingOpen: false,
  isKeyboardShortcutsOpen: false,
  isAgentChatOpen: false,
  isAgentStreamOpen: false,
  panelPresentationRequest: null,
  overlayView: 'home',
  trainingSubview: 'runs',
  trainingSelection: { workspaceId: null, campaignId: null },

  setView: (view) => set({ currentView: view, overlayView: null }),

  openOverlay: (view) => {
    if (view === 'workspace') {
      get().closeOverlay()
      return
    }
    if (view === 'autoresearch') {
      get().openTraining('autoresearch', undefined, 'replace')
      return
    }
    if (view === 'training') {
      get().openTraining('runs')
      return
    }
    set({ overlayView: view })
    const target = canonicalViewUrl(view)
    if (target && typeof window !== 'undefined') {
      window.history.pushState({}, '', target)
    }
  },

  openTraining: (subview = 'runs', selection, historyMode = 'push') => {
    const prior = get().trainingSelection
    const nextSelection = subview === 'runs'
      ? { workspaceId: null, campaignId: null }
      : {
          workspaceId: selection?.workspaceId ?? prior.workspaceId,
          campaignId: selection?.campaignId ?? prior.campaignId,
        }
    set({
      overlayView: 'training',
      trainingSubview: subview,
      trainingSelection: nextSelection,
    })
    const target = canonicalTrainingUrl(subview, nextSelection)
    if (target && typeof window !== 'undefined') {
      window.history[historyMode === 'replace' ? 'replaceState' : 'pushState']({}, '', target)
    }
  },

  hydrateNavigationFromUrl: (search) => {
    if (typeof window === 'undefined' && search === undefined) return
    const params = new URLSearchParams(search ?? window.location.search)
    const view = params.get('view')
    if (view !== 'training' && view !== 'autoresearch') {
      if (view === 'workspace') {
        set({ overlayView: null })
      } else if (view && CANONICAL_OVERLAY_VIEWS.has(view as ViewMode)) {
        set({ overlayView: view as ViewMode })
      } else {
        set({ overlayView: 'home' })
      }
      return
    }
    const legacy = view === 'autoresearch'
    const subview: TrainingSubview = legacy || params.get('tab') === 'autoresearch'
      ? 'autoresearch'
      : 'runs'
    const selection: TrainingSelection = subview === 'autoresearch'
      ? {
          workspaceId: normalizedId(params.get('workspace_id')),
          campaignId: normalizedId(params.get('campaign_id')),
        }
      : { workspaceId: null, campaignId: null }
    set({
      overlayView: 'training',
      trainingSubview: subview,
      trainingSelection: selection,
    })
    if (legacy && typeof window !== 'undefined') {
      const target = canonicalTrainingUrl(subview, selection)
      if (target) window.history.replaceState({}, '', target)
    }
  },

  closeOverlay: () => {
    set({ overlayView: null })
    const target = canonicalViewUrl('workspace')
    if (target && typeof window !== 'undefined') {
      window.history.pushState({}, '', target)
    }
  },

  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

  setSidebarOpen: (open) => set({ isSidebarOpen: open }),

  setSidebarMode: (mode) => set({ sidebarMode: mode, isSidebarOpen: true }),

  setSettingsOpen: (open) => set({ isSettingsOpen: open }),

  setCommandPaletteOpen: (open) => set({ isCommandPaletteOpen: open }),

  setOnboardingOpen: (open) => set({ isOnboardingOpen: open }),

  setKeyboardShortcutsOpen: (open) => set({ isKeyboardShortcutsOpen: open }),

  setAgentChatOpen: (open) => set({ isAgentChatOpen: open }),

  toggleAgentChat: () => set((state) => ({ isAgentChatOpen: !state.isAgentChatOpen })),

  setAgentStreamOpen: (open) => set({ isAgentStreamOpen: open }),

  toggleAgentStream: () => set((state) => ({ isAgentStreamOpen: !state.isAgentStreamOpen })),

  presentWorkspacePanel: (workspaceId, panelId, presentation = 'focus') => set({
    panelPresentationRequest: { workspaceId, panelId, presentation, requestedAt: Date.now() }
  }),

  clearPanelPresentationRequest: (requestedAt) => set((state) => (
    state.panelPresentationRequest?.requestedAt === requestedAt
      ? { panelPresentationRequest: null }
      : state
  )),
}))
