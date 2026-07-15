import { create } from 'zustand'

export type ViewMode = 'home' | 'workspace' | 'training' | 'autoresearch' | 'router' | 'traces' | 'factory' | 'evaluator' | 'guardrails' | 'profiler' | 'models' | 'huggingface' | 'integration' | 'achievements' | 'orchestrator' | 'pipeline' | 'download'

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

  // Actions
  setView: (view: ViewMode) => void
  openOverlay: (view: ViewMode) => void
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

export const useUIStore = create<UIState>((set) => ({
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

  setView: (view) => set({ currentView: view, overlayView: null }),

  openOverlay: (view) => set({ overlayView: view }),

  closeOverlay: () => set({ overlayView: null }),

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
