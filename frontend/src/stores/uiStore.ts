import { create } from 'zustand'

export type ViewMode = 'home' | 'workspace' | 'training' | 'router' | 'traces' | 'factory' | 'evaluator' | 'guardrails' | 'profiler' | 'models' | 'huggingface' | 'integration' | 'achievements'

interface UIState {
  // Navigation
  currentView: ViewMode
  isSidebarOpen: boolean
  isSettingsOpen: boolean
  isCommandPaletteOpen: boolean
  isOnboardingOpen: boolean
  isKeyboardShortcutsOpen: boolean

  // Overlays - terminals persist behind these
  overlayView: ViewMode | null

  // Actions
  setView: (view: ViewMode) => void
  openOverlay: (view: ViewMode) => void
  closeOverlay: () => void
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setSettingsOpen: (open: boolean) => void
  setCommandPaletteOpen: (open: boolean) => void
  setOnboardingOpen: (open: boolean) => void
  setKeyboardShortcutsOpen: (open: boolean) => void
}

export const useUIStore = create<UIState>((set) => ({
  currentView: 'home',
  isSidebarOpen: false,
  isSettingsOpen: false,
  isCommandPaletteOpen: false,
  isOnboardingOpen: false,
  isKeyboardShortcutsOpen: false,
  overlayView: 'home',

  setView: (view) => set({ currentView: view, overlayView: null }),

  openOverlay: (view) => set({ overlayView: view }),

  closeOverlay: () => set({ overlayView: null }),

  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

  setSidebarOpen: (open) => set({ isSidebarOpen: open }),

  setSettingsOpen: (open) => set({ isSettingsOpen: open }),

  setCommandPaletteOpen: (open) => set({ isCommandPaletteOpen: open }),

  setOnboardingOpen: (open) => set({ isOnboardingOpen: open }),

  setKeyboardShortcutsOpen: (open) => set({ isKeyboardShortcutsOpen: open })
}))
