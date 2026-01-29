import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface CanvasControlState {
  // Global controls
  globalPaused: boolean

  // View options - what to show on nodes
  showMetrics: boolean
  showToolHistory: boolean
  showRecentFiles: boolean

  // Canvas settings
  gridEnabled: boolean
  snapToGrid: boolean
  showMiniMap: boolean

  // Panel state
  isPanelCollapsed: boolean

  // Actions
  setGlobalPaused: (paused: boolean) => void
  toggleGlobalPaused: () => void

  setShowMetrics: (show: boolean) => void
  setShowToolHistory: (show: boolean) => void
  setShowRecentFiles: (show: boolean) => void

  setGridEnabled: (enabled: boolean) => void
  setSnapToGrid: (snap: boolean) => void
  setShowMiniMap: (show: boolean) => void

  setPanelCollapsed: (collapsed: boolean) => void
  togglePanelCollapsed: () => void

  // Batch update
  updateSettings: (settings: Partial<Omit<CanvasControlState, 'updateSettings' | 'setGlobalPaused' | 'toggleGlobalPaused' | 'setShowMetrics' | 'setShowToolHistory' | 'setShowRecentFiles' | 'setGridEnabled' | 'setSnapToGrid' | 'setShowMiniMap' | 'setPanelCollapsed' | 'togglePanelCollapsed'>>) => void
}

export const useCanvasControlStore = create<CanvasControlState>()(
  persist(
    (set) => ({
      // Defaults
      globalPaused: false,
      showMetrics: true,
      showToolHistory: true,
      showRecentFiles: true,
      gridEnabled: true,
      snapToGrid: true,
      showMiniMap: true,
      isPanelCollapsed: false,

      // Actions
      setGlobalPaused: (paused) => set({ globalPaused: paused }),
      toggleGlobalPaused: () => set((state) => ({ globalPaused: !state.globalPaused })),

      setShowMetrics: (show) => set({ showMetrics: show }),
      setShowToolHistory: (show) => set({ showToolHistory: show }),
      setShowRecentFiles: (show) => set({ showRecentFiles: show }),

      setGridEnabled: (enabled) => set({ gridEnabled: enabled }),
      setSnapToGrid: (snap) => set({ snapToGrid: snap }),
      setShowMiniMap: (show) => set({ showMiniMap: show }),

      setPanelCollapsed: (collapsed) => set({ isPanelCollapsed: collapsed }),
      togglePanelCollapsed: () => set((state) => ({ isPanelCollapsed: !state.isPanelCollapsed })),

      updateSettings: (settings) => set(settings)
    }),
    {
      name: 'bashgym-canvas-controls',
      partialize: (state) => ({
        showMetrics: state.showMetrics,
        showToolHistory: state.showToolHistory,
        showRecentFiles: state.showRecentFiles,
        gridEnabled: state.gridEnabled,
        snapToGrid: state.snapToGrid,
        showMiniMap: state.showMiniMap,
        isPanelCollapsed: state.isPanelCollapsed
        // Don't persist globalPaused - always start unpaused
      })
    }
  )
)
