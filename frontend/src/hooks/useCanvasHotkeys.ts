import { useEffect, useCallback } from 'react'
import { useTerminalStore, useCanvasControlStore } from '../stores'

export interface UseCanvasHotkeysOptions {
  enabled?: boolean
  onFitView?: () => void
  onFocusPanel?: (panelId: string) => void
}

export function useCanvasHotkeys({
  enabled = true,
  onFitView,
  onFocusPanel
}: UseCanvasHotkeysOptions = {}) {
  const {
    panels,
    activePanelId,
    setActivePanel,
    sessions,
    updateSession
  } = useTerminalStore()

  const {
    gridEnabled,
    showMiniMap,
    setGridEnabled,
    setShowMiniMap,
    toggleGlobalPaused
  } = useCanvasControlStore()

  // Focus session by number (1-9)
  const focusSessionByNumber = useCallback((num: number) => {
    const panelArray = Array.from(panels)
    if (num > 0 && num <= panelArray.length) {
      const panel = panelArray[num - 1]
      setActivePanel(panel.id)
      onFocusPanel?.(panel.id)
    }
  }, [panels, setActivePanel, onFocusPanel])

  // Cycle through sessions with Tab
  const cycleSession = useCallback((reverse: boolean = false) => {
    const panelArray = Array.from(panels)
    if (panelArray.length === 0) return

    const currentIndex = panelArray.findIndex(p => p.id === activePanelId)
    let nextIndex: number

    if (reverse) {
      nextIndex = currentIndex <= 0 ? panelArray.length - 1 : currentIndex - 1
    } else {
      nextIndex = currentIndex >= panelArray.length - 1 ? 0 : currentIndex + 1
    }

    const nextPanel = panelArray[nextIndex]
    setActivePanel(nextPanel.id)
    onFocusPanel?.(nextPanel.id)
  }, [panels, activePanelId, setActivePanel, onFocusPanel])

  // Toggle grid
  const toggleGrid = useCallback(() => {
    setGridEnabled(!gridEnabled)
  }, [gridEnabled, setGridEnabled])

  // Toggle minimap
  const toggleMinimap = useCallback(() => {
    setShowMiniMap(!showMiniMap)
  }, [showMiniMap, setShowMiniMap])

  // Pause/resume focused agent
  const toggleFocusedAgentPause = useCallback(() => {
    if (!activePanelId) return

    const panel = panels.find(p => p.id === activePanelId)
    if (!panel?.terminalId) return

    const session = sessions.get(panel.terminalId)
    if (!session) return

    updateSession(panel.terminalId, { isPaused: !session.isPaused })
  }, [activePanelId, panels, sessions, updateSession])

  // Handle keydown events
  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      // Ignore if typing in an input, textarea, or contenteditable
      const target = event.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return
      }

      // Check for number keys 1-9 (without modifiers for canvas shortcuts)
      if (!event.ctrlKey && !event.altKey && !event.metaKey) {
        const num = parseInt(event.key, 10)
        if (num >= 1 && num <= 9) {
          event.preventDefault()
          focusSessionByNumber(num)
          return
        }
      }

      // Tab - cycle through sessions (Shift+Tab for reverse)
      if (event.key === 'Tab' && !event.ctrlKey && !event.altKey && !event.metaKey) {
        event.preventDefault()
        cycleSession(event.shiftKey)
        return
      }

      // F - fit view (without modifiers)
      if (event.key.toLowerCase() === 'f' && !event.ctrlKey && !event.altKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault()
        onFitView?.()
        return
      }

      // G - toggle grid
      if (event.key.toLowerCase() === 'g' && !event.ctrlKey && !event.altKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault()
        toggleGrid()
        return
      }

      // M - toggle minimap
      if (event.key.toLowerCase() === 'm' && !event.ctrlKey && !event.altKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault()
        toggleMinimap()
        return
      }

      // Space - pause/resume focused agent
      if (event.key === ' ' && !event.ctrlKey && !event.altKey && !event.metaKey) {
        event.preventDefault()
        toggleFocusedAgentPause()
        return
      }

      // P - pause/resume all (with Shift)
      if (event.key.toLowerCase() === 'p' && event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
        event.preventDefault()
        toggleGlobalPaused()
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    enabled,
    focusSessionByNumber,
    cycleSession,
    toggleGrid,
    toggleMinimap,
    toggleFocusedAgentPause,
    toggleGlobalPaused,
    onFitView
  ])

  return {
    focusSessionByNumber,
    cycleSession,
    toggleGrid,
    toggleMinimap,
    toggleFocusedAgentPause
  }
}

// Canvas shortcuts reference for documentation
export const CANVAS_SHORTCUTS = [
  { keys: ['1-9'], description: 'Focus session by number' },
  { keys: ['Tab'], description: 'Cycle through sessions' },
  { keys: ['Shift', 'Tab'], description: 'Cycle sessions (reverse)' },
  { keys: ['F'], description: 'Fit view to all nodes' },
  { keys: ['G'], description: 'Toggle grid' },
  { keys: ['M'], description: 'Toggle minimap' },
  { keys: ['Space'], description: 'Pause/resume focused agent' },
  { keys: ['Shift', 'P'], description: 'Pause/resume all agents' }
]
