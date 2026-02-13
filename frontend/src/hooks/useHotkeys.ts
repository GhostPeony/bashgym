import { useEffect, useCallback } from 'react'
import { useTerminalStore, useThemeStore, useUIStore, useTrainingStore } from '../stores'

type HotkeyHandler = (event: KeyboardEvent) => void

interface HotkeyConfig {
  key: string
  ctrl?: boolean
  meta?: boolean
  alt?: boolean
  shift?: boolean
  handler: HotkeyHandler
}

export function useHotkeys(hotkeys: HotkeyConfig[]) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      for (const hotkey of hotkeys) {
        const ctrlMatch = hotkey.ctrl ? event.ctrlKey || event.metaKey : true
        const metaMatch = hotkey.meta ? event.metaKey : true
        const altMatch = hotkey.alt ? event.altKey : !event.altKey
        const shiftMatch = hotkey.shift ? event.shiftKey : !event.shiftKey
        const keyMatch = event.key.toLowerCase() === hotkey.key.toLowerCase()

        if (ctrlMatch && metaMatch && altMatch && shiftMatch && keyMatch) {
          event.preventDefault()
          hotkey.handler(event)
          return
        }
      }
    },
    [hotkeys]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

// Pre-configured global hotkeys hook
export function useGlobalHotkeys() {
  const { createTerminal, activePanelId, removePanel, panels, setActivePanel } = useTerminalStore()
  const { toggleTheme } = useThemeStore()
  const {
    closeOverlay,
    overlayView,
    toggleSidebar,
    setSettingsOpen,
    setKeyboardShortcutsOpen,
    isKeyboardShortcutsOpen,
    isSettingsOpen,
    isSidebarOpen,
    openOverlay,
    setCommandPaletteOpen
  } = useUIStore()
  const { currentRun, pauseTraining, resumeTraining } = useTrainingStore()

  useHotkeys([
    // === General ===

    // Ctrl+K: Command palette
    {
      key: 'k',
      ctrl: true,
      handler: () => {
        setCommandPaletteOpen(true)
      }
    },
    // Ctrl+,: Open settings
    {
      key: ',',
      ctrl: true,
      handler: () => {
        setSettingsOpen(!isSettingsOpen)
      }
    },
    // Ctrl+D: Toggle theme
    {
      key: 'd',
      ctrl: true,
      handler: () => {
        toggleTheme()
      }
    },
    // Ctrl+?: Show keyboard shortcuts
    {
      key: '/',
      ctrl: true,
      shift: true,
      handler: () => {
        setKeyboardShortcutsOpen(!isKeyboardShortcutsOpen)
      }
    },
    // Escape: Close overlay/modal
    {
      key: 'Escape',
      handler: () => {
        // Close modals first, then overlays
        if (isKeyboardShortcutsOpen) {
          setKeyboardShortcutsOpen(false)
        } else if (isSettingsOpen) {
          setSettingsOpen(false)
        } else if (overlayView) {
          closeOverlay()
        }
      }
    },

    // === Terminal ===

    // Ctrl+N: New terminal
    {
      key: 'n',
      ctrl: true,
      handler: () => {
        createTerminal()
      }
    },
    // Ctrl+W: Close active panel
    {
      key: 'w',
      ctrl: true,
      handler: () => {
        if (activePanelId) {
          removePanel(activePanelId)
        }
      }
    },
    // Ctrl+1-9: Focus terminal by number
    ...Array.from({ length: 9 }, (_, i) => ({
      key: String(i + 1),
      ctrl: true,
      handler: () => {
        if (panels[i]) {
          setActivePanel(panels[i].id)
        }
      }
    })),

    // === Navigation ===

    // Ctrl+B: Toggle sidebar
    {
      key: 'b',
      ctrl: true,
      handler: () => {
        toggleSidebar()
      }
    },
    // Ctrl+Shift+T: Training dashboard
    {
      key: 't',
      ctrl: true,
      shift: true,
      handler: () => {
        if (overlayView === 'training') {
          closeOverlay()
        } else {
          openOverlay('training')
        }
      }
    },
    // Ctrl+Shift+R: Router dashboard
    {
      key: 'r',
      ctrl: true,
      shift: true,
      handler: () => {
        if (overlayView === 'router') {
          closeOverlay()
        } else {
          openOverlay('router')
        }
      }
    },
    // Ctrl+Shift+F: Factory dashboard
    {
      key: 'f',
      ctrl: true,
      shift: true,
      handler: () => {
        if (overlayView === 'factory') {
          closeOverlay()
        } else {
          openOverlay('factory')
        }
      }
    },
    // Ctrl+Shift+E: Evaluator dashboard
    {
      key: 'e',
      ctrl: true,
      shift: true,
      handler: () => {
        if (overlayView === 'evaluator') {
          closeOverlay()
        } else {
          openOverlay('evaluator')
        }
      }
    },

    // === Training ===

    // Ctrl+Enter: Start training (no-op if already running)
    {
      key: 'Enter',
      ctrl: true,
      handler: () => {
        // Training start requires config — just navigate to training dashboard
        if (!currentRun) {
          openOverlay('training')
        }
      }
    },
    // Ctrl+Shift+P: Pause/resume training
    {
      key: 'p',
      ctrl: true,
      shift: true,
      handler: () => {
        if (currentRun?.status === 'running') {
          pauseTraining()
        } else if (currentRun?.status === 'paused') {
          resumeTraining()
        }
      }
    }
  ])
}
