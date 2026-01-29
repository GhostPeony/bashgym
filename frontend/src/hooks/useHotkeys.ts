import { useEffect, useCallback } from 'react'
import { useTerminalStore, useThemeStore, useUIStore } from '../stores'

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
  const { createTerminal, activePanelId, removePanel } = useTerminalStore()
  const { toggleTheme } = useThemeStore()
  const { closeOverlay, overlayView } = useUIStore()

  useHotkeys([
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
    // Ctrl+D: Toggle theme
    {
      key: 'd',
      ctrl: true,
      handler: () => {
        toggleTheme()
      }
    },
    // Escape: Close overlay
    {
      key: 'Escape',
      handler: () => {
        if (overlayView) {
          closeOverlay()
        }
      }
    }
  ])
}
