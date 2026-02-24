import { useState, useEffect } from 'react'
import {
  Menu,
  Plus,
  Settings,
  Sun,
  Moon,
  X,
  Keyboard,
  Home,
  Minus,
  Square,
  Copy
} from 'lucide-react'
import { useThemeStore, useUIStore, useTerminalStore } from '../../stores'

// Detect if running in Electron
const isElectron = typeof window !== 'undefined' && !!(window as any).bashgym?.window

export function NavigationBar() {
  const { theme, toggleTheme } = useThemeStore()
  const { toggleSidebar, overlayView, openOverlay, closeOverlay } = useUIStore()
  const { createTerminal } = useTerminalStore()
  const [isMaximized, setIsMaximized] = useState(false)

  // Track maximized state
  useEffect(() => {
    if (!isElectron) return
    const windowApi = (window as any).bashgym.window

    const checkMaximized = async () => {
      const maximized = await windowApi.isMaximized()
      setIsMaximized(maximized)
    }

    checkMaximized()

    // Re-check on resize (covers maximize/unmaximize/restore events)
    const handleResize = () => checkMaximized()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const handleAddTerminal = () => {
    createTerminal()
    // If we're on home, go to workspace when adding a terminal
    if (overlayView === 'home') {
      closeOverlay()
    }
  }

  const handleGoHome = () => {
    openOverlay('home')
  }

  const handleCloseOverlay = () => {
    // If on an overlay (not home), go back to workspace
    if (overlayView && overlayView !== 'home') {
      closeOverlay()
    }
  }

  // Get breadcrumb text for current view
  const getBreadcrumbText = () => {
    if (!overlayView) return 'Workspace'
    if (overlayView === 'home') return null
    if (overlayView === 'factory') return 'Data Factory'
    if (overlayView === 'huggingface') return 'HuggingFace'
    return overlayView.charAt(0).toUpperCase() + overlayView.slice(1)
  }

  const breadcrumb = getBreadcrumbText()

  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-border bg-background-card titlebar-drag">
      {/* Left Section */}
      <div className="flex items-center gap-3 titlebar-no-drag">
        {/* Menu Toggle */}
        <button
          onClick={toggleSidebar}
          className="btn-icon group"
          title="Toggle menu"
        >
          <Menu className="w-5 h-5 text-text-secondary group-hover:text-accent" />
        </button>

        {/* Home Button */}
        <button
          onClick={handleGoHome}
          className={`btn-icon ${
            overlayView === 'home'
              ? 'bg-accent/10 text-accent'
              : 'text-text-secondary'
          }`}
          title="Home"
        >
          <Home className="w-5 h-5" />
        </button>

        {/* Logo / Title - Clickable to go home */}
        <button
          onClick={handleGoHome}
          className="flex items-center gap-2"
        >
          <img src="/ghost-icon.png" alt="BashGym" className="w-6 h-6 object-cover" />
          <span className="font-brand font-semibold text-lg">
            <span className="text-accent">/</span>
            <span className="text-text-primary">BashGym</span>
          </span>
        </button>

        {/* Breadcrumb for current view */}
        {breadcrumb && (
          <div className="flex items-center font-brand text-lg">
            <span className="text-text-muted mx-1">/</span>
            <span className="text-text-secondary font-medium">{breadcrumb}</span>
          </div>
        )}
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-2 titlebar-no-drag">
        {/* Close overlay button - only show when in an overlay (not home or workspace) */}
        {overlayView && overlayView !== 'home' && (
          <button
            onClick={handleCloseOverlay}
            className="btn-icon text-text-secondary"
            title="Return to workspace (Escape)"
          >
            <X className="w-5 h-5" />
          </button>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="btn-icon text-text-secondary hover:text-accent"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode (Ctrl+D)`}
        >
          {theme === 'dark' ? (
            <Sun className="w-5 h-5" />
          ) : (
            <Moon className="w-5 h-5" />
          )}
        </button>

        {/* Add Terminal */}
        <button
          onClick={handleAddTerminal}
          className="btn-icon text-text-secondary hover:text-accent"
          title="Add terminal (Ctrl+N)"
        >
          <Plus className="w-5 h-5" />
        </button>

        {/* Keyboard Shortcuts */}
        <button
          onClick={() => useUIStore.getState().setKeyboardShortcutsOpen(true)}
          className="btn-icon text-text-secondary hover:text-accent"
          title="Keyboard shortcuts (Ctrl+?)"
        >
          <Keyboard className="w-5 h-5" />
        </button>

        {/* Settings */}
        <button
          onClick={() => useUIStore.getState().setSettingsOpen(true)}
          className="btn-icon text-text-secondary hover:text-accent"
          title="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>

        {/* Window Controls — Electron only */}
        {isElectron && (
          <>
            <div className="w-px h-5 bg-border mx-1" />
            <button
              onClick={() => (window as any).bashgym.window.minimize()}
              className="btn-icon text-text-secondary hover:text-accent"
              title="Minimize"
            >
              <Minus className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                (window as any).bashgym.window.maximize()
                // State will update via resize listener
              }}
              className="btn-icon text-text-secondary hover:text-accent"
              title={isMaximized ? 'Restore' : 'Maximize'}
            >
              {isMaximized ? (
                <Copy className="w-3.5 h-3.5" />
              ) : (
                <Square className="w-3.5 h-3.5" />
              )}
            </button>
            <button
              onClick={() => (window as any).bashgym.window.close()}
              className="btn-icon text-text-secondary hover:text-red-500 hover:bg-red-500/10"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </>
        )}
      </div>
    </header>
  )
}
