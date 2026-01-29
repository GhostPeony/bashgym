import {
  Menu,
  Plus,
  Settings,
  Sun,
  Moon,
  X,
  Keyboard,
  Home
} from 'lucide-react'
import { useThemeStore, useUIStore, useTerminalStore } from '../../stores'

export function NavigationBar() {
  const { theme, toggleTheme } = useThemeStore()
  const { toggleSidebar, overlayView, openOverlay, closeOverlay } = useUIStore()
  const { createTerminal } = useTerminalStore()

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
    <header className="h-12 flex items-center justify-between px-4 border-b border-border-subtle bg-background-secondary/80 backdrop-blur-xl titlebar-drag">
      {/* Left Section */}
      <div className="flex items-center gap-3 titlebar-no-drag">
        {/* Menu Toggle */}
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          title="Toggle menu"
        >
          <Menu className="w-5 h-5 text-text-secondary" />
        </button>

        {/* Home Button */}
        <button
          onClick={handleGoHome}
          className={`p-2 rounded-lg transition-colors ${
            overlayView === 'home'
              ? 'bg-primary/10 text-primary'
              : 'hover:bg-background-tertiary text-text-secondary'
          }`}
          title="Home"
        >
          <Home className="w-5 h-5" />
        </button>

        {/* Logo / Title - Clickable to go home */}
        <button
          onClick={handleGoHome}
          className="flex items-center gap-2 hover:opacity-80 transition-opacity"
        >
          <img src="/ghost-icon.png" alt="BashGym" className="w-6 h-6 rounded-lg object-cover" />
          <span className="font-semibold" style={{ color: '#70A6C1' }}>/BashGym</span>
        </button>

        {/* Breadcrumb for current view */}
        {breadcrumb && (
          <div className="flex items-center">
            <span className="text-text-muted mx-1">/</span>
            <span className="text-text-secondary">{breadcrumb}</span>
          </div>
        )}
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-2 titlebar-no-drag">
        {/* Close overlay button - only show when in an overlay (not home or workspace) */}
        {overlayView && overlayView !== 'home' && (
          <button
            onClick={handleCloseOverlay}
            className="p-2 rounded-lg hover:bg-background-tertiary transition-colors text-text-secondary hover:text-text-primary"
            title="Return to workspace (Escape)"
          >
            <X className="w-5 h-5" />
          </button>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode (Ctrl+D)`}
        >
          {theme === 'dark' ? (
            <Sun className="w-5 h-5 text-text-secondary" />
          ) : (
            <Moon className="w-5 h-5 text-text-secondary" />
          )}
        </button>

        {/* Add Terminal */}
        <button
          onClick={handleAddTerminal}
          className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          title="Add terminal (Ctrl+N)"
        >
          <Plus className="w-5 h-5 text-text-secondary" />
        </button>

        {/* Keyboard Shortcuts */}
        <button
          onClick={() => useUIStore.getState().setKeyboardShortcutsOpen(true)}
          className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          title="Keyboard shortcuts (Ctrl+?)"
        >
          <Keyboard className="w-5 h-5 text-text-secondary" />
        </button>

        {/* Settings */}
        <button
          onClick={() => useUIStore.getState().setSettingsOpen(true)}
          className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          title="Settings"
        >
          <Settings className="w-5 h-5 text-text-secondary" />
        </button>
      </div>
    </header>
  )
}
