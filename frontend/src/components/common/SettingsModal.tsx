import { useState, useEffect } from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { useUIStore } from '../../stores'
import { HooksSection, ModelsSection } from '../settings'
import { Settings, Cpu, Terminal, Keyboard, Info, Moon, Sun, Monitor } from 'lucide-react'

type SettingsTab = 'general' | 'models' | 'capture' | 'terminal' | 'api'
type Theme = 'light' | 'dark' | 'system'

export function SettingsModal() {
  const { isSettingsOpen, setSettingsOpen } = useUIStore()
  const [activeTab, setActiveTab] = useState<SettingsTab>('general')
  const [theme, setTheme] = useState<Theme>('dark')

  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [wsUrl, setWsUrl] = useState('ws://localhost:8000/ws')

  // Load saved settings
  useEffect(() => {
    const savedTheme = localStorage.getItem('ghost-gym-theme') as Theme
    if (savedTheme) setTheme(savedTheme)

    const savedApiUrl = localStorage.getItem('ghost-gym-api-url')
    if (savedApiUrl) setApiUrl(savedApiUrl)

    const savedWsUrl = localStorage.getItem('ghost-gym-ws-url')
    if (savedWsUrl) setWsUrl(savedWsUrl)
  }, [])

  const handleSave = () => {
    localStorage.setItem('ghost-gym-theme', theme)
    localStorage.setItem('ghost-gym-api-url', apiUrl)
    localStorage.setItem('ghost-gym-ws-url', wsUrl)
    setSettingsOpen(false)
  }

  const handleThemeChange = (newTheme: Theme) => {
    setTheme(newTheme)
    // Apply theme immediately
    if (newTheme === 'dark' || (newTheme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  const tabs = [
    { id: 'general' as const, label: 'General', icon: Settings },
    { id: 'models' as const, label: 'Models', icon: Cpu },
    { id: 'capture' as const, label: 'Trace Capture', icon: Terminal },
    { id: 'api' as const, label: 'API', icon: Info },
  ]

  return (
    <Modal
      isOpen={isSettingsOpen}
      onClose={() => setSettingsOpen(false)}
      title="Settings"
      description="Configure BashGym"
      size="xl"
      footer={
        <>
          <Button variant="secondary" onClick={() => setSettingsOpen(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSave}>
            Save Changes
          </Button>
        </>
      }
    >
      <div className="flex gap-6 h-[450px]">
        {/* Sidebar Tabs */}
        <div className="w-36 flex-shrink-0 space-y-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-primary/10 text-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 overflow-y-auto overflow-x-hidden">
          {/* General Tab */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-medium text-text-primary mb-4">Appearance</h3>
                <div className="space-y-3">
                  <label className="text-xs text-text-secondary">Theme</label>
                  <div className="flex gap-2">
                    {[
                      { value: 'light' as const, icon: Sun, label: 'Light' },
                      { value: 'dark' as const, icon: Moon, label: 'Dark' },
                      { value: 'system' as const, icon: Monitor, label: 'System' },
                    ].map(({ value, icon: Icon, label }) => (
                      <button
                        key={value}
                        onClick={() => handleThemeChange(value)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                          theme === value
                            ? 'border-primary bg-primary/10 text-primary'
                            : 'border-border-subtle text-text-secondary hover:border-border-color hover:text-text-primary'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span className="text-sm">{label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="pt-4 border-t border-border-subtle">
                <h4 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                  <Keyboard className="w-4 h-4" />
                  Keyboard Shortcuts
                </h4>
                <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                  {[
                    { label: 'New Terminal', key: 'Ctrl+N' },
                    { label: 'Close Panel', key: 'Ctrl+W' },
                    { label: 'Toggle Theme', key: 'Ctrl+D' },
                    { label: 'Close Overlay', key: 'Escape' },
                    { label: 'Command Palette', key: 'Ctrl+K' },
                    { label: 'Settings', key: 'Ctrl+,' },
                  ].map(shortcut => (
                    <div key={shortcut.key} className="flex items-center justify-between py-1">
                      <span className="text-text-secondary">{shortcut.label}</span>
                      <kbd className="px-2 py-0.5 rounded bg-background-tertiary border border-border-subtle text-text-muted text-xs">
                        {shortcut.key}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>

              {/* About */}
              <div className="pt-4 border-t border-border-subtle">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-text-primary">BashGym</p>
                    <p className="text-xs text-text-muted">A Self-Improving Agentic Development Gym</p>
                  </div>
                  <span className="text-xs text-text-muted">v1.0.0</span>
                </div>
              </div>
            </div>
          )}

          {/* Models Tab - always mounted for background loading */}
          <div className={activeTab === 'models' ? 'h-full' : 'hidden'}>
            <ModelsSection />
          </div>

          {/* Trace Capture Tab */}
          {activeTab === 'capture' && (
            <HooksSection />
          )}

          {/* API Tab */}
          {activeTab === 'api' && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-text-primary">API Configuration</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-text-secondary mb-2">API URL</label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="input w-full text-sm"
                    placeholder="http://localhost:8000"
                  />
                </div>
                <div>
                  <label className="block text-xs text-text-secondary mb-2">WebSocket URL</label>
                  <input
                    type="text"
                    value={wsUrl}
                    onChange={(e) => setWsUrl(e.target.value)}
                    className="input w-full text-sm"
                    placeholder="ws://localhost:8000/ws"
                  />
                </div>
              </div>
              <p className="text-xs text-text-muted">
                Configure the backend API endpoints. Changes require a page reload to take effect.
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}
