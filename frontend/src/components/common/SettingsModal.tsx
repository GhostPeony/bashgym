import { useState, useEffect } from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { useUIStore, useAccentStore, ACCENT_PRESETS } from '../../stores'
import { HooksSection, ModelsSection } from '../settings'
import { Settings, Cpu, Terminal, Info, Keyboard, Moon, Sun, Monitor, Dices, RotateCcw } from 'lucide-react'

type SettingsTab = 'general' | 'models' | 'capture' | 'terminal' | 'api'
type Theme = 'light' | 'dark' | 'system'

export function SettingsModal() {
  const { isSettingsOpen, setSettingsOpen } = useUIStore()
  const { accentHue, setAccentHue, randomizeHue, resetHue } = useAccentStore()
  const [activeTab, setActiveTab] = useState<SettingsTab>('general')
  const [theme, setTheme] = useState<Theme>('dark')

  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [wsUrl, setWsUrl] = useState('ws://localhost:8000/ws')

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
            Save
          </Button>
        </>
      }
    >
      <div className="flex gap-6 h-[450px]">
        {/* Sidebar Tabs */}
        <div className="w-36 flex-shrink-0 space-y-1 border-r border-border pr-4">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 text-sm font-mono font-semibold tracking-wide transition-colors relative ${
                activeTab === tab.id
                  ? 'menu-item-active rounded-brutal'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-secondary rounded-brutal'
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
              {/* Appearance Section */}
              <div>
                <div className="tag mb-4">
                  <span>APPEARANCE</span>
                </div>

                {/* Theme Selector */}
                <div className="space-y-3 mb-6">
                  <label className="text-xs font-mono uppercase tracking-widest text-text-secondary">Theme</label>
                  <div className="flex gap-2">
                    {[
                      { value: 'light' as const, icon: Sun, label: 'Light' },
                      { value: 'dark' as const, icon: Moon, label: 'Dark' },
                      { value: 'system' as const, icon: Monitor, label: 'System' },
                    ].map(({ value, icon: Icon, label }) => (
                      <button
                        key={value}
                        onClick={() => handleThemeChange(value)}
                        className={`flex items-center gap-2 px-4 py-2 font-mono text-sm font-semibold border-brutal transition-press rounded-brutal ${
                          theme === value
                            ? 'border-border bg-accent-light text-accent-dark shadow-brutal-sm'
                            : 'border-border-subtle text-text-secondary hover:border-border hover:text-text-primary'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span>{label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Accent Color Slider */}
                <div className="space-y-3">
                  <label className="text-xs font-mono uppercase tracking-widest text-text-secondary">Accent Color</label>

                  {/* Slider Row */}
                  <div className="flex items-center gap-3">
                    <button
                      onClick={randomizeHue}
                      className="btn-icon w-8 h-8 flex-shrink-0"
                      title="Random color"
                    >
                      <Dices className="w-4 h-4" />
                    </button>

                    <input
                      type="range"
                      min="0"
                      max="360"
                      value={accentHue}
                      onChange={(e) => setAccentHue(Number(e.target.value))}
                      className="hue-slider flex-1"
                    />

                    <button
                      onClick={resetHue}
                      className="btn-icon w-8 h-8 flex-shrink-0"
                      title="Reset to Wisteria"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Preset Chips */}
                  <div className="flex flex-wrap gap-2 mt-2">
                    {ACCENT_PRESETS.map(preset => (
                      <button
                        key={preset.name}
                        onClick={() => setAccentHue(preset.hue)}
                        className={`flex items-center gap-2 px-3 py-1.5 text-xs font-mono font-semibold border-brutal rounded-brutal transition-press ${
                          accentHue === preset.hue
                            ? 'border-border shadow-brutal-sm'
                            : 'border-border-subtle hover:border-border'
                        }`}
                      >
                        <span
                          className="w-3 h-3 rounded-full border border-border"
                          style={{ backgroundColor: `hsl(${preset.hue}, 30%, 66%)` }}
                        />
                        {preset.name}
                      </button>
                    ))}
                  </div>

                  {/* Live Preview Swatches */}
                  <div className="flex items-center gap-3 mt-3">
                    <span className="text-xs font-mono text-text-muted uppercase tracking-widest">Preview:</span>
                    <div className="flex gap-2">
                      <div
                        className="w-8 h-8 border-brutal border-border rounded-brutal"
                        style={{ backgroundColor: `hsl(${accentHue}, 30%, 66%)` }}
                        title="Accent"
                      />
                      <div
                        className="w-8 h-8 border-brutal border-border rounded-brutal"
                        style={{ backgroundColor: `hsl(${accentHue}, 35%, 80%)` }}
                        title="Light"
                      />
                      <div
                        className="w-8 h-8 border-brutal border-border rounded-brutal"
                        style={{ backgroundColor: `hsl(${accentHue}, 30%, 47%)` }}
                        title="Dark"
                      />
                    </div>
                    <span className="text-xs font-mono text-text-muted">{accentHue}deg</span>
                  </div>
                </div>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="section-divider" />
              <div>
                <div className="tag mb-4">
                  <span>SHORTCUTS</span>
                </div>
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
                      <kbd className="px-2 py-0.5 font-mono text-xs border-brutal border-border bg-background-secondary rounded-brutal text-text-muted">
                        {shortcut.key}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>

              {/* About */}
              <div className="section-divider" />
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-brand text-text-primary">BashGym</p>
                  <p className="text-xs text-text-muted">A Self-Improving Agentic Development Gym</p>
                </div>
                <span className="text-xs font-mono text-text-muted">v1.0.0</span>
              </div>
            </div>
          )}

          {/* Models Tab */}
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
              <div className="tag mb-4">
                <span>API</span>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-mono uppercase tracking-widest text-text-secondary mb-2">API URL</label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="input w-full text-sm"
                    placeholder="http://localhost:8000"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono uppercase tracking-widest text-text-secondary mb-2">WebSocket URL</label>
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
                Changes require a page reload to take effect.
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}
