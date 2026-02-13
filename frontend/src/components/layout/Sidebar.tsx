import { useState, useEffect } from 'react'
import {
  Home,
  Terminal,
  Sparkles,
  BarChart3,
  Settings,
  CheckCircle,
  XCircle,
  Wifi,
  WifiOff,
  ChevronRight,
  FileStack,
  Layers,
  FlaskConical,
  GitBranch,
  Shield,
  Activity,
  Cloud,
  Link2,
  Trophy
} from 'lucide-react'
import { useUIStore, useTrainingStore } from '../../stores'
import { hooksApi, systemApi } from '../../services/api'
import { clsx } from 'clsx'

interface MenuItemProps {
  icon: React.ReactNode
  label: string
  onClick: () => void
  active?: boolean
  badge?: string | number
  primary?: boolean
}

function MenuItem({ icon, label, onClick, active, badge, primary }: MenuItemProps) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-brutal transition-colors text-left',
        active
          ? 'menu-item-active rounded-brutal'
          : 'text-text-secondary hover:text-text-primary hover:bg-background-secondary',
        primary && 'font-medium'
      )}
    >
      <span className="flex-shrink-0">{icon}</span>
      <span className={clsx('flex-1 text-sm', primary && 'font-medium')}>{label}</span>
      {badge !== undefined && (
        <span className="font-mono text-xs border border-border bg-accent-light text-accent-dark px-2 py-0.5 rounded-brutal">
          {badge}
        </span>
      )}
    </button>
  )
}

function MenuSection({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      {title && (
        <h3 className="px-3 py-2 font-mono text-xs uppercase tracking-widest text-text-muted">
          {title}
        </h3>
      )}
      <div className="space-y-1">{children}</div>
    </div>
  )
}

type SecondaryViewId = 'traces' | 'models' | 'evaluator' | 'router' | 'guardrails' | 'profiler' | 'huggingface' | 'integration' | 'achievements'

interface CollapsibleSectionProps {
  title: string
  items: Array<{ id: SecondaryViewId; icon: React.ReactNode; label: string }>
  defaultExpanded?: boolean
}

function CollapsibleSection({ title, items, defaultExpanded = false }: CollapsibleSectionProps) {
  const { overlayView, openOverlay, setSidebarOpen } = useUIStore()
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  const handleClick = (id: SecondaryViewId) => {
    if (overlayView === id) {
      openOverlay('home')
    } else {
      openOverlay(id)
    }
  }

  const hasActiveItem = items.some(item => overlayView === item.id)

  return (
    <div className="mb-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={clsx(
          'w-full flex items-center gap-2 px-3 py-2 font-mono text-xs uppercase tracking-widest transition-colors',
          hasActiveItem ? 'text-accent-dark' : 'text-text-muted hover:text-text-secondary'
        )}
      >
        <ChevronRight className={clsx('w-3 h-3 transition-transform', isExpanded && 'rotate-90')} />
        {title}
      </button>
      {isExpanded && (
        <div className="space-y-1 mt-1">
          {items.map((item) => (
            <MenuItem
              key={item.id}
              icon={item.icon}
              label={item.label}
              onClick={() => handleClick(item.id)}
              active={overlayView === item.id}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function SecondarySections() {
  const libraryItems: CollapsibleSectionProps['items'] = [
    { id: 'traces', icon: <FileStack className="w-4 h-4" />, label: 'Traces' },
    { id: 'models', icon: <Layers className="w-4 h-4" />, label: 'Models' }
  ]

  const toolsItems: CollapsibleSectionProps['items'] = [
    { id: 'evaluator', icon: <FlaskConical className="w-4 h-4" />, label: 'Evaluator' },
    { id: 'router', icon: <GitBranch className="w-4 h-4" />, label: 'Router' },
    { id: 'guardrails', icon: <Shield className="w-4 h-4" />, label: 'Guardrails' },
    { id: 'profiler', icon: <Activity className="w-4 h-4" />, label: 'Profiler' }
  ]

  const connectionsItems: CollapsibleSectionProps['items'] = [
    { id: 'huggingface', icon: <Cloud className="w-4 h-4" />, label: 'HuggingFace' },
    { id: 'integration', icon: <Link2 className="w-4 h-4" />, label: 'Integration' }
  ]

  const progressItems: CollapsibleSectionProps['items'] = [
    { id: 'achievements', icon: <Trophy className="w-4 h-4" />, label: 'Achievements' }
  ]

  return (
    <div className="mb-4 pt-2 border-t border-border">
      <CollapsibleSection title="Library" items={libraryItems} defaultExpanded />
      <CollapsibleSection title="Tools" items={toolsItems} defaultExpanded />
      <CollapsibleSection title="Progress" items={progressItems} defaultExpanded />
      <CollapsibleSection title="Connections" items={connectionsItems} defaultExpanded />
    </div>
  )
}

export function Sidebar() {
  const { isSidebarOpen, setSidebarOpen, overlayView, openOverlay, closeOverlay, setSettingsOpen } = useUIStore()
  const { currentRun } = useTrainingStore()

  // Status state
  const [hooksInstalled, setHooksInstalled] = useState<boolean | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)

  // Fetch status on mount
  useEffect(() => {
    const fetchStatus = async () => {
      // Check hooks
      const hooksResult = await hooksApi.getStatus()
      if (hooksResult.ok && hooksResult.data) {
        setHooksInstalled(hooksResult.data.all_installed)
      } else {
        setHooksInstalled(false)
      }

      // Check API
      const healthResult = await systemApi.health()
      setApiConnected(healthResult.ok)
    }

    if (isSidebarOpen) {
      fetchStatus()
    }
  }, [isSidebarOpen])

  const handleNavClick = (view: 'home' | 'training' | 'factory' | null) => {
    if (view === null) {
      closeOverlay() // Go to workspace (terminals)
    } else if (overlayView === view) {
      closeOverlay() // Toggle off
    } else {
      openOverlay(view)
    }
  }

  if (!isSidebarOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ backgroundColor: 'rgba(27, 32, 64, 0.3)' }}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <aside className="fixed left-0 top-12 bottom-8 w-64 bg-background-card border-r border-border z-50 overflow-y-auto">
        <div className="p-4">
          {/* Header - Clickable to go home */}
          <button
            onClick={() => handleNavClick('home')}
            className="flex items-center gap-3 mb-6 w-full text-left hover-press transition-press"
          >
            <img src="/ghost-icon.png" alt="BashGym" className="w-10 h-10 object-cover" />
            <div>
              <h2 className="font-brand font-semibold text-lg">
                <span className="text-accent">/</span>BashGym
              </h2>
              <p className="text-xs text-text-muted">Agentic Development</p>
            </div>
          </button>

          {/* Primary Navigation */}
          <MenuSection>
            <MenuItem
              icon={<Home className="w-4 h-4" />}
              label="Home"
              onClick={() => handleNavClick('home')}
              active={overlayView === 'home'}
              primary
            />
            <MenuItem
              icon={<Terminal className="w-4 h-4" />}
              label="Workspace"
              onClick={() => handleNavClick(null)}
              active={overlayView === null}
              primary
            />
            <MenuItem
              icon={<Sparkles className="w-4 h-4" />}
              label="Data Factory"
              onClick={() => handleNavClick('factory')}
              active={overlayView === 'factory'}
              primary
            />
            <MenuItem
              icon={<BarChart3 className="w-4 h-4" />}
              label="Training"
              onClick={() => handleNavClick('training')}
              active={overlayView === 'training'}
              badge={currentRun ? 'Live' : undefined}
              primary
            />
          </MenuSection>

          {/* Secondary Sections */}
          <SecondarySections />

          {/* Status Section */}
          <MenuSection title="Status">
            <div className="px-3 py-2 space-y-2">
              {/* API Connection */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {apiConnected ? (
                    <Wifi className="w-3.5 h-3.5 text-status-success" />
                  ) : (
                    <WifiOff className="w-3.5 h-3.5 text-status-error" />
                  )}
                  <span className="font-mono text-xs text-text-secondary">API</span>
                </div>
                <span className={clsx(
                  'font-mono text-xs border rounded-brutal px-2 py-0.5',
                  apiConnected
                    ? 'border-status-success text-status-success'
                    : 'border-status-error text-status-error'
                )}>
                  {apiConnected === null ? '...' : apiConnected ? 'Connected' : 'Offline'}
                </span>
              </div>

              {/* Hooks Status */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {hooksInstalled ? (
                    <CheckCircle className="w-3.5 h-3.5 text-status-success" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-status-warning" />
                  )}
                  <span className="font-mono text-xs text-text-secondary">Hooks</span>
                </div>
                <span className={clsx(
                  'font-mono text-xs border rounded-brutal px-2 py-0.5',
                  hooksInstalled
                    ? 'border-status-success text-status-success'
                    : 'border-status-warning text-status-warning'
                )}>
                  {hooksInstalled === null ? '...' : hooksInstalled ? 'Installed' : 'Not installed'}
                </span>
              </div>

              {/* Setup prompt if hooks not installed */}
              {hooksInstalled === false && (
                <button
                  onClick={() => {
                    setSettingsOpen(true)
                    setSidebarOpen(false)
                  }}
                  className="btn-secondary mt-2 w-full font-mono text-xs text-center py-1.5"
                >
                  Setup Hooks →
                </button>
              )}
            </div>
          </MenuSection>

          {/* Settings */}
          <div className="pt-2 border-t border-border">
            <MenuItem
              icon={<Settings className="w-4 h-4" />}
              label="Settings"
              onClick={() => {
                setSettingsOpen(true)
                setSidebarOpen(false)
              }}
            />
          </div>
        </div>
      </aside>
    </>
  )
}
