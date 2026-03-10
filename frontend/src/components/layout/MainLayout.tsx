import { lazy, Suspense, useState } from 'react'
import { MessageSquare } from 'lucide-react'
import { NavigationBar } from './NavigationBar'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'
import { TrainingDashboard } from '../training/TrainingDashboard'
import { RouterDashboard } from '../router/RouterDashboard'
import { TraceBrowser } from '../traces/TraceBrowser'
import { FactoryDashboard } from '../factory/FactoryDashboard'
import { EvaluatorDashboard } from '../evaluator/EvaluatorDashboard'
import { GuardrailsDashboard } from '../guardrails/GuardrailsDashboard'
import { ProfilerDashboard } from '../profiler/ProfilerDashboard'
import { ModelBrowser, ModelProfilePage, ModelComparison, ModelTrends } from '../models'
import { HFDashboard } from '../huggingface'
import { AchievementsView } from '../achievements/AchievementsView'
import { HomeScreen, TutorialChecklist, TutorialTooltip } from '../home'
import { KeyboardShortcutsModal } from '../common/KeyboardShortcutsModal'
import { useUIStore } from '../../stores'
import { isElectron, isWeb } from '../../utils/platform'

// Electron-only components — tree-shaken from web builds because isElectron
// is a build-time constant (VITE_MODE !== 'web' → true in Electron, false in web)
const TerminalGrid = isElectron
  ? lazy(() => import('../terminal/TerminalGrid').then(m => ({ default: m.TerminalGrid })))
  : null
const AgentChat = isElectron
  ? lazy(() => import('../agent/AgentChat').then(m => ({ default: m.AgentChat })))
  : null
const OrchestratorDashboard = isElectron
  ? lazy(() => import('../orchestrator/OrchestratorDashboard').then(m => ({ default: m.OrchestratorDashboard })))
  : null
const PipelineDashboard = isElectron
  ? lazy(() => import('../pipeline/PipelineDashboard').then(m => ({ default: m.PipelineDashboard })))
  : null
const IntegrationDashboard = isElectron
  ? lazy(() => import('../integration/IntegrationDashboard').then(m => ({ default: m.IntegrationDashboard })))
  : null

// Web-only components
const DownloadPage = isWeb
  ? lazy(() => import('../download/DownloadPage').then(m => ({ default: m.DownloadPage })))
  : null

type ModelSubView = 'browser' | 'profile' | 'comparison' | 'trends'

function LazyFallback() {
  return <div className="flex-1 flex items-center justify-center text-text-muted font-mono text-sm">Loading...</div>
}

export function MainLayout() {
  const { isSidebarOpen, overlayView, openOverlay, isAgentChatOpen, toggleAgentChat } = useUIStore()
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
  const [modelSubView, setModelSubView] = useState<ModelSubView>('browser')
  const [compareModelIds, setCompareModelIds] = useState<string[]>([])

  const showHome = overlayView === 'home'
  const showWorkspace = overlayView === null

  return (
    <div className="h-screen flex flex-col bg-background-primary">
      {/* Top Navigation */}
      <NavigationBar />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Collapsible Sidebar */}
        <Sidebar />

        {/* Workspace Area */}
        <main className="flex-1 flex flex-col">
          {/* Home Screen */}
          {showHome && (
            <div className="flex-1 overflow-auto">
              <HomeScreen />
            </div>
          )}

          {/* Terminal Grid - Always rendered, persists in background (Electron only) */}
          {TerminalGrid && (
            <div className={`flex-1 ${showWorkspace ? 'block' : 'hidden'}`}>
              <Suspense fallback={<LazyFallback />}>
                <TerminalGrid />
              </Suspense>
            </div>
          )}

          {/* Web: show home when workspace is selected (no terminal) */}
          {!TerminalGrid && showWorkspace && (
            <div className="flex-1 overflow-auto">
              <HomeScreen />
            </div>
          )}

          {/* Dashboard Overlays */}
          {overlayView === 'training' && (
            <div className="flex-1 overflow-auto">
              <TrainingDashboard />
            </div>
          )}

          {overlayView === 'router' && (
            <div className="flex-1 overflow-auto">
              <RouterDashboard />
            </div>
          )}

          {overlayView === 'traces' && (
            <div className="flex-1 overflow-auto">
              <TraceBrowser />
            </div>
          )}

          {overlayView === 'factory' && (
            <div className="flex-1 overflow-auto">
              <FactoryDashboard />
            </div>
          )}

          {overlayView === 'evaluator' && (
            <div className="flex-1 overflow-auto">
              <EvaluatorDashboard />
            </div>
          )}

          {overlayView === 'guardrails' && (
            <div className="flex-1 overflow-auto">
              <GuardrailsDashboard />
            </div>
          )}

          {overlayView === 'profiler' && (
            <div className="flex-1 overflow-auto">
              <ProfilerDashboard />
            </div>
          )}

          {overlayView === 'models' && (
            <div className="flex-1 overflow-auto">
              {modelSubView === 'comparison' ? (
                <ModelComparison
                  modelIds={compareModelIds}
                  onBack={() => {
                    setModelSubView('browser')
                    setCompareModelIds([])
                  }}
                  onAddModel={() => setModelSubView('browser')}
                  onRemoveModel={(id) => {
                    setCompareModelIds(prev => prev.filter(m => m !== id))
                    if (compareModelIds.length <= 2) {
                      setModelSubView('browser')
                    }
                  }}
                />
              ) : modelSubView === 'trends' ? (
                <ModelTrends
                  onBack={() => setModelSubView('browser')}
                  onSelectModel={(modelId) => {
                    setSelectedModelId(modelId)
                    setModelSubView('profile')
                  }}
                />
              ) : selectedModelId ? (
                <ModelProfilePage
                  modelId={selectedModelId}
                  onBack={() => {
                    setSelectedModelId(null)
                    setModelSubView('browser')
                  }}
                  onCompare={(modelIds) => {
                    setCompareModelIds(modelIds)
                    setModelSubView('comparison')
                  }}
                />
              ) : (
                <ModelBrowser
                  onSelectModel={(modelId) => {
                    setSelectedModelId(modelId)
                    setModelSubView('profile')
                  }}
                  onTrainNew={() => openOverlay('training')}
                  onCompare={(modelIds) => {
                    setCompareModelIds(modelIds)
                    setModelSubView('comparison')
                  }}
                  onViewTrends={() => setModelSubView('trends')}
                />
              )}
            </div>
          )}

          {overlayView === 'huggingface' && (
            <div className="flex-1 overflow-auto">
              <HFDashboard />
            </div>
          )}

          {overlayView === 'integration' && IntegrationDashboard && (
            <div className="flex-1 overflow-auto">
              <Suspense fallback={<LazyFallback />}>
                <IntegrationDashboard />
              </Suspense>
            </div>
          )}

          {overlayView === 'achievements' && (
            <div className="flex-1 overflow-auto">
              <AchievementsView />
            </div>
          )}

          {overlayView === 'orchestrator' && OrchestratorDashboard && (
            <div className="flex-1 overflow-auto">
              <Suspense fallback={<LazyFallback />}>
                <OrchestratorDashboard />
              </Suspense>
            </div>
          )}

          {overlayView === 'pipeline' && PipelineDashboard && (
            <div className="flex-1 overflow-auto">
              <Suspense fallback={<LazyFallback />}>
                <PipelineDashboard />
              </Suspense>
            </div>
          )}

          {overlayView === 'download' && DownloadPage && (
            <div className="flex-1 overflow-auto">
              <Suspense fallback={<LazyFallback />}>
                <DownloadPage />
              </Suspense>
            </div>
          )}
        </main>
      </div>

      {/* Bottom Status Bar */}
      <StatusBar />

      {/* Tutorial Components */}
      <TutorialChecklist />
      <TutorialTooltip />

      {/* Global Modals */}
      <KeyboardShortcutsModal />

      {/* Agent Chat Panel — Electron only (spawns Claude CLI subprocesses) */}
      {AgentChat && (
        <Suspense fallback={null}>
          <AgentChat />
        </Suspense>
      )}

      {/* Floating Agent Button — Electron only */}
      {AgentChat && !isAgentChatOpen && (
        <button
          onClick={toggleAgentChat}
          className="fixed bottom-6 right-6 z-40 w-12 h-12 border-brutal border-border rounded-full bg-accent text-white shadow-brutal flex items-center justify-center hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-all"
          title="Open Peony Agent"
        >
          <MessageSquare className="w-5 h-5" />
        </button>
      )}
    </div>
  )
}
