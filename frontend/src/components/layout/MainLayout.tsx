import { useState } from 'react'
import { NavigationBar } from './NavigationBar'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'
import { TerminalGrid } from '../terminal/TerminalGrid'
import { TrainingDashboard } from '../training/TrainingDashboard'
import { RouterDashboard } from '../router/RouterDashboard'
import { TraceBrowser } from '../traces/TraceBrowser'
import { FactoryDashboard } from '../factory/FactoryDashboard'
import { EvaluatorDashboard } from '../evaluator/EvaluatorDashboard'
import { GuardrailsDashboard } from '../guardrails/GuardrailsDashboard'
import { ProfilerDashboard } from '../profiler/ProfilerDashboard'
import { ModelBrowser, ModelProfilePage, ModelComparison, ModelTrends } from '../models'
import { HFDashboard } from '../huggingface'
import { IntegrationDashboard } from '../integration/IntegrationDashboard'
import { AchievementsView } from '../achievements/AchievementsView'
import { HomeScreen, TutorialChecklist, TutorialTooltip } from '../home'
import { KeyboardShortcutsModal } from '../common/KeyboardShortcutsModal'
import { useUIStore } from '../../stores'

type ModelSubView = 'browser' | 'profile' | 'comparison' | 'trends'

export function MainLayout() {
  const { isSidebarOpen, overlayView, openOverlay } = useUIStore()
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

          {/* Terminal Grid - Always rendered, persists in background */}
          <div className={`flex-1 ${showWorkspace ? 'block' : 'hidden'}`}>
            <TerminalGrid />
          </div>

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

          {overlayView === 'integration' && (
            <div className="flex-1 overflow-auto">
              <IntegrationDashboard />
            </div>
          )}

          {overlayView === 'achievements' && (
            <div className="flex-1 overflow-auto">
              <AchievementsView />
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
    </div>
  )
}
