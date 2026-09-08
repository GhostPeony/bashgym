import { useCallback, useEffect, useState } from 'react'
import { BookOpen, Flower2, Home, Layers3, LogOut, Settings2 } from 'lucide-react'
import { AutoResearchControlRoom } from '../autoresearch/AutoResearchControlRoom'
import { StudioResources } from './StudioResources'
import { useAuthStore } from '../../stores/authStore'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import { useUIStore } from '../../stores/uiStore'
import { campaignApi } from '../../services/api'
import {
  readStudioView,
  resolveStudioWorkspace,
  studioViews,
  type StudioView
} from './studioNavigation'
import '../../styles/studio.css'

const navigation = {
  home: { label: 'Home', icon: Home },
  setup: { label: 'Setup', icon: Settings2 },
  experiments: { label: 'Experiments', icon: BookOpen },
  resources: { label: 'Resources', icon: Layers3 }
}

export function ResearchStudio() {
  const [view, setView] = useState<StudioView>(() => readStudioView(window.location.hash))
  const workspaces = useWorkspaceStore((state) => state.workspaces)
  const activeWorkspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const selection = useUIStore((state) => state.trainingSelection)
  const workspaceId = selection.workspaceId || activeWorkspaceId
  const workspaceName =
    workspaces.find((item) => item.id === workspaceId)?.name || 'Research project'
  const logout = useAuthStore((state) => state.logout)
  const [allowedWorkspaces, setAllowedWorkspaces] = useState<string[] | null>(null)
  const [accessError, setAccessError] = useState<string | null>(null)
  const loadAccess = useCallback(async () => {
    setAccessError(null)
    const response = await campaignApi.capabilities()
    const ids = response.data?.workspace_ids
    if (!response.ok || !Array.isArray(ids) || !resolveStudioWorkspace(ids, null)) {
      setAccessError(
        response.error ||
          'This session has no available research projects. Pair with a project-scoped code and retry.'
      )
      return
    }
    setAllowedWorkspaces(ids)
  }, [])
  useEffect(() => {
    void loadAccess()
  }, [loadAccess])
  useEffect(() => {
    if (!allowedWorkspaces) return
    const hydrate = () => {
      const params = new URLSearchParams(window.location.search)
      const requested = params.get('workspace_id')
      const authorizedWorkspace = resolveStudioWorkspace(allowedWorkspaces, requested)
      useUIStore.getState().openTraining(
        'autoresearch',
        {
          workspaceId: authorizedWorkspace,
          campaignId: requested === authorizedWorkspace ? params.get('campaign_id') : null
        },
        'replace'
      )
      setView(readStudioView(window.location.hash))
    }
    hydrate()
    window.addEventListener('hashchange', hydrate)
    window.addEventListener('popstate', hydrate)
    return () => {
      window.removeEventListener('hashchange', hydrate)
      window.removeEventListener('popstate', hydrate)
    }
  }, [allowedWorkspaces])
  const projectReady = Boolean(
    allowedWorkspaces && selection.workspaceId && allowedWorkspaces.includes(selection.workspaceId)
  )
  return (
    <div className="research-studio studio-shell">
      <a
        className="studio-skip"
        href="#studio-main"
        onClick={(event) => {
          event.preventDefault()
          document.getElementById('studio-main')?.focus()
        }}
      >
        Skip to research
      </a>
      <aside className="studio-rail" aria-label="Project rail">
        <a className="studio-wordmark" href="#home">
          <Flower2 aria-hidden="true" />
          <span>BashGym</span>
        </a>
        <p className="studio-eyebrow">Research studio</p>
        <div className="studio-project">
          <span className="studio-project-dot" />
          {workspaceName}
        </div>
        <nav aria-label="Studio navigation">
          {studioViews.map((item) => {
            const Icon = navigation[item].icon
            return (
              <a key={item} href={`#${item}`} aria-current={view === item ? 'page' : undefined}>
                <Icon size={18} aria-hidden="true" />
                {navigation[item].label}
              </a>
            )
          })}
        </nav>
        <div className="studio-rail-note">
          <span className="studio-eyebrow">The research loop</span>
          <p>
            Ask a clear question.
            <br />
            Change one thing.
            <br />
            Let the evidence lead.
          </p>
        </div>
        <button className="studio-signout" onClick={() => void logout()}>
          <LogOut size={16} />
          Disconnect session
        </button>
      </aside>
      <main id="studio-main" className="studio-main" tabIndex={-1}>
        <header className="studio-page-header">
          <span className="studio-eyebrow">
            {workspaceName} / {navigation[view].label}
          </span>
          <span className="studio-edition">BashGym · Research journal</span>
        </header>
        <div className="studio-page-intro">
          <h1>
            {view === 'home'
              ? 'A little progress, thoughtfully made.'
              : view === 'setup'
                ? 'Give your experiment a foundation.'
                : view === 'experiments'
                  ? 'The experiment journal.'
                  : 'Your research materials.'}
          </h1>
          <p>
            {view === 'home'
              ? 'Your objective, the latest evidence, and the next considered step.'
              : view === 'setup'
                ? 'Choose registered materials, agree on the evaluation, and prepare a bounded campaign.'
                : view === 'experiments'
                  ? 'Follow the question from hypothesis to a decision you can explain.'
                  : 'Review the campaign’s registered inputs and inspect the evidence behind each result.'}
          </p>
        </div>
        {!projectReady ? (
          <section className="studio-paper" role={accessError ? 'alert' : 'status'}>
            <h2>
              {accessError ? 'Project access needs attention' : 'Opening your research project…'}
            </h2>
            {accessError && (
              <>
                <p>{accessError}</p>
                <button className="studio-primary" onClick={() => void loadAccess()}>
                  Retry project access
                </button>
              </>
            )}
          </section>
        ) : view === 'resources' ? (
          <StudioResources>
            <AutoResearchControlRoom studioView={view} />
          </StudioResources>
        ) : (
          <AutoResearchControlRoom studioView={view} />
        )}
      </main>
    </div>
  )
}
