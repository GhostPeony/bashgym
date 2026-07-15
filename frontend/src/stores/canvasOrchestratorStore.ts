import { create } from 'zustand'
import type { DesignerJobStatus, RuntimeJob } from '../services/api'
import { useTerminalStore, type CanvasEdge, type Panel } from './terminalStore'
import { useWorkspaceStore } from './workspaceStore'
import {
  selectDesignerPanelForJob,
  type DesignerCanvasMetadata,
} from './designerCanvasLifecycle'
import {
  recipeFromTrainingProgress,
  recipeFromTrainingQueued,
  recipeFromWorkspaceIntent,
  shouldReuseTrainingOrigin,
  statusVisualForTraining,
  type CanvasNodeRecipe,
  type TrainingProgressPayload,
  type TrainingQueuedPayload,
  type WorkspaceCanvasIntentPayload,
} from '../components/terminal/canvasRecipes'
import { findDynamicNodePosition } from '../components/terminal/canvasPlacement'

interface CanvasOrchestratorState {
  handledKeys: Record<string, number>
  handleWorkspaceIntent: (payload: WorkspaceCanvasIntentPayload) => void
  handleTrainingQueued: (payload: TrainingQueuedPayload) => void
  handleTrainingProgress: (payload: TrainingProgressPayload) => void
  handleTrainingTerminalStatus: (runId: string, status: string) => void
  handleDesignerJob: (job: DesignerJobStatus, metadata?: DesignerCanvasMetadata) => void
  handleRuntimeJob: (job: RuntimeJob) => void
}

function panelMatchesRecipe(panel: Panel, recipe: CanvasNodeRecipe): boolean {
  if (panel.type !== recipe.type) return false
  if (recipe.type === 'skilllab') return true
  const cfg = panel.adapterConfig || {}
  const runId = recipe.config.runId
  const correlationId = recipe.config.correlationId
  if (runId && cfg.runId === runId) return true
  if (correlationId && cfg.correlationId === correlationId) return true
  return false
}

/**
 * Resolve the origin panel within the ACTIVE workspace. When the recipe names
 * an explicit origin that isn't materialized here (it lives in a background
 * workspace), return the sentinel so the caller skips materialization instead
 * of grafting the node onto the wrong canvas.
 */
const ORIGIN_ELSEWHERE = Symbol('origin-elsewhere')

function findOriginPanel(recipe: CanvasNodeRecipe): Panel | undefined | typeof ORIGIN_ELSEWHERE {
  const state = useTerminalStore.getState()
  if (recipe.config.runtimeDiscovered) return undefined
  const hasExplicitOrigin = Boolean(recipe.originPanelId || recipe.originTerminalId)
  if (recipe.originPanelId) {
    const byPanel = state.panels.find((panel) => panel.id === recipe.originPanelId)
    if (byPanel) return byPanel
  }
  if (recipe.originTerminalId) {
    const byTerminal = state.panels.find((panel) => panel.terminalId === recipe.originTerminalId)
    if (byTerminal) return byTerminal
  }
  if (hasExplicitOrigin) return ORIGIN_ELSEWHERE

  const sessions = Array.from(state.sessions.values()).sort((a, b) => b.lastActivity - a.lastActivity)
  const activeAgent =
    sessions.find((session) => session.status === 'running' || session.status === 'tool_calling') ||
    sessions.find((session) => session.agentKind)
  if (activeAgent) {
    return state.panels.find((panel) => panel.terminalId === activeAgent.id)
  }

  return state.panels.find((panel) => panel.id === state.activePanelId)
}

function placeNearOrigin(panelId: string, originPanel: Panel | undefined, type: Panel['type']) {
  const state = useTerminalStore.getState()
  const activeSessionPanel = Array.from(state.sessions.values())
    .filter((session) => session.status === 'running' || session.status === 'tool_calling')
    .sort((left, right) => right.lastActivity - left.lastActivity)
    .map((session) => state.panels.find((panel) => panel.terminalId === session.id))
    .find((panel) => panel && state.canvasNodes.has(panel.id))
  const activePanel = state.activePanelId
    ? state.panels.find((panel) => panel.id === state.activePanelId)
    : undefined
  const anchorPanel = originPanel || activeSessionPanel || activePanel
  const anchor = anchorPanel ? state.canvasNodes.get(anchorPanel.id)?.position : undefined
  const occupied = Array.from(state.canvasNodes.values())
    .filter((node) => node.panelId !== panelId)
    .map((node) => node.position)

  state.updateCanvasNode(panelId, findDynamicNodePosition(type, anchor, occupied))
}

function upsertEdge(source: string | undefined, target: string): void {
  if (!source || source === target) return
  const state = useTerminalStore.getState()
  const exists = state.canvasEdges.some(
    (edge) =>
      (edge.source === source && edge.target === target) ||
      (edge.source === target && edge.target === source)
  )
  if (exists) return
  const edge: CanvasEdge = {
    id: `edge-${source}-${target}`,
    source,
    target,
  }
  state.setCanvasEdges([...state.canvasEdges, edge])
}

function materializeRecipe(recipe: CanvasNodeRecipe): string | null {
  if (useWorkspaceStore.getState().switching) {
    console.info('[canvas-orchestrator] dropped recipe mid-switch', recipe.key)
    return null
  }
  const state = useTerminalStore.getState()
  const matchingPanel = state.panels.find((panel) => panelMatchesRecipe(panel, recipe))
  const resolved = findOriginPanel(recipe)
  if (resolved === ORIGIN_ELSEWHERE && !matchingPanel) {
    console.info('[canvas-orchestrator] origin not in active workspace, skipping', recipe.key)
    return null
  }
  const originPanel = resolved === ORIGIN_ELSEWHERE ? undefined : resolved
  const existing = matchingPanel ?? (
    shouldReuseTrainingOrigin(recipe, originPanel) ? originPanel : undefined
  )
  const existingConfig = existing?.adapterConfig || {}
  const visual =
    recipe.type === 'training'
      ? statusVisualForTraining(String(recipe.config.status || existingConfig.status || 'planned'))
      : recipe.visual

  const mergedConfig = {
    ...existingConfig,
    ...recipe.config,
    originPanelId: recipe.config.originPanelId || originPanel?.id,
    visual,
  }

  const panelId = existing?.id ?? state.addPanel({
    type: recipe.type,
    title: recipe.title,
    adapterConfig: mergedConfig,
  })

  if (existing) {
    state.updatePanelConfig(panelId, mergedConfig)
  } else {
    placeNearOrigin(panelId, originPanel, recipe.type)
  }

  upsertEdge(originPanel?.id, panelId)
  if (recipe.type === 'skilllab') state.setActivePanel(panelId)
  return panelId
}

export const useCanvasOrchestratorStore = create<CanvasOrchestratorState>((set, get) => ({
  handledKeys: {},

  handleWorkspaceIntent: (payload) => {
    const originPanel = payload.source?.panel_id
      ? useTerminalStore.getState().panels.find((panel) => panel.id === payload.source?.panel_id)
      : undefined
    const recipe = recipeFromWorkspaceIntent(payload, originPanel?.id)
    if (!recipe) return
    if (materializeRecipe(recipe) === null) return
    set((state) => ({ handledKeys: { ...state.handledKeys, [recipe.key]: Date.now() } }))
  },

  handleTrainingQueued: (payload) => {
    if (!payload.run_id) return
    const recipe = recipeFromTrainingQueued(payload)
    if (materializeRecipe(recipe) === null) return
    set((state) => ({ handledKeys: { ...state.handledKeys, [recipe.key]: Date.now() } }))
  },

  handleTrainingProgress: (payload) => {
    if (!payload.run_id) return
    const key = `training:${payload.run_id}`
    const exists = useTerminalStore.getState().panels.some((panel) => (
      panel.type === 'training' && panel.adapterConfig?.runId === payload.run_id
    ))
    if (!exists) {
      if (materializeRecipe(recipeFromTrainingProgress(payload)) === null) return
    } else {
      get().handleTrainingTerminalStatus(payload.run_id, 'running')
    }
    set((state) => ({ handledKeys: { ...state.handledKeys, [key]: Date.now() } }))
  },

  handleTrainingTerminalStatus: (runId, status) => {
    if (useWorkspaceStore.getState().switching) return
    const state = useTerminalStore.getState()
    const panel = state.panels.find((candidate) => (
      candidate.type === 'training' && candidate.adapterConfig?.runId === runId
    ))
    if (!panel) return
    const adapterConfig = {
      ...(panel.adapterConfig || {}),
      status,
      visual: statusVisualForTraining(status),
    }
    state.updatePanelConfig(panel.id, adapterConfig)
  },

  handleDesignerJob: (job, metadata = {}) => {
    if (useWorkspaceStore.getState().switching) return
    const state = useTerminalStore.getState()
    const requestedOriginPanelId = metadata.originPanelId || job.origin?.panel_id
    const originPanel = (
      requestedOriginPanelId
        ? state.panels.find((candidate) => candidate.id === requestedOriginPanelId)
        : undefined
    ) ?? (
      job.origin?.terminal_id
        ? state.panels.find((candidate) => candidate.terminalId === job.origin?.terminal_id)
        : undefined
    )
    const panel = selectDesignerPanelForJob(state.panels, job, originPanel?.id)
    const adapterConfig = {
      ...(panel?.adapterConfig || {}),
      ...(metadata.config || {}),
      designerJobId: job.job_id,
      runtimeJobId: metadata.runtimeDiscovered ? job.job_id : undefined,
      runtimePid: metadata.runtimePid,
      runtimeStartedAt: job.started_at,
      runtimeDiscovered: metadata.runtimeDiscovered ?? false,
      originPanelId: originPanel?.id,
      originTerminalId: job.origin?.terminal_id,
      originAgent: job.origin?.agent,
      correlationId: job.correlation_id,
      status: job.status,
      pipeline: job.pipeline,
      numRecords: job.num_records,
      jobName: job.job_name,
      dataset: job.dataset,
      model: job.model,
      provider: job.provider,
      execution: job.execution,
      outputDir: job.output_dir,
      progress: job.progress ? { ...job.progress } : undefined,
      error: job.error,
    }
    const panelId = panel?.id ?? state.addPanel({
      type: 'designer',
      title: 'Data Designer',
      adapterConfig,
    })
    if (panel) state.updatePanelConfig(panelId, adapterConfig)
    else placeNearOrigin(panelId, originPanel, 'designer')
    upsertEdge(originPanel?.id, panelId)
    set((current) => ({
      handledKeys: { ...current.handledKeys, [`designer:${job.job_id}`]: Date.now() },
    }))
  },

  handleRuntimeJob: (job) => {
    if (useWorkspaceStore.getState().switching) return
    const progress = job.progress
      ? { ...job.progress }
      : undefined
    const summary = [
      `Observed ${job.script} (PID ${job.pid})`,
      progress?.total ? `${progress.current}/${progress.total} ${progress.unit}` : undefined,
      job.artifacts.length ? `${job.artifacts.length} recent artifacts` : undefined,
    ].filter(Boolean).join(' · ')

    if (job.kind === 'designer') {
      get().handleDesignerJob({
        job_id: job.job_id,
        status: job.status,
        pipeline: job.pipeline || job.script,
        num_records: job.progress?.total ?? 0,
        progress: job.progress?.total
          ? { ...job.progress, total: job.progress.total }
          : undefined,
        job_name: job.job_name,
        dataset: job.dataset,
        model: job.model,
        provider: job.provider,
        execution: job.execution,
        started_at: job.started_at,
        output_dir: job.output_dir || undefined,
      }, {
        runtimePid: job.pid,
        runtimeDiscovered: true,
        config: { summary },
      })
      return
    }

    const recipe = recipeFromTrainingQueued({
      run_id: job.job_id,
      status: job.status,
      strategy: job.strategy || 'training',
      dataset_path: job.output_dir || undefined,
      compute_target: 'local',
    })
    recipe.config = {
      ...recipe.config,
      runtimeJobId: job.job_id,
      runtimePid: job.pid,
      runtimeStartedAt: job.started_at,
      runtimeDiscovered: true,
      progress,
      summary,
      status: job.status,
    }
    recipe.visual = statusVisualForTraining(job.status)
    materializeRecipe(recipe)
  },
}))
