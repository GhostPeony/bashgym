import { create } from 'zustand'
import { useTerminalStore, type CanvasEdge, type Panel } from './terminalStore'
import { useWorkspaceStore } from './workspaceStore'
import {
  recipeFromTrainingProgress,
  recipeFromTrainingQueued,
  recipeFromWorkspaceIntent,
  statusVisualForTraining,
  type CanvasNodeRecipe,
  type TrainingProgressPayload,
  type TrainingQueuedPayload,
  type WorkspaceCanvasIntentPayload,
} from '../components/terminal/canvasRecipes'

interface CanvasOrchestratorState {
  handledKeys: Record<string, number>
  handleWorkspaceIntent: (payload: WorkspaceCanvasIntentPayload) => void
  handleTrainingQueued: (payload: TrainingQueuedPayload) => void
  handleTrainingProgress: (payload: TrainingProgressPayload) => void
  handleTrainingTerminalStatus: (runId: string, status: string) => void
}

function panelMatchesRecipe(panel: Panel, recipe: CanvasNodeRecipe): boolean {
  if (panel.type !== recipe.type) return false
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
  const originPos = originPanel ? state.canvasNodes.get(originPanel.id)?.position : undefined
  const existingPositions = new Set(
    Array.from(state.canvasNodes.values()).map((node) => `${node.position.x}:${node.position.y}`)
  )
  const laneOffset =
    type === 'evals'
      ? { x: 900, y: 40 }
      : type === 'huggingface'
        ? { x: 450, y: 260 }
        : type === 'toolkit'
          ? { x: 0, y: 300 }
          : { x: 460, y: -60 }
  const base = originPos
    ? { x: originPos.x + laneOffset.x, y: originPos.y + laneOffset.y }
    : { x: 50 + state.panels.length * 30, y: 50 + state.panels.length * 24 }

  let pos = base
  for (let i = 0; i < 8 && existingPositions.has(`${pos.x}:${pos.y}`); i += 1) {
    pos = { x: base.x + i * 40, y: base.y + i * 34 }
  }
  state.updateCanvasNode(panelId, pos)
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
  const existing = state.panels.find((panel) => panelMatchesRecipe(panel, recipe))
  const resolved = findOriginPanel(recipe)
  if (resolved === ORIGIN_ELSEWHERE && !existing) {
    console.info('[canvas-orchestrator] origin not in active workspace, skipping', recipe.key)
    return null
  }
  const originPanel = resolved === ORIGIN_ELSEWHERE ? undefined : resolved
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
}))
