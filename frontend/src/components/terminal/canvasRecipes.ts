import type { PanelType } from '../../stores/terminalStore'

export type NodeVisualHue = 'marigold' | 'moss' | 'sky' | 'orchid' | 'wisteria' | 'poppy'
export type NodeVisualPhase = 'planned' | 'queued' | 'running' | 'waiting' | 'completed' | 'failed'
export type NodeVisualIntensity = 'quiet' | 'active' | 'urgent'
export type NodeVisualMotion = 'none' | 'enter-from-origin' | 'edge-pulse' | 'state-strip'

export interface NodeVisualRecipe {
  hue: NodeVisualHue
  phase: NodeVisualPhase
  intensity: NodeVisualIntensity
  icon: string
  motion: NodeVisualMotion
}

export interface CanvasNodeRecipe {
  key: string
  type: PanelType
  title: string
  config: Record<string, unknown>
  originPanelId?: string
  originTerminalId?: string
  correlationId?: string
  edgeType?: 'initiated' | 'produces' | 'context'
  visual: NodeVisualRecipe
}

export interface WorkspaceCanvasIntentPayload {
  type: string
  workspace_id?: string
  source?: {
    kind?: string
    terminal_id?: string
    panel_id?: string
    agent?: string
  }
  correlation_id?: string
  title?: string
  summary?: string
  entity?: Record<string, unknown>
  suggested_nodes?: Array<{
    recipe?: string
    title?: string
    config?: Record<string, unknown>
  }>
}

export interface TrainingQueuedPayload {
  run_id: string
  status?: string
  strategy?: string
  base_model?: string
  dataset_path?: string
  origin?: {
    kind?: string
    terminal_id?: string
    panel_id?: string
    agent?: string
  }
  correlation_id?: string
  compute_target?: string
}

export interface TrainingProgressPayload {
  run_id: string
  strategy?: string
  compute_target?: string
}

function trainingVisual(phase: NodeVisualPhase, motion: NodeVisualMotion): NodeVisualRecipe {
  return {
    hue: 'marigold',
    phase,
    intensity: phase === 'failed' ? 'urgent' : phase === 'planned' ? 'quiet' : 'active',
    icon: 'dumbbell',
    motion,
  }
}

function skillLabVisual(phase: NodeVisualPhase, motion: NodeVisualMotion): NodeVisualRecipe {
  return {
    hue: 'wisteria',
    phase,
    intensity: phase === 'failed' ? 'urgent' : phase === 'running' ? 'active' : 'quiet',
    icon: 'skill-lab',
    motion,
  }
}

export function recipeFromWorkspaceIntent(
  payload: WorkspaceCanvasIntentPayload,
  originPanelId?: string,
): CanvasNodeRecipe | null {
  const explicitSkillLab = payload.suggested_nodes?.find((node) => node.recipe === 'skill_lab')
  const entityKind = String(payload.entity?.kind ?? '')
  const looksLikeSkillLab = entityKind === 'skill_lab' || Boolean(explicitSkillLab)
  if (looksLikeSkillLab) {
    const config = explicitSkillLab?.config || {}
    const status = String(config.status || payload.entity?.status || 'prepared')
    const phase: NodeVisualPhase = status === 'running' || status === 'queued'
      ? 'running'
      : status === 'failed'
        ? 'failed'
        : status === 'completed'
          ? 'completed'
          : 'planned'
    const correlationId = payload.correlation_id ?? `skill-lab-${Date.now()}`
    const visual = skillLabVisual(phase, phase === 'running' ? 'state-strip' : 'enter-from-origin')
    return {
      key: `skilllab:${payload.workspace_id || 'default'}`,
      type: 'skilllab',
      title: explicitSkillLab?.title || payload.title || 'Skill Lab',
      originPanelId: payload.source?.panel_id || originPanelId,
      originTerminalId: payload.source?.terminal_id,
      correlationId,
      edgeType: 'context',
      visual,
      config: {
        ...config,
        originPanelId: payload.source?.panel_id || originPanelId,
        originTerminalId: payload.source?.terminal_id,
        originAgent: payload.source?.agent,
        correlationId,
        summary: payload.summary,
        visual,
      },
    }
  }

  const explicitTraining = payload.suggested_nodes?.find((node) => node.recipe === 'training.run')
  const looksLikeTraining =
    payload.type === 'training.prep.started' ||
    entityKind === 'training_run' ||
    Boolean(explicitTraining)

  if (!looksLikeTraining) return null

  const strategy =
    String(explicitTraining?.config?.strategy ?? payload.entity?.strategy ?? 'sft') || 'sft'
  const correlationId = payload.correlation_id ?? `intent-${Date.now()}`
  const title = explicitTraining?.title || payload.title || `${strategy.toUpperCase()} Run`

  return {
    key: `training:${payload.entity?.run_id || correlationId}`,
    type: 'training',
    title,
    originPanelId: payload.source?.panel_id || originPanelId,
    originTerminalId: payload.source?.terminal_id,
    correlationId,
    edgeType: 'initiated',
    visual: trainingVisual('planned', 'enter-from-origin'),
    config: {
      ...explicitTraining?.config,
      strategy,
      status: explicitTraining?.config?.status || 'planned',
      runId: payload.entity?.run_id || undefined,
      originPanelId: payload.source?.panel_id || originPanelId,
      originTerminalId: payload.source?.terminal_id,
      originAgent: payload.source?.agent,
      correlationId,
      summary: payload.summary,
      visual: trainingVisual('planned', 'enter-from-origin'),
    },
  }
}

export function recipeFromTrainingQueued(
  payload: TrainingQueuedPayload,
  originPanelId?: string,
): CanvasNodeRecipe {
  const strategy = payload.strategy || 'sft'
  const correlationId = payload.correlation_id
  const status = payload.status === 'pending' ? 'queued' : payload.status || 'queued'
  return {
    key: `training:${payload.run_id}`,
    type: 'training',
    title: `${strategy.toUpperCase()} Run`,
    originPanelId: payload.origin?.panel_id || originPanelId,
    originTerminalId: payload.origin?.terminal_id,
    correlationId,
    edgeType: 'initiated',
    visual: trainingVisual('queued', 'enter-from-origin'),
    config: {
      runId: payload.run_id,
      strategy,
      status,
      baseModel: payload.base_model,
      datasetPath: payload.dataset_path,
      originPanelId: payload.origin?.panel_id || originPanelId,
      originTerminalId: payload.origin?.terminal_id,
      originAgent: payload.origin?.agent,
      correlationId,
      computeTarget: payload.compute_target,
      visual: trainingVisual('queued', 'enter-from-origin'),
    },
  }
}

export function recipeFromTrainingProgress(
  payload: TrainingProgressPayload,
  originPanelId?: string,
): CanvasNodeRecipe {
  return {
    key: `training:${payload.run_id}`,
    type: 'training',
    title: `${(payload.strategy || 'training').toUpperCase()} Run`,
    originPanelId,
    edgeType: 'initiated',
    visual: trainingVisual('running', 'state-strip'),
    config: {
      runId: payload.run_id,
      strategy: payload.strategy,
      status: 'running',
      originPanelId,
      computeTarget: payload.compute_target,
      visual: trainingVisual('running', 'state-strip'),
    },
  }
}

export function statusVisualForTraining(status: string | undefined): NodeVisualRecipe {
  const normalized = status === 'pending' ? 'queued' : status || 'planned'
  const phase =
    normalized === 'running' || normalized === 'starting'
      ? 'running'
      : normalized === 'completed'
        ? 'completed'
        : normalized === 'failed'
          ? 'failed'
          : normalized === 'paused'
            ? 'waiting'
            : normalized === 'queued'
              ? 'queued'
              : 'planned'
  return trainingVisual(phase, phase === 'running' ? 'state-strip' : 'none')
}

/** A configured-but-unbound Training node becomes the run it launches. */
export function shouldReuseTrainingOrigin(
  recipe: CanvasNodeRecipe,
  originPanel?: { type: PanelType; adapterConfig?: Record<string, unknown> },
): boolean {
  if (recipe.type !== 'training' || originPanel?.type !== 'training') return false
  return !originPanel.adapterConfig?.runId
}
