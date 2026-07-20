export interface TrainingOrigin {
  kind?: string
  workspace_id?: string
  terminal_id?: string
  panel_id?: string
  agent?: string
}

export interface TrainingQueuedPayload {
  run_id: string
  status?: string
  strategy?: string
  base_model?: string
  dataset_path?: string
  origin?: TrainingOrigin
  correlation_id?: string
  compute_target?: string
}

export interface TrainingLaunchConfiguration {
  strategy: string
  baseModel: string
  datasetPath: string
  useNemoGym?: boolean
  useRemoteSSH?: boolean
  deviceId?: string
}

export interface TrainingOriginState {
  panels: readonly {
    id: string
    type: string
    terminalId?: string
  }[]
  sessions: ReadonlyMap<string, { agentKind?: string }>
  activePanelId: string | null
  activeSessionId: string | null
}

export interface TrainingQueuedFallback {
  strategy: string
  baseModel: string
  datasetPath: string
  origin: TrainingOrigin
  correlationId: string
  computeTarget: string
}

const ACTIVE_TRAINING_STATUSES = new Set(['starting', 'running', 'paused'])

/** A retained terminal run remains inspectable, but only active states are live. */
export function isTrainingRunActive(status: string | null | undefined): boolean {
  return status ? ACTIVE_TRAINING_STATUSES.has(status) : false
}

function originForPanel(
  panel: TrainingOriginState['panels'][number],
  state: TrainingOriginState
): TrainingOrigin {
  const session = panel.terminalId ? state.sessions.get(panel.terminalId) : undefined
  return {
    kind: panel.type === 'terminal' ? 'terminal' : 'panel',
    panel_id: panel.id,
    ...(panel.terminalId ? { terminal_id: panel.terminalId } : {}),
    ...(session?.agentKind ? { agent: session.agentKind } : {})
  }
}

/** Prefer the active surface, then the active terminal, so a run stays with its source. */
export function resolveTrainingOrigin(state: TrainingOriginState): TrainingOrigin {
  const activePanel = state.panels.find((panel) => panel.id === state.activePanelId)
  if (activePanel) return originForPanel(activePanel, state)

  const activeTerminalPanel = state.panels.find(
    (panel) => panel.terminalId === state.activeSessionId
  )
  if (activeTerminalPanel) return originForPanel(activeTerminalPanel, state)

  return { kind: 'workspace' }
}

/** Mirrors the backend target precedence, keeping canvas provenance trustworthy. */
export function inferTrainingComputeTarget(config: TrainingLaunchConfiguration): string {
  if (config.useRemoteSSH) {
    return config.deviceId ? `ssh:${config.deviceId}` : 'ssh:remote'
  }
  return config.useNemoGym ? 'cloud' : 'local'
}

export function createTrainingCorrelationId(now = Date.now(), random = Math.random): string {
  const entropy = Math.floor(random() * 0x1000000)
    .toString(36)
    .padStart(4, '0')
  return `training-${now.toString(36)}-${entropy}`
}

function trainingOriginFromResponse(
  value: Record<string, unknown> | undefined
): TrainingOrigin | undefined {
  if (!value) return undefined
  const kind = typeof value.kind === 'string' ? value.kind : undefined
  const terminalId = typeof value.terminal_id === 'string' ? value.terminal_id : undefined
  const panelId = typeof value.panel_id === 'string' ? value.panel_id : undefined
  const agent = typeof value.agent === 'string' ? value.agent : undefined
  if (!kind && !terminalId && !panelId && !agent) return undefined
  return {
    ...(kind ? { kind } : {}),
    ...(terminalId ? { terminal_id: terminalId } : {}),
    ...(panelId ? { panel_id: panelId } : {}),
    ...(agent ? { agent } : {})
  }
}

/** Convert a REST response into the same idempotent path used by WebSocket events. */
export function trainingQueuedPayloadFromResponse(
  response: {
    run_id: string
    status: string
    strategy: string
    origin?: Record<string, unknown>
    correlation_id?: string
    compute_target?: string
  },
  fallback?: TrainingQueuedFallback
): TrainingQueuedPayload {
  return {
    run_id: response.run_id,
    status: response.status,
    strategy: response.strategy || fallback?.strategy,
    base_model: fallback?.baseModel,
    dataset_path: fallback?.datasetPath,
    origin: trainingOriginFromResponse(response.origin) ?? fallback?.origin,
    correlation_id: response.correlation_id ?? fallback?.correlationId,
    compute_target: response.compute_target ?? fallback?.computeTarget
  }
}
